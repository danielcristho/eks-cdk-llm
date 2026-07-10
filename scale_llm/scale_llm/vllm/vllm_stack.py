from aws_cdk import Stack, CfnOutput, aws_eks as eks
from constructs import Construct


class VllmStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        cluster: eks.Cluster,
        model_bucket_name: str,
        kuberay_operator,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

        """KubeRay caches serveConfigV2 by its own text and skips resubmitting when it matches, so a
        serve_vllm.py-only edit (same YAML, different mounted script) would otherwise go unnoticed.
        Folding the script's hash into an (unused) arg forces the YAML to change whenever it does.
        """
        import hashlib
        script_sha = hashlib.sha256(_load_serve_script().encode()).hexdigest()[:12]

        """
        NVIDIA device plugin, makes nvidia.com/gpu requestable as a resource.
        release= kept short: the chart renders 2 DaemonSets (plugin + mps
        control daemon), and CDK's auto-generated release name pushed both
        names past K8s's 63-char limit, truncating them to the same name.
        """
        nvidia_plugin = cluster.add_helm_chart(
            "NvidiaDevicePlugin",
            chart="nvidia-device-plugin",
            repository="https://nvidia.github.io/k8s-device-plugin",
            version="0.17.1",
            release="nvidia-device-plugin",
            namespace="kube-system",
            values={
                "nodeSelector": {"workload": "gpu"},
                "tolerations": [
                    {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
                ],
            },
        )

        # serve_vllm.py embedded as ConfigMap, mounted into workers at /serve-scripts
        serve_configmap = cluster.add_manifest(
            "VllmServeConfigMap",
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {
                    "name": "vllm-serve-script",
                    "namespace": "kuberay",
                },
                "data": {
                    "serve_vllm.py": _load_serve_script(),
                },
            },
        )
        serve_configmap.node.add_dependency(kuberay_operator)

        # Runs vLLM as a Ray Serve deployment
        # Ray routes requests across replicas, autoscaler scales workers, Karpenter provisions GPU nodes
        ray_service = cluster.add_manifest(
            "VllmRayService",
            {
                "apiVersion": "ray.io/v1",
                "kind": "RayService",
                "metadata": {
                    "name": "vllm",
                    "namespace": "kuberay",
                },
                "spec": {
                    "serveConfigV2": (
                        "applications:\n"
                        "  - name: vllm\n"
                        "    import_path: serve_vllm:build_app\n"
                        "    route_prefix: /\n"
                        "    runtime_env:\n"
                        "      pip:\n"
                        "        - vllm==0.8.5\n"
                        "        - ray[serve]==2.43.0\n"  # matches cluster image; vllm forbids ray==2.44.*
                        "        - numpy<2\n"             # vllm pulls numpy 2.x, breaks image's numpy-1.x pyarrow
                        "      env_vars:\n"
                        "        PYTHONPATH: /serve-scripts\n"
                        f"    args:\n"
                        f"      model: \"{model_id}\"\n"
                        "      dtype: half\n"
                        "      quantization: awq\n"
                        "      max_model_len: 4096\n"
                        "      download_dir: /model-cache\n"
                        f"      _script_sha: \"{script_sha}\"\n"  # unused by build_app; busts KubeRay's config cache
                        "    deployments:\n"
                        "      - name: VLLMDeployment\n"
                        "        max_ongoing_requests: 5\n"
                        "        autoscaling_config:\n"
                        "          min_replicas: 0\n"
                        "          max_replicas: 2\n"  # set max 2 × g4dn.xlarge
                        "          target_ongoing_requests: 2\n"
                        "        ray_actor_options:\n"
                        "          num_gpus: 1\n"
                        "          memory: 12288\n"
                    ),
                    "rayClusterConfig": {
                        "rayVersion": "2.43.0",
                        "enableInTreeAutoscaling": True,
                        "headGroupSpec": {
                            "rayStartParams": {
                                "dashboard-host": "0.0.0.0",
                                "num-gpus": "0",
                            },
                            "template": {
                                "metadata": {"labels": {"role": "ray-head"}},
                                "spec": {
                                    "nodeSelector": {"role": "system"},
                                    "containers": [
                                        {
                                            "name": "ray-head",
                                            "image": "rayproject/ray:2.43.0-gpu",
                                            "ports": [
                                                {"containerPort": 6379,  "name": "gcs"},
                                                {"containerPort": 10001, "name": "client"},
                                                {"containerPort": 8265,  "name": "dashboard"},
                                                {"containerPort": 8000,  "name": "serve"},
                                            ],
                                            # 6Gi limit: importing vllm/torch for the Serve app build alone needs 2-3GB
                                            "resources": {
                                                "requests": {"cpu": "1", "memory": "3Gi"},
                                                "limits":   {"cpu": "2", "memory": "6Gi"},
                                            },
                                            "volumeMounts": [
                                                {"name": "serve-scripts", "mountPath": "/serve-scripts"},
                                            ],
                                        }
                                    ],
                                    "volumes": [
                                        {
                                            "name": "serve-scripts",
                                            "configMap": {"name": "vllm-serve-script"},
                                        }
                                    ],
                                },
                            },
                        },
                        "workerGroupSpecs": [
                            {
                                "groupName": "gpu-workers",
                                "replicas": 0,
                                "minReplicas": 0,
                                "maxReplicas": 2,  # set max 2 × g4dn.xlarge
                                "rayStartParams": {"num-gpus": "1"},
                                "template": {
                                    "metadata": {
                                        "labels": {"role": "ray-worker", "workload": "gpu"}
                                    },
                                    "spec": {
                                        "tolerations": [
                                            {
                                                "key": "nvidia.com/gpu",
                                                "operator": "Exists",
                                                "effect": "NoSchedule",
                                            }
                                        ],
                                        "nodeSelector": {"workload": "gpu"},
                                        "terminationGracePeriodSeconds": 120,
                                        "initContainers": [
                                            {
                                                # Pull a cached copy of the model from S3 before vLLM starts
                                                "name": "model-cache-sync",
                                                "image": "amazon/aws-cli:2.17.62",
                                                "command": [
                                                    "sh", "-c",
                                                    "aws s3 sync s3://$MODEL_BUCKET/model-cache /model-cache || true",
                                                ],
                                                "env": [
                                                    {"name": "AWS_DEFAULT_REGION", "value": self.region},
                                                    {"name": "MODEL_BUCKET",       "value": model_bucket_name},
                                                ],
                                                "volumeMounts": [
                                                    {"name": "model-cache", "mountPath": "/model-cache"},
                                                ],
                                            }
                                        ],
                                        "containers": [
                                            {
                                                "name": "ray-worker",
                                                "image": "rayproject/ray:2.43.0-gpu",
                                                # g4dn.xlarge has 16GiB RAM; leave headroom for the model + KV cache
                                                "resources": {
                                                    "requests": {
                                                        "cpu":    "2",
                                                        "memory": "12Gi",
                                                    },
                                                    "limits": {
                                                        "cpu":            "4",
                                                        "memory":         "14Gi",
                                                        "nvidia.com/gpu": "1",
                                                    },
                                                },
                                                "env": [
                                                    {"name": "AWS_DEFAULT_REGION", "value": self.region},
                                                    {"name": "MODEL_BUCKET",       "value": model_bucket_name},
                                                ],
                                                "volumeMounts": [
                                                    {"name": "model-cache",   "mountPath": "/model-cache"},
                                                    {"name": "serve-scripts", "mountPath": "/serve-scripts"},
                                                ],
                                            }
                                        ],
                                        "volumes": [
                                            {"name": "model-cache",   "emptyDir": {"sizeLimit": "80Gi"}},
                                            {"name": "serve-scripts", "configMap": {"name": "vllm-serve-script"}},
                                        ],
                                    },
                                },
                            }
                        ],
                    },
                },
            },
        )
        ray_service.node.add_dependency(serve_configmap)
        ray_service.node.add_dependency(nvidia_plugin)

        # NLB to expose the OpenAI-compatible API externally
        nlb_svc = cluster.add_manifest(
            "VllmNlbService",
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": "vllm-nlb",
                    "namespace": "kuberay",
                    "annotations": {
                        "service.beta.kubernetes.io/aws-load-balancer-type": "nlb",
                        "service.beta.kubernetes.io/aws-load-balancer-scheme": "internet-facing",
                    },
                },
                "spec": {
                    "type": "LoadBalancer",
                    "selector": {"role": "ray-head"},
                    "ports": [
                        {"name": "http", "port": 80, "targetPort": 8000, "protocol": "TCP"}
                    ],
                },
            },
        )
        nlb_svc.node.add_dependency(ray_service)

        CfnOutput(
            self, "VllmEndpointNote",
            value="Run: kubectl get svc vllm-nlb -n kuberay -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'",
            description="Command to get the NLB endpoint after deployment",
        )


def _load_serve_script() -> str:
    """Read serve_vllm.py's contents to embed in the ConfigMap above."""
    import pathlib
    script_path = pathlib.Path(__file__).parent / "serve_vllm.py"
    if script_path.exists():
        return script_path.read_text()
    return ""