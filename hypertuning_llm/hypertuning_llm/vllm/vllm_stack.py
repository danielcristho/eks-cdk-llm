from aws_cdk import Stack, CfnOutput, aws_eks as eks
from constructs import Construct
import hashlib
import os

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

        model_uri = f"s3://{model_bucket_name}/{os.environ['MODEL_PREFIX']}"
        ray_version = os.environ["RAY_VERSION"]
        ray_image = f"rayproject/ray:{ray_version}-gpu"
        max_replicas = int(os.environ["VLLM_MAX_REPLICAS"])
        script_sha = hashlib.sha256(_load_serve_script().encode()).hexdigest()[:12]
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

        # Runs vLLM as a Ray Serve deployment, routes requests across replicas, autoscaler scales workers, Karpenter provisions GPU nodes
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
                        f"        - vllm=={os.environ['VLLM_VERSION']}\n"
                        f"        - ray[serve]=={ray_version}\n"
                        "        - numpy<2\n"
                        "      env_vars:\n"
                        "        PYTHONPATH: /serve-scripts\n"
                        "    args:\n"
                        "      model: /model-cache/current\n"
                        f"      served_model_name: {os.environ['SERVED_MODEL_NAME']}\n"
                        f"      dtype: {os.environ['DTYPE']}\n"
                        f"      quantization: {os.environ['QUANTIZATION']}\n"
                        f"      max_model_len: {os.environ['MAX_MODEL_LEN']}\n"
                        "      download_dir: /model-cache\n"
                        f"      _script_sha: \"{script_sha}\"\n"
                        "    deployments:\n"
                        "      - name: VLLMDeployment\n"
                        "        max_ongoing_requests: 5\n"
                        "        autoscaling_config:\n"
                        # 1, not 0: a new cluster takes traffic once a replica has the new model loaded
                        "          min_replicas: 1\n"
                        f"          max_replicas: {max_replicas}\n"  # set max 2 × g4dn.xlarge
                        "          target_ongoing_requests: 2\n"
                        "        ray_actor_options:\n"
                        "          num_gpus: 1\n"
                        "          memory: 12288\n"
                    ),
                    "rayClusterConfig": {
                        "rayVersion": ray_version,
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
                                            "image": ray_image,
                                            "ports": [
                                                {"containerPort": 6379,  "name": "gcs"},
                                                {"containerPort": 10001, "name": "client"},
                                                {"containerPort": 8265,  "name": "dashboard"},
                                                {"containerPort": 8000,  "name": "serve"},
                                            ],
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
                                "maxReplicas": max_replicas,  # set max 2 × g4dn.xlarge
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
                                                "name": "model-cache-sync",
                                                "image": os.environ["AWS_CLI_IMAGE"],
                                                "command": [
                                                    "sh", "-c",
                                                    "aws s3 sync \"$MODEL_URI\" /model-cache/current",
                                                ],
                                                "env": [
                                                    {"name": "AWS_DEFAULT_REGION", "value": self.region},
                                                    {"name": "MODEL_URI",          "value": model_uri},
                                                ],
                                                "volumeMounts": [
                                                    {"name": "model-cache", "mountPath": "/model-cache"},
                                                ],
                                            }
                                        ],
                                        "containers": [
                                            {
                                                "name": "ray-worker",
                                                "image": ray_image,
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

        CfnOutput(
            self, "VllmEndpointNote",
            value="In-cluster: http://vllm-serve-svc.kuberay.svc.cluster.local:8000. "
                  "Access from local: kubectl port-forward -n kuberay svc/vllm-serve-svc 8000:8000",
            description="How to reach vLLM after deployment",
        )


def _load_serve_script() -> str:
    """Read serve_vllm.py's contents to embed in the ConfigMap above."""
    import pathlib
    script_path = pathlib.Path(__file__).parent / "serve_vllm.py"
    if script_path.exists():
        return script_path.read_text()
    return ""
