from aws_cdk import Stack, CfnOutput, aws_eks as eks
from constructs import Construct

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

        model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

        # NVIDIA device plugin for GPU scheduling
        nvidia_plugin = cluster.add_helm_chart(
            "NvidiaDevicePlugin",
            chart="nvidia-device-plugin",
            repository="https://nvidia.github.io/k8s-device-plugin",
            version="0.17.1",
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

        # RayService — runs vLLM as a Ray Serve deployment
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
                        "      working_dir: /serve-scripts\n"
                        "      pip:\n"
                        "        - vllm==0.8.5\n"
                        f"    args:\n"
                        f"      model: \"{model_id}\"\n"
                        "      dtype: half\n"
                        "      quantization: awq\n"
                        "      max_model_len: 4096\n"
                        "      download_dir: /model-cache\n"
                        "    deployments:\n"
                        "      - name: VLLMDeployment\n"
                        "        num_replicas: 1\n"
                        "        max_ongoing_requests: 5\n"
                        "        autoscaling_config:\n"
                        "          min_replicas: 0\n"
                        "          max_replicas: 2\n"  # max 2 × g4dn.xlarge (account limit)
                        "          target_ongoing_requests: 2\n"
                        "        ray_actor_options:\n"
                        "          num_gpus: 1\n"
                        "          memory: 12288\n"
                    ),
                    "rayClusterConfig": {
                        "rayVersion": "2.44.1",
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
                                            "image": "rayproject/ray-ml:2.44.1-gpu",
                                            "ports": [
                                                {"containerPort": 6379,  "name": "gcs"},
                                                {"containerPort": 10001, "name": "client"},
                                                {"containerPort": 8265,  "name": "dashboard"},
                                                {"containerPort": 8000,  "name": "serve"},
                                            ],
                                            "resources": {
                                                "requests": {"cpu": "1", "memory": "2Gi"},
                                                "limits":   {"cpu": "2", "memory": "4Gi"},
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
                                "maxReplicas": 2,  # max 2 × g4dn.xlarge = 8 vCPU (account limit)
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
                                        "containers": [
                                            {
                                                "name": "ray-worker",
                                                "image": "rayproject/ray-ml:2.44.1-gpu",
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
    import pathlib
    script_path = pathlib.Path(__file__).parent / "serve_vllm.py"
    if script_path.exists():
        return script_path.read_text()
    return ""