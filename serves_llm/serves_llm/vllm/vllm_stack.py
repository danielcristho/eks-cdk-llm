import os
from aws_cdk import Stack, aws_eks as eks
from constructs import Construct


class VllmStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, cluster: eks.Cluster, model_bucket_name: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        hf_token = os.environ.get("HF_TOKEN", "")
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        # Install NVIDIA device plugin so EKS can schedule GPU workloads
        cluster.add_helm_chart(
            "NvidiaDevicePlugin",
            chart="nvidia-device-plugin",
            repository="https://nvidia.github.io/k8s-device-plugin",
            namespace="kube-system",
            values={"tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]},
        )

        cluster.add_manifest("VllmDeployment", {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "vllm", "namespace": "default"},
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app": "vllm"}},
                "template": {
                    "metadata": {"labels": {"app": "vllm"}},
                    "spec": {
                        "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
                        "nodeSelector": {"workload": "gpu"},
                        "containers": [{
                            "name": "vllm",
                            "image": "vllm/vllm-openai:latest",
                            "args": [
                                "--model", model_id,
                                "--download-dir", "/model-cache",
                                "--dtype", "float16",
                                "--max-model-len", "4096",
                            ],
                            "env": [
                                {"name": "HF_TOKEN", "value": hf_token},
                                {"name": "AWS_DEFAULT_REGION", "value": self.region},
                                {"name": "MODEL_BUCKET", "value": model_bucket_name},
                            ],
                            "ports": [{"containerPort": 8000}],
                            "resources": {
                                "limits": {"nvidia.com/gpu": "1"},
                                "requests": {"memory": "20Gi", "cpu": "4"},
                            },
                            "volumeMounts": [{"name": "model-cache", "mountPath": "/model-cache"}],
                            "readinessProbe": {
                                "httpGet": {"path": "/health", "port": 8000},
                                "initialDelaySeconds": 120,
                                "periodSeconds": 15,
                            },
                        }],
                        "volumes": [{"name": "model-cache", "emptyDir": {}}],
                    },
                },
            },
        })

        cluster.add_manifest("VllmService", {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "vllm",
                "namespace": "default",
                "annotations": {"service.beta.kubernetes.io/aws-load-balancer-type": "nlb"},
            },
            "spec": {
                "type": "LoadBalancer",
                "selector": {"app": "vllm"},
                "ports": [{"port": 80, "targetPort": 8000, "protocol": "TCP"}],
            },
        })
