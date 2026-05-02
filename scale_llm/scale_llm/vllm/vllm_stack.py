from aws_cdk import Stack, CfnOutput, aws_eks as eks
from constructs import Construct

import os

class VllmStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, cluster: eks.Cluster, model_bucket_name: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

        # Install NVIDIA device plugin for GPU scheduling
        # Using helm chart
        # Using nodeselector to ensure it runs on GPU node
        cluster.add_helm_chart(
            "NvidiaDevicePlugin",
            chart="nvidia-device-plugin",
            repository="https://nvidia.github.io/k8s-device-plugin",
            namespace="kube-system",
            values={
                "nodeSelector": {"workload": "gpu"},
                "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
            },
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
                                "--dtype", "half",
                                "--quantization", "awq",
                                "--max-model-len", "4096",
                            ],
                            "env": [
                                {"name": "AWS_DEFAULT_REGION", "value": self.region},
                                {"name": "MODEL_BUCKET", "value": model_bucket_name},
                                {"name": "VLLM_PORT", "value": "8000"},
                            ],
                            "ports": [{"containerPort": 8000}],
                            "resources": {
                                "limits": {"nvidia.com/gpu": "1"},
                                "requests": {"memory": "12Gi", "cpu": "2"},
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

        # Internal cluster URL for the vLLM service
        self.vllm_url = os.environ.get("VLLM_URL", "http://vllm.default.svc.cluster.local:80")

        CfnOutput(self, "VllmUrl",
            value=self.vllm_url,
            description="Internal vLLM service URL",
        )