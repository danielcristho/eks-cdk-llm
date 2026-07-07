from aws_cdk import Stack, CfnOutput, aws_eks as eks
from constructs import Construct


class RayStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        cluster: eks.Cluster,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Install KubeRay operator
        self.kuberay_operator = cluster.add_helm_chart(
            "KubeRayOperator",
            chart="kuberay-operator",
            repository="https://ray-project.github.io/kuberay-helm/",
            version="1.3.0",
            namespace="kuberay",
            create_namespace=True,
            values={
                "resources": {
                    "requests": {"cpu": "100m", "memory": "128Mi"},
                    "limits":   {"cpu": "500m", "memory": "256Mi"},
                }
            },
        )

        # RayCluster — head node (CPU) + GPU workers (0 --> N via autoscaler)
        ray_cluster = cluster.add_manifest(
            "RayCluster",
            {
                "apiVersion": "ray.io/v1",
                "kind": "RayCluster",
                "metadata": {
                    "name": "vllm-cluster",
                    "namespace": "kuberay",
                    "labels": {"app": "vllm-cluster"},
                },
                "spec": {
                    "rayVersion": "2.44.1",
                    "enableInTreeAutoscaling": True,
                    "autoscalerOptions": {
                        "upscalingMode": "Default",
                        "idleTimeoutSeconds": 60,
                        "resources": {
                            "requests": {"cpu": "250m", "memory": "256Mi"},
                            "limits":   {"cpu": "500m", "memory": "512Mi"},
                        },
                    },
                    "headGroupSpec": {
                        "rayStartParams": {
                            "dashboard-host": "0.0.0.0",
                            "num-gpus": "0",
                        },
                        "serviceType": "ClusterIP",
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
                                        "env": [
                                            {"name": "RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING", "value": "1"},
                                        ],
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
                            "maxReplicas": 2,
                            "rayStartParams": {"num-gpus": "1"},
                            "template": {
                                "metadata": {"labels": {"role": "ray-worker", "workload": "gpu"}},
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
                                            "volumeMounts": [
                                                {"name": "model-cache", "mountPath": "/model-cache"}
                                            ],
                                        }
                                    ],
                                    "volumes": [
                                        {"name": "model-cache", "emptyDir": {"sizeLimit": "80Gi"}}
                                    ],
                                },
                            },
                        }
                    ],
                },
            },
        )
        ray_cluster.node.add_dependency(self.kuberay_operator)

        # ClusterIP service to reach the Ray head from inside the cluster
        ray_serve_svc = cluster.add_manifest(
            "RayServeService",
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": "ray-serve",
                    "namespace": "kuberay",
                    "labels": {"app": "vllm-cluster"},
                },
                "spec": {
                    "type": "ClusterIP",
                    "selector": {"role": "ray-head"},
                    "ports": [
                        {"name": "serve",     "port": 8000,  "targetPort": 8000},
                        {"name": "dashboard", "port": 8265,  "targetPort": 8265},
                        {"name": "client",    "port": 10001, "targetPort": 10001},
                    ],
                },
            },
        )
        ray_serve_svc.node.add_dependency(ray_cluster)

        CfnOutput(
            self, "RayDashboard",
            value="http://ray-serve.kuberay.svc.cluster.local:8265",
            description="Ray Dashboard (port-forward to access locally)",
        )
