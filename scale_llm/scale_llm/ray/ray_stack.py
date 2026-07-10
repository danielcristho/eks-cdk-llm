from aws_cdk import Stack, aws_eks as eks
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

        # Install KubeRay operator; exposed so VllmStack can depend on it
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
