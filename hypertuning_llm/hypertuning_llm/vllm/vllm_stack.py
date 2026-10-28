from aws_cdk import Stack, CfnOutput, aws_eks as eks
from constructs import Construct
import hashlib

class VllmStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        cluster: eks.Cluster,
        kuberay_operator,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # passed to the RayService so a serve_vllm.py change restarts the Serve app
        self.script_sha = hashlib.sha256(_load_serve_script().encode()).hexdigest()[:12]
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

        # The RayService itself lives in manifests/vllm (Helm chart synced by Argo CD); these are what it needs
        self.prerequisites = [serve_configmap, nvidia_plugin]

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
