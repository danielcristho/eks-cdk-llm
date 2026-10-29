import os

from aws_cdk import Stack, CfnOutput, aws_eks as eks
from constructs import Construct


# chart value -> .env key; modelPrefix is not here on purpose, it lives in Git (manifests/vllm/values.yaml)
CHART_ENV = {
    "rayVersion": "RAY_VERSION",
    "vllmVersion": "VLLM_VERSION",
    "awsCliImage": "AWS_CLI_IMAGE",
    "maxReplicas": "VLLM_MAX_REPLICAS",
    "servedModelName": "SERVED_MODEL_NAME",
    "dtype": "DTYPE",
    "quantization": "QUANTIZATION",
    "maxModelLen": "MAX_MODEL_LEN",
}


class ArgocdStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        cluster: eks.Cluster,
        repo_url: str,
        revision: str,
        model_bucket_name: str,
        script_sha: str,
        prerequisites: list,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        argocd = cluster.add_helm_chart(
            "ArgoCD",
            chart="argo-cd",
            repository="https://argoproj.github.io/argo-helm",
            version="10.9.6",
            release="argocd",
            namespace="argocd",
            create_namespace=True,
            values={
                "global": {"nodeSelector": {"role": "system"}},
                # no SSO or notifications for a demo cluster, saves memory on the system nodes
                "dex": {"enabled": False},
                "notifications": {"enabled": False},
                "configs": {
                    # UI over kubectl port-forward, so plain HTTP
                    "params": {"server.insecure": True},
                    # poll Git every ~30s instead of the default 120s + 60s jitter
                    "cm": {"timeout.reconciliation": "30s", "timeout.reconciliation.jitter": "0s"},
                },
            },
        )

        # Git holds which model is served (manifests/vllm/values.yaml); CDK fills in what this
        # environment knows and the public repo shouldn't: bucket, region, serve script hash
        app = cluster.add_manifest(
            "VllmApplication",
            {
                "apiVersion": "argoproj.io/v1alpha1",
                "kind": "Application",
                "metadata": {
                    "name": "vllm",
                    "namespace": "argocd",
                    # deleting the Application deletes the RayService, so destroy doesn't strand GPU nodes
                    "finalizers": ["resources-finalizer.argocd.argoproj.io"],
                },
                "spec": {
                    "project": "default",
                    "source": {
                        "repoURL": repo_url,
                        "targetRevision": revision,
                        "path": "hypertuning_llm/manifests/vllm",
                        "helm": {
                            "parameters": [
                                {"name": "modelBucket", "value": model_bucket_name},
                                {"name": "region", "value": self.region},
                                {"name": "scriptSha", "value": script_sha},
                                *({"name": k, "value": os.environ[v]} for k, v in CHART_ENV.items()),
                            ],
                        },
                    },
                    "destination": {"server": "https://kubernetes.default.svc", "namespace": "kuberay"},
                    # selfHeal reverts manual kubectl edits back to what Git says
                    "syncPolicy": {"automated": {"prune": True, "selfHeal": True}},
                },
            },
        )
        app.node.add_dependency(argocd)
        for p in prerequisites:
            app.node.add_dependency(p)

        CfnOutput(
            self, "ArgocdNote",
            value="UI: just argocd-ui (localhost:8081, user admin). Password: just argocd-password",
            description="How to reach Argo CD after deployment",
        )
