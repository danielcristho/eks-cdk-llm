import pathlib
import yaml

from aws_cdk import (
    Stack,
    CfnOutput,
    aws_eks as eks,
    aws_iam as iam,
    aws_s3 as s3,
)
from constructs import Construct

# Consts
MANIFESTS_DIR = pathlib.Path(__file__).parent / "manifests"
_STRIP_RESOURCES = {
    ("ServiceAccount", "mysql"),
    ("Service", "mysql"),
    ("Service", "minio-service"),
    ("PersistentVolumeClaim", "minio-pvc"),
    ("PersistentVolumeClaim", "mysql-pv-claim"),
    ("Deployment", "minio"),
    ("Deployment", "mysql"),
    ("ServiceAccount", "kubeflow-pipelines-cache"),
    ("ServiceAccount", "kubeflow-pipelines-cache-deployer-sa"),
    ("Deployment", "cache-server"),
    ("Deployment", "cache-deployer-deployment"),
    ("Service", "cache-server"),
}
_FRONTEND_IMAGE = "ghcr.io/kubeflow/kfp-frontend:2.4.0"
_ARGOEXEC_IMAGE = "quay.io/argoproj/argoexec:v3.4.16"
_S3_IRSA_SERVICE_ACCOUNTS = {"pipeline-runner", "argo"}

class KubeflowStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        cluster: eks.Cluster,
        model_bucket: s3.Bucket,
        db_endpoint: str,
        db_username: str,
        db_password: str,
        s3_role: iam.Role,
        s3_access_key_id: str,
        s3_secret_access_key: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.db_endpoint = db_endpoint
        self.db_username = db_username
        self.db_password = db_password
        self.s3_role = s3_role
        self.s3_access_key_id = s3_access_key_id
        self.s3_secret_access_key = s3_secret_access_key

        cluster_scoped = cluster.add_manifest(
            "KubeflowClusterScoped",
            *_load_docs(MANIFESTS_DIR / "kfp-cluster-scoped.yaml"),
        )

        patched_docs = self._patch_manifest(
            _load_docs(MANIFESTS_DIR / "kfp-pipelines-base.yaml"),
            model_bucket.bucket_name,
        )
        pipelines = cluster.add_manifest("KubeflowPipelines", *patched_docs)
        pipelines.node.add_dependency(cluster_scoped)

        # deploy_model runs as pipeline-runner and needs to repoint the vllm RayService in another namespace
        rollout_rbac = cluster.add_manifest(
            "KfpRayServiceRollout",
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "Role",
                "metadata": {"name": "rayservice-rollout", "namespace": "kuberay"},
                "rules": [{
                    "apiGroups": ["ray.io"],
                    "resources": ["rayservices"],
                    "resourceNames": ["vllm"],
                    "verbs": ["get", "update"],
                }],
            },
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "RoleBinding",
                "metadata": {"name": "rayservice-rollout", "namespace": "kuberay"},
                "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "Role", "name": "rayservice-rollout"},
                "subjects": [{"kind": "ServiceAccount", "name": "pipeline-runner", "namespace": "kubeflow"}],
            },
        )
        rollout_rbac.node.add_dependency(pipelines)

        CfnOutput(
            self, "KfpUiPortForward",
            value="kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80",
            description="No public LB for the KFP dashboard; port-forward to reach it",
        )

    def _patch_manifest(self, docs: list, bucket_name: str) -> list:
        """Swap upstream's in-cluster MySQL/MinIO wiring for RDS + S3 + IRSA."""
        patched = []
        for doc in docs:
            key = (doc.get("kind"), doc.get("metadata", {}).get("name"))
            if key in _STRIP_RESOURCES:
                continue

            if key == ("ConfigMap", "pipeline-install-config"):
                doc["data"]["dbHost"] = self.db_endpoint
                doc["data"]["mysqlHost"] = self.db_endpoint
                doc["data"]["bucketName"] = bucket_name

            elif key == ("Secret", "mysql-secret"):
                doc["stringData"] = {"username": self.db_username, "password": self.db_password}

            elif key == ("Secret", "mlpipeline-minio-artifact"):
                doc["stringData"] = {"accesskey": self.s3_access_key_id, "secretkey": self.s3_secret_access_key}

            elif key == ("Deployment", "ml-pipeline-ui"):
                doc["spec"]["template"]["spec"]["containers"][0]["image"] = _FRONTEND_IMAGE

            elif key == ("ConfigMap", "kfp-launcher"):
                doc["data"]["defaultPipelineRoot"] = f"s3://{bucket_name}/v2/artifacts?region={self.region}"

            elif key == ("Deployment", "workflow-controller"):
                args = doc["spec"]["template"]["spec"]["containers"][0]["args"]
                idx = args.index("--executor-image")
                args[idx + 1] = _ARGOEXEC_IMAGE

            elif key == ("ConfigMap", "workflow-controller-configmap"):
                doc["data"]["artifactRepository"] = (
                    "archiveLogs: true\n"
                    "s3:\n"
                    f'  endpoint: "s3.{self.region}.amazonaws.com"\n'
                    f'  bucket: "{bucket_name}"\n'
                    '  keyFormat: "pipelines/{{workflow.name}}/{{workflow.creationTimestamp.Y}}/'
                    '{{workflow.creationTimestamp.m}}/{{workflow.creationTimestamp.d}}/{{pod.name}}"\n'
                    "  insecure: false\n"
                    "  useSDKCreds: true\n"
                )

            elif key == ("Deployment", "ml-pipeline"):
                env = doc["spec"]["template"]["spec"]["containers"][0]["env"]
                for e in env:
                    if e["name"] == "OBJECTSTORECONFIG_SECURE":
                        e["value"] = "true"
                env += [
                    {"name": "OBJECTSTORECONFIG_HOST", "value": f"s3.{self.region}.amazonaws.com"},
                    {"name": "OBJECTSTORECONFIG_REGION", "value": self.region},
                    {"name": "OBJECTSTORECONFIG_PORT", "value": ""},
                ]

            elif key[0] == "ServiceAccount" and key[1] in _S3_IRSA_SERVICE_ACCOUNTS:
                doc.setdefault("metadata", {}).setdefault("annotations", {})[
                    "eks.amazonaws.com/role-arn"
                ] = self.s3_role.role_arn

            patched.append(doc)
        return patched


def _load_docs(path: pathlib.Path) -> list:
    return [d for d in yaml.safe_load_all(path.read_text()) if d]
