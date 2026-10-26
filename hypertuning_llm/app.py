#!/usr/bin/env python3
import os

from dotenv import load_dotenv
load_dotenv()

import aws_cdk as cdk

from hypertuning_llm.eks.eks_stack import EksStack
from hypertuning_llm.karpenter.karpenter_stack import KarpenterStack
from hypertuning_llm.ray.ray_stack import RayStack
from hypertuning_llm.vllm.vllm_stack import VllmStack
from hypertuning_llm.kubeflow.kubeflow_stack import KubeflowStack

app = cdk.App()

env = cdk.Environment(
    account=os.getenv("AWS_DEFAULT_ACCOUNT"),
    region=os.getenv("AWS_DEFAULT_REGION"),
)

# 1. Core infrastructure: VPC, EKS cluster, system node group, Karpenter IAM roles
eks_stack = EksStack(
    app,
    "eks-stack",
    env=env,
)

# 2. Karpenter: controller Helm chart + GPU NodePool + EC2NodeClass
#    Depends on EksStack for cluster, IRSA roles, and tagged subnets.
karpenter_stack = KarpenterStack(
    app,
    "karpenter-stack",
    cluster=eks_stack.cluster,
    cluster_name=eks_stack.cluster_name,
    karpenter_controller_role=eks_stack.karpenter_controller_role,
    karpenter_node_role=eks_stack.karpenter_node_role,
    cluster_security_group_id=eks_stack.cluster_security_group_id,
    env=env,
)
karpenter_stack.add_dependency(eks_stack)

# 3. Ray: KubeRay Operator + RayCluster (head + GPU workers)
#    GPU workers start at 0 replicas; Karpenter provisions nodes on demand.
ray_stack = RayStack(
    app,
    "ray-stack",
    cluster=eks_stack.cluster,
    env=env,
)
ray_stack.add_dependency(karpenter_stack)

# 4. vLLM: NVIDIA device plugin + RayService (vLLM via Ray Serve)
#    Ray Serve autoscales vLLM replicas; Karpenter follows with GPU nodes.
vllm_stack = VllmStack(
    app,
    "vllm-stack",
    cluster=eks_stack.cluster,
    model_bucket_name=eks_stack.model_bucket.bucket_name,
    kuberay_operator=ray_stack.kuberay_operator,
    env=env,
)
vllm_stack.add_dependency(ray_stack)

# 5. Kubeflow Pipelines: RDS + S3-backed KFP standalone (no in-cluster MySQL/MinIO PVCs)
kubeflow_stack = KubeflowStack(
    app,
    "kubeflow-stack",
    cluster=eks_stack.cluster,
    model_bucket=eks_stack.model_bucket,
    db_endpoint=eks_stack.kfp_db_instance.db_instance_endpoint_address,
    db_username=eks_stack.kfp_db_username,
    db_password=eks_stack.kfp_db_password,
    s3_role=eks_stack.kfp_s3_role,
    s3_access_key_id=eks_stack.kfp_s3_access_key_id,
    s3_secret_access_key=eks_stack.kfp_s3_secret_access_key,
    env=env,
)
kubeflow_stack.add_dependency(vllm_stack)

app.synth()
