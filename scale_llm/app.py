#!/usr/bin/env python3
import os

from dotenv import load_dotenv
load_dotenv()

import aws_cdk as cdk

from scale_llm.eks.eks_stack import EksStack
from scale_llm.karpenter.karpenter_stack import KarpenterStack
from scale_llm.ray.ray_stack import RayStack
from scale_llm.vllm.vllm_stack import VllmStack

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

app.synth()