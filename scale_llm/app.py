#!/usr/bin/env python3
import os

from dotenv import load_dotenv
load_dotenv()

import aws_cdk as cdk

from scale_llm.eks.eks_stack import EksStack
from scale_llm.vllm.vllm_stack import VllmStack
from scale_llm.ray.ray_stack import RayStack
from scale_llm.karpenter.karpenter_stack import KarpenterStack

app = cdk.App()

env = cdk.Environment(
    account=os.getenv("AWS_DEFAULT_ACCOUNT"),
    region=os.getenv("AWS_DEFAULT_REGION"),
)

eks_stack = EksStack(
    app, 
    "eks-stack", 
    env=env
)

ray_stack = RayStack(
    app,
    "ray-stack",
    cluster=eks_stack.cluster,
    env=env,
)

karpenter_stack = KarpenterStack(
    app,
    "karpenter-stack",
    cluster=eks_stack.cluster,
    env=env,
)

VllmStack(
    app,
    "vllm-stack",
    cluster=eks_stack.cluster,
    model_bucket_name=eks_stack.model_bucket.bucket_name,
    env=env,
)

app.synth()