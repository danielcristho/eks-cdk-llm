import os
from aws_cdk import (
    Stack,
    aws_eks as eks,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_s3 as s3,
    RemovalPolicy,
)
from aws_cdk.lambda_layer_kubectl_v34 import KubectlV34Layer
from constructs import Construct


class EksStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        kubectl_layer = KubectlV34Layer(self, "KubectlLayer")

        vpc = ec2.Vpc(self, "EksVpc", max_azs=2)

        self.cluster = eks.Cluster(
            self,
            "EksCluster",
            version=eks.KubernetesVersion.V1_34,
            vpc=vpc,
            default_capacity=0,
            kubectl_layer=kubectl_layer,
        )

        # General purpose node group (non-GPU)
        self.cluster.add_nodegroup_capacity(
            "ManagedNodeGroup",
            desired_size=2,
            min_size=1,
            max_size=3,
            instance_types=[ec2.InstanceType("t3.medium")],
        )

        # GPU node group for vLLM
        gpu_node_role = iam.Role(
            self,
            "GpuNodeRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEKSWorkerNodePolicy"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryReadOnly"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEKS_CNI_Policy"),
            ],
        )

        self.cluster.add_nodegroup_capacity(
            "GpuNodeGroup",
            desired_size=1,
            min_size=0,
            max_size=2,
            instance_types=[ec2.InstanceType("g4dn.xlarge")],
            node_role=gpu_node_role,
            labels={"workload": "gpu"},
            taints=[
                eks.TaintSpec(
                    key="nvidia.com/gpu",
                    value="true",
                    effect=eks.TaintEffect.NO_SCHEDULE,
                )
            ],
        )

        self.cluster.aws_auth.add_user_mapping(
            iam.User.from_user_name(self, "AdminUser", os.environ["AWS_ADMIN_USER"]),
            groups=["system:masters"],
        )

        # S3 bucket for model weights
        self.model_bucket = s3.Bucket(
            self,
            "ModelBucket",
            removal_policy=RemovalPolicy.RETAIN,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )

        # Allow GPU nodes to read from the model bucket
        self.model_bucket.grant_read(gpu_node_role)