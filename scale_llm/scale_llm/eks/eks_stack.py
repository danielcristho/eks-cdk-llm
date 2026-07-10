from aws_cdk import (
    Stack,
    Tags,
    CfnJson,
    aws_eks as eks,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_s3 as s3,
    RemovalPolicy,
)
from aws_cdk.lambda_layer_kubectl_v34 import KubectlV34Layer
from constructs import Construct

import os

class EksStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        kubectl_layer = KubectlV34Layer(self, "KubectlLayer")

        # VPC for the cluster, 2 AZs is enough for this
        self.vpc = ec2.Vpc(self, "EksVpc", max_azs=2)

        cluster_name = os.environ.get("CLUSTER_NAME", "eks-llm-scale")
        self.cluster_name = cluster_name

        # default_capacity=0, node group defined separately below
        self.cluster = eks.Cluster(
            self,
            cluster_name,
            cluster_name=cluster_name,
            version=eks.KubernetesVersion.V1_34,
            vpc=self.vpc,
            default_capacity=0,
            kubectl_layer=kubectl_layer,
        )

        """
        CPU node group for system pods (CoreDNS, kube-proxy) and the Ray head.
        t3.large, not t3.medium, since importing vllm/torch for Ray Serve's
        app build step alone needs 2-3GB, which OOMs a smaller node.
        """
        self.cluster.add_nodegroup_capacity(
            "ManagedNodeGroup",
            desired_size=3,
            min_size=3,
            max_size=4,
            disk_size=50,
            instance_types=[ec2.InstanceType("t3.large")],
            ami_type=eks.NodegroupAmiType.AL2023_X86_64_STANDARD,
            labels={"role": "system"},
        )

        # Give ourselves kubectl access to the cluster
        self.cluster.aws_auth.add_user_mapping(
            iam.User.from_user_name(self, "AdminUser", os.environ["AWS_ADMIN_USER"]),
            groups=["system:masters"],
        )

        # IAM role for EC2 instances provisioned by Karpenter
        self.karpenter_node_role = iam.Role(
            self,
            "KarpenterNodeRole",
            role_name=f"KarpenterNodeRole-{cluster_name}",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEKSWorkerNodePolicy"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryReadOnly"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEKS_CNI_Policy"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore"),
            ],
        )

        # Instance profile Karpenter attaches to the nodes it launches
        iam.CfnInstanceProfile(
            self,
            "KarpenterInstanceProfile",
            instance_profile_name=f"KarpenterNodeInstanceProfile-{cluster_name}",
            roles=[self.karpenter_node_role.role_name],
        )

        # Let nodes bootstrapped with this role join the cluster
        self.cluster.aws_auth.add_role_mapping(
            self.karpenter_node_role,
            groups=["system:bootstrappers", "system:nodes"],
            username="system:node:{{EC2PrivateDNSName}}",
        )

        # IRSA condition for the Karpenter controller role, below
        oidc_issuer = self.cluster.cluster_open_id_connect_issuer
        irsa_condition = CfnJson(
            self,
            "KarpenterIrsaCondition",
            value={
                f"{oidc_issuer}:sub": "system:serviceaccount:karpenter:karpenter",
                f"{oidc_issuer}:aud": "sts.amazonaws.com",
            },
        )

        # Role Karpenter's controller pod assumes via IRSA
        self.karpenter_controller_role = iam.Role(
            self,
            "KarpenterControllerRole",
            role_name=f"KarpenterControllerRole-{cluster_name}",
            assumed_by=iam.FederatedPrincipal(
                federated=self.cluster.open_id_connect_provider.open_id_connect_provider_arn,
                assume_role_action="sts:AssumeRoleWithWebIdentity",
                conditions={"StringEquals": irsa_condition},
            ),
        )

        # Karpenter's controller talks to the K8s API as this IAM role
        self.cluster.aws_auth.add_role_mapping(
            self.karpenter_controller_role,
            groups=["system:masters"],
        )

        # Lets Karpenter create/destroy the EC2 instances it launches
        self.karpenter_controller_role.add_to_policy(
            iam.PolicyStatement(
                sid="KarpenterEC2",
                actions=[
                    "ec2:CreateFleet",
                    "ec2:CreateLaunchTemplate",
                    "ec2:CreateTags",
                    "ec2:DeleteLaunchTemplate",
                    "ec2:DescribeAvailabilityZones",
                    "ec2:DescribeImages",
                    "ec2:DescribeInstances",
                    "ec2:DescribeInstanceTypeOfferings",
                    "ec2:DescribeInstanceTypes",
                    "ec2:DescribeLaunchTemplates",
                    "ec2:DescribeSecurityGroups",
                    "ec2:DescribeSpotPriceHistory",
                    "ec2:DescribeSubnets",
                    "ec2:RunInstances",
                    "ec2:TerminateInstances",
                ],
                resources=["*"],
            )
        )

        # Lets Karpenter pass the node role to EC2 at launch time
        self.karpenter_controller_role.add_to_policy(
            iam.PolicyStatement(
                sid="KarpenterIAM",
                actions=["iam:PassRole"],
                resources=[self.karpenter_node_role.role_arn],
            )
        )

        """
        EC2NodeClass.spec.role (not .instanceProfile) makes Karpenter create
        and manage its own instance profile per NodeClass, so it needs full
        lifecycle permissions on instance profiles, not just PassRole.
        """
        self.karpenter_controller_role.add_to_policy(
            iam.PolicyStatement(
                sid="KarpenterInstanceProfile",
                actions=[
                    "iam:CreateInstanceProfile",
                    "iam:TagInstanceProfile",
                    "iam:AddRoleToInstanceProfile",
                    "iam:RemoveRoleFromInstanceProfile",
                    "iam:DeleteInstanceProfile",
                    "iam:GetInstanceProfile",
                ],
                resources=["*"],
            )
        )

        # Needed to look up AMIs and price out instance types
        self.karpenter_controller_role.add_to_policy(
            iam.PolicyStatement(
                sid="KarpenterSSMPricing",
                actions=["ssm:GetParameter", "pricing:GetProducts"],
                resources=["*"],
            )
        )

        # Needed to read cluster info when reconciling NodePools
        self.karpenter_controller_role.add_to_policy(
            iam.PolicyStatement(
                sid="KarpenterEKS",
                actions=["eks:DescribeCluster"],
                resources=[self.cluster.cluster_arn],
            )
        )

        # Tag subnets so Karpenter can discover them via EC2NodeClass
        for subnet in self.vpc.private_subnets:
            Tags.of(subnet).add("karpenter.sh/discovery", cluster_name)

        # Passed to KarpenterStack; Tags.of() is a no-op on this imported SG
        self.cluster_security_group_id = self.cluster.cluster_security_group_id

        # S3 bucket for model weights
        self.model_bucket = s3.Bucket(
            self,
            "ModelBucket",
            bucket_name=os.environ.get("AWS_BUCKET"),
            removal_policy=RemovalPolicy.RETAIN,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )

        # Allow Karpenter GPU nodes to read model weights
        self.model_bucket.grant_read(self.karpenter_node_role)