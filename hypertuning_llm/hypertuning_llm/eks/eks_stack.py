from aws_cdk import (
    Stack,
    Tags,
    CfnJson,
    CustomResource,
    Duration,
    aws_eks as eks,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_s3 as s3,
    aws_rds as rds,
    custom_resources as cr,
    RemovalPolicy,
)
from aws_cdk.lambda_layer_kubectl_v34 import KubectlV34Layer
from constructs import Construct

import os

class EksStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        kubectl_layer = KubectlV34Layer(self, "KubectlLayer")

        # single NAT gateway is fine for a demo cluster; AZ-redundant NAT isn't worth 2x the hourly charge here
        self.vpc = ec2.Vpc(self, "EksVpc", max_azs=2, nat_gateways=1)
        self.vpc.add_gateway_endpoint("S3GatewayEndpoint", service=ec2.GatewayVpcEndpointAwsService.S3)

        cluster_name = os.environ.get("CLUSTER_NAME", "eks-llm-scale")
        self.cluster_name = cluster_name

        self.cluster = eks.Cluster(
            self,
            cluster_name,
            cluster_name=cluster_name,
            version=eks.KubernetesVersion.V1_34,
            vpc=self.vpc,
            default_capacity=0,
            kubectl_layer=kubectl_layer,
        )

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

        # comma-separated, e.g. "onomi,comday"
        for i, user_name in enumerate(os.environ["AWS_ADMIN_USER"].split(",")):
            self.cluster.aws_auth.add_user_mapping(
                iam.User.from_user_name(self, f"AdminUser{i or ''}", user_name.strip()),
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

        self.cluster.aws_auth.add_role_mapping(
            self.karpenter_node_role,
            groups=["system:bootstrappers", "system:nodes"],
            username="system:node:{{EC2PrivateDNSName}}",
        )

        oidc_issuer = self.cluster.cluster_open_id_connect_issuer
        irsa_condition = CfnJson(
            self,
            "KarpenterIrsaCondition",
            value={
                f"{oidc_issuer}:sub": "system:serviceaccount:karpenter:karpenter",
                f"{oidc_issuer}:aud": "sts.amazonaws.com",
            },
        )

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

        self.karpenter_controller_role.add_to_policy(
            iam.PolicyStatement(
                sid="KarpenterSSMPricing",
                actions=["ssm:GetParameter", "pricing:GetProducts"],
                resources=["*"],
            )
        )

        self.karpenter_controller_role.add_to_policy(
            iam.PolicyStatement(
                sid="KarpenterEKS",
                actions=["eks:DescribeCluster"],
                resources=[self.cluster.cluster_arn],
            )
        )

        for subnet in self.vpc.private_subnets:
            Tags.of(subnet).add("karpenter.sh/discovery", cluster_name)

        self.cluster_security_group_id = self.cluster.cluster_security_group_id

        self.model_bucket = s3.Bucket.from_bucket_name(
            self, "ModelBucket", os.environ.get("AWS_BUCKET")
        )

        # Allow Karpenter GPU nodes to read model weights
        self.model_bucket.grant_read(self.karpenter_node_role)

        kfp_db_sg = ec2.SecurityGroup(
            self, "KfpDbSecurityGroup", vpc=self.vpc, description="Allow MySQL from the EKS cluster"
        )
        kfp_db_sg.add_ingress_rule(
            ec2.Peer.security_group_id(self.cluster_security_group_id),
            ec2.Port.tcp(3306),
            "EKS cluster to KFP metadata DB",
        )

        self.kfp_db_instance = rds.DatabaseInstance(
            self,
            "KfpMetadataDb",
            engine=rds.DatabaseInstanceEngine.mysql(version=rds.MysqlEngineVersion.VER_8_0),
            instance_type=ec2.InstanceType.of(ec2.InstanceClass.T4G, ec2.InstanceSize.MICRO),
            vpc=self.vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            security_groups=[kfp_db_sg],
            credentials=rds.Credentials.from_generated_secret("kfpadmin"),
            allocated_storage=20,
            multi_az=False,
            removal_policy=RemovalPolicy.DESTROY,
            deletion_protection=False,
        )

        db_secret_reader_fn = lambda_.Function(
            self, "KfpDbSecretReaderFn",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="index.handler",
            timeout=Duration.seconds(30),
            code=lambda_.Code.from_inline(
                "import json, boto3\n"
                "def handler(event, context):\n"
                "    if event['RequestType'] == 'Delete':\n"
                "        return {}\n"
                "    secret = boto3.client('secretsmanager').get_secret_value(\n"
                "        SecretId=event['ResourceProperties']['SecretArn'])\n"
                "    data = json.loads(secret['SecretString'])\n"
                "    return {'Data': {'Username': data['username'], 'Password': data['password']}}\n"
            ),
        )
        self.kfp_db_instance.secret.grant_read(db_secret_reader_fn)
        db_secret_reader = CustomResource(
            self, "KfpDbSecretReader",
            service_token=cr.Provider(self, "KfpDbSecretReaderProvider", on_event_handler=db_secret_reader_fn).service_token,
            properties={"SecretArn": self.kfp_db_instance.secret.secret_arn},
        )
        self.kfp_db_username = db_secret_reader.get_att_string("Username")
        self.kfp_db_password = db_secret_reader.get_att_string("Password")

        kfp_irsa_condition = CfnJson(
            self,
            "KfpS3IrsaCondition",
            value={
                f"{oidc_issuer}:aud": "sts.amazonaws.com",
                f"{oidc_issuer}:sub": [
                    f"system:serviceaccount:kubeflow:{sa}" for sa in ("pipeline-runner", "argo")
                ],
            },
        )
        self.kfp_s3_role = iam.Role(
            self,
            "KfpS3Role",
            role_name=f"KfpS3Role-{cluster_name}",
            assumed_by=iam.FederatedPrincipal(
                federated=self.cluster.open_id_connect_provider.open_id_connect_provider_arn,
                assume_role_action="sts:AssumeRoleWithWebIdentity",
                conditions={"ForAnyValue:StringEquals": kfp_irsa_condition},
            ),
        )
        self.model_bucket.grant_read_write(self.kfp_s3_role)

        kfp_s3_user = iam.User(self, "KfpS3User", user_name=f"KfpS3User-{cluster_name}")
        self.model_bucket.grant_read_write(kfp_s3_user)
        kfp_s3_access_key = iam.AccessKey(self, "KfpS3UserAccessKey", user=kfp_s3_user)
        self.kfp_s3_access_key_id = kfp_s3_access_key.access_key_id
        self.kfp_s3_secret_access_key = kfp_s3_access_key.secret_access_key.unsafe_unwrap()
