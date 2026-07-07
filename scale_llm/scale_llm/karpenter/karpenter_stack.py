from aws_cdk import Stack, aws_eks as eks, aws_iam as iam
from constructs import Construct


class KarpenterStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        cluster: eks.Cluster,
        cluster_name: str,
        karpenter_controller_role: iam.Role,
        karpenter_node_role: iam.Role,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Install Karpenter controller via Helm
        karpenter_chart = cluster.add_helm_chart(
            "Karpenter",
            chart="karpenter",
            repository="oci://public.ecr.aws/karpenter/karpenter",
            version="1.4.0",
            namespace="karpenter",
            create_namespace=True,
            values={
                "serviceAccount": {
                    "annotations": {
                        "eks.amazonaws.com/role-arn": karpenter_controller_role.role_arn,
                    }
                },
                "settings": {
                    "clusterName": cluster_name,
                },
                "controller": {
                    "resources": {
                        "requests": {"cpu": "250m", "memory": "256Mi"},
                        "limits":   {"cpu": "1",    "memory": "512Mi"},
                    }
                },
            },
        )

        # EC2NodeClass — defines AMI, subnets, and security groups for GPU nodes
        # Subnets and SGs are discovered via the karpenter.sh/discovery tag set in EksStack
        ec2_node_class = cluster.add_manifest(
            "GpuEC2NodeClass",
            {
                "apiVersion": "karpenter.k8s.aws/v1",
                "kind": "EC2NodeClass",
                "metadata": {"name": "gpu"},
                "spec": {
                    "amiSelectorTerms": [{"alias": "al2023@latest"}],
                    "subnetSelectorTerms": [
                        {"tags": {"karpenter.sh/discovery": cluster_name}}
                    ],
                    "securityGroupSelectorTerms": [
                        {"tags": {"karpenter.sh/discovery": cluster_name}}
                    ],
                    "role": karpenter_node_role.role_name,
                    "blockDeviceMappings": [
                        {
                            "deviceName": "/dev/xvda",
                            "ebs": {
                                # 100 GB — vLLM image + model cache needs headroom
                                "volumeSize": "100Gi",
                                "volumeType": "gp3",
                                "encrypted": True,
                                "deleteOnTermination": True,
                            },
                        }
                    ],
                    "tags": {
                        "Name": f"karpenter-gpu-node-{cluster_name}",
                        "project": "eks-cdk-mlops",
                    },
                },
            },
        )
        ec2_node_class.node.add_dependency(karpenter_chart)

        # NodePool to definee which GPU instances Karpenter can provision
        gpu_node_pool = cluster.add_manifest(
            "GpuNodePool",
            {
                "apiVersion": "karpenter.sh/v1",
                "kind": "NodePool",
                "metadata": {"name": "gpu"},
                "spec": {
                    "template": {
                        "metadata": {
                            "labels": {"workload": "gpu"},
                        },
                        "spec": {
                            "nodeClassRef": {
                                "group": "karpenter.k8s.aws",
                                "kind": "EC2NodeClass",
                                "name": "gpu",
                            },
                            "requirements": [
                                {
                                    
                                    "key": "node.kubernetes.io/instance-type",
                                    "operator": "In",
                                    "values": ["g4dn.xlarge"],
                                },
                                {
                                    "key": "karpenter.sh/capacity-type",
                                    "operator": "In",
                                    "values": ["on-demand"],
                                },
                                {
                                    "key": "kubernetes.io/arch",
                                    "operator": "In",
                                    "values": ["amd64"],
                                },
                                {
                                    "key": "kubernetes.io/os",
                                    "operator": "In",
                                    "values": ["linux"],
                                },
                            ],
                            "taints": [
                                {
                                    "key": "nvidia.com/gpu",
                                    "value": "true",
                                    "effect": "NoSchedule",
                                }
                            ],
                        },
                    },
                    # 2× g4dn.xlarge = 8 vCPU
                    "limits": {"nvidia.com/gpu": "2"},
                    "disruption": {
                        # Terminate idle GPU nodes after 30s, try to save cost :)
                        "consolidationPolicy": "WhenEmpty",
                        "consolidateAfter": "30s",
                    },
                },
            },
        )
        gpu_node_pool.node.add_dependency(ec2_node_class)
