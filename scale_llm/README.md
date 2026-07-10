# LLMs on EKS: Scaling with Ray & Karpenter

Hi there, it's been 2 months.

Back in May, we deployed an LLM on Amazon EKS using vLLM. It gave me a much better understanding of what it takes to serve an LLM in production.

But after getting it up and running, one question kept coming to mind: *What happens when a single GPU isn't enough?*

![Lindalee Cat GIF](https://res.cloudinary.com/diunivf9n/image/upload/v1783432480/lindalee-cat_em4vzn.gif)

Sure, we could add more replicas. But those replicas still need GPU nodes, and keeping expensive GPU instances running 24/7 just in case traffic spikes doesn't sound like a great idea.

So in this post, we'll extend the previous setup by introducing [Ray](https://github.com/ray-project/ray) for distributed inference and [Karpenter for AWS](https://github.com/aws/karpenter-provider-aws) for dynamic GPU node provisioning, allowing the cluster to scale up (and back down) as demand changes.

## TL;DR

- Running distributed LLM inference with Ray
- Autoscaling worker pods using KubeRay
- Provisioning GPU nodes dynamically with Karpenter
- Extending the previous vLLM deployment to scale on demand

## Why Ray?

Kubernetes can certainly scale pods, but serving LLMs comes with a few extra challenges. A single GPU can only handle so much, and once traffic grows, we need a way to distribute inference requests accross multiple workers instead of relying on a single pod.

[Ray](https://github.com/ray-project/ray) is built for distirubuted AI workloads, making it natural fit here. I mainly chose it because:

- **Distributed inference** — spreads requests across multiple workers instead of relying on a single replica.
- **Built-in autoscaler** — adjusts the number of Ray workers based on workload.
- **Kubernetes-native** — integrates nicely with EKS through [KubeRay](https://github.com/ray-project/kuberay).

Rather than treating each pod as an isolated deployment, Ray lets them work together as a single inference cluster.

## Why Karpenter?

Ray can create additional worker pods, but it can't magically create GPU node. If every GPU node is already busy, those worker pods will remain in the 'Pending' state until Kubernetes finds available capacity.

That's where [Karpenter](https://github.com/aws/karpenter-provider-aws) comes in:

- **Provision nodes on demand** — launches new GPU instances when the cluster runs out of capacity.
- **No fixed node groups** — avoids keeping idle GPU instances running all day.
- **Scale back down** — automatically removes unused nodes to reduce infrastructure costs.

Together, [Ray](https://github.com/ray-project/ray) and [Karpenter](https://github.com/aws/karpenter-provider-aws) solve different parts of the scaling problem. Ray decides *when* more workers are needed, while Karpenter makes sure there's actually somewhere for those workers to run.

## What We Tryna Build

This time, the architecture is a little different.

Instead of running a single vLLM pod, we'll build a small Ray cluster on Amazon EKS. Ray will distribute inference requests across multiple workers, while Karpenter automatically provisions GPU nodes whenever the cluster needs more capacity.

The flow looks like this:

- User sends an inference request
- Ray routes the request to an available worker
- Workers run vLLM to generate responses
- Ray creates additional workers as traffic increases
- Karpenter provisions new GPU nodes when the cluster runs out of capacity
- Unused workers and nodes are removed once traffic drops

![Project Arch](https://res.cloudinary.com/diunivf9n/image/upload/v1783517665/project_arch_servesllm_g8whow.webp)

### The Stack

We'll reuse most of the infrastructure from the previous post. This time, we're adding a few more components to make the deployment capable of scaling automatically.

- **Amazon EKS** - runs the Kubernetes workloads, including Ray and vLLM.
- **Amazon S3** - stores artifacts such as model files used by the inference service.
- **AWS CDK** - provisions the infrastructure as code.
- **Ray** - distributes inference requests across multiple workers.
- **Karpenter** - provisions GPU nodes on demand.
- **vLLM** - serves the LLM through an OpenAI-compatible API.
- **Meta Llama 3.1** - the language model used for inference.
- **NVIDIA Device Plugin** — exposes GPU resources so workloads like vLLM can request and use NVIDIA GPUs.

You can see the stack below.

![Stack Overview](https://res.cloudinary.com/diunivf9n/image/upload/v1783518249/scale_llm_xr64jp.webp)

## The Code

We'll reuse most of the infrastructure from the previous post and build on top of it. This project is organized into separate CDK stacks, with each stack responsible for a specific part of the deployment.

```text
scale_llm/
├── eks/
├── karpenter/
├── ray/
└── vllm/
```

This separation keeps each component independent and easier to maintain. Let's go through each stack and see what it does.

### EKS Stack

The `EksStack` reuses the same cluster setup from the previous post, but this time there's only one node group instead of two.

```python
# CPU nodes, runs system pods (CoreDNS, kube-proxy) and the Ray head
self.cluster.add_nodegroup_capacity(
    "ManagedNodeGroup",
    desired_size=3,
    min_size=3,
    max_size=4,
    disk_size=50,
    instance_types=[ec2.InstanceType("t3.medium")],
    ami_type=eks.NodegroupAmiType.AL2023_X86_64_STANDARD,
    labels={"role": "system"},
)
```

No GPU node group this time. GPU capacity is fully dynamic now, Karpenter provisions and terminates GPU nodes on its own, so there's nothing static to define here.

`desired_size` went from 2 to 3, and `disk_size` from the default 20GB to 50GB. The Ray head pod ended up sharing this node group with CoreDNS and kube-proxy, and once you add KubeRay's autoscaler sidecar (which rides along with the head pod), 2 nodes just didn't leave enough CPU headroom. The bigger disk is there because pulling Ray's CUDA image on a 20GB root volume was triggering disk-pressure evictions.

Since Karpenter creates its own EC2 instances and needs to register them with the cluster, it needs an IAM role of its own:

```python
self.karpenter_node_role = iam.Role(
    self, "KarpenterNodeRole",
    assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
    managed_policies=[
        iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEKSWorkerNodePolicy"),
        iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryReadOnly"),
        iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEKS_CNI_Policy"),
        iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore"),
    ],
)
```

The Karpenter controller itself also needs a role, one it assumes via IRSA so it can call the EC2 and EKS APIs:

```python
oidc_issuer = self.cluster.cluster_open_id_connect_issuer
irsa_condition = CfnJson(self, "KarpenterIrsaCondition", value={
    f"{oidc_issuer}:sub": "system:serviceaccount:karpenter:karpenter",
    f"{oidc_issuer}:aud": "sts.amazonaws.com",
})

self.karpenter_controller_role = iam.Role(
    self, "KarpenterControllerRole",
    assumed_by=iam.FederatedPrincipal(
        federated=self.cluster.open_id_connect_provider.open_id_connect_provider_arn,
        assume_role_action="sts:AssumeRoleWithWebIdentity",
        conditions={"StringEquals": irsa_condition},
    ),
)

# Karpenter talks to the K8s API server as this IAM role, same as `aws eks get-token` does,
# not through its pod's ServiceAccount token — so it also needs an aws-auth mapping.
self.cluster.aws_auth.add_role_mapping(self.karpenter_controller_role, groups=["system:masters"])
```

Last, we tag the private subnets so Karpenter knows where it's allowed to launch nodes:

```python
for subnet in self.vpc.private_subnets:
    Tags.of(subnet).add("karpenter.sh/discovery", cluster_name)
```

A few things worth noting:

- IRSA alone isn't enough for the controller, it also needs the `aws-auth` role mapping, otherwise it gets stuck on "the server has asked for the client to provide credentials"
- the cluster's security group is passed to the Karpenter stack by ID instead of by tag — `Tags.of(cluster.cluster_security_group)` is a silent no-op in CDK since it's an imported reference, not a resource this stack owns
- `karpenter_node_role` also gets a matching `aws-auth` mapping (`system:bootstrappers` / `system:nodes`), same as any other node group's role

:::note
The system node group is now:

| Node      | vCPU| Memory  | Count |
|-----------|-----|---------|-------|
| t3.large  | 2   | 8Gi     | 3     |

GPU nodes are launched by Karpenter, so there's no fixed count, they only exist while there's demand for them.
:::

### Karpenter Stack

The `KarpenterStack` installs the Karpenter controller via Helm, then defines the two CRDs that tell it what to provision: an `EC2NodeClass` (AMI, subnets, security group, IAM role) and a `NodePool` (instance types, limits, disruption behavior).

```python
karpenter_chart = cluster.add_helm_chart(
    "Karpenter",
    chart="karpenter",
    repository="oci://public.ecr.aws/karpenter/karpenter",
    version="1.4.0",
    namespace="karpenter",
    create_namespace=True,
    values={
        "serviceAccount": {
            "name": "karpenter",
            "annotations": {"eks.amazonaws.com/role-arn": karpenter_controller_role.role_arn},
        },
        "settings": {"clusterName": cluster_name},
    },
)
```

`serviceAccount.name` is pinned to `"karpenter"` instead of letting Helm auto-generate one from the release name. The IRSA trust policy back in `EksStack` hardcodes `system:serviceaccount:karpenter:karpenter` as the expected subject, so if the actual SA name doesn't match that exactly, the pod fails `sts:AssumeRoleWithWebIdentity` with an access denied.

Next, the `EC2NodeClass`:

```python
ec2_node_class = cluster.add_manifest("GpuEC2NodeClass", {
    "apiVersion": "karpenter.k8s.aws/v1",
    "kind": "EC2NodeClass",
    "metadata": {"name": "gpu"},
    "spec": {
        "amiSelectorTerms": [{"alias": "al2023@latest"}],
        "subnetSelectorTerms": [{"tags": {"karpenter.sh/discovery": cluster_name}}],
        "securityGroupSelectorTerms": [{"id": cluster_security_group_id}],
        "role": karpenter_node_role.role_name,
        "blockDeviceMappings": [{
            "deviceName": "/dev/xvda",
            "ebs": {"volumeSize": "100Gi", "volumeType": "gp3", "encrypted": True, "deleteOnTermination": True},
        }],
    },
})
```

And the `NodePool`:

```python
gpu_node_pool = cluster.add_manifest("GpuNodePool", {
    "apiVersion": "karpenter.sh/v1",
    "kind": "NodePool",
    "metadata": {"name": "gpu"},
    "spec": {
        "template": {
            "metadata": {"labels": {"workload": "gpu"}},
            "spec": {
                "nodeClassRef": {"group": "karpenter.k8s.aws", "kind": "EC2NodeClass", "name": "gpu"},
                "requirements": [
                    {"key": "node.kubernetes.io/instance-type", "operator": "In", "values": ["g4dn.xlarge"]},
                    {"key": "karpenter.sh/capacity-type", "operator": "In", "values": ["on-demand"]},
                ],
                "taints": [{"key": "nvidia.com/gpu", "value": "true", "effect": "NoSchedule"}],
            },
        },
        "limits": {"nvidia.com/gpu": "2"},
        "disruption": {"consolidationPolicy": "WhenEmpty", "consolidateAfter": "30s"},
    },
})
```

A few things worth noting:

- `spec.role` (not `spec.instanceProfile`) tells Karpenter to create and manage its own IAM instance profile at runtime, that's why `EksStack` grants the controller role extra `iam:CreateInstanceProfile` / `iam:GetInstanceProfile` permissions
- the `nvidia.com/gpu=true:NoSchedule` taint keeps everything except GPU workloads off these nodes, it's matched by a toleration on the Ray GPU worker pods in `VllmStack`
- `limits.nvidia.com/gpu: "2"` caps Karpenter at 2 GPU nodes at a time, matching this account's GPU instance quota
- `consolidateAfter: "30s"` terminates idle GPU nodes fast, `g4dn.xlarge` isn't something you want sitting idle

### Ray Stack

The `RayStack` is small, it only installs the KubeRay operator:

```python
self.kuberay_operator = cluster.add_helm_chart(
    "KubeRayOperator",
    chart="kuberay-operator",
    repository="https://ray-project.github.io/kuberay-helm/",
    version="1.3.0",
    namespace="kuberay",
    create_namespace=True,
    values={
        "resources": {
            "requests": {"cpu": "100m", "memory": "128Mi"},
            "limits": {"cpu": "500m", "memory": "256Mi"},
        }
    },
)
```

The operator itself doesn't run a Ray cluster, it watches for `RayCluster` / `RayService` custom resources and reconciles whatever it finds. The actual head + worker cluster is defined in `VllmStack`, as part of the `RayService` spec below.

`self.kuberay_operator` is exposed from this stack so `VllmStack` can declare an explicit dependency on it (`node.add_dependency(kuberay_operator)`). Without that, CDK has no way of knowing the `kuberay` namespace and its CRDs need to exist before the manifests in `VllmStack` are applied, and they'd occasionally get applied out of order.

### vLLM Stack

The `VllmStack` ties everything together. It installs the NVIDIA device plugin, ships the Ray Serve entrypoint script as a ConfigMap, and defines a `RayService` that runs vLLM as an autoscaling Ray Serve deployment.

```python
nvidia_plugin = cluster.add_helm_chart(
    "NvidiaDevicePlugin",
    chart="nvidia-device-plugin",
    repository="https://nvidia.github.io/k8s-device-plugin",
    version="0.17.1",
    namespace="kube-system",
    values={
        "nodeSelector": {"workload": "gpu"},
        "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
    },
)
```

Same as the previous post, this is what makes `nvidia.com/gpu` requestable as a resource on whatever GPU nodes Karpenter launches.

The Ray Serve app script (`serve_vllm.py`) gets mounted into the containers through a ConfigMap:

```python
serve_configmap = cluster.add_manifest("VllmServeConfigMap", {
    "apiVersion": "v1",
    "kind": "ConfigMap",
    "metadata": {"name": "vllm-serve-script", "namespace": "kuberay"},
    "data": {"serve_vllm.py": _load_serve_script()},
})
serve_configmap.node.add_dependency(kuberay_operator)
```

Then the `RayService` itself, this is the bulk of the stack:

```python
ray_service = cluster.add_manifest("VllmRayService", {
    "apiVersion": "ray.io/v1",
    "kind": "RayService",
    "metadata": {"name": "vllm", "namespace": "kuberay"},
    "spec": {
        "serveConfigV2": (
            "applications:\n"
            "  - name: vllm\n"
            "    import_path: serve_vllm:build_app\n"
            "    route_prefix: /\n"
            "    runtime_env:\n"
            "      working_dir: /serve-scripts\n"
            "      pip:\n"
            "        - vllm==0.8.5\n"
            f"    args:\n"
            f"      model: \"{model_id}\"\n"
            "      dtype: half\n"
            "      quantization: awq\n"
            "      max_model_len: 4096\n"
            "      download_dir: /model-cache\n"
            "    deployments:\n"
            "      - name: VLLMDeployment\n"
            "        num_replicas: 1\n"
            "        max_ongoing_requests: 5\n"
            "        autoscaling_config:\n"
            "          min_replicas: 0\n"
            "          max_replicas: 2\n"
            "          target_ongoing_requests: 2\n"
            "        ray_actor_options:\n"
            "          num_gpus: 1\n"
        ),
        "rayClusterConfig": {
            "rayVersion": "2.44.1",
            "enableInTreeAutoscaling": True,
            "headGroupSpec": {
                "rayStartParams": {"dashboard-host": "0.0.0.0", "num-gpus": "0"},
                "template": {
                    "spec": {
                        "nodeSelector": {"role": "system"},
                        "containers": [{
                            "name": "ray-head",
                            "image": "rayproject/ray:2.44.1-gpu",
                            "resources": {"requests": {"cpu": "1", "memory": "2Gi"}},
                        }],
                    },
                },
            },
            "workerGroupSpecs": [{
                "groupName": "gpu-workers",
                "replicas": 0,
                "minReplicas": 0,
                "maxReplicas": 2,
                "rayStartParams": {"num-gpus": "1"},
                "template": {
                    "spec": {
                        "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
                        "nodeSelector": {"workload": "gpu"},
                        "containers": [{
                            "name": "ray-worker",
                            "image": "rayproject/ray:2.44.1-gpu",
                            "resources": {
                                "requests": {"cpu": "2", "memory": "12Gi"},
                                "limits": {"cpu": "4", "memory": "14Gi", "nvidia.com/gpu": "1"},
                            },
                        }],
                    },
                },
            }],
        },
    },
})
```

A few things worth noting:

- `min_replicas: 0` / `max_replicas: 2` on the deployment's `autoscaling_config` is what drives everything else, Ray Serve scales vLLM replicas based on `target_ongoing_requests`, and each new replica needs a GPU worker pod, which needs a GPU node, which is where Karpenter comes in
- `enableInTreeAutoscaling: true` turns on KubeRay's autoscaler sidecar in the head pod, that's the thing actually creating/removing worker pods, separate from Ray Serve's own replica autoscaling above it
- worker `replicas` start at 0, so no GPU node exists at deploy time, Karpenter only provisions one once a worker pod goes `Pending`
- the head pod runs on the system node group (`nodeSelector: role: system`) with `num-gpus: 0`, it only routes and manages the cluster, it doesn't need a GPU itself
- image is `rayproject/ray:2.44.1-gpu`, not `ray-ml:2.44.1-gpu`, that `ray-ml` tag turned out to be deprecated on Docker Hub for this version

There's a catch with scaling GPU workers to zero: every time Karpenter provisions a fresh node, `download_dir: /model-cache` starts out empty, so vLLM re-downloads the ~5.7GB model from HuggingFace on every scale-up. Fine occasionally, annoying if traffic is spiky and nodes keep cycling. So the worker pod also gets an `initContainer` that pulls a cached copy from S3 first:

```python
"initContainers": [
    {
        "name": "model-cache-sync",
        "image": "amazon/aws-cli:2.17.62",
        "command": [
            "sh", "-c",
            "aws s3 sync s3://$MODEL_BUCKET/model-cache /model-cache || true",
        ],
        "env": [
            {"name": "AWS_DEFAULT_REGION", "value": self.region},
            {"name": "MODEL_BUCKET",       "value": model_bucket_name},
        ],
        "volumeMounts": [{"name": "model-cache", "mountPath": "/model-cache"}],
    }
],
```

The `|| true` matters, it makes this best-effort: an empty or missing bucket just falls through to the normal HuggingFace download, nothing breaks on first run. The GPU node's IAM role (`karpenter_node_role`) already has `grant_read` on the model bucket from `EksStack`, so no extra permissions are needed here.

This only helps once the bucket actually has something in it, though, S3 doesn't populate itself. After the first worker pod finishes downloading the model, seed the cache once:

```bash
kubectl cp kuberay/<worker-pod>:/model-cache ./model-cache
aws s3 sync ./model-cache s3://<your-bucket>/model-cache
```

After that, every future scale-up pulls from S3 (fast, same-region, effectively free) instead of hitting HuggingFace over the internet.

Last, the same NLB pattern as before, just pointed at the Ray head's Serve port instead of a plain vLLM pod:

```python
nlb_svc = cluster.add_manifest("VllmNlbService", {
    "apiVersion": "v1",
    "kind": "Service",
    "metadata": {
        "name": "vllm-nlb",
        "namespace": "kuberay",
        "annotations": {
            "service.beta.kubernetes.io/aws-load-balancer-type": "nlb",
            "service.beta.kubernetes.io/aws-load-balancer-scheme": "internet-facing",
        },
    },
    "spec": {
        "type": "LoadBalancer",
        "selector": {"role": "ray-head"},
        "ports": [{"port": 80, "targetPort": 8000}],
    },
})
```

## Deploy

```bash
cdk bootstrap   # first time only
cdk deploy --all
```

This deploys the four stacks in order: `eks-stack` → `karpenter-stack` → `ray-stack` → `vllm-stack`. The cluster itself takes the longest, 15-20 minutes, so grab a coffee.

Once `eks-stack` is up, the system node group is there:

```bash
$ kubectl get nodes
NAME                          STATUS   ROLES    AGE   VERSION
ip-10-0-13-44.ec2.internal    Ready    <none>   4m    v1.34.7-eks-40737a8
ip-10-0-31-9.ec2.internal     Ready    <none>   4m    v1.34.7-eks-40737a8
ip-10-0-52-201.ec2.internal   Ready    <none>   4m    v1.34.7-eks-40737a8
```

Three CPU nodes, no GPU nodes. Nothing's asking for a GPU yet, so Karpenter hasn't provisioned anything.

Once `karpenter-stack` is up, the controller should be running:

```bash
$ kubectl get pods -n karpenter
NAME                                                         READY   STATUS    RESTARTS   AGE
eksstackeksllmscalechartkarpenter29d18760-7b8d7cbf68-bhtfd   1/1     Running   0          11m
eksstackeksllmscalechartkarpenter29d18760-7b8d7cbf68-nmh8m   1/1     Running   0          11m
```

And once `ray-stack` + `vllm-stack` finish, the Ray head comes up on the system nodes:

```bash
$ kubectl get pods -n kuberay
NAME                        READY   STATUS    RESTARTS   AGE
kuberay-operator-...         1/1    Running   0          5m
vllm-raycluster-...-head     2/2    Running   0          3m
```

No worker pod yet, `RayService` was deployed with `replicas: 0`. This is also where things stopped being straightforward. On paper the head comes up, Serve builds the `vllm` app, and you're done — in practice it took five redeploys to get there, none of them infrastructure problems, all of them Ray Serve / Helm gotchas that only show up once something actually tries to run:

```bash
$ kubectl get rayservice vllm -n kuberay
NAME   SERVICE STATUS   NUM SERVE ENDPOINTS
vllm
```

Empty `SERVICE STATUS` isn't a good sign. The real status lives behind the head's dashboard API, not in this column:

```bash
kubectl run -n kuberay checkstatus --image=curlimages/curl:8.10.1 --restart=Never -i --rm --timeout=20s -- \
  curl -s http://<raycluster>-head-svc.kuberay.svc.cluster.local:8265/api/serve/applications/
# look at .applications.vllm.status and .message
```

What that turned up, in the order they surfaced:

- **`working_dir: /serve-scripts` rejected.** Ray Serve's `runtime_env.working_dir` only accepts remote URIs (`file://`, `s3://`, …), not a bare local path — even though that's exactly where the ConfigMap mounts the script. Fix: drop `working_dir` entirely, set `PYTHONPATH: /serve-scripts` under `runtime_env.env_vars` instead, so `import_path: serve_vllm:build_app` still resolves.
- **`num_replicas` + `autoscaling_config` together.** Ray Serve won't accept a fixed replica count alongside an autoscaling config — drop `num_replicas`, autoscaling owns replica count now.
- **`vllm==0.8.5` forbids `ray==2.44.*`.** The runtime_env's `pip: [vllm==0.8.5]` pulls in a newer Ray by default, and Ray refuses to run a pip-isolated env whose Ray version doesn't match the cluster's. vLLM 0.8.5 also explicitly excludes `2.44.*`, which is what the cluster's `rayproject/ray:2.44.1-gpu` image runs. Pinning `ray[serve]==2.44.1` in the pip list just trades one error for another — had to move the whole cluster to `rayproject/ray:2.43.0-gpu` (image *and* `rayVersion` field) and pin `ray[serve]==2.43.0` instead.
- **Worker "OOM" that wasn't OOM.** Once the pip env resolved, the build task started crashing with `WorkerCrashedError` and a log line blaming "K8s pod memory limits." The node group (`t3.medium`, 4GiB) got bumped to `t3.large` and the head's memory limit from 4Gi to 6Gi — crash didn't go away. The cgroup's own `memory.events` showed `oom_kill: 0` the whole time; it was never actually OOM.
- **The real cause: NumPy 2.x vs. a NumPy-1.x-built pyarrow.** `vllm`'s pip install upgrades NumPy to 2.x, which breaks the base image's `pyarrow` (compiled against NumPy 1.x, used internally by Ray's own serialization). Every worker crashed on `import pyarrow` with `AttributeError: _ARRAY_API not found`. Fix: add `numpy<2` to the pip list.
- **`build_app()`'s signature was wrong.** Past all the environment issues, Ray Serve finally tried to actually call the app builder — `TypeError: Application builder functions should take exactly one parameter, a dictionary`. `serve_vllm.py`'s `build_app` took `model`, `dtype`, etc. as separate keyword arguments instead of one `args: dict`. Straightforward fix, but easy to miss since nothing catches it until the app actually deploys.
- **KubeRay caches `serveConfigV2` by its own text, not by what's mounted.** Editing `serve_vllm.py` alone (the ConfigMap) didn't get picked up — operator logs showed `"shouldUpdate": false, reason: "Current V2 Serve config matches cached Serve config."` on every reconcile. It only re-submits when the YAML string itself changes. Fix: fold a hash of the script into an (unused) arg in `args:`, so any script edit changes the YAML and busts the cache automatically.
- **The NVIDIA device plugin never scheduled at all.** With the app finally `RUNNING`, the GPU worker pod sat `Pending` — `Insufficient nvidia.com/gpu`. The `nvidia-device-plugin` Helm chart ships a default `nodeAffinity` requiring an NFD/GPU-Operator label (`nvidia.com/gpu.present`, among others) that nothing in this cluster sets, so it never matched our Karpenter-provisioned node. Passing `affinity: {}` to override it turned out to be a no-op (Helm doesn't let an empty map clear a chart's default nested value) — the actual fix was adding `nvidia.com/gpu.present` and `nvidia.com/mps.capable` labels directly on the Karpenter GPU `NodePool`'s node template.
- **...which then collided with itself.** The chart actually renders *two* DaemonSets (the plugin and an MPS control daemon), and CDK's auto-generated Helm release name was long enough that both DaemonSets' names got silently truncated to the same 63-character Kubernetes limit — one clobbered the other, and the survivor happened to be the MPS daemon, not the plugin. Passing an explicit short `release=` name to `add_helm_chart(...)` fixed it; the leftover daemonset from the old release name had to be deleted by hand since changing the release name doesn't clean up the old one.

None of this was visible from `kubectl get pods` — the head and worker pods looked healthy the entire time. The only way to see any of it was polling `/api/serve/applications/` directly.

Once all of the above landed, a fresh `cdk deploy eks-stack` finally gets a clean result:

```bash
$ curl -s http://<raycluster>-head-svc.kuberay.svc.cluster.local:8265/api/serve/applications/ | jq '.applications.vllm.status'
"RUNNING"
```

Now send it a request (see [Inference](#inference) below) and watch the autoscaler react. The worker pod shows up first, `Pending`, since there's no GPU node to put it on:

```bash
$ kubectl get pods -n kuberay -w
vllm-raycluster-...-worker-gpu-xxxx   0/1   Pending   0   3s
```

Karpenter picks that up within a few seconds and launches a `g4dn.xlarge`:

```bash
$ kubectl get nodes -l workload=gpu -w
NAME                          STATUS     ROLES    AGE   VERSION
ip-10-0-44-171.ec2.internal   NotReady   <none>   8s    v1.34.7-eks-40737a8
ip-10-0-44-171.ec2.internal   Ready      <none>   38s   v1.34.7-eks-40737a8
```

Once the node is `Ready`, the worker pod schedules and vLLM starts loading the model on it, same download-and-load as the previous post, just happening on a node that didn't exist a minute ago:

```bash
$ kubectl logs -f -n kuberay -l ray.io/group=gpu-workers -c ray-worker
...
INFO 07-08 ... non-default args: {'model': 'hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4', 'dtype': 'half', 'quantization': 'awq', 'max_model_len': 4096}
...
```

Stop sending traffic, and once Ray Serve scales the deployment back down to 0 replicas, the worker pod goes away, and 30 seconds later (`consolidateAfter: "30s"`) so does the GPU node:

```bash
$ kubectl get nodes -l workload=gpu
No resources found
```

That's the whole loop: no GPU node running until there's real demand for one, and nothing left running once demand disappears.

:::danger[Clean Up Resources]
Run `cdk destroy --all` when you're done. Karpenter-provisioned nodes aren't tracked by CloudFormation the same way a static node group is, so double check with `kubectl get nodes` and the EC2 console that nothing GPU-shaped got left behind.
:::

References:

- [AWS Labs: KubeRay Operator Add-on](https://github.com/awslabs/cdk-eks-blueprints/blob/main/docs/addons/kuberay-operator.md)
- [Flaticon: User Icon](https://www.flaticon.com/free-icon/user_6175043)
- [TypingMind: LLaMA Model Icon](https://custom.typingmind.com/tools/model-icons/llama)
