# LLM on EKS: Serving with vLLM

Last year, I mentioned that I'm interested in learning MLOps and LLMs. At first it was just curiosity, but over time I wanted to actually try building something—not just reading about it.

This post is a small step in that direction: serving an LLM using [vLLM](https://github.com/vllm-project/vllm), deployed on [Amazon EKS](https://aws.amazon.com/eks), provisioned the infra using [AWS CDK](https://github.com/aws/aws-cdk), and wrapped into a simple chatbot using [Streamlit](https://github.com/streamlit/streamlit).

## TL;DR

- I'm exploring MLOps + LLMs by building a real project
- Using vLLM for inference
- Running on EKS with CDK
- Simple chatbot with Streamlit

## What We Tryna Build

The idea is simple: build a small chatbot powered by an LLM and run the model on Kubernetes.

I'm not focusing on training models here. I just want to understand how to serve an LLM properly.

The flow looks like this:

- User interacts with a chatbot (running locally)
- The chatbot sends a request to a vLLM API
- The model processes the request and returns a response
- The vLLM service runs on Amazon EKS

![Project Architecture](https://res.cloudinary.com/diunivf9n/image/upload/v1777186530/vllm-eks_ybekjc.webp)

## The Stack

- **vLLM** — inference engine for the LLM. Fast, supports streaming, and exposes an OpenAI-compatible API out of the box.
- **Amazon EKS** — The Kubernetes service on AWS to run the vLLM workload.
- **AWS CDK** — infrastructure as code to manage AWS infra, at this time I'll using Python. One `cdk deploy` and everything is provisioned.
- **Streamlit** — simple chatbot UI that talks to the vLLM endpoint.

## Why vLLM?

There are a few ways to serve an LLM — you could use TGI, Triton, or just raw HuggingFace `transformers`. I went with vLLM for a few reasons:

- **PagedAttention** — manages GPU memory more efficiently, which matters a lot on a single `g4dn.xlarge`
- **OpenAI-compatible API** — the chatbot can use the `openai` Python SDK without any changes
- **Streaming support** — responses stream token by token, which makes the chatbot feel more responsive

## Why EKS?

I could've just spun up an EC2 instance and SSH'd in. But that's not really MLOps — that's just running a script on a server.

EKS gives us a proper environment to run GPU workloads: node groups, taints and tolerations to make sure only the vLLM pod lands on the GPU node, and a LoadBalancer service to expose the endpoint.

## The Code

### EKS Stack

The `EksStack` provisions everything at the infrastructure level: VPC, EKS cluster, node groups, and an S3 bucket for model storage.

```python
vpc = ec2.Vpc(self, "EksVpc", max_azs=2)

cluster = eks.Cluster(
    self, "EksCluster",
    version=eks.KubernetesVersion.V1_34,
    vpc=vpc,
    default_capacity=0,
    kubectl_layer=kubectl_layer,
)
```

`default_capacity=0` means no default node group — we define our own below.

We have two node groups:

```python
# 1. CPU, runs system pods (CoreDNS, kube-proxy, etc.)
cluster.add_nodegroup_capacity(
    "ManagedNodeGroup",
    desired_size=1,
    instance_types=[ec2.InstanceType("t3.medium")],
    ami_type=eks.NodegroupAmiType.AL2023_X86_64_STANDARD,
)

# 2. GPU, for running vLLM
cluster.add_nodegroup_capacity(
    "GpuNodeGroup",
    desired_size=1,
    instance_types=[ec2.InstanceType("g4dn.xlarge")],
    ami_type=eks.NodegroupAmiType.AL2023_X86_64_NEURON,
    labels={"workload": "gpu"},
    taints=[
        eks.TaintSpec(
            key="nvidia.com/gpu",
            value="true",
            effect=eks.TaintEffect.NO_SCHEDULE,
        )
    ],
)
```

The taint `nvidia.com/gpu=true:NoSchedule` on the GPU node group means no pod will be scheduled there unless it explicitly tolerates it. This keeps system pods off the GPU node.

The S3 bucket is for model weights, and the GPU node role gets read access to it:

```python
self.model_bucket = s3.Bucket(self, "ModelBucket", ...)
self.model_bucket.grant_read(gpu_node_role)
```

:::note
We will create those instances:

| Node      | vCPU| Memory  |
|-----------|-----|---------|
| t3.medium | 2   | 4Gi     |
| g4dn.xlarge | 4 | 8Gi     | 
:::

### vLLM Stack

The `VllmStack` takes the cluster from `EksStack` and deploys vLLM on top of it.

First, we install the NVIDIA device plugin via Helm. This is what makes EKS aware of the GPU on the node — without it, you can't request `nvidia.com/gpu` as a resource in your pod spec.

```python
cluster.add_helm_chart(
    "NvidiaDevicePlugin",
    chart="nvidia-device-plugin",
    repository="https://nvidia.github.io/k8s-device-plugin",
    namespace="kube-system",
    values={"tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]},
)
```

Note the toleration on the plugin itself — it needs to run on the GPU node to expose the GPU, so it has to tolerate the taint we set earlier.

Then the vLLM Deployment:

```python
cluster.add_manifest("VllmDeployment", {
    ...
    "spec": {
        "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
        "nodeSelector": {"workload": "gpu"},
        "containers": [{
            "image": "vllm/vllm-openai:latest",
            "args": [
                "--model", "meta-llama/Llama-3.1-8B-Instruct",
                "--dtype", "float16",
                "--max-model-len", "4096",
            ],
            "resources": {
                "limits": {"nvidia.com/gpu": "1"},
            },
        }],
    },
})
```

A few things worth noting:

- `nodeSelector: workload=gpu` pins the pod to the GPU node group
- `nvidia.com/gpu: 1` requests exactly one GPU
- `dtype: float16` keeps memory usage in check on the 16GB VRAM of `g4dn.xlarge`
- `max-model-len: 4096` caps the context window to avoid OOM

Finally, a LoadBalancer service to expose the endpoint publicly:

```python
cluster.add_manifest("VllmService", {
    "kind": "Service",
    "metadata": {
        "annotations": {"service.beta.kubernetes.io/aws-load-balancer-type": "nlb"},
    },
    "spec": {
        "type": "LoadBalancer",
        "ports": [{"port": 80, "targetPort": 8000}],
    },
})
```

## Deploy

```bash
cdk bootstrap   # first time only
cdk deploy --all
```

Wait for the vLLM pod to be ready (~5-10 minutes, model is downloaded from HuggingFace on first start):

```bash
kubectl get pods -w
kubectl logs -f deployment/vllm
```

## Inference

Once the pod is running, grab the NLB endpoint:

```bash
kubectl get svc vllm -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
```

Check that the model is loaded:

```bash
curl http://<nlb-endpoint>/v1/models
```

You should see something like:

```json
{
  "object": "list",
  "data": [{
    "id": "meta-llama/Llama-3.1-8B-Instruct",
    "object": "model",
    "owned_by": "vllm"
  }]
}
```

Send it a prompt:

```bash
curl http://<nlb-endpoint>/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "What is MLOps?",
    "max_tokens": 200,
    "temperature": 0
  }'
```

If you get a response back — the model is live. 🎉

A working API endpoint is great, but typing `curl` commands is not exactly a great user experience. Let's build a proper chatbot UI on top of this.

![That's not enough](https://res.cloudinary.com/diunivf9n/image/upload/v1777184871/not-enough-batman_jpvyc6.gif)

## Chatbot with Streamlit

So let's built a simple chatbot using [Streamlit](https://github.com/streamlit/streamlit) that talks directly to the vLLM.

The nice part? Since vLLM exposes an OpenAI-compatible API, we can just use the openai Python SDK without any efforts.

### Setup

Install the dependencies:

```bash
pip install streamlit openapi
```

Let's create a simple UI:

```bash
mkdir src
touch src/app.y
```

```py title='app.py'
import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

VLLM_URL = os.environ["VLLM_ENDPOINT_URL"]
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

client = OpenAI(base_url=f"{VLLM_URL}/v1", api_key="none")

st.set_page_config(page_title="Llama 3 Chatbot", page_icon="🦙")
st.title("🦙 Llama 3 Chatbot")
st.caption("Powered by vLLM on EKS")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("How is you day? Say something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=MODEL_ID,
            messages=st.session_state.messages,
            stream=True,
        )
        response = st.write_stream(chunk.choices[0].delta.content or "" for chunk in stream)

    st.session_state.messages.append({"role": "assistant", "content": response})
```

Run the UI:

```bash
streamlit run app.py
```

Aight. Thanks for reading this post, hope you found something useful 🚀
