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

### The Stack

- **[vLLM](https://github.com/vllm-project/vllm)** — inference engine for the LLM. Fast, supports streaming, and exposes an OpenAI-compatible API out of the box.
- **[Amazon EKS](https://aws.amazon.com/eks)** — managed Kubernetes to run the vLLM workload on a GPU node.
- **[AWS CDK](https://github.com/aws/aws-cdk)** — infrastructure as code in Python. One `cdk deploy` and everything is provisioned.
- **[Streamlit](https://github.com/streamlit/streamlit)** — simple chatbot UI that talks to the vLLM endpoint.

#### Why vLLM?

There are a few ways to serve an LLM — you could use TGI, Triton, or just raw HuggingFace `transformers`. I went with vLLM for a few reasons:

- **PagedAttention** — manages GPU memory more efficiently, which matters a lot on a single `g4dn.xlarge`
- **OpenAI-compatible API** — the chatbot can use the `openai` Python SDK without any changes
- **Streaming support** — responses stream token by token, which makes the chatbot feel more responsive

#### Why EKS?

I could've just spun up an EC2 instance and SSH'd in. But that's not really MLOps — that's just running a script on a server.

EKS gives us a proper environment to run GPU workloads: node groups, taints and tolerations to make sure only the vLLM pod lands on the GPU node, and a LoadBalancer service to expose the endpoint.

## Project Structure

```
serves_llm/
├── serves_llm/
│   ├── eks/
│   │   └── eks_stack.py      # VPC, EKS cluster, GPU node group, S3 bucket
│   └── vllm/
│       └── vllm_stack.py     # vLLM deployment + LoadBalancer service
├── src/
│   └── app.py                # Streamlit chatbot
└── app.py                    # CDK entrypoint
```

## Prerequisites

- AWS CLI
- AWS CDK CLI: `npm install -g aws-cdk`
- Python 3.9+
- GPU quota for `g4dn.xlarge` — request via [Service Quotas](https://console.aws.amazon.com/servicequotas) ("Running On-Demand G and VT instances", minimum 4 vCPU)
- HuggingFace account with access to [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

## Setup

```bash
git clone https://github.com/your-username/eks-cdk-mlops
cd serves_llm

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in HF_TOKEN, CDK_DEFAULT_ACCOUNT, CDK_DEFAULT_REGION, AWS_ADMIN_USER
```

## Deploy

```bash
cdk bootstrap   # first time only
cdk deploy --all
```

This provisions the EKS cluster first (`EksStack`), then deploys vLLM on top of it (`VllmStack`).

After deploy, grab the NLB endpoint:

```bash
kubectl get svc vllm -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
```

Add it to `.env` as `VLLM_ENDPOINT_URL`, then run the chatbot:

```bash
streamlit run src/app.py
```

## What's Next

This is Phase 1. Phase 2 is packaging the chatbot and deploying it to EKS as well — so the whole thing runs in the cloud, not just the model.

---

![That's not enough](https://res.cloudinary.com/diunivf9n/image/upload/v1777184871/not-enough-batman_jpvyc6.gif)

Aight. Thanks for reading this post, hope you found something useful 🚀
