# EKS CDK LLM

A repository for exploring LLM serving and MLOps on Amazon EKS, built with AWS CDK.

![The Final Architecture](https://res.cloudinary.com/diunivf9n/image/upload/v1793366609/Archi-Page-3.drawio_lgyv3h.png)

## Projects

Each folder is its own CDK app and its own blog post. Each one starts as a copy of the one before it, so every post's code stays runnable as published.

| # | Project | What it does | Blog post |
|---|---------|--------------|-----------|
| 1 | **[Serving LLMs on EKS](./serves_llm/README.md)** | EKS cluster serving `Llama-3.1-8B-Instruct` (AWQ) with vLLM | [LLM on EKS: Serving with vLLM](https://danielcristho.site/blog/llm-on-eks-vllm) |
| 2 | **[Scaling LLMs on EKS](./scale_llm/README.md)** | Autoscaling inference with Ray + KubeRay, GPU nodes on demand via Karpenter | [LLM on EKS: Scaling with Ray & Karpenter](https://danielcristho.site/blog/llm-on-eks-scale-llm) |
| 3 | **[Training Pipelines on EKS](./pipeline_llm/README.md)** | LoRA fine-tuning and AWQ quantization of `Qwen2.5-1.5B-Instruct` with Kubeflow Pipelines, backed by RDS and S3 | [LLM on EKS: Training Pipelines with Kubeflow](https://danielcristho.site/blog/llm-on-eks-pipeline-kubeflow/) |
| 4 | **[Rolling Out Models](./hypertuning_llm/README-rollout.md)** | Serving the pipeline's model from S3, switching models with a Git commit through Argo CD | Draft, [in the repo](./hypertuning_llm/README-rollout.md) |
| 5 | **[Hyperparameter Tuning](./hypertuning_llm/README.md)** | Ray Tune searches LoRA settings, and a new model ships when it beats the old one | Draft, [in the repo](./hypertuning_llm/README.md) |
| 6 | **[Making LLMs Reliable on EKS](./survive_llm/README.md)** | Surviving an AZ outage with zonal shift (Amazon ARC): a warm vLLM replica per AZ, one NAT gateway per AZ, and the Gradio chat UI running in the cluster | Draft, [in the repo](./survive_llm/README.md) |

**Stack:** AWS CDK, Amazon EKS, vLLM, Ray, Karpenter, Kubeflow Pipelines, Argo CD, Amazon Application Recovery Controller (ARC), Gradio

## The full journey in one project

[`survive_llm`](./survive_llm/) carries every step of the series: Serve (vLLM) → Scale (Ray + Karpenter) → Fine-tune (Kubeflow: prep, LoRA, AWQ) → Serve from S3 (Argo CD rollout) → Survive (ARC zonal shift). Its stacks deploy in this order:

| Stack | What it adds |
|---|---|
| `eks-stack` | VPC (2 AZs, one NAT gateway each), EKS with zonal shift on, system node group, IAM for Karpenter, RDS for Kubeflow |
| `karpenter-stack` | Karpenter and the GPU NodePool (`g4dn.xlarge`, on demand) |
| `ray-stack` | KubeRay operator |
| `vllm-stack` | NVIDIA device plugin and the Ray Serve script for vLLM |
| `kubeflow-stack` | Kubeflow Pipelines on RDS + S3 |
| `argocd-stack` | Argo CD, syncing the vLLM RayService chart from `survive_llm/manifests/vllm` |
| `chat-stack` | The Gradio chat UI, calling vLLM through its in-cluster Service |

## Quick start

Requirements: an AWS account with G/VT instance vCPU quota, the AWS CLI, AWS CDK CLI, `kubectl`, [`just`](https://github.com/casey/just) and Python 3.12. The `justfile` recipes use POSIX shell, so on Windows run them from Git Bash or WSL.

```bash
cd survive_llm
cp .env.example .env   # fill in account, region, admin user, bucket, cluster name
just deploy            # refuses to run with uncommitted changes in the folder
just kubeconfig
just chat-ui           # chat UI at http://localhost:7860
just destroy           # GPU nodes and NAT gateways cost money: don't leave them running
```

Zonal shift recipes: `just zones`, `just zonal-shift-start <az-id> 30m`, `just zonal-shift-status`, `just zonal-shift-cancel <id>`.

## Before you deploy

Each project creates real AWS resources, including GPU instances and a NAT gateway. Every folder has a `justfile`: `just deploy` to bring it up, `just destroy` when you're done.

The projects share stack names (`eks-stack`, `vllm-stack`, ...), so deploy one project at a time per account and region, and destroy it before deploying the next.
