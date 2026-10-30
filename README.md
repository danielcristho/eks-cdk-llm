# EKS CDK LLM

A repository for exploring LLM serving and MLOps on Amazon EKS, built with AWS CDK.

![The Final Architecture](https://res.cloudinary.com/diunivf9n/image/upload/v1793366609/Archi-Page-3.drawio_lgyv3h.png)

## Projects

| # | Project | What it does | Blog post |
|---|---------|--------------|-----------|
| 1 | **[Serving LLMs on EKS](./serves_llm/README.md)** | EKS cluster serving `Llama-3.1-8B-Instruct` (AWQ) with vLLM | [LLM on EKS: Serving with vLLM](https://danielcristho.site/blog/llm-on-eks-vllm) |
| 2 | **[Scaling LLMs on EKS](./scale_llm/README.md)** | Autoscaling inference with Ray + KubeRay, GPU nodes on demand via Karpenter | [LLM on EKS: Scaling with Ray & Karpenter](https://danielcristho.site/blog/llm-on-eks-scale-llm) |
| 3 | **[Training Pipelines on EKS](./pipeline_llm/README.md)** | LoRA fine-tuning and AWQ quantization of `Qwen2.5-1.5B-Instruct` with Kubeflow Pipelines, backed by RDS and S3 | [LLM on EKS: Training Pipelines with Kubeflow](https://danielcristho.site/blog/llm-on-eks-pipeline-kubeflow/) |
| 4 | **[Rolling Out Models](./hypertuning_llm/README-rollout.md)** | Serving the pipeline's model from S3, switching models with a Git commit through Argo CD | coming soon |
| 5 | **[Hyperparameter Tuning](./hypertuning_llm/README.md)** | Ray Tune searches LoRA settings, and a new model ships when it beats the old one | coming soon |
| 6 | **Making LLMs Reliable on EKS** | Surviving an AZ outage with zonal shift (Amazon ARC) | coming soon |

**Stack:** AWS CDK, Amazon EKS, vLLM, Ray, Karpenter, Kubeflow Pipelines, Argo CD

## Before you deploy

Each project creates real AWS resources, including GPU instances and a NAT gateway. Every folder has a `justfile`: `just deploy` to bring it up, `just destroy` when you're done.
