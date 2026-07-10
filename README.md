# EKS CDK LLM

A repository for exploring LLM Serving and MLOps practices on AWS Kubernetes (EKS) using Infrastructure as Code (AWS CDK).

## Projects

| Project | Description | Blog Post |
|---------|-------------|------|
| **[Serving LLMs on EKS](./serves_llm/README.md)** | Creating EKS cluster and serving `Llama-3.1-8B-Instruct` (AWQ quantized) LLM | [LLM on EKS: Serving with vLLM](https://danielcristho.site/blog/llm-on-eks-vllm)|
| **[Scaling LLMs on EKS](./scale_llm/README.md)** | Autoscaling `Llama-3.1-8B-Instruct` inference with Ray + KubeRay, GPU nodes provisioned on demand via Karpenter | [LLM on EKS: Scaling with Ray & Karpenter](https://danielcristho.site/blog/llm-on-eks-scale-llm)|

