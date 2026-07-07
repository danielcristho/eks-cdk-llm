# LLMs on EKS: Scaling with Ray & Karpenter

Hi there, it's been 2 months.

Back in May, we deployed an LLM on Amazon EKS using vLLM. It gave me a much better understanding of what it takes to serve an LLM in production.

But after getting it up and running, one question kept coming to mind: What happens when a single GPU isn't enough?

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

References:

- [AWS Labs: KubeRay Operator Add-on](https://github.com/awslabs/cdk-eks-blueprints/blob/main/docs/addons/kuberay-operator.md)