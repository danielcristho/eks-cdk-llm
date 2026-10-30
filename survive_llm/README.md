# LLM on EKS: Surviving an AZ Failure with ARC Zonal Shift

Hi there! 👋🏽 In the previous posts, the stack serves, scales and fine-tunes `Qwen2.5-1.5B-Instruct` on EKS. Before production, a teammate asked: "what if one Availability Zone breaks, when a new GPU replica takes ~8 minutes to serve traffic?"

So in this post, we'll test Amazon Application Recovery Controller (ARC) zonal shift on this stack, and look at where static stability and GPU cost savings pull in opposite directions.

## TL;DR

<!-- TODO: from a real zonal shift run -->

## What Changed from the Previous Post

- **EKS zonal shift on.** `eks.Cluster` has no prop for it, so an `AwsCustomResource` calls `UpdateClusterConfig` with `zonalShiftConfig.enabled=true`.
- **One NAT Gateway per AZ.** With one NAT, losing its AZ cuts egress (S3 sync, pip installs) for the AZ we shift traffic to.
- **One warm vLLM replica per AZ.** `min_replicas: 2` plus a zone `topologySpreadConstraint` on the GPU workers, so the healthy AZ already serves when the shift starts.
- **`just` recipes** to start, inspect and cancel a zonal shift.
- **The chat UI runs in the cluster.** `chat-stack` deploys the Gradio app (`src/chat_gradio`) from a ConfigMap, and it calls vLLM through `vllm-serve-svc`. Through `kubectl port-forward` to vLLM, a laptop would pin one pod and skip the Service that zonal shift steers. Open it with `just chat-ui`.

<!-- TODO: architecture diagram, 2 AZs -->

## Static Stability vs Scale to Zero

<!-- TODO: scale_llm scaled GPUs to zero; surviving an AZ means paying for a warm GPU per AZ, 24/7. Cost of 2 x g4dn.xlarge vs the ~8 min to bring a replica up -->

## What Zonal Shift Moves (and What It Doesn't)

<!-- TODO: EndpointSlices drop the AZ's endpoints, managed node groups stop launching there; Karpenter-launched GPU nodes, the Ray head (single point of failure, GCS fault tolerance would need external Redis), RDS single-AZ for KFP -->

## Running the Test

```bash
just zones                                 # AZ name -> AZ ID, nodes per zone
just locust                                # steady load on vllm-serve-svc
just zonal-shift-start <az-id> 30m         # shift away from one AZ
just zonal-shift-status                    # applied weights per AZ
just zonal-shift-cancel <zonal-shift-id>   # end it early
```

<!-- TODO: errors and latency during the shift, from Locust -->

## Results

<!-- TODO -->

## Deploy

<!-- TODO: same as the previous post; check the G/VT vCPU quota covers 4 x g4dn.xlarge (2 warm + 2 during a blue/green upgrade) -->

## Conclusion

<!-- TODO -->
