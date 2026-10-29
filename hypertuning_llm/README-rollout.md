# LLM on EKS: Serving the Pipeline's Model and Rolling Out New Versions

Hi there! 👋🏽 In the previous post, a Kubeflow pipeline fine-tuned `Qwen2.5-1.5B-Instruct` with LoRA, quantized it with AWQ, and pushed it to S3. But our serving stack didn't use that model at all. vLLM was serving a Llama model from Hugging Face.

So in this post, we'll close that gap. vLLM loads the pipeline's model from S3, every model version gets its own S3 prefix, and switching versions is a blue/green rollout with no downtime.

## TL;DR

- Each GPU worker pulls the model from S3 in an init container before vLLM starts. 1.6 GB synced in about 6 seconds through the S3 Gateway Endpoint.
- vLLM serves it under a stable name, `qwen-lora`, so clients don't change when the model behind it does.
- Model versions are write-once S3 prefixes (`pipeline-output/model-v1`, `pipeline-output/<run-id>`). `MODEL_PREFIX` picks which one gets served.
- Changing `MODEL_PREFIX` makes KubeRay build a new RayCluster with the new model, and traffic moves once it serves. Rolling back is the same thing in reverse.
- vLLM is reachable from inside the cluster, no public load balancer in front of the GPU.
- A new GPU replica takes about 8 minutes from request to serving, and Karpenter accounts for less than one of them.

## Serving the Model from S3

<!-- TODO: init container `aws s3 sync "$MODEL_URI" /model-cache/current`, emptyDir, why per-pod sync instead of shared storage, served_model_name "qwen-lora", serve_vllm.py changes -->

### The IMDS Hop Limit

<!-- TODO: the init container couldn't get node IAM credentials; IMDSv2 hop limit 1 blocks pods, httpPutResponseHopLimit: 2 on the EC2NodeClass; verified live 2026-10-27 (9 files / 1.6 GB in 6s, exit=0) -->

## Versioning the Model

Every model gets its own S3 prefix, and we write each prefix once. The problem with the previous post was `pipeline-output/latest`: every `pipeline_llm` run overwrites it, so the name stays the same while the model behind it changes. I copied that model to `pipeline-output/model-v1`. Runs of this project's pipeline write to `pipeline-output/<run-id>`, and a run ID is unique.

`MODEL_PREFIX` in `.env` decides which version is served:

```
MODEL_PREFIX=pipeline-output/model-v1
```

<!-- TODO: why `latest` was a problem (Git/config says one thing, S3 contents change underneath), .env.example, `just deploy` refusing a dirty tree -->

## Rolling Out Model B

Model A is what's serving now: `pipeline-output/model-v1`. Model B is any newer version.

<!-- TODO: MODEL_URI in rayClusterConfig vs serveConfigV2 (zero-downtime NewCluster upgrade vs in-place restart), KubeRay's default vllm-serve-svc (ClusterIP) follows the active cluster, min_replicas=1, 2x GPU during the switch, deploy_model step + RBAC for pipeline-runner, rollback = point MODEL_PREFIX back. Results from the live test: errors during the switch (Locust), time from change to traffic moved -->

## Keeping vLLM Private

<!-- TODO: dropped the internet-facing NLB (unauthenticated GPU endpoint), vllm-serve-svc ClusterIP, `just vllm-forward` for local access, in-cluster URL for apps -->

## Load Testing

<!-- TODO: tests/load_test.py (`just load`, stdlib, runs in a pod) for numbers, Locust (`just locust`, manifests/locust-pod.yaml) for the live chart; TOKENS row; target_ongoing_requests 2 -> second replica -->

### How Long a New Replica Takes

Measured on 2026-10-27, one run, polling every ~20s:

| Stage | Time (UTC) | Elapsed |
|---|---|---|
| Worker pod created | 01:56:16 | |
| Karpenter's GPU node (`g4dn.xlarge`) ready | 01:56:55 | 39s |
| Worker container running (image pull, S3 sync, init) | 01:59:55 | ~3m |
| vLLM loaded, Ray Serve app `RUNNING` | 02:04:04 | ~4m |
| **Total** | | **~7m48s** |

Karpenter isn't the bottleneck here. Most of the time goes to pulling the Ray GPU image and installing vLLM at startup.

<!-- TODO: compare with scale_llm's ~9m27s (different model, source and scenario, so not apples to apples); baking vLLM into the image as the next step -->

## Chatting with the Model

<!-- TODO: src/chat_gradio and src/chat_streamlit, knowledge base in the system prompt, per-request stats (ttft, prompt/completion tokens, tok/s) via stream_options include_usage; prompt tokens ~1.5k because of the knowledge base -->

## Deploy

<!-- TODO: .env from .env.example, just deploy, just vllm-forward, cleanup order (delete ec2nodeclass gpu before just destroy) -->

## Conclusion

<!-- TODO -->
