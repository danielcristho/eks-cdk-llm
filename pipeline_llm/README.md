# LLM on EKS: Training Pipelines with Kubeflow

Hi there! 👋🏽 Let's continue this series. Previous posts got a Llama-3.1-8B-Instruct serving on EKS, then made it autoscale with Ray and Karpenter. This time it's about closing the loop: fine-tuning the model, quantizing it, and pushing it back into the same serving stack, all orchestrated as a repeatable pipeline instead of a one-off script.

## TL;DR

- Kubeflow Pipelines standalone, running on the same EKS cluster as the earlier posts, orchestrates a 3-step fine-tuning pipeline: prep data, LoRA fine-tune, AWQ quantize, then push straight to S3.
- KFP's default in-cluster MySQL and MinIO are swapped for Amazon RDS and S3, no PersistentVolumeClaims, no EBS CSI driver needed.
- Ray stays dedicated to serving (autoscaling vLLM); training runs as plain pods through Argo Workflows, not Ray jobs (more on why below).
- GPU steps land on the same Karpenter-provisioned `g4dn.xlarge` NodePool from the previous post.
- The pipeline writes to the same S3 bucket `vllm_stack.py` already reads from; actually serving this pipeline's output still means pointing `vllm_stack.py`'s hardcoded `model_id` at it (more on that below).

## The Stack

- Amazon EKS: runs Kubeflow Pipelines, Argo Workflows, and Ray workloads.
- Amazon RDS for MySQL: stores Kubeflow Pipelines and ML Metadata.
- Amazon S3: stores pipeline artifacts and model assets.
- AWS CDK: provisions the infrastructure as code.
- Kubeflow Pipelines: orchestrates machine learning workflows.
- Argo Workflows: executes each pipeline step.
- vLLM: serves the trained model through an OpenAI-compatible API.

## How Does It All Fit Together?

![Architecture diagram](PLACEHOLDER-architecture-diagram.png)

Everything still on the same EKS cluster from the earlier posts. Karpenter still provisions GPU nodes on demand, the same `g4dn.xlarge` NodePool that scales vLLM replicas up and down. What's new here is Kubeflow Pipelines, installed standalone (not the full Kubeflow platform) and backed by RDS + S3 instead of its default in-cluster MySQL and MinIO.

The flow: a pipeline gets compiled locally and uploaded to Kubeflow Pipelines' UI or API. Triggering a run hands it to Argo Workflows, which schedules each step as a regular pod. The GPU steps land on the same Karpenter NodePool vLLM uses, tolerating the same `nvidia.com/gpu` taint. The final step pushes the fine-tuned, re-quantized model into the same S3 bucket `vllm_stack.py` already reads from, so there's no manual file shuffling between training output and the serving path, though pointing vLLM at a specific run's output still means updating `vllm_stack.py`'s `model_id`.

### Why Kubeflow over Ray Tune + Ray Train?

Ray is already part of the stack (`ray_stack.py`), so why not use Ray Tune or Ray Train for training instead of adding Kubeflow?

Ray and Kubeflow solve different problems. Ray Tune or Ray Train is about **execution**: running training or hyperparameter search across multiple GPUs or nodes. Kubeflow Pipelines is about **orchestrating** the process: managing a multi-step workflow from data prep to training to quantization to upload as a tracked DAG. Each run keeps its history, artifact lineage is clear (which dataset produced which model), there's a UI to see which step failed, and pipelines can be reused and versioned.

For the fine-tuning here (LoRA on a single `g4dn.xlarge` GPU), Ray Train/Tune's multi-node scaling isn't really needed; its value shows up once training requires multiple nodes. Worth being honest about too: the existing Ray cluster in this project is dedicated to serving (autoscaling vLLM), not training. `finetune_pipeline.py` runs as a regular pod through Argo Workflows, not as a Ray job. If the scope later expands to distributed fine-tuning, the two can work together: a Kubeflow step submitting a Ray Train job to the same cluster, rather than replacing one with the other.

### Why RDS and S3?

Kubeflow Pipelines ships with in-cluster MySQL and MinIO by default, which is great for getting started locally. Since this series focuses on Amazon EKS, those get replaced with managed AWS services instead.

Amazon RDS stores pipeline metadata and ML Metadata, while Amazon S3 stores pipeline artifacts. This keeps state outside the Kubernetes cluster, making the platform easier to operate and more resilient. It also sidesteps a real constraint of this cluster: no EBS CSI driver is installed (model weights already live in S3, Ray's model-cache uses `emptyDir`), so the default MySQL/MinIO PersistentVolumeClaims would just sit `Pending` forever.

## How Is the Kubeflow Pipelines Manifest Vendored?

Kubeflow Pipelines standalone ships as Kustomize, not Helm, so `cluster.add_helm_chart()` doesn't apply here. Instead, the upstream manifest is rendered once with `kustomize build` and vendored into `pipeline_llm/kubeflow/manifests/`, then patched at synth time (in `kubeflow_stack.py`) to swap the default in-cluster MySQL/MinIO for RDS + S3:

```bash
kustomize build "https://github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=2.2.0" \
  > pipeline_llm/kubeflow/manifests/kfp-cluster-scoped.yaml

kustomize build "https://github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-emissary?ref=2.2.0" \
  > pipeline_llm/kubeflow/manifests/kfp-pipelines-base.yaml
```

## What Does the Pipeline Actually Do?

`finetune_pipeline.py` compiles down to three steps:

1. **`prepare_dataset`**: pulls a small instruction-tuning subset (`mlabonne/guanaco-llama2-1k`) and writes it out as JSONL. CPU-only, no GPU node needed.
2. **`finetune_lora`**: LoRA fine-tunes the base model for a single epoch, then merges the adapter into full-precision weights. Runs on a GPU node.
3. **`quantize_and_upload`**: AWQ-quantizes the merged model back to INT4, then uploads it to `s3://<bucket>/pipeline-output/latest/`, inside the same bucket `vllm_stack.py` reads from. Also GPU.

This isn't chasing accuracy. It's a small dataset and one epoch, deliberately. The point is proving the pipeline mechanics actually work end-to-end: GPU scheduling through Karpenter, artifact handoff between steps, and a model landing back in the serving path without manual intervention.

### Why a Different Base Model for Training?

The earlier posts in this series serve `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`, already quantized to INT4. The fine-tuning pipeline here doesn't start from that checkpoint. LoRA fine-tuning needs to update full-precision weights; you can't meaningfully train on top of a checkpoint that's already been quantized down to 4-bit. So the pipeline starts from full-precision weights, fine-tunes with LoRA, merges the adapter, then re-quantizes the result with AWQ as the last step, landing it back in S3 in the same INT4 format `vllm_stack.py` already expects.

For the base weights, the first attempts used `NousResearch/Meta-Llama-3.1-8B-Instruct` instead of Meta's official `meta-llama/Llama-3.1-8B-Instruct` repo: same weights, same license, an ungated re-upload that needs no `HF_TOKEN`. That kept the pipeline credential-free, but ran into a real wall on `g4dn.xlarge`'s T4 (14.74GB usable VRAM): QLoRA training itself worked, but merging the LoRA adapter back into full-precision weights afterward didn't. A full fp16 reload for an 8B model doesn't physically fit, and reloading in 4-bit or 8-bit to merge instead left the state dict partially quantized, breaking the AWQ step downstream with shape and dtype errors that no amount of cleanup fixed.

Rather than add a second GPU node pool just for the merge step, `base_model` defaults to `Qwen/Qwen2.5-1.5B-Instruct` now. At that size, fp16 training and merging both fit comfortably on the same T4 with room to spare, no 4-bit/8-bit quantization needed anywhere in `finetune_lora`, which also means the whole class of merge-time dtype bugs above simply doesn't come up. The trade-off: `vllm_stack.py` still serves `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` from the earlier posts, so this pipeline's output isn't a drop-in replacement for it. Swapping to a matching base model (or back to Llama, on bigger GPU) is still just the `base_model` pipeline parameter.

## Deploy

```bash
cdk bootstrap
cdk deploy --all
```

Getting `kubeflow-stack` to a healthy state took a few real rounds of debugging, worth being honest about rather than pretending the first deploy just worked:

- An EC2 security group rule description containing `->`, rejected by AWS's character allowlist for rule descriptions.
- CloudFormation doesn't resolve `{{resolve:secretsmanager:...}}` dynamic references inside Custom Resource properties, and `eks.KubernetesManifest` is one, so the RDS password needed a small Lambda that reads Secrets Manager directly instead of the usual `secretValueFromJson()` trick.
- Three separate container images upstream had quietly moved off `gcr.io` since this manifest was pinned to 2.2.0 (`ml-pipeline-ui`'s frontend, `argoexec`, and a transient registry blip on `nvidia-device-plugin`), each needing its own fixed reference.
- `mlpipeline-minio-artifact`'s default demo credentials (`minio`/`minio123`) short-circuited KFP's object store client straight into a static-credential path, authenticating against real S3 with garbage keys instead of ever attempting IRSA. Blanking that Secret out was the actual fix, not the IRSA role debugging it took to find that.

None of these are infrastructure decisions worth re-litigating, they're the drift that shows up whenever you vendor a manifest instead of tracking a moving target. `cdk destroy --all` cleanly tears everything down afterward, with the same two Karpenter-related teardown gotchas from the last two posts (an `EC2NodeClass` finalizer and an orphaned IAM instance profile), fixed the same way as before.

Redeploying from scratch weeks later (after the base model swap below) surfaced two more:

- The S3 model bucket has `RemovalPolicy.RETAIN`, so it survived the teardown with real data in it. Redeploying `eks_stack.py` tried to `s3.Bucket(...)` a bucket name that already existed and CloudFormation rejected the changeset outright:

  ```
  Early validation failed for change set cdk-deploy-change-set:
  eks-stack/ModelBucket/Resource  (AWS::S3::Bucket ModelBucketB33D855B)
    Resource of type 'AWS::S3::Bucket' with identifier 'eks-llm-pipeline-model-bucket' already exists.
  ```

  Fixed by referencing the existing bucket instead of recreating it: `s3.Bucket.from_bucket_name(self, "ModelBucket", bucket_name)`.

- `cluster.add_helm_chart()`/`add_manifest()`, called from `karpenter_stack.py`, `ray_stack.py`, `vllm_stack.py`, and `kubeflow_stack.py`, all parent to the `Cluster` construct that `eks_stack.py` owns. CDK resolves which CloudFormation stack a resource belongs to by walking up the construct tree, not by which file the code lives in, so every one of those Helm charts and manifests physically deploys as part of `eks-stack`, regardless of which Python stack class wrote the call. Running `cdk deploy eks-stack` alone deployed the entire app, Karpenter, Ray, vLLM, and Kubeflow Pipelines included; the later `cdk deploy karpenter-stack`/`ray-stack`/`vllm-stack`/`kubeflow-stack` commands just confirmed to CDK that nothing had changed:

  ```
  eks-stack: deploying... [1/3]
  ✅  eks-stack (no changes)
  ✨  Deployment time: 0s

  karpenter-stack: deploying... [2/3]
  ✅  karpenter-stack
  ✨  Deployment time: 12.93s
  ```

## Running the Pipeline

Compiled `finetune_pipeline.py` locally (`python finetune_pipeline.py`, producing `finetune_pipeline.yaml`), uploaded it through the KFP UI, and hit **Create run**. The first step, `prepare-dataset`, needs no GPU and runs in seconds:

```sh
[KFP Executor 2026-08-28 11:41:26,465 INFO]: Loading KFP component "prepare_dataset" from /tmp/tmp.pfCjK0T5YM/ephemeral_component.py
...
Downloading data: 100%|██████████| 967k/967k [00:00<00:00, 2.06MB/s]
Generating train split: 100%|██████████| 1000/1000 [00:00<00:00, 86752.38 examples/s]
Creating json from Arrow format: 100%|██████████| 1/1 [00:00<00:00, 95.20ba/s]
[KFP Executor 2026-08-28 11:41:28,539 INFO]: Wrote executor output file to /tmp/kfp_outputs/output_metadata.json.
I0828 11:41:28.988313 24 object_store.go:175] uploadFile(localFilePath="...", blobFilePath="prepare-dataset/dataset_out")
I0828 11:41:29.152326 24 launcher_v2.go:151] publish success.
```

Dataset pulled from Hugging Face, subset generated, uploaded to S3. Then it's `finetune-lora`'s turn, which needs a GPU node Karpenter doesn't have sitting around idle. The first two attempts after the base model swap both died here, not from pipeline code, from Karpenter deleting the node it had just provisioned:

```
{"message":"disrupting node(s)","reason":"empty","decision":"delete","disrupted-node-count":1,
 "disrupted-nodes":[{"Node":{"name":"ip-10-0-204-81.ec2.internal"},"capacity-type":"on-demand","instance-type":"g4dn.xlarge"}]}
```

The GPU `NodePool` had `consolidationPolicy: WhenEmpty`, `consolidateAfter: "30s"`. A brand-new node needs time to join, have the NVIDIA device plugin register `nvidia.com/gpu`, then get a pod actually bound to it, routinely more than 30 seconds. Karpenter's disruption loop ran its "empty" check in that window, saw zero pods yet, and deleted the node out from under `finetune-lora` mid-schedule, twice in a row on two separate runs. Bumping `consolidateAfter` to `"5m"` fixed it; the third run's node survived long enough to actually get used.

The next retry failed differently, a packaging conflict, not infra this time:

```
ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4
```

`pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` ships `numpy==1.26.4` via conda; `datasets==2.21.0` pulls in an unpinned (so: latest) `pyarrow` built against a NumPy 2.0 ABI. Forcing `numpy>=2.0` directly seemed like the fix, until pip's resolver pointed out why that numpy was there in the first place:

```
ERROR: Cannot install accelerate==0.33.0, numpy>=2.0, peft==0.12.0 and transformers==4.44.2 because
these package versions have conflicting dependencies.
ERROR: ResolutionImpossible
```

`transformers==4.44.2`, `peft==0.12.0`, and `accelerate==0.33.0` all still pin `numpy<2`. The actual fix was pinning `pyarrow<17` instead, so pip resolves a pyarrow build compatible with the numpy the other packages already require, rather than fighting them over numpy itself.

With both of those fixed, the run that finally went through pulled the base model from S3 (cached there by an earlier attempt, behind the VPC gateway endpoint, no NAT charge) instead of Hugging Face:

```
Fetching 10 files: 100%|██████████| 10/10 [00:09<00:00,  1.01it/s]
Generating train split: 200 examples [00:00, 476.92 examples/s]
Map: 100%|██████████| 200/200 [00:00<00:00, 710.15 examples/s]
  0%|          | 0/25 [00:00<?, ?it/s]...100%|██████████| 25/25 [00:43<00:00,  1.73s/it]
{'loss': 3.9413, 'grad_norm': 5.671746730804443, 'learning_rate': 3.2000000000000005e-05, 'epoch': 0.4}
{'loss': 3.3486, 'grad_norm': 5.183830261230469, 'learning_rate': 1.2e-05, 'epoch': 0.8}
{'train_runtime': 43.2565, 'train_samples_per_second': 4.624, 'train_steps_per_second': 0.578, 'train_loss': 3.5962399291992186, 'epoch': 1.0}
```

Nine seconds to fetch the cached base model, 43 seconds to train, no download-shards wait, no checkpoint-shard loading wait, no `peft`/`bnb` merge warnings either, since there's no quantized base to merge back into anymore. Compare that to the original Llama attempt's multi-minute shard downloads before training even started: this is what dropping the whole 4-bit/8-bit merge path actually bought.

`quantize_and_upload` picked the merged model up and AWQ-quantized it, all 28 layers, about 10 minutes:

```
AWQ: 100%|██████████| 28/28 [10:22<00:00, 22.24s/it]
```

and pushed the result to S3:

```
$ aws s3 ls s3://eks-llm-pipeline-model-bucket/pipeline-output/latest/ --human-readable
2026-10-22 22:18:51    1.5 GiB model.safetensors
2026-10-22 22:18:51      986 Bytes config.json
2026-10-22 22:18:54      7.3 KiB tokenizer_config.json
2026-10-22 22:18:54      6.7 MiB tokenizer.json
```

```
$ kubectl get workflow qwen-lora-finetune-lhrv5 -n kubeflow
NAME                       STATUS      AGE
qwen-lora-finetune-lhrv5   Succeeded   19m
```

End to end, for real, with a model sitting in S3 to show for it.

## Conclusion

Kubeflow Pipelines standalone, backed by RDS and S3 instead of its default in-cluster MySQL and MinIO, now runs on the same EKS cluster as the earlier posts and closes the loop this series set out to close: a model trained, quantized, and pushed into the same S3 path vLLM already reads from, as a repeatable pipeline instead of a one-off script.

Getting there took a real base-model swap partway through. The original plan kept Llama-3.1-8B throughout the whole series, but a `g4dn.xlarge` T4 can QLoRA-train an 8B model fine while still not being able to merge the adapter back into full precision afterward, no reload path (fp16, 4-bit, or 8-bit) worked cleanly. Rather than add a bigger GPU node pool just for one step, the pipeline moved to Qwen2.5-1.5B-Instruct, small enough that the entire merge-dtype problem stopped applying. That's the honest trade-off of this post: the fine-tuned output isn't a drop-in replacement for what `vllm_stack.py` currently serves, wiring the two together for real is still a `model_id` change away, not automatic.

The rest of the bugs were the usual kind for this series, real and not especially glamorous: a stale `gcr.io` reference, a CloudFormation limitation around Custom Resources, demo credentials shadowing IRSA, an S3 bucket that outlived its stack, a CDK construct-parenting quirk that deployed more than one command implied it would, a packaging conflict between `pyarrow` and three pinned ML libraries, and a Karpenter consolidation window too short for its own device plugin to catch up. None of them were exotic. All of them were real, and the pipeline runs end to end now because each one got root-caused rather than worked around.

References:

- [Kubeflow](https://www.kubeflow.org)
- [Machine Learning with Kubeflow on Amazon EKS with Amazon EFS](https://aws.amazon.com/blogs/storage/machine-learning-with-kubeflow-on-amazon-eks-with-amazon-efs)
- [AWS Labs: Deploy Kubeflow on AWS](https://awslabs.github.io/kubeflow-manifests/docs/deployment)
- [Kubeflow Pipelines manifests (kustomize/env/platform-agnostic-emissary)](https://github.com/kubeflow/pipelines/tree/2.2.0/manifests/kustomize/env/platform-agnostic-emissary)
- [Kubeflow Pipelines manifests (kustomize/cluster-scoped-resources)](https://github.com/kubeflow/pipelines/tree/2.2.0/manifests/kustomize/cluster-scoped-resources)
- [AWS Labs Kubeflow manifests (RDS + S3 overlay)](https://github.com/awslabs/kubeflow-manifests/tree/main/awsconfigs/apps/pipeline)
