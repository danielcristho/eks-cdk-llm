# LLM on EKS: Hyperparameter Tuning with Ray Tune

Hi there! 👋🏽 In the previous post, a Kubeflow pipeline fine-tuned `Qwen2.5-1.5B-Instruct` with LoRA, quantized it with AWQ, and pushed it to S3. Two things were left open. The LoRA settings were guesses, and the serving stack never actually picked up the pipeline's output. This post closes both: the pipeline now searches for better settings with Ray Tune, and when it finds one, it rolls the new model out to vLLM with no downtime.

## TL;DR

- A new `tune_hyperparams` step runs Ray Tune inside a single GPU pod, searching LoRA `r`, `lora_alpha`, and `learning_rate` on a held-out eval split.
- ASHA stops weak trials after one epoch, so 6 trials cost far less than 6 full trainings.
- The previous post's hand-picked settings run as one of the trials (the baseline), so we can see exactly where that guess ranks.
- The pipeline only continues if the best config beats the baseline. Otherwise it stops and the served model stays as it is.
- Each run writes its model to its own S3 prefix, and a final `deploy_model` step points the vLLM RayService at it. KubeRay brings up a new cluster with the new model and switches traffic only once it's serving.

## Why Tune at All?

In the previous post, the LoRA settings (`r=16`, `alpha=32`, `lr=5e-5`) were common defaults, not measured choices. The pipeline trained on all 200 examples with no held-out set, so the only signal was training loss, which was still dropping when the single epoch ended. That tells us the model was learning, not whether these settings were any good. `5e-5` is also on the low side for LoRA, which is usually trained somewhere around `1e-4` to `2e-4`.

So the problem isn't that the first model was proven bad. It's that we never measured it. This post adds that measurement: hold out 20% of the data, try several LoRA configurations under identical conditions, and keep the one with the lowest eval loss. The old defaults run as one of the trials, so if they turn out to be the best, that's a real answer too. We'd know they're good instead of guessing.

## What Changed in the Pipeline

```
prepare_dataset -> tune_hyperparams -> [beats baseline?] -> finetune_lora -> quantize_and_upload -> deploy_model
```

<!-- TODO: architecture diagram -->

### Ray Tune in One Pod

<!-- TODO: why local mode on one GPU pod instead of the serving RayCluster, search space, ASHA, max_concurrent_trials=1, reuse_actors=False -->

### The Baseline Gate

<!-- TODO: points_to_evaluate, dsl.If on beats_baseline, why skipping saves GPU time -->

## Rolling Out Model B

Model A is what's serving now: the previous post's output at `s3://<bucket>/pipeline-output/latest`. Model B is whatever this pipeline produces when tuning finds a better config.

<!-- TODO: MODEL_URI in rayClusterConfig vs serveConfigV2 (zero-downtime NewCluster upgrade vs in-place restart), serveService so the NLB follows the active cluster, min_replicas=1, stable served_model_name "qwen-lora", versioned S3 prefix for rollback, RBAC for pipeline-runner, IMDS hop limit fix -->

## Chatting with the Model

<!-- TODO: Streamlit chatbot (src/app.py) pointed at the NLB, before/after rollout -->

## Deploy

<!-- TODO -->

## Running the Pipeline

<!-- TODO: trials table from the KFP UI, best config vs baseline, rollout logs -->

## Conclusion

<!-- TODO -->
