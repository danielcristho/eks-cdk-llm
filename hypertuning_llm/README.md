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

## Does the Chatbot Need Fine-Tuning at All?

Tuning answers "which LoRA config is best?". There's an earlier question: "does LoRA beat the base model at all?". If it doesn't, tuning optimizes something we don't need.

For the AWS Community Day Jakarta chatbot, the answer is probably no. What users ask about are facts: times, rooms, speakers. Facts belong in the prompt, not the weights. The whole event knowledge base (`src/knowledge_base.md`, regenerated from the event API by `src/build_kb.py`) is about 1.5k tokens, which fits in `max_model_len` 4096, so `Qwen2.5-1.5B-Instruct` reads it from the system prompt with no retrieval index. Fine-tuning changes how the model answers, not what it reliably knows, and an agenda baked into weights goes stale the moment the schedule changes.

Eval loss isn't chatbot quality either. A lower loss doesn't mean the model answers "what time is the LLM talk?" correctly.

### What Fine-Tuning Could Still Teach

Since the knowledge base is injected into every request, training has to use the same shape: context in the prompt, answer drawn from it. The goal isn't to memorize the agenda but to teach the behavior around it: find the answer in the context, and say "I don't know" when it isn't there. That second part is where a 1.5B model is most likely to slip and invent a time or a speaker.

**Dataset, generated from `src/knowledge_base.md` with templates (no LLM in the loop):**

- Each session yields a few questions, in Indonesian and English: when is X, which room is X in, who presents X.
- Each example's context is a mini knowledge base: the line holding the answer plus ~8 random distractor lines. That keeps examples around 300 tokens instead of 1.5k, which matters on a T4.
- Unanswerable examples: the same question with the answer line removed from the context, where the target reply is "not in the event info, check jakarta.awscommunity.id/comday". Plus questions the event page never covers (parking, wifi, dress code).
- Split by session, not by row. Sessions in the test set never appear in training, so the test score measures lookup, not recall.

**Scoring, automatic:**

- Answerable: correct if the reply contains the right time, room or name.
- Unanswerable: correct if the reply says it doesn't know and contains no invented time or name.

**Three models, one test set:**

| Model | Question it answers |
|---|---|
| Base Qwen + knowledge base | the no-training baseline |
| LoRA, old settings (`r=16`, `alpha=32`, `lr=5e-5`) | does fine-tuning help? |
| LoRA, Ray Tune's best config | does tuning help on top of that? |

Ray Tune keeps `eval_loss` to rank trials, since it's cheap per trial, but the rollout decision uses test accuracy. Ray Tune runs only if the old-settings LoRA beats the base model. If it doesn't, that's the finding: for a fact-driven chatbot, a knowledge base in the prompt is enough, and neither fine-tuning nor tuning adds anything.

My guess: the base model already handles simple lookups like "what time is X?". The gap, if any, shows up in "I don't know" answers and in multi-step questions like "what's on the Main Stage after lunch?".

**Has to change before this runs:**

- The dataset. `mlabonne/guanaco-llama2-1k` uses the Llama-2 `[INST]` template; the new one is built with Qwen's chat template.
- Loss masking. `finetune_lora` and `tune_hyperparams` set `labels = input_ids`, so loss covers the prompt too. With a context-heavy prompt, that loss is mostly about copying the knowledge base. Prompt tokens should be masked (`-100`) so loss only counts the answer.

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

<!-- TODO: MODEL_URI in rayClusterConfig vs serveConfigV2 (zero-downtime NewCluster upgrade vs in-place restart), KubeRay's default vllm-serve-svc (ClusterIP) follows the active cluster, no public NLB on the GPU, min_replicas=1, stable served_model_name "qwen-lora", versioned S3 prefix for rollback, RBAC for pipeline-runner, IMDS hop limit fix -->

## Chatting with the Model

<!-- TODO: chatbot (src/chat_gradio, src/chat_streamlit) calling vllm-serve-svc in-cluster, knowledge base as system prompt, test accuracy for base vs old-settings LoRA vs tuned LoRA -->

## Deploy

<!-- TODO -->

## Running the Pipeline

<!-- TODO: trials table from the KFP UI, best config vs baseline, rollout logs -->

## Conclusion

<!-- TODO -->
