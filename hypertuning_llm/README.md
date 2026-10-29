# LLM on EKS: Hyperparameter Tuning with Ray Tune

Hi there! 👋🏽 In the previous post, vLLM started serving our fine-tuned `Qwen2.5-1.5B-Instruct` from S3, and we can switch model versions with a blue/green rollout (see [README-rollout.md](./README-rollout.md)). But one thing bugged me: the LoRA settings behind that model were a guess.

So in this post, we'll measure them. The pipeline searches for better settings with Ray Tune, and it rolls out a new model when one of them wins.

## TL;DR

- A new `tune_hyperparams` step runs Ray Tune inside a single GPU pod, searching LoRA `r`, `lora_alpha`, and `learning_rate` on a held-out eval split.
- ASHA stops weak trials after one epoch, so 6 trials cost a lot less than 6 full trainings.
- The previous post's hand-picked settings run as one of the trials (the baseline), so we can see where that guess ranks.
- If the best config doesn't beat the baseline, the pipeline stops and the served model stays as it is.
- Each run writes its model to its own S3 prefix, and a final `deploy_model` step rolls it out with the blue/green mechanism from the previous post.

## Why Tune at All?

In the previous post, I picked the LoRA settings (`r=16`, `alpha=32`, `lr=5e-5`) because they're common defaults, not because I measured them. The pipeline trained on all 200 examples with no held-out set. The one signal we had was training loss, and it was going down when the epoch ended. That tells us the model was learning, not that these settings were good. And `5e-5` is on the low side for LoRA, most setups use something between `1e-4` and `2e-4`.

So the problem isn't that the first model is bad. We didn't measure it. In this post we hold out 20% of the data, try several LoRA configs under the same conditions, and keep the one with the lowest eval loss. The old defaults run as one of the trials. If they win, that's a real answer too: we'd know they're good instead of guessing.

## Does the Chatbot Need Fine-Tuning at All?

Tuning answers "which LoRA config is best?". But there's a question before that: "does LoRA beat the base model at all?". If it doesn't, we're tuning something we don't need.

For the AWS Community Day Jakarta chatbot, my guess is no. People ask about facts: times, rooms, speakers. Facts go in the prompt, not in the weights. The whole event knowledge base (`src/knowledge_base.md`, generated from the event API by `src/build_kb.py`) is around 1.5k tokens. That fits in `max_model_len` 4096, so `Qwen2.5-1.5B-Instruct` can read it from the system prompt without a retrieval index. Fine-tuning changes how the model answers, not what it knows. And an agenda baked into the weights goes stale once the schedule changes.

Eval loss isn't chatbot quality either. A lower loss doesn't mean the model gets "what time is the LLM talk?" right.

### What Fine-Tuning Could Teach

The knowledge base goes into every request, so training has to look the same: context in the prompt, answer taken from it. The goal isn't to memorize the agenda. It's to teach the behavior around it: find the answer in the context, and say "I don't know" when it isn't there. That second part is where a 1.5B model tends to slip and make up a time or a speaker.

**Dataset, generated from `src/knowledge_base.md` with templates (no LLM in the loop):**

- Each session gives a few questions, in Indonesian and English: when is X, which room is X in, who presents X.
- Each example's context is a mini knowledge base: the line with the answer plus ~8 random lines around it. That keeps examples around 300 tokens instead of 1.5k, which matters on a T4.
- Unanswerable examples: the same question with the answer line removed from the context, and the target reply is "not in the event info, check jakarta.awscommunity.id/comday". Plus questions the event page doesn't cover (parking, wifi, dress code).
- Split by session, not by row. Sessions in the test set don't show up in training, so the test score measures lookup, not memory.

**Scoring, automatic:**

- Answerable: correct if the reply contains the right time, room or name.
- Unanswerable: correct if the reply says it doesn't know and contains no invented time or name.

**Three models, one test set:**

| Model | Question it answers |
|---|---|
| Base Qwen + knowledge base | the no-training baseline |
| LoRA, old settings (`r=16`, `alpha=32`, `lr=5e-5`) | does fine-tuning help? |
| LoRA, Ray Tune's best config | does tuning help on top of that? |

Ray Tune ranks trials with `eval_loss` because it's cheap per trial, but the rollout decision uses test accuracy. And we run Ray Tune when the old-settings LoRA beats the base model. If it doesn't, that's the finding: for a chatbot about facts, a knowledge base in the prompt is enough, and fine-tuning or tuning doesn't add anything.

My guess: the base model can handle simple lookups like "what time is X?". If there's a gap, it shows up in "I don't know" answers and in multi-step questions like "what's on the Main Stage after lunch?".

**Has to change before this runs:**

- The dataset. `mlabonne/guanaco-llama2-1k` uses the Llama-2 `[INST]` template; the new one is built with Qwen's chat template.
- Loss masking. `finetune_lora` and `tune_hyperparams` set `labels = input_ids`, so the loss covers the prompt too. With a long context in the prompt, most of that loss is about copying the knowledge base. Prompt tokens need to be masked (`-100`) so the loss counts the answer and nothing else.

## What Changed in the Pipeline

```
prepare_dataset -> tune_hyperparams -> [beats baseline?] -> finetune_lora -> quantize_and_upload -> deploy_model
```

<!-- TODO: architecture diagram -->

### Ray Tune in One Pod

<!-- TODO: why local mode on one GPU pod instead of the serving RayCluster, search space, ASHA, max_concurrent_trials=1, reuse_actors=False -->

### The Baseline Gate

<!-- TODO: points_to_evaluate, dsl.If on beats_baseline, why skipping saves GPU time -->

## Rolling Out the Winner

<!-- TODO: deploy_model points the RayService at pipeline-output/<run-id>; the rollout itself is covered in README-rollout.md. With Argo CD, deploy_model's kubectl patch gets reverted by selfHeal; the pipeline should commit modelPrefix in manifests/vllm/values.yaml instead -->

## Results

<!-- TODO: test accuracy for base vs old-settings LoRA vs tuned LoRA, a few answers side by side (e.g. the parking question) -->

## Chatting with the Model

<!-- TODO: the chatbot setup is in README-rollout.md; here, the same questions against the tuned model -->

## Deploy

<!-- TODO -->

## Running the Pipeline

<!-- TODO: trials table from the KFP UI, best config vs baseline, rollout logs -->

## Conclusion

<!-- TODO -->
