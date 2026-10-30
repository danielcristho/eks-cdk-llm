"""
Compiles to finetune_pipeline.yaml.
Data-prep -> Ray Tune LoRA search -> LoRA fine-tune (best config) -> AWQ quantize -> S3.
"""

from typing import NamedTuple

from kfp import dsl, compiler, kubernetes

GPU_BASE_IMAGE = "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"


@dsl.component(base_image="python:3.11-slim", packages_to_install=["datasets==2.21.0"])
def prepare_dataset(dataset_out: dsl.Output[dsl.Dataset], num_examples: int = 200):
    """Pull a small instruction-tuning subset and write it as JSONL."""
    from datasets import load_dataset

    ds = load_dataset("mlabonne/guanaco-llama2-1k", split=f"train[:{num_examples}]")
    ds.to_json(dataset_out.path)


@dsl.component(
    base_image=GPU_BASE_IMAGE,
    packages_to_install=[
        "transformers==4.44.2", "peft==0.12.0", "accelerate==0.33.0", "datasets==2.21.0",
        "boto3>=1.28.0", "pyarrow<17", "ray[tune]==2.40.0",
    ],
)
def tune_hyperparams(
    dataset_in: dsl.Input[dsl.Dataset],
    trials_report: dsl.Output[dsl.Markdown],
    bucket_name: str,
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    num_trials: int = 6,
    max_epochs: int = 3,
) -> NamedTuple("BestConfig", [("lora_r", int), ("lora_alpha", int), ("learning_rate", float), ("beats_baseline", bool)]):
    """Search LoRA rank, alpha, and learning rate with Ray Tune + ASHA; return the best config."""
    import os
    from collections import namedtuple

    import boto3
    import ray
    import torch
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from ray import train, tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search import BasicVariantGenerator

    # pipeline_llm's hand-picked defaults, run as one of the trials so the report shows where they rank
    baseline = {"lora_r": 16, "lora_alpha": 32, "learning_rate": 5e-5}

    # same S3 base-model cache as finetune_lora, so the search skips the Hugging Face download through NAT
    cache_prefix = f"base-model-cache/{base_model.replace('/', '_')}"
    local_dir = "/tmp/base-model"
    s3 = boto3.client("s3")
    try:
        s3.head_object(Bucket=bucket_name, Key=f"{cache_prefix}/config.json")
        cached = True
    except s3.exceptions.ClientError:
        cached = False

    if cached:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket_name, Prefix=f"{cache_prefix}/"):
            for obj in page.get("Contents", []):
                dest = os.path.join(local_dir, obj["Key"][len(cache_prefix) + 1:])
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                s3.download_file(bucket_name, obj["Key"], dest)
    else:
        snapshot_download(repo_id=base_model, local_dir=local_dir)
        for root, _, files in os.walk(local_dir):
            for name in files:
                path = os.path.join(root, name)
                s3.upload_file(path, bucket_name, f"{cache_prefix}/{os.path.relpath(path, local_dir)}")

    splits = load_dataset("json", data_files=dataset_in.path, split="train").train_test_split(test_size=0.2, seed=42)
    train_path, eval_path = "/tmp/tune-train.jsonl", "/tmp/tune-eval.jsonl"
    splits["train"].to_json(train_path)
    splits["test"].to_json(eval_path)

    use_gpu = torch.cuda.is_available()

    def trial(config):
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from ray import train
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

        def tokenize(batch):
            out = tokenizer(batch["text"], truncation=True, max_length=512, padding="max_length")
            out["labels"] = out["input_ids"].copy()
            return out

        train_ds = load_dataset("json", data_files=train_path, split="train").map(tokenize, batched=True, remove_columns=["text"])
        eval_ds = load_dataset("json", data_files=eval_path, split="train").map(tokenize, batched=True, remove_columns=["text"])

        model = AutoModelForCausalLM.from_pretrained(
            local_dir,
            torch_dtype=torch.float16 if use_gpu else torch.float32,
            device_map={"": 0} if use_gpu else None,
        )
        model = get_peft_model(model, LoraConfig(
            r=config["lora_r"], lora_alpha=config["lora_alpha"], lora_dropout=0.05, task_type="CAUSAL_LM",
        ))

        # one report per epoch is what ASHA compares trials on, and where it stops the losers
        class ReportToTune(TrainerCallback):
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                train.report({"eval_loss": metrics["eval_loss"], "epoch": state.epoch})

        Trainer(
            model=model,
            args=TrainingArguments(
                output_dir="/tmp/tune-out",
                num_train_epochs=max_epochs,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=2,
                learning_rate=config["learning_rate"],
                eval_strategy="epoch",
                save_strategy="no",
                logging_steps=10,
                fp16=use_gpu,
                report_to=[],
            ),
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            callbacks=[ReportToTune()],
        ).train()

    ray.init(num_gpus=1 if use_gpu else 0, include_dashboard=False)
    tuner = tune.Tuner(
        tune.with_resources(trial, {"cpu": 2, "gpu": 1 if use_gpu else 0}),
        param_space={
            "lora_r": tune.choice([8, 16, 32]),
            "lora_alpha": tune.choice([16, 32, 64]),
            "learning_rate": tune.loguniform(5e-5, 5e-4),
        },
        tune_config=tune.TuneConfig(
            metric="eval_loss",
            mode="min",
            num_samples=num_trials,
            search_alg=BasicVariantGenerator(points_to_evaluate=[baseline]),
            # one T4, one trial at a time (also keeps CPU smoke runs from running trials in parallel)
            max_concurrent_trials=1,
            scheduler=ASHAScheduler(max_t=max_epochs, grace_period=1, reduction_factor=2),
            # a fresh worker process per trial, so each trial's GPU memory is released when it ends
            reuse_actors=False,
        ),
        run_config=train.RunConfig(storage_path="/tmp/ray-results", name="lora-tune"),
    )
    results = tuner.fit()
    best = results.get_best_result()

    df = results.get_dataframe()
    rows = [
        f"| {r['config/lora_r']} | {r['config/lora_alpha']} | {r['config/learning_rate']:.2e} "
        f"| {int(r['training_iteration'])} | {r['eval_loss']:.4f} "
        f"| {'baseline' if all(r[f'config/{k}'] == v for k, v in baseline.items()) else ''} |"
        for _, r in df.sort_values("eval_loss").iterrows()
    ]
    report = "\n".join([
        "| lora_r | lora_alpha | learning_rate | epochs run | eval_loss | note |",
        "|---|---|---|---|---|---|",
        *rows,
    ])
    print(report)
    with open(trials_report.path, "w") as f:
        f.write(report)

    BestConfig = namedtuple("BestConfig", ["lora_r", "lora_alpha", "learning_rate", "beats_baseline"])
    return BestConfig(
        best.config["lora_r"], best.config["lora_alpha"], best.config["learning_rate"], best.config != baseline,
    )


@dsl.component(
    base_image=GPU_BASE_IMAGE,
    packages_to_install=[
        "transformers==4.44.2", "peft==0.12.0", "accelerate==0.33.0", "datasets==2.21.0",
        "boto3>=1.28.0",
        # transformers/peft/accelerate at these versions pin numpy<2, so force numpy up is a
        # dead end; pin pyarrow down instead, since datasets pulls it unpinned and latest
        # pyarrow needs a numpy 2.0 ABI the base image's numpy 1.26.4 doesn't have
        "pyarrow<17",
    ],
)
def finetune_lora(
    dataset_in: dsl.Input[dsl.Dataset],
    model_out: dsl.Output[dsl.Model],
    bucket_name: str,
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    epochs: int = 1,
    lora_r: int = 16,
    lora_alpha: int = 32,
    learning_rate: float = 5e-5,
):
    """LoRA fine-tune base_model on the prepared dataset, then merge and save full weights."""
    import os
    import torch
    import boto3
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    """
    Base weights get pulled from Hugging Face once and cached in S3 (behind a VPC gateway
    endpoint, so reads are free, no NAT charge); every retry after that loads from S3 instead
    of re-downloading over the internet. Debugging sessions retry this step a lot, and repeated
    multi-GB downloads through NAT were the single biggest cost driver the last time around.
    """
    # not "model-cache/" - vllm_stack.py's init container already syncs that whole prefix
    # for the serving model, and would needlessly pull this raw base-model cache down too
    cache_prefix = f"base-model-cache/{base_model.replace('/', '_')}"
    local_dir = "/tmp/base-model"
    s3 = boto3.client("s3")
    try:
        s3.head_object(Bucket=bucket_name, Key=f"{cache_prefix}/config.json")
        cached = True
    except s3.exceptions.ClientError:
        cached = False

    if cached:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket_name, Prefix=f"{cache_prefix}/"):
            for obj in page.get("Contents", []):
                dest = os.path.join(local_dir, obj["Key"][len(cache_prefix) + 1:])
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                s3.download_file(bucket_name, obj["Key"], dest)
    else:
        snapshot_download(repo_id=base_model, local_dir=local_dir)
        for root, _, files in os.walk(local_dir):
            for name in files:
                path = os.path.join(root, name)
                s3.upload_file(path, bucket_name, f"{cache_prefix}/{os.path.relpath(path, local_dir)}")

    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    use_gpu = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        torch_dtype=torch.float16 if use_gpu else torch.float32,
        device_map={"": 0} if use_gpu else None,
    )
    model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, task_type="CAUSAL_LM"))

    def tokenize(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=512, padding="max_length")
        out["labels"] = out["input_ids"].copy()  # causal LM: Trainer needs labels to compute loss
        return out

    ds = load_dataset("json", data_files=dataset_in.path, split="train")
    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

    Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/lora-out",
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=learning_rate,
            logging_steps=10,
            save_strategy="no",
            fp16=use_gpu,
            report_to=[],
        ),
        train_dataset=ds,
    ).train()

    merged = model.merge_and_unload()
    merged.save_pretrained(model_out.path)
    tokenizer.save_pretrained(model_out.path)


@dsl.component(
    base_image=GPU_BASE_IMAGE,
    packages_to_install=["autoawq==0.2.6", "transformers==4.44.2", "boto3>=1.28.0"],
)
def quantize_and_upload(model_in: dsl.Input[dsl.Model], bucket_name: str, s3_prefix: str, run_id: str) -> str:
    """AWQ-quantize the merged model, then push it to its own versioned S3 prefix; returns the model's S3 URI."""
    import pathlib
    import subprocess

    """
    autoawq/transformers pull in a torchvision build incompatible with this image's torch==2.4.0
    (AttributeError: torch.library has no attribute register_fake), even when torchvision is
    pinned to a matching version. We have no use for torchvision at all here (text-only model),
    and transformers gracefully skips its vision imports when the package isn't installed, so
    just remove it before transformers/awq ever get imported.
    """
    subprocess.run(["pip", "uninstall", "-y", "-q", "torchvision"], check=False)

    import boto3
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_in.path)
    model = AutoAWQForCausalLM.from_pretrained(model_in.path)
    model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"})

    out_dir = pathlib.Path("/tmp/awq-out")
    model.save_quantized(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # one prefix per run, so the model being replaced stays in S3 for rollback
    prefix = f"{s3_prefix}/{run_id}"
    s3 = boto3.client("s3")
    for f in out_dir.rglob("*"):
        if f.is_file():
            s3.upload_file(str(f), bucket_name, f"{prefix}/{f.name}")
    return f"s3://{bucket_name}/{prefix}"


def _on_gpu_node(task: dsl.PipelineTask) -> dsl.PipelineTask:
    """karpenter's gpu NodePool: labels workload=gpu, taints nvidia.com/gpu=true:NoSchedule."""
    task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)
    kubernetes.add_node_selector(task, label_key="workload", label_value="gpu")
    kubernetes.add_toleration(task, key="nvidia.com/gpu", operator="Equal", value="true", effect="NoSchedule")
    return task


@dsl.pipeline(
    name="qwen-lora-tune-finetune",
    description="Prep data -> Ray Tune LoRA search -> if it beats the baseline: LoRA fine-tune -> AWQ quantize -> S3",
)
def finetune_pipeline(
    bucket_name: str,
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    num_examples: int = 200,
    num_trials: int = 6,
    tune_max_epochs: int = 3,
    s3_prefix: str = "pipeline-output",
):
    prep = prepare_dataset(num_examples=num_examples)

    search = tune_hyperparams(
        dataset_in=prep.outputs["dataset_out"], bucket_name=bucket_name, base_model=base_model,
        num_trials=num_trials, max_epochs=tune_max_epochs,
    )
    _on_gpu_node(search)

    # baseline won: the served model uses the best config, so skip the GPU steps and keep it
    with dsl.If(search.outputs["beats_baseline"] == True):  # noqa: E712
        # ASHA lets a trial reach max_t if it keeps winning, so the best config is proven at that length
        train = finetune_lora(
            dataset_in=prep.outputs["dataset_out"], bucket_name=bucket_name, base_model=base_model,
            epochs=tune_max_epochs, lora_r=search.outputs["lora_r"], lora_alpha=search.outputs["lora_alpha"],
            learning_rate=search.outputs["learning_rate"],
        )
        _on_gpu_node(train)

        quant = quantize_and_upload(
            model_in=train.outputs["model_out"], bucket_name=bucket_name, s3_prefix=s3_prefix,
            run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
        )
        _on_gpu_node(quant)
        # rollout is a Git commit: set modelPrefix in manifests/vllm/values.yaml to pipeline-output/<run-id>


if __name__ == "__main__":
    compiler.Compiler().compile(finetune_pipeline, __file__.replace(".py", ".yaml"))
