"""
Run prepare_dataset -> tune_hyperparams -> finetune_lora locally on CPU with a tiny Qwen2
model, before spending any GPU time on EKS. Calls each component's python_func directly,
so it exercises the real component code without KFP. The base-model cache step talks to the
real S3 bucket (the tiny model is a few MB).

    HT_TEST_BUCKET=eks-llm-pipeline-model-bucket python tests/smoke_pipeline_local.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from survive_llm.kubeflow.pipelines.finetune_pipeline import (  # noqa: E402
    finetune_lora,
    prepare_dataset,
    tune_hyperparams,
)

TINY_MODEL = "yujiepan/qwen2.5-tiny-random"


class Artifact:
    def __init__(self, path):
        self.path = path


def main():
    bucket = os.environ.get("HT_TEST_BUCKET", "eks-llm-pipeline-model-bucket")
    work = tempfile.mkdtemp(prefix="ht-smoke-")

    dataset = Artifact(os.path.join(work, "dataset.jsonl"))
    prepare_dataset.python_func(dataset_out=dataset, num_examples=40)

    report = Artifact(os.path.join(work, "trials.md"))
    best = tune_hyperparams.python_func(
        dataset_in=dataset, trials_report=report, bucket_name=bucket,
        base_model=TINY_MODEL, num_trials=2, max_epochs=2,
    )
    print("best config:", best)
    assert best.lora_r in (8, 16, 32)
    assert best.lora_alpha in (16, 32, 64)
    assert 5e-5 <= best.learning_rate <= 5e-4
    report_text = open(report.path).read()
    assert report_text.count("\n") >= 3, "trials report should have a header plus one row per trial"
    assert "baseline" in report_text, "pipeline_llm defaults should run as one of the trials"
    assert isinstance(best.beats_baseline, bool)

    model = Artifact(os.path.join(work, "model"))
    finetune_lora.python_func(
        dataset_in=dataset, model_out=model, bucket_name=bucket, base_model=TINY_MODEL, epochs=1,
        lora_r=best.lora_r, lora_alpha=best.lora_alpha, learning_rate=best.learning_rate,
    )
    assert os.path.exists(os.path.join(model.path, "config.json")), "merged model was not saved"
    print("SMOKE_OK", work)


if __name__ == "__main__":
    main()
