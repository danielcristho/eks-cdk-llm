# AGENT.md

Four independent AWS CDK (Python) apps for serving and training LLMs on EKS, one
per blog post. No shared code between them; treat each as its own project root.

See also: [code_style.md](code_style.md), [commits.md](commits.md).

## Projects

- **`serves_llm/`**: baseline. EKS cluster + single vLLM Deployment
  (Llama-3.1-8B-AWQ) on one GPU node, plus a Streamlit chatbot in `src/app.py`.
- **`scale_llm/`**: autoscaling via 4 stacks deployed in order (see `app.py`):
  1. `eks/eks_stack.py`: VPC, EKS cluster, CPU node group, Karpenter IAM roles, S3 model bucket
  2. `karpenter/karpenter_stack.py`: Karpenter controller + GPU NodePool/EC2NodeClass
  3. `ray/ray_stack.py`: KubeRay operator + RayCluster, GPU workers start at 0
  4. `vllm/vllm_stack.py` + `serve_vllm.py`: NVIDIA device plugin + RayService running vLLM via Ray Serve

  Ray Serve autoscales vLLM replicas; Karpenter provisions/removes GPU nodes to match.
- **`pipeline_llm/`**: `scale_llm` + a 5th stack, `kubeflow/kubeflow_stack.py`
  (standalone Kubeflow Pipelines backed by RDS + S3). The pipeline in
  `kubeflow/pipelines/finetune_pipeline.py` LoRA fine-tunes and AWQ-quantizes
  `Qwen2.5-1.5B-Instruct` and pushes it to S3.
- **`hypertuning_llm/`**: WIP. `pipeline_llm` + a Ray Tune step (ASHA, baseline
  trial) before fine-tuning, and a `deploy_model` step that rolls the new model
  out to the vLLM RayService. Local CPU smoke test in `tests/smoke_pipeline_local.py`.
- **`survive_llm/`**: WIP. `hypertuning_llm` + ARC zonal shift: EKS zonal shift
  enabled through an `AwsCustomResource`, one NAT per AZ, one warm vLLM replica
  per AZ (`min_replicas: 2` + zone topology spread), `just zonal-shift-*` recipes.

## Commands (run inside the relevant project dir)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env   # fill in AWS_DEFAULT_ACCOUNT, AWS_ADMIN_USER, AWS_BUCKET, CLUSTER_NAME

cdk diff               # per-stack: cdk diff <StackName>
cdk deploy --all       # respects the stack dependency order in app.py
cdk destroy --all      # tear down: GPU nodes are expensive, don't leave them running
```

`serves_llm/` also has a `justfile` (`just deploy`, `just load-kubeconfig`, `just get-nodes`, `just get-pods`).

Compile the pipeline (`pipeline_llm`, `hypertuning_llm`):

```bash
python <project>/kubeflow/pipelines/finetune_pipeline.py   # writes finetune_pipeline.yaml next to it
```

## Conventions

- Each stack lives in its own package: `<project>/<name>/<name>_stack.py`, exporting a `<Name>Stack` class consumed from `app.py`.
- Config comes from `.env` (loaded via `python-dotenv` in `app.py`), never hardcode account/region/bucket names.
- `cdk.out/`, `.venv/`, `__pycache__/` are generated: don't edit or commit into them.
- The user runs `cdk deploy` / `cdk destroy`; agents don't deploy.

## Gotchas

- New AWS accounts default to 0 vCPU quota for G/VT on-demand instances: request an increase before deploying GPU node groups.
- `tests/unit/test_<project>_stack.py` in every project is unmodified `cdk init` boilerplate (imports a stack module that doesn't exist): don't trust it as a working example, and don't wire CI to it as-is.
- `pipeline_llm/` and `hypertuning_llm/` have no `.env.example`; copy `scale_llm/.env.example` and change the bucket and cluster names.
- `hypertuning_llm` pins `aws-cdk-lib==2.261.0`: newer releases emit cloud assembly schema 55, which CDK CLI 2.1132 can't read.
- If `cdk synth` fails writing the jsii cache, run it with `JSII_RUNTIME_PACKAGE_CACHE=disabled`.
