# Code style

Clean Code, SOLID and DRY apply to every part of eks-cdk-mlops: the CDK stacks,
the Kubeflow pipeline components, the Ray Serve script and the Streamlit chatbot.

## Project shape

Each top-level folder (`serves_llm`, `scale_llm`, `pipeline_llm`,
`hypertuning_llm`) is its own CDK app and its own blog post. Each one is copied
from the previous one, not imported from it, so every post's code stays runnable
as it was published. A change in one project does not get backported to the
others unless asked.

## Clean Code

- Write the smallest code that solves the actual problem. Delete before adding.
- Prefer boring and obvious over clever.
- Names carry the meaning, so most code needs no comment.
- A function does one thing at one level of abstraction.
- Files stay under 500 lines; split by responsibility.
- No dead code, no commented-out code.
- No silent failure: no `|| true` on a step that has to work, no bare `except`.
  A failed S3 sync or model download stops the pod with a clear error.
- No hardcoded AWS account, region, bucket or cluster name; they come from
  `.env` through `python-dotenv` in `app.py`.

## SOLID, as this project reads it

- **Single responsibility.** One stack per concern: `eks -> karpenter -> ray ->
  vllm -> kubeflow`, each in `<project>/<name>/<name>_stack.py` exporting a
  `<Name>Stack`. One KFP component per pipeline step: `prepare_dataset`,
  `tune_hyperparams`, `finetune_lora`, `quantize_and_upload`, `deploy_model`.
- **Open/closed.** A base class or helper layer only at a seam with a second
  real implementation. Elsewhere, the concrete stack or function.
- **Liskov substitution.** A local stand-in owes the real thing's behavior, not
  just its signature. The CPU smoke test with a tiny random model checks the
  code path, not model quality, and is never quoted as a result.
- **Interface segregation.** A stack takes the few constructs it uses as
  constructor arguments (`cluster`, `model_bucket_name`), not the whole stack.
- **Dependency inversion.** `app.py` wires stacks together and sets the order
  with `add_dependency`. A stack does not look up another stack's resources by
  name.

SOLID misapplied is a base class for every stack and a module for every
function, the opposite of clean code. A new abstraction names the second
implementation that needs it.

## DRY

One source of truth per piece of knowledge, within a project:

- versions pinned once: `requirements.txt`, Helm chart versions, container
  images, `packages_to_install` per component, `pip` in the RayService
  `runtime_env`;
- shared values passed down from `app.py` (bucket name, cluster) instead of
  repeated per stack;
- the KFP manifest kept in `kubeflow/manifests/*.yaml` and patched in
  `kubeflow_stack.py`, not rewritten in Python.

KFP lightweight components run in their own container and cannot import the
project, so each one keeps its imports inside the function. Code shared by two
components (the S3 base-model cache) is repeated on purpose; extract it only by
moving to a shared base image.

Extend an existing helper before writing a second one. Do not merge code that
only looks alike: DRY is about one piece of knowledge, not similar lines.

## Configuration

- Per-account values live in `.env` (`AWS_DEFAULT_ACCOUNT`,
  `AWS_DEFAULT_REGION`, `AWS_BUCKET`, `CLUSTER_NAME`), with `.env.example`
  listing every key.
- Values that change per run (number of examples, trials, epochs, base model,
  S3 prefix) are pipeline parameters with defaults, set in the KFP UI, not
  edited in code.
- Facts of the setup stay in code: namespaces (`kuberay`, `kubeflow`), the GPU
  instance type, taints and labels that other manifests select on.

## Cost

- GPU nodes come from Karpenter and scale to zero when idle; nothing GPU-bound
  runs on the system node group.
- Validate offline first: a CPU smoke test with a tiny model
  (`tests/smoke_pipeline_local.py`) before spending GPU time on EKS.
- Deploy, demo, destroy. Nothing is left running between sessions.

## Blog numbers

- Every log line, timing, loss or cost in a README comes from a real run, as
  pasted output or a screenshot. An estimate is labelled as an estimate.
- A value picked by hand (a default LoRA rank, a learning rate) is said to be
  picked by hand, not presented as tuned.

## Comments

- Default to none. A comment earns its place for a non-obvious why or
  constraint: a gotcha, a workaround, a limit of a tool.
- Short, single-line `#`. If the why needs more than one sentence, a
  `"""..."""` block above the code instead of stacked `#` lines.
- One line at the top of a file says what it holds, as a module docstring.
- A KFP component gets a one-line docstring saying what the step does.
- No adverbs ("only", "safely", "usually", "never", "already"), no tags.
- Never delete or reword comments the user wrote, even when they break these
  rules; point it out at most.

## Python

- Python 3.12, a `.venv` per project (`python3 -m venv .venv`,
  `pip install -r requirements.txt`).
- Type hints on functions; KFP needs them on every component parameter.
- Tests live in each project's `tests/`; a check is the smallest runnable
  thing that fails when the logic breaks.
