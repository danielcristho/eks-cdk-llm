# Commits

- Commit when asked, not before.
- One change per commit: a stack, a component, a README section. Split a mixed
  working tree into several commits.
- One-line message, Conventional Commits prefix, lowercase, no trailing period:
  `feat:`, `fix:`, `docs:`, `chore:`, `test:`, `refactor:`. Name the thing
  changed: `feat: add deploy_model step (roll out to vLLM RayService)`.
- Never commit `.env`, `cdk.out/`, `.venv/`, `__pycache__/` or `.brainstorm/`.
