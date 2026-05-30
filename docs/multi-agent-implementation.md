# Multi-Agent Implementation Setup

This repo uses Agent Orchestrator to coordinate MLX REAP implementation work.

## Role Split

- Orchestrator: OpenCode using `ollama/deepseek-v4-pro:cloud`
  - planning, issue triage, implementation decomposition, PR review, docs review, and worker coordination
  - read-mostly; implementation edits should be delegated to workers
- Worker: Codex using `gpt-5.5-xhigh`
  - code implementation, tests, narrow issue-scoped docs, and verification
  - one worker per issue or tightly scoped issue slice

## Project Invariants

- The existing PyTorch/CUDA REAP implementation remains the production and official experimentation workflow.
- The MLX backend is a parallel Apple Silicon experimentation backend.
- MLX implementation choices must respect MLX constraints directly.
- Qwen3 is the bootstrap/reference adapter, not the final architecture boundary.
- Support for arbitrary compatible MoE weights must be adapter-driven.

## Local Prerequisites

- `ao` from `@aoagents/ao`
- `tmux`
- `gh` authenticated for `egesabanci/reap-mlx`
- `opencode`
- `ollama` with `deepseek-v4-pro:cloud` available locally as model metadata
- Codex CLI authenticated for the requested `gpt-5.5-xhigh` model

Check the local setup:

```bash
ao --version
tmux -V
gh auth status
opencode --version
ollama list | rg 'deepseek-v4-pro:cloud'
opencode models ollama | rg 'deepseek-v4-pro:cloud'
```

If OpenCode does not show the model, refresh Ollama Cloud model metadata:

```bash
ollama pull deepseek-v4-pro:cloud
```

## Start AO

From the repo root:

```bash
ao doctor
ao start reap-mlx
```

The dashboard runs on `http://localhost:3000` by default.

## Spawn Work

Use issue numbers from the `MLX Backend Implementation` milestone.

```bash
ao spawn 1
ao spawn 2
ao batch-spawn 3 4
```

The default worker is Codex with `gpt-5.5-xhigh`. Use explicit overrides only for exceptional cases:

```bash
ao spawn 4 --agent codex
```

## Recommended Execution Order

1. Finish foundation and dependency boundaries.
2. Add the MLX model loading and smoke-test harness.
3. Define the adapter and MoE weight layout contract.
4. Implement routing capture against adapter-described MoE weights.
5. Implement pruning.
6. Implement merging.
7. Add cross-model validation fixtures and Apple Silicon workflow docs.
8. Run final parity and limitation documentation passes.

## Worker Hand-Off Checklist

Each worker should report:

- issue number and branch
- files changed
- implementation summary
- tests or verification commands run
- known limitations or skipped checks
- follow-up issue links if new scope is discovered
