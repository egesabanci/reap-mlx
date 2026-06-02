# REAP MLX Technical Documentation

This directory contains maintainer-focused reference documentation for REAP MLX.
It describes the behavior implemented in `src/reap` and covered by
`tests/test_mlx_*.py`.

REAP MLX applies Router-weighted Expert Activation Pruning (REAP) to MLX-LM
Mixture-of-Experts models on Apple Silicon. The runtime pipeline is deliberately
linear:

```txt
load -> calibrate -> observe -> prune -> save -> reload -> smoke -> metrics
```

The package is also intentionally import-light. Importing `reap` or its public
modules must not import MLX, MLX-LM, datasets, Torch, or vLLM until the runtime
function that needs those packages executes.

## Documentation Map

| Document | Purpose |
| --- | --- |
| [Pipeline](pipeline.md) | End-to-end execution flow and phase responsibilities. |
| [Architecture](architecture.md) | Module boundaries, dependency timing, data flow, and invariants. |
| [Model Adapters](model-adapters.md) | Supported MoE layouts and adapter extension contract. |
| [Calibration](calibration.md) | Dataset loading, text extraction, tokenization, and sequence shape rules. |
| [Observation And Metrics](observation-and-metrics.md) | Router semantics, layer replay, and pruning metric accumulation. |
| [Pruning](pruning.md) | Saliency methods, expert ranking, in-place slicing, and config updates. |
| [Save Reload Validation](save-reload-validation.md) | Artifact saving, reload checks, shape validation, and smoke generation. |
| [CLI](cli.md) | Command-line options, examples, wrapper script, and failure behavior. |
| [Telemetry](telemetry.md) | `validation-metrics.json` schema and interpretation guide. |
| [Development](development.md) | Test commands, import-safety checks, and extension workflow. |

## Core Guarantees

REAP MLX currently guarantees:

- package imports stay light and do not pull in runtime-heavy dependencies;
- calibration sequences are unpadded batch-size-1 token arrays;
- model-family differences are isolated behind adapters and routers;
- observation replays model layers and collects selected-expert statistics only;
- pruning mutates live MLX-LM-style modules by slicing expert-stacked arrays on
  dimension 0;
- `num_experts` and `num_experts_per_tok` are updated after pruning, with top-k
  clamped to the retained expert count;
- saved artifacts are reloaded and validated before a run is considered
  successful;
- every run writes structured telemetry, including failure telemetry when the
  pipeline raises after startup.

## Supported Model Families

| Adapter | Model family | MoE module location | Notes |
| --- | --- | --- | --- |
| `lfm2_moe` | Liquid LFM2.5 MoE MLX-LM models | `layer.feed_forward` | Supports attention and conv/SSM operators plus optional `expert_bias`. |
| `qwen3_moe` | Qwen3-MoE MLX-LM-style models | `layer.mlp` | Default adapter when LFM2 cannot be inferred. |

The implementation assumes a uniform post-pruning expert count across MoE
layers because the saved config uses a single global `num_experts` value.

## Quick Maintainer Commands

```bash
uv sync --group dev
uv run python -m pytest -q tests/test_mlx_*.py
uv run python -m reap.entrypoint --help
uv lock --check
git diff --check
```

