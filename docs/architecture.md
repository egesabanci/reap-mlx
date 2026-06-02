# Architecture

REAP MLX is organized around a small set of modules with explicit dependency
boundaries. The core design goal is to make model-family differences local while
keeping the main pruning pipeline inspectable.

## Module Responsibilities

| Module | Responsibility | Heavy runtime imports |
| --- | --- | --- |
| `reap.__init__` | Package root. Exposes no public runtime symbols. | None |
| `reap.data` | Calibration text extraction and tokenization. | `datasets` only when loading without injection |
| `reap.model_adapters` | Model layout discovery and config updates. | `mlx_lm` only for mask helper calls |
| `reap.router` | Architecture-specific selected-router behavior. | `mlx.core` only when routing executes |
| `reap.metrics` | NumPy `PruningState` accumulation. | None beyond NumPy |
| `reap.observer` | MLX layer replay and selected-expert metric collection. | `mlx.core` when observing |
| `reap.prune` | In-place expert slicing and pruning method resolution. | None beyond NumPy |
| `reap.save` | Save, reload, shape validation, and smoke generation. | `mlx_lm` when default save/load/generate executes |
| `reap.validation_metrics` | Structured run telemetry. | `mlx.core` only for memory sampling |
| `reap.entrypoint` | CLI parsing and phase orchestration. | Deferred through called functions |

## Import-Safety Boundary

Import safety is a hard contract. Tests install a subprocess import blocker and
verify that importing each module does not import:

- `torch`
- `vllm`
- `mlx`
- `mlx_lm`
- `datasets` where applicable

This lets users run `python -m reap.entrypoint --help` and import package modules
without requiring Apple Silicon MLX runtime packages at import time.

When adding code, keep runtime imports inside the function that directly needs
the dependency. Do not add top-level imports for MLX, MLX-LM, Hugging Face
datasets, Torch, or vLLM.

## Data Flow

```txt
CLI args
  -> RunMetrics.record_run_config
  -> MLX-LM load
  -> adapter inference
  -> calibration records
  -> token sequences
  -> observer reports
  -> keep indices by layer
  -> mutated model/config
  -> saved artifact
  -> reloaded model/config
  -> validation metrics
```

The live model object is mutated during pruning. The config mapping is also
mutated before save so MLX-LM writes the pruned expert count.

## Adapter Boundary

Adapters describe architecture layout only. They do not perform routing, metric
accumulation, pruning, saving, or generation.

An adapter provides:

- model layer access;
- MoE layer identification;
- access to the MoE module for a layer;
- access to the dense MLP module for a non-MoE layer;
- per-layer MoE metadata: expert count, top-k, normalization flag, and optional
  LFM2 expert-bias flag.

Router classes are separate because the selected-route calculation differs
between model families.

## Observer Boundary

The observer does not collect all expert outputs. It only asks the MoE
`switch_mlp` for selected experts returned by the router. This keeps the metric
surface aligned with pruning methods and avoids full expert materialization.

The observer returns plain dictionaries containing NumPy-compatible arrays. That
keeps pruning independent of MLX graph state.

## Pruning Boundary

Pruning is implemented as first-dimension slicing of expert-stacked fields. The
module supports regular attributes and MLX-LM modules exposing `.get(field)`.

Sliceable fields include:

- switch projection `weight`, `scales`, `biases`, `bias`;
- gate `weight`, `bias`, `e_score_correction_bias` for Qwen3-style gates;
- gate `weight`, `scales`, `biases`, `bias` for LFM2-style gates;
- LFM2 `expert_bias` when enabled or present.

Pruning updates runtime attributes after slicing so later forward passes and
reload validation see the new expert count.

## Metrics Boundary

`RunMetrics` records high-level telemetry. It does not control pipeline
behavior. A failed metrics write is logged in the CLI exception path but should
not hide the original runtime exception.

`PruningState` records pruning saliency inputs. It is separate from
`RunMetrics` because it is part of observation and pruning behavior, not only
telemetry.

## Design Invariants

- `compression_ratio` must be finite and in `[0, 1)`.
- `max_samples` and `max_seq_length` must be positive integers.
- Calibration sequences must contain at least one token.
- Observer input supports `[seq]` and `[1, seq]`; larger batches are rejected.
- Router input supports `[tokens, hidden]` and `[batch, seq, hidden]`.
- Saliency arrays must be one-dimensional and match the layer expert count.
- Saliency arrays must not contain NaN values.
- Saved artifacts must reload and validate before the run is marked successful.

