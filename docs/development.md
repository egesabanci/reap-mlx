# Development

This document summarizes local development checks and extension expectations for
REAP MLX.

## Environment

Requirements:

- Apple Silicon macOS for real MLX runtime execution;
- Python 3.12 or 3.13;
- `uv`.

Install dependencies:

```bash
uv sync --group dev
```

## Core Checks

Run the focused test suite:

```bash
uv run python -m pytest -q tests/test_mlx_*.py
```

The routers mirror each mlx-lm release's routing, so router or observer changes
should also pass against the other supported mlx-lm line, for example:

```bash
uv run --with "mlx-lm==0.32.2" python -m pytest -q tests/test_mlx_*.py
```

Run CLI help:

```bash
uv run python -m reap.entrypoint --help
```

Lint (ruff with the `E`, `F`, `W`, `I`, `SIM` rules at 88 columns):

```bash
uv run ruff check src tests
```

Check lockfile and whitespace:

```bash
uv lock --check
git diff --check
```

CI runs the lockfile check, lint, tests, and CLI smoke on Python 3.12 (the
required `tests` check), plus the test suite on Python 3.13.

## Import-Safety Tests

Import safety is tested with subprocess import blockers. The suite verifies that
module imports do not import heavy runtime packages such as:

- `torch`
- `vllm`
- `mlx`
- `mlx_lm`
- `datasets`

When adding code, keep imports lazy unless the dependency is already safe at
module import time. If a runtime package is needed, import it inside the
function that executes the runtime behavior.

## Test Layout

| Test file | Coverage |
| --- | --- |
| `test_mlx_no_torch_import.py` | Package root import safety. |
| `test_mlx_cli.py` | CLI argument validation, orchestration, progress, failure metrics. |
| `test_mlx_data.py` | Calibration loading, text extraction, tokenizer handling. |
| `test_mlx_model_adapters.py` | Adapter inference, layer config, config updates. |
| `test_mlx_router.py` | Qwen, LFM2, and Mixtral/PhiMoE router semantics for mlx-lm 0.31 and 0.32. |
| `test_mlx_metrics.py` | `PruningState` accumulation and reports. |
| `test_mlx_observer.py` | Layer replay, masks, selected metrics, LFM2 operators. |
| `test_mlx_prune.py` | Expert ranking, slicing, metadata/config updates, failures. |
| `test_mlx_save.py` | Save/reload validation, artifact checks, smoke generation. |
| `test_mlx_validation_metrics.py` | Telemetry JSON, timings, throughput, failure payload. |
| `test_mlx_checkpoint.py` | Checkpoint schema, round trips, atomic writes. |
| `test_mlx_eval_nll.py` | Calibration NLL across fp32/fp16/bf16 logits. |
| `test_mlx_e2e.py` | Tiny random mlx-lm models per family (bf16, 4-bit, and on mlx-lm 0.32 nvfp4 with per-expert global scales): CLI prune/save/reload and bit-exact replay parity. |

Some tests require MLX or MLX-LM and skip when those packages are unavailable.

## Adding A Pruning Method

To add a pruning method:

1. Ensure the observer report includes the required per-expert one-dimensional
   array.
2. Add the method key to `_SUPPORTED_PRUNE_METHODS`.
3. Add aliases to `_PRUNE_METHOD_ALIASES` if needed.
4. Add tests for method resolution and pruning with that saliency key.
5. Update [Pruning](pruning.md) and README method tables.

The pruning method must rank higher values as more important.

## Adding A Model Adapter

For a new model family:

1. Implement adapter layout methods and `MoeLayerConfig` extraction.
2. Add router behavior matching the architecture's MLX-LM routing semantics in
   every supported mlx-lm release.
3. Decide whether existing slicing rules cover the family.
4. Extend save/reload validation for any new expert-stacked fields.
5. Add fixture-style unit tests that do not require real model downloads, and
   add the family to `tests/test_mlx_e2e.py` (tiny random mlx-lm model in bf16
   and 4-bit); fixtures alone miss quantized-gate and config-key bugs.
6. Add the model type to `SUPPORTED_MODEL_TYPES`.
7. Preserve import-safety tests.

Use the existing Qwen3 and LFM2 adapters as the reference pattern.

## Real Model Runs

Unit tests use small fixtures. Before merging risky runtime changes, run at
least one small real pruning job on Apple Silicon:

```bash
MAX_SAMPLES=8 MAX_SEQ_LENGTH=1024 uv run \
  bash experiments/mlx-pruning.sh \
  LiquidAI/LFM2.5-8B-A1B-MLX-4bit \
  theblackcat102/evol-codealpaca-v1 \
  reap \
  0.25 \
  42
```

Do not commit outputs under `artifacts/`.

## Documentation Updates

When behavior changes, update the matching technical doc:

- pipeline or CLI behavior: `pipeline.md`, `cli.md`;
- adapter or router behavior: `model-adapters.md`,
  `observation-and-metrics.md`;
- saliency or slicing behavior: `pruning.md`;
- save/reload behavior: `save-reload-validation.md`;
- metrics schema: `telemetry.md`;
- development workflow: `development.md`.

