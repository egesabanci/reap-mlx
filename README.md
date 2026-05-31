# REAP MLX

Apple Silicon / MLX implementation of Router-weighted Expert Activation
Pruning (REAP) for MLX-LM MoE models.

This repository is now focused on MLX-LM. The original CUDA/PyTorch
experiment scripts, plotting assets, Docker setup, paper reproduction
materials, and eager Torch source have been removed.

## Current Scope

The working path is:

```text
MLX-LM model -> calibration data -> selected-expert observation -> REAP pruning
-> save -> reload -> generation smoke -> validation metrics
```

Implemented MLX architecture adapters:

- `qwen3_moe`
- `lfm2_moe`, validated with `LiquidAI/LFM2.5-8B-A1B-MLX-4bit`

The implementation lives under:

```text
src/reap/
```

It is intentionally import-light: importing `reap` must not import
Torch, vLLM, MLX, or MLX-LM unless an execution function explicitly needs them.

## Setup

Create a local virtual environment and install the MLX runtime dependencies:

```bash
uv sync --group dev
```

Run commands with `PYTHONPATH=src` while the command-line interface remains
module-local.

## Run MLX Pruning

Small LFM2.5 smoke run:

```bash
PYTHONPATH=src MAX_SAMPLES=1 MAX_SEQ_LENGTH=64 \
  OUTPUT_DIR=artifacts/mlx/lfm2-e2e-smoke \
  bash experiments/mlx-pruning.sh \
  LiquidAI/LFM2.5-8B-A1B-MLX-4bit \
  theblackcat102/evol-codealpaca-v1 \
  reap \
  0.25 \
  42
```

The run writes a structured metrics artifact:

```text
<OUTPUT_DIR>/validation-metrics.json
```

That file records model metadata, runtime versions, calibration statistics,
timings, throughput, MLX memory samples, observer summaries, pruning decisions,
save/reload validation, artifact sizes, and smoke-generation results.

## Tests

Focused MLX test suite:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_mlx_no_torch_import.py \
  tests/test_mlx_router.py \
  tests/test_mlx_metrics.py \
  tests/test_mlx_model_adapters.py \
  tests/test_mlx_observer.py \
  tests/test_mlx_prune.py \
  tests/test_mlx_save.py \
  tests/test_mlx_data.py \
  tests/test_mlx_cli.py \
  tests/test_mlx_validation_metrics.py
```

## Notes

- `artifacts/` is ignored and is the intended location for local pruned models,
  validation metrics, and scratch chat helpers.
- The license file is retained until the replacement license/notice is decided.
