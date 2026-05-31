# REAP MLX

Apple Silicon / MLX implementation of Router-weighted Expert Activation
Pruning (REAP) for MLX-LM MoE models.

This repository is now focused on the MLX backend. The original CUDA/PyTorch
experiment scripts, plotting assets, Docker setup, and paper reproduction
materials have been removed or are being retired in follow-up cleanup work.

## Current Scope

The working path is:

```text
MLX-LM model -> calibration data -> selected-expert observation -> REAP pruning
-> save -> reload -> generation smoke -> validation metrics
```

Implemented MLX architecture adapters:

- `qwen3_moe`
- `lfm2_moe`, validated with `LiquidAI/LFM2.5-8B-A1B-MLX-4bit`

The MLX backend lives under:

```text
src/reap/backends/mlx/
```

It is intentionally import-light: importing `reap.backends.mlx` must not import
Torch, vLLM, MLX, or MLX-LM unless an execution function explicitly needs them.

## Setup

Create a local virtual environment and install the MLX runtime dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install mlx mlx-lm datasets huggingface-hub safetensors transformers pytest
```

Run commands with `PYTHONPATH=src` until packaging metadata is simplified for
the MLX-only project.

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
- The repository still contains some original Torch/CUDA source during the
  cleanup transition. That code is no longer the target path and will be removed
  in a follow-up MLX-only cleanup PR.
- The license file is retained until the replacement license/notice is decided.
