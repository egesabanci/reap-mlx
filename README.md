<div align="center">

# REAP MLX

**Apple Silicon REAP expert pruning for MLX-LM MoE models.**

[Quick Start](#quick-start) |
[Workflow](#workflow) |
[Supported Models](#supported-models) |
[CLI Reference](#cli-reference) |
[Metrics](#metrics) |
[Technical Docs](#technical-docs) |
[References](#references) |
[Development](#development) |
[License](#license)

![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue)
![Runtime](https://img.shields.io/badge/Runtime-MLX-black)
![Package](https://img.shields.io/badge/Package-reap--mlx-green)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

REAP MLX applies Router-weighted Expert Activation Pruning (REAP) to
MLX-LM mixture-of-experts models on Apple Silicon. It loads an MLX-LM model,
collects selected-expert activation statistics on calibration data, prunes
low-saliency experts in place, saves the pruned model, reloads it, and writes
validation telemetry for the run.

Use it when you want local MoE pruning experiments on Apple Silicon without a
CUDA, PyTorch, vLLM, plotting, or paper-reproduction stack.

## Highlights

- **MLX-only runtime**: the execution path is built around `mlx` and `mlx-lm`.
- **Adapter-based architecture support**: model-family differences are isolated
  in small adapters under `src/reap/model_adapters.py`.
- **Calibration-driven pruning**: observation records selected routes, router
  scores, output norms, weighted activation metrics, and max activations.
- **Save/reload validation**: runs through `mlx_lm.save`, reloads the artifact,
  validates expert counts, and can run a generation smoke test.
- **Structured telemetry**: each run writes `validation-metrics.json` with
  model metadata, timings, memory samples, throughput, pruning decisions, and
  artifact sizes.
- **Import-light package**: importing `reap` must not import MLX, MLX-LM,
  datasets, Torch, or vLLM.

## Quick Start

Prerequisites:

- Apple Silicon macOS environment
- Python 3.12 or 3.13
- `uv`
- An MLX-LM compatible MoE model, local path, or Hugging Face repo

Install the development environment:

```bash
git clone git@github.com:egesabanci/reap-mlx.git
cd reap-mlx
uv sync --group dev
```

Check the CLI:

```bash
uv run python -m reap.entrypoint --help
```

Run the focused test suite:

```bash
uv run python -m pytest -q tests/test_mlx_*.py
```

## Run Pruning

The experiment wrapper uses `LiquidAI/LFM2.5-8B-A1B-MLX-4bit` by default and
writes artifacts under `artifacts/mlx/`.

```bash
MAX_SAMPLES=8 MAX_SEQ_LENGTH=1024 uv run \
  bash experiments/mlx-pruning.sh \
  LiquidAI/LFM2.5-8B-A1B-MLX-4bit \
  theblackcat102/evol-codealpaca-v1 \
  reap \
  0.25 \
  42
```

Equivalent direct CLI call:

```bash
uv run python -m reap.entrypoint \
  --model-name LiquidAI/LFM2.5-8B-A1B-MLX-4bit \
  --dataset-name theblackcat102/evol-codealpaca-v1 \
  --prune-method reap \
  --compression-ratio 0.25 \
  --max-samples 8 \
  --max-seq-length 1024 \
  --seed 42 \
  --output-dir artifacts/mlx/lfm2-smoke \
  --verbose
```

The output directory contains the saved MLX-LM artifact and a metrics file:

```txt
artifacts/mlx/lfm2-smoke/
  config.json
  *.safetensors or *.npz
  tokenizer files
  validation-metrics.json
```

## Workflow

The pruning pipeline is intentionally linear and inspectable.

| Step | What happens | Evidence |
| --- | --- | --- |
| 1. Load | Load model, tokenizer, and config through `mlx_lm.load(..., return_config=True)`. | model metadata |
| 2. Calibrate | Load and tokenize unpadded batch-size-1 calibration sequences. | sample and token counts |
| 3. Observe | Replay layers and collect selected expert route statistics. | per-layer observer metrics |
| 4. Prune | Keep highest-saliency experts and slice expert-stacked arrays. | retained expert ids |
| 5. Save | Save the mutated model and updated config with `mlx_lm.save`. | artifact paths and sizes |
| 6. Validate | Reload the saved model and optionally run a generation smoke test. | reload and smoke status |

## Supported Models

| Adapter | Model family | Status |
| --- | --- | --- |
| `lfm2_moe` | Liquid LFM2.5 MoE MLX-LM models | Validated with `LiquidAI/LFM2.5-8B-A1B-MLX-4bit` |
| `qwen3_moe` | Qwen3-MoE MLX-LM-style models | Adapter and unit coverage present |

Adding a model family should usually mean adding or extending an adapter in
`src/reap/model_adapters.py`, router logic in `src/reap/router.py`, and pruning
coverage in `tests/test_mlx_model_adapters.py`, `tests/test_mlx_router.py`, and
`tests/test_mlx_prune.py`.

## Pruning Methods

`--prune-method` selects the per-expert saliency score used to rank experts.
Higher scores are kept.

| Method | Meaning |
| --- | --- |
| `reap` | Weighted expert activation norm divided by expert frequency. |
| `expert_frequency` | Count of selected router assignments. |
| `frequency` | Alias for `expert_frequency`. |
| `weighted_expert_frequency_sum` | Sum of selected router scores. |
| `weighted_frequency_sum` | Alias for `weighted_expert_frequency_sum`. |
| `ean_sum` | Sum of selected expert output norms. |
| `ean_mean` | Mean selected expert output norm. |
| `weighted_ean_sum` | Router-score-weighted sum of selected expert output norms. |
| `max_activations` | Maximum selected expert output activation. |

`--compression-ratio` must be in `[0, 1)`. For each MoE layer, REAP MLX prunes
`int(num_experts * compression_ratio)` experts and always keeps at least one
expert.

## CLI Reference

Run:

```bash
uv run python -m reap.entrypoint --help
```

Core options:

| Option | Default | Description |
| --- | --- | --- |
| `--model-name` | required | MLX-LM model path or Hugging Face repo id. |
| `--dataset-name` | required | Hugging Face dataset name for calibration. |
| `--split` | `train` | Dataset split. |
| `--dataset-config-name` | unset | Optional dataset config name. |
| `--prune-method` | `reap` | Saliency metric used for expert ranking. |
| `--compression-ratio` | `0.25` | Fraction of experts to remove per MoE layer. |
| `--max-samples` | `128` | Number of non-empty calibration samples. |
| `--max-seq-length` | `2048` | Maximum token length per calibration sample. |
| `--seed` | `42` | Dataset shuffle seed when shuffle is available. |
| `--output-dir` | required | Directory for the pruned MLX-LM artifact. |
| `--metrics-file` | `validation-metrics.json` | Metrics filename or absolute path. |
| `--verbose` | off | Print pipeline progress. |
| `--no-smoke` | off | Skip generation smoke after save/reload validation. |

## Metrics

Every run writes structured validation telemetry. The JSON includes:

- run configuration and runtime versions;
- model architecture, adapter, expert count, top-k, and quantization metadata;
- calibration sample counts, token counts, and token throughput;
- per-phase timings for load, calibration, observe, prune, save, reload, and
  smoke validation;
- MLX memory samples when available;
- observer summaries and saliency inputs;
- per-layer retained and removed experts;
- output artifact sizes;
- reload validation and generation smoke results.

Use this file to compare pruning settings, calibration size, runtime cost, and
artifact size across runs.

## Technical Docs

Maintainer-focused reference documentation is available in
[docs/index.md](docs/index.md). It covers the pipeline, architecture, model
adapters, calibration, observation metrics, pruning semantics, save/reload
validation, CLI behavior, telemetry, and development workflow.

## References

REAP MLX is an independent MLX-focused implementation. Some implementation
decisions are inspired by the original
[CerebrasResearch/reap](https://github.com/CerebrasResearch/reap) repository.

The pruning method is based on the paper
[REAP the Experts: Why Pruning Prevails for One-Shot MoE compression](https://arxiv.org/pdf/2510.13999).

## Repository Layout

```txt
reap-mlx/
  README.md
  LICENSE
  pyproject.toml
  uv.lock
  experiments/
    mlx-pruning.sh
  docs/
    index.md
    *.md
  src/
    reap/
      data.py
      entrypoint.py
      metrics.py
      model_adapters.py
      observer.py
      prune.py
      router.py
      save.py
      validation_metrics.py
  tests/
    test_mlx_*.py
```

## Development

Install and sync dependencies:

```bash
uv sync --group dev
```

Run tests:

```bash
uv run python -m pytest -q tests/test_mlx_*.py
```

Run import-safety and CLI smoke checks:

```bash
uv run python -c "import reap; import reap.entrypoint"
uv run python -m reap.entrypoint --help
uv lock --check
```

Check formatting hygiene before committing:

```bash
git diff --check
```

## License

REAP MLX is released under the MIT License. See [LICENSE](LICENSE).
