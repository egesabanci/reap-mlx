# CLI

The command-line entry point is:

```bash
uv run python -m reap.entrypoint
```

It orchestrates model loading, calibration, observation, pruning, save/reload
validation, optional smoke generation, and telemetry writing.

## Required Options

| Option | Description |
| --- | --- |
| `--model-name` | MLX-LM model path or Hugging Face repository id. |
| `--dataset-name` | Hugging Face dataset name for calibration. |
| `--output-dir` | Destination directory for the pruned MLX-LM artifact. |

## Optional Options

| Option | Default | Description |
| --- | --- | --- |
| `--split` | `train` | Dataset split passed to `load_dataset`. |
| `--dataset-config-name` | unset | Optional Hugging Face dataset config name. |
| `--prune-method` | `reap` | Saliency key or alias used to rank experts. |
| `--compression-ratio` | `0.25` | Fraction of experts to remove per MoE layer. |
| `--max-samples` | `128` | Maximum non-empty calibration samples. |
| `--num-calibration-sequences` | alias | Alias for `--max-samples`. |
| `--max-seq-length` | `2048` | Maximum tokens per calibration sequence. |
| `--eval-frequency` | `1` | Evaluate the MLX graph every N observation layers. |
| `--seed` | `42` | Shuffle seed when the dataset supports shuffle. |
| `--metrics-file` | `validation-metrics.json` | Metrics filename or absolute path. |
| `--verbose` | off | Print phase progress messages. |
| `--no-smoke` | off | Skip generation smoke after reload validation. |
| `--smoke-prompt` | `What is your name?` | Prompt for generation smoke validation. |
| `--smoke-max-tokens` | `16` | Maximum tokens for generation smoke validation. |

## Validation

Argument validation happens before model or dataset loading.

The CLI rejects:

- `compression_ratio` outside `[0, 1)`;
- non-finite compression ratios such as `nan`;
- unsupported prune methods;
- `max_samples < 1`;
- `max_seq_length < 1`;
- `eval_frequency < 1`;
- `smoke_max_tokens < 1`.

Invalid arguments exit through `argparse` with code 2.

## Direct Example

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

Expected output directory:

```txt
artifacts/mlx/lfm2-smoke/
  config.json
  *.safetensors or *.npz
  tokenizer files
  validation-metrics.json
```

## Experiment Wrapper

`experiments/mlx-pruning.sh` provides a shell wrapper:

```bash
MAX_SAMPLES=8 MAX_SEQ_LENGTH=1024 uv run \
  bash experiments/mlx-pruning.sh \
  LiquidAI/LFM2.5-8B-A1B-MLX-4bit \
  theblackcat102/evol-codealpaca-v1 \
  reap \
  0.25 \
  42
```

Positional arguments:

```txt
1. MODEL_NAME
2. DATASET
3. PRUNE_METHOD
4. COMPRESSION_RATIO
5. SEED
```

Environment variables:

| Variable | Default |
| --- | --- |
| `MAX_SAMPLES` | `8` |
| `MAX_SEQ_LENGTH` | `1024` |
| `OUTPUT_DIR` | `artifacts/mlx/$(basename "$MODEL")/${METHOD}-${RATIO}-seed-${SEED}` |

## Progress Messages

With normal execution, the CLI emits phase messages prefixed with `[reap-mlx]`:

```txt
load: loading MLX-LM model
calibrate: loading calibration sequences
observe: collecting pruning metrics
prune: mutating selected experts
save: saving pruned model and validating reload
reload/smoke: validation complete
done: saved MLX pruned model to ...
```

The messages are emitted regardless of `--verbose`; `--verbose` controls Python
logging level.

## Failure Behavior

`KeyboardInterrupt` returns exit code 130 and prints an interrupted message.

For other exceptions, the CLI writes failed telemetry when possible and then
re-raises the exception. The failure payload includes:

- pipeline phase;
- exception type;
- exception message;
- elapsed seconds before failure;
- memory sample at failure.
