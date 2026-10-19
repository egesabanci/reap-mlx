# Evaluating pruned models

REAP MLX validates structure (reload + optional generation smoke) and can
optionally compare **next-token NLL** before and after pruning:

```bash
uv run python -m reap.entrypoint \
  --model-name ... \
  --dataset-name ... \
  --output-dir artifacts/mlx/run \
  --eval-calibration-nll \
  --eval-calibration-sequences 8
```

With `--eval-calibration-nll`, the loader fetches `--eval-calibration-sequences`
extra calibration sequences and holds them out of observation, so the NLL is not
measured on the data that chose the experts. A baseline NLL is computed on the
unpruned model right before pruning, and the same sequences are scored again
on the reloaded pruned model. The `quality` block reports:

- `mean_nll`, `perplexity`, `token_count`, `sequence_count` for the pruned model;
- `baseline` with the same fields for the unpruned model;
- `delta_mean_nll` (pruned minus baseline; lower is better);
- `evaluation_set`: `held_out`, or `calibration` when the dataset had no records
  left to hold out and the observed sequences were reused (in-sample).

Resumed runs skip calibration, so they report `status: "skipped"`.

The NLL signal is a quick regression check, not a substitute for task benchmarks.

Logits are converted to float32 before the log-softmax, so bfloat16/float16
checkpoints (most MLX conversions) get a precise value. If no sequence can be
evaluated, the `quality` block reports `status: "skipped"` together with the
last error that occurred.

## Third-party eval stacks

The `third-party/` directory holds reference harnesses (EvalPlus, LiveCodeBench,
HELM, EvalScope, etc.). They are **not** installed as package dependencies and
are **not** run by CI. To evaluate a pruned artifact:

1. Save a pruned model to `output_dir` with REAP MLX.
2. Serve or load that path with your preferred stack (vLLM, mlx_lm, transformers).
3. Point the harness at the served endpoint or local path.

Example with mlx_lm generation smoke only:

```bash
uv run python -c "
from mlx_lm import load, generate
model, tok = load('artifacts/mlx/run')
print(generate(model, tok, prompt='Hello', max_tokens=32))
"
```

For code benchmarks, follow the upstream README inside each `third-party/*`
project after installing its own environment.
