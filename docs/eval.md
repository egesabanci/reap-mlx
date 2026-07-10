# Evaluating pruned models

REAP MLX validates structure (reload + optional generation smoke) and can
optionally report **calibration mean NLL** after reload:

```bash
uv run python -m reap.entrypoint \
  --model-name ... \
  --dataset-name ... \
  --output-dir artifacts/mlx/run \
  --eval-calibration-nll \
  --eval-calibration-sequences 8
```

The NLL signal is a quick regression check, not a substitute for task benchmarks.

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
