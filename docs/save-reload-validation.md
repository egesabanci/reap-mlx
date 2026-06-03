# Save Reload Validation

Saving and validation are implemented in `reap.save`. A pruning run is marked
successful only after the mutated model is saved, reloaded, and shape-validated.

## Entry Point

```python
save_pruned_model(
    model,
    tokenizer,
    config,
    output_dir,
    original_model_name,
    *,
    adapter=None,
    expected_expert_count=None,
    smoke_fn=None,
    smoke_prompt="What is your name?",
    smoke_max_tokens=16,
    load_fn=None,
    save_fn=None,
) -> SaveReloadResult
```

`model` and `config` are the already-pruned live objects.

## SaveReloadResult

```python
SaveReloadResult(
    output_dir: Path,
    reloaded_model: Any,
    reloaded_tokenizer: Any,
    reloaded_config: Mapping[str, Any],
    expected_expert_count: int,
    smoke_result: Any = None,
    metrics: Mapping[str, Any] | None = None,
)
```

The result metrics include save/reload timings, artifact summary, and smoke
metadata.

## Save Behavior

The default save function is imported lazily:

```python
from mlx_lm import utils
utils.save(...)
```

It is called with:

```python
save_fn(
    dst_path=str(output_path),
    src_path_or_repo=str(original_model_name),
    model=model,
    tokenizer=tokenizer,
    config=config,
)
```

The helper uses the passed config mapping. It must not rely on `model.config`.

`output_dir` is created if needed. If it already exists as a file, saving raises
`OSError`.

## Required Saved Artifacts

After save, the output directory must contain:

- `config.json`
- at least one weight artifact matching `*.safetensors` or `*.npz`

The artifact summary includes files matching:

- `*.safetensors`
- `*.npz`
- `*.json`
- `*.model`
- `*.txt`

## Reload Behavior

The default reload function is imported lazily:

```python
from mlx_lm import load
```

If `load_fn` supports `return_config`, it is called as:

```python
load_fn(str(output_dir), return_config=True)
```

and must return:

```python
(model, tokenizer, config)
```

If `return_config` is not supported, it is called as:

```python
load_fn(str(output_dir))
```

and must return:

```python
(model, tokenizer)
```

In that fallback path, `config.json` is read directly.

## Expected Expert Count

Expected expert count comes from:

1. explicit `expected_expert_count`, when provided;
2. `config["num_experts"]`.

The value must be positive. If both are missing, validation raises.

## Reloaded Config Validation

Reloaded config must include `num_experts`, and it must match the expected
expert count.

This catches save paths that failed to write the pruned config.

## Reloaded Shape Validation

The adapter is used to find MoE layers in the reloaded model. At least one
adapter-visible MoE layer must exist.

For every MoE layer, validation checks:

- adapter-reported `layer_config.num_experts` equals expected expert count;
- `switch_mlp.gate_proj.weight` first dimension equals expected expert count;
- `switch_mlp.up_proj.weight` first dimension equals expected expert count;
- `switch_mlp.down_proj.weight` first dimension equals expected expert count;
- `gate.weight` first dimension equals expected expert count.

For LFM2, `expert_bias` is also validated when `use_expert_bias` is true or the
attribute exists.

## Smoke Generation

`generation_smoke` runs a short generation with these default CLI settings:

```txt
prompt: "What is your name?"
max_tokens: 16
```

Use `--smoke-prompt` and `--smoke-max-tokens` to override those values for a
run. The configured prompt and token limit are recorded in smoke metrics.

If the tokenizer has a chat template and `apply_chat_template`, the prompt is
wrapped as a user message with `add_generation_prompt=True`.

The default generator is imported lazily:

```python
from mlx_lm import generate
```

Smoke runs on the reloaded model and reloaded tokenizer. `--no-smoke` disables
smoke by passing `smoke_fn=None`.

## Common Failure Meanings

| Failure | Likely cause |
| --- | --- |
| Missing `config.json` | Save function did not write a complete MLX-LM artifact. |
| Missing weight artifact | Save function wrote config/tokenizer files but no model weights. |
| Reload config expert mismatch | Mutated config was not saved or wrong artifact reloaded. |
| First-dimension mismatch | Expert-stacked weights were not sliced consistently. |
| LFM2 expert bias mismatch | `expert_bias` was not pruned or reloaded with stale shape. |
