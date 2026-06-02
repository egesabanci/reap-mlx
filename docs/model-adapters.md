# Model Adapters

Model adapters isolate MLX-LM architecture layout differences from the rest of
the pruning pipeline.

The adapter API is intentionally informal: adapters are regular Python objects
with methods called by observer, pruning, validation, and telemetry code. The
current adapters are `Qwen3MoeModelAdapter` and `Lfm2MoeModelAdapter`.

## Shared Layer Discovery

`get_model_layers(model)` checks common MLX-LM object layouts in this order:

1. `model.model.layers`
2. `model.layers`
3. `model.model.model.layers`

If none exists, it raises a `ValueError`.

## MoE Layer Config

Adapters return `MoeLayerConfig`:

```python
MoeLayerConfig(
    num_experts: int,
    top_k: int,
    norm_topk_prob: bool,
    adapter_name: str,
    use_expert_bias: bool = False,
)
```

Live module attributes are preferred over config values. This matters after
pruning, because the live model may already have updated `num_experts` and
top-k attributes.

Config conventions supported by adapters:

| Concept | Live attrs | Config keys |
| --- | --- | --- |
| Expert count | `num_experts` | `num_experts` |
| Top-k | `top_k`, `num_experts_per_tok` | `num_experts_per_tok`, `top_k` |
| Top-k normalization | `norm_topk_prob` | `norm_topk_prob` |
| LFM2 expert bias | `use_expert_bias` | `use_expert_bias` |

## Qwen3-MoE Adapter

Adapter name: `qwen3_moe`

Expected layer shape:

```txt
layer.mlp
  gate
  switch_mlp
    gate_proj
    up_proj
    down_proj
```

A layer is considered MoE when `layer.mlp.switch_mlp` exists. Dense layers still
use `layer.mlp` for the dense MLP path.

The adapter updates config through `update_qwen3_moe_config`:

- set `config["num_experts"]`;
- set `config["num_experts_per_tok"]`;
- update `config["top_k"]` only when that key already exists;
- clamp top-k to at most the retained expert count.

## LFM2-MoE Adapter

Adapter name: `lfm2_moe`

Expected layer shape:

```txt
layer.feed_forward
  gate
  switch_mlp
    gate_proj
    up_proj
    down_proj
  expert_bias  # optional, required when use_expert_bias is true
```

A layer is considered MoE when `layer.feed_forward.switch_mlp` exists. Dense
layers use `layer.feed_forward` for the dense path.

LFM2 layers can be attention or conv/SSM operator layers. The observer uses:

- `layer.self_attn` for attention layers;
- `layer.conv` for non-attention layers;
- `layer.is_attention_layer` when present to choose between the two.

The LFM2 config update currently delegates to the Qwen3 update logic and
preserves unrelated keys such as `use_expert_bias`.

## Adapter Inference

`infer_model_adapter(model, config)` selects LFM2 when:

- `config["model_type"] == "lfm2_moe"`;
- any configured architecture starts with `"Lfm2"`;
- the model layout includes any `layer.feed_forward.switch_mlp`.

Otherwise it returns the Qwen3 adapter.

## Shared Experts

`get_shared_expert(moe)` supports both:

- `moe.shared_experts`
- `moe.shared_expert`

Plural is preferred when both are present. Shared expert output participates in
the hidden-state flow but is not included in selected-expert saliency metrics.

## Adding A Model Family

To add a new MoE family:

1. Add an adapter with the same method surface:
   - `adapter_name`
   - `layers(model)`
   - `identify_moe_layers(model)`
   - `is_moe_layer(layer)`
   - `get_moe(layer)`
   - `get_dense_mlp(layer)`
   - `get_layer_config(layer, config)`
2. Add or extend router logic for the architecture's selected-route semantics.
3. Add pruning support if the expert-stacked field layout differs from Qwen3 or
   LFM2.
4. Update save/reload shape validation if new expert-stacked fields must be
   validated.
5. Add fixture-style tests for adapter inference, routing, pruning, observer
   replay, and reload validation.

Keep any new runtime imports lazy and add import-safety coverage.

