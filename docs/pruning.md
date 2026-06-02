# Pruning

Pruning is implemented in `reap.prune`. It ranks experts per MoE layer using
observer data, slices expert-stacked model fields in place, and updates runtime
metadata plus config values.

## Entry Point

```python
prune_experts(
    model,
    config,
    observer_data,
    prune_method,
    compression_ratio,
    *,
    adapter=None,
) -> dict[int, np.ndarray]
```

Returns ascending retained expert indices for each pruned MoE layer.

## Supported Methods

| Method | Description |
| --- | --- |
| `reap` | `weighted_ean_sum / expert_frequency`, with zero-frequency experts reporting zero. |
| `expert_frequency` | Count of selected router assignments. |
| `frequency` | Alias for `expert_frequency`. |
| `weighted_expert_frequency_sum` | Sum of selected router scores. |
| `weighted_frequency_sum` | Alias for `weighted_expert_frequency_sum`. |
| `ean_sum` | Sum of selected expert output norms. |
| `ean_mean` | Mean selected expert output norm. |
| `weighted_ean_sum` | Router-score-weighted sum of selected output norms. |
| `max_activations` | Maximum selected expert output activation. |

Higher scores are kept.

## Compression Ratio

`compression_ratio` must be in `[0, 1)`.

For each MoE layer:

```txt
num_to_prune = int(num_experts * compression_ratio)
retained_count = max(num_experts - num_to_prune, 1)
```

This means pruning is floor-based. For example, with 3 experts and ratio
`1 / 3`, `int(3 * 1 / 3) == 1`, so 2 experts are retained.

## Expert Ranking

`compute_keep_indices` validates that saliency is one-dimensional, contains no
NaN values, and that `retained_count` is in `[1, num_experts]`.

Ranking uses:

```python
np.lexsort((expert_ids, -saliency))
```

This keeps highest saliency first and breaks ties by lower expert id. The final
returned keep indices are sorted in ascending order so slicing preserves the
original expert order among retained experts.

## Qwen3 Slicing

Qwen3 pruning requires:

```txt
moe.switch_mlp.gate_proj.weight
moe.switch_mlp.up_proj.weight
moe.switch_mlp.down_proj.weight
moe.gate.weight
```

For switch projections, any present fields are sliced:

- `weight`
- `scales`
- `biases`
- `bias`

For the gate, any present fields are sliced:

- `weight`
- `bias`
- `e_score_correction_bias`

Each sliced value must expose a shape and have first dimension equal to the
pre-pruning expert count.

## LFM2 Slicing

LFM2 pruning slices the same switch projection fields:

- `weight`
- `scales`
- `biases`
- `bias`

For the LFM2 gate, any present fields are sliced:

- `weight`
- `scales`
- `biases`
- `bias`

If `use_expert_bias` is enabled or `expert_bias` exists, `expert_bias` is also
sliced on dimension 0. If expert bias is enabled but missing, pruning raises a
`ValueError`.

## Runtime Metadata Updates

After slicing, pruning updates live module attributes:

- `moe.num_experts`
- any existing top-k attrs among `top_k`, `num_experts_per_tok`, `k`
- gate attrs `num_experts` and `n_routed_experts` when present
- `gate.top_k` when present

New top-k is:

```txt
min(old_top_k, retained_count)
```

## Config Updates

After all MoE layers are pruned, config is updated with:

- `config["num_experts"] = retained_count`
- `config["num_experts_per_tok"] = min(top_k, retained_count)`
- `config["top_k"]` only if that key already exists

All pruned MoE layers must retain the same expert count. If different layers
would retain different counts, pruning raises a `ValueError` because the current
config format records one global `num_experts` value.

## Validation Failures

Pruning raises before mutation or during mutation for:

- unsupported adapter names;
- invalid compression ratios;
- missing observer data for an adapter-visible MoE layer;
- missing observer saliency key;
- saliency length mismatch;
- NaN saliency values;
- missing required switch projections or gate weights;
- sliceable fields whose first dimension does not match `num_experts`;
- LFM2 expert bias enabled but missing.

Because pruning mutates in place, callers should treat failures during pruning
as invalidating the live model object for save purposes.

