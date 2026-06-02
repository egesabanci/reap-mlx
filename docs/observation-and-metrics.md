# Observation And Metrics

Observation is the calibration-time phase that measures selected expert usage
and activation saliency. It is implemented by `reap.observer.observe_model` and
`reap.metrics.PruningState`.

## Observer Contract

```python
observe_model(
    model,
    calibration_sequences,
    config=None,
    *,
    adapter=None,
    debug_memory=False,
    eval_fn=None,
    mask_fn=None,
) -> dict[int, dict[str, Any]]
```

Returns observer reports keyed by MoE layer index. Dense layers are replayed but
do not receive reports.

`eval_fn` defaults to `mx.eval`. It is called after every layer to force MLX
evaluation boundaries and avoid unbounded lazy graph growth across layers and
sequences.

## Qwen3 Replay

For each sequence:

1. Batch token IDs.
2. Embed tokens.
3. For every layer:
   - build an attention mask unless sequence length is 1;
   - run `input_layernorm`;
   - run `self_attn(normalized, mask, cache=None)`;
   - add the attention residual;
   - run `post_attention_layernorm`;
   - run either MoE observation or dense MLP;
   - add the MLP residual;
   - evaluate the hidden state.

Qwen3 MoE layers route with `Qwen3MoeRouter`.

## LFM2 Replay

For each sequence:

1. Batch token IDs.
2. Embed tokens.
3. Build attention and SSM masks once for the sequence unless sequence length is
   1 and no custom mask function is provided.
4. For every layer:
   - run `operator_norm`;
   - run `self_attn(..., mask=attn_mask, cache=None)` for attention layers;
   - run `conv(..., mask=conv_mask, cache=None)` for non-attention layers;
   - add the operator residual;
   - run `ffn_norm`;
   - run either MoE observation or dense feed-forward;
   - add the feed-forward residual;
   - evaluate the hidden state.

LFM2 MoE layers route with `Lfm2MoeRouter`.

## Router Results

Routers return:

```python
RouterResult(
    indices,
    scores,
    logits=None,
    score_mode="actual",
)
```

`indices` and `scores` share shape:

```txt
[..., top_k]
```

The leading dimensions match the input hidden states without the hidden
dimension.

## Qwen3 Router Semantics

Qwen3 routing:

1. Accepts hidden states shaped `[tokens, hidden]` or `[batch, seq, hidden]`.
2. Flattens leading dimensions before gate projection.
3. Computes gate logits with `moe.gate`.
4. Applies `mx.softmax(logits, axis=-1, precise=True)`.
5. Selects top-k experts through `mx.argpartition`.
6. Gathers selected softmax scores.
7. Optionally renormalizes selected scores when `norm_topk_prob` is enabled.
8. Reshapes selected indices and scores back to leading dimensions.

Live `top_k` on the module is preferred over config top-k.

## LFM2 Router Semantics

LFM2 routing:

1. Accepts hidden states shaped `[tokens, hidden]` or `[batch, seq, hidden]`.
2. Computes gate logits with `moe.gate`.
3. Casts logits to `mx.float32`.
4. Applies `mx.softmax(logits, axis=-1)`.
5. Adds `expert_bias` before top-k selection when `use_expert_bias` is enabled.
6. Selects top-k experts through `mx.argpartition`.
7. Gathers selected scores.
8. Optionally renormalizes selected scores with an epsilon denominator.
9. Casts scores back to the hidden-state dtype.

When `use_expert_bias` is true, `moe.expert_bias` is required.

## Selected Expert Outputs

For a routed MoE layer, observation calls:

```python
selected_outputs = switch_mlp(moe_input, routing.indices)
```

The expected shape is:

```txt
[..., top_k, hidden]
```

The observer computes the MoE output as:

```python
(selected_outputs * routing.scores[..., None]).sum(axis=-2)
```

If a shared expert exists, its output is added to the hidden-state flow. Shared
expert output is not included in selected-expert saliency accumulation.

## PruningState

`PruningState` is a NumPy accumulator for one MoE layer.

Tracked fields:

| Field | Shape | Meaning |
| --- | --- | --- |
| `total_tokens` | scalar | Number of routed token positions, not multiplied by top-k. |
| `expert_frequency` | `[num_experts]` | Count of selected router assignments per expert. |
| `pairwise_expert_frequency` | `[num_experts, num_experts]` | Batch-level pairwise frequency summary. |
| `ean_sum` | `[num_experts]` | Sum of selected expert output L2 norms. |
| `weighted_ean_sum` | `[num_experts]` | Sum of output norms multiplied by router score. |
| `weighted_expert_frequency_sum` | `[num_experts]` | Sum of selected router scores. |
| `max_activations` | `[num_experts]` | Maximum selected expert output activation. |

`accumulate` can accept a `RouterResult` or explicit `indices` and `scores`.
Callers must also provide either selected expert outputs or precomputed selected
output norms and maxes.

## Observer Report Keys

`PruningState.report()` returns:

| Key | Meaning |
| --- | --- |
| `total_tokens` | Routed token positions. |
| `expert_frequency` | Selected-route counts. |
| `pairwise_expert_frequency` | Batch pairwise frequency matrix. |
| `expert_proba` | `expert_frequency / max(total_tokens, 1)`. |
| `ean_sum` | Sum of selected output norms. |
| `ean_mean` | `ean_sum / max(expert_frequency, eps)`. |
| `weighted_ean_sum` | Router-score-weighted output norm sum. |
| `weighted_expert_frequency_sum` | Selected router score sum. |
| `reap` | `weighted_ean_sum / max(expert_frequency, eps)`. |
| `max_activations` | Max selected output activation. |

Never-selected experts report finite zeros for mean-style metrics.

## Metric Caveats

`expert_frequency.sum()` can exceed `total_tokens` when `top_k > 1`, because
frequency counts selected routes and each token can select multiple experts.

`pairwise_expert_frequency` is collected for telemetry and future analysis. It
is not currently a supported pruning method.

