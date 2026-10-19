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
    eval_frequency=1,
    eval_fn=None,
    mask_fn=None,
    print_fn=None,
    verify_replay=True,
) -> dict[int, dict[str, Any]]
```

Returns observer reports keyed by MoE layer index. Dense layers are replayed but
do not receive reports.

`eval_fn` defaults to `mx.eval`. By default, it is called after every layer to
force MLX evaluation boundaries and avoid unbounded lazy graph growth across
layers and sequences.

`eval_frequency` controls that boundary. Values greater than 1 evaluate every N
layers and also flush the final layer of each sequence when needed. Higher
values reduce GPU synchronization barriers but increase peak memory.

Selected-expert statistics do not add synchronization points of their own:
each MoE layer reduces its routed outputs on device to per-route float32 norms
and maxes (plus per-token shared-expert norms), and those small arrays are
evaluated and accumulated at the next eval boundary. The accumulated
statistics are identical for every `eval_frequency`.

Out-of-memory errors are recognised in both forms they take: `MemoryError`
and the `RuntimeError` MLX raises for allocation failures (for example
`[metal::malloc] Attempting to allocate ... greater than the maximum allowed
buffer size`). Other `RuntimeError` failures are re-raised so real graph bugs
are not masked as memory pressure.

An out-of-memory error from `eval_fn` clears the MLX buffer cache and falls back
to `eval_frequency=1` for the rest of the current sequence.

An out-of-memory error anywhere else in a sequence rolls observer state back to
the snapshot taken before that sequence started (so MoE layers never disagree
on how many sequences they saw), clears the buffer cache, stops observation,
and keeps the fully completed sequences. Every layer report carries
`observed_sequences`; the CLI fails the run when it is below
`--min-calibration-samples` unless `--allow-partial-calibration` is passed.

A Metal command-buffer failure ("Insufficient Memory") can abort the process
inside MLX before Python sees an exception; lower `--max-seq-length` if that
happens.

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
   - evaluate the hidden state when the configured eval boundary is reached.

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
   - evaluate the hidden state when the configured eval boundary is reached.

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

`indices` and `scores` are in `argpartition` partition order. The top-k experts
are present, but they are not guaranteed to be sorted by descending score.

## Qwen3 Router Semantics

Qwen3 routing:

1. Accepts hidden states shaped `[tokens, hidden]` or `[batch, seq, hidden]`.
2. Flattens leading dimensions before gate projection.
3. Computes gate logits with `moe.gate`.
4. Applies `mx.softmax(logits, axis=-1, precise=True)`.
5. Selects top-k experts through `mx.argpartition`, using the same form as the
   mlx-lm architecture: ascending (`argpartition(gates, kth=-k)[..., -k:]`) for
   `qwen3_moe`, descending (`argpartition(-gates, kth=k-1)[..., :k]`) for
   `qwen2_moe` and `olmoe`. The two forms keep different experts on exact ties
   at the top-k boundary, which bfloat16 gates produce regularly.
6. Gathers selected softmax scores.
7. Optionally renormalizes selected scores with the shared epsilon denominator
   when `norm_topk_prob` is enabled.
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
8. Optionally renormalizes selected scores with the shared epsilon denominator.
9. Casts scores back to the hidden-state dtype.

When `use_expert_bias` is true, `moe.expert_bias` is required.

LFM2 returns two score tensors:

- `scores`: bias-adjusted values used to weight expert outputs in the residual
  (matches mlx-lm forward).
- `saliency_scores`: **pure softmax** probabilities at the selected indices
  (bias-free), used by `PruningState.accumulate` for `weighted_ean_sum` and
  `reap`.

This keeps REAP ranking aligned with routing probability rather than additive
expert bias, while the residual path still matches the live model.

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

## Replay Parity Check

The replay only produces correct saliency if the routers reproduce the live
model's routing exactly. On the first calibration sequence, every MoE layer's
replayed output is therefore also compared with the live MoE block's own
forward pass (`moe(moe_input)`). The error is the largest per-token L2
difference divided by the mean per-token L2 norm of the live output. Supported
architectures replay bit-exactly (error `0.0`); if the error exceeds
`REPLAY_PARITY_TOLERANCE` (`1e-2`), observation raises before the remaining
sequences are processed. That typically means the installed mlx-lm changed how
the architecture routes.

The measured error is reported per layer as `replay_parity_error`. Pass
`verify_replay=False` (CLI: `--skip-replay-check`) to disable the check. MoE
blocks that are not callable (test fixtures) are not compared.

## PruningState

`PruningState` is a NumPy accumulator for one MoE layer.

Tracked fields:

| Field | Shape | Meaning |
| --- | --- | --- |
| `total_tokens` | scalar | Number of routed token positions, not multiplied by top-k. |
| `expert_frequency` | `[num_experts]` | Count of selected router assignments per expert. |
| `ean_sum` | `[num_experts]` | Sum of selected expert output L2 norms. |
| `weighted_ean_sum` | `[num_experts]` | Sum of output norms multiplied by router score. |
| `weighted_expert_frequency_sum` | `[num_experts]` | Sum of selected router scores. |
| `max_activations` | `[num_experts]` | Maximum selected expert output activation. |

Optional debug tracking can be enabled with
`PruningState.initialize(num_experts, track_pairwise=True)`. This allocates and
reports `pairwise_expert_frequency` with shape `[num_experts, num_experts]`.
Default observation leaves it disabled to avoid O(num_experts²) hot-path work.

`accumulate` can accept a `RouterResult` or explicit `indices` and `scores`.
Callers must also provide either selected expert outputs or precomputed selected
output norms and maxes.

## Observer Report Keys

`PruningState.report()` returns:

| Key | Meaning |
| --- | --- |
| `total_tokens` | Routed token positions. |
| `expert_frequency` | Selected-route counts. |
| `expert_proba` | `expert_frequency / max(total_slots, 1)` where `total_slots` counts every selected (token, expert) slot (`total_tokens * top_k` accumulation). |
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

`pairwise_expert_frequency` is not currently a supported pruning method and is
disabled by default.
