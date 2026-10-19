"""Layerwise observer for MLX-backed MoE models.

The observer uses explicit MLX layer replay. It records only selected expert
outputs for pruning metrics and avoids importing optional MLX runtime packages
until observation is executed.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from reap.metrics import PruningState
from reap.model_adapters import (
    get_shared_expert,
    infer_model_adapter,
    make_attention_mask,
    make_ssm_mask,
    unsupported_architecture_message,
)
from reap.router import Lfm2MoeRouter, MixtralMoeRouter, Qwen3MoeRouter


logger = logging.getLogger(__name__)

# On the first calibration sequence every replayed MoE output is also compared
# with the live MoE block's own forward pass. Supported architectures replay
# bit-exactly; routing drift (for example an mlx-lm release that changes the
# gating) sends whole tokens to other experts and shows up as errors of order
# 0.1-1, far above this tolerance.
REPLAY_PARITY_TOLERANCE = 1e-2

# MLX reports allocation failures as RuntimeError, for example
# "[metal::malloc] Attempting to allocate ... greater than the maximum allowed
# buffer size", not as MemoryError. Lower-cased substrings that identify them.
_OOM_MESSAGE_MARKERS = (
    "malloc",
    "out of memory",
    "insufficient memory",
    "outofmemory",
)


def observe_model(
    model: Any,
    calibration_sequences: list[Any],
    config: Mapping[str, Any] | None = None,
    *,
    adapter: Any | None = None,
    debug_memory: bool = False,
    eval_frequency: int = 1,
    eval_fn: Callable[[Any], Any] | None = None,
    mask_fn: Callable[..., Any] | None = None,
    print_fn: Callable[[str], Any] | None = None,
    verify_replay: bool = True,
) -> dict[int, dict[str, Any]]:
    """Collect pruning-compatible observer data with explicit MLX layer replay.

    With ``verify_replay`` the first sequence is also run through each live
    MoE block, and observation fails if the replay drifts from it by more than
    ``REPLAY_PARITY_TOLERANCE``. Each layer report then carries the measured
    ``replay_parity_error``.
    """
    mx = _require_mlx_core()
    config = {} if config is None else config
    adapter = infer_model_adapter(model, config) if adapter is None else adapter
    if adapter is None:
        raise ValueError(unsupported_architecture_message(config))
    eval_frequency = _validate_eval_frequency(eval_frequency)
    eval_fn = mx.eval if eval_fn is None else eval_fn

    if getattr(adapter, "adapter_name", None) == "lfm2_moe":
        layout = _Lfm2Layout(mask_fn)
    else:
        # qwen3_moe and mixtral_moe share the standard attention+mlp residual layout.
        layout = _StandardLayout(mask_fn)
    return _observe_layers(
        model,
        calibration_sequences,
        config,
        adapter=adapter,
        layout=layout,
        debug_memory=debug_memory,
        eval_frequency=eval_frequency,
        eval_fn=eval_fn,
        print_fn=print_fn,
        verify_replay=verify_replay,
    )


class _StandardLayout:
    """Pre-norm self-attention then MoE/MLP (Qwen family, Mixtral, PhiMoE)."""

    def __init__(self, mask_fn: Callable[..., Any] | None) -> None:
        self.mask_fn = mask_fn

    def sequence_masks(self, hidden_states: Any) -> Any | None:
        # Build the causal mask once per sequence and reuse it across layers.
        # This assumes a static mask that depends only on sequence length, which
        # holds for the supported decoder-only architectures. An injected
        # mask_fn is called per layer instead.
        if self.mask_fn is not None:
            return None
        return _attention_mask(
            hidden_states,
            sequence_length=hidden_states.shape[1],
            mask_fn=None,
        )

    def mix(self, layer: Any, hidden_states: Any, sequence_masks: Any) -> tuple[Any, Any]:
        """Apply the token mixer; return (residual stream, MoE/MLP input)."""
        mask = (
            sequence_masks
            if self.mask_fn is None
            else _attention_mask(
                hidden_states,
                sequence_length=hidden_states.shape[1],
                mask_fn=self.mask_fn,
            )
        )
        normalized = _call_required(layer, "input_layernorm", hidden_states)
        hidden_states = hidden_states + _run_attention(layer, normalized, mask)
        return hidden_states, _call_required(
            layer, "post_attention_layernorm", hidden_states
        )


class _Lfm2Layout:
    """LFM2: attention or short-conv operator, then the feed-forward block."""

    def __init__(self, mask_fn: Callable[..., Any] | None) -> None:
        self.mask_fn = mask_fn

    def sequence_masks(self, hidden_states: Any) -> tuple[Any | None, Any | None]:
        return _lfm2_masks(
            hidden_states,
            sequence_length=hidden_states.shape[1],
            mask_fn=self.mask_fn,
        )

    def mix(self, layer: Any, hidden_states: Any, sequence_masks: Any) -> tuple[Any, Any]:
        """Apply the token mixer; return (residual stream, MoE/MLP input)."""
        attn_mask, conv_mask = sequence_masks
        operator_mask = attn_mask if _is_lfm2_attention_layer(layer) else conv_mask
        hidden_states = _run_lfm2_operator(layer, hidden_states, operator_mask)
        return hidden_states, _call_required(layer, "ffn_norm", hidden_states)


def _observe_layers(
    model: Any,
    calibration_sequences: list[Any],
    config: Mapping[str, Any],
    *,
    adapter: Any,
    layout: _StandardLayout | _Lfm2Layout,
    debug_memory: bool,
    eval_frequency: int,
    eval_fn: Callable[[Any], Any],
    print_fn: Callable[[str], Any] | None,
    verify_replay: bool,
) -> dict[int, dict[str, Any]]:
    requested_eval_frequency = eval_frequency
    mx = _require_mlx_core()
    layers = adapter.layers(model)
    moe_layer_indices = set(adapter.identify_moe_layers(model))
    embed_tokens = _get_embed_tokens(model)
    router_cls = _router_cls_for_adapter(adapter)
    parity_errors: dict[int, float] = {}

    accumulators = _initialize_accumulators(
        layers,
        moe_layer_indices,
        adapter=adapter,
        config=config,
    )

    completed_sequences = 0
    for seq_idx, sequence in enumerate(calibration_sequences):
        # Snapshot before the sequence so a mid-sequence MemoryError cannot leave
        # earlier MoE layers with extra tokens relative to later layers.
        snapshots = {
            layer_idx: state.snapshot() for layer_idx, state in accumulators.items()
        }
        try:
            # Reset to the requested cadence for each sequence so a transient
            # eval failure in one sequence does not permanently downgrade the
            # whole run to eval_frequency=1.
            eval_frequency = requested_eval_frequency
            tokens = _batch_tokens(mx, sequence)
            h = embed_tokens(tokens)
            sequence_masks = layout.sequence_masks(h)
            # Per-layer statistics stay lazy until the next eval boundary, so
            # --eval-frequency N really means one graph sync per N layers.
            pending: list[_PendingLayerStats] = []

            for layer_idx, layer in enumerate(layers):
                h, moe_input = layout.mix(layer, h, sequence_masks)

                if layer_idx in moe_layer_indices:
                    moe_out, stats = _observe_selected_moe_layer(
                        layer,
                        moe_input,
                        accumulators[layer_idx],
                        adapter=adapter,
                        config=config,
                        router_cls=router_cls,
                    )
                    pending.append(stats)
                    if verify_replay and seq_idx == 0:
                        error = _replay_parity_error(
                            adapter.get_moe(layer), moe_input, moe_out
                        )
                        if error is not None:
                            _check_replay_parity(error, layer_idx=layer_idx, config=config)
                            parity_errors[layer_idx] = error
                    h = h + moe_out
                else:
                    dense_mlp = adapter.get_dense_mlp(layer)
                    h = h + dense_mlp(moe_input)

                if _should_eval_layer(layer_idx, len(layers), eval_frequency):
                    eval_frequency = _run_eval_boundary(
                        eval_fn,
                        h,
                        layer_idx=layer_idx,
                        layer_count=len(layers),
                        eval_frequency=eval_frequency,
                        print_fn=print_fn,
                    )
                    _flush_layer_stats(mx, pending)
                    pending = []
                if debug_memory:
                    _log_memory(mx, layer_idx)
            completed_sequences += 1
        except (MemoryError, RuntimeError) as exc:
            if not _is_out_of_memory(exc):
                raise
            for layer_idx, snap in snapshots.items():
                accumulators[layer_idx].restore(snap)
            _release_cached_memory(mx)
            _emit_oom_salvage(
                print_fn,
                seq_idx=seq_idx,
                total_sequences=len(calibration_sequences),
                completed_sequences=completed_sequences,
            )
            break

    observer_data = _report_with_layer_guard(accumulators)
    for layer_idx, layer_data in observer_data.items():
        layer_data["observed_sequences"] = completed_sequences
        if layer_idx in parity_errors:
            layer_data["replay_parity_error"] = parity_errors[layer_idx]
    return observer_data


def _is_out_of_memory(exc: BaseException) -> bool:
    """Return True for MemoryError and MLX allocation-failure RuntimeErrors."""
    if isinstance(exc, MemoryError):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return any(marker in message for marker in _OOM_MESSAGE_MARKERS)


def _release_cached_memory(mx: Any) -> None:
    """Drop MLX's buffer cache so a retry after an allocation failure can fit."""
    clear_cache = getattr(mx, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()


def _replay_parity_error(moe: Any, moe_input: Any, replay_output: Any) -> float | None:
    """Return how far the replayed MoE output is from the live block's output.

    The error is the largest per-token L2 difference divided by the mean
    per-token L2 norm of the live output. Returns None when the MoE block is
    not callable (fixture objects), since there is nothing to compare against.
    """
    if not callable(moe):
        return None
    mx = _require_mlx_core()
    reference = moe(moe_input).astype(mx.float32)
    hidden_size = reference.shape[-1]
    reference = reference.reshape(-1, hidden_size)
    delta = replay_output.astype(mx.float32).reshape(-1, hidden_size) - reference
    token_error = mx.sqrt(mx.sum(delta * delta, axis=-1))
    token_norm = mx.sqrt(mx.sum(reference * reference, axis=-1))
    error = mx.max(token_error) / mx.maximum(mx.mean(token_norm), 1e-12)
    return float(error.item())


def _check_replay_parity(
    error: float,
    *,
    layer_idx: int,
    config: Mapping[str, Any],
) -> None:
    if error <= REPLAY_PARITY_TOLERANCE:
        return
    raise ValueError(
        f"Replay parity check failed for MoE layer {layer_idx} "
        f"(model_type={config.get('model_type')!r}): the replayed MoE output "
        f"differs from the model's own forward pass by {error:.3g} (tolerance "
        f"{REPLAY_PARITY_TOLERANCE:g}, relative to the mean token norm). The "
        "installed mlx-lm probably routes this architecture differently from "
        "reap-mlx, so expert saliency would be measured on the wrong routes. "
        "Pass --skip-replay-check (observe_model(verify_replay=False)) to "
        "override."
    )


def _router_cls_for_adapter(adapter: Any) -> type:
    name = getattr(adapter, "adapter_name", None)
    if name == "lfm2_moe":
        return Lfm2MoeRouter
    if name == "mixtral_moe":
        return MixtralMoeRouter
    return Qwen3MoeRouter


def _require_mlx_core():
    try:
        import mlx.core as mx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "observe_model requires the optional 'mlx' package to execute. "
            "Install MLX in the active environment before observing."
        ) from exc
    return mx


def _validate_eval_frequency(eval_frequency: int) -> int:
    value = int(eval_frequency)
    if value < 1:
        raise ValueError(
            f"eval_frequency must be a positive integer, got {eval_frequency}."
        )
    return value


def _should_eval_layer(layer_idx: int, layer_count: int, eval_frequency: int) -> bool:
    return (layer_idx + 1) % eval_frequency == 0 or layer_idx == layer_count - 1


def _get_embed_tokens(model: Any) -> Callable[[Any], Any]:
    model_body = getattr(model, "model", None)
    embed_tokens = getattr(model_body, "embed_tokens", None)
    if callable(embed_tokens):
        return embed_tokens

    embed_tokens = getattr(model, "embed_tokens", None)
    if callable(embed_tokens):
        return embed_tokens

    raise ValueError(
        "Model does not expose an embed_tokens callable at "
        "model.model.embed_tokens or model.embed_tokens."
    )


def _batch_tokens(mx: Any, sequence: Any) -> Any:
    input_ids = sequence.get("input_ids") if isinstance(sequence, Mapping) else sequence
    tokens = mx.array(input_ids)

    if tokens.ndim == 0:
        raise ValueError("Calibration sequences must contain at least one token.")
    if tokens.ndim == 1:
        if tokens.shape[0] == 0:
            raise ValueError("Calibration sequences must contain at least one token.")
        return tokens[None, :]
    if tokens.ndim == 2:
        if tokens.shape[0] != 1:
            raise ValueError(
                "MLX observer only supports unpadded batch-size-1 sequences."
            )
        if tokens.shape[1] == 0:
            raise ValueError("Calibration sequences must contain at least one token.")
        return tokens

    raise ValueError(
        "Calibration sequences must have shape [seq] or [1, seq], "
        f"got {tokens.shape}."
    )


def _report_with_layer_guard(accumulators: dict[int, PruningState]) -> dict[int, dict]:
    """Build observer data, surfacing NaN/Inf with the offending layer index."""
    observer_data: dict[int, dict] = {}
    for layer_idx, state in accumulators.items():
        try:
            observer_data[layer_idx] = state.report()
        except ValueError as exc:
            raise ValueError(f"Layer {layer_idx}: {exc}") from exc
    return observer_data


def _initialize_accumulators(
    layers: Any,
    moe_layer_indices: set[int],
    *,
    adapter: Any,
    config: Mapping[str, Any],
) -> dict[int, PruningState]:
    accumulators: dict[int, PruningState] = {}
    for layer_idx in sorted(moe_layer_indices):
        layer_config = adapter.get_layer_config(layers[layer_idx], config)
        accumulators[layer_idx] = PruningState.initialize(layer_config.num_experts)
    return accumulators


def _attention_mask(
    hidden_states: Any,
    *,
    sequence_length: int,
    mask_fn: Callable[..., Any] | None,
) -> Any | None:
    if mask_fn is not None:
        return mask_fn(hidden_states, cache=None)
    if sequence_length == 1:
        return None
    return make_attention_mask(hidden_states, cache=None)


def _lfm2_masks(
    hidden_states: Any,
    *,
    sequence_length: int,
    mask_fn: Callable[..., Any] | None,
) -> tuple[Any | None, Any | None]:
    if sequence_length == 1 and mask_fn is None:
        return None, None
    if mask_fn is not None:
        return (
            _call_mask_fn(mask_fn, hidden_states, kind="attention"),
            _call_mask_fn(mask_fn, hidden_states, kind="ssm"),
        )
    return (
        make_attention_mask(hidden_states, cache=None),
        make_ssm_mask(hidden_states, cache=None),
    )


def _call_mask_fn(
    mask_fn: Callable[..., Any],
    hidden_states: Any,
    *,
    kind: str,
) -> Any:
    try:
        return mask_fn(hidden_states, cache=None, kind=kind)
    except TypeError:
        return mask_fn(hidden_states, cache=None)


def _run_attention(layer: Any, normalized: Any, mask: Any | None) -> Any:
    """Run self-attention on pre-normalized hidden states, returning residual."""
    self_attn = getattr(layer, "self_attn", None)
    if not callable(self_attn):
        raise ValueError("Layer does not expose a callable self_attn module.")
    attention_output = self_attn(normalized, mask, cache=None)
    if isinstance(attention_output, tuple):
        attention_output = attention_output[0]
    return attention_output


def _is_lfm2_attention_layer(layer: Any) -> bool:
    is_attention_layer = getattr(layer, "is_attention_layer", None)
    if is_attention_layer is not None:
        return bool(is_attention_layer)
    # Without an explicit flag, fall back to checking self_attn. If the layer
    # has both self_attn and conv (hybrid), this heuristic is ambiguous --
    # raise a clear error instead of silently picking the wrong operator.
    has_attn = callable(getattr(layer, "self_attn", None))
    has_conv = callable(getattr(layer, "conv", None))
    if has_attn and has_conv:
        raise ValueError(
            "Layer has both self_attn and conv modules but no "
            "is_attention_layer flag. Cannot determine operator type.",
        )
    return has_attn


def _run_lfm2_operator(layer: Any, h: Any, mask: Any | None) -> Any:
    normalized = _call_required(layer, "operator_norm", h)
    if _is_lfm2_attention_layer(layer):
        operator = getattr(layer, "self_attn", None)
        if not callable(operator):
            raise ValueError(
                "LFM2 attention layer does not expose a callable self_attn module."
            )
        operator_output = operator(normalized, mask=mask, cache=None)
    else:
        operator = getattr(layer, "conv", None)
        if not callable(operator):
            raise ValueError("LFM2 conv layer does not expose a callable conv module.")
        operator_output = operator(normalized, mask=mask, cache=None)

    if isinstance(operator_output, tuple):
        operator_output = operator_output[0]
    return h + operator_output


@dataclass
class _PendingLayerStats:
    """One MoE layer's selected-route statistics, waiting for an eval boundary.

    All arrays are float32/uint32 reductions shaped like the routed indices
    (``[..., top_k]``, or ``[..., ]`` per token for the shared expert), so
    evaluating them moves only a few values per token to NumPy.
    """

    state: PruningState
    indices: Any
    scores: Any
    output_norms: Any
    output_maxes: Any
    shared_output_norms: Any | None = None

    def arrays(self) -> list[Any]:
        arrays = [self.indices, self.scores, self.output_norms, self.output_maxes]
        if self.shared_output_norms is not None:
            arrays.append(self.shared_output_norms)
        return arrays

    def accumulate(self) -> None:
        self.state.accumulate(
            indices=self.indices,
            scores=self.scores,
            selected_output_norms=self.output_norms,
            selected_output_maxes=self.output_maxes,
        )
        if self.shared_output_norms is not None:
            self.state.accumulate_shared_expert_norms(self.shared_output_norms)


def _flush_layer_stats(mx: Any, pending: list[_PendingLayerStats]) -> None:
    """Evaluate queued layer statistics in one call and accumulate them."""
    if not pending:
        return
    mx.eval(*(array for stats in pending for array in stats.arrays()))
    for stats in pending:
        stats.accumulate()


def _token_l2_norms(mx: Any, values: Any) -> Any:
    return mx.sqrt(mx.sum(values * values, axis=-1))


def _observe_selected_moe_layer(
    layer: Any,
    moe_input: Any,
    state: PruningState,
    *,
    adapter: Any,
    config: Mapping[str, Any],
    router_cls: type,
) -> tuple[Any, _PendingLayerStats]:
    """Replay one MoE layer; return its output and its pending statistics."""
    mx = _require_mlx_core()
    moe = adapter.get_moe(layer)
    routing = router_cls(moe, config)(moe_input)
    switch_mlp = getattr(moe, "switch_mlp", None)
    if not callable(switch_mlp):
        raise ValueError("MoE layer does not expose a callable switch_mlp module.")

    selected_outputs = switch_mlp(moe_input, routing.indices)
    # Validate switch_mlp output shape early so a mismatch produces a
    # descriptive REAP error instead of an opaque downstream MLX shape error.
    expected_shape = (*routing.indices.shape, moe_input.shape[-1])
    if tuple(selected_outputs.shape) != tuple(expected_shape):
        raise ValueError(
            "switch_mlp returned shape "
            f"{tuple(selected_outputs.shape)}, expected {tuple(expected_shape)} "
            "(indices shape + hidden dim) for this MoE layer.",
        )
    saliency_scores = routing.saliency_scores
    if saliency_scores is None:
        saliency_scores = routing.scores
    # Metrics path: reduce on device in float32 (NumPy cannot read bfloat16)
    # down to per-route norms and maxes. Residual path: keep the original
    # selected_outputs / scores dtypes so the replay matches mlx-lm forward
    # numerics for later layers.
    outputs_for_metrics = selected_outputs.astype(mx.float32)
    stats = _PendingLayerStats(
        state=state,
        indices=routing.indices,
        scores=saliency_scores.astype(mx.float32),
        output_norms=_token_l2_norms(mx, outputs_for_metrics),
        output_maxes=mx.max(mx.abs(outputs_for_metrics), axis=-1),
    )

    moe_out = (selected_outputs * routing.scores[..., None]).sum(axis=-2)
    shared_expert = get_shared_expert(moe)
    if shared_expert is not None:
        shared_out = shared_expert(moe_input)
        # Optional gated shared experts (e.g. Qwen2-MoE).
        shared_gate = getattr(moe, "shared_expert_gate", None)
        if callable(shared_gate):
            shared_out = mx.sigmoid(shared_gate(moe_input)) * shared_out
        stats.shared_output_norms = _token_l2_norms(
            mx, shared_out.astype(mx.float32)
        )
        moe_out = moe_out + shared_out
    return moe_out, stats


def _run_eval_boundary(
    eval_fn: Callable[[Any], Any],
    h: Any,
    *,
    layer_idx: int,
    layer_count: int,
    eval_frequency: int,
    print_fn: Callable[[str], Any] | None,
) -> int:
    """Evaluate the graph at a layer boundary.

    Only out-of-memory errors (``MemoryError`` or MLX allocation-failure
    ``RuntimeError``) trigger a fallback to ``eval_frequency=1``. Other errors
    are re-raised so real graph bugs are not masked as memory pressure.
    """
    try:
        eval_fn(h)
        return eval_frequency
    except (MemoryError, RuntimeError) as eval_err:
        if not _is_out_of_memory(eval_err):
            raise
        logger.warning(
            "eval_fn out of memory (%s) at layer %d/%d (eval_frequency=%d): %s. "
            "Falling back to eval_frequency=1 for remaining layers.",
            type(eval_err).__name__,
            layer_idx,
            layer_count,
            eval_frequency,
            eval_err,
        )
        if print_fn is not None:
            print_fn(
                f"[reap-mlx] observe: eval_fn out of memory "
                f"({type(eval_err).__name__}) at layer {layer_idx}/{layer_count}; "
                "falling back to eval_frequency=1 for this sequence.",
            )
        _release_cached_memory(_require_mlx_core())
        try:
            eval_fn(h)
        except (MemoryError, RuntimeError) as retry_err:
            if _is_out_of_memory(retry_err):
                logger.error(
                    "eval_fn retry also ran out of memory at layer %d. Raising.",
                    layer_idx,
                )
            raise
        return 1


def _emit_oom_salvage(
    print_fn: Callable[[str], Any] | None,
    *,
    seq_idx: int,
    total_sequences: int,
    completed_sequences: int,
) -> None:
    logger.warning(
        "Out of memory while processing sequence %d/%d. "
        "Discarding partial progress for this sequence and stopping. "
        "Completed sequences retained: %d.",
        seq_idx + 1,
        total_sequences,
        completed_sequences,
    )
    if print_fn is not None:
        print_fn(
            f"[reap-mlx] observe: out of memory at sequence "
            f"{seq_idx + 1}/{total_sequences}; "
            f"salvaging {completed_sequences} fully completed sequence(s) "
            "(partial sequence rolled back)."
        )


def _call_required(layer: Any, attr: str, h: Any) -> Any:
    module = getattr(layer, attr, None)
    if not callable(module):
        raise ValueError(f"Layer does not expose a callable {attr} module.")
    return module(h)


def _log_memory(mx: Any, layer_idx: int) -> None:
    memory_parts = []
    for name in ("get_active_memory", "get_peak_memory", "get_cache_memory"):
        getter = getattr(mx, name, None)
        if callable(getter):
            memory_parts.append(f"{name}={getter()}")
    if memory_parts:
        logger.debug("MLX memory after layer %s: %s", layer_idx, ", ".join(memory_parts))


__all__ = ["observe_model"]
