"""Layerwise observer for MLX-backed MoE models.

The observer uses explicit MLX layer replay. It records only selected expert
outputs for pruning metrics and avoids importing optional MLX runtime packages
until observation is executed.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Callable

from reap.metrics import PruningState
from reap.model_adapters import (
    get_shared_expert,
    infer_model_adapter,
    make_attention_mask,
    make_ssm_mask,
)
from reap.router import Lfm2MoeRouter, Qwen3MoeRouter


logger = logging.getLogger(__name__)


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
) -> dict[int, dict[str, Any]]:
    """Collect pruning-compatible observer data with explicit MLX layer replay."""
    mx = _require_mlx_core()
    config = {} if config is None else config
    adapter = infer_model_adapter(model, config) if adapter is None else adapter
    if adapter is None:
        raise ValueError(
            "Could not determine the MoE architecture adapter for this model. "
            "observe_model requires an MoE model (Qwen3-MoE or LFM2-MoE) with at "
            "least one MoE layer."
        )
    eval_frequency = _validate_eval_frequency(eval_frequency)
    eval_fn = mx.eval if eval_fn is None else eval_fn

    if getattr(adapter, "adapter_name", None) == "lfm2_moe":
        return _observe_lfm2_model(
            model,
            calibration_sequences,
            config,
            adapter=adapter,
            debug_memory=debug_memory,
            eval_frequency=eval_frequency,
            eval_fn=eval_fn,
            mask_fn=mask_fn,
        )

    return _observe_qwen3_model(
        model,
        calibration_sequences,
        config,
        adapter=adapter,
        debug_memory=debug_memory,
        eval_frequency=eval_frequency,
        eval_fn=eval_fn,
        mask_fn=mask_fn,
    )


def _observe_qwen3_model(
    model: Any,
    calibration_sequences: list[Any],
    config: Mapping[str, Any],
    *,
    adapter: Any,
    debug_memory: bool,
    eval_frequency: int,
    eval_fn: Callable[[Any], Any],
    mask_fn: Callable[..., Any] | None,
) -> dict[int, dict[str, Any]]:
    mx = _require_mlx_core()
    layers = adapter.layers(model)
    moe_layer_indices = set(adapter.identify_moe_layers(model))
    embed_tokens = _get_embed_tokens(model)

    accumulators = _initialize_accumulators(
        layers,
        moe_layer_indices,
        adapter=adapter,
        config=config,
    )

    for sequence in calibration_sequences:
        tokens = _batch_tokens(mx, sequence)
        h = embed_tokens(tokens)
        default_mask = (
            _attention_mask(
                h,
                sequence_length=tokens.shape[-1],
                mask_fn=None,
            )
            if mask_fn is None
            else None
        )

        for layer_idx, layer in enumerate(layers):
            mask = (
                default_mask
                if mask_fn is None
                else _attention_mask(
                    h,
                    sequence_length=tokens.shape[-1],
                    mask_fn=mask_fn,
                )
            )
            h = _run_attention(layer, h, mask)
            moe_input = _call_required(layer, "post_attention_layernorm", h)

            if layer_idx in moe_layer_indices:
                h = h + _observe_selected_moe_layer(
                    layer,
                    moe_input,
                    accumulators[layer_idx],
                    adapter=adapter,
                    config=config,
                    router_cls=Qwen3MoeRouter,
                )
            else:
                dense_mlp = adapter.get_dense_mlp(layer)
                h = h + dense_mlp(moe_input)

            if _should_eval_layer(layer_idx, len(layers), eval_frequency):
                try:
                    eval_fn(h)
                except (RuntimeError, MemoryError) as eval_err:
                    logger.warning(
                        "eval_fn failed at layer %d/%d (eval_frequency=%d): %s. "
                        "Falling back to eval_frequency=1 for remaining layers.",
                        layer_idx, len(layers), eval_frequency, eval_err,
                    )
                    eval_frequency = 1
                    try:
                        eval_fn(h)
                    except (RuntimeError, MemoryError):
                        logger.error(
                            "eval_fn retry also failed at layer %d. Raising.",
                            layer_idx,
                        )
                        raise
            if debug_memory:
                _log_memory(mx, layer_idx)

    return _report_with_layer_guard(accumulators)


def _observe_lfm2_model(
    model: Any,
    calibration_sequences: list[Any],
    config: Mapping[str, Any],
    *,
    adapter: Any,
    debug_memory: bool,
    eval_frequency: int,
    eval_fn: Callable[[Any], Any],
    mask_fn: Callable[..., Any] | None,
) -> dict[int, dict[str, Any]]:
    mx = _require_mlx_core()
    layers = adapter.layers(model)
    moe_layer_indices = set(adapter.identify_moe_layers(model))
    embed_tokens = _get_embed_tokens(model)

    accumulators = _initialize_accumulators(
        layers,
        moe_layer_indices,
        adapter=adapter,
        config=config,
    )

    for sequence in calibration_sequences:
        tokens = _batch_tokens(mx, sequence)
        h = embed_tokens(tokens)
        attn_mask, conv_mask = _lfm2_masks(
            h,
            sequence_length=tokens.shape[-1],
            mask_fn=mask_fn,
        )

        for layer_idx, layer in enumerate(layers):
            operator_mask = attn_mask if _is_lfm2_attention_layer(layer) else conv_mask
            h_mid = _run_lfm2_operator(layer, h, operator_mask)
            ffn_input = _call_required(layer, "ffn_norm", h_mid)

            if layer_idx in moe_layer_indices:
                h = h_mid + _observe_selected_moe_layer(
                    layer,
                    ffn_input,
                    accumulators[layer_idx],
                    adapter=adapter,
                    config=config,
                    router_cls=Lfm2MoeRouter,
                )
            else:
                dense_mlp = adapter.get_dense_mlp(layer)
                h = h_mid + dense_mlp(ffn_input)

            if _should_eval_layer(layer_idx, len(layers), eval_frequency):
                try:
                    eval_fn(h)
                except (RuntimeError, MemoryError) as eval_err:
                    logger.warning(
                        "eval_fn failed at layer %d/%d (eval_frequency=%d): %s. "
                        "Falling back to eval_frequency=1 for remaining layers.",
                        layer_idx, len(layers), eval_frequency, eval_err,
                    )
                    eval_frequency = 1
                    try:
                        eval_fn(h)
                    except (RuntimeError, MemoryError):
                        logger.error(
                            "eval_fn retry also failed at layer %d. Raising.",
                            layer_idx,
                        )
                        raise
            if debug_memory:
                _log_memory(mx, layer_idx)

    return _report_with_layer_guard(accumulators)


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


def _run_attention(layer: Any, h: Any, mask: Any | None) -> Any:
    normalized = _call_required(layer, "input_layernorm", h)
    self_attn = getattr(layer, "self_attn", None)
    if not callable(self_attn):
        raise ValueError("Layer does not expose a callable self_attn module.")

    attention_output = self_attn(normalized, mask, cache=None)
    if isinstance(attention_output, tuple):
        attention_output = attention_output[0]
    return h + attention_output


def _is_lfm2_attention_layer(layer: Any) -> bool:
    is_attention_layer = getattr(layer, "is_attention_layer", None)
    if is_attention_layer is not None:
        return bool(is_attention_layer)
    return callable(getattr(layer, "self_attn", None))


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


def _observe_selected_moe_layer(
    layer: Any,
    moe_input: Any,
    state: PruningState,
    *,
    adapter: Any,
    config: Mapping[str, Any],
    router_cls: type,
) -> Any:
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
    state.accumulate(
        indices=routing.indices,
        scores=saliency_scores.astype(mx.float32),
        selected_outputs=selected_outputs.astype(mx.float32),
    )

    moe_out = (selected_outputs * routing.scores[..., None]).sum(axis=-2)
    shared_expert = get_shared_expert(moe)
    if shared_expert is not None:
        moe_out = moe_out + shared_expert(moe_input)
    return moe_out


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
