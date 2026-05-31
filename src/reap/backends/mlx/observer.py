"""Layerwise observer for MLX-backed MoE models.

The MLX observer uses explicit layer replay instead of PyTorch hooks. It records
only selected expert outputs for pruning metrics and avoids importing optional
MLX runtime packages until observation is executed.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Callable

from reap.backends.mlx.metrics import PruningState
from reap.backends.mlx.model_adapters import (
    Qwen3MoeModelAdapter,
    get_shared_expert,
    make_attention_mask,
)
from reap.backends.mlx.router import Qwen3MoeRouter


logger = logging.getLogger(__name__)


def observe_model(
    model: Any,
    calibration_sequences: list[Any],
    config: Mapping[str, Any] | None = None,
    *,
    adapter: Any | None = None,
    debug_memory: bool = False,
    eval_fn: Callable[[Any], Any] | None = None,
    mask_fn: Callable[..., Any] | None = None,
) -> dict[int, dict[str, Any]]:
    """Collect pruning-compatible observer data with explicit MLX layer replay."""
    mx = _require_mlx_core()
    adapter = Qwen3MoeModelAdapter() if adapter is None else adapter
    config = {} if config is None else config
    eval_fn = mx.eval if eval_fn is None else eval_fn

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

        for layer_idx, layer in enumerate(layers):
            mask = _attention_mask(
                h,
                sequence_length=tokens.shape[-1],
                mask_fn=mask_fn,
            )
            h = _run_attention(layer, h, mask)
            moe_input = _call_required(layer, "post_attention_layernorm", h)

            if layer_idx in moe_layer_indices:
                h = h + _observe_moe_layer(
                    layer,
                    moe_input,
                    accumulators[layer_idx],
                    adapter=adapter,
                    config=config,
                )
            else:
                dense_mlp = adapter.get_dense_mlp(layer)
                h = h + dense_mlp(moe_input)

            eval_fn(h)
            if debug_memory:
                _log_memory(mx, layer_idx)

    return {layer_idx: state.report() for layer_idx, state in accumulators.items()}


def _require_mlx_core():
    try:
        import mlx.core as mx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "observe_model requires the optional 'mlx' package to execute. "
            "Install MLX in the active environment before observing."
        ) from exc
    return mx


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


def _run_attention(layer: Any, h: Any, mask: Any | None) -> Any:
    normalized = _call_required(layer, "input_layernorm", h)
    self_attn = getattr(layer, "self_attn", None)
    if not callable(self_attn):
        raise ValueError("Layer does not expose a callable self_attn module.")

    attention_output = self_attn(normalized, mask, cache=None)
    if isinstance(attention_output, tuple):
        attention_output = attention_output[0]
    return h + attention_output


def _observe_moe_layer(
    layer: Any,
    moe_input: Any,
    state: PruningState,
    *,
    adapter: Any,
    config: Mapping[str, Any],
) -> Any:
    moe = adapter.get_moe(layer)
    routing = Qwen3MoeRouter(moe, config)(moe_input)
    switch_mlp = getattr(moe, "switch_mlp", None)
    if not callable(switch_mlp):
        raise ValueError("MoE layer does not expose a callable switch_mlp module.")

    selected_outputs = switch_mlp(moe_input, routing.indices)
    state.accumulate(routing, selected_outputs=selected_outputs)

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
