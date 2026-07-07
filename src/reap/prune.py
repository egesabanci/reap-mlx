"""Expert pruning for adapter-described MLX MoE modules.

This module mutates live MLX-LM-style modules by slicing expert-stacked arrays
on their first dimension. It intentionally avoids importing MLX or MLX-LM at
module import time.
"""

from __future__ import annotations
import logging

from collections.abc import Mapping, MutableMapping
from typing import Any

import numpy as np

from reap.model_adapters import (
    infer_model_adapter,
    update_lfm2_moe_config,
    update_qwen3_moe_config,
)

logger = logging.getLogger(__name__)


_PRUNE_METHOD_ALIASES = {
    "frequency": "expert_frequency",
    "weighted_frequency_sum": "weighted_expert_frequency_sum",
}

_SUPPORTED_PRUNE_METHODS = {
    "expert_frequency",
    "ean_sum",
    "ean_mean",
    "weighted_ean_sum",
    "weighted_expert_frequency_sum",
    "reap",
    "max_activations",
}

_SLICE_FIELD_NAMES = ("weight", "scales", "biases", "bias")
_SWITCH_PROJECTION_NAMES = ("gate_proj", "up_proj", "down_proj")
_TOP_K_ATTRS = ("top_k", "num_experts_per_tok", "k")


def prune_experts(
    model: Any,
    config: MutableMapping[str, Any],
    observer_data: Mapping[int, Mapping[str, Any]],
    prune_method: str,
    compression_ratio: float,
    *,
    adapter: Any | None = None,
) -> dict[int, np.ndarray]:
    """Prune adapter-discovered MLX MoE experts in place.

    This mutates both the live model and the passed config mapping in place.
    Copy config before calling if pre-pruning values are still needed.

    Returns a mapping from layer index to ascending retained expert indices.
    """
    adapter = infer_model_adapter(model, config) if adapter is None else adapter
    _validate_adapter(adapter)
    _validate_compression_ratio(compression_ratio)

    layers = adapter.layers(model)
    keep_by_layer: dict[int, np.ndarray] = {}
    config_num_experts: int | None = None
    config_top_k: int | None = None

    total_pruned_global = 0
    for layer_idx in adapter.identify_moe_layers(model):
        if layer_idx not in observer_data:
            raise ValueError(
                f"Missing observer data for MoE layer {layer_idx}. "
                f"Available layers: {sorted(observer_data)}."
            )

        layer = layers[layer_idx]
        layer_config = adapter.get_layer_config(layer, config)
        retained_count = _retained_expert_count(
            layer_config.num_experts,
            compression_ratio,
        )
        saliency = _saliency_scores(
            observer_data[layer_idx],
            prune_method,
            num_experts=layer_config.num_experts,
            layer_idx=layer_idx,
        )
        keep_indices = compute_keep_indices(saliency, retained_count)

        if adapter.adapter_name == "lfm2_moe":
            _prune_lfm2_moe_layer(
                adapter.get_moe(layer),
                keep_indices,
                num_experts=layer_config.num_experts,
                old_top_k=layer_config.top_k,
                layer_idx=layer_idx,
            )
        else:
            _prune_qwen3_moe_layer(
                adapter.get_moe(layer),
                keep_indices,
                num_experts=layer_config.num_experts,
                old_top_k=layer_config.top_k,
                layer_idx=layer_idx,
            )
        keep_by_layer[layer_idx] = keep_indices
        if retained_count >= layer_config.num_experts:
            logger.warning(
                "Layer %d compression_ratio=%s retains all %d experts "
                "(no pruning).",
                layer_idx,
                compression_ratio,
                layer_config.num_experts,
            )
        else:
            total_pruned_global += layer_config.num_experts - retained_count

        new_top_k = min(layer_config.top_k, retained_count)
        if config_num_experts is None:
            config_num_experts = retained_count
            config_top_k = new_top_k
        elif config_num_experts != retained_count:
            raise ValueError(
                "MLX MoE config update requires all pruned layers to retain "
                f"the same expert count. Layer {layer_idx} retained "
                f"{retained_count}, expected {config_num_experts}."
            )
        else:
            config_top_k = min(config_top_k or new_top_k, new_top_k)

    if config_num_experts is not None:
        update_config = (
            update_lfm2_moe_config
            if adapter.adapter_name == "lfm2_moe"
            else update_qwen3_moe_config
        )
        update_config(
            config,
            num_experts=config_num_experts,
            top_k=config_top_k or config_num_experts,
        )

    if total_pruned_global == 0:
        logger.warning(
            "compression_ratio=%s resulted in zero experts being pruned "
            "across all layers. Use a larger ratio to reduce the model.",
            compression_ratio,
        )

    return keep_by_layer


def resolve_prune_method(prune_method: str) -> str:
    """Return the observer-data key for a supported prune method or alias."""
    resolved = _PRUNE_METHOD_ALIASES.get(prune_method, prune_method)
    if resolved not in _SUPPORTED_PRUNE_METHODS:
        supported = sorted(_SUPPORTED_PRUNE_METHODS | set(_PRUNE_METHOD_ALIASES))
        raise ValueError(
            f"Unsupported prune method {prune_method!r}. "
            f"Supported methods: {supported}."
        )
    return resolved


def compute_keep_indices(saliency: Any, retained_count: int) -> np.ndarray:
    """Keep the highest-saliency experts and return them in ascending order."""
    saliency_array = np.asarray(saliency, dtype=np.float64)
    if saliency_array.ndim != 1:
        raise ValueError(
            "saliency scores must be a one-dimensional array, got "
            f"shape {saliency_array.shape}."
        )
    retained_count = int(retained_count)
    if retained_count < 1 or retained_count > saliency_array.shape[0]:
        raise ValueError(
            "retained_count must be in [1, num_experts], got "
            f"{retained_count} for {saliency_array.shape[0]} experts."
        )
    if np.isnan(saliency_array).any():
        raise ValueError("saliency scores must not contain NaN values.")

    expert_ids = np.arange(saliency_array.shape[0])
    ranked = np.lexsort((expert_ids, -saliency_array))
    return np.sort(ranked[:retained_count]).astype(np.int64, copy=False)


def slice_first_dim(
    module: Any,
    keep_indices: np.ndarray,
    *,
    num_experts: int,
    required: bool = False,
    field_names: tuple[str, ...] = _SLICE_FIELD_NAMES,
) -> bool:
    """Slice present module fields on dimension 0.

    Returns ``True`` if at least one field was sliced.
    """
    sliced_any = False
    keep_list = _keep_list(keep_indices)

    for field_name in field_names:
        value = _get_module_value(module, field_name)
        if value is None:
            continue
        _validate_first_dim(
            value,
            field_name=field_name,
            num_experts=num_experts,
        )
        _set_module_value(module, field_name, value[keep_list])
        sliced_any = True

    if required and not sliced_any:
        raise ValueError(
            "Expected module to expose at least one sliceable field from "
            f"{field_names}."
        )
    return sliced_any


def _validate_adapter(adapter: Any) -> None:
    adapter_name = getattr(adapter, "adapter_name", None)
    if adapter_name not in {"qwen3_moe", "lfm2_moe"}:
        raise ValueError(
            "MLX expert pruning currently supports the qwen3_moe and "
            "lfm2_moe adapters only; "
            f"got {adapter_name!r}."
        )


def _validate_compression_ratio(compression_ratio: float) -> None:
    try:
        ratio = float(compression_ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError("compression_ratio must be a number in [0, 1).") from exc

    if ratio < 0.0 or ratio >= 1.0:
        raise ValueError(
            f"compression_ratio must be in [0, 1), got {compression_ratio}."
        )


def _retained_expert_count(num_experts: int, compression_ratio: float) -> int:
    num_experts = int(num_experts)
    num_to_prune = int(num_experts * float(compression_ratio))
    return max(num_experts - num_to_prune, 1)


def _saliency_scores(
    layer_observer_data: Mapping[str, Any],
    prune_method: str,
    *,
    num_experts: int,
    layer_idx: int,
) -> np.ndarray:
    resolved_method = resolve_prune_method(prune_method)
    if resolved_method not in layer_observer_data:
        raise ValueError(
            f"Prune method {prune_method!r} resolved to {resolved_method!r}, "
            f"but that key is missing from observer data for layer {layer_idx}. "
            f"Available keys: {sorted(layer_observer_data)}."
        )

    scores = np.asarray(layer_observer_data[resolved_method], dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError(
            f"Observer metric {resolved_method!r} for layer {layer_idx} must be "
            f"one-dimensional, got shape {scores.shape}."
        )
    if scores.shape[0] != num_experts:
        raise ValueError(
            f"Observer metric {resolved_method!r} for layer {layer_idx} has "
            f"{scores.shape[0]} scores, expected {num_experts}."
        )
    return scores


def _prune_qwen3_moe_layer(
    moe: Any,
    keep_indices: np.ndarray,
    *,
    num_experts: int,
    old_top_k: int,
    layer_idx: int,
) -> None:
    switch_mlp = getattr(moe, "switch_mlp", None)
    if switch_mlp is None:
        raise ValueError(f"MoE layer {layer_idx} does not expose switch_mlp.")

    for projection_name in _SWITCH_PROJECTION_NAMES:
        projection = getattr(switch_mlp, projection_name, None)
        if projection is None:
            raise ValueError(
                f"MoE layer {layer_idx} switch_mlp is missing "
                f"{projection_name}."
            )
        if _get_module_value(projection, "weight") is None:
            raise ValueError(
                f"MoE layer {layer_idx} switch_mlp.{projection_name} is "
                "missing required weight."
            )
        slice_first_dim(
            projection,
            keep_indices,
            num_experts=num_experts,
            required=True,
        )

    gate = getattr(moe, "gate", None)
    if gate is None:
        raise ValueError(f"MoE layer {layer_idx} does not expose a gate.")
    if _get_module_value(gate, "weight") is None:
        raise ValueError(f"MoE layer {layer_idx} gate is missing required weight.")
    slice_first_dim(
        gate,
        keep_indices,
        num_experts=num_experts,
        required=True,
        field_names=("weight", "bias", "e_score_correction_bias"),
    )

    retained_count = len(keep_indices)
    new_top_k = min(int(old_top_k), retained_count)
    _update_runtime_attrs(moe, gate, retained_count=retained_count, top_k=new_top_k)


def _prune_lfm2_moe_layer(
    moe: Any,
    keep_indices: np.ndarray,
    *,
    num_experts: int,
    old_top_k: int,
    layer_idx: int,
) -> None:
    switch_mlp = getattr(moe, "switch_mlp", None)
    if switch_mlp is None:
        raise ValueError(f"MoE layer {layer_idx} does not expose switch_mlp.")

    for projection_name in _SWITCH_PROJECTION_NAMES:
        projection = getattr(switch_mlp, projection_name, None)
        if projection is None:
            raise ValueError(
                f"MoE layer {layer_idx} switch_mlp is missing "
                f"{projection_name}."
            )
        if _get_module_value(projection, "weight") is None:
            raise ValueError(
                f"MoE layer {layer_idx} switch_mlp.{projection_name} is "
                "missing required weight."
            )
        slice_first_dim(
            projection,
            keep_indices,
            num_experts=num_experts,
            required=True,
        )

    gate = getattr(moe, "gate", None)
    if gate is None:
        raise ValueError(f"MoE layer {layer_idx} does not expose a gate.")
    if _get_module_value(gate, "weight") is None:
        raise ValueError(f"MoE layer {layer_idx} gate is missing required weight.")
    slice_first_dim(
        gate,
        keep_indices,
        num_experts=num_experts,
        required=True,
        field_names=("weight", "scales", "biases", "bias"),
    )

    if getattr(moe, "use_expert_bias", False) or hasattr(moe, "expert_bias"):
        expert_bias = getattr(moe, "expert_bias", None)
        if expert_bias is None:
            raise ValueError(
                f"MoE layer {layer_idx} use_expert_bias is enabled but "
                "expert_bias is missing."
            )
        setattr(
            moe,
            "expert_bias",
            _slice_value_first_dim(
                expert_bias,
                keep_indices,
                num_experts=num_experts,
                field_name="expert_bias",
            ),
        )

    retained_count = len(keep_indices)
    new_top_k = min(int(old_top_k), retained_count)
    _update_runtime_attrs(moe, gate, retained_count=retained_count, top_k=new_top_k)


def _update_runtime_attrs(
    moe: Any,
    gate: Any,
    *,
    retained_count: int,
    top_k: int,
) -> None:
    setattr(moe, "num_experts", retained_count)
    for attr in _TOP_K_ATTRS:
        if hasattr(moe, attr):
            setattr(moe, attr, top_k)

    for attr in ("num_experts", "n_routed_experts"):
        if hasattr(gate, attr):
            setattr(gate, attr, retained_count)
    if hasattr(gate, "top_k"):
        setattr(gate, "top_k", top_k)


def _get_module_value(module: Any, field_name: str) -> Any | None:
    getter = getattr(module, "get", None)
    if callable(getter):
        try:
            value = getter(field_name)
        except (KeyError, TypeError, AttributeError):
            value = None
        if value is not None:
            return value
    return getattr(module, field_name, None)


def _set_module_value(module: Any, field_name: str, value: Any) -> None:
    setattr(module, field_name, value)


def _slice_value_first_dim(
    value: Any,
    keep_indices: np.ndarray,
    *,
    num_experts: int,
    field_name: str,
) -> Any:
    _validate_first_dim(value, field_name=field_name, num_experts=num_experts)
    return value[_keep_list(keep_indices)]


def _validate_first_dim(value: Any, *, field_name: str, num_experts: int) -> None:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 1:
        raise ValueError(f"{field_name} must expose a non-empty shape.")
    if int(shape[0]) != int(num_experts):
        raise ValueError(
            f"{field_name} first dimension must match num_experts={num_experts}, "
            f"got shape {shape}."
        )


def _keep_list(keep_indices: np.ndarray) -> list[int]:
    return [int(idx) for idx in np.asarray(keep_indices, dtype=np.int64).tolist()]


__all__ = [
    "compute_keep_indices",
    "prune_experts",
    "resolve_prune_method",
    "slice_first_dim",
]
