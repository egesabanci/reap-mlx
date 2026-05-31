"""Model adapter helpers for MLX-backed MoE models.

This module is intentionally import-light. It describes MLX-LM-style model
layouts through plain Python attribute access and keeps optional MLX-LM imports
inside the functions that need them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence


@dataclass(frozen=True)
class MoeLayerConfig:
    """Architecture-neutral MoE layer metadata needed by MLX pipeline stages."""

    num_experts: int
    top_k: int
    norm_topk_prob: bool
    adapter_name: str = "qwen3_moe"


def _lookup_attr_path(root: Any, path: tuple[str, ...]) -> Any | None:
    current = root
    for attr in path:
        current = getattr(current, attr, None)
        if current is None:
            return None
    return current


def get_model_layers(model: Any) -> Sequence[Any]:
    """Return decoder layers from common MLX-LM model layouts."""
    candidate_paths = (
        ("model", "layers"),
        ("layers",),
        ("model", "model", "layers"),
    )
    for path in candidate_paths:
        layers = _lookup_attr_path(model, path)
        if layers is not None:
            return layers

    raise ValueError(
        "Could not find model layers. Expected one of: "
        "model.model.layers, model.layers, or model.model.model.layers."
    )


def get_shared_expert(moe: Any) -> Any | None:
    """Return the shared expert module if this MoE block exposes one."""
    for attr in ("shared_experts", "shared_expert"):
        shared = getattr(moe, attr, None)
        if shared is not None:
            return shared
    return None


def _config_value(
    config: Mapping[str, Any] | None,
    *keys: str,
    default: Any = None,
) -> Any:
    if config is None:
        return default
    for key in keys:
        value = config.get(key)
        if value is not None:
            return value
    return default


def _live_value(module: Any, *attrs: str) -> Any:
    for attr in attrs:
        value = getattr(module, attr, None)
        if value is not None:
            return value
    return None


def _live_or_config_value(
    module: Any,
    live_attrs: tuple[str, ...],
    config: Mapping[str, Any] | None,
    config_keys: tuple[str, ...],
    *,
    default: Any = None,
) -> Any:
    value = _live_value(module, *live_attrs)
    if value is not None:
        return value
    return _config_value(config, *config_keys, default=default)


def _positive_int(value: Any, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required.")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def update_qwen3_moe_config(
    config: MutableMapping[str, Any],
    *,
    num_experts: int,
    top_k: int,
) -> MutableMapping[str, Any]:
    """Update a Qwen3-MoE config dict after expert pruning."""
    num_experts = _positive_int(num_experts, "num_experts")
    top_k = min(_positive_int(top_k, "top_k"), num_experts)

    config["num_experts"] = num_experts
    config["num_experts_per_tok"] = top_k
    if "top_k" in config:
        config["top_k"] = top_k
    return config


def make_attention_mask(hidden_states: Any, cache: Any | None = None) -> Any:
    """Build an MLX-LM causal attention mask using the installed MLX-LM helper."""
    try:
        from mlx_lm.models.base import create_attention_mask
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "make_attention_mask requires the optional 'mlx_lm' package. "
            "Install MLX-LM before constructing MLX attention masks."
        ) from exc

    return create_attention_mask(hidden_states, cache=cache)


class Qwen3MoeModelAdapter:
    """Qwen3-MoE adapter for MLX-LM-style model objects."""

    adapter_name = "qwen3_moe"

    def layers(self, model: Any) -> Sequence[Any]:
        return get_model_layers(model)

    def identify_moe_layers(self, model: Any) -> list[int]:
        return [
            layer_idx
            for layer_idx, layer in enumerate(self.layers(model))
            if self.is_moe_layer(layer)
        ]

    def is_moe_layer(self, layer: Any) -> bool:
        mlp = getattr(layer, "mlp", None)
        return mlp is not None and getattr(mlp, "switch_mlp", None) is not None

    def get_moe(self, layer: Any) -> Any:
        if not self.is_moe_layer(layer):
            raise ValueError("Layer does not expose a Qwen3-style MoE mlp.switch_mlp.")
        return layer.mlp

    def get_dense_mlp(self, layer: Any) -> Any:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise ValueError("Layer does not expose an mlp module.")
        return mlp

    def get_layer_config(
        self,
        layer: Any,
        config: Mapping[str, Any] | None = None,
    ) -> MoeLayerConfig:
        moe = self.get_moe(layer)

        num_experts = _positive_int(
            _live_or_config_value(
                moe,
                ("num_experts",),
                config,
                ("num_experts",),
            ),
            "num_experts",
        )
        top_k = _positive_int(
            _live_or_config_value(
                moe,
                ("top_k", "num_experts_per_tok"),
                config,
                ("num_experts_per_tok", "top_k"),
            ),
            "top_k",
        )
        norm_topk_prob = bool(
            _live_or_config_value(
                moe,
                ("norm_topk_prob",),
                config,
                ("norm_topk_prob",),
                default=False,
            )
        )

        return MoeLayerConfig(
            num_experts=num_experts,
            top_k=top_k,
            norm_topk_prob=norm_topk_prob,
            adapter_name=self.adapter_name,
        )


__all__ = [
    "MoeLayerConfig",
    "Qwen3MoeModelAdapter",
    "get_model_layers",
    "get_shared_expert",
    "make_attention_mask",
    "update_qwen3_moe_config",
]
