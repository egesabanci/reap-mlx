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
    use_expert_bias: bool = False


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


def update_lfm2_moe_config(
    config: MutableMapping[str, Any],
    *,
    num_experts: int,
    top_k: int,
) -> MutableMapping[str, Any]:
    """Update an LFM2 MoE config dict after expert pruning."""
    return update_qwen3_moe_config(config, num_experts=num_experts, top_k=top_k)


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


def make_ssm_mask(hidden_states: Any, cache: Any | None = None) -> Any:
    """Build an MLX-LM SSM/conv mask using the installed MLX-LM helper."""
    try:
        from mlx_lm.models.base import create_ssm_mask
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "make_ssm_mask requires the optional 'mlx_lm' package. "
            "Install MLX-LM before constructing MLX SSM masks."
        ) from exc

    return create_ssm_mask(hidden_states, cache=cache)


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


class Lfm2MoeModelAdapter:
    """Liquid LFM2.5 MoE adapter for MLX-LM-style model objects."""

    adapter_name = "lfm2_moe"

    def layers(self, model: Any) -> Sequence[Any]:
        return get_model_layers(model)

    def identify_moe_layers(self, model: Any) -> list[int]:
        return [
            layer_idx
            for layer_idx, layer in enumerate(self.layers(model))
            if self.is_moe_layer(layer)
        ]

    def is_moe_layer(self, layer: Any) -> bool:
        feed_forward = getattr(layer, "feed_forward", None)
        return (
            feed_forward is not None
            and getattr(feed_forward, "switch_mlp", None) is not None
        )

    def get_moe(self, layer: Any) -> Any:
        if not self.is_moe_layer(layer):
            raise ValueError(
                "Layer does not expose an LFM2-style MoE feed_forward.switch_mlp."
            )
        return layer.feed_forward

    def get_dense_mlp(self, layer: Any) -> Any:
        feed_forward = getattr(layer, "feed_forward", None)
        if feed_forward is None:
            raise ValueError("Layer does not expose a feed_forward module.")
        return feed_forward

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
        use_expert_bias = bool(
            _live_or_config_value(
                moe,
                ("use_expert_bias",),
                config,
                ("use_expert_bias",),
                default=False,
            )
        )

        return MoeLayerConfig(
            num_experts=num_experts,
            top_k=top_k,
            norm_topk_prob=norm_topk_prob,
            adapter_name=self.adapter_name,
            use_expert_bias=use_expert_bias,
        )


def infer_model_adapter(
    model: Any | None = None,
    config: Mapping[str, Any] | None = None,
) -> Any:
    """Infer the MLX architecture adapter from config or model layout."""
    model_type = _config_value(config, "model_type")
    architectures = _config_value(config, "architectures", default=()) or ()
    if model_type == "lfm2_moe" or any(
        str(architecture).startswith("Lfm2") for architecture in architectures
    ):
        return Lfm2MoeModelAdapter()

    if model is not None:
        try:
            layers = get_model_layers(model)
        except ValueError:
            layers = ()
        if any(
            getattr(getattr(layer, "feed_forward", None), "switch_mlp", None)
            is not None
            for layer in layers
        ):
            return Lfm2MoeModelAdapter()

    return Qwen3MoeModelAdapter()


__all__ = [
    "Lfm2MoeModelAdapter",
    "MoeLayerConfig",
    "Qwen3MoeModelAdapter",
    "get_model_layers",
    "get_shared_expert",
    "infer_model_adapter",
    "make_attention_mask",
    "make_ssm_mask",
    "update_lfm2_moe_config",
    "update_qwen3_moe_config",
]
