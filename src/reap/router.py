"""Router adapters for MLX-backed MoE models.

This module is intentionally import-light: importing it must not require MLX,
MLX-LM, or other runtime packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


_NORM_EPSILON: float = 1e-20


@dataclass(frozen=True)
class RouterResult:
    """Architecture-neutral selected-router output.

    Indices and scores are in argpartition order, not sorted by score.

    ``scores`` are the scores actually used to weight expert outputs in the
    forward pass (e.g. bias-adjusted for LFM2). ``saliency_scores``, when
    provided, are bias-free scores intended for REAP metric accumulation so
    that saliency reflects pure routing probability rather than additive
    expert bias; consumers fall back to ``scores`` when it is None.
    """

    indices: Any
    scores: Any
    logits: Any | None = None
    score_mode: str = "actual"
    saliency_scores: Any | None = None


def _require_mlx_core():
    try:
        import mlx.core as mx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MLX router adapters require the optional 'mlx' package to execute. "
            "Install MLX in the active environment before routing."
        ) from exc
    return mx


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


def _live_or_config_value(
    module: Any,
    live_attr: str,
    config: Mapping[str, Any] | None,
    *config_keys: str,
    default: Any = None,
) -> Any:
    value = getattr(module, live_attr, None)
    if value is not None:
        return value
    return _config_value(config, *config_keys, default=default)


class Qwen3MoeRouter:
    """Qwen3-MoE router adapter matching MLX-LM routing semantics."""

    def __init__(
        self,
        mlp_layer: Any,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.gate = getattr(mlp_layer, "gate", None)
        if self.gate is None:
            raise ValueError("Qwen3MoeRouter requires an MLX MoE layer with a gate.")

        self.top_k = _live_or_config_value(
            mlp_layer,
            "top_k",
            config,
            "num_experts_per_tok",
            "top_k",
        )
        if self.top_k is None:
            raise ValueError(
                "Qwen3MoeRouter requires top-k from mlp_layer.top_k, "
                "config['num_experts_per_tok'], or config['top_k']."
            )
        self.top_k = int(self.top_k)
        if self.top_k < 1:
            raise ValueError(f"top_k must be positive, got {self.top_k}.")

        self.norm_topk_prob = bool(
            _live_or_config_value(
                mlp_layer,
                "norm_topk_prob",
                config,
                "norm_topk_prob",
                default=False,
            )
        )

    def __call__(self, hidden_states: Any) -> RouterResult:
        mx = _require_mlx_core()

        if hidden_states.ndim not in (2, 3):
            raise ValueError(
                "Qwen3MoeRouter expects hidden states with shape "
                "[tokens, hidden] or [batch, seq, hidden], got "
                f"{hidden_states.shape}."
            )

        leading_shape = hidden_states.shape[:-1]
        hidden_size = hidden_states.shape[-1]
        flat_hidden_states = hidden_states.reshape(-1, hidden_size)

        logits = self.gate(flat_hidden_states)
        num_experts = logits.shape[-1]
        if self.top_k > num_experts:
            raise ValueError(
                f"top_k={self.top_k} cannot exceed num_experts={num_experts}."
            )

        # Qwen3-MoE architecture uses precise softmax (`precise=True`) to match
        # the upstream MLX-LM forward pass behavior for this architecture.
        # The LFM2-MoE router below intentionally omits `precise` for the same reason:
        # it mirrors that architecture's own MLX-LM forward pass.
        # See docs/observation-and-metrics.md for full router semantics.
        gates = mx.softmax(logits, axis=-1, precise=True)
        flat_indices = mx.argpartition(gates, kth=-self.top_k, axis=-1)[
            ..., -self.top_k :
        ]
        flat_scores = mx.take_along_axis(gates, flat_indices, axis=-1)

        if self.norm_topk_prob:
            flat_scores = flat_scores / (
                flat_scores.sum(axis=-1, keepdims=True) + _NORM_EPSILON
            )

        output_shape = (*leading_shape, self.top_k)
        return RouterResult(
            indices=flat_indices.reshape(output_shape),
            scores=flat_scores.reshape(output_shape),
            logits=None,
            score_mode="actual",
        )


class Lfm2MoeRouter:
    """Liquid LFM2.5 MoE router matching MLX-LM routing semantics."""

    def __init__(
        self,
        moe_layer: Any,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.gate = getattr(moe_layer, "gate", None)
        if self.gate is None:
            raise ValueError("Lfm2MoeRouter requires an MLX MoE layer with a gate.")

        self.top_k = _live_or_config_value(
            moe_layer,
            "top_k",
            config,
            "num_experts_per_tok",
            "top_k",
        )
        if self.top_k is None:
            raise ValueError(
                "Lfm2MoeRouter requires top-k from moe_layer.top_k, "
                "config['num_experts_per_tok'], or config['top_k']."
            )
        self.top_k = int(self.top_k)
        if self.top_k < 1:
            raise ValueError(f"top_k must be positive, got {self.top_k}.")

        self.norm_topk_prob = bool(
            _live_or_config_value(
                moe_layer,
                "norm_topk_prob",
                config,
                "norm_topk_prob",
                default=False,
            )
        )
        self.use_expert_bias = bool(
            _live_or_config_value(
                moe_layer,
                "use_expert_bias",
                config,
                "use_expert_bias",
                default=False,
            )
        )
        self.expert_bias = getattr(moe_layer, "expert_bias", None)
        if self.use_expert_bias and self.expert_bias is None:
            raise ValueError(
                "Lfm2MoeRouter requires moe_layer.expert_bias when "
                "use_expert_bias is enabled."
            )

    def __call__(self, hidden_states: Any) -> RouterResult:
        mx = _require_mlx_core()

        if hidden_states.ndim not in (2, 3):
            raise ValueError(
                "Lfm2MoeRouter expects hidden states with shape "
                "[tokens, hidden] or [batch, seq, hidden], got "
                f"{hidden_states.shape}.",
            )

        # Flatten to 2D for the gate (consistent with Qwen3MoeRouter).
        leading_shape = hidden_states.shape[:-1]
        hidden_size = hidden_states.shape[-1]
        flat_hidden_states = hidden_states.reshape(-1, hidden_size)

        logits = self.gate(flat_hidden_states).astype(mx.float32)
        num_experts = logits.shape[-1]
        if self.top_k > num_experts:
            raise ValueError(
                f"top_k={self.top_k} cannot exceed num_experts={num_experts}.",
            )

        # LFM2 architecture does NOT use precise softmax \xe2\x80\x94 intentional
        gates = mx.softmax(logits, axis=-1)
        if self.use_expert_bias:
            gates = gates + self.expert_bias

        flat_indices = mx.argpartition(gates, kth=-self.top_k, axis=-1)[
            ..., -self.top_k :
        ]
        flat_scores = mx.take_along_axis(gates, flat_indices, axis=-1)
        if self.norm_topk_prob:
            flat_scores = flat_scores / (
                mx.sum(flat_scores, axis=-1, keepdims=True) + _NORM_EPSILON
            )
        scores = flat_scores.astype(hidden_states.dtype)

        # Saliency scores: pure softmax before expert_bias
        pure_gates = mx.softmax(logits, axis=-1)
        pure_flat_scores = mx.take_along_axis(pure_gates, flat_indices, axis=-1)
        if self.norm_topk_prob:
            pure_flat_scores = pure_flat_scores / (
                mx.sum(pure_flat_scores, axis=-1, keepdims=True) + _NORM_EPSILON
            )
        saliency_scores = pure_flat_scores.astype(hidden_states.dtype)

        output_shape = (*leading_shape, self.top_k)
        return RouterResult(
            indices=flat_indices.reshape(output_shape),
            scores=scores.reshape(output_shape),
            logits=None,
            score_mode="actual",
            saliency_scores=saliency_scores.reshape(output_shape),
        )


__all__ = ["Lfm2MoeRouter", "Qwen3MoeRouter", "RouterResult"]
