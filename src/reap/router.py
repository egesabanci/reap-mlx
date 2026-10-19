"""Router adapters for MLX-backed MoE models.

This module is intentionally import-light: importing it must not require MLX,
MLX-LM, or other runtime packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

_NORM_EPSILON: float = 1e-20
# mlx-lm 0.32's sigmoid-gated LFM2-MoE renormalizes with this epsilon instead.
_SIGMOID_NORM_EPSILON: float = 1e-6

# mlx-lm partitions Qwen2-MoE and OLMoE gates in descending order
# (argpartition(-gates, kth=k-1)[..., :k]) but Qwen3-MoE in ascending order
# (argpartition(gates, kth=-k)[..., -k:]). On exact ties at the top-k boundary,
# which are common with bfloat16 gates, the two forms keep different experts,
# so the replay must use the same form as the architecture it mirrors.
_DESCENDING_TOPK_MODEL_TYPES = frozenset({"qwen2_moe", "olmoe"})


@dataclass(frozen=True)
class RouterResult:
    """Architecture-neutral selected-router output.

    Indices and scores are in the order the mirrored mlx-lm router emits them
    (argpartition order for most architectures), not sorted by score.

    ``scores`` are the scores actually used to weight expert outputs in the
    forward pass (e.g. bias-adjusted for LFM2 under mlx-lm 0.31).
    ``saliency_scores``, when provided, are bias-free scores intended for REAP
    metric accumulation so that saliency reflects pure routing probability
    rather than additive expert bias; consumers fall back to ``scores`` when
    it is None.
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
        model_type = _config_value(config, "model_type", default="")
        self.descending_topk = str(model_type) in _DESCENDING_TOPK_MODEL_TYPES

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
        if self.descending_topk:
            flat_indices = mx.argpartition(-gates, kth=self.top_k - 1, axis=-1)[
                ..., : self.top_k
            ]
        else:
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
    """Liquid LFM2.5 MoE router matching MLX-LM routing semantics.

    mlx-lm 0.31 adds ``expert_bias`` to softmax gates, so the bias moves both
    the expert choice and the weights. mlx-lm 0.32 follows HF Transformers:
    sigmoid gates, ``expert_bias`` only for choosing the top-k, and weights
    scaled by ``routed_scaling_factor``. Only the 0.32 block stores
    ``routed_scaling_factor``, so its presence selects the sigmoid routing.
    """

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
        self.routed_scaling_factor = getattr(
            moe_layer, "routed_scaling_factor", None
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

        logits = self.gate(flat_hidden_states)
        num_experts = logits.shape[-1]
        if self.top_k > num_experts:
            raise ValueError(
                f"top_k={self.top_k} cannot exceed num_experts={num_experts}.",
            )

        if self.routed_scaling_factor is None:
            flat_indices, flat_scores, saliency_scores = self._softmax_routes(
                mx, logits
            )
        else:
            flat_indices, flat_scores = self._sigmoid_routes(mx, logits)
            # The weights carry no expert_bias, so they are the saliency scores.
            saliency_scores = None

        output_shape = (*leading_shape, self.top_k)
        if saliency_scores is not None:
            saliency_scores = saliency_scores.astype(hidden_states.dtype).reshape(
                output_shape
            )
        return RouterResult(
            indices=flat_indices.reshape(output_shape),
            scores=flat_scores.astype(hidden_states.dtype).reshape(output_shape),
            logits=None,
            score_mode="actual",
            saliency_scores=saliency_scores,
        )

    def _softmax_routes(self, mx: Any, logits: Any) -> tuple[Any, Any, Any]:
        """mlx-lm 0.31: biased softmax gates choose and weight the experts."""
        logits = logits.astype(mx.float32)
        # LFM2 architecture does NOT use precise softmax; this is intentional.
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

        # Saliency scores: pure softmax before expert_bias
        pure_gates = mx.softmax(logits, axis=-1)
        pure_flat_scores = mx.take_along_axis(pure_gates, flat_indices, axis=-1)
        if self.norm_topk_prob:
            pure_flat_scores = pure_flat_scores / (
                mx.sum(pure_flat_scores, axis=-1, keepdims=True) + _NORM_EPSILON
            )
        return flat_indices, flat_scores, pure_flat_scores

    def _sigmoid_routes(self, mx: Any, logits: Any) -> tuple[Any, Any]:
        """mlx-lm 0.32: expert_bias only chooses; weights are unbiased sigmoids."""
        # Same dtypes as mlx-lm: sigmoid in the gate dtype, selection in float32.
        routing_weights = mx.sigmoid(logits)
        selection_scores = routing_weights
        if self.use_expert_bias:
            selection_scores = routing_weights.astype(mx.float32) + self.expert_bias

        flat_indices = mx.argpartition(selection_scores, kth=-self.top_k, axis=-1)[
            ..., -self.top_k :
        ]
        flat_scores = mx.take_along_axis(routing_weights, flat_indices, axis=-1)
        if self.norm_topk_prob:
            flat_scores = flat_scores / (
                mx.sum(flat_scores, axis=-1, keepdims=True) + _SIGMOID_NORM_EPSILON
            )
        return flat_indices, flat_scores * self.routed_scaling_factor


class MixtralMoeRouter:
    """Mixtral-style router: top-k on raw gate logits, then softmax on selected.

    mlx-lm 0.32 routes PhiMoE with HF's sparsemixer instead. Only that block
    stores ``jitter_eps``, so its presence selects the sparsemixer routing.
    """

    def __init__(
        self,
        moe_layer: Any,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.gate = getattr(moe_layer, "gate", None)
        if self.gate is None:
            raise ValueError("MixtralMoeRouter requires an MoE layer with a gate.")
        self.top_k = _live_or_config_value(
            moe_layer,
            "top_k",
            config,
            "num_experts_per_tok",
            "top_k",
        )
        if self.top_k is None:
            self.top_k = _live_or_config_value(
                moe_layer,
                "num_experts_per_tok",
                config,
                "num_experts_per_tok",
            )
        if self.top_k is None:
            raise ValueError(
                "MixtralMoeRouter requires top-k from the MoE module or config."
            )
        self.top_k = int(self.top_k)
        if self.top_k < 1:
            raise ValueError(f"top_k must be positive, got {self.top_k}.")
        self.jitter_eps = getattr(moe_layer, "jitter_eps", None)

    def __call__(self, hidden_states: Any) -> RouterResult:
        mx = _require_mlx_core()
        if hidden_states.ndim not in (2, 3):
            raise ValueError(
                "MixtralMoeRouter expects hidden states with shape "
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
        if self.jitter_eps is not None:
            flat_indices, flat_scores = _sparsemixer(
                mx, logits, top_k=self.top_k, jitter_eps=self.jitter_eps
            )
        else:
            # Match mlx-lm Mixtral: partition on raw logits, softmax only on top-k.
            flat_indices = mx.argpartition(-logits, kth=self.top_k - 1, axis=-1)[
                ..., : self.top_k
            ]
            flat_scores = mx.take_along_axis(logits, flat_indices, axis=-1)
            flat_scores = mx.softmax(flat_scores, axis=-1, precise=True)
        output_shape = (*leading_shape, self.top_k)
        return RouterResult(
            indices=flat_indices.reshape(output_shape),
            scores=flat_scores.reshape(output_shape),
            logits=None,
            score_mode="actual",
        )


def _sparsemixer(
    mx: Any,
    gates: Any,
    *,
    top_k: int,
    jitter_eps: float,
) -> tuple[Any, Any]:
    """Inference-path sparsemixer, op for op as mlx-lm 0.32 routes PhiMoE.

    Each pass picks the best remaining expert and weights it by a softmax over
    the experts whose logits are within a ``2 * jitter_eps`` relative band of
    it, so indices come out in descending-logit order.
    """
    masked = gates
    indices, scores = [], []
    for _ in range(top_k):
        top = masked.max(axis=-1, keepdims=True)
        index = mx.argmax(masked, axis=-1, keepdims=True)
        factor = mx.maximum(mx.abs(gates), top)
        band = mx.where((top - gates) / factor > 2 * jitter_eps, -mx.inf, masked)
        band = mx.softmax(band, axis=-1, precise=True)
        scores.append(mx.take_along_axis(band, index, axis=-1))
        indices.append(index)
        masked = mx.put_along_axis(
            masked, index, mx.array(-mx.inf, masked.dtype), axis=-1
        )
    return mx.concatenate(indices, axis=-1), mx.concatenate(scores, axis=-1)


__all__ = ["Lfm2MoeRouter", "MixtralMoeRouter", "Qwen3MoeRouter", "RouterResult"]
