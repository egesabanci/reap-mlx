"""NumPy pruning accumulators for MLX-backed MoE observation.

MLX observer code should pass compact selected-route arrays here after forcing
any needed MLX evaluation boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


_FLOAT_EPS = np.finfo(np.float64).eps


@dataclass
class PruningState:
    """Pruning-only observer state for one MoE layer."""

    num_experts: int
    total_tokens: int
    expert_frequency: np.ndarray
    pairwise_expert_frequency: np.ndarray
    ean_sum: np.ndarray
    weighted_ean_sum: np.ndarray
    weighted_expert_frequency_sum: np.ndarray
    max_activations: np.ndarray

    @classmethod
    def initialize(cls, num_experts: int) -> "PruningState":
        """Create zero-initialized pruning state for ``num_experts`` experts."""
        num_experts = int(num_experts)
        if num_experts < 1:
            raise ValueError(f"num_experts must be positive, got {num_experts}.")

        return cls(
            num_experts=num_experts,
            total_tokens=0,
            expert_frequency=np.zeros(num_experts, dtype=np.int64),
            pairwise_expert_frequency=np.zeros(
                (num_experts, num_experts),
                dtype=np.int64,
            ),
            ean_sum=np.zeros(num_experts, dtype=np.float64),
            weighted_ean_sum=np.zeros(num_experts, dtype=np.float64),
            weighted_expert_frequency_sum=np.zeros(num_experts, dtype=np.float64),
            max_activations=np.zeros(num_experts, dtype=np.float32),
        )

    def accumulate(
        self,
        routing: Any | None = None,
        *,
        indices: Any | None = None,
        scores: Any | None = None,
        selected_outputs: Any | None = None,
        selected_output_norms: Any | None = None,
        selected_output_maxes: Any | None = None,
    ) -> "PruningState":
        """Accumulate one selected-route batch into this state.

        ``indices`` and ``scores`` must share shape ``[..., top_k]``. Pass either
        selected expert outputs shaped ``[..., top_k, hidden]`` or precomputed
        norms and maxes shaped like ``indices``.
        """
        if routing is not None:
            if indices is not None or scores is not None:
                raise ValueError(
                    "Pass either routing or explicit indices/scores, not both."
                )
            indices = getattr(routing, "indices", None)
            scores = getattr(routing, "scores", None)

        if indices is None or scores is None:
            raise ValueError("accumulate requires selected expert indices and scores.")

        indices_array = np.asarray(indices, dtype=np.int64)
        scores_array = np.asarray(scores, dtype=np.float64)
        self._validate_route_arrays(indices_array, scores_array)

        if indices_array.shape[-1] == 0:
            raise ValueError("indices must include at least one selected expert.")

        token_count = int(np.prod(indices_array.shape[:-1]))
        if token_count == 0:
            return self

        flat_indices = indices_array.reshape(-1)
        min_index = int(flat_indices.min())
        max_index = int(flat_indices.max())
        if min_index < 0 or max_index >= self.num_experts:
            raise ValueError(
                "selected expert indices must be in [0, num_experts): "
                f"got min={min_index}, max={max_index}, "
                f"num_experts={self.num_experts}."
            )

        norms_array = self._selected_output_norms(
            indices_array,
            selected_outputs=selected_outputs,
            selected_output_norms=selected_output_norms,
        )
        maxes_array = self._selected_output_maxes(
            indices_array,
            selected_outputs=selected_outputs,
            selected_output_maxes=selected_output_maxes,
        )

        batch_frequency = np.bincount(
            flat_indices,
            minlength=self.num_experts,
        )[: self.num_experts].astype(np.int64, copy=False)

        self.total_tokens += token_count
        self.expert_frequency += batch_frequency
        self.pairwise_expert_frequency += (
            batch_frequency[:, None] + batch_frequency[None, :]
        )

        flat_norms = norms_array.reshape(-1).astype(np.float64, copy=False)
        flat_scores = scores_array.reshape(-1)
        flat_maxes = maxes_array.reshape(-1).astype(np.float32, copy=False)

        np.add.at(self.ean_sum, flat_indices, flat_norms)
        np.add.at(self.weighted_ean_sum, flat_indices, flat_norms * flat_scores)
        np.add.at(self.weighted_expert_frequency_sum, flat_indices, flat_scores)
        np.maximum.at(self.max_activations, flat_indices, flat_maxes)

        return self

    def report(self) -> dict[str, Any]:
        """Return pruning data compatible with the existing observer schema."""
        token_denominator = max(self.total_tokens, 1)
        count_denominator = np.maximum(
            self.expert_frequency.astype(np.float64),
            _FLOAT_EPS,
        )

        return {
            "total_tokens": int(self.total_tokens),
            "expert_frequency": self.expert_frequency.copy(),
            "pairwise_expert_frequency": self.pairwise_expert_frequency.copy(),
            "expert_proba": self.expert_frequency.astype(np.float64)
            / token_denominator,
            "ean_sum": self.ean_sum.copy(),
            "ean_mean": (self.ean_sum / count_denominator).astype(np.float32),
            "weighted_ean_sum": self.weighted_ean_sum.copy(),
            "weighted_expert_frequency_sum": (
                self.weighted_expert_frequency_sum.copy()
            ),
            "reap": (self.weighted_ean_sum / count_denominator).astype(np.float32),
            "max_activations": self.max_activations.copy(),
        }

    def _validate_route_arrays(
        self,
        indices_array: np.ndarray,
        scores_array: np.ndarray,
    ) -> None:
        if indices_array.ndim < 1:
            raise ValueError(
                "indices must have shape [..., top_k], got a scalar value."
            )
        if indices_array.shape != scores_array.shape:
            raise ValueError(
                "indices and scores must have the same shape: "
                f"{indices_array.shape} != {scores_array.shape}."
            )

    def _selected_output_norms(
        self,
        indices_array: np.ndarray,
        *,
        selected_outputs: Any | None,
        selected_output_norms: Any | None,
    ) -> np.ndarray:
        if selected_output_norms is not None:
            norms_array = np.asarray(selected_output_norms, dtype=np.float64)
            self._validate_selected_stat_shape(
                "selected_output_norms",
                norms_array,
                indices_array.shape,
            )
            return norms_array

        if selected_outputs is None:
            raise ValueError(
                "selected_outputs or selected_output_norms must be provided "
                "for non-empty accumulation."
            )

        outputs_array = np.asarray(selected_outputs)
        self._validate_selected_output_shape(outputs_array, indices_array.shape)
        return np.linalg.norm(outputs_array, axis=-1).astype(np.float64, copy=False)

    def _selected_output_maxes(
        self,
        indices_array: np.ndarray,
        *,
        selected_outputs: Any | None,
        selected_output_maxes: Any | None,
    ) -> np.ndarray:
        if selected_output_maxes is not None:
            maxes_array = np.asarray(selected_output_maxes, dtype=np.float32)
            self._validate_selected_stat_shape(
                "selected_output_maxes",
                maxes_array,
                indices_array.shape,
            )
            return maxes_array

        if selected_outputs is None:
            raise ValueError(
                "selected_outputs or selected_output_maxes must be provided "
                "for non-empty accumulation."
            )

        outputs_array = np.asarray(selected_outputs)
        self._validate_selected_output_shape(outputs_array, indices_array.shape)
        return np.max(outputs_array, axis=-1).astype(np.float32, copy=False)

    def _validate_selected_stat_shape(
        self,
        name: str,
        values: np.ndarray,
        expected_shape: tuple[int, ...],
    ) -> None:
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {values.shape}."
            )

    def _validate_selected_output_shape(
        self,
        values: np.ndarray,
        route_shape: tuple[int, ...],
    ) -> None:
        if values.shape[:-1] != route_shape:
            raise ValueError(
                "selected_outputs must have shape [..., top_k, hidden] matching "
                f"indices shape {route_shape}, got {values.shape}."
            )
        if values.shape[-1] == 0:
            raise ValueError("selected_outputs hidden dimension must be non-empty.")


__all__ = ["PruningState"]
