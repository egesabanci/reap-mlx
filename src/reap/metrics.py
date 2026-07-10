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
    total_slots: int
    expert_frequency: np.ndarray
    pairwise_expert_frequency: np.ndarray | None
    ean_sum: np.ndarray
    weighted_ean_sum: np.ndarray
    weighted_expert_frequency_sum: np.ndarray
    max_activations: np.ndarray
    shared_expert_ean_sum: float = 0.0
    shared_expert_tokens: int = 0

    @classmethod
    def initialize(
        cls,
        num_experts: int,
        *,
        track_pairwise: bool = False,
    ) -> "PruningState":
        """Create zero-initialized pruning state for ``num_experts`` experts."""
        num_experts = int(num_experts)
        if num_experts < 1:
            raise ValueError(f"num_experts must be positive, got {num_experts}.")

        pairwise = (
            np.zeros((num_experts, num_experts), dtype=np.int64)
            if track_pairwise
            else None
        )
        return cls(
            num_experts=num_experts,
            total_tokens=0,
            total_slots=0,
            expert_frequency=np.zeros(num_experts, dtype=np.int64),
            pairwise_expert_frequency=pairwise,
            ean_sum=np.zeros(num_experts, dtype=np.float64),
            weighted_ean_sum=np.zeros(num_experts, dtype=np.float64),
            weighted_expert_frequency_sum=np.zeros(num_experts, dtype=np.float64),
            max_activations=np.zeros(num_experts, dtype=np.float32),
            shared_expert_ean_sum=0.0,
            shared_expert_tokens=0,
        )

    def snapshot(self) -> dict[str, Any]:
        """Return a deep copy of accumulator fields for mid-sequence rollback."""
        return {
            "total_tokens": int(self.total_tokens),
            "total_slots": int(self.total_slots),
            "expert_frequency": self.expert_frequency.copy(),
            "pairwise_expert_frequency": (
                None
                if self.pairwise_expert_frequency is None
                else self.pairwise_expert_frequency.copy()
            ),
            "ean_sum": self.ean_sum.copy(),
            "weighted_ean_sum": self.weighted_ean_sum.copy(),
            "weighted_expert_frequency_sum": self.weighted_expert_frequency_sum.copy(),
            "max_activations": self.max_activations.copy(),
            "shared_expert_ean_sum": float(self.shared_expert_ean_sum),
            "shared_expert_tokens": int(self.shared_expert_tokens),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore accumulator fields from :meth:`snapshot`."""
        self.total_tokens = int(snapshot["total_tokens"])
        self.total_slots = int(snapshot["total_slots"])
        self.expert_frequency = np.asarray(
            snapshot["expert_frequency"], dtype=np.int64
        ).copy()
        pairwise = snapshot.get("pairwise_expert_frequency")
        self.pairwise_expert_frequency = (
            None if pairwise is None else np.asarray(pairwise, dtype=np.int64).copy()
        )
        self.ean_sum = np.asarray(snapshot["ean_sum"], dtype=np.float64).copy()
        self.weighted_ean_sum = np.asarray(
            snapshot["weighted_ean_sum"], dtype=np.float64
        ).copy()
        self.weighted_expert_frequency_sum = np.asarray(
            snapshot["weighted_expert_frequency_sum"], dtype=np.float64
        ).copy()
        self.max_activations = np.asarray(
            snapshot["max_activations"], dtype=np.float32
        ).copy()
        self.shared_expert_ean_sum = float(snapshot.get("shared_expert_ean_sum", 0.0))
        self.shared_expert_tokens = int(snapshot.get("shared_expert_tokens", 0))

    def accumulate_shared_expert(self, shared_outputs: Any) -> None:
        """Record shared-expert activation energy (not used for expert ranking)."""
        outputs = np.asarray(shared_outputs, dtype=np.float64)
        if outputs.size == 0:
            return
        # Mean L2 norm over token positions.
        if outputs.ndim >= 2:
            token_norms = np.linalg.norm(outputs.reshape(-1, outputs.shape[-1]), axis=-1)
            self.shared_expert_ean_sum += float(token_norms.sum())
            self.shared_expert_tokens += int(token_norms.size)
        else:
            self.shared_expert_ean_sum += float(np.linalg.norm(outputs))
            self.shared_expert_tokens += 1

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
        # total_slots counts every selected (token, expert) slot, i.e.
        # total_tokens * top_k. expert_frequency is summed over slots, so the
        # true per-expert probability is frequency / total_slots -- using
        # total_tokens instead would let expert_proba exceed 1.0 for top_k > 1.
        slot_count = token_count * int(indices_array.shape[-1])
        self.total_slots += slot_count
        self.expert_frequency += batch_frequency
        if self.pairwise_expert_frequency is not None:
            # Per-token co-occurrence: count how often experts i and j are
            # selected for the SAME token. A batch-level outer product of
            # bincount frequencies would inflate co-occurrence for experts
            # selected on different tokens, so build a per-token one-hot
            # matrix and accumulate one_hot.T @ one_hot instead.
            top_k = indices_array.shape[-1]
            per_token = indices_array.reshape(token_count, top_k)
            one_hot = np.zeros((token_count, self.num_experts), dtype=np.int64)
            np.add.at(
                one_hot,
                (np.arange(token_count)[:, None], per_token),
                1,
            )
            self.pairwise_expert_frequency += one_hot.T @ one_hot

        flat_norms = norms_array.reshape(-1).astype(np.float64, copy=False)
        flat_scores = scores_array.reshape(-1)
        flat_maxes = maxes_array.reshape(-1).astype(np.float32, copy=False)

        np.add.at(self.ean_sum, flat_indices, flat_norms)
        np.add.at(self.weighted_ean_sum, flat_indices, flat_norms * flat_scores)
        np.add.at(self.weighted_expert_frequency_sum, flat_indices, flat_scores)
        np.maximum.at(self.max_activations, flat_indices, flat_maxes)

        return self

    def _require_finite_accumulators(self) -> None:
        """Raise if any accumulated statistic contains NaN or Inf.

        Catching this here (per-layer, before saliency is derived) avoids
        wasting the entire observation run only to fail later in
        compute_keep_indices with a context-free "saliency scores must not
        contain NaN values" error.
        """
        checks = {
            "ean_sum": self.ean_sum,
            "weighted_ean_sum": self.weighted_ean_sum,
            "weighted_expert_frequency_sum": self.weighted_expert_frequency_sum,
            "max_activations": self.max_activations,
            "expert_frequency": self.expert_frequency,
        }
        for name, values in checks.items():
            arr = np.asarray(values)
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"PruningState metric {name!r} contains non-finite "
                    f"values (NaN/Inf): {arr.tolist()}",
                )

    def report(self) -> dict[str, Any]:
        """Return pruning data compatible with the existing observer schema."""
        self._require_finite_accumulators()
        slot_denominator = max(self.total_slots, 1)
        count_denominator = np.maximum(
            self.expert_frequency.astype(np.float64),
            _FLOAT_EPS,
        )

        shared_den = max(int(self.shared_expert_tokens), 1)
        report = {
            "total_tokens": int(self.total_tokens),
            "total_slots": int(self.total_slots),
            "expert_frequency": self.expert_frequency.copy(),
            # expert_proba is a true probability distribution over experts:
            # frequency (slot counts) divided by total_slots (total slots).
            "expert_proba": self.expert_frequency.astype(np.float64)
            / slot_denominator,
            "ean_sum": self.ean_sum.copy(),
            "ean_mean": (self.ean_sum / count_denominator).astype(np.float32),
            "weighted_ean_sum": self.weighted_ean_sum.copy(),
            "weighted_expert_frequency_sum": (
                self.weighted_expert_frequency_sum.copy()
            ),
            "reap": (self.weighted_ean_sum / count_denominator).astype(np.float32),
            "max_activations": self.max_activations.copy(),
            # Shared experts are never pruned; these are diagnostic only.
            "shared_expert_ean_sum": float(self.shared_expert_ean_sum),
            "shared_expert_ean_mean": float(self.shared_expert_ean_sum / shared_den),
            "shared_expert_tokens": int(self.shared_expert_tokens),
        }
        if self.pairwise_expert_frequency is not None:
            report["pairwise_expert_frequency"] = (
                self.pairwise_expert_frequency.copy()
            )
        return report

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
        return np.max(np.abs(outputs_array), axis=-1).astype(np.float32, copy=False)

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
