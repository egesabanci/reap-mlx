"""
Module for merging experts in Mixture of Experts (MoE) layers based on pre-defined clusters.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Union, Any, Callable
from enum import Enum
import torch
import torch.nn as nn

from reap.model_util import MODEL_ATTRS
from reap.permute import (
    DirectAlignmentPermuter,
    assert_improved_weight_dist,
    assert_not_equal,
    assert_invariance,
    PERMUTER_REGISTRY,
)

FP32_EPS = torch.finfo(torch.float32).eps

logger = logging.getLogger(__name__)


class MergeMethod(str, Enum):
    """Available methods for merging experts."""

    FREQUENCY_WEIGHTED_AVERAGE = "frequency_weighted_average"
    TIES = "ties"
    MULTISLERP = "multislerp"
    SCE = "sce"
    KARCHER = "karcher"
    SUBMOE = "submoe"


class MoEExpertMerger:
    """
    Class for merging experts in MoE layers based on pre-defined clusters.
    """

    def __init__(
        self,
        moe: nn.Module,
        cluster_label: torch.Tensor,
        expert_proba: torch.Tensor,
        model_attrs: Dict[str, Any],
        merge_method: Union[str, MergeMethod] = MergeMethod.FREQUENCY_WEIGHTED_AVERAGE,
        dom_as_base: bool = True,
        permute: str | None = None,
        tie_tensors: bool = False,
        **merge_kwargs,
    ):
        """
        Initialize the expert merger with pre-defined clusters.

        Args:
            cluster_label: Tensor of clusters assignments with each tensor element
                corresponding to an expert in the uncompressed model
            expert_proba: Tensor of expert probabilities
            model_attrs: Dictionary of attributes for the MoE layer
            merge_method: Method to use for merging experts
            dom_as_base: If True, the dominant expert is used as the base for merging with multi-SLERP.
            merge_kwargs: Additional keyword arguments for the merge method
        """
        self.moe = moe
        self.cluster_label = cluster_label
        self.expert_proba = expert_proba
        self.model_attrs = model_attrs
        self.fused = False
        if model_attrs.get("fused", False):
            self.fused = True
        self.dom_as_base = dom_as_base
        self.permute = permute
        self.tie_tensors = tie_tensors
        self.merge_method = (
            MergeMethod(merge_method) if isinstance(merge_method, str) else merge_method
        )
        self.merge_kwargs = merge_kwargs or {}

    @torch.no_grad()
    def merge_experts(
        self,
    ) -> None:
        """
        Merge moe experts in-place based on the pre-defined clusters.
        """
        merge_fn = self._get_merge_function()
        # Process each cluster
        for cluster_id in self.cluster_label.unique():
            experts = getattr(self.moe, self.model_attrs["experts"])
            expert_indices = torch.where(self.cluster_label == cluster_id)[0].tolist()
            if len(expert_indices) == 1:
                continue
            logger.debug(
                f"Processing cluster {cluster_id} with experts {expert_indices}"
            )

            dom_expert = self._get_dominant_expert(
                cluster_id, self.cluster_label, self.expert_proba
            )
            non_dom_indices = [i for i in expert_indices if i != dom_expert]
            if len(non_dom_indices) == 0:
                logger.debug(
                    f"Only one expert in cluster {cluster_id}, skipping merge."
                )
                continue

            logger.debug(f"Dominant expert for cluster {cluster_id} is {dom_expert}")

            # If permute is enabled, apply permutation to align experts
            if self.permute:
                permuter_cls = PERMUTER_REGISTRY.get(self.permute)
                logger.info(
                    f"Permuting experts in cluster {cluster_id} with {self.permute} method"
                )
                permuter = permuter_cls(self.model_attrs)
                permuter.permute(
                    experts,
                    expert_indices,
                    dom_expert_idx=dom_expert,
                )

            if not self.fused:
                dom_tensors = self._get_tensors(experts[dom_expert])
                other_tensors = [
                    self._get_tensors(experts[idx]) for idx in non_dom_indices
                ]
            else:
                dom_tensors = self._get_tensors_fused(experts, dom_expert)
                other_tensors = [
                    self._get_tensors_fused(experts, idx) for idx in non_dom_indices
                ]

            # Group corresponding parameters across experts
            for param_name in dom_tensors:
                if self.dom_as_base and self.merge_method in [MergeMethod.TIES, MergeMethod.MULTISLERP, MergeMethod.SCE, MergeMethod.KARCHER]:
                    base = dom_tensors[param_name]
                    tensors_to_merge = [t[param_name] for t in other_tensors]
                    tensor_weights = self.expert_proba[non_dom_indices]
                else:
                    base = None
                    tensors_to_merge = [dom_tensors[param_name]] + [
                        t[param_name] for t in other_tensors
                    ]
                    tensor_weights = torch.concat(
                        (
                            self.expert_proba[dom_expert].unsqueeze(dim=0),
                            self.expert_proba[non_dom_indices],
                        )
                    )

                # Merge the tensors using the selected method
                merged_tensor = merge_fn(
                    tensors=tensors_to_merge,
                    tensor_weights=tensor_weights,
                    base_tensor=base,
                    **self.merge_kwargs,
                )
                if self.tie_tensors:
                    if self.fused:
                        raise NotImplementedError(
                            "Tying tensors is not supported for fused experts."
                        )
                    self._tie_tensors(experts, expert_indices, dom_expert, param_name, merged_tensor)
                    continue
                # copy experts
                for expert_idx in expert_indices:
                    if not self.fused:
                        self._set_tensor(experts[expert_idx], param_name, merged_tensor)
                    else:
                        # For fused experts, set the merged tensor directly
                        experts = getattr(self.moe, self.model_attrs["experts"])
                        getattr(
                            experts, self.model_attrs[param_name]
                        ).data[expert_idx] = merged_tensor

    def _get_merge_function(self) -> Callable:
        """Get the appropriate merge function based on the selected method."""
        if self.merge_method == MergeMethod.FREQUENCY_WEIGHTED_AVERAGE:
            return self.frequency_weighted_average_merge
        elif self.merge_method == MergeMethod.TIES:
            return self._ties_merge
        elif self.merge_method == MergeMethod.MULTISLERP:
            return self._multislerp_merge
        elif self.merge_method == MergeMethod.SCE:
            return sce_merge
        elif self.merge_method == MergeMethod.KARCHER:
            return karcher_merge_tensors
        elif self.merge_method == MergeMethod.SUBMOE:
            return submoe

        else:
            raise NotImplementedError(f"Unknown merge method {self.merge_method}")

    @staticmethod
    def frequency_weighted_average_merge(
        tensors: List[torch.Tensor],
        tensor_weights: torch.Tensor | None = None,
        **kwargs,
    ) -> nn.Module:
        if getattr(kwargs, "base_tensor", None) is not None:
            raise ValueError(
                "The frequency weighted average merge does not support a base tensor."
            )
        if tensor_weights is None:
            tensor_weights = torch.tensor([1.0 / len(tensors)] * len(tensors))

        tensor_list = torch.stack(
            [t * tensor_weights[idx] for idx, t in enumerate(tensors)], dim=0
        )
        merged_tensor = torch.sum(tensor_list, dim=0) / (
            torch.sum(tensor_weights, dim=0) + FP32_EPS
        )
        return merged_tensor

    @staticmethod
    def _ties_merge(
        tensors: List[torch.Tensor],
        tensor_weights: torch.Tensor | None = None,
        *,
        base_tensor: torch.Tensor | None = None,
        select_top_k: float = 0.1,
        scaling: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        tvs = [base_tensor - t for t in tensors]

        # mag prune
        if select_top_k < 1:
            top_k = int(len(base_tensor.flatten()) * select_top_k)
            for idx, tv in enumerate(tvs):
                keep_idx = torch.topk(
                    tv.abs().flatten(), k=top_k, largest=True, sorted=False
                ).indices
                tv_pruned = torch.where(
                    torch.zeros_like(tv.flatten(), dtype=torch.bool).scatter(
                        dim=0, index=keep_idx, value=True
                    ),
                    tv.flatten(),
                    torch.zeros_like(tv.flatten(), dtype=tv.dtype),
                )
                tvs[idx] = tv_pruned.reshape(tv.shape)

        # re-weight TVs
        if tensor_weights is None:
            # uniform weights
            tensor_weights = torch.ones(
                len(tvs), dtype=tvs.dtype, device=tvs.device
            ) / len(tvs)
        for idx, tv in enumerate(tvs):
            tvs[idx] = (tv * tensor_weights[idx]) / tensor_weights.sum(dim=0)

        # sign consensus
        tvs = torch.stack(tvs, dim=0)
        mask = get_sign_mask(tvs, method="sum")
        delta = (tvs * mask).sum(dim=0)

        # merge
        return (base_tensor + delta * scaling).to(base_tensor.dtype)

    @staticmethod
    def _multislerp_merge(
        tensors: List[torch.Tensor],
        tensor_weights: torch.Tensor | None = None,
        *,
        base_tensor: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Merge tensors using multislerp."""
        if tensor_weights is None:
            tensor_weights = [1.0 / len(tensors)] * len(tensors)

        merged_weight = multislerp(
            tensors,
            weight=tensor_weights,
            base_tensor=base_tensor,
            normalize_weights=True,
        )

        return merged_weight

    def _get_dominant_expert(
        self, cluster_idx: int, cluster_labels: torch.Tensor, expert_proba: torch.Tensor
    ) -> int:
        """Get the dominant expert for a given cluster index based on frequency.

        Args:
            cluster_idx: Index of the cluster to find the dominant expert for.
            cluster_labels: Tensor containing expert labels for each expert.
            expert_freq: Tensor containing frequency of each expert.

        Returns:
            The index of the dominant expert in the specified cluster.
        """
        # Get experts in this cluster
        experts_in_cluster = torch.where(cluster_labels == cluster_idx)[0]

        # Get probalities for these experts
        proba = expert_proba[experts_in_cluster]

        # Return the index of the expert with maximum frequency
        return experts_in_cluster[torch.argmax(proba)].item()

    def _get_tensors(self, expert: nn.Module) -> dict[str, torch.Tensor]:
        """Get weight tensors from the dominant expert and other experts."""
        w1_name = self.model_attrs["up_proj"]
        w2_name = self.model_attrs["down_proj"]
        w3_name = self.model_attrs["gate_proj"]
        return {
            "up_proj": getattr(expert, w1_name).weight.data,
            "down_proj": getattr(expert, w2_name).weight.data,
            "gate_proj": getattr(expert, w3_name).weight.data,
        }

    def _get_tensors_fused(
        self, experts: nn.Module, expert_index: int
    ) -> Dict[str, torch.Tensor]:
        """Get weight tensors from fused experts"""
        return {
            # will contain up_gate_proj
            "up_proj": getattr(experts, self.model_attrs["up_proj"]).data[expert_index],
            "down_proj": getattr(experts, self.model_attrs["down_proj"]).data[
                expert_index
            ],
        }

    def _set_tensor(
        self, expert: nn.Module, tensor_name: str, merged_tensor: torch.Tensor
    ) -> None:
        """Set weight tensor of expert"""
        attr_name = self.model_attrs[tensor_name]
        getattr(expert, attr_name).weight.data.copy_(merged_tensor)

    def _tie_tensors(
            self,
            experts: List[nn.Module],
            expert_indices: List[int],
            dom_expert: int,
            tensor_name: str,
            merged_tensor: torch.Tensor,
    ) -> None:
        """Set merged tensor as the dom tensor and tie weights of non-dom experts"""
        attr_name = self.model_attrs[tensor_name]
        # Update the dominant expert's weight with the merged tensor
        getattr(experts[dom_expert], attr_name).weight.data.copy_(merged_tensor)
        # Tie other experts' parameters to the dominant expert's
        for idx in expert_indices:
            if idx == dom_expert:
                continue
            expert = experts[idx]
            setattr(expert, attr_name, getattr(experts[dom_expert], attr_name))
            # if not hasattr(expert, "_dynamic_tied_weights_keys"):
            #     expert._dynamic_tied_weights_keys = []
            # if attr_name not in expert._dynamic_tied_weights_keys:
            #     expert._dynamic_tied_weights_keys.append(attr_name)


def get_sign_mask(
    delta: torch.Tensor,
    method: str = "sum",
):
    """Returns a mask determining which delta vectors should be merged
    into the final model.

    For the methodology described in the TIES paper use 'sum'. For a
    simpler naive count of signs, use 'count'."""
    mask_dtype = delta.dtype

    sign = delta.sign().to(mask_dtype)

    if method == "sum":
        sign_weight = delta.sum(dim=0)
        majority_sign = (sign_weight >= 0).to(mask_dtype) * 2 - 1
        del sign_weight
    elif method == "count":
        majority_sign = (sign.sum(dim=0) >= 0).to(mask_dtype) * 2 - 1
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')

    return sign == majority_sign


def multislerp(
    tensors: List[torch.Tensor],
    weight: List[float],
    base_tensor: Optional[torch.Tensor] = None,
    normalize_weights: bool = True,
    eps: float = 1e-8,
):
    """
    Implements barycentric interpolation on a hypersphere.

    The approach:
    1. Project points onto a tangent space at their weighted Euclidean mean.
    2. Perform the interpolation in the tangent space.
    3. Project the result back to the hypersphere.

    Limitations:
    - The weighted sum of the input tensors must not be zero.
    - The tensors must not be all parallel or antiparallel.

    Args:
        tensors: List of tensors to interpolate
        weight: List of weights for each tensor
        base_tensor: Optional tensor defining the origin of the hypersphere
        normalize_weights: If True, the weights will be normalized to sum to 1
        eps: Small constant for numerical stability
    """
    if len(tensors) == 1:
        # No interpolation needed
        return tensors[0]

    tensors = torch.stack(tensors, dim=0)
    if base_tensor is not None:
        tensors -= base_tensor

    tensors_flat = tensors.view(tensors.shape[0], -1)

    weights = weight.detach().clone().to(tensors.device)
    if normalize_weights:
        weights = weights / weights.sum()

    # Project to unit hypersphere
    norms = torch.norm(tensors_flat, dim=-1, keepdim=True)
    unit_tensors = tensors_flat / (norms + eps)

    mean = (unit_tensors * weights.view(-1, 1)).sum(0)
    mean_norm = torch.norm(mean)
    # print(mean_norm)
    if mean_norm < eps:
        if tensors.shape[0] == 2:
            # fallback to linear interpolation
            res = (tensors[0] * weights[0] + tensors[1] * weights[1]).view(
                tensors.shape[1:]
            )
            if base_tensor is not None:
                res = res + base_tensor
            return res
        raise ValueError(
            "The weighted sum of the input tensors is zero. This occurs when "
            "antipodal vectors or sets of vectors have weights that exactly "
            "balance out (e.g., vectors a,-a with equal weights). Try using "
            "different weights if you have antipodal vectors."
        )
    mean = mean / mean_norm

    # Project to tangent space
    dots = (unit_tensors * mean).sum(-1, keepdim=True)
    tangent_vectors = unit_tensors - dots * mean

    # Interpolate
    tangent_result = (tangent_vectors * weights.view(-1, 1)).sum(0)

    # Project back to sphere using exponential map
    tangent_norm = torch.norm(tangent_result) + eps
    result = mean * torch.cos(tangent_norm) + tangent_result * (
        torch.sin(tangent_norm) / tangent_norm
    )

    avg_norm = (norms.squeeze(-1) * weights).sum()
    result = result * avg_norm
    result = result.view(tensors.shape[1:])

    if base_tensor is not None:
        result = result + base_tensor

    return result


def sce_merge(
    tensors: List[torch.Tensor],
    base_tensor: torch.Tensor,
    int8_mask: bool = False,
    select_top_k: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    if not tensors:
        return base_tensor
    mask_dtype = torch.int8 if int8_mask else base_tensor.dtype
    task_vectors = torch.stack([t - base_tensor for t in tensors], dim=0)

    if select_top_k < 1:
        mask = sce_mask(task_vectors, select_top_k, mask_dtype)
        task_vectors = task_vectors * mask.unsqueeze(0)

    erase_mask = get_sign_mask(task_vectors, method="sum")

    tv_weights = sce_weight(task_vectors)
    while tv_weights.dim() < task_vectors.dim():
        tv_weights = tv_weights.unsqueeze(-1)

    erased_weights = tv_weights * erase_mask
    merged_tv = (task_vectors * erased_weights).sum(dim=0)
    final_tv = merged_tv / torch.sum(erased_weights, dim=0).clamp(min=1e-6)

    return (base_tensor + final_tv).squeeze(dim=0)


def sce_weight(tvs: torch.Tensor) -> torch.Tensor:
    weights = torch.mean(tvs**2, dim=list(range(1, tvs.dim())))
    weight_sum = torch.sum(weights).item()
    if abs(weight_sum) < 1e-6:
        return torch.ones_like(weights) / weights.shape[0]
    return weights / weight_sum


def sce_mask(
    tvs: torch.Tensor, density: float, mask_dtype: Optional[torch.dtype] = None
):
    if density <= 0:
        return torch.zeros_like(tvs, dtype=mask_dtype)
    if density >= 1:
        return torch.ones_like(tvs, dtype=mask_dtype)

    var = torch.var(tvs, dim=0, unbiased=False)
    nonzero = torch.count_nonzero(var)
    k = int(nonzero * density)
    if k == 0:
        return torch.zeros_like(tvs, dtype=mask_dtype)

    _, indices = torch.topk(var.abs().view(-1), k=k, largest=True)
    mask = torch.zeros_like(var, dtype=mask_dtype)
    mask.view(-1)[indices] = 1
    return mask


def karcher_merge_tensors(tensors, tensor_weights, max_iter=10, tol=1e-5, **kwargs):
    """
    Implements weight fusion based on the Riemannian (Karcher) mean concept.

    Args:
        tensors: List of tensors to merge
        tensor_weights: List of weights for each tensor
        max_iter: Maximum number of iterations for the Karcher mean algorithm
        tol: Convergence tolerance

    Returns:
        Merged tensor using Karcher mean algorithm
    """
    alphas = tensor_weights
    if len(tensors) == 1:
        return tensors[0]

    # Calculate norms and unit vectors
    norms = []
    units = []
    for t in tensors:
        t_float = t.float()
        n = torch.linalg.norm(t_float)
        n_val = n.item()
        if n_val == 0.0:
            norms.append(0.0)
            units.append(torch.zeros_like(t))
        else:
            norms.append(n_val)
            units.append((t / n).to(t.dtype))

    # Select non-zero weight vectors
    valid_indices = [i for i, n in enumerate(norms) if n > tol]
    if not valid_indices:
        return torch.zeros_like(tensors[0])

    valid_alphas = [alphas[i] for i in valid_indices]
    alpha_sum = sum(valid_alphas)
    normalized_alphas = [a / alpha_sum for a in valid_alphas]
    valid_units = [units[i] for i in valid_indices]

    # Initial guess: Normalized weighted arithmetic mean
    u = torch.zeros_like(valid_units[0])
    for a, ui in zip(normalized_alphas, valid_units):
        u += a * ui
    norm_u = torch.linalg.norm(u.float()).item()
    if norm_u < tol:
        u = valid_units[0].clone()
    else:
        u = (u / norm_u).to(u.dtype)

    # Iterative Karcher mean computation
    for _ in range(max_iter):
        T = torch.zeros_like(u)
        for a, ui in zip(normalized_alphas, valid_units):
            # Flatten tensor for dot product calculation
            dot = torch.clamp(torch.dot(u.flatten(), ui.flatten()), -1.0, 1.0)
            theta = torch.arccos(dot)
            theta_val = theta.item()
            if theta_val < tol:
                continue
            else:
                # Ensure tensor operations
                sin_theta = torch.sin(theta)
                T += a * (theta / sin_theta) * (ui - dot * u)

        # Convert norm_T to tensor
        norm_T = torch.linalg.norm(T.float())
        if norm_T.item() < tol:
            break

        # Use tensor for trigonometric calculations
        cos_norm_T = torch.cos(norm_T)
        sin_norm_T = torch.sin(norm_T)
        u = (cos_norm_T * u + sin_norm_T * (T / norm_T)).to(u.dtype)

        # Ensure u is a unit vector
        u_norm = torch.linalg.norm(u.float())
        if u_norm.item() > tol:
            u = (u / u_norm).to(u.dtype)

    # Global scale: Weighted sum of original tensor norms (including zero vectors)
    s = 0.0
    for a, n in zip(alphas, norms):
        s += a * n

    return s * u


def expert_weight_similarity(
    experts: List[nn.Module],
    model_attrs: Dict[str, Any],
    method: str = "cosine",
) -> torch.Tensor:
    up_proj_name = model_attrs["up_proj"]
    down_proj_name = model_attrs["down_proj"]
    gate_proj_name = model_attrs["gate_proj"]

    pairwise_similarity = torch.empty((len(experts), len(experts)))
    flattened_experts = [
        torch.stack(
            [
                getattr(expert, up_proj_name).weight.flatten(),
                getattr(expert, down_proj_name).weight.flatten(),
                getattr(expert, gate_proj_name).weight.flatten(),
            ]
        )
        for expert in experts
    ]
    for i in range(len(experts)):
        for j in range(len(experts)):
            if i == j:
                this_sim = float("inf")
            else:
                if method == "cosine":
                    this_sim = torch.nn.functional.cosine_similarity(
                        flattened_experts[i], flattened_experts[j]
                    )
                elif method == "euclidean":
                    this_sim = -torch.norm(flattened_experts[i] - flattened_experts[j])
                else:
                    raise ValueError(f"Unknown similarity method: {method}")
            pairwise_similarity[i, j] = this_sim
            pairwise_similarity[j, i] = this_sim
    return pairwise_similarity


@torch.no_grad()
def submoe(tensors, tensor_weights, **kwargs):
    if getattr(kwargs, "base_tensor", None) is not None:
        raise ValueError(
            "SubMoE merge does not support dom_as_base=True."
        )
    # Merge experts in aligned subspace
    # normalize tensor_weights
    tensor_weights = tensor_weights / torch.sum(tensor_weights)
    d_type = tensors[0].dtype
    concat_t = torch.cat(tensors, dim=1).to(torch.float32)
    U, S, V = torch.linalg.svd(concat_t, full_matrices=False)
    V = V.reshape(len(tensors), V.shape[0], -1)
    for i, v in enumerate(V):
        V[i] = v * tensor_weights[i]
    V_merged = torch.sum(V, dim=0)
    merged_t = (U @ torch.diag(S)) @ V_merged
    merged_t = merged_t.to(d_type)
    return merged_t
