from typing import Any
from abc import ABC, abstractmethod
import copy
import gc

import torch
from torch import nn
import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment

import logging

logger = logging.getLogger(__name__)


def _weight_match_dist(a, dom):
    return torch.cdist(a, dom, p=2) ** 2


def assert_invariance(permuted_expert, orig_expert, model_attrs):
    up_proj = getattr(permuted_expert, model_attrs["up_proj"])
    inp = torch.rand(
        (1, up_proj.weight.shape[1]),
        dtype=up_proj.weight.dtype,
        device=up_proj.weight.device,
    )
    out1 = permuted_expert(inp)
    out2 = orig_expert(inp)
    if not torch.allclose(out1, out2, atol=1e-2):
        logger.warning(
            "Output of permuted expert should match original expert. "
            "Sum(abs(out1 - out2)) = {}".format(torch.sum(torch.abs(out1 - out2)))
        )


def assert_improved_weight_dist(permuted_expert, orig_expert, dom_expert, model_attrs):
    orig_dist = 0
    permuted_dist = 0
    for attr in ["up_proj", "gate_proj", "down_proj"]:
        permuted_weight = getattr(permuted_expert, model_attrs[attr]).weight
        orig_weight = getattr(orig_expert, model_attrs[attr]).weight
        dom_weight = getattr(dom_expert, model_attrs[attr]).weight
        orig_dist += _weight_match_dist(orig_weight, dom_weight).sum().item()
        permuted_dist += _weight_match_dist(permuted_weight, dom_weight).sum().item()
    if not permuted_dist < orig_dist:
        logger.warning(
            "Permuted expert should have a lower distance to the original expert than the dominant expert. ({}) > ({})".format(
                permuted_dist, orig_dist
            )
        )
    return permuted_dist, orig_dist


def assert_not_equal(permuted_expert, orig_expert, model_attrs):
    for attr in ["up_proj", "gate_proj", "down_proj"]:
        permuted_weight = getattr(permuted_expert, model_attrs[attr]).weight
        orig_weight = getattr(orig_expert, model_attrs[attr]).weight
        if torch.equal(permuted_weight, orig_weight):
            logger.warning(
                f"Permuted expert's {attr} weights should not be equal to the original expert's weights."
            )


class ExpertPermuter(ABC):
    def __init__(self, model_attrs: dict[str, Any]):
        self.model_attrs = model_attrs
        self.fused = False
        if model_attrs.get("fused", False):
            self.fused = True

    @torch.no_grad()
    def permute(
        self, experts: list[nn.Module], expert_indices: list[int], dom_expert_idx: int
    ):
        if self.fused:
            self._fused_permute(experts, expert_indices, dom_expert_idx)
        else:
            self._permute(experts, expert_indices, dom_expert_idx)

    @abstractmethod
    def _permute(
        self,
        experts: list[nn.Module],
        expert_indices: list[int],
        dom_expert_idx: int,
    ):
        """
        Abstract method to permute experts to a canonical order.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _fused_permute(
        self,
        experts: nn.Module,
        expert_indices: list[int],
        dom_expert_idx: int,
    ):
        """
        Abstract method to permute experts in a fused model.
        Must be implemented by subclasses.
        """
        pass

    def _run_assertions(
        self, permuted_expert, orig_expert, cost_matrix_np, row_ind, col_ind
    ):
        try:
            assert_invariance(permuted_expert, orig_expert, self.model_attrs)
            assert_not_equal(permuted_expert, orig_expert, self.model_attrs)
            permuted_cost = cost_matrix_np[row_ind, col_ind].sum()
            original_cost = np.trace(cost_matrix_np)
            assert permuted_cost <= original_cost, (
                f"Permuted cost {permuted_cost} should be less than or equal to original cost {original_cost}."
            )
        except AssertionError as e:
            logger.warning(f"Assertion failed during permutation: {e}")
            print(e)
            pass


class WeightMatchingPermuter(ExpertPermuter):
    def _permute(
        self,
        experts: list[nn.Module],
        expert_indices: list[int],
        dom_expert_idx: int,
    ):
        """Permutes experts using weight matching."""
        for expert_idx in expert_indices:
            if expert_idx == dom_expert_idx:
                continue
            expert = experts[expert_idx]
            orig_expert = copy.deepcopy(expert)
            cost_matrix = self._expert_cost_matrix(
                expert, experts[dom_expert_idx], self.model_attrs
            )
            cost_matrix_np = cost_matrix.cpu().to(torch.float16).numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
            permutation = torch.tensor(col_ind, dtype=torch.long).argsort()
            self.apply_permutation(expert, permutation, self.model_attrs)
            logger.debug("Checking expert %d...", expert_idx)
            self._run_assertions(expert, orig_expert, cost_matrix_np, row_ind, col_ind)

    def _expert_cost_matrix(self, expert, dominant_expert, model_attrs):
        """
        Computes the L1 distance between two experts.
        """
        up = _weight_match_dist(
            getattr(expert, model_attrs["up_proj"]).weight,
            getattr(dominant_expert, model_attrs["up_proj"]).weight,
        )
        gate = _weight_match_dist(
            getattr(expert, model_attrs["gate_proj"]).weight,
            getattr(dominant_expert, model_attrs["gate_proj"]).weight,
        )
        down = _weight_match_dist(
            getattr(expert, model_attrs["down_proj"]).weight.T,
            getattr(dominant_expert, model_attrs["down_proj"]).weight.T,
        )
        return up + gate + down

    def apply_permutation(self, expert, permutation, model_attrs):
        """
        Applies a permutation to the weights of an expert.
        """
        up_proj = getattr(expert, model_attrs["up_proj"])
        gate_proj = getattr(expert, model_attrs["gate_proj"])
        down_proj = getattr(expert, model_attrs["down_proj"])

        up_proj.weight.data = up_proj.weight[permutation]
        gate_proj.weight.data = gate_proj.weight[permutation]
        down_proj.weight.data = down_proj.weight[:, permutation]

    def _fused_permute(
        self,
        experts: nn.Module,
        expert_indices: list[int],
        dom_expert_idx: int,
    ):
        """Permutes experts in a fused model using weight matching.
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))
        """
        if len(expert_indices) == 1:
            return  # No permutation needed if only one expert
        orig_experts = copy.deepcopy(experts).cpu()
        up_gate_proj_param = getattr(experts, self.model_attrs["up_proj"])
        down_proj_param = getattr(experts, self.model_attrs["down_proj"])
        device = up_gate_proj_param.device

        up_gate_proj = up_gate_proj_param.data.cpu()
        down_proj = down_proj_param.data.cpu()

        expert_dim = down_proj.shape[1]

        dom_gate = up_gate_proj[dom_expert_idx, :, :expert_dim]
        dom_up = up_gate_proj[dom_expert_idx, :, expert_dim:]
        dom_down = down_proj[dom_expert_idx]

        for expert_idx in expert_indices:
            if expert_idx == dom_expert_idx:
                continue
            this_gate_proj = up_gate_proj[expert_idx, :, :expert_dim]
            this_up_proj = up_gate_proj[expert_idx, :, expert_dim:]
            this_down_proj = down_proj[expert_idx]

            up_cost = _weight_match_dist(
                this_up_proj.T,
                dom_up.T,
            )
            gate_cost = _weight_match_dist(
                this_gate_proj.T,
                dom_gate.T,
            )
            down_cost = _weight_match_dist(this_down_proj, dom_down)
            cost_matrix = up_cost + gate_cost + down_cost
            del up_cost, gate_cost, down_cost
            cost_matrix_np = cost_matrix.to(torch.float16).numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
            permutation = torch.tensor(col_ind, dtype=torch.long).argsort()

            up_gate_proj[expert_idx, :, :expert_dim] = this_gate_proj[:, permutation]
            up_gate_proj[expert_idx, :, expert_dim:] = this_up_proj[:, permutation]
            down_proj[expert_idx, :] = this_down_proj[permutation, :]
            del this_down_proj, this_gate_proj, this_up_proj
            gc.collect()
            torch.cuda.empty_cache()

        up_gate_proj_param.data = up_gate_proj.to(device)
        down_proj_param.data = down_proj.to(device)

        # Check permutation invariance and weights changed
        input = torch.rand(
            (up_gate_proj.shape[0], up_gate_proj.shape[1]),
            dtype=up_gate_proj.dtype,
            device=device,
        )
        orig_out = orig_experts.to(device)(input)
        permuted_out = experts(input)
        if not torch.allclose(orig_out, permuted_out, atol=1e-2):
            logger.warning(
                "Output of permuted experts should match original expert. "
                "Sum(abs(out1 - out2)) = {}".format(
                    torch.sum(torch.abs(orig_out - permuted_out))
                )
            )
        del input
        del orig_out
        del permuted_out
        
        if torch.allclose(
            orig_experts.gate_up_proj.cpu(), experts.gate_up_proj.cpu()
        ) or torch.allclose(orig_experts.down_proj.cpu(), experts.down_proj.cpu()):
            logger.warning(
                "Permuted experts' weights should not be equal to the original experts'"
                " weights."
            )


class DirectAlignmentPermuter(ExpertPermuter):
    def _permute(
        self,
        experts: list[nn.Module],
        dom_expert_idx: int,
    ):
        for expert in experts:
            if expert is not experts[dom_expert_idx]:
                cost_matrix = self._expert_cost_matrix(
                    expert, experts[dom_expert_idx], self.model_attrs
                )
                cost_matrix_np = cost_matrix.cpu().to(torch.float16).numpy()
                row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
                permutation = torch.tensor(col_ind, dtype=torch.long)
                self.apply_permutation_direct_alignment(
                    expert, permutation, self.model_attrs
                )

    def _l2_dist(self, a, dom):
        return torch.cdist(a, dom, p=2) ** 2

    def _expert_cost_matrix(self, expert, dominant_expert, model_attrs):
        """
        Computes the L1 distance between two experts.
        """
        up = self._l2_dist(
            getattr(expert, model_attrs["up_proj"]).weight.T,
            getattr(dominant_expert, model_attrs["up_proj"]).weight.T,
        )
        gate = self._l2_dist(
            getattr(expert, model_attrs["gate_proj"]).weight.T,
            getattr(dominant_expert, model_attrs["gate_proj"]).weight.T,
        )
        down = self._l2_dist(
            getattr(expert, model_attrs["down_proj"]).weight,
            getattr(dominant_expert, model_attrs["down_proj"]).weight,
        )
        return up + gate + down

    def apply_permutation_direct_alignment(self, expert, permutation, model_attrs):
        """
        Applies a permutation to the weights of an expert.
        """
        up_proj = getattr(expert, model_attrs["up_proj"])
        gate_proj = getattr(expert, model_attrs["gate_proj"])
        down_proj = getattr(expert, model_attrs["down_proj"])

        up_proj.weight.data = up_proj.weight.data[:, permutation]
        gate_proj.weight.data = gate_proj.weight.data[:, permutation]
        down_proj.weight.data = down_proj.weight.data[permutation, :]


PERMUTER_REGISTRY = {
    "wm": WeightMatchingPermuter,
    "direct": DirectAlignmentPermuter,
}
