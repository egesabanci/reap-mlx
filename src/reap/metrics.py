import numpy as np
import torch
import torch.nn.functional as F
import math  # For pi constant in comments
from typing import List, Dict, Callable
import logging


logger = logging.getLogger(__name__)

# --- Standalone Distance Functions ---

CHUNK_SIZE=16 # Chunk size for distance calculations to avoid OOM

def angular_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the normalized angular distance between two tensors of vectors.
    The distance is the angle in radians / pi, calculated as acos(cosine_similarity).
    Range: [0, 1]

    Args:
        x: Tensor of shape (..., N, D)
        y: Tensor of shape (..., M, D)

    Returns:
        A tensor of normalized pairwise distances of shape (..., N, M).
    """
    # Since this is used for online metrics, we chunk to avoid OOM
    chunks = max(1, int(x.shape[0] // CHUNK_SIZE))
    similarities = []
    for x_chunk, y_chunk in zip(x.chunk(chunks, dim=0), y.chunk(chunks, dim=0)):
        similarities.append(F.cosine_similarity(x_chunk, y_chunk, dim=-1))
    similarity = torch.cat(similarities, dim=0)

    # Clamp similarity to the valid range [-1, 1] to avoid NaNs from acos
    # due to floating-point inaccuracies.
    clamped_similarity = torch.clamp(similarity, -1.0, 1.0)

    # Calculate the angle. The result is in radians.
    angle = torch.acos(clamped_similarity)
    return angle / math.pi  # Normalize to [0, 1] range


def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes 1 - Cosine Similarity between two tensors of vectors.
    Range: [0, 2]
    """
    # Since this is used for online metrics, we chunk to avoid OOM
    chunks = max(1, int(x.shape[0] // CHUNK_SIZE))
    similarities = []
    for x_chunk, y_chunk in zip(x.chunk(chunks, dim=0), y.chunk(chunks, dim=0)):
        similarities.append(F.cosine_similarity(x_chunk, y_chunk, dim=-1))
    similarity = torch.cat(similarities, dim=0)
    return 1.0 - similarity


def cka_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes 1 - Centered Kernel Alignment (CKA) using a linear kernel.
    """
    x_centered = x - x.mean(dim=-1, keepdim=True)
    y_centered = y - y.mean(dim=-1, keepdim=True)
    cka_similarity = F.cosine_similarity(x_centered, y_centered, dim=-1)
    return 1.0 - cka_similarity


def js_divergence(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the Jensen-Shannon Divergence. Assumes inputs are logits
    and applies softmax to create probability distributions.
    """
    # Add a small epsilon for numerical stability with log
    epsilon = 1e-10
    x_dist = F.softmax(x, dim=-1)
    y_dist = F.softmax(y, dim=-1)

    m_dist = 0.5 * (x_dist + y_dist)

    kl_x_m = F.kl_div((m_dist + epsilon).log(), x_dist, reduction="none").sum(dim=-1)
    kl_y_m = F.kl_div((m_dist + epsilon).log(), y_dist, reduction="none").sum(dim=-1)

    jsd = 0.5 * (kl_x_m + kl_y_m)
    return jsd


def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the Euclidean distance between two vectors.
    The distance is computed as the L2 norm of the difference between the vectors.
    Range: [0, inf)

    Args:
        x: Tensor of shape (..., M, D)
        y: Tensor of shape (..., M, D)

    Returns:
        A tensor of pairwise distances in shape (..., N, M)
    """
    return torch.norm(x - y, dim=-1)


distance_metrics = {
    "angular": angular_distance,
    "euclidean": euclidean_distance,
    "jsd": js_divergence,
    "cka": cka_distance,
    "cosine": cosine_distance,
}
get_distance_fn = lambda metric: distance_metrics.get(metric)


def ttm_online(
    activations: torch.Tensor,  # (num_experts, seq, hidden_dim)
    selected: List[torch.Tensor],
    distance_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_experts: int,
    pairwise_expert_frequency: torch.Tensor,
) -> np.ndarray:
    """
    Calculates a pairwise N x N distance matrix using a vectorized approach.
    """
    device = activations.device
    pairwise_distances = torch.zeros(
        (num_experts, num_experts), device=device, dtype=activations[0].dtype
    )

    E, S, H = activations.shape
    K = selected.shape[1]
    act_tensor_permuted = activations.permute(1, 0, 2)  # S, E, H

    selected_acts = torch.gather(
        act_tensor_permuted, 1, selected.unsqueeze(-1).expand(-1, -1, H)
    )

    dist_matrix = distance_callable(
        selected_acts.unsqueeze(2), act_tensor_permuted.unsqueeze(1)
    )  # Shape: (S, K, E)

    # Vectorized accumulation using scatter_add_
    # Create an index for the 'selected expert' dimension
    idx0 = selected.view(S * K)

    # Flatten the distance matrix to match the indices
    flat_dists = dist_matrix.view(S * K, E)

    # For each of the S*K selections, we have E distances.
    # We need to add these E distances to the correct row in the temp matrix.
    # The row index is given by the selected expert index.
    pairwise_distances.scatter_add_(0, idx0.unsqueeze(-1).expand(-1, E), flat_dists)

    # Symmetrize the matrices, add i,j to j,i
    pairwise_distances = pairwise_distances + pairwise_distances.T

    # normalize by the sum of tokens router to either i or j
    pairwise_distances = pairwise_distances / pairwise_expert_frequency
    pairwise_distances = pairwise_distances.nan_to_num(0)  # Replace NaNs with 0

    # Ensure the diagonal is zero
    pairwise_distances.fill_diagonal_(0)

    return pairwise_distances


def ca_dist_online(
    activations: torch.Tensor,  # (seq, num_experts, hidden_dim)
    distance_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
):
    # (total_seq_len, num_experts, hidden_dim)
    permuted_activations = activations.permute(1, 0, 2)  
    # we will chunk along seq len dim. 
    distance_matrix = distance_callable(
        permuted_activations.unsqueeze(1), permuted_activations.unsqueeze(2)
    ).mean(dim=0)
    return distance_matrix  # (num_experts, num_experts)


def get_routed_characteristic_activation(
    activations: torch.Tensor,
    selected_experts: torch.Tensor,
    expert_frequency: torch.Tensor,
    device: torch.device,
    hidden_dim: int,
    num_experts: int,
) -> torch.Tensor:
    # seq, n_expert, hidden
    activations_permuted = activations.permute(1, 0, 2)

    # Shape: (seq, K, 1) -> (seq, K, hidden)
    index_for_gather = selected_experts.unsqueeze(-1).expand(-1, -1, hidden_dim)

    # Shape: (seq, K, hidden)
    gathered_activations = activations_permuted.gather(dim=1, index=index_for_gather)

    # Flatten the gathered activations and the expert indices.
    # src shape: (seq * K, hidden)
    src = gathered_activations.reshape(-1, hidden_dim)
    # index shape: (seq * K, 1) -> (seq * K, hidden)
    index_for_scatter = selected_experts.reshape(-1, 1).expand(-1, hidden_dim)

    # We are scattering along dim=0 (the n_experts dimension).
    characteristic_activation = torch.zeros(
        num_experts, hidden_dim, dtype=torch.float64, device=device
    )
    characteristic_activation = characteristic_activation.scatter_add_(
        dim=0, index=index_for_scatter, src=src.to(torch.float64)
    )
    # Normalize by the expert frequency
    characteristic_activation = characteristic_activation / expert_frequency.unsqueeze(
        -1
    )
    characteristic_activation = characteristic_activation.nan_to_num(
        0
    )  # Replace NaNs with 0
    return characteristic_activation


class OnlineStatsTracker:
    """
    A numerically stable tracker for online mean and variance using Welford's algorithm
    and Kahan summation for updating the mean.

    This is designed to handle a large number of updates without significant precision loss.
    """

    def __init__(
        self,
        shape: tuple,
        count_shape: tuple | int = 1,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initializes the tracker.

        Args:
            shape (tuple): The shape of the tensor statistic being tracked (e.g., (num_experts, hidden_dim)).
            device (torch.device): The device to store the state on.
        """
        self.shape = shape
        self.count_shape = count_shape
        self.device = device
        self.dtype = dtype

        # Welford's algorithm state
        self.count = torch.zeros(
            count_shape, dtype=torch.long, device=self.device, requires_grad=False
        )
        self.mean = torch.zeros(
            shape, dtype=dtype, device=self.device, requires_grad=False
        )

        # Kahan summation compensation for the mean
        self.mean_compensation = torch.zeros(
            shape, dtype=dtype, device=self.device, requires_grad=False
        )

    def update(self, new_mean: torch.Tensor, new_count: int | torch.Tensor):
        """
        Update the statistics with a new batch of data.

        Args:
            new_mean (torch.Tensor): A tensor of new data to update state.
            new_count (int | torch.Tensor): The count of new data points in the batch to
                normalize the mean entires with.
        """
        new_count = new_count.to(self.device, torch.long)
        new_mean = new_mean.to(self.device, dtype=self.dtype)

        # Welford's algorithm
        updated_count = self.count + new_count
        delta = new_mean - self.mean

        # Kahan Summation
        # `y` is the new term to add to the mean, corrected by the old compensation.
        y = delta * new_count / updated_count
        y = y.nan_to_num(0)  # Replace NaNs with 0 in case of updated_count being zero
        y = y - self.mean_compensation
        # `t` is the new provisional mean.
        t = self.mean + y
        # `self.mean_compensation` is the new error (the part that was lost).
        self.mean_compensation = (t - self.mean) - y
        # `self.mean` is the new, more accurate mean.
        self.mean = t
        # End Kahan Summation
        self.count = updated_count
