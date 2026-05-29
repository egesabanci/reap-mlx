import torch
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from torch import nn


def restricted_hierarchical_clustering(
    distances: torch.Tensor,
    method: str,
    n_clusters: int,
    max_cluster_size: int,
):
    """
    Performs hierarchical clustering with a maximum cluster size constraint.

    Will return cluster assignments into n_clusters where the maximum size of any
    cluster is max_cluster_size.

    Args:
        distances (torch.Tensor): A square tensor of pairwise distances.
        method (str): The linkage algorithm to use.
        n_clusters (int): The desired number of clusters.
        max_cluster_size (int): The maximum number of points in a cluster.

    Returns:
        np.ndarray: An array of cluster labels.
    """
    n_samples = distances.shape[0]
    final_labels = torch.arange(n_samples, dtype=torch.int)
    next_cluster_id = n_samples
    distances = distances.fill_diagonal_(float("inf"))
    cluster_sizes = torch.full((n_samples,), 1.0, dtype=torch.float)

    while len(torch.unique(final_labels)) > n_clusters:
        values, idx = torch.sort(distances.flatten(), descending=False, dim=-1)
        valid_merge = False
        for i, next_merge_min_idx in enumerate(idx):
            if values[i] == float("inf"):
                raise ValueError(
                    "No valid merges found. Check your parameters or data."
                )

            row_idx = (next_merge_min_idx // n_samples).item()
            col_idx = (next_merge_min_idx % n_samples).item()
            proposed_cluster_idx = min(row_idx, col_idx)
            other_cluster_idx = max(row_idx, col_idx)

            proposed_cluster_merge_size = cluster_sizes[
                [proposed_cluster_idx, other_cluster_idx]
            ].sum()
            if proposed_cluster_merge_size > max_cluster_size:
                continue  # Skip if the proposed merge exceeds max cluster size
            else:
                valid_merge = True
                break

        if not valid_merge:
            raise ValueError("No valid merges found. Check your parameters or data.")

        if valid_merge:
            final_labels[final_labels == other_cluster_idx] = proposed_cluster_idx
            cluster_sizes[proposed_cluster_idx] = (
                cluster_sizes[proposed_cluster_idx] + cluster_sizes[other_cluster_idx]
            )
            cluster_sizes[other_cluster_idx] = float("inf")
            next_cluster_id += 1
            # linkage update
            if method == "average":
                new_distances = (
                    distances[proposed_cluster_idx, :] + distances[other_cluster_idx, :]
                ) / 2
                distances[proposed_cluster_idx, :] = new_distances
                distances[:, proposed_cluster_idx] = new_distances
            else:
                raise NotImplementedError(f"Linkage method {method} not implemented")

            # prune larger index after update
            distances[other_cluster_idx, :] = float("inf")
            distances[:, other_cluster_idx] = float("inf")
    # make contigious
    contigious_labels = final_labels.clone()
    for new_cluster_id, label in enumerate(torch.unique(final_labels)):
        contigious_labels[final_labels == label] = new_cluster_id
    return contigious_labels.numpy()
