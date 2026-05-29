from typing import List, Callable, Dict, Optional

import numpy as np
import torch
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.vq import kmeans2
import logging


def get_penalty_vector(
    expert_probablities: torch.Tensor,
    temperature: float | None = None,
) -> torch.Tensor:
    if temperature is not None and temperature != 0:
        penalty = torch.softmax(expert_probablities / temperature, dim=0)
    else:
        penalty = expert_probablities / expert_probablities.sum()
    return penalty


@torch.no_grad()
def dynamic_frequency_penalized_clustering(
    distances: torch.Tensor,
    expert_probablities: torch.Tensor,
    n_clusters: int,
    softmax_temperature: float | None = 1.0,
) -> torch.Tensor:
    """Frequency penalized hierarchical clustering.

    Args:
        distances (torch.Tensor): NxN matrix of pairwise distances between experts.
        expert_frequencies (torch.Tensor): N vector of frequencies for each expert.
        n_clusters (int): number of clusters to form.
        softmax_temperature (float | None): Temperature for softmax scaling of frequencies.
            If None, frequencies are normalizded by their sum.

    Returns:
        torch.Tensor: Cluster assignments for each expert.
    """
    # Copy distances
    distances = distances.clone()
    expert_probablities = expert_probablities.clone()

    device = distances.device
    num_experts = distances.shape[0]
    distances = distances.fill_diagonal_(
        float("inf")
    )  # Set diagonal to inf to avoid self-merging

    # Initialize clusters
    clusters = torch.tensor([i for i in range(num_experts)])

    # Perform clustering
    while len(torch.unique(clusters)) > n_clusters:
        penalty = get_penalty_vector(expert_probablities, softmax_temperature)
        penalty_matrix = penalty.unsqueeze(0) + penalty.unsqueeze(1)
        penalized_distances = distances * penalty_matrix
        penalized_distances[penalized_distances.isnan()] = float("inf")
        min_idx = torch.argmin(penalized_distances).item()
        i, j = (
            min_idx // penalized_distances.shape[0],
            min_idx % penalized_distances.shape[0],
        )
        if i > j:  # always merge the larger index into the smaller one
            i, j = j, i

        # We merge cluster j to cluster i, so other clusters to cluster j will be inf. (cluster j dissapears)
        # And the distance from cluster i to other clusters will be updated based on the linkage method.
        for k in range(distances.shape[0]):
            if k != i and k != j:  # skip the merged cluster
                new_dist = (
                    distances[i, k] * penalty[i] + distances[j, k] * penalty[j]
                ) / (penalty[i] + penalty[j])
                distances[i, k] = new_dist
                distances[k, i] = new_dist

        distances[i, i] = float("inf")
        distances[j, :] = float("inf")
        distances[:, j] = float("inf")
        expert_probablities[i] += expert_probablities[j]
        expert_probablities[j] = 0.0  # Mark the merged cluster's frequency as 0

        print(f"clusters: {len(torch.unique(clusters))}, merge ({i}, {j})")
        cj = clusters[j]
        # Merge cluster j to cluster i
        clusters[clusters == cj] = clusters[i]

    # Reassign cluster IDs to be contiguous
    clusters_ids = {c.item(): i for i, c in enumerate(torch.unique(clusters))}
    reindexed_clusters = [clusters_ids[c.item()] for c in clusters]
    return torch.tensor(reindexed_clusters, device=device)


def hierarchical_clustering(
    distances: torch.Tensor,
    method: str,
    n_clusters: int,
):
    condensed_dist = squareform(distances.fill_diagonal_(0))
    all_clusters = linkage(condensed_dist, method=method)
    cluster_labels = linkage_to_labels(all_clusters, n_clusters)
    return cluster_labels


def linkage_to_labels(linkage_matrix: np.ndarray, num_clusters: int) -> np.ndarray:
    """
    Converts a SciPy linkage matrix into a flat array of cluster labels.

    This function simulates the clustering process described by the linkage
    matrix until the desired number of clusters is reached. It then assigns
    a unique label to each of these final clusters.

    This is functionally similar to scipy.cluster.hierarchy.fcluster(Z, k, criterion='maxclust').

    Args:
        linkage_matrix (np.ndarray): The hierarchical clustering linkage matrix,
                                     e.g., from scipy.cluster.hierarchy.linkage.
        num_clusters (int): The desired number of final clusters.

    Returns:
        np.ndarray: An array of size n (number of original data points), where
                    the value at each index is the integer label (from 1 to
                    num_clusters) of the cluster the point belongs to.
    """
    # The number of original observations is the number of rows in the
    # linkage matrix plus one.
    n_samples = linkage_matrix.shape[0] + 1

    if num_clusters < 1 or num_clusters > n_samples:
        raise ValueError(
            f"Number of clusters must be between 1 and {n_samples} (the number of samples)."
        )

    # Initialize an array where each data point is in its own cluster,
    # labeled by its own index.
    # These labels will be updated as clusters are merged.
    labels = np.arange(n_samples)

    # We perform n - k merges to end up with k clusters.
    # The linkage_matrix has n-1 rows, so we iterate through the first n-k rows.
    num_merges = n_samples - num_clusters

    for i in range(num_merges):
        # Get the two clusters being merged in this step.
        # Indices < n_samples are original data points.
        # Indices >= n_samples are newly formed clusters from previous steps.
        cluster_1_id = int(linkage_matrix[i, 0])
        cluster_2_id = int(linkage_matrix[i, 1])

        # The new cluster's ID is n_samples + i.
        new_cluster_id = n_samples + i

        # Find all points belonging to the two clusters being merged and
        # assign them to the new cluster.
        mask = (labels == cluster_1_id) | (labels == cluster_2_id)
        labels[mask] = new_cluster_id

    # The labels are currently large, arbitrary numbers (e.g., 3, 12, 8).
    # We remap them to be consecutive integers from 1 to num_clusters.
    unique_final_labels = np.unique(labels)

    # Create a mapping from the old cluster IDs to the new ones (1, 2, ..., k)
    label_map = {old_label: i for i, old_label in enumerate(unique_final_labels)}

    # Apply the mapping to get the final, clean labels
    final_labels = np.array([label_map[label] for label in labels])

    # final_labels = labels
    return final_labels


def multi_layer_hierarchical_clustering(
    distances: dict[int, torch.Tensor],
    num_layers: int,
    method: str,
    n_clusters: int,
) -> dict[int, np.ndarray]:
    """
    Performs hierarchical clustering jointly across groups of consecutive layers.

    This method groups layers by `num_layers` and then greedily merges clusters
    with the minimum distance across all layers within a group until a target
    total number of clusters for that group is reached.

    Args:
        distances (dict[int, torch.Tensor]): A dictionary mapping layer index to distance matrices.
        num_layers (int): The number of consecutive layers to group for joint clustering.
        method (str): The linkage method to use (e.g., 'average', 'single').
        n_clusters (int): The target number of clusters per layer on average.

    Returns:
        dict[int, np.ndarray]: A dictionary mapping layer index to cluster labels.
    """
    all_cluster_labels = {}
    sorted_layer_indices = sorted(distances.keys())

    for layer_idx in sorted_layer_indices:
        # distances[layer_idx].fill_diagonal_(float("inf"))
        distances[layer_idx].fill_diagonal_(0.) # Set diagonal to 0 for squareform fn

    for i in range(0, len(sorted_layer_indices), num_layers):
        layer_indices_in_group = sorted_layer_indices[i : i + num_layers]
        num_layers_in_group = len(layer_indices_in_group)

        if num_layers_in_group == 0:
            continue

        if num_layers_in_group == 1 and num_layers > 1:
            logging.warning(
                f"The last layer group is a singleton (layer {layer_indices_in_group[0]}). "
                f"This can happen if the total number of layers is not divisible by num_layers."
            )

        group_distances = [distances[idx] for idx in layer_indices_in_group]
        num_experts_per_layer = [d.shape[0] for d in group_distances]
        total_experts = sum(num_experts_per_layer)
        target_total_clusters = n_clusters * num_layers_in_group

        linkage_matrices = [
            linkage(squareform(d.cpu().numpy().astype(np.float64)), method=method)
            for d in group_distances
        ]

        merge_distances = [Z[:, 2] for Z in linkage_matrices]
        merge_indices = [0] * num_layers_in_group
        num_merges_per_layer = [0] * num_layers_in_group
        num_merges_to_perform = total_experts - target_total_clusters

        for _ in range(num_merges_to_perform):
            min_dist = float("inf")
            best_layer_in_group = -1
            for j in range(num_layers_in_group):
                if merge_indices[j] < len(merge_distances[j]):
                    current_dist = merge_distances[j][merge_indices[j]]
                    if current_dist < min_dist:
                        min_dist = current_dist
                        best_layer_in_group = j

            if best_layer_in_group != -1:
                merge_indices[best_layer_in_group] += 1
                num_merges_per_layer[best_layer_in_group] += 1
            else:
                # No more merges possible
                break

        for j in range(num_layers_in_group):
            original_layer_idx = layer_indices_in_group[j]
            num_final_clusters = num_experts_per_layer[j] - num_merges_per_layer[j]
            labels = linkage_to_labels(linkage_matrices[j], num_final_clusters)
            all_cluster_labels[original_layer_idx] = torch.tensor(labels)

    return all_cluster_labels


def kmeans_clustering(
    data: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """
    Performs k-means clustering using scipy.cluster.vq.kmeans2.

    Args:
        data (np.ndarray): The data to cluster, with shape (n_samples, n_features).
        n_clusters (int): The number of clusters to form.

    Returns:
        np.ndarray: An array of cluster labels.
    """
    # The kmeans2 function returns the centroids and the labels.
    # We are interested in the labels.
    _, labels = kmeans2(data, n_clusters, minit="++")
    return labels


def mc_smoe_clustering(
    distances: dict[int, torch.Tensor],
    expert_proba: dict[int, torch.Tensor],
    total_clusters: int,
) -> dict[int, np.ndarray]:
    """
    Performs clustering using the MC-SMoE algorithm.

    Args:
        distances (dict[int, torch.Tensor]): Expert pairwise distances matrices per
            layer based on router logits. Tensor shape (num_expert, num_experts)
        expert_proba (dict[int, torch.Tensor]): A list of tensors containing the normalized
            expert probabilities for each layer. Tensor shape (num_experts, )
        total_clusters (int): The target number of clusters across all layers.

    Returns:
        dict[int, torch.Tensor]: A dictionary mapping layer index to cluster labels.
            For each layer, the tensor index corresponds to the original expert and the
            value at that index is the assigned cluster label. Tensors shape
            (num_experts, )
    """
    num_layers = len(expert_proba)
    if num_layers == 0:
        return {}

    if total_clusters < num_layers:
        raise ValueError(
            f"total_clusters ({total_clusters}) must be at least the number of "
            f"layers ({num_layers}) to ensure at least one dominant expert per layer."
        )

    # Step 1: Identify dominant experts globally, ensuring at least one per layer.
    dominant_experts = set()

    # First, select the most probable expert from each layer to be dominant.
    for layer_idx, probas in expert_proba.items():
        most_probable_expert_idx = torch.argmax(probas).item()
        dominant_experts.add((layer_idx, most_probable_expert_idx))

    # Then, gather all other experts as candidates for the remaining dominant slots.
    non_dominant_candidates = []
    for layer_idx, probas in expert_proba.items():
        for expert_idx in range(probas.shape[0]):
            if (layer_idx, expert_idx) not in dominant_experts:
                proba = probas[expert_idx].item()
                non_dominant_candidates.append((proba, layer_idx, expert_idx))

    # Sort candidates by probability to find the best ones globally.
    non_dominant_candidates.sort(key=lambda x: x[0], reverse=True)

    # Fill the remaining cluster slots with the top candidates.
    num_remaining_clusters = total_clusters - len(dominant_experts)
    for i in range(min(num_remaining_clusters, len(non_dominant_candidates))):
        _, layer_idx, expert_idx = non_dominant_candidates[i]
        dominant_experts.add((layer_idx, expert_idx))

    # Step 2: Assign cluster labels based on similarity to dominant experts.
    all_cluster_labels = {}
    for layer_idx in distances:
        num_experts = expert_proba[layer_idx].shape[0]
        labels = torch.full((num_experts,), -1, dtype=torch.long)

        layer_dominant_indices = sorted(
            [exp_idx for l_idx, exp_idx in dominant_experts if l_idx == layer_idx]
        )

        if not layer_dominant_indices:
            if num_experts > 0:
                labels.zero_()
            all_cluster_labels[layer_idx] = labels
            continue

        layer_dominant_tensor = torch.tensor(layer_dominant_indices, dtype=torch.long)

        # Assign each non-dominant expert to the most similar dom expert (min. distance)
        for expert_idx in range(num_experts):
            if (layer_idx, expert_idx) in dominant_experts:
                # A dominant expert defines its own cluster.
                labels[expert_idx] = expert_idx
            else:
                # Find the closest dominant expert using the distance matrix.
                dist_to_dominants = distances[layer_idx][
                    expert_idx, layer_dominant_tensor
                ]
                closest_dominant_local_idx = torch.argmin(dist_to_dominants)
                closest_dominant_global_idx = layer_dominant_indices[
                    closest_dominant_local_idx
                ]
                labels[expert_idx] = closest_dominant_global_idx

        # Remap labels to be contiguous from 0.
        unique_labels = torch.unique(labels)
        label_map = {old_label.item(): i for i, old_label in enumerate(unique_labels)}
        remapped_labels = torch.tensor(
            [label_map[l.item()] for l in labels], dtype=torch.long
        )
        all_cluster_labels[layer_idx] = remapped_labels

    return all_cluster_labels


class KMeansCostTable:
    MAX_DIST = 1000.0

    def __init__(self, distances: torch.Tensor, num_merges_to_perform: int):
        """
        Initializes the cost table.

        Args:
            distances (torch.Tensor): A square tensor of pairwise distances.
            num_merges_to_perform (int): The maximum number of merges to pre-compute costs for.
        """
        self.distances = distances.fill_diagonal_(self.MAX_DIST)
        self.num_experts = distances.shape[0]
        self.num_merges_to_perform = num_merges_to_perform
        
        # cost_table[i] will store the cost of performing i+1 merges.
        self.cost_table = torch.full((num_merges_to_perform,), float("inf"))
        
        # Dictionaries to store results, keyed by the number of merges (e.g., 1, 2, ...).
        self.labels = {}
        self.centroids = {}
        self._populate_table()

    def _populate_table(self):
        # Calculate the cost for each possible number of merges up to the specified limit.
        for num_merges in range(1, self.num_merges_to_perform + 1):
            # k is the number of clusters after `num_merges`.
            k = self.num_experts - num_merges
            if k <= 0:
                continue

            centroids_np, labels_np = kmeans2(self.distances, k=k, minit="++")

            centroids = torch.tensor(centroids_np, device=self.distances.device)
            labels = torch.tensor(labels_np, device=self.distances.device)

            # Store results keyed by the number of merges.
            self.centroids[num_merges] = centroids
            self.labels[num_merges] = labels
            cost = self._calculate_merge_cost(centroids, labels)
            
            # The cost for `num_merges` is stored at index `num_merges - 1`.
            self.cost_table[num_merges - 1] = cost

    def _calculate_merge_cost(self, centroids: torch.Tensor, labels: torch.Tensor) -> float:
        # Calculate the total intra-cluster distance.
        total_cost = 0
        for cluster_idx in torch.unique(labels):
            experts_in_cluster = torch.where(labels == cluster_idx)[0]
            if len(experts_in_cluster) == 0:
                continue
            
            cluster_centroid = centroids[cluster_idx]
            per_expert_l2_cost = torch.linalg.norm(
                self.distances[experts_in_cluster] - cluster_centroid, dim=-1
            )
            total_cost += torch.sum(per_expert_l2_cost)
        return total_cost.item()

    @staticmethod
    def return_optimal_merge(
        cost_tables: list["KMeansCostTable"], num_merges_to_perform: int
    ) -> list[torch.Tensor]:
        # Tracks the number of merges assigned to each table.
        num_merges_per_table = [0] * len(cost_tables)
        
        for _ in range(num_merges_to_perform):
            min_cost = float("inf")
            best_table_idx = -1
            
            # Find the table where the *next* merge has the lowest cost.
            for table_idx, cost_table in enumerate(cost_tables):
                merges_done = num_merges_per_table[table_idx]
                if merges_done < cost_table.num_merges_to_perform:
                    # Cost for the next merge (merges_done + 1) is at index `merges_done`.
                    cost_of_next_merge = cost_table.cost_table[merges_done]
                    if cost_of_next_merge < min_cost:
                        min_cost = cost_of_next_merge
                        best_table_idx = table_idx

            if best_table_idx != -1:
                num_merges_per_table[best_table_idx] += 1
            else:
                # No more merges are possible across any table.
                break

        final_labels = []
        for i, cost_table in enumerate(cost_tables):
            merges_for_this_table = num_merges_per_table[i]
            if merges_for_this_table > 0:
                # Labels for M merges are stored at key M.
                final_labels.append(cost_table.labels[merges_for_this_table])
            else:
                # No merges were performed, so each expert is its own cluster.
                num_experts = cost_table.num_experts
                final_labels.append(
                    torch.arange(num_experts, device=cost_table.distances.device)
                )
        return final_labels


def multi_layer_kmeans_clustering(
    distances: dict[int, torch.Tensor],
    num_layers: int,
    n_clusters: int,
) -> dict[int, np.ndarray]:
    """
    Performs k-means clustering jointly across groups of consecutive layers
    using a greedy merging strategy based on pre-computed k-means costs.

    Args:
        distances (dict[int, torch.Tensor]): A dictionary of embedding tensors for each layer.
        num_layers (int): The number of consecutive layers to group for joint clustering.
        n_clusters (int): The target number of clusters per layer on average.

    Returns:
        dict[int, np.ndarray]: A dictionary mapping layer index to cluster labels.
    """
    all_cluster_labels = {}
    sorted_layer_indices = sorted(distances.keys())

    for group_start_idx in range(0, len(sorted_layer_indices), num_layers):
        layer_indices_in_group = sorted_layer_indices[
            group_start_idx : group_start_idx + num_layers
        ]
        num_layers_in_group = len(layer_indices_in_group)

        if num_layers_in_group == 0:
            continue

        group_distances = [distances[i] for i in layer_indices_in_group]
        num_experts_per_layer = [d.shape[0] for d in group_distances]
        total_experts = sum(num_experts_per_layer)
        target_total_clusters = n_clusters * num_layers_in_group
        num_merges_to_perform = total_experts - target_total_clusters

        if num_merges_to_perform <= 0:
            # No merges needed, each expert is its own cluster.
            for i, original_layer_idx in enumerate(layer_indices_in_group):
                num_experts = num_experts_per_layer[i]
                all_cluster_labels[original_layer_idx] = torch.arange(num_experts)
            continue

        cost_tables = []
        for d in group_distances:
            num_experts = d.shape[0]
            # Max merges for a layer is num_experts - 1 (to get 1 cluster).
            max_merges = num_experts - 1
            if max_merges > 0:
                cost_tables.append(KMeansCostTable(d, max_merges))
            else:
                # This layer has 0 or 1 expert, so it can't be merged.
                # We add a placeholder to keep indices aligned.
                cost_tables.append(None)

        # Filter out layers that couldn't be merged.
        valid_cost_tables = [ct for ct in cost_tables if ct is not None]
        if not valid_cost_tables:
            # Handle case where no layers in the group can be merged.
            for i, original_layer_idx in enumerate(layer_indices_in_group):
                num_experts = num_experts_per_layer[i]
                all_cluster_labels[original_layer_idx] = torch.arange(num_experts)
            continue

        # Distribute merges greedily based on cost.
        final_labels_for_valid_tables = KMeansCostTable.return_optimal_merge(
            valid_cost_tables, num_merges_to_perform
        )

        # Map the results back to the original layer indices.
        valid_table_iter = iter(final_labels_for_valid_tables)
        for i, original_layer_idx in enumerate(layer_indices_in_group):
            if cost_tables[i] is not None:
                labels = next(valid_table_iter)
            else:
                # This layer had no merges performed.
                num_experts = num_experts_per_layer[i]
                labels = torch.arange(num_experts, device=distances[original_layer_idx].device)
            
            # Remap labels to be contiguous from 0.
            unique_final_labels = torch.unique(labels)
            label_map = {
                old_label.item(): i for i, old_label in enumerate(unique_final_labels)
            }
            remapped_labels = torch.tensor([label_map[l.item()] for l in labels])
            all_cluster_labels[original_layer_idx] = remapped_labels

    return all_cluster_labels



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



class KMeansCostTableV2:
    # intended to work on characteristic_activations not pairwise distances

    def __init__(self, ca: torch.Tensor, num_merges_to_perform: int):
        """
        Initializes the cost table.

        Args:
            ca (torch.Tensor): A square tensor of characteristic activations.
            num_merges_to_perform (int): The maximum number of merges to pre-compute costs for.
        """
        self.ca = ca / torch.linalg.norm(ca, dim=-1, keepdim=True)  # normalize to unit sphere so that euclidean ~ cosine
        self.num_experts = ca.shape[0]
        self.num_merges_to_perform = num_merges_to_perform
        
        # cost_table[i] will store the cost of performing i+1 merges.
        self.cost_table = torch.full((num_merges_to_perform,), float("inf"))
        
        # Dictionaries to store results, keyed by the number of merges (e.g., 1, 2, ...).
        self.labels = {}
        self.centroids = {}
        self._populate_table()

    def _populate_table(self):
        # Calculate the cost for each possible number of merges up to the specified limit.
        for num_merges in range(1, self.num_merges_to_perform + 1):
            # k is the number of clusters after `num_merges`.
            k = self.num_experts - num_merges
            if k <= 0:
                continue

            centroids_np, labels_np = kmeans2(self.ca, k=k, minit="++")

            centroids = torch.tensor(centroids_np, device=self.ca.device)
            labels = torch.tensor(labels_np, device=self.ca.device)

            # Store results keyed by the number of merges.
            self.centroids[num_merges] = centroids
            self.labels[num_merges] = labels
            cost = self._calculate_merge_cost(centroids, labels)
            
            # The cost for `num_merges` is stored at index `num_merges - 1`.
            self.cost_table[num_merges - 1] = cost

    def _calculate_merge_cost(self, centroids: torch.Tensor, labels: torch.Tensor) -> float:
        # Calculate the total intra-cluster distance.
        total_cost = 0
        for cluster_idx in torch.unique(labels):
            experts_in_cluster = torch.where(labels == cluster_idx)[0]
            if len(experts_in_cluster) == 0:
                continue
            
            cluster_centroid = centroids[cluster_idx]
            # per_expert_cosine_dist = 1 - (
            #     self.ca[experts_in_cluster] @ cluster_centroid
            # ) / (torch.linalg.norm(cluster_centroid) + 1e-8)
            # total_cost += torch.sum(per_expert_cosine_dist) 
            # NOTE: For spherical kmeans we should use cosine dist, but follow paper and use euclidean here for cost
            per_expert_l2_cost = torch.linalg.norm(
                self.ca[experts_in_cluster] - cluster_centroid, dim=-1
            )
            total_cost += torch.sum(per_expert_l2_cost)
        return total_cost.item()

    @staticmethod
    def return_optimal_merge(
        cost_tables: list["KMeansCostTable"], num_merges_to_perform: int
    ) -> list[torch.Tensor]:
        # Tracks the number of merges assigned to each table.
        num_merges_per_table = [0] * len(cost_tables)
        
        for _ in range(num_merges_to_perform):
            min_cost = float("inf")
            best_table_idx = -1
            
            # Find the table where the *next* merge has the lowest cost.
            for table_idx, cost_table in enumerate(cost_tables):
                merges_done = num_merges_per_table[table_idx]
                if merges_done < cost_table.num_merges_to_perform:
                    # Cost for the next merge (merges_done + 1) is at index `merges_done`.
                    cost_of_next_merge = cost_table.cost_table[merges_done]
                    if cost_of_next_merge < min_cost:
                        min_cost = cost_of_next_merge
                        best_table_idx = table_idx

            if best_table_idx != -1:
                num_merges_per_table[best_table_idx] += 1
            else:
                # No more merges are possible across any table.
                break

        final_labels = []
        for i, cost_table in enumerate(cost_tables):
            merges_for_this_table = num_merges_per_table[i]
            if merges_for_this_table > 0:
                # Labels for M merges are stored at key M.
                final_labels.append(cost_table.labels[merges_for_this_table])
            else:
                # No merges were performed, so each expert is its own cluster.
                num_experts = cost_table.num_experts
                final_labels.append(
                    torch.arange(num_experts, device=cost_table.distances.device)
                )
        return final_labels


def multi_layer_kmeans_clustering_on_ca(
    distances: dict[int, torch.Tensor],
    num_layers: int,
    n_clusters: int,
) -> dict[int, np.ndarray]:
    """
    Performs k-means clustering jointly across groups of consecutive layers
    using a greedy merging strategy based on pre-computed k-means costs.

    Args:
        distances (dict[int, torch.Tensor]): A dictionary of embedding tensors for each layer.
        num_layers (int): The number of consecutive layers to group for joint clustering.
        n_clusters (int): The target number of clusters per layer on average.

    Returns:
        dict[int, np.ndarray]: A dictionary mapping layer index to cluster labels.
    """
    all_cluster_labels = {}
    sorted_layer_indices = sorted(distances.keys())

    for group_start_idx in range(0, len(sorted_layer_indices), num_layers):
        layer_indices_in_group = sorted_layer_indices[
            group_start_idx : group_start_idx + num_layers
        ]
        num_layers_in_group = len(layer_indices_in_group)

        if num_layers_in_group == 0:
            continue

        group_distances = [distances[i] for i in layer_indices_in_group]
        num_experts_per_layer = [d.shape[0] for d in group_distances]
        total_experts = sum(num_experts_per_layer)
        target_total_clusters = n_clusters * num_layers_in_group
        num_merges_to_perform = total_experts - target_total_clusters

        if num_merges_to_perform <= 0:
            # No merges needed, each expert is its own cluster.
            for i, original_layer_idx in enumerate(layer_indices_in_group):
                num_experts = num_experts_per_layer[i]
                all_cluster_labels[original_layer_idx] = torch.arange(num_experts)
            continue

        cost_tables = []
        for d in group_distances:
            num_experts = d.shape[0]
            # Max merges for a layer is num_experts - 1 (to get 1 cluster).
            max_merges = num_experts - 1
            if max_merges > 0:
                cost_tables.append(KMeansCostTableV2(d, max_merges))
            else:
                # This layer has 0 or 1 expert, so it can't be merged.
                # We add a placeholder to keep indices aligned.
                cost_tables.append(None)

        # Filter out layers that couldn't be merged.
        valid_cost_tables = [ct for ct in cost_tables if ct is not None]
        if not valid_cost_tables:
            # Handle case where no layers in the group can be merged.
            for i, original_layer_idx in enumerate(layer_indices_in_group):
                num_experts = num_experts_per_layer[i]
                all_cluster_labels[original_layer_idx] = torch.arange(num_experts)
            continue

        # Distribute merges greedily based on cost.
        final_labels_for_valid_tables = KMeansCostTableV2.return_optimal_merge(
            valid_cost_tables, num_merges_to_perform
        )

        # Map the results back to the original layer indices.
        valid_table_iter = iter(final_labels_for_valid_tables)
        for i, original_layer_idx in enumerate(layer_indices_in_group):
            if cost_tables[i] is not None:
                labels = next(valid_table_iter)
            else:
                # This layer had no merges performed.
                num_experts = num_experts_per_layer[i]
                labels = torch.arange(num_experts, device=distances[original_layer_idx].device)
            
            # Remap labels to be contiguous from 0.
            unique_final_labels = torch.unique(labels)
            label_map = {
                old_label.item(): i for i, old_label in enumerate(unique_final_labels)
            }
            remapped_labels = torch.tensor([label_map[l.item()] for l in labels])
            all_cluster_labels[original_layer_idx] = remapped_labels

    return all_cluster_labels