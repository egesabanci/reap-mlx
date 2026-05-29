import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import warnings

def _save_fig(fig, plot_path: pathlib.Path):
    fig.savefig(f"{plot_path}.png", dpi=600, bbox_inches='tight')
    fig.savefig(f"{plot_path}.pdf", dpi=600, bbox_inches='tight')
    
def _plot_layer_clusters(cluster_label: torch.Tensor, plot_path: pathlib.Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, ax = plt.subplots(figsize=(12, 6))
        cluster_sizes = torch.bincount(cluster_label).cpu().numpy()
        sns.barplot(x=range(len(cluster_sizes)), y=cluster_sizes, ax=ax)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Cluster Size")
        ax.tick_params(axis='x', labelrotation=90)
        _save_fig(fig, plot_path)

def plot_cluster_analysis(
    cluster_labels: dict[int, torch.Tensor],
    plot_dir: pathlib.Path,
    skip_first: bool,
    skip_last: bool,
):
    # layerwise clusters
    total_singletons = []
    total_non_singletons = []
    total_non_singleton_sizes = []
    num_remaining_experts_per_layer = []
    num_layers = len(cluster_labels)
    for i, (layer, cluster_label) in enumerate(cluster_labels.items()):
        layer_plot_dir = plot_dir / "layers" / f"layer_{layer}"
        layer_plot_dir.parent.mkdir(parents=True, exist_ok=True)
        _plot_layer_clusters(cluster_label, layer_plot_dir)
        cluster_sizes = torch.bincount(cluster_label)
        non_singletons = torch.argwhere(cluster_sizes != 1)
        num_singletons = len(torch.unique(cluster_label)) - non_singletons.shape[0]
        total_singletons.append(num_singletons)
        total_non_singletons.append(non_singletons.shape[0])
        size_non_singletons = (cluster_sizes[cluster_sizes!=1]).sum().item()
        total_non_singleton_sizes.append(size_non_singletons)

        if (skip_first and i == 0) or (skip_last and i == num_layers - 1):
            num_remaining_experts = len(cluster_label)
        else:
            num_remaining_experts = len(torch.unique(cluster_label))
        num_remaining_experts_per_layer.append(num_remaining_experts)


    average_non_singleton_sizes = torch.tensor(total_non_singleton_sizes) / torch.tensor(total_non_singletons)
    average_non_singleton_sizes = average_non_singleton_sizes.nan_to_num(0)
    average_non_singleton_sizes = average_non_singleton_sizes.tolist()
    # overall plots
    # singletons
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=list(cluster_labels.keys()), y=total_singletons, ax=ax)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Number of Singletons")
    _save_fig(fig, plot_dir / "singletons_per_layer")

    # non-singletons
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=list(cluster_labels.keys()), y=total_non_singletons, ax=ax)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Number of Non-Singletons")
    _save_fig(fig, plot_dir / "non_singletons_per_layer")

    # non-singleton sizes
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=list(cluster_labels.keys()), y=total_non_singleton_sizes, ax=ax)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Non-Singleton Clusters Sizes")
    _save_fig(fig, plot_dir / "non_singleton_sizes_per_layer")
    
    # average non-singleton sizes
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=list(cluster_labels.keys()), y=average_non_singleton_sizes, ax=ax)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Average Non-Singleton Clusters Sizes")
    _save_fig(fig, plot_dir / "average_non_singleton_sizes_per_layer")

    # merged experts per layer
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=list(cluster_labels.keys()), y=num_remaining_experts_per_layer, ax=ax)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Number of Remaining Experts")
    _save_fig(fig, plot_dir / "num_remaining_experts_per_layer")
