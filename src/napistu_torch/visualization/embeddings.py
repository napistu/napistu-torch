from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def plot_coordinates_with_masks(
    coordinates,
    masks,
    mask_names,
    figsize=(15, 10),
    ncols=3,
    cmap_bg="lightgray",
    cmap_fg="red",
    alpha=0.6,
    s=10,
):
    """
    Plot 2D coordinates with binary masks overlaid separately for each category.

    Parameters
    ----------
    coordinates : array-like, shape (n_points, 2)
        2D coordinates (e.g., UMAP layout)
    masks : array-like, shape (n_points, n_categories)
        Binary mask matrix where each column represents a category
    mask_names : list of str
        Names of the categories (one per column of masks)
    figsize : tuple, optional
        Figure size (width, height)
    ncols : int, optional
        Number of columns in the subplot grid
    cmap_bg : str, optional
        Color for points where mask is False (0)
    cmap_fg : str, optional
        Color for points where mask is True (1)
    alpha : float, optional
        Transparency of points
    s : float, optional
        Size of points

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : array of matplotlib.axes.Axes
        Array of subplot axes
    """
    # Convert to numpy if tensors
    if torch.is_tensor(coordinates):
        coordinates = coordinates.cpu().numpy()
    if torch.is_tensor(masks):
        masks = masks.cpu().numpy()

    # Ensure masks is 2D
    if masks.ndim == 1:
        masks = masks.reshape(-1, 1)

    n_categories = masks.shape[1]

    # Calculate grid dimensions
    nrows = (n_categories + ncols - 1) // ncols

    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Flatten axes for easier iteration
    if n_categories == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten() if nrows > 1 else np.array(axes).flatten()

    # Plot each category
    for idx, (ax, category_name) in enumerate(zip(axes[:n_categories], mask_names)):
        mask = masks[:, idx].astype(bool)

        # Plot background points (where mask is False)
        ax.scatter(
            coordinates[~mask, 0],
            coordinates[~mask, 1],
            c=cmap_bg,
            s=s,
            alpha=alpha,
            label="Other",
        )

        # Plot foreground points (where mask is True)
        if mask.sum() > 0:  # Only plot if there are any True values
            ax.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                c=cmap_fg,
                s=s * 1.5,
                alpha=alpha,
                label=category_name,
            )

        ax.set_title(
            f"{category_name} edge predic({mask.sum()} points)", fontsize=10, loc="left"
        )
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.legend(markerscale=2, fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_categories, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    return fig, axes


@torch.no_grad()
def layout_tsne(
    embeddings: Union[torch.Tensor, np.ndarray],
    filtering_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    n_components: int = 2,
    perplexity: int = 30,
    random_state: int = 42,
):
    """
    Layout embeddings in 2D using t-SNE with sensible defaults.

    For large datasets (>10K), t-SNE becomes impractically slow.
    Use filtering_mask to subset the data or consider using UMAP instead.

    Parameters
    ----------
    embeddings : Union[torch.Tensor, np.ndarray]
        Precomputed node embeddings. Shape [num_nodes, embedding_dim].
    filtering_mask : Union[torch.Tensor, np.ndarray] or None, optional
        Boolean mask of shape (num_nodes,) to select subset of embeddings.
        If None, uses all embeddings, by default None
    n_components : int, optional
        Number of dimensions, by default 2
    perplexity : int, optional
        Balance between local and global structure, by default 30.
        Reasonable range: 5-50 depending on dataset size
    random_state : int, optional
        Random seed for reproducibility, by default 42

    Returns
    -------
    numpy.ndarray
        Array of shape (n_selected, n_components) containing 2D embeddings
    """
    if isinstance(embeddings, torch.Tensor):
        z = embeddings.detach().to(dtype=torch.float32)
    else:
        z = torch.as_tensor(embeddings, dtype=torch.float32)

    mask = _prepare_filtering_mask(z, filtering_mask)

    z = z[mask].cpu().numpy()

    reducer = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate="auto",
        n_iter=1000,
        random_state=random_state,
        init="pca",
        metric="cosine",
    )

    z_2d = reducer.fit_transform(z)
    return z_2d


@torch.no_grad()
def layout_umap(
    embeddings: Union[torch.Tensor, np.ndarray],
    filtering_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    n_components: int = 2,
    n_neighbors: int = 15,
    random_state: int = 42,
):
    """
    Layout embeddings in 2D using UMAP with sensible defaults.

    UMAP is generally preferred for embeddings: faster, more stable,
    and better at preserving both local and global structure. UMAP
    scales well to large datasets (100K+ samples).

    Note: Requires umap-learn package. Install with:
        pip install napistu-torch[viz]
    or
        pip install umap-learn

    Parameters
    ----------
    embeddings : Union[torch.Tensor, numpy.ndarray]
        Precomputed node embeddings. Shape [num_nodes, embedding_dim].
    filtering_mask : Union[torch.Tensor, numpy.ndarray] or None, optional
        Boolean mask of shape (num_nodes,) to select subset of embeddings.
        If None, uses all embeddings, by default None
    n_components : int, optional
        Number of dimensions, by default 2
    n_neighbors : int, optional
        Size of local neighborhood, by default 15.
        Reasonable range: 5-50 depending on desired granularity
    random_state : int, optional
        Random seed for reproducibility, by default 42

    Returns
    -------
    numpy.ndarray
        Array of shape (n_selected, n_components) containing 2D embeddings

    Raises
    ------
    ImportError
        If umap-learn is not installed
    """
    if not UMAP_AVAILABLE:
        raise ImportError(
            "UMAP is not installed. Install it with:\n" "  pip install umap-learn"
        )

    if isinstance(embeddings, torch.Tensor):
        z = embeddings.detach().to(dtype=torch.float32)
    else:
        z = torch.as_tensor(embeddings, dtype=torch.float32)

    mask = _prepare_filtering_mask(z, filtering_mask)

    z = z[mask].cpu().numpy()

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
    )

    z_2d = reducer.fit_transform(z)
    return z_2d


def _prepare_filtering_mask(
    embeddings: torch.Tensor,
    filtering_mask: Optional[Union[torch.Tensor, np.ndarray]],
) -> torch.Tensor:
    if filtering_mask is None:
        return torch.ones(
            embeddings.shape[0], dtype=torch.bool, device=embeddings.device
        )

    if isinstance(filtering_mask, torch.Tensor):
        mask = filtering_mask.to(dtype=torch.bool, device=embeddings.device)
    else:
        mask = torch.as_tensor(
            filtering_mask, dtype=torch.bool, device=embeddings.device
        )

    if mask.ndim != 1 or mask.shape[0] != embeddings.shape[0]:
        raise ValueError(
            "filtering_mask must be 1D and match number of rows in embeddings"
        )

    return mask
