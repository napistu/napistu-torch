import torch
from sklearn.manifold import TSNE

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


@torch.no_grad()
def layout_tsne(
    model, filtering_mask=None, n_components=2, perplexity=30, random_state=42
):
    """
    Layout embeddings in 2D using t-SNE with sensible defaults.

    For large datasets (>10K), t-SNE becomes impractically slow.
    Use filtering_mask to subset the data or consider using UMAP instead.

    Parameters
    ----------
    model : torch.nn.Module
        Model with embedding layer
    filtering_mask : torch.Tensor or None, optional
        Boolean mask of shape (n_features,) to select subset of embeddings.
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
    model.eval()

    if hasattr(model, "embedding"):
        z = model.embedding.weight.data
    else:
        z = model()

    # Apply filtering mask
    if filtering_mask is None:
        filtering_mask = torch.ones(z.shape[0], dtype=torch.bool, device=z.device)

    z = z[filtering_mask].cpu().numpy()

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
    model, filtering_mask=None, n_components=2, n_neighbors=15, random_state=42
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
    model : torch.nn.Module
        Model with embedding layer
    filtering_mask : torch.Tensor or None, optional
        Boolean mask of shape (n_features,) to select subset of embeddings.
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

    model.eval()

    if hasattr(model, "embedding"):
        z = model.embedding.weight.data
    else:
        z = model()

    # Apply filtering mask
    if filtering_mask is None:
        filtering_mask = torch.ones(z.shape[0], dtype=torch.bool, device=z.device)

    z = z[filtering_mask].cpu().numpy()

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
    )

    z_2d = reducer.fit_transform(z)
    return z_2d
