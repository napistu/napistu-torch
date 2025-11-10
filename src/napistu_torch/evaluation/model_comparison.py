"""Functions for comparing model's vertex embeddings."""

import logging
from itertools import combinations
from typing import Union

import numpy as np
import pandas as pd
import torch

from napistu_torch.utils.tensor_utils import (
    compute_cosine_distances_torch,
    compute_spearman_correlation_torch,
)
from napistu_torch.utils.torch_utils import ensure_device, memory_manager

logger = logging.getLogger(__name__)


def compare_embeddings(
    embeddings: dict[str, np.ndarray],
    device: Union[str, torch.device],
) -> pd.DataFrame:
    """
    Compare the vertex embeddings of multiple models.

    Parameters
    ----------
    embeddings : dict[str, np.ndarray]
        A dictionary of model names and their embeddings.
    device : Union[str, torch.device]
        The device to use when computing and comparing distances.

    Returns
    -------
    pd.DataFrame
        A dataframe of the comparisons.
    """

    device = ensure_device(device)

    # check array compatibility (must have same # of rows)
    n_rows_dict = {k: v.shape[0] for k, v in embeddings.items()}
    if len(set(n_rows_dict.values())) != 1:
        raise ValueError("All embeddings must have the same number of rows")
    n_rows = list(n_rows_dict.values())[0]

    # Convert embeddings to PyTorch tensors and compute distances with memory management
    distances = {}
    with memory_manager(device):
        for model_name, embedding in embeddings.items():
            logger.info(f"Computing distances for {model_name}...")
            distances[model_name] = compute_cosine_distances_torch(embedding, device)

    # Compare distance matrices pairwise - all unique pairs from model_prefixes
    # Use upper triangle only (exclude diagonal and avoid redundancy)
    mask = np.triu_indices(n_rows, k=1)  # k=1 excludes diagonal

    comparisons = list()
    for model1, model2 in combinations(embeddings.keys(), 2):
        logger.info(f"Comparing {model1} vs {model2}...")
        with memory_manager(device):

            dist1_flat = distances[model1][mask]
            dist2_flat = distances[model2][mask]

            # Spearman correlation using PyTorch
            rho = compute_spearman_correlation_torch(dist1_flat, dist2_flat, device)

            comparison_summary = {
                "model1": model1,
                "model2": model2,
                "spearman_rho": rho,
            }

            # Clean up intermediate arrays
            del dist1_flat, dist2_flat
            comparisons.append(comparison_summary)

    return pd.DataFrame(comparisons)
