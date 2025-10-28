"""Module for loading pathway representations from GCS"""

import logging
import os
from typing import Optional

from napistu.gcs.constants import (
    GCS_ASSETS_NAMES,
    GCS_SUBASSET_NAMES,
)
from napistu.gcs.downloads import load_public_napistu_asset

from napistu_torch.napistu_data_store import NapistuDataStore

logger = logging.getLogger(__name__)


def gcs_model_to_store(
    napistu_data_dir: str,
    store_dir: str,
    asset_name: str = GCS_ASSETS_NAMES.HUMAN_CONSENSUS,
    asset_version: Optional[str] = None,
    overwrite_data_dir: bool = False,
    overwrite_store_dir: bool = False,
) -> NapistuDataStore:
    """
    Download a model from GCS and save it to a local directory to initialize a NapistuDataStore.

    Parameters
    ----------
    store_dir : str
        The directory to save the model to
    asset_name : str
        The name of the asset to download (default: GCS_ASSETS_NAMES.HUMAN_CONSENSUS)
    asset_version : Optional[str]
        The version of the asset to download (default: None, for the latest version)
    overwrite_data_dir : bool
        Whether to overwrite the existing napistu_data_dir
    overwrite_store_dir : bool
        Whether to overwrite the existing store_dir

    Returns
    -------
    NapistuDataStore
        The NapistuDataStore object

    """

    if overwrite_data_dir:
        # force a fresh download
        _ = load_public_napistu_asset(
            asset_name,
            napistu_data_dir,
            subasset=GCS_SUBASSET_NAMES.NAPISTU_GRAPH,
            version=asset_version,
            overwrite=True,
        )

    # use the existing asset paths and download only if needed
    subasset_paths = {
        subasset: load_public_napistu_asset(
            asset_name,
            napistu_data_dir,
            subasset=subasset,
            version=asset_version,
        )
        for subasset in [GCS_SUBASSET_NAMES.NAPISTU_GRAPH, GCS_SUBASSET_NAMES.SBML_DFS]
    }

    if not os.path.isdir(store_dir) or overwrite_store_dir:

        napistu_data_store = NapistuDataStore.create(
            store_dir=store_dir,
            sbml_dfs_path=subasset_paths[GCS_SUBASSET_NAMES.SBML_DFS],
            napistu_graph_path=subasset_paths[GCS_SUBASSET_NAMES.NAPISTU_GRAPH],
            copy_to_store=True,
            overwrite=True,
        )

    else:
        napistu_data_store = NapistuDataStore(store_dir)

    return napistu_data_store
