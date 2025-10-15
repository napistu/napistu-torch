from typing import Any, Callable, Dict, Optional, Union

import pandas as pd
import torch
from napistu.constants import SBML_DFS
from napistu.network.constants import (
    ADDING_ENTITY_DATA_DEFS,
    IGRAPH_DEFS,
)
from napistu.network.ng_core import NapistuGraph
from napistu.sbml_dfs_core import SBML_dfs
from torch_geometric.data import Data

from napistu_torch.load import encoding
from napistu_torch.load.constants import (
    EDGE_DEFAULT_TRANSFORMS,
    SPLITTING_STRATEGIES,
    VALID_SPLITTING_STRATEGIES,
    VERTEX_DEFAULT_TRANSFORMS,
)
from napistu_torch.load.encoders import DEFAULT_ENCODERS
from napistu_torch.load.encoding import EncodingManager
from napistu_torch.ml.constants import TRAINING
from napistu_torch.ml.stratification import create_split_masks, train_test_val_split


def augment_napistu_graph(
    sbml_dfs: SBML_dfs, napistu_graph: NapistuGraph, inplace: bool = False
) -> None:
    """
    Augment the NapistuGraph with information from the SBML_dfs.

    This function adds summaries of the SBML_dfs to the NapistuGraph,
    and extends the graph with reaction and species data from the SBML_dfs.

    Parameters
    ----------
    sbml_dfs : SBML_dfs
        The SBML_dfs to augment the NapistuGraph with.
    napistu_graph : NapistuGraph
        The NapistuGraph to augment.
    inplace : bool, default=False
        If True, modify the NapistuGraph in place.
        If False, return a new NapistuGraph with the augmentations.

    Returns
    -------
    None
        Modifies the NapistuGraph in place.
    """

    if not inplace:
        napistu_graph = napistu_graph.copy()

    # augment napistu graph with infomration from the sbml_dfs
    napistu_graph.add_sbml_dfs_summaries(
        sbml_dfs, stratify_by_bqb=False, add_name_prefixes=True
    )

    # add reactions_data to edges
    napistu_graph.add_all_entity_data(
        sbml_dfs, SBML_DFS.REACTIONS, overwrite=True, add_name_prefixes=True
    )

    napistu_graph.add_all_entity_data(
        sbml_dfs,
        SBML_DFS.SPECIES,
        mode=ADDING_ENTITY_DATA_DEFS.EXTEND,
        add_name_prefixes=True,
    )

    return None if inplace else napistu_graph


def napistu_graph_to_pyg(
    napistu_graph: NapistuGraph,
    splitting_strategy: str,
    vertex_default_transforms: Dict[str, Dict] = VERTEX_DEFAULT_TRANSFORMS,
    vertex_transforms: Dict[str, Dict] = None,
    edge_default_transforms: Dict[str, Dict] = EDGE_DEFAULT_TRANSFORMS,
    edge_transforms: Dict[str, Dict] = None,
    auto_encode: bool = True,
    encoders: Dict = DEFAULT_ENCODERS,
    verbose: bool = True,
    **strategy_kwargs: Any,
) -> Union[Data, dict[str, Data]]:
    """
    Convert a NapistuGraph to PyTorch Geometric Data object(s) with specified splitting strategy.

    This function transforms a NapistuGraph (representing a biological network) into
    a PyTorch Geometric Data object suitable for graph neural network training.
    Node and edge features are automatically encoded using configurable transformers.

    Parameters
    ----------
    napistu_graph : NapistuGraph
        The input graph to convert
    splitting_strategy : str
        One of: 'edge_mask', 'vertex_mask', 'no_mask', 'inductive'
    napistu_graph : NapistuGraph
        The NapistuGraph object containing the biological network data.
        Must have vertices (nodes) and edges with associated attributes.
    vertex_transforms : Optional[Dict[str, Dict]], default=None
        Optional override configuration for vertex (node) feature encoding.
        If provided, will be merged with vertex_default_transforms using the
        merge strategy from compose_configs.
    edge_transforms : Optional[Dict[str, Dict]], default=None
        Optional override configuration for edge feature encoding.
        If provided, will be merged with edge_default_transforms using the
        merge strategy from compose_configs.
    vertex_default_transforms : Optional[Dict[str, Dict]], default=VERTEX_DEFAULT_TRANSFORMS
        Default encoding configuration for vertex features. By default, encodes:
        - node_type and species_type as categorical features using OneHotEncoder
    edge_default_transforms : Optional[Dict[str, Dict]], default=EDGE_DEFAULT_TRANSFORMS
        Default encoding configuration for edge features. By default, encodes:
        - direction and sbo_term as categorical features using OneHotEncoder
        - stoichiometry, weight, and upstream_weight as numerical features using StandardScaler
        - r_isreversible as boolean features using passthrough
    encoders : Dict, default=DEFAULT_ENCODERS
        Dictionary of encoders to use for encoding. This is passed to the encoding.compose_encoding_configs function and auto_encode function.
    auto_encode : bool, default=True
        If True, autoencode attributes that are not explicitly encoded (and which are not part of NEVER_ENCODE).
    verbose : bool, default=False
        If True, log detailed information about config composition and encoding.
    **strategy_kwargs : Any
        Strategy-specific arguments:
        - For 'edge_mask': train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
        - For 'vertex_mask': train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
        - For 'inductive': num_hops=2, train_ratio=0.8, etc.

    Returns
    -------
    Data
        PyTorch Geometric Data object containing:
        - x : torch.Tensor
            Node features tensor of shape (num_nodes, num_node_features)
        - edge_index : torch.Tensor
            Edge connectivity tensor of shape (2, num_edges) with source and target indices
        - edge_attr : torch.Tensor
            Edge features tensor of shape (num_edges, num_edge_features)
        - vertex_feature_names : list[str]
            List of vertex feature names
        - edge_feature_names : list[str]
            List of edge feature names
        - optional, train_mask : torch.Tensor
            Mask tensor for train split
        - optional, test_mask : torch.Tensor
            Mask tensor for test split
        - optional, val_mask : torch.Tensor
            Mask tensor for validation split

    Or, if splitting_strategy is 'inductive', a dictionary of Data objects

    Examples
    --------
    >>> # Edge masking with custom split ratios
    >>> data = napistu_graph_to_pyg_data(
    ...     ng,
    ...     splitting_strategy='edge_mask',
    ...     train_ratio=0.7,
    ...     val_ratio=0.15,
    ...     test_ratio=0.15
    ... )

    >>> # Vertex masking with default splits
    >>> data = napistu_graph_to_pyg_data(ng, splitting_strategy='vertex_mask')

    >>> # Inductive split with custom parameters
    >>> data_dict = napistu_graph_to_pyg_data(
    ...     ng,
    ...     splitting_strategy='inductive',
    ...     num_hops=3,
    ...     train_ratio=0.8
    ... )
    """

    if not isinstance(napistu_graph, NapistuGraph):
        raise ValueError("napistu_graph must be a NapistuGraph object")

    if splitting_strategy not in VALID_SPLITTING_STRATEGIES:
        raise ValueError(
            f"splitting_strategy must be one of {VALID_SPLITTING_STRATEGIES}, "
            f"got '{splitting_strategy}'"
        )

    # Get the strategy function
    strategy_func = SPLITTING_STRATEGY_FUNCTIONS[splitting_strategy]

    # Call with all standard arguments plus any strategy-specific kwargs
    return strategy_func(
        napistu_graph=napistu_graph,
        vertex_default_transforms=vertex_default_transforms,
        vertex_transforms=vertex_transforms,
        edge_default_transforms=edge_default_transforms,
        edge_transforms=edge_transforms,
        auto_encode=auto_encode,
        encoders=encoders,
        verbose=verbose,
        **strategy_kwargs,
    )


# private utils


def _napistu_graph_to_pyg_edge_mask(
    napistu_graph: NapistuGraph,
    vertex_default_transforms: Dict[str, Dict] = VERTEX_DEFAULT_TRANSFORMS,
    vertex_transforms: Dict[str, Dict] = None,
    edge_default_transforms: Dict[str, Dict] = EDGE_DEFAULT_TRANSFORMS,
    edge_transforms: Dict[str, Dict] = None,
    auto_encode: bool = True,
    encoders: Dict = DEFAULT_ENCODERS,
    verbose: bool = True,
) -> dict[str, Data]:
    """NapistuGraph to PyG Data object with edge masks split across train, test, and validation edge sets."""

    # 1. extract vertex and edge DataFrames and set encodings
    vertex_df, edge_df, vertex_encoding_manager, edge_encoding_manager = (
        _standardize_graph_dfs_and_encodings(
            napistu_graph=napistu_graph,
            vertex_default_transforms=vertex_default_transforms,
            vertex_transforms=vertex_transforms,
            edge_default_transforms=edge_default_transforms,
            edge_transforms=edge_transforms,
            auto_encode=auto_encode,
            encoders=encoders,
        )
    )

    # 2. Encode vertices
    encoded_vertices, vertex_feature_names = encoding.encode_dataframe(
        vertex_df, vertex_encoding_manager, verbose=verbose
    )

    # 3. Split vertices into train/test/val
    edge_splits = train_test_val_split(edge_df, return_dict=True)

    # 4. Create masks (one mask per split, all same length as vertex_df)
    masks = create_split_masks(edge_df, edge_splits)

    # 5. Fit encoders on just the training split
    fitted_edge_encoders = encoding.fit_encoders(
        edge_splits[TRAINING.TRAIN],  # Fit on train only!
        edge_encoding_manager,
        verbose=verbose,
    )

    # 6. Transform all vertices
    encoded_edges, edge_feature_names = encoding.transform_dataframe(
        edge_df, fitted_edge_encoders  # Transform all vertices
    )

    # 7. Create edge index from all edges
    edge_index = torch.tensor(
        edge_df[[IGRAPH_DEFS.SOURCE, IGRAPH_DEFS.TARGET]].values.T, dtype=torch.long
    )

    # 8. Create PyG Data object
    return Data(
        x=torch.tensor(encoded_vertices, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=torch.tensor(encoded_edges, dtype=torch.float),
        vertex_feature_names=vertex_feature_names,
        edge_feature_names=edge_feature_names,
        **masks,  # Unpack train_mask, test_mask, val_mask
    )


def _napistu_graph_to_pyg_inductive(
    napistu_graph: NapistuGraph,
    vertex_default_transforms=VERTEX_DEFAULT_TRANSFORMS,
    vertex_transforms=None,
    edge_default_transforms=EDGE_DEFAULT_TRANSFORMS,
    edge_transforms=None,
    encoders=DEFAULT_ENCODERS,
    auto_encode=True,
    verbose=True,
):
    """
    Create PyG Data objects from a NapistuGraph with an inductive split into train, test, and validation sets.
    """

    # 1. extract vertex and edge DataFrames and set encodings
    vertex_df, edge_df, vertex_encoding_manager, edge_encoding_manager = (
        _standardize_graph_dfs_and_encodings(
            napistu_graph=napistu_graph,
            vertex_default_transforms=vertex_default_transforms,
            vertex_transforms=vertex_transforms,
            edge_default_transforms=edge_default_transforms,
            edge_transforms=edge_transforms,
            auto_encode=auto_encode,
            encoders=encoders,
        )
    )

    # 2. encode features for all vertices
    vertex_features, _ = encoding.encode_dataframe(
        vertex_df, vertex_encoding_manager, verbose=verbose
    )

    # 3. split edges into train/test/val
    edge_splits = train_test_val_split(edge_df, return_dict=True)

    # 4. fit encoders to the training edges
    edge_encoder = encoding.fit_encoders(
        edge_splits[TRAINING.TRAIN], edge_encoding_manager
    )

    pyg_data = dict()
    for k, edges in edge_splits.items():
        # encode each strata using the train encoder
        edge_features, _ = encoding.transform_dataframe(edges, edge_encoder)

        # 5. Reformat the NapistuGraph's edgelist as from-to indices
        edge_index = edge_index = torch.tensor(
            edges[[IGRAPH_DEFS.SOURCE, IGRAPH_DEFS.TARGET]].values.T, dtype=torch.long
        )

        # 6. Create PyG Data
        pyg_data[k] = Data(
            x=torch.tensor(vertex_features, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=torch.tensor(edge_features, dtype=torch.float),
            num_nodes=vertex_df.shape[0],
        )

    return pyg_data


def _napistu_graph_to_pyg_no_mask(
    napistu_graph: NapistuGraph,
    vertex_transforms: Optional[Dict[str, Dict]] = None,
    edge_transforms: Optional[Dict[str, Dict]] = None,
    vertex_default_transforms: Optional[Dict[str, Dict]] = VERTEX_DEFAULT_TRANSFORMS,
    edge_default_transforms: Optional[Dict[str, Dict]] = EDGE_DEFAULT_TRANSFORMS,
    encoders: Dict = DEFAULT_ENCODERS,
    auto_encode: bool = True,
    verbose: bool = False,
) -> Data:
    """Create a PyTorch Geometric Data object from a NapistuGraph without any splitting/masking of vertices or edges"""

    # 1. extract vertex and edge DataFrames and set encodings
    vertex_df, edge_df, vertex_encoding_manager, edge_encoding_manager = (
        _standardize_graph_dfs_and_encodings(
            napistu_graph=napistu_graph,
            vertex_default_transforms=vertex_default_transforms,
            vertex_transforms=vertex_transforms,
            edge_default_transforms=edge_default_transforms,
            edge_transforms=edge_transforms,
            auto_encode=auto_encode,
            encoders=encoders,
        )
    )

    # 2. Encode node and edge data in numpy arrays
    vertex_features, vertex_feature_names = encoding.encode_dataframe(
        vertex_df, vertex_encoding_manager, verbose=verbose
    )
    edge_features, edge_feature_names = encoding.encode_dataframe(
        edge_df, edge_encoding_manager, verbose=verbose
    )

    # 3. Reformat the NapistuGraph's edgelist as from-to indices
    edge_index = torch.tensor(
        [[e.source, e.target] for e in napistu_graph.es], dtype=torch.long
    ).T

    # 4. Create PyG Data
    data = Data(
        x=torch.tensor(vertex_features, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_features, dtype=torch.float),
        vertex_feature_names=vertex_feature_names,
        edge_feature_names=edge_feature_names,
    )

    return data


def _napistu_graph_to_pyg_vertex_mask(
    napistu_graph: NapistuGraph,
    vertex_default_transforms: Dict[str, Dict] = VERTEX_DEFAULT_TRANSFORMS,
    vertex_transforms: Dict[str, Dict] = None,
    edge_default_transforms: Dict[str, Dict] = EDGE_DEFAULT_TRANSFORMS,
    edge_transforms: Dict[str, Dict] = None,
    auto_encode: bool = True,
    encoders: Dict = DEFAULT_ENCODERS,
    verbose: bool = True,
) -> dict[str, Data]:
    """
    Create PyG Data objects from a NapistuGraph with vertex masks split across train, test, and validation vertex sets.
    """

    # 1. extract vertex and edge DataFrames and set encodings
    vertex_df, edge_df, vertex_encoding_manager, edge_encoding_manager = (
        _standardize_graph_dfs_and_encodings(
            napistu_graph=napistu_graph,
            vertex_default_transforms=vertex_default_transforms,
            vertex_transforms=vertex_transforms,
            edge_default_transforms=edge_default_transforms,
            edge_transforms=edge_transforms,
            auto_encode=auto_encode,
            encoders=encoders,
        )
    )

    # 2. Encode edges
    encoded_edges, edge_feature_names = encoding.encode_dataframe(
        edge_df, edge_encoding_manager, verbose=verbose
    )

    # 3. Split vertices into train/test/val
    vertex_splits = train_test_val_split(vertex_df, return_dict=True)

    # 4. Create masks (one mask per split, all same length as vertex_df)
    masks = create_split_masks(vertex_df, vertex_splits)

    # 5. Fit encoders on just the training split
    fitted_vertex_encoders = encoding.fit_encoders(
        vertex_splits["train"],  # Fit on train only!
        vertex_encoding_manager,
        verbose=verbose,
    )

    # 6. Transform all vertices
    vertex_features, vertex_feature_names = encoding.transform_dataframe(
        vertex_df, fitted_vertex_encoders  # Transform all vertices
    )

    # 7. Create edge index from all edges
    edge_index = torch.tensor(
        edge_df[[IGRAPH_DEFS.SOURCE, IGRAPH_DEFS.TARGET]].values.T, dtype=torch.long
    )

    # 8. Create PyG Data object
    return Data(
        x=torch.tensor(vertex_features, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=torch.tensor(encoded_edges, dtype=torch.float),
        vertex_feature_names=vertex_feature_names,
        edge_feature_names=edge_feature_names,
        **masks,  # Unpack train_mask, test_mask, val_mask
    )


def _standardize_graph_dfs_and_encodings(
    napistu_graph: NapistuGraph,
    vertex_default_transforms: Dict[str, Dict],
    vertex_transforms: Dict[str, Dict],
    edge_default_transforms: Dict[str, Dict],
    edge_transforms: Dict[str, Dict],
    auto_encode: bool,
    encoders: Dict = DEFAULT_ENCODERS,
) -> tuple[pd.DataFrame, pd.DataFrame, EncodingManager, EncodingManager]:
    """
    Standardize the node and edge DataFrames and encoding managers for a NapistuGraph.

    This is a common pattern to prepare a NapistuGraph for encoding as matrices of vertex and edge features.

    Parameters
    ----------
    napistu_graph : NapistuGraph
        The NapistuGraph to standardize
    vertex_default_transforms : Dict[str, Dict]
        The default vertex transformations to apply
    vertex_transforms : Dict[str, Dict]
        Additional vertex transformations to apply
    edge_default_transforms : Dict[str, Dict]
        The default edge transformations to apply
    edge_transforms : Dict[str, Dict]
        Additional edge transformations to apply
    auto_encode : bool
        Whether to automatically select an appropriate encoding for unaccounted for attributes
    encoders : Dict
        The encoders to use

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, EncodingManager, EncodingManager]

    - vertex_df : pd.DataFrame
        The vertex DataFrame
    - edge_df : pd.DataFrame
        The edge DataFrame
    - vertex_encoding_manager : EncodingManager
        The vertex encoding manager
    - edge_encoding_manager : EncodingManager
        The edge encoding manager

    """
    # 1. Extract node data as DataFrame
    vertex_df, edge_df = napistu_graph.to_pandas_dfs()

    # 2. combine defaults with overrides
    vertex_encoding_manager = encoding.compose_encoding_configs(
        vertex_default_transforms, vertex_transforms, encoders
    )
    edge_encoding_manager = encoding.compose_encoding_configs(
        edge_default_transforms, edge_transforms, encoders
    )

    # 3. optionally, automatically select an appropriate encoding for unaccounted for attributes
    if auto_encode:
        vertex_encoding_manager = encoding.auto_encode(
            vertex_df, vertex_encoding_manager
        )
        edge_encoding_manager = encoding.auto_encode(edge_df, edge_encoding_manager)

    return vertex_df, edge_df, vertex_encoding_manager, edge_encoding_manager


# Strategy registry
SPLITTING_STRATEGY_FUNCTIONS: Dict[str, Callable] = {
    SPLITTING_STRATEGIES.EDGE_MASK: _napistu_graph_to_pyg_edge_mask,
    SPLITTING_STRATEGIES.VERTEX_MASK: _napistu_graph_to_pyg_vertex_mask,
    SPLITTING_STRATEGIES.NO_MASK: _napistu_graph_to_pyg_no_mask,
    SPLITTING_STRATEGIES.INDUCTIVE: _napistu_graph_to_pyg_inductive,
}
