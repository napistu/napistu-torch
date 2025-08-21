from typing import Dict, Optional

import torch
from napistu.network.constants import NAPISTU_GRAPH_EDGES, NAPISTU_GRAPH_VERTICES
from napistu.network.ng_core import NapistuGraph
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch_geometric.data import Data

from napistu_torch.load import transforms
from napistu_torch.load.constants import TRANSFORMATION

# Node configuration
VERTEX_DEFAULT_TRANSFORMS = {
    "cat": {
        TRANSFORMATION.COLUMNS: [
            NAPISTU_GRAPH_VERTICES.NODE_TYPE,
            NAPISTU_GRAPH_VERTICES.SPECIES_TYPE,
        ],
        TRANSFORMATION.TRANSFORMER: OneHotEncoder(handle_unknown="ignore"),
    }
}

# Edge configuration
EDGE_DEFAULT_TRANSFORMS = {
    "cat": {
        TRANSFORMATION.COLUMNS: [
            NAPISTU_GRAPH_EDGES.DIRECTION,
            NAPISTU_GRAPH_EDGES.SBO_TERM,
        ],
        TRANSFORMATION.TRANSFORMER: OneHotEncoder(handle_unknown="ignore"),
    },
    "num": {
        TRANSFORMATION.COLUMNS: [
            NAPISTU_GRAPH_EDGES.STOICHIOMETRY,
            NAPISTU_GRAPH_EDGES.WEIGHT,
            NAPISTU_GRAPH_EDGES.UPSTREAM_WEIGHT,
        ],
        TRANSFORMATION.TRANSFORMER: StandardScaler(),
    },
    "bool": {
        TRANSFORMATION.COLUMNS: [NAPISTU_GRAPH_EDGES.R_ISREVERSIBLE],
        TRANSFORMATION.TRANSFORMER: TRANSFORMATION.PASSTHROUGH,
    },
}


def napistu_graph_to_pyg_data(
    napistu_graph: NapistuGraph,
    vertex_transforms: Optional[Dict[str, Dict]] = None,
    edge_transforms: Optional[Dict[str, Dict]] = None,
    vertex_default_transforms: Optional[Dict[str, Dict]] = VERTEX_DEFAULT_TRANSFORMS,
    edge_default_transforms: Optional[Dict[str, Dict]] = EDGE_DEFAULT_TRANSFORMS,
    verbose: bool = False,
) -> Data:
    """Convert a NapistuGraph to a PyTorch Geometric Data object with encoded features.

    This function transforms a NapistuGraph (representing a biological network) into
    a PyTorch Geometric Data object suitable for graph neural network training.
    Node and edge features are automatically encoded using configurable transformers.

    Parameters
    ----------
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
    verbose : bool, default=False
        If True, log detailed information about config composition and encoding.

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
        - num_nodes : int
            Number of nodes in the graph

    Raises
    ------
    ValueError
        If the NapistuGraph is empty or invalid.
    KeyError
        If required vertex or edge attributes are missing from the graph.
    TypeError
        If the NapistuGraph object is not a valid NapistuGraph instance.

    Examples
    --------
    >>> from napistu.network.ng_core import NapistuGraph
    >>> from napistu_torch.load.napistu_graphs import napistu_graph_to_pyg_data
    >>>
    >>> # Load a NapistuGraph
    >>> graph = NapistuGraph.from_file("pathway.sbml")
    >>>
    >>> # Convert to PyG Data with default encoding
    >>> data = napistu_graph_to_pyg_data(graph)
    >>> print(f"Graph has {data.num_nodes} nodes and {data.edge_index.shape[1]} edges")
    >>> print(f"Node features: {data.x.shape}, Edge features: {data.edge_attr.shape}")
    >>>
    >>> # Convert with custom vertex encoding
    >>> custom_vertex_transforms = {
    ...     'custom_cat': {
    ...         'columns': ['node_type'],
    ...         'transformer': OneHotEncoder(sparse_output=False)
    ...     }
    ... }
    >>> data = napistu_graph_to_pyg_data(graph, vertex_transforms=custom_vertex_transforms)
    >>>
    >>> # Use in PyTorch Geometric model
    >>> import torch_geometric.nn as gnn
    >>> model = gnn.GCNConv(data.x.shape[1], 64)
    >>> out = model(data.x, data.edge_index)

    Notes
    -----
    - Node features are automatically encoded from vertex attributes
    - Edge features are automatically encoded from edge attributes
    - The function handles missing attributes gracefully by using 'ignore' in OneHotEncoder
    - Boolean features are passed through without transformation
    - Numerical features are standardized using StandardScaler
    - The resulting Data object is ready for PyTorch Geometric models
    """
    # 1. Extract node data as DataFrame
    vertex_df, edge_df = napistu_graph.to_pandas_dfs()

    # 2. Encode node and edge data in numpy arrays
    vertex_features, _ = transforms.encode_dataframe(
        vertex_df, vertex_default_transforms, vertex_transforms, verbose=verbose
    )
    edge_features, _ = transforms.encode_dataframe(
        edge_df, edge_default_transforms, edge_transforms, verbose=verbose
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
        num_nodes=len(napistu_graph.vs),
    )

    return data
