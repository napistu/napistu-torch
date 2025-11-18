import inspect
import logging
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd
import torch
from napistu.constants import SBML_DFS
from napistu.network.constants import (
    ADDING_ENTITY_DATA_DEFS,
    IGRAPH_DEFS,
    NAPISTU_GRAPH,
    NAPISTU_GRAPH_EDGES,
    NAPISTU_GRAPH_VERTICES,
    SINGULAR_GRAPH_ENTITIES,
    VALID_VERTEX_SBML_DFS_SUMMARIES,
)
from napistu.network.ng_core import NapistuGraph
from napistu.sbml_dfs_core import SBML_dfs

from napistu_torch.labels.constants import (
    LABEL_TYPE,
    LABELING,
    TASK_TYPES,
)
from napistu_torch.labels.create import create_vertex_labels
from napistu_torch.labels.labeling_manager import LABELING_MANAGERS, LabelingManager
from napistu_torch.load import encoding
from napistu_torch.load.constants import (
    EDGE_DEFAULT_TRANSFORMS,
    IGNORED_EDGE_ATTRIBUTES,
    IGNORED_IF_CONSTANT_EDGE_ATTRIBUTES,
    IGNORED_IF_CONSTANT_VERTEX_ATTRIBUTES,
    IGNORED_VERTEX_ATTRIBUTES,
    SPLITTING_STRATEGIES,
    VALID_SPLITTING_STRATEGIES,
    VERTEX_DEFAULT_TRANSFORMS,
)
from napistu_torch.load.encoders import DEFAULT_ENCODERS
from napistu_torch.load.encoding import EncodingManager
from napistu_torch.ml.constants import TRAINING
from napistu_torch.ml.splitting import create_split_masks, train_test_val_split
from napistu_torch.napistu_data import NapistuData

# Set up logger
logger = logging.getLogger(__name__)


def augment_napistu_graph(
    sbml_dfs: SBML_dfs,
    napistu_graph: NapistuGraph,
    sbml_dfs_summary_types: list = VALID_VERTEX_SBML_DFS_SUMMARIES,
    ignored_attributes: dict[str, list[str]] = {
        NAPISTU_GRAPH.EDGES: IGNORED_EDGE_ATTRIBUTES,
        NAPISTU_GRAPH.VERTICES: IGNORED_VERTEX_ATTRIBUTES,
    },
    ignored_if_constant_attributes: dict[str, dict[str, Any]] = {
        NAPISTU_GRAPH.EDGES: IGNORED_IF_CONSTANT_EDGE_ATTRIBUTES,
        NAPISTU_GRAPH.VERTICES: IGNORED_IF_CONSTANT_VERTEX_ATTRIBUTES,
    },
    inplace: bool = False,
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
    sbml_dfs_summary_types : list, optional
        Types of summaries to include. Defaults to all valid summary types.
    ignored_attributes : dict[str, list[str]], optional
        A dictionary of attribute types and lists of attributes to ignore. Defaults to IGNORED_EDGE_ATTRIBUTES and IGNORED_VERTEX_ATTRIBUTES.
    ignored_if_constant_attributes : dict[str, dict[str, Any]], optional
        A dictionary of attribute types and dicts mapping attribute names to values to check against. For example, {"some_attr": 0} means check if all values are 0 or None. Defaults to IGNORED_IF_CONSTANT_EDGE_ATTRIBUTES and IGNORED_IF_CONSTANT_VERTEX_ATTRIBUTES.
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
    if len(sbml_dfs_summary_types) > 0:
        logger.info(
            f"Augmenting `NapistuGraph` with `SBML_dfs`' summaries: {sbml_dfs_summary_types}"
        )
        napistu_graph.add_sbml_dfs_summaries(
            sbml_dfs,
            summary_types=sbml_dfs_summary_types,
            stratify_by_bqb=False,
            add_name_prefixes=True,
            binarize=True,
        )
    else:
        logger.info(
            "Skipping augmentation of `NapistuGraph` with `SBML_dfs` summaries since `sbml_dfs_summary_types` is empty"
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

    # drop ignored attributes which aren't needed
    _ignore_graph_attributes(napistu_graph, ignored_attributes)
    # ignore attributes if they are a constant specified value or missing for all vertices/edges
    _ignore_if_constant(napistu_graph, ignored_if_constant_attributes)

    return None if inplace else napistu_graph


def construct_vertex_labeled_napistu_data(
    sbml_dfs: SBML_dfs,
    napistu_graph: NapistuGraph,
    splitting_strategy: str = SPLITTING_STRATEGIES.VERTEX_MASK,
    label_type: Union[str, LabelingManager] = LABEL_TYPE.SPECIES_TYPE,
    task_type: str = TASK_TYPES.CLASSIFICATION,
    labeling_managers: Optional[Dict[str, LabelingManager]] = LABELING_MANAGERS,
    name: Optional[str] = None,
    deduplicate_features: bool = True,
    **kwargs,
) -> Union[NapistuData, Dict[str, NapistuData]]:
    """
    Construct a PyG data object for supervised training tasks using a SBML_dfs and NapistuGraph.

    This function handles the workflow for supervised learning tasks where labels are derived
    from graph attributes. The process is:
    1. Extract labels from the original graph (before augmentation) - labels may depend on
        attributes that exist in the original graph
    2. Augment the graph with SBML_dfs data (sources, reactions, species) to add features
    3. Remove attributes that should not be encoded as features (e.g., the label attribute itself)
    4. Encode the augmented graph (with excluded attributes removed) into NapistuData

    Note: Labels are created from the original graph because they may depend on specific attributes
    that must be present before augmentation. However, the final NapistuData uses the augmented
    graph to ensure all SBML_dfs-derived features are included.

    Parameters
    ----------
    sbml_dfs : SBML_dfs
        The SBML_dfs to augment the NapistuGraph with.
    napistu_graph : NapistuGraph
        The NapistuGraph to augment.
    splitting_strategy : str
        The strategy to use for splitting the data into train/test/val sets.
    label_type : Union[str, LabelingManager]
        The type of labels to use for the supervised task.
    task_type : str
        The type of task to use for the supervised task (classification or regression)
    labeling_managers : Optional[Dict[str, LabelingManager]]
        A dictionary of LabelingManager objects for each label type. If label_type is a LabelingManager, this is ignored.
    name : Optional[str], default=None
        Name for the NapistuData object. If None, uses the default name.
    **kwargs : dict
        Additional keyword arguments to pass to the NapistuData constructor.

    Returns
    -------
    NapistuData : Union[NapistuData, Dict[str, NapistuData]]
        A NapistuData object containing the augmented NapistuGraph and labels.
        The labeling manager is embedded in the NapistuData object.
        If splitting_strategy is 'inductive', returns Dict[str, NapistuData] with keys
        'train', 'test', 'validation' (or subset thereof).

    Examples
    --------
    >>> # Vertex masking with default splits
    >>> data = construct_vertex_labeled_napistu_data(ng, splitting_strategy='vertex_mask')
    """

    # Step 1: Extract labels from the original graph (before augmentation)
    # Labels must be created from the original graph because they may depend on
    # attributes that need to be present in the graph before augmentation
    labels, labeling_manager = create_vertex_labels(
        napistu_graph,
        label_type=label_type,
        task_type=task_type,
        labeling_managers=labeling_managers,
    )

    # Step 2: Augment the graph with SBML_dfs data to add features
    # This adds source summaries, reaction data, and species data to the graph
    working_napistu_graph = augment_napistu_graph(
        sbml_dfs,
        napistu_graph,
        sbml_dfs_summary_types=labeling_manager.augment_summary_types,
        inplace=False,
    )

    # Step 3: Remove attributes that should not be encoded as features
    # This prevents data leakage - if we're predicting 'species_type', we don't want
    # 'species_type' to be an input feature (the model would just look at it directly).
    # The remove_attributes() call physically deletes these attributes from the graph
    # so they won't be present when the graph is converted to a DataFrame for encoding.
    # For example:
    #   - SPECIES_TYPE label excludes: [SPECIES_TYPE]
    #   - NODE_TYPE label excludes: [NODE_TYPE, SPECIES_TYPE]
    if len(labeling_manager.exclude_vertex_attributes) > 0:
        working_napistu_graph.remove_attributes(
            NAPISTU_GRAPH.VERTICES, labeling_manager.exclude_vertex_attributes
        )

        # Also remove excluded attributes from the default transforms configuration
        # so the encoding system doesn't try to encode attributes that no longer exist
        vertex_default_transforms = {
            k: v - set(labeling_manager.exclude_vertex_attributes)
            for k, v in VERTEX_DEFAULT_TRANSFORMS.items()
        }
        vertex_default_transforms = {
            k: v for k, v in vertex_default_transforms.items() if len(v) > 0
        }
    else:
        vertex_default_transforms = VERTEX_DEFAULT_TRANSFORMS

    # Step 4: Encode the augmented graph (with excluded attributes removed) into NapistuData
    # Use the working graph which has both the augmentations AND the correct attributes removed
    napistu_data = napistu_graph_to_napistu_data(
        working_napistu_graph,  # Use augmented graph with SBML_dfs data and excluded attributes removed
        splitting_strategy=splitting_strategy,
        vertex_default_transforms=vertex_default_transforms,
        labels=labels,
        name=name,
        labeling_manager=labeling_manager,
        deduplicate_features=deduplicate_features,
        **kwargs,
    )

    return napistu_data


def construct_unlabeled_napistu_data(
    sbml_dfs: SBML_dfs,
    napistu_graph: NapistuGraph,
    splitting_strategy: str = SPLITTING_STRATEGIES.NO_MASK,
    name: Optional[str] = None,
    **kwargs,
) -> Union[NapistuData, Dict[str, NapistuData]]:
    """
    Construct a NapistuData object from an SBML_dfs and NapistuGraph.

    This function augments the NapistuGraph with SBML_dfs summaries and reaction data,
    and then encodes the graph into a NapistuData object.

    Parameters
    ----------
    sbml_dfs : SBML_dfs
        The SBML_dfs to augment the NapistuGraph with.
    napistu_graph : NapistuGraph
        The NapistuGraph to augment.
    splitting_strategy : str, optional
        The splitting strategy to use for the NapistuData object.
        Defaults to SPLITTING_STRATEGIES.NO_MASK.
    name : Optional[str], default=None
        Name for the NapistuData object. If None, uses the default name.
    **kwargs:
        Additional keyword arguments to pass to napistu_graph_to_napistu_data.

    Returns
    -------
    NapistuData : Union[NapistuData, Dict[str, NapistuData]]
        A NapistuData object containing the augmented NapistuGraph.
        If splitting_strategy is 'inductive', returns Dict[str, NapistuData] with keys
        'train', 'test', 'validation' (or subset thereof).

    Examples
    --------
    >>> # Unmasked data with default splits
    >>> data = construct_unlabeled_napistu_data(ng, splitting_strategy='no_mask')
    """

    working_napistu_graph = augment_napistu_graph(
        sbml_dfs, napistu_graph, inplace=False
    )

    napistu_data = napistu_graph_to_napistu_data(
        working_napistu_graph,
        splitting_strategy=splitting_strategy,
        name=name,
        **kwargs,
    )

    return napistu_data


def napistu_graph_to_napistu_data(
    napistu_graph: NapistuGraph,
    splitting_strategy: str,
    vertex_default_transforms: Union[
        Dict[str, Dict], EncodingManager
    ] = VERTEX_DEFAULT_TRANSFORMS,
    vertex_transforms: Optional[Union[Dict[str, Dict], EncodingManager]] = None,
    edge_default_transforms: Union[
        Dict[str, Dict], EncodingManager
    ] = EDGE_DEFAULT_TRANSFORMS,
    edge_transforms: Optional[Union[Dict[str, Dict], EncodingManager]] = None,
    auto_encode: bool = True,
    encoders: Dict[str, Any] = DEFAULT_ENCODERS,
    deduplicate_features: bool = True,
    verbose: bool = True,
    labels: Optional[torch.Tensor] = None,
    name: Optional[str] = None,
    labeling_manager: Optional[LabelingManager] = None,
    **strategy_kwargs: Any,
) -> Union[NapistuData, Dict[str, NapistuData]]:
    """
    Convert a NapistuGraph to NapistuData object(s) with specified splitting strategy.

    This function transforms a NapistuGraph (representing a biological network) into
    a NapistuData object (based on PyTorch Geometric Data object) suitable for graph
    neural network training. Node and edge features are automatically encoded using
    configurable transformers.

    Parameters
    ----------
    napistu_graph : NapistuGraph
        The input graph to convert
    splitting_strategy : str
        One of: 'edge_mask', 'vertex_mask', 'no_mask', 'inductive'
    napistu_graph : NapistuGraph
        The NapistuGraph object containing the biological network data.
        Must have vertices (nodes) and edges with associated attributes.
    vertex_transforms : Optional[Union[Dict[str, Dict], EncodingManager]], default=None
        Optional override configuration for vertex (node) feature encoding.
        If provided, will be merged with vertex_default_transforms using the
        merge strategy from compose_configs.
    edge_transforms : Optional[Union[Dict[str, Dict], EncodingManager]], default=None
        Optional override configuration for edge feature encoding.
        If provided, will be merged with edge_default_transforms using the
        merge strategy from compose_configs.
    vertex_default_transforms : Union[Dict[str, Dict], EncodingManager], default=VERTEX_DEFAULT_TRANSFORMS
        Default encoding configuration for vertex features. By default, encodes:
        - node_type and species_type as categorical features using OneHotEncoder
    edge_default_transforms : Union[Dict[str, Dict], EncodingManager], default=EDGE_DEFAULT_TRANSFORMS
        Default encoding configuration for edge features. By default, encodes:
        - direction and sbo_term as categorical features using OneHotEncoder
        - stoichiometry, weight, and upstream_weight as numerical features using StandardScaler
        - r_isreversible as boolean features using passthrough
    encoders : Dict[str, Any], default=DEFAULT_ENCODERS
        Dictionary of encoders to use for encoding. This is passed to the encoding.compose_encoding_configs function and auto_encode function.
    auto_encode : bool, default=True
        If True, autoencode attributes that are not explicitly encoded (and which are not part of NEVER_ENCODE).
    deduplicate_features : bool, default=True
        If True, deduplicate identical features and name the resulting columns using the shortest
        common prefix of the merged columns.
    verbose : bool, default=False
        If True, log detailed information about config composition and encoding.
    labels : Optional[torch.Tensor], default=None
        Optional labels tensor to set as 'y' attribute in the resulting NapistuData object(s).
    name : Optional[str], default=None
        Name for the NapistuData object(s). If None, uses the default name.
    labeling_manager : Optional[LabelingManager], default=None
        Labeling manager used to encode the labels. Should be provided when labels are present.
    **strategy_kwargs : Any
        Strategy-specific arguments:
        - For 'edge_mask': train_size=0.8, val_size=0.1, test_size=0.1
        - For 'vertex_mask': train_size=0.8, val_size=0.1, test_size=0.1
        - For 'inductive': num_hops=2, train_size=0.8, etc.

    Returns
    -------
    Union[NapistuData, Dict[str, NapistuData]]
        NapistuData object (subclass of PyTorch Geometric Data) containing:
        - x : torch.Tensor
            Node features tensor of shape (num_nodes, num_node_features)
        - edge_index : torch.Tensor
            Edge connectivity tensor of shape (2, num_edges) with source and target indices
        - edge_attr : torch.Tensor
            Edge features tensor of shape (num_edges, num_edge_features)
        - edge_weight : torch.Tensor, optional
            1D tensor of original edge weights for scalar weight-based models
        - vertex_feature_names : List[str]
            List of vertex feature names
        - edge_feature_names : List[str]
            List of edge feature names
        - vertex_feature_name_aliases : Dict[str, str]
            Mapping from vertex feature names to their canonical names (for deduplicated features)
        - edge_feature_name_aliases : Dict[str, str]
            Mapping from edge feature names to their canonical names (for deduplicated features)
        - optional, y : torch.Tensor
            Labels tensor if labels parameter was provided

        If splitting_strategy is 'inductive', returns Dict[str, NapistuData] with keys
        'train', 'test', 'validation' (or subset thereof).

    Examples
    --------
    >>> # Edge masking with custom split ratios
    >>> data = napistu_graph_to_napistu_data(
    ...     ng,
    ...     splitting_strategy='edge_mask',
    ...     train_size=0.7,
    ...     val_size=0.15,
    ...     test_size=0.15
    ... )

    >>> # Vertex masking with default splits
    >>> data = napistu_graph_to_napistu_data(ng, splitting_strategy='vertex_mask')

    >>> # Inductive split with custom parameters
    >>> data_dict = napistu_graph_to_napistu_data(
    ...     ng,
    ...     splitting_strategy='inductive',
    ...     num_hops=3,
    ...     train_size=0.8
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

    # Filter strategy_kwargs to only include parameters that the strategy function accepts
    strategy_sig = inspect.signature(strategy_func)
    strategy_params = set(strategy_sig.parameters.keys())

    # Identify ignored parameters and warn about them
    ignored_params = set(strategy_kwargs.keys()) - strategy_params
    if ignored_params:
        logger.warning(
            f"The following parameters were ignored by '{splitting_strategy}' strategy: {sorted(ignored_params)}. "
            f"Only parameters accepted by this strategy will be used."
        )

    # Only include kwargs that are valid parameters for this strategy function
    filtered_kwargs = {
        key: value for key, value in strategy_kwargs.items() if key in strategy_params
    }

    # Generate descriptive name if not provided
    if name is None:
        name = _name_napistu_data(
            splitting_strategy=splitting_strategy,
            labels=labels,
            labeling_manager=labeling_manager,
        )

    # Call with all standard arguments plus filtered strategy-specific kwargs
    return strategy_func(
        napistu_graph=napistu_graph,
        name=name,
        vertex_default_transforms=vertex_default_transforms,
        vertex_transforms=vertex_transforms,
        edge_default_transforms=edge_default_transforms,
        edge_transforms=edge_transforms,
        auto_encode=auto_encode,
        encoders=encoders,
        deduplicate_features=deduplicate_features,
        verbose=verbose,
        labels=labels,
        labeling_manager=labeling_manager,
        **filtered_kwargs,
    )


# private utils


def _extract_edge_weights(edge_df: pd.DataFrame) -> Optional[torch.Tensor]:
    """
    Extract original edge weights from edge DataFrame.

    Parameters
    ----------
    edge_df : pd.DataFrame
        Edge DataFrame containing weight information

    Returns
    -------
    Optional[torch.Tensor]
        1D tensor of original edge weights, or None if no weights found
    """
    from napistu.network.constants import NAPISTU_GRAPH_EDGES

    if NAPISTU_GRAPH_EDGES.WEIGHT in edge_df.columns:
        weights = edge_df[NAPISTU_GRAPH_EDGES.WEIGHT].values
        return torch.tensor(weights, dtype=torch.float)
    else:
        logger.warning("No edge weights found in edge DataFrame")
    return None


def _ignore_graph_attributes(
    napistu_graph: NapistuGraph,
    ignored_attributes: dict[str, list[str]] = {
        NAPISTU_GRAPH.EDGES: IGNORED_EDGE_ATTRIBUTES,
        NAPISTU_GRAPH.VERTICES: IGNORED_VERTEX_ATTRIBUTES,
    },
) -> None:
    """
    Remove specified attributes from vertices or edges.

    This function removes the specified attributes from either vertices or edges. This is generally to restrict the vertex and edge encodings to a manageable size.
    """
    for entity_type in [NAPISTU_GRAPH.EDGES, NAPISTU_GRAPH.VERTICES]:
        if entity_type not in ignored_attributes:
            continue

        if entity_type == NAPISTU_GRAPH.EDGES:
            existing_attributes = napistu_graph.es.attributes()
        else:
            existing_attributes = napistu_graph.vs.attributes()

        to_be_removed_attributes = set(ignored_attributes[entity_type]) & set(
            existing_attributes
        )

        if len(to_be_removed_attributes) > 0:
            entity_name = SINGULAR_GRAPH_ENTITIES[entity_type]
            logger.info(
                f"Removing the following {entity_name} attributes: {to_be_removed_attributes}"
            )
            napistu_graph.remove_attributes(entity_type, to_be_removed_attributes)

    return None


def _ignore_if_constant(
    napistu_graph: NapistuGraph,
    attributes_to_check: dict[str, dict[str, Any]] = {
        NAPISTU_GRAPH.EDGES: IGNORED_IF_CONSTANT_EDGE_ATTRIBUTES,
        NAPISTU_GRAPH.VERTICES: IGNORED_IF_CONSTANT_VERTEX_ATTRIBUTES,
    },
) -> None:
    """
    Remove attributes from vertices or edges if they are constant at a specific value or missing for all vertices/edges.

    This function checks specified attributes and removes them if:
    - All vertices/edges have the specified value (or None/missing)
    - All vertices/edges have missing/None values

    Parameters
    ----------
    napistu_graph : NapistuGraph
        The NapistuGraph to check and modify.
    attributes_to_check : dict[str, dict[str, Any]], optional
        Dictionary mapping NAPISTU_GRAPH.EDGES/VERTICES to dicts mapping attribute names
        to values to check against. For example, {"some_attr": 0} means check if all
        values are 0 or None. Defaults to empty dicts.

    Returns
    -------
    None
        Modifies the NapistuGraph in place.
    """
    for entity_type in [NAPISTU_GRAPH.EDGES, NAPISTU_GRAPH.VERTICES]:
        if entity_type not in attributes_to_check:
            continue

        if entity_type == NAPISTU_GRAPH.EDGES:
            existing_attributes = napistu_graph.es.attributes()
            entity_sequence = napistu_graph.es
        else:
            existing_attributes = napistu_graph.vs.attributes()
            entity_sequence = napistu_graph.vs

        attrs_to_check = attributes_to_check[entity_type]
        to_be_removed_attributes = set()

        for attr, expected_value in attrs_to_check.items():
            if attr not in existing_attributes:
                continue
            values = entity_sequence[attr]
            # Filter out NaN values and create set of unique observed values
            non_nan_values = [v for v in values if not pd.isna(v)]
            unique_values = set(non_nan_values)
            # Remove if all values are NaN or if only expected_value is present
            if len(unique_values) == 0 or unique_values == {expected_value}:
                to_be_removed_attributes.add(attr)

        if len(to_be_removed_attributes) > 0:
            entity_name = SINGULAR_GRAPH_ENTITIES[entity_type]
            logger.info(
                f"Removing constant or all-missing {entity_name} attributes: {to_be_removed_attributes}"
            )
            napistu_graph.remove_attributes(entity_type, to_be_removed_attributes)

    return None


def _napistu_graph_to_edge_masked_napistu_data(
    napistu_graph: NapistuGraph,
    name: str,
    vertex_default_transforms: Union[
        Dict[str, Dict], EncodingManager
    ] = VERTEX_DEFAULT_TRANSFORMS,
    vertex_transforms: Optional[Union[Dict[str, Dict], EncodingManager]] = None,
    edge_default_transforms: Union[
        Dict[str, Dict], EncodingManager
    ] = EDGE_DEFAULT_TRANSFORMS,
    edge_transforms: Optional[Union[Dict[str, Dict], EncodingManager]] = None,
    auto_encode: bool = True,
    encoders: Dict[str, Any] = DEFAULT_ENCODERS,
    deduplicate_features: bool = True,
    train_size: float = 0.7,
    test_size: float = 0.15,
    val_size: float = 0.15,
    verbose: bool = True,
    labels: Optional[torch.Tensor] = None,
    labeling_manager: Optional[LabelingManager] = None,
) -> NapistuData:
    """NapistuGraph to NapistuData object with edge masks split across train, test, and validation edge sets."""

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
    encoded_vertices, vertex_feature_names, vertex_feature_name_aliases = (
        encoding.encode_dataframe(
            vertex_df,
            vertex_encoding_manager,
            deduplicate=deduplicate_features,
            verbose=verbose,
        )
    )

    # 3. Split vertices into train/test/val
    edge_splits = train_test_val_split(
        edge_df,
        train_size=train_size,
        test_size=test_size,
        val_size=val_size,
        return_dict=True,
    )

    # 4. Create masks (one mask per split, all same length as vertex_df)
    masks = create_split_masks(edge_df, edge_splits)

    # 5. Fit encoders on just the training split
    fitted_edge_encoders = encoding.fit_encoders(
        edge_splits[TRAINING.TRAIN],  # Fit on train only!
        edge_encoding_manager,
        verbose=verbose,
    )

    # 6. Transform all edges
    encoded_edges, edge_feature_names, edge_feature_name_aliases = (
        encoding.transform_dataframe(
            edge_df,
            fitted_edge_encoders,
            deduplicate=deduplicate_features,  # Transform all edges
        )
    )

    # 7. Create edge index from all edges
    edge_index = torch.tensor(
        edge_df[[IGRAPH_DEFS.SOURCE, IGRAPH_DEFS.TARGET]].values.T, dtype=torch.long
    )

    # 8. Extract original edge weights
    edge_weights = _extract_edge_weights(edge_df)

    # 9. Extract minimal NapistuGraph attributes for debugging/validation
    ng_vertex_names, ng_edge_names = _get_napistu_graph_names(vertex_df, edge_df)

    # 10. Create NapistuData object
    return NapistuData(
        x=torch.tensor(encoded_vertices, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=torch.tensor(encoded_edges, dtype=torch.float),
        edge_weight=edge_weights,
        vertex_feature_names=vertex_feature_names,
        edge_feature_names=edge_feature_names,
        vertex_feature_name_aliases=vertex_feature_name_aliases,
        edge_feature_name_aliases=edge_feature_name_aliases,
        ng_vertex_names=ng_vertex_names,
        ng_edge_names=ng_edge_names,
        y=labels,
        name=name,
        splitting_strategy=SPLITTING_STRATEGIES.EDGE_MASK,
        labeling_manager=labeling_manager,
        **masks,  # Unpack train_mask, test_mask, val_mask
    )


def _napistu_graph_to_inductive_napistu_data(
    napistu_graph: NapistuGraph,
    name: str,
    vertex_default_transforms: Union[
        Dict[str, Dict], EncodingManager
    ] = VERTEX_DEFAULT_TRANSFORMS,
    vertex_transforms: Optional[Union[Dict[str, Dict], EncodingManager]] = None,
    edge_default_transforms: Union[
        Dict[str, Dict], EncodingManager
    ] = EDGE_DEFAULT_TRANSFORMS,
    edge_transforms: Optional[Union[Dict[str, Dict], EncodingManager]] = None,
    encoders: Dict[str, Any] = DEFAULT_ENCODERS,
    auto_encode: bool = True,
    deduplicate_features: bool = True,
    train_size: float = 0.7,
    test_size: float = 0.15,
    val_size: float = 0.15,
    verbose: bool = True,
    labels: Optional[torch.Tensor] = None,
    labeling_manager: Optional[LabelingManager] = None,
) -> Dict[str, NapistuData]:
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
    vertex_features, vertex_feature_names, vertex_feature_name_aliases = (
        encoding.encode_dataframe(
            vertex_df,
            vertex_encoding_manager,
            deduplicate=deduplicate_features,
            verbose=verbose,
        )
    )

    # 3. split edges into train/test/val
    edge_splits = train_test_val_split(
        edge_df,
        train_size=train_size,
        test_size=test_size,
        val_size=val_size,
        return_dict=True,
    )

    # 4. fit encoders to the training edges
    edge_encoder = encoding.fit_encoders(
        edge_splits[TRAINING.TRAIN], edge_encoding_manager
    )

    pyg_data = dict()
    for k, edges in edge_splits.items():
        # encode each strata using the train encoder
        edge_features, edge_feature_names, edge_feature_name_aliases = (
            encoding.transform_dataframe(
                edges, edge_encoder, deduplicate=deduplicate_features
            )
        )

        # 5. Reformat the NapistuGraph's edgelist as from-to indices
        edge_index = torch.tensor(
            edges[[IGRAPH_DEFS.SOURCE, IGRAPH_DEFS.TARGET]].values.T, dtype=torch.long
        )

        # 6. Extract original edge weights for this split
        edge_weights = _extract_edge_weights(edges)

        # 7. Extract minimal NapistuGraph attributes for debugging/validation
        ng_vertex_names, ng_edge_names = _get_napistu_graph_names(vertex_df, edges)

        # 8. Create NapistuData
        pyg_data[k] = NapistuData(
            x=torch.tensor(vertex_features, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=torch.tensor(edge_features, dtype=torch.float),
            edge_weight=edge_weights,
            ng_vertex_names=ng_vertex_names,
            ng_edge_names=ng_edge_names,
            vertex_feature_names=vertex_feature_names,
            edge_feature_names=edge_feature_names,
            vertex_feature_name_aliases=vertex_feature_name_aliases,
            edge_feature_name_aliases=edge_feature_name_aliases,
            y=labels,
            name=f"{name}_{k}",
            splitting_strategy=SPLITTING_STRATEGIES.INDUCTIVE,
            labeling_manager=labeling_manager,
        )

    return pyg_data


def _napistu_graph_to_unmasked_napistu_data(
    napistu_graph: NapistuGraph,
    name: str,
    vertex_transforms: Optional[Union[Dict[str, Dict], EncodingManager]] = None,
    edge_transforms: Optional[Union[Dict[str, Dict], EncodingManager]] = None,
    vertex_default_transforms: Union[
        Dict[str, Dict], EncodingManager
    ] = VERTEX_DEFAULT_TRANSFORMS,
    edge_default_transforms: Union[
        Dict[str, Dict], EncodingManager
    ] = EDGE_DEFAULT_TRANSFORMS,
    encoders: Dict[str, Any] = DEFAULT_ENCODERS,
    auto_encode: bool = True,
    deduplicate_features: bool = True,
    verbose: bool = False,
    labels: Optional[torch.Tensor] = None,
    labeling_manager: Optional[LabelingManager] = None,
) -> NapistuData:
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
    vertex_features, vertex_feature_names, vertex_feature_name_aliases = (
        encoding.encode_dataframe(
            vertex_df,
            vertex_encoding_manager,
            deduplicate=deduplicate_features,
            verbose=verbose,
        )
    )
    edge_features, edge_feature_names, edge_feature_name_aliases = (
        encoding.encode_dataframe(
            edge_df,
            edge_encoding_manager,
            deduplicate=deduplicate_features,
            verbose=verbose,
        )
    )

    # 3. Reformat the NapistuGraph's edgelist as from-to indices
    edge_index = torch.tensor(
        [[e.source, e.target] for e in napistu_graph.es], dtype=torch.long
    ).T

    # 4. Extract original edge weights
    edge_weights = _extract_edge_weights(edge_df)

    # 5. Extract minimal NapistuGraph attributes for debugging/validation
    ng_vertex_names, ng_edge_names = _get_napistu_graph_names(vertex_df, edge_df)

    # 6. Create NapistuData
    return NapistuData(
        x=torch.tensor(vertex_features, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_features, dtype=torch.float),
        edge_weight=edge_weights,
        vertex_feature_names=vertex_feature_names,
        edge_feature_names=edge_feature_names,
        vertex_feature_name_aliases=vertex_feature_name_aliases,
        edge_feature_name_aliases=edge_feature_name_aliases,
        y=labels,
        ng_vertex_names=ng_vertex_names,
        ng_edge_names=ng_edge_names,
        name=name,
        splitting_strategy=SPLITTING_STRATEGIES.NO_MASK,
        labeling_manager=labeling_manager,
    )


def _napistu_graph_to_vertex_masked_napistu_data(
    napistu_graph: NapistuGraph,
    name: str,
    vertex_default_transforms: Union[
        Dict[str, Dict], EncodingManager
    ] = VERTEX_DEFAULT_TRANSFORMS,
    vertex_transforms: Optional[Union[Dict[str, Dict], EncodingManager]] = None,
    edge_default_transforms: Union[
        Dict[str, Dict], EncodingManager
    ] = EDGE_DEFAULT_TRANSFORMS,
    edge_transforms: Optional[Union[Dict[str, Dict], EncodingManager]] = None,
    encoders: Dict[str, Any] = DEFAULT_ENCODERS,
    auto_encode: bool = True,
    deduplicate_features: bool = True,
    train_size: float = 0.7,
    test_size: float = 0.15,
    val_size: float = 0.15,
    verbose: bool = True,
    labels: Optional[torch.Tensor] = None,
    labeling_manager: Optional[LabelingManager] = None,
) -> NapistuData:
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
    encoded_edges, edge_feature_names, edge_feature_name_aliases = (
        encoding.encode_dataframe(
            edge_df,
            edge_encoding_manager,
            deduplicate=deduplicate_features,
            verbose=verbose,
        )
    )

    # 3. Split vertices into train/test/val
    vertex_splits = train_test_val_split(
        vertex_df,
        train_size=train_size,
        test_size=test_size,
        val_size=val_size,
        return_dict=True,
    )

    # 4. Create masks (one mask per split, all same length as vertex_df)
    masks = create_split_masks(vertex_df, vertex_splits)

    # 5. Fit encoders on just the training split
    fitted_vertex_encoders = encoding.fit_encoders(
        vertex_splits[TRAINING.TRAIN],  # Fit on train only!
        vertex_encoding_manager,
        verbose=verbose,
    )

    # 6. Transform all vertices
    vertex_features, vertex_feature_names, vertex_feature_name_aliases = (
        encoding.transform_dataframe(
            vertex_df,
            fitted_vertex_encoders,
            deduplicate=deduplicate_features,  # Transform all vertices
        )
    )

    # 7. Create edge index from all edges
    edge_index = torch.tensor(
        edge_df[[IGRAPH_DEFS.SOURCE, IGRAPH_DEFS.TARGET]].values.T, dtype=torch.long
    )

    # 8. Extract original edge weights
    edge_weights = _extract_edge_weights(edge_df)

    # 9. Extract minimal NapistuGraph attributes for debugging/validation
    ng_vertex_names, ng_edge_names = _get_napistu_graph_names(vertex_df, edge_df)

    # 10. Create NapistuData object
    return NapistuData(
        x=torch.tensor(vertex_features, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=torch.tensor(encoded_edges, dtype=torch.float),
        edge_weight=edge_weights,
        vertex_feature_names=vertex_feature_names,
        edge_feature_names=edge_feature_names,
        vertex_feature_name_aliases=vertex_feature_name_aliases,
        edge_feature_name_aliases=edge_feature_name_aliases,
        ng_vertex_names=ng_vertex_names,
        ng_edge_names=ng_edge_names,
        y=labels,  # NapistuData will handle None gracefully
        name=name,
        splitting_strategy=SPLITTING_STRATEGIES.VERTEX_MASK,
        labeling_manager=labeling_manager,
        **masks,  # Unpack train_mask, test_mask, val_mask
    )


def _name_napistu_data(
    splitting_strategy: str,
    labels: Optional[torch.Tensor] = None,
    labeling_manager: Optional[LabelingManager] = None,
) -> str:
    """
    Generate a descriptive name for NapistuData objects based on labeling and splitting strategy.

    This function creates a filename-friendly name that includes information about:
    - The labeling strategy (if supervised)
    - The splitting strategy used

    Parameters
    ----------
    splitting_strategy : Optional[str], default=None
        The splitting strategy used (e.g., 'vertex_mask', 'edge_mask', 'no_mask', 'inductive').
    labels: Optional[torch.Tensor], default=None
        The labels for the data. If None, indicates unlabeled data.
    labeling_manager : Optional[LabelingManager], default=None
        The labeling manager used for supervised tasks. If provided, this can improve the labels of data objects supporting supervised learning.

    Returns
    -------
    str
        A descriptive name suitable for use as a filename.

    Examples
    --------
    >>> # Supervised data with vertex masking
    >>> name = _name_napistu_data(splitting_strategy="vertex_mask", labels=labels, labeling_manager=labeling_manager)
    >>> print(name)  # "supervised_species_type_vertex_mask"

    >>> # Unlabeled data with no masking
    >>> name = _name_napistu_data(splitting_strategy="no_mask")
    >>> print(name)  # "unlabeled"
    """
    name_parts = []

    # Determine if its labeled or unlabeled
    if labels is not None:
        name_parts.append(LABELING.LABELED)
        if labeling_manager is not None:
            name_parts.append(labeling_manager.label_attribute)
    else:
        name_parts.append(LABELING.UNLABELED)

    # Add splitting strategy
    if splitting_strategy != SPLITTING_STRATEGIES.NO_MASK:
        name_parts.append(splitting_strategy)

    return "_".join(name_parts)


def _get_napistu_graph_names(
    vertex_df: pd.DataFrame, edge_df: pd.DataFrame
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Extract minimal NapistuGraph attributes for NapistuData storage.

    This function extracts only the essential attributes needed for debugging
    and validation, keeping file sizes small while preserving the ability to
    look up the full NapistuGraph when needed.

    Parameters
    ----------
    vertex_df : pd.DataFrame
        Full vertex DataFrame from NapistuGraph
    edge_df : pd.DataFrame
        Full edge DataFrame from NapistuGraph

    Returns
    -------
    tuple[pd.Series, pd.DataFrame]
        - vertex_names: Series of vertex names aligned with tensor rows
        - edge_names: DataFrame with 'from' and 'to' columns aligned with tensor columns


    """
    # Extract vertex names as a Series (aligned with tensor rows)
    if NAPISTU_GRAPH_VERTICES.NAME not in vertex_df.columns:
        raise ValueError("Vertex DataFrame must contain 'name' column")
    vertex_names = vertex_df[NAPISTU_GRAPH_VERTICES.NAME].copy()

    # Extract edge from/to names as a DataFrame (aligned with tensor columns)
    if (
        NAPISTU_GRAPH_EDGES.FROM not in edge_df.columns
        or NAPISTU_GRAPH_EDGES.TO not in edge_df.columns
    ):
        raise ValueError("Edge DataFrame must contain 'from' and 'to' columns")
    edge_names = edge_df[[NAPISTU_GRAPH_EDGES.FROM, NAPISTU_GRAPH_EDGES.TO]].copy()

    logger.debug(
        f"Extracted {len(vertex_names)} vertex names and {len(edge_names)} edge name pairs"
    )

    return vertex_names, edge_names


def _standardize_graph_dfs_and_encodings(
    napistu_graph: NapistuGraph,
    vertex_default_transforms: Union[Dict[str, Dict], EncodingManager],
    vertex_transforms: Optional[Union[Dict[str, Dict], EncodingManager]],
    edge_default_transforms: Union[Dict[str, Dict], EncodingManager],
    edge_transforms: Optional[Union[Dict[str, Dict], EncodingManager]],
    auto_encode: bool,
    encoders: Dict[str, Any] = DEFAULT_ENCODERS,
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
    SPLITTING_STRATEGIES.EDGE_MASK: _napistu_graph_to_edge_masked_napistu_data,
    SPLITTING_STRATEGIES.VERTEX_MASK: _napistu_graph_to_vertex_masked_napistu_data,
    SPLITTING_STRATEGIES.NO_MASK: _napistu_graph_to_unmasked_napistu_data,
    SPLITTING_STRATEGIES.INDUCTIVE: _napistu_graph_to_inductive_napistu_data,
}
