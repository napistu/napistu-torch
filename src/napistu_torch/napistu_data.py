"""
NapistuData - A PyTorch Geometric Data subclass for Napistu networks.

This class extends PyG's Data class with Napistu-specific functionality
including safe save/load methods and additional utilities.
"""

import copy
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from napistu.network.constants import (
    NAPISTU_GRAPH,
    NAPISTU_GRAPH_EDGES,
    NAPISTU_GRAPH_VERTICES,
)
from napistu.network.ng_core import NapistuGraph
from torch_geometric.data import Data

from napistu_torch.constants import (
    NAPISTU_DATA,
    NAPISTU_DATA_DEFAULT_NAME,
)
from napistu_torch.labels.apply import decode_labels
from napistu_torch.labels.labeling_manager import LabelingManager
from napistu_torch.load.constants import (
    EDGE_DEFAULT_TRANSFORMS,
    ENCODING_MANAGER,
    VALID_SPLITTING_STRATEGIES,
    VERTEX_DEFAULT_TRANSFORMS,
)
from napistu_torch.load.encoders import DEFAULT_ENCODERS
from napistu_torch.load.encoding import (
    expand_deduplicated_features,
    fit_encoders,
    transform_dataframe,
)
from napistu_torch.load.encoding_manager import EncodingManager
from napistu_torch.ml.constants import DEVICE

logger = logging.getLogger(__name__)


class NapistuData(Data):
    """
    A PyTorch Geometric Data subclass for Napistu biological networks.

    This class extends PyG's Data class with Napistu-specific functionality
    including safe save/load methods and additional utilities for working
    with biological network data.

    Parameters
    ----------
    x : torch.Tensor
        Node feature matrix with shape [num_nodes, num_node_features]
    edge_index : torch.Tensor
        Graph connectivity in COO format with shape [2, num_edges]
    edge_attr : torch.Tensor
        Edge feature matrix with shape [num_edges, num_edge_features]
    edge_weight : torch.Tensor, optional
        Edge weights tensor with shape [num_edges]
    y : torch.Tensor, optional
        Node labels tensor with shape [num_nodes] for supervised learning tasks
    vertex_feature_names : List[str], optional
        Names of vertex features for interpretability
    edge_feature_names : List[str], optional
        Names of edge features for interpretability
    vertex_feature_name_aliases : Dict[str, str], optional
        Mapping from vertex feature names to their canonical names (for deduplicated features)
    edge_feature_name_aliases : Dict[str, str], optional
        Mapping from edge feature names to their canonical names (for deduplicated features)
    ng_vertex_names : pd.Series, optional
        Minimal vertex names from the original NapistuGraph. Series aligned with
        the vertex tensor (x) - each element corresponds to a vertex in the same
        order as the tensor rows. Used for debugging and validation of tensor alignment.
    ng_edge_names : pd.DataFrame, optional
        Minimal edge names from the original NapistuGraph. DataFrame with 'from' and 'to'
        columns aligned with the edge tensor (edge_index, edge_attr) - each row corresponds
        to an edge in the same order as the tensor columns. Used for debugging and validation.
    name: str = NAPISTU_DATA_DEFAULT_NAME,
        Name of the NapistuData object. Used for summaries and for organizing objects in the NapistuDataStore.
    splitting_strategy: Optional[str] = None,
        Strategy used to split the data into train/test/val sets. This occurs upstream but the approach is tracked as a reference here.
    labeling_manager: Optional[LabelingManager] = None,
        Labeling manager used to encode the labels. This is used to decode the labels back to the original values for validation purposes.
    **kwargs
        Additional attributes to store in the data object

    Public Methods
    --------------
    copy()
        Create a deep copy of the NapistuData object
    estimate_memory_footprint()
        Estimate memory footprint of the NapistuData object
    get_edge_feature_names()
        Get the names of edge features
    get_edge_names()
        Get the edge names from the original NapistuGraph
    get_edge_weights()
        Get edge weights as a 1D tensor
    get_feature_by_name(feature_name)
        Get a feature by name from the NapistuData object
    get_vertex_feature_names()
        Get the names of vertex features
    get_vertex_names()
        Get the vertex names from the original NapistuGraph
    load(filepath, map_location="cpu")
        Load a NapistuData object from disk
    save(filepath)
        Save the NapistuData object to disk
    show_memory_footprint()
        Display memory footprint of the NapistuData object
    summary()
        Get a summary of the NapistuData object
    trim(keep_edge_attr=True, keep_labels=True, keep_masks=True, inplace=False)
        Trim the NapistuData object to keep only the specified attributes
    unencode_features(napistu_graph, attribute_type, attribute, encoding_manager=None)
        Unencode features from the NapistuData object

    Private Methods
    ---------------
    _validate_edge_encoding(napistu_graph, edge_attribute, encoding_manager=None)
        Validate the edge encoding of the NapistuData object
    _validate_labels(napistu_graph, labeling_manager)
        Validate the labels of the NapistuData object
    _validate_vertex_encoding(napistu_graph, vertex_attribute, encoding_manager=None)
        Validate the vertex encoding of the NapistuData object


    Examples
    --------
    >>> # Create a NapistuData object (x, edge_index, and edge_attr are required)
    >>> data = NapistuData(
    ...     x=torch.randn(100, 10),                    # Required: node features
    ...     edge_index=torch.randint(0, 100, (2, 200)), # Required: graph connectivity
    ...     edge_attr=torch.randn(200, 5),             # Required: edge features
    ...     y=torch.randint(0, 3, (100,)),             # Optional: node labels
    ...     vertex_feature_names=['feature_1', 'feature_2', ...],  # Optional
    ...     edge_feature_names=['weight', 'direction', ...],       # Optional
    ...     ng_vertex_names=vertex_names_series,        # Optional: minimal vertex names
    ...     ng_edge_names=edge_names_df,                # Optional: minimal edge names
    ... )
    >>>
    >>> # Save and load
    >>> data.save('my_network.pt')
    >>> loaded_data = NapistuData.load('my_network.pt')
    """

    def __init__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        vertex_feature_names: Optional[List[str]] = None,
        edge_feature_names: Optional[List[str]] = None,
        vertex_feature_name_aliases: Optional[Dict[str, str]] = None,
        edge_feature_name_aliases: Optional[Dict[str, str]] = None,
        ng_vertex_names: Optional[pd.Series] = None,
        ng_edge_names: Optional[pd.DataFrame] = None,
        name: str = NAPISTU_DATA_DEFAULT_NAME,
        splitting_strategy: Optional[str] = None,
        labeling_manager: Optional[LabelingManager] = None,
        **kwargs,
    ):
        # Validate required parameters
        if x is None:
            raise ValueError("x (node feature matrix) is required and cannot be None")
        if edge_index is None:
            raise ValueError(
                "edge_index (graph connectivity) is required and cannot be None"
            )
        if edge_attr is None:
            raise ValueError(
                "edge_attr (edge feature matrix) is required and cannot be None"
            )
        if name is None or not isinstance(name, str):
            raise ValueError(
                "name (name of the NapistuData object) is required and must be a string"
            )

        # Build parameters dict, only including non-None values
        params = {
            NAPISTU_DATA.X: x,
            NAPISTU_DATA.EDGE_INDEX: edge_index,
            NAPISTU_DATA.EDGE_ATTR: edge_attr,
            NAPISTU_DATA.EDGE_WEIGHT: edge_weight,
            NAPISTU_DATA.NAME: name,
        }

        # Add optional parameters if they are not None
        if y is not None:
            if not isinstance(y, torch.Tensor):
                raise ValueError("if provided, y (node labels) must be a torch.Tensor")
            params[NAPISTU_DATA.Y] = y

        if vertex_feature_names is not None:
            if not isinstance(vertex_feature_names, list):
                raise ValueError(
                    "if provided, vertex_feature_names must be a list of strings"
                )
            if len(vertex_feature_names) != x.shape[1]:
                raise ValueError(
                    "if provided, vertex_feature_names must be a list of strings with the same length as the number of columns in x"
                )
            params[NAPISTU_DATA.VERTEX_FEATURE_NAMES] = vertex_feature_names

        if vertex_feature_name_aliases is not None:
            if not isinstance(vertex_feature_name_aliases, dict):
                raise ValueError(
                    "if provided, vertex_feature_name_aliases must be a dict"
                )
            params[NAPISTU_DATA.VERTEX_FEATURE_NAME_ALIASES] = (
                vertex_feature_name_aliases
            )

        if edge_feature_name_aliases is not None:
            if not isinstance(edge_feature_name_aliases, dict):
                raise ValueError(
                    "if provided, edge_feature_name_aliases must be a dict"
                )
            params[NAPISTU_DATA.EDGE_FEATURE_NAME_ALIASES] = edge_feature_name_aliases

        if edge_feature_names is not None:
            if not isinstance(edge_feature_names, list):
                raise ValueError(
                    "if provided, edge_feature_names must be a list of strings"
                )
            if len(edge_feature_names) != edge_attr.shape[1]:
                raise ValueError(
                    "if provided, edge_feature_names must be a list of strings with the same length as the number of columns in edge_attr"
                )
            params[NAPISTU_DATA.EDGE_FEATURE_NAMES] = edge_feature_names

        if ng_vertex_names is not None:
            if not isinstance(ng_vertex_names, pd.Series):
                raise ValueError("if provided, ng_vertex_names must be a pd.Series")
            if ng_vertex_names.name != NAPISTU_GRAPH_VERTICES.NAME:
                raise ValueError(
                    "if provided, ng_vertex_names must have a name attribute of 'name'"
                )
            params[NAPISTU_DATA.NG_VERTEX_NAMES] = ng_vertex_names

        if ng_edge_names is not None:
            if not isinstance(ng_edge_names, pd.DataFrame):
                raise ValueError("if provided, ng_edge_names must be a pd.DataFrame")
            if ng_edge_names.shape[1] != 2:
                raise ValueError("if provided, ng_edge_names must have 2 columns")
            EXPECTED_EDGE_NAMES = [NAPISTU_GRAPH_EDGES.FROM, NAPISTU_GRAPH_EDGES.TO]
            if not all(col in ng_edge_names.columns for col in EXPECTED_EDGE_NAMES):
                raise ValueError(
                    f"if provided, ng_edge_names must have columns '{EXPECTED_EDGE_NAMES}'"
                )
            params[NAPISTU_DATA.NG_EDGE_NAMES] = ng_edge_names

        if splitting_strategy is not None:
            if not isinstance(splitting_strategy, str):
                raise ValueError("if provided, splitting_strategy must be a string")
            if splitting_strategy not in VALID_SPLITTING_STRATEGIES:
                raise ValueError(
                    f"if provided, splitting_strategy must be one of {VALID_SPLITTING_STRATEGIES}"
                )
            params[NAPISTU_DATA.SPLITTING_STRATEGY] = splitting_strategy

        if labeling_manager is not None:
            if not isinstance(labeling_manager, LabelingManager):
                raise ValueError(
                    "if provided, labeling_manager must be a LabelingManager object"
                )
            if y is None:
                logger.warning(
                    "Labeling manager provided but no labels are present in the data. The labeling manager will be ignored."
                )
            else:
                params[NAPISTU_DATA.LABELING_MANAGER] = labeling_manager

        # Add any non-None kwargs
        params.update({k: v for k, v in kwargs.items() if v is not None})

        super().__init__(**params)

    def copy(self) -> "NapistuData":
        """
        Create a deep copy of the NapistuData object.
        """
        return copy.deepcopy(self)

    def estimate_memory_footprint(self) -> Dict[str, Optional[int]]:
        """
        Estimate memory footprint of the NapistuData object.

        Calculates the memory usage (in bytes) for each major component
        of the data object, including node features, edge index, edge
        attributes, and training/validation/test masks.

        Returns
        -------
        Dict[str, Optional[int]]
            Dictionary containing memory usage in bytes for each component:
            - "node_features": Memory used by node features (x)
            - "edge_index": Memory used by edge index
            - "edge_attr": Memory used by edge attributes
            - "train_mask": Memory used by train mask (None if not present)
            - "val_mask": Memory used by validation mask (None if not present)
            - "test_mask": Memory used by test mask (None if not present)
            - "total": Total memory usage in bytes

        Examples
        --------
        >>> footprint = data.estimate_memory_footprint()
        >>> print(f"Total memory: {footprint['total'] / 1e9:.2f} GB")
        >>> print(f"Node features: {footprint['node_features'] / 1e9:.2f} GB")
        """
        memory_dict: Dict[str, Optional[int]] = {
            NAPISTU_DATA.X: None,
            NAPISTU_DATA.EDGE_INDEX: None,
            NAPISTU_DATA.EDGE_ATTR: None,
            NAPISTU_DATA.TRAIN_MASK: None,
            NAPISTU_DATA.VAL_MASK: None,
            NAPISTU_DATA.TEST_MASK: None,
            "total": 0,
        }

        total_bytes = 0

        # Node features
        if hasattr(self, NAPISTU_DATA.X) and self.x is not None:
            node_bytes = self.x.element_size() * self.x.nelement()
            memory_dict[NAPISTU_DATA.X] = node_bytes
            total_bytes += node_bytes

        # Edge index
        if hasattr(self, NAPISTU_DATA.EDGE_INDEX) and self.edge_index is not None:
            edge_index_bytes = (
                self.edge_index.element_size() * self.edge_index.nelement()
            )
            memory_dict[NAPISTU_DATA.EDGE_INDEX] = edge_index_bytes
            total_bytes += edge_index_bytes

        # Edge attributes
        if hasattr(self, NAPISTU_DATA.EDGE_ATTR) and self.edge_attr is not None:
            edge_attr_bytes = self.edge_attr.element_size() * self.edge_attr.nelement()
            memory_dict[NAPISTU_DATA.EDGE_ATTR] = edge_attr_bytes
            total_bytes += edge_attr_bytes

        # Masks
        for mask_name in [
            NAPISTU_DATA.TRAIN_MASK,
            NAPISTU_DATA.VAL_MASK,
            NAPISTU_DATA.TEST_MASK,
        ]:
            if hasattr(self, mask_name):
                mask = getattr(self, mask_name)
                if mask is not None:
                    mask_bytes = mask.element_size() * mask.nelement()
                    memory_dict[mask_name] = mask_bytes
                    total_bytes += mask_bytes

        memory_dict["total"] = total_bytes
        return memory_dict

    def get_edge_feature_names(self) -> Optional[List[str]]:
        """
        Get the names of edge features.

        Returns
        -------
        Optional[List[str]]
            List of edge feature names, or None if not available
        """
        result = getattr(self, NAPISTU_DATA.EDGE_FEATURE_NAMES, None)
        if result is None:
            logger.warning(
                "Edge feature names not found in NapistuData. "
                f"Attribute '{NAPISTU_DATA.EDGE_FEATURE_NAMES}' is missing."
            )
        return result

    def get_edge_names(self) -> Optional[pd.Index]:
        """
        Get the edge names as a pandas Index.

        Returns
        -------
        Optional[pd.Index]
            Pandas Index of edge names, or None if not available
        """
        result = getattr(self, NAPISTU_DATA.NG_EDGE_NAMES, None)
        if result is None:
            logger.warning(
                "Edge names not found in NapistuData. "
                f"Attribute '{NAPISTU_DATA.NG_EDGE_NAMES}' is missing."
            )
        return result

    def get_edge_weights(self) -> Optional[torch.Tensor]:
        """
        Get edge weights as a 1D tensor.

        This method provides access to the original edge weights stored in the
        edge_weight attribute, which is the standard PyG convention for scalar
        edge weights.

        Returns
        -------
        Optional[torch.Tensor]
            1D tensor of edge weights, or None if not available

        Examples
        --------
        >>> weights = data.get_edge_weights()
        >>> if weights is not None:
        ...     print(f"Edge weights shape: {weights.shape}")
        ...     print(f"Mean weight: {weights.mean():.3f}")
        """
        result = getattr(self, NAPISTU_DATA.EDGE_WEIGHT, None)
        if result is None:
            logger.warning(
                "Edge weights not found in NapistuData. "
                f"Attribute '{NAPISTU_DATA.EDGE_WEIGHT}' is missing."
            )
        return result

    def get_feature_by_name(self, feature_name: str) -> torch.Tensor:
        """
        Get a feature by name from the NapistuData object.

        Parameters
        ----------
        feature_name : str
            The name of the feature to get

        Returns
        -------
        torch.Tensor
            The feature tensor
        """
        vertex_feature_names = self.get_vertex_feature_names()
        if feature_name not in vertex_feature_names:
            raise ValueError(
                f"Feature name {feature_name} not found in NapistuData vertex feature names"
            )
        feature_idx = vertex_feature_names.index(feature_name)
        return self.x[:, feature_idx]

    def get_features_by_regex(
        self, regex: str, return_suffixes: bool = False
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Get features by regex from the NapistuData object.

        Parameters
        ----------
        regex: str
            The regex to search for in the vertex feature names
        return_suffixes: bool
            If True, return the substring following the regex as feature_names

        Returns:
        Tuple[torch.Tensor, List[str]]
            The features tensor and feature names

            features: torch.Tensor
                The features tensor
            feature_names: List[str]
                The feature names

        Examples
        --------
        >>> features, feature_names = napistu_data.get_features_by_regex("__source")
        >>> print(features.shape)
        >>> print(feature_names)
        (100, 5)
        ['source_1', 'source_2', 'source_3', 'source_4', 'source_5']
        """

        vertex_feature_names = self.get_vertex_feature_names()
        mask = [bool(re.search(regex, x)) for x in vertex_feature_names]
        if not any(mask):
            raise ValueError(f"No features found with regex {regex}")

        matching_feature_names = [
            name for name, match in zip(vertex_feature_names, mask) if match
        ]
        if return_suffixes:
            # Check if regex already has capturing groups - if so, throw an exception
            if "(" in regex and ")" in regex:
                raise ValueError(
                    f"Regex '{regex}' already contains capturing groups. When return_suffixes=True, the regex should not contain parentheses."
                )

            # No capturing groups, create one for the suffix
            matching_feature_names = [
                re.search(regex + r"(.*)", x).group(1) for x in matching_feature_names
            ]
            matching_feature_names = [x.lstrip("_") for x in matching_feature_names]

        indices = [i for i, m in enumerate(mask) if m]
        selected_features = self.x[:, indices]
        return selected_features, matching_feature_names

    def get_vertex_feature_names(self) -> Optional[List[str]]:
        """
        Get the names of vertex features.

        Returns
        -------
        Optional[List[str]]
            List of vertex feature names, or None if not available
        """
        result = getattr(self, NAPISTU_DATA.VERTEX_FEATURE_NAMES, None)
        if result is None:
            logger.warning(
                "Vertex feature names not found in NapistuData. "
                f"Attribute '{NAPISTU_DATA.VERTEX_FEATURE_NAMES}' is missing."
            )
        return result

    def get_vertex_names(self) -> Optional[pd.Index]:
        """
        Get the vertex names as a pandas Index.

        Returns
        -------
        Optional[pd.Index]
            Pandas Index of vertex names, or None if not available
        """
        result = getattr(self, NAPISTU_DATA.NG_VERTEX_NAMES, None)
        if result is None:
            logger.warning(
                "Vertex names not found in NapistuData. "
                f"Attribute '{NAPISTU_DATA.NG_VERTEX_NAMES}' is missing."
            )
        return result

    @classmethod
    def load(
        cls, filepath: Union[str, Path], map_location: str = DEVICE.CPU
    ) -> "NapistuData":
        """
        Load a NapistuData object from disk.

        This method automatically uses weights_only=False to ensure compatibility
        with PyG Data objects, which contain custom classes that aren't allowed
        with the default weights_only=True setting in PyTorch 2.6+.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path to the saved data object
        map_location : str, default='cpu'
            Device to map tensors to (e.g., 'cpu', 'cuda:0'). Defaults to 'cpu'
            for universal compatibility.

        Returns
        -------
        NapistuData
            The loaded NapistuData object

        Raises
        ------
        FileNotFoundError
            If the file doesn't exist
        RuntimeError
            If loading fails
        TypeError
            If the loaded object is not a NapistuData or Data object

        Examples
        --------
        >>> data = NapistuData.load('my_network.pt')  # Loads to CPU by default
        >>> data = NapistuData.load('my_network.pt', map_location='cuda:0')  # Load to GPU

        Notes
        -----
        This method uses weights_only=False by default because PyG Data objects
        contain custom classes that aren't allowed with weights_only=True.
        Only use this with trusted files, as it can result in arbitrary code execution.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File n?ot found: {filepath}")

        try:
            # Always use weights_only=False for PyG compatibility
            data = torch.load(filepath, weights_only=False, map_location=map_location)

            # Convert to NapistuData if it's a regular Data object
            if isinstance(data, NapistuData):
                return data
            else:
                raise TypeError(
                    f"Loaded object is not a NapistuData object, got {type(data)}. "
                    "This may indicate a corrupted file or incorrect file type."
                )

        except Exception as e:
            raise RuntimeError(
                f"Failed to load NapistuData object from {filepath}: {e}"
            ) from e

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the NapistuData object to disk.

        This method provides a safe way to save NapistuData objects, ensuring
        compatibility with PyTorch's security features.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path where to save the data object

        Examples
        --------
        >>> data.save('my_network.pt')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self, filepath)

    def show_memory_footprint(self) -> None:
        """
        Display memory footprint of the NapistuData object.

        Prints a formatted breakdown of memory usage for each component
        of the data object in gigabytes (GB), showing node features,
        edge index, edge attributes, and training/validation/test masks.

        Examples
        --------
        >>> data.show_memory_footprint()
        Node features: 0.05 GB
        Edge index: 0.01 GB
        Edge attributes: 0.20 GB
        train_mask: 0.00 GB
        val_mask: 0.00 GB
        test_mask: 0.00 GB

        Total data: 0.26 GB
        """
        memory_dict = self.estimate_memory_footprint()

        # Node features
        if memory_dict[NAPISTU_DATA.X] is not None:
            print(f"Node features: {memory_dict[NAPISTU_DATA.X] / 1e9:.2f} GB")

        # Edge index
        if memory_dict[NAPISTU_DATA.EDGE_INDEX] is not None:
            print(f"Edge index: {memory_dict[NAPISTU_DATA.EDGE_INDEX] / 1e9:.2f} GB")

        # Edge attributes
        if memory_dict[NAPISTU_DATA.EDGE_ATTR] is not None:
            print(
                f"Edge attributes: {memory_dict[NAPISTU_DATA.EDGE_ATTR] / 1e9:.2f} GB"
            )

        # Masks
        for mask_name in [
            NAPISTU_DATA.TRAIN_MASK,
            NAPISTU_DATA.VAL_MASK,
            NAPISTU_DATA.TEST_MASK,
        ]:
            if memory_dict[mask_name] is not None:
                print(f"{mask_name}: {memory_dict[mask_name] / 1e9:.3f} GB")

        print(f"\nTotal data: {memory_dict['total'] / 1e9:.2f} GB")

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the NapistuData object.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing summary information about the data object
        """
        summary_dict = {
            NAPISTU_DATA.NAME: self.name,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_node_features": self.num_node_features,
            "num_edge_features": self.num_edge_features,
            "has_vertex_feature_names": hasattr(
                self, NAPISTU_DATA.VERTEX_FEATURE_NAMES
            ),
            "has_edge_feature_names": hasattr(self, NAPISTU_DATA.EDGE_FEATURE_NAMES),
            "has_edge_weights": hasattr(self, NAPISTU_DATA.EDGE_WEIGHT),
            "has_ng_vertex_names": hasattr(self, NAPISTU_DATA.NG_VERTEX_NAMES),
            "has_ng_edge_names": hasattr(self, NAPISTU_DATA.NG_EDGE_NAMES),
            "has_splitting_strategy": hasattr(self, NAPISTU_DATA.SPLITTING_STRATEGY),
            "has_labeling_manager": hasattr(self, NAPISTU_DATA.LABELING_MANAGER),
        }

        return summary_dict

    def trim(
        self,
        keep_edge_attr: bool = True,
        keep_labels: bool = True,
        keep_masks: bool = True,
        inplace: bool = False,
    ) -> "NapistuData":
        """
        Create a memory-optimized copy with only essential training attributes.

        This method creates a new NapistuData object with only the core attributes
        needed for training, stripping away all metadata and debugging information.

        **What's Always Kept:**
        - x (node features)
        - edge_index (graph structure)
        - edge_weight (if present)

        **What's Always Removed:**
        - ng_vertex_names, ng_edge_names (pandas objects)
        - vertex_feature_names, edge_feature_names (metadata)
        - name, splitting_strategy, labeling_manager (metadata)

        Parameters
        ----------
        keep_edge_attr : bool, default=True
            Whether to keep edge_attr. Set False if not using edge features
            (e.g., no edge encoder). **Major memory savings for large graphs.**
        keep_labels : bool, default=True
            Whether to keep y (node labels). Set False for unlabeled tasks.
        keep_masks : bool, default=True
            Whether to keep train_mask, val_mask, test_mask.
            Set False if using custom splitting.
        inplace: bool, default=False
            Whether to modify the current object in place or return a new object.

        Returns
        -------
        NapistuData
            New trimmed NapistuData object with minimal attributes

        Examples
        --------
        >>> # Default - keep everything except metadata
        >>> trimmed = data.trim()
        >>>
        >>> # No edge features needed (biggest memory savings)
        >>> trimmed = data.trim(keep_edge_attr=False)
        >>>
        >>> # Unlabeled learning
        >>> trimmed = data.trim(keep_labels=False)
        >>>
        >>> # Check memory savings
        >>> print(f"Before: {data.estimate_memory():.2f} GB")
        >>> print(f"After: {trimmed.estimate_memory():.2f} GB")
        >>>
        >>> # Minimal for inference (no edge features, labels, or masks)
        >>> trimmed = data.trim(
        ...     keep_edge_attr=False,
        ...     keep_labels=False,
        ...     keep_masks=False
        ... )

        Notes
        -----
        **Memory Impact (10M edges example):**
        - Removing edge_attr: saves ~4 GB (100 features) to ~0.4 GB (10 features)
        - Removing pandas names: saves ~10-100 MB
        - Removing labels/masks: saves ~10-50 MB
        """

        new_attrs = {
            NAPISTU_DATA.X: self.x,
            NAPISTU_DATA.EDGE_INDEX: self.edge_index,
            # edge_attr is always included to satisfy NapistuData.__init__ requirements
            # If keep_edge_attr=False, use empty tensor (saves memory while keeping API compatibility)
            NAPISTU_DATA.EDGE_ATTR: (
                self.edge_attr if keep_edge_attr else torch.empty((self.num_edges, 0))
            ),
            NAPISTU_DATA.NAME: NAPISTU_DATA_DEFAULT_NAME + "_trimmed",
        }

        # Add edge_weight if present
        if hasattr(self, NAPISTU_DATA.EDGE_WEIGHT) and self.edge_weight is not None:
            new_attrs[NAPISTU_DATA.EDGE_WEIGHT] = self.edge_weight

        # Add labels if requested
        if keep_labels and hasattr(self, NAPISTU_DATA.Y) and self.y is not None:
            new_attrs[NAPISTU_DATA.Y] = self.y

        # Add masks if requested
        if keep_masks:
            for mask_name in [
                NAPISTU_DATA.TRAIN_MASK,
                NAPISTU_DATA.VAL_MASK,
                NAPISTU_DATA.TEST_MASK,
            ]:
                if hasattr(self, mask_name):
                    mask = getattr(self, mask_name)
                    if mask is not None:
                        new_attrs[mask_name] = mask

        if inplace:
            # Modify the object in place by clearing all attributes and setting new ones
            for key in list(self.keys()):
                delattr(self, key)
            for key, value in new_attrs.items():
                setattr(self, key, value)
            return None
        else:
            return NapistuData(**new_attrs)

    def unencode_features(
        self,
        napistu_graph: NapistuGraph,
        attribute_type: str,
        attribute: str,
        encoding_manager: Optional[EncodingManager] = None,
    ) -> pd.Series:
        """
        Unencode features from the NapistuData object back to the original values.

        This only categorical and passthrough encoding and is useful for validation purposes
        to ensure that encoded features are proprely aligned with their values in their original NapistuGraph.

        Parameters
        ----------
        napistu_graph : NapistuGraph
            The NapistuGraph object containing the original values
        attribute_type : str
            The type of attribute to unencode ("vertices" or "edges")
        attribute : str
            An attribute to unencode (e.g., "node_type" or "species_type")
        encoding_manager : Optional[EncodingManager]
            The encoding manager to use to unencode the features.
            If this is not provided then the default encoding managers will be used.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the unencoded features
        """

        if attribute_type == NAPISTU_GRAPH.VERTICES:
            attribute_values = napistu_graph.get_vertex_series(attribute)
            encoded_features = self.x
            feature_names = self.vertex_feature_names
            feature_name_aliases = self.vertex_feature_name_aliases
        elif attribute_type == NAPISTU_GRAPH.EDGES:
            attribute_values = napistu_graph.get_edge_series(attribute)
            encoded_features = self.edge_attr
            feature_names = self.edge_feature_names
            feature_name_aliases = self.edge_feature_name_aliases
        else:
            raise ValueError(f"Invalid attribute type: {attribute_type}")

        if encoding_manager is None:
            if attribute_type == NAPISTU_GRAPH.VERTICES:
                encoding_manager = EncodingManager(
                    VERTEX_DEFAULT_TRANSFORMS, encoders=DEFAULT_ENCODERS
                )
            elif attribute_type == NAPISTU_GRAPH.EDGES:
                encoding_manager = EncodingManager(
                    EDGE_DEFAULT_TRANSFORMS, encoders=DEFAULT_ENCODERS
                )
        elif not isinstance(encoding_manager, EncodingManager):
            ValueError(
                f"Invalid value for `encoding_manager` it shoudl be either None or an EncodingManager object but was given a {type(encoding_manager)}: {encoding_manager}"
            )

        # Filter encoding manager to only include the attribute of interest
        filtered_config = {}
        for transform_name, transform_config in encoding_manager.config_.items():
            if attribute in transform_config[ENCODING_MANAGER.COLUMNS]:
                # Create new config with only this attribute
                filtered_config[transform_name] = {
                    ENCODING_MANAGER.COLUMNS: [attribute],
                    ENCODING_MANAGER.TRANSFORMER: transform_config[
                        ENCODING_MANAGER.TRANSFORMER
                    ],
                }

        if not filtered_config:
            raise ValueError(f"Attribute '{attribute}' not found in encoding manager")

        # Create filtered encoding manager for just this attribute
        filtered_manager = EncodingManager(filtered_config)

        # Fit the encoder on the single-column DataFrame
        fitted_encoder = fit_encoders(attribute_values.to_frame(), filtered_manager)
        _, actual_transformer, _ = fitted_encoder.transformers_[0]

        # Get feature names for this specific attribute to find column indices
        _, fitted_feature_names, _ = transform_dataframe(
            attribute_values.to_frame(), fitted_encoder, deduplicate=False
        )

        # do we need columns which are only present as aliases?
        if any(name in feature_name_aliases for name in fitted_feature_names):
            # need to expand the encoding
            required_aliases = {
                k: v
                for k, v in feature_name_aliases.items()
                if k in fitted_feature_names
            }
            encoded_features, feature_names = expand_deduplicated_features(
                encoded_features, feature_names, required_aliases
            )

        # Find which columns in encoded_features correspond to this attribute
        col_indices = [feature_names.index(fname) for fname in fitted_feature_names]

        # Extract relevant columns from encoded features
        if isinstance(encoded_features, torch.Tensor):
            relevant_features = encoded_features[:, col_indices].cpu().numpy()
        else:
            relevant_features = encoded_features[:, col_indices]

        # Inverse transform using the actual transformer
        if actual_transformer == ENCODING_MANAGER.PASSTHROUGH:
            # For passthrough, just extract the column directly
            decoded_values = relevant_features.flatten()
        else:
            # For OneHotEncoder and other transformers with inverse_transform
            decoded = actual_transformer.inverse_transform(relevant_features)
            decoded_values = decoded.flatten()

        return pd.Series(decoded_values, name=attribute)

    def _validate_vertex_encoding(
        self,
        napistu_graph: NapistuGraph,
        vertex_attribute: str,
        encoding_manager: Optional[EncodingManager] = None,
    ) -> bool:
        """
        Validate consistency between encoded values and original NapistuGraph vertex values.

        This method compares the vertex values recovered from encoding
        in the NapistuData object with the original vertex values stored in
        the NapistuGraph object to ensure data consistency.

        Parameters
        ----------
        napistu_graph : NapistuGraph
            The NapistuGraph object containing the original categorical values
        categorical_vertex_attribute : str
            The name of the categorical vertex attribute to validate (e.g., 'node_type')

        Returns
        -------
        bool
            True if the encoding is consistent, False otherwise

        Raises
        ------
        ValueError
            If the categorical attribute is not found in the NapistuGraph,
            if vertex names don't match between NapistuData and NapistuGraph,
            or if there are encoding inconsistencies.

        Examples
        --------
        >>> # Validate node_type encoding consistency
        >>> is_consistent = napistu_data._validate_vertex_encoding(napistu_graph, 'node_type')
        >>> print(f"Encoding is consistent: {is_consistent}")
        True

        >>> # Validate a different categorical attribute
        >>> is_consistent = napistu_data._validate_vertex_encoding(napistu_graph, 'species_type')
        >>> print(f"Species type encoding is consistent: {is_consistent}")
        True
        """
        # Get the categorical values from NapistuGraph
        graph_values = napistu_graph.get_vertex_series(vertex_attribute)

        # Get the recovered values from encoding in NapistuData using unencode_features
        data_values = self.unencode_features(
            napistu_graph=napistu_graph,
            attribute_type=NAPISTU_GRAPH.VERTICES,
            attribute=vertex_attribute,
            encoding_manager=encoding_manager,
        )

        # Get vertex names for alignment
        if (
            not hasattr(self, NAPISTU_DATA.NG_VERTEX_NAMES)
            or getattr(self, NAPISTU_DATA.NG_VERTEX_NAMES) is None
        ):
            raise ValueError(
                f"Validation not available - the `{NAPISTU_DATA.NG_VERTEX_NAMES}` attribute is required for this method."
            )
        data_vertex_names = getattr(self, NAPISTU_DATA.NG_VERTEX_NAMES)

        # Align the graph values with the NapistuData vertex ordering
        # Create a DataFrame for easier merging
        graph_df = pd.DataFrame(
            {
                "graph_vertex_name": graph_values.index,
                "graph_vertex_value": graph_values.values,
            }
        )

        # Merge with NapistuData vertex names to get aligned graph values
        # Convert data_vertex_names Series to DataFrame for merging
        data_vertex_names_df = data_vertex_names.to_frame("graph_vertex_name")
        aligned_graph = data_vertex_names_df.merge(
            graph_df, on="graph_vertex_name", how="left"
        )
        graph_values_aligned = aligned_graph["graph_vertex_value"]

        # Debug: Check if we have any matches
        matches_found = aligned_graph["graph_vertex_value"].notna().sum()
        if matches_found == 0:
            raise ValueError(
                f"No matching vertex names found between NapistuData and NapistuGraph. "
                f"NapistuData vertex names: {data_vertex_names.tolist()[:5]}... "
                f"NapistuGraph vertex names: {graph_values.index.tolist()[:5]}..."
            )

        # Create masks for valid (non-null) values in both series
        graph_valid_mask = ~graph_values_aligned.isna()
        data_valid_mask = ~data_values.isna()

        # Check if the non-null masks are identical
        if not graph_valid_mask.equals(data_valid_mask):
            graph_null_count = (~graph_valid_mask).sum()
            data_null_count = (~data_valid_mask).sum()
            raise ValueError(
                f"Non-null masks don't match between graph and data values. "
                f"Graph values non-null count: {graph_valid_mask.sum()}, "
                f"Data values non-null count: {data_valid_mask.sum()}, "
                f"Graph values null count: {graph_null_count}, "
                f"Data values null count: {data_null_count}"
            )

        # Compare only valid values (since masks are identical, we can use either)
        graph_valid = graph_values_aligned[graph_valid_mask]
        data_valid = data_values[graph_valid_mask]

        # Check for exact matches
        matches = graph_valid == data_valid

        if not matches.all():
            # Find mismatches for detailed error reporting
            mismatches = ~matches
            mismatch_indices = matches.index[mismatches]

            mismatch_details = []
            for idx in mismatch_indices:
                graph_val = graph_valid[idx]
                data_val = data_valid[idx]
                vertex_name = data_vertex_names.iloc[
                    data_vertex_names.index.get_loc(idx)
                ]
                mismatch_details.append(
                    f"Vertex '{vertex_name}': graph='{graph_val}', data='{data_val}'"
                )

            raise ValueError(
                f"Encoding validation failed for {vertex_attribute}. "
                f"Found {mismatches.sum()} mismatches out of {len(matches)} valid comparisons:\n"
                + "\n".join(mismatch_details)
            )

        return True

    def _validate_edge_encoding(
        self,
        napistu_graph: NapistuGraph,
        edge_attribute: str,
        encoding_manager: Optional[EncodingManager] = None,
    ) -> bool:
        """
        Validate consistency between encoded values and original NapistuGraph edge values.

        This method compares the edge values recovered from encoding
        in the NapistuData object with the original edge values stored in
        the NapistuGraph object to ensure data consistency.

        Parameters
        ----------
        napistu_graph : NapistuGraph
            The NapistuGraph object containing the original edge values
        edge_attribute : str
            The name of the edge attribute to validate (e.g., 'r_irreversible')
        encoding_manager : Optional[EncodingManager]
            The encoding manager to use to unencode the features.
            If this is not provided then the default encoding managers will be used.

        Returns
        -------
        bool
            True if the encoding is consistent, False otherwise

        Raises
        ------
        ValueError
            If the edge attribute is not found in the NapistuGraph,
            if edge names don't match between NapistuData and NapistuGraph,
            or if there are encoding inconsistencies.

        Examples
        --------
        >>> # Validate r_irreversible encoding consistency
        >>> is_consistent = napistu_data._validate_edge_encoding(napistu_graph, 'r_irreversible')
        >>> print(f"Encoding is consistent: {is_consistent}")
        True

        >>> # Validate a different edge attribute
        >>> is_consistent = napistu_data._validate_edge_encoding(napistu_graph, 'weight')
        >>> print(f"Weight encoding is consistent: {is_consistent}")
        True
        """
        # Get the edge values from NapistuGraph
        graph_values = napistu_graph.get_edge_series(edge_attribute)

        # Get the recovered values from encoding in NapistuData using unencode_features
        data_values = self.unencode_features(
            napistu_graph=napistu_graph,
            attribute_type=NAPISTU_GRAPH.EDGES,
            attribute=edge_attribute,
            encoding_manager=encoding_manager,
        )

        # Get edge names for alignment
        if (
            not hasattr(self, NAPISTU_DATA.NG_EDGE_NAMES)
            or getattr(self, NAPISTU_DATA.NG_EDGE_NAMES) is None
        ):
            raise ValueError(
                f"Validation not available - the `{NAPISTU_DATA.NG_EDGE_NAMES}` attribute is required for this method."
            )
        data_edge_names = getattr(self, NAPISTU_DATA.NG_EDGE_NAMES)

        # Align the graph values with the NapistuData edge ordering
        # Convert the Series with MultiIndex to DataFrame
        graph_df = graph_values.to_frame("graph_edge_value").reset_index()

        # Merge with NapistuData edge names to get aligned graph values
        # Convert data_edge_names DataFrame for merging
        aligned_graph = data_edge_names.merge(
            graph_df, on=[NAPISTU_GRAPH_EDGES.FROM, NAPISTU_GRAPH_EDGES.TO], how="left"
        )
        graph_values_aligned = aligned_graph["graph_edge_value"]

        # Debug: Check if we have any matches
        matches_found = aligned_graph["graph_edge_value"].notna().sum()
        if matches_found == 0:
            raise ValueError(
                f"No matching edge names found between NapistuData and NapistuGraph. "
                f"NapistuData edge names: {data_edge_names.head().to_dict()}... "
                f"NapistuGraph edge names: {graph_values.index.tolist()[:5]}..."
            )

        # Create masks for valid (non-null) values in both series
        graph_valid_mask = ~graph_values_aligned.isna()
        data_valid_mask = ~data_values.isna()

        # Check if the non-null masks are identical
        if not graph_valid_mask.equals(data_valid_mask):
            graph_null_count = (~graph_valid_mask).sum()
            data_null_count = (~data_valid_mask).sum()
            raise ValueError(
                "Non-null masks don't match between graph and data values. "
                f"Graph values non-null count: {graph_valid_mask.sum()}, "
                f"Data values non-null count: {data_valid_mask.sum()}, "
                f"Graph values null count: {graph_null_count}, "
                f"Data values null count: {data_null_count}"
            )

        # Compare only valid values (since masks are identical, we can use either)
        graph_valid = graph_values_aligned[graph_valid_mask]
        data_valid = data_values[graph_valid_mask]

        # Check for exact matches
        matches = graph_valid == data_valid

        if not matches.all():
            # Find mismatches for detailed error reporting
            mismatches = ~matches
            mismatch_indices = matches.index[mismatches]

            mismatch_details = []
            for idx in mismatch_indices:
                graph_val = graph_valid[idx]
                data_val = data_valid[idx]
                edge_info = data_edge_names.iloc[data_edge_names.index.get_loc(idx)]
                mismatch_details.append(
                    f"Edge '{edge_info['from']} -> {edge_info['to']}': graph='{graph_val}', data='{data_val}'"
                )

            raise ValueError(
                f"Encoding validation failed for {edge_attribute}. "
                f"Found {mismatches.sum()} mismatches out of {len(matches)} valid comparisons:\n"
                + "\n".join(mismatch_details)
            )

        return True

    def _validate_labels(
        self,
        napistu_graph: NapistuGraph,
        labeling_manager: LabelingManager,
    ) -> bool:
        """
        Validate consistency between encoded labels and original NapistuGraph vertex labels.

        This method compares the labels recovered from encoding
        in the NapistuData object with the original labels stored in
        the NapistuGraph object to ensure data consistency.

        Parameters
        ----------
        napistu_graph : NapistuGraph
            The NapistuGraph object containing the original vertex labels
        labeling_manager: LabelingManager
            The labeling manager used to decode the encoded labels

        Returns
        -------
        bool
            True if the label encoding is consistent, False otherwise

        Raises
        ------
        ValueError
            If the NapistuData object doesn't have encoded labels (y attribute),
            if vertex names don't match between NapistuData and NapistuGraph,
            or if there are label encoding inconsistencies.

        Examples
        --------
        >>> # Validate label encoding consistency
        >>> is_consistent = napistu_data._validate_labels(napistu_graph, labeling_manager)
        >>> print(f"Label encoding is consistent: {is_consistent}")
        True
        """

        # Check if NapistuData has encoded labels
        if not hasattr(self, NAPISTU_DATA.Y) or getattr(self, NAPISTU_DATA.Y) is None:
            raise ValueError(
                "NapistuData object does not have encoded labels (y attribute)"
            )

        # Get the encoded labels and vertex names from NapistuData
        encoded_labels = getattr(self, NAPISTU_DATA.Y)
        vertex_names = getattr(self, NAPISTU_DATA.NG_VERTEX_NAMES, None)

        if vertex_names is None:
            raise ValueError(
                f"Validation not available - the `{NAPISTU_DATA.NG_VERTEX_NAMES}` attribute is required for this method."
            )

        # Verify dimensions match
        if len(encoded_labels) != len(vertex_names):
            raise ValueError(
                f"Label count ({len(encoded_labels)}) should match vertex count ({len(vertex_names)})"
            )
        if len(encoded_labels) != self.num_nodes:
            raise ValueError(
                f"Label count ({len(encoded_labels)}) should match node count ({self.num_nodes})"
            )

        # Decode the labels using the utility function
        decoded_labels = pd.Series(decode_labels(encoded_labels, labeling_manager))

        # Get the corresponding labels from the NapistuGraph using merge
        vertex_df = napistu_graph.get_vertex_dataframe()
        vertex_names_df = pd.DataFrame({NAPISTU_GRAPH_VERTICES.NAME: vertex_names})

        # Merge vertex names with the vertex DataFrame to get labels
        merged_df = vertex_names_df.merge(
            vertex_df[[NAPISTU_GRAPH_VERTICES.NAME, labeling_manager.label_attribute]],
            on=NAPISTU_GRAPH_VERTICES.NAME,
            how="left",
        )
        graph_labels = merged_df[labeling_manager.label_attribute]

        # Create mask for valid (non-null) values in both decoded and graph labels
        decoded_valid_mask = decoded_labels.notna()
        graph_valid_mask = graph_labels.notna()
        valid_mask = decoded_valid_mask & graph_valid_mask

        # Compare only valid values
        decoded_valid = decoded_labels[valid_mask]
        graph_valid = graph_labels[valid_mask]

        if len(decoded_valid) != len(graph_valid):
            raise ValueError("Valid label counts should match")

        # Check for exact matches
        matches = decoded_valid == graph_valid

        if not matches.all():
            # Find mismatches for detailed error reporting
            mismatches = ~matches
            mismatch_indices = matches.index[mismatches]

            mismatch_details = []
            for idx in mismatch_indices:
                decoded_val = decoded_valid[idx]
                graph_val = graph_valid[idx]
                vertex_name = vertex_names.iloc[vertex_names.index.get_loc(idx)]
                mismatch_details.append(
                    f"Vertex '{vertex_name}': decoded='{decoded_val}', graph='{graph_val}'"
                )

            raise ValueError(
                f"Label encoding validation failed. "
                f"Found {mismatches.sum()} mismatches out of {len(matches)} valid comparisons:\n"
                + "\n".join(mismatch_details)
            )

        # Additional verification: check that we have some non-null labels
        non_null_decoded = decoded_labels[decoded_labels.notna()]
        non_null_graph = graph_labels[graph_labels.notna()]

        if len(non_null_decoded) == 0:
            raise ValueError("Should have some non-null decoded labels")
        if len(non_null_graph) == 0:
            raise ValueError("Should have some non-null graph labels")
        if len(non_null_decoded) != len(non_null_graph):
            raise ValueError("Non-null label counts should match")

        return True

    def __repr__(self) -> str:
        """String representation of the NapistuData object."""
        summary = self.summary()
        return (
            f"NapistuData(num_nodes={summary['num_nodes']}, "
            f"num_edges={summary['num_edges']}, "
            f"num_node_features={summary['num_node_features']}, "
            f"num_edge_features={summary['num_edge_features']})"
        )
