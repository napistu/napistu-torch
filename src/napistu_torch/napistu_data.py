"""
NapistuData - A PyTorch Geometric Data subclass for Napistu networks.

This class extends PyG's Data class with Napistu-specific functionality
including safe save/load methods and additional utilities.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch
from torch_geometric.data import Data

from napistu_torch.constants import NAPISTU_DATA


class NapistuData(Data):
    """
    A PyTorch Geometric Data subclass for Napistu biological networks.

    This class extends PyG's Data class with Napistu-specific functionality
    including safe save/load methods and additional utilities for working
    with biological network data.

    Parameters
    ----------
    x : torch.Tensor, optional
        Node feature matrix with shape [num_nodes, num_node_features]
    edge_index : torch.Tensor, optional
        Graph connectivity in COO format with shape [2, num_edges]
    edge_attr : torch.Tensor, optional
        Edge feature matrix with shape [num_edges, num_edge_features]
    edge_weight : torch.Tensor, optional
        Edge weights tensor with shape [num_edges]
    y : torch.Tensor, optional
        Node labels tensor with shape [num_nodes] for supervised learning tasks
    vertex_feature_names : List[str], optional
        Names of vertex features for interpretability
    edge_feature_names : List[str], optional
        Names of edge features for interpretability
    ng_vertex_names : pd.Series, optional
        Minimal vertex names from the original NapistuGraph. Series aligned with
        the vertex tensor (x) - each element corresponds to a vertex in the same
        order as the tensor rows. Used for debugging and validation of tensor alignment.
    ng_edge_names : pd.DataFrame, optional
        Minimal edge names from the original NapistuGraph. DataFrame with 'from' and 'to'
        columns aligned with the edge tensor (edge_index, edge_attr) - each row corresponds
        to an edge in the same order as the tensor columns. Used for debugging and validation.
    **kwargs
        Additional attributes to store in the data object

    Examples
    --------
    >>> # Create a NapistuData object
    >>> data = NapistuData(
    ...     x=torch.randn(100, 10),
    ...     edge_index=torch.randint(0, 100, (2, 200)),
    ...     edge_attr=torch.randn(200, 5),
    ...     y=torch.randint(0, 3, (100,)),  # Node labels
    ...     vertex_feature_names=['feature_1', 'feature_2', ...],
    ...     edge_feature_names=['weight', 'direction', ...],
    ...     ng_vertex_names=vertex_names_series,  # Optional: minimal vertex names
    ...     ng_edge_names=edge_names_df,          # Optional: minimal edge names
    ... )
    >>>
    >>> # Save and load
    >>> data.save('my_network.pt')
    >>> loaded_data = NapistuData.load('my_network.pt')
    """

    def __init__(
        self,
        x: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        vertex_feature_names: Optional[List[str]] = None,
        edge_feature_names: Optional[List[str]] = None,
        ng_vertex_names: Optional[pd.Series] = None,
        ng_edge_names: Optional[pd.DataFrame] = None,
        **kwargs,
    ):
        # Build parameters dict, only including non-None values
        params = {
            NAPISTU_DATA.X: x,
            NAPISTU_DATA.EDGE_INDEX: edge_index,
            NAPISTU_DATA.EDGE_ATTR: edge_attr,
            NAPISTU_DATA.EDGE_WEIGHT: edge_weight,
        }

        # Only add y if it's not None
        if y is not None:
            params[NAPISTU_DATA.Y] = y

        # Add any non-None kwargs
        params.update({k: v for k, v in kwargs.items() if v is not None})

        super().__init__(**params)

        # Store feature names for interpretability
        if vertex_feature_names is not None:
            self.vertex_feature_names = vertex_feature_names
        if edge_feature_names is not None:
            self.edge_feature_names = edge_feature_names

        # Store minimal NapistuGraph attributes for debugging and validation
        if ng_vertex_names is not None:
            self.ng_vertex_names = ng_vertex_names
        if ng_edge_names is not None:
            self.ng_edge_names = ng_edge_names

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

    @classmethod
    def load(
        cls, filepath: Union[str, Path], map_location: str = "cpu"
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
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            # Always use weights_only=False for PyG compatibility
            data = torch.load(filepath, weights_only=False, map_location=map_location)

            # Convert to NapistuData if it's a regular Data object
            if isinstance(data, Data) and not isinstance(data, NapistuData):
                napistu_data = NapistuData()
                napistu_data.__dict__.update(data.__dict__)
                return napistu_data
            elif isinstance(data, NapistuData):
                return data
            else:
                raise TypeError(
                    f"Loaded object is not a NapistuData or Data object, got {type(data)}. "
                    "This may indicate a corrupted file or incorrect file type."
                )

        except Exception as e:
            raise RuntimeError(
                f"Failed to load NapistuData object from {filepath}: {e}"
            ) from e

    def get_vertex_feature_names(self) -> Optional[List[str]]:
        """
        Get the names of vertex features.

        Returns
        -------
        Optional[List[str]]
            List of vertex feature names, or None if not available
        """
        return getattr(self, NAPISTU_DATA.VERTEX_FEATURE_NAMES, None)

    def get_edge_feature_names(self) -> Optional[List[str]]:
        """
        Get the names of edge features.

        Returns
        -------
        Optional[List[str]]
            List of edge feature names, or None if not available
        """
        return getattr(self, NAPISTU_DATA.EDGE_FEATURE_NAMES, None)

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
        return getattr(self, NAPISTU_DATA.EDGE_WEIGHT, None)

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the NapistuData object.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing summary information about the data object
        """
        summary_dict = {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_node_features": self.num_node_features,
            "num_edge_features": self.num_edge_features,
            "has_vertex_feature_names": hasattr(
                self, NAPISTU_DATA.VERTEX_FEATURE_NAMES
            ),
            "has_edge_feature_names": hasattr(self, NAPISTU_DATA.EDGE_FEATURE_NAMES),
            "has_edge_weights": hasattr(self, NAPISTU_DATA.EDGE_WEIGHT),
        }

        if hasattr(self, NAPISTU_DATA.VERTEX_FEATURE_NAMES):
            summary_dict[NAPISTU_DATA.VERTEX_FEATURE_NAMES] = self.vertex_feature_names
        if hasattr(self, NAPISTU_DATA.EDGE_FEATURE_NAMES):
            summary_dict[NAPISTU_DATA.EDGE_FEATURE_NAMES] = self.edge_feature_names

        # Add any additional attributes
        for key, value in self.__dict__.items():
            if key not in summary_dict and not key.startswith("_"):
                if isinstance(value, torch.Tensor):
                    summary_dict[key] = f"Tensor{list(value.shape)}"
                else:
                    summary_dict[key] = str(value)[:100]  # Truncate long strings

        return summary_dict

    def __repr__(self) -> str:
        """String representation of the NapistuData object."""
        summary = self.summary()
        return (
            f"NapistuData(num_nodes={summary['num_nodes']}, "
            f"num_edges={summary['num_edges']}, "
            f"num_node_features={summary['num_node_features']}, "
            f"num_edge_features={summary['num_edge_features']})"
        )
