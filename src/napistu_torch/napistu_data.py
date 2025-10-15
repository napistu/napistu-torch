"""
NapistuData - A PyTorch Geometric Data subclass for Napistu networks.

This class extends PyG's Data class with Napistu-specific functionality
including safe save/load methods and additional utilities.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch_geometric.data import Data


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
    vertex_feature_names : List[str], optional
        Names of vertex features for interpretability
    edge_feature_names : List[str], optional
        Names of edge features for interpretability
    **kwargs
        Additional attributes to store in the data object

    Examples
    --------
    >>> # Create a NapistuData object
    >>> data = NapistuData(
    ...     x=torch.randn(100, 10),
    ...     edge_index=torch.randint(0, 100, (2, 200)),
    ...     edge_attr=torch.randn(200, 5),
    ...     vertex_feature_names=['feature_1', 'feature_2', ...],
    ...     edge_feature_names=['weight', 'direction', ...]
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
        vertex_feature_names: Optional[List[str]] = None,
        edge_feature_names: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, **kwargs)

        # Store feature names for interpretability
        if vertex_feature_names is not None:
            self.vertex_feature_names = vertex_feature_names
        if edge_feature_names is not None:
            self.edge_feature_names = edge_feature_names

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
        return getattr(self, "vertex_feature_names", None)

    def get_edge_feature_names(self) -> Optional[List[str]]:
        """
        Get the names of edge features.

        Returns
        -------
        Optional[List[str]]
            List of edge feature names, or None if not available
        """
        return getattr(self, "edge_feature_names", None)

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
            "has_vertex_feature_names": hasattr(self, "vertex_feature_names"),
            "has_edge_feature_names": hasattr(self, "edge_feature_names"),
        }

        if hasattr(self, "vertex_feature_names"):
            summary_dict["vertex_feature_names"] = self.vertex_feature_names
        if hasattr(self, "edge_feature_names"):
            summary_dict["edge_feature_names"] = self.edge_feature_names

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
