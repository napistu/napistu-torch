import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import torch

from napistu_torch.constants import VERTEX_TENSOR
from napistu_torch.ml.constants import DEVICE
from napistu_torch.napistu_data import NapistuData

logger = logging.getLogger(__name__)


class VertexTensor:
    """
    Container for vertex-aligned tensors with metadata.

    Keeps tensors aligned with NapistuGraph vertices, storing the necessary
    metadata to validate alignment and interpret features.

    Attributes
    ----------
    data : torch.Tensor
        The vertex-aligned tensor with shape [num_vertices, num_features]
    feature_names : List[str]
        Names of features (columns)
    vertex_names : pd.Series
        Vertex names aligned with tensor rows
    name : str
        Name/identifier for this tensor (e.g., "pathway_memberships")
    description : Optional[str]
        Human-readable description of what this tensor represents

    Public Methods
    --------------
    save(filepath)
        Save the VertexTensor to disk
    load(filepath, map_location="cpu")
        Load a VertexTensor from disk
    align_to_napistu_data(napistu_data, inplace=True)
        Align this VertexTensor to match the vertex ordering in NapistuData
    """

    def __init__(
        self,
        data: torch.Tensor,
        feature_names: List[str],
        vertex_names: pd.Series,
        name: str,
        description: Optional[str] = None,
    ):
        self.data = data
        self.feature_names = feature_names
        self.vertex_names = vertex_names
        self.name = name
        self.description = description

    def align_to_napistu_data(
        self, napistu_data: NapistuData, inplace: bool = True
    ) -> "VertexTensor":
        """
        Align this VertexTensor to match the vertex ordering in NapistuData.

        Validates that vertex names match and reorders the tensor data if
        necessary to align with the NapistuData vertex ordering.

        Parameters
        ----------
        napistu_data : NapistuData
            The NapistuData object to align to
        inplace : bool, default=True
            If True, modify this VertexTensor in place. If False, return
            a new VertexTensor with the alignment applied.

        Returns
        -------
        VertexTensor
            The aligned VertexTensor (self if inplace=True, new instance otherwise)

        Raises
        ------
        ValueError
            If the number of vertices don't match or if vertex names don't match
        """
        # Validate number of vertices
        if self.data.shape[0] != napistu_data.x.shape[0]:
            raise ValueError(
                f"Vertex tensor and NapistuData have different numbers of rows: "
                f"{self.data.shape[0]} != {napistu_data.x.shape[0]}"
            )

        nd_vertex_names = napistu_data.get_vertex_names()

        # Handle case where NapistuData has no vertex names
        if nd_vertex_names is None:
            logger.warning(
                "No vertex names found in NapistuData, so `vertex_tensor`'s "
                "alignment to NapistuData will be assumed but not confirmed"
            )
            return self if inplace else self.copy()

        # Case 1: Vertex names are identical and in the same order
        if (nd_vertex_names.values == self.vertex_names.values).all():
            logger.debug("Vertex names already aligned with NapistuData")
            return self if inplace else self.copy()

        # Case 2: Same vertex names but different order
        if set(nd_vertex_names.values) == set(self.vertex_names.values):
            # Create a mapping from vertex names to indices in this VertexTensor
            vt_name_to_idx = {
                name: idx for idx, name in enumerate(self.vertex_names.values)
            }

            # Create reordering indices based on nd_vertex_names order
            reorder_indices = [vt_name_to_idx[name] for name in nd_vertex_names.values]

            # Reorder data
            reordered_data = self.data[reorder_indices]

            logger.info(
                f"Reordered VertexTensor '{self.name}' to match NapistuData vertex ordering"
            )

            if inplace:
                self.data = reordered_data
                self.vertex_names = nd_vertex_names.copy()
                return self
            else:
                return VertexTensor(
                    data=reordered_data,
                    feature_names=self.feature_names.copy(),
                    vertex_names=nd_vertex_names.copy(),
                    name=self.name,
                    description=self.description,
                )

        # Case 3: Vertex names don't match
        raise ValueError(
            f"Vertex names in VertexTensor '{self.name}' and NapistuData do not match"
        )

    def copy(self) -> "VertexTensor":
        """Create a deep copy of this VertexTensor."""
        return VertexTensor(
            data=self.data.clone(),
            feature_names=self.feature_names.copy(),
            vertex_names=self.vertex_names.copy(),
            name=self.name,
            description=self.description,
        )

    @classmethod
    def load(
        cls, filepath: Union[str, Path], map_location: str = DEVICE.CPU
    ) -> "VertexTensor":
        """Load from disk."""
        saved = torch.load(filepath, weights_only=False, map_location=map_location)
        return cls(**saved)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save to disk."""
        torch.save(
            {
                VERTEX_TENSOR.DATA: self.data,
                VERTEX_TENSOR.FEATURE_NAMES: self.feature_names,
                VERTEX_TENSOR.VERTEX_NAMES: self.vertex_names,
                VERTEX_TENSOR.NAME: self.name,
                VERTEX_TENSOR.DESCRIPTION: self.description,
            },
            filepath,
        )
