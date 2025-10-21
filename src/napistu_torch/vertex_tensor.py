from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import torch

from napistu_torch.constants import VERTEX_TENSOR
from napistu_torch.ml.constants import DEVICE


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
    tensor_name : str
        Name/identifier for this tensor (e.g., "pathway_memberships")
    description : Optional[str]
        Human-readable description of what this tensor represents
    """

    def __init__(
        self,
        data: torch.Tensor,
        feature_names: List[str],
        vertex_names: pd.Series,
        tensor_name: str,
        description: Optional[str] = None,
    ):
        self.data = data
        self.feature_names = feature_names
        self.vertex_names = vertex_names
        self.tensor_name = tensor_name
        self.description = description

    def save(self, filepath: Union[str, Path]) -> None:
        """Save to disk."""
        torch.save(
            {
                VERTEX_TENSOR.DATA: self.data,
                VERTEX_TENSOR.FEATURE_NAMES: self.feature_names,
                VERTEX_TENSOR.VERTEX_NAMES: self.vertex_names,
                VERTEX_TENSOR.TENSOR_NAME: self.tensor_name,
                VERTEX_TENSOR.DESCRIPTION: self.description,
            },
            filepath,
        )

    @classmethod
    def load(
        cls, filepath: Union[str, Path], map_location: str = DEVICE.CPU
    ) -> "VertexTensor":
        """Load from disk."""
        saved = torch.load(filepath, weights_only=False, map_location=map_location)
        return cls(**saved)
