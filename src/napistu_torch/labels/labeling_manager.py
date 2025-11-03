"""Labeling manager configuration and validation for napistu-torch."""

from typing import Any, Dict, List, Optional

from napistu.network.constants import (
    NAPISTU_GRAPH_VERTICES,
    VALID_VERTEX_SBML_DFS_SUMMARIES,
    VERTEX_SBML_DFS_SUMMARIES,
)
from pydantic import BaseModel, Field, field_validator

from napistu_torch.labels.constants import LABEL_TYPE


class LabelingManager(BaseModel):
    """Configuration for label-specific featurization strategies.

    This class organizes and validates the attributes needed for different
    labeling approaches in molecular network analysis tasks.

    Attributes
    ----------
    label_attribute : str
        The vertex attribute to use as the target label
    exclude_vertex_attributes : List[str]
        Vertex attributes to exclude from feature extraction
    augment_summary_types : List[str]
        SBML DFS summary types to add during graph augmentation.
        Used by augment_napistu_graph when calling add_sbml_dfs_summaries.
    label_names : Optional[Dict[int, Any]]
        Optional lookup table mapping label integers to their original names.
        Used to track the mapping between encoded integers and original label values.

    Public Methods
    --------------
    get_label_names()
        Get the label names mapping, returning empty dict if None
    to_dict()
        Convert the labeling strategy to a dictionary
    from_dict(config)
        Create a LabelingManager from a dictionary configuration
    """

    label_attribute: str
    exclude_vertex_attributes: List[str] = Field(default_factory=list)
    augment_summary_types: List[str] = Field(default_factory=list)
    label_names: Optional[Dict[int, Any]] = Field(default=None)

    @field_validator("label_attribute")
    @classmethod
    def validate_label_attribute(cls, v: str) -> str:
        """Validate that the label attribute is not empty."""
        if not v or not isinstance(v, str):
            raise ValueError("label_attribute must be a non-empty string")
        return v

    @field_validator("exclude_vertex_attributes")
    @classmethod
    def validate_exclude_attributes(cls, v: List[str]) -> List[str]:
        """Validate that excluded attributes are strings."""
        if not isinstance(v, list):
            raise ValueError("exclude_vertex_attributes must be a list")
        for attr in v:
            if not isinstance(attr, str) or not attr:
                raise ValueError(
                    "All exclude_vertex_attributes must be non-empty strings"
                )
        return v

    @field_validator("augment_summary_types")
    @classmethod
    def validate_summary_types(cls, v: List[str]) -> List[str]:
        """Validate that summary types are valid SBML DFS summary types."""
        if not isinstance(v, list):
            raise ValueError("augment_summary_types must be a list")

        valid_summary_types = list(VALID_VERTEX_SBML_DFS_SUMMARIES)

        for summary_type in v:
            if not isinstance(summary_type, str) or not summary_type:
                raise ValueError("All summary types must be non-empty strings")
            if summary_type not in valid_summary_types:
                raise ValueError(
                    f"Invalid summary_type '{summary_type}'. "
                    f"Must be one of: {valid_summary_types}"
                )
        return v

    def get_label_names(self) -> Dict[int, Any]:
        """Get the label names mapping, returning empty dict if None."""
        return self.label_names if self.label_names is not None else {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert the labeling strategy to a dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "LabelingManager":
        """Create a LabelingManager from a dictionary configuration."""
        return cls.model_validate(config)


# Predefined labeling strategies
LABELING_MANAGERS = {
    LABEL_TYPE.SPECIES_TYPE: LabelingManager(
        label_attribute=NAPISTU_GRAPH_VERTICES.SPECIES_TYPE,
        exclude_vertex_attributes=[NAPISTU_GRAPH_VERTICES.SPECIES_TYPE],
        augment_summary_types=[VERTEX_SBML_DFS_SUMMARIES.SOURCES],
    ),
    LABEL_TYPE.NODE_TYPE: LabelingManager(
        label_attribute=NAPISTU_GRAPH_VERTICES.NODE_TYPE,
        exclude_vertex_attributes=[
            NAPISTU_GRAPH_VERTICES.NODE_TYPE,
            NAPISTU_GRAPH_VERTICES.SPECIES_TYPE,
        ],
        augment_summary_types=[VERTEX_SBML_DFS_SUMMARIES.SOURCES],
    ),
}
