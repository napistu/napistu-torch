"""
Virtual cell foundation models: embeddings, weights, attention, and ETL.
"""

from napistu_torch.foundation_models.attention_patterns import (
    AttentionPatternsInputs,
    LayerwiseAttentionInputs,
    aggregate_embedding_comparisons_over_categories,
    validate_embedding_comparisons_settings,
)
from napistu_torch.foundation_models.constants import (
    COMPARE_EMBEDDINGS_COMPARISONS,
    COMPARE_EMBEDDINGS_SETTINGS,
    FM_DEFS,
    FM_EDGELIST,
    FM_LAYER_CONSENSUS_METHODS,
    GROUP_SCOPING_FIELDS,
    MODEL_NICE_NAMES,
    SCOPING_FIELDS,
    SCPRINT_DEFS,
    SCPRINT_VERSIONS,
)
from napistu_torch.foundation_models.etl import (
    populate_lamin_db,
    process_aidocell,
    process_scfoundation,
    process_scgpt,
    process_scprint,
)
from napistu_torch.foundation_models.foundation_models import (
    AttentionLayer,
    FoundationModel,
    FoundationModels,
    FoundationModelStore,
    FoundationModelWeights,
    GeneAnnotations,
    ModelMetadata,
)
from napistu_torch.foundation_models.gene_embeddings import (
    DatasetGeneEmbeddings,
    GeneEmbeddings,
    GeneEmbeddingsSet,
)

__all__ = [
    "AttentionLayer",
    "AttentionPatternsInputs",
    "COMPARE_EMBEDDINGS_COMPARISONS",
    "COMPARE_EMBEDDINGS_SETTINGS",
    "DatasetGeneEmbeddings",
    "FM_DEFS",
    "FM_EDGELIST",
    "FM_LAYER_CONSENSUS_METHODS",
    "FoundationModel",
    "FoundationModelStore",
    "FoundationModels",
    "FoundationModelWeights",
    "GeneAnnotations",
    "GeneEmbeddings",
    "GeneEmbeddingsSet",
    "GROUP_SCOPING_FIELDS",
    "LayerwiseAttentionInputs",
    "MODEL_NICE_NAMES",
    "ModelMetadata",
    "SCOPING_FIELDS",
    "SCPRINT_DEFS",
    "SCPRINT_VERSIONS",
    "aggregate_embedding_comparisons_over_categories",
    "populate_lamin_db",
    "process_aidocell",
    "process_scfoundation",
    "process_scgpt",
    "process_scprint",
    "validate_embedding_comparisons_settings",
]
