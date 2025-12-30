from types import SimpleNamespace

from napistu.constants import SBML_DFS
from napistu.network.constants import (
    IGRAPH_DEFS,
    NAPISTU_GRAPH_EDGES,
    NAPISTU_GRAPH_VERTICES,
)

# artifact defs

DEFAULT_ARTIFACTS_NAMES = SimpleNamespace(
    UNLABELED="unlabeled",
    EDGE_PREDICTION="edge_prediction",
    RELATION_PREDICTION="relation_prediction",
    SPECIES_TYPE_PREDICTION="species_type_prediction",
    COMPREHENSIVE_PATHWAY_MEMBERSHIPS="comprehensive_pathway_memberships",
    EDGE_STRATA_BY_EDGE_SBO_TERMS="edge_strata_by_edge_sbo_terms",
    EDGE_STRATA_BY_NODE_SPECIES_TYPE="edge_strata_by_node_species_type",
    EDGE_STRATA_BY_NODE_TYPE="edge_strata_by_node_type",
    NAME_TO_SID_MAP="name_to_sid_map",
    SPECIES_IDENTIFIERS="species_identifiers",
)

ARTIFACT_DEFS = SimpleNamespace(
    NAME="name",
    ARTIFACT_TYPE="artifact_type",
    CREATION_FUNC="creation_func",
    DESCRIPTION="description",
)

STRATIFY_BY_ARTIFACT_NAMES = {
    DEFAULT_ARTIFACTS_NAMES.EDGE_STRATA_BY_EDGE_SBO_TERMS,
    DEFAULT_ARTIFACTS_NAMES.EDGE_STRATA_BY_NODE_SPECIES_TYPE,
    DEFAULT_ARTIFACTS_NAMES.EDGE_STRATA_BY_NODE_TYPE,
}

MERGE_RARE_STRATA_DEFS = SimpleNamespace(
    OTHER="other",
    OTHER_RELATION="other relation",
)

# transformation defs

ENCODING_MANAGER = SimpleNamespace(
    COLUMNS="columns",
    TRANSFORMER="transformer",
    # attributes
    FIT="fit",
    TRANSFORM="transform",
    PASSTHROUGH="passthrough",
    # merges
    BASE="base",
    OVERRIDE="override",
)

ENCODING_MANAGER_TABLE = SimpleNamespace(
    TRANSFORM_NAME="transform_name",
    COLUMN="column",
    TRANSFORMER_TYPE="transformer_type",
)

# encodings

ENCODINGS = SimpleNamespace(
    CATEGORICAL="categorical",
    NUMERIC="numeric",
    SPARSE_CATEGORICAL="sparse_categorical",
    SPARSE_NUMERIC="sparse_numeric",
    BINARY="binary",
)

NEVER_ENCODE = {
    SBML_DFS.SC_ID,
    SBML_DFS.S_ID,
    SBML_DFS.C_ID,
    SBML_DFS.R_ID,
    IGRAPH_DEFS.INDEX,
    IGRAPH_DEFS.NAME,
    IGRAPH_DEFS.SOURCE,
    IGRAPH_DEFS.TARGET,
    NAPISTU_GRAPH_VERTICES.NODE_NAME,
    NAPISTU_GRAPH_EDGES.FROM,
    NAPISTU_GRAPH_EDGES.TO,
}

# Node configuration
VERTEX_DEFAULT_TRANSFORMS = {
    ENCODINGS.CATEGORICAL: {
        NAPISTU_GRAPH_VERTICES.NODE_TYPE,
    },
    ENCODINGS.SPARSE_CATEGORICAL: {
        NAPISTU_GRAPH_VERTICES.SPECIES_TYPE,
    },
}

# Edge configuration
EDGE_DEFAULT_TRANSFORMS = {
    ENCODINGS.CATEGORICAL: {
        NAPISTU_GRAPH_EDGES.DIRECTION,
        NAPISTU_GRAPH_EDGES.SBO_TERM_DOWNSTREAM,
        NAPISTU_GRAPH_EDGES.SBO_TERM_UPSTREAM,
    },
    ENCODINGS.NUMERIC: {
        NAPISTU_GRAPH_EDGES.STOICHIOMETRY_DOWNSTREAM,
        NAPISTU_GRAPH_EDGES.STOICHIOMETRY_UPSTREAM,
        NAPISTU_GRAPH_EDGES.WEIGHT,
        NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM,
    },
    ENCODINGS.BINARY: {
        NAPISTU_GRAPH_EDGES.R_ISREVERSIBLE,
    },
}

# splitting strategies

SPLITTING_STRATEGIES = SimpleNamespace(
    EDGE_MASK="edge_mask",
    VERTEX_MASK="vertex_mask",
    NO_MASK="no_mask",
    INDUCTIVE="inductive",
)

VALID_SPLITTING_STRATEGIES = list(SPLITTING_STRATEGIES.__dict__.values())

# stratification

STRATIFY_BY = SimpleNamespace(
    EDGE_SBO_TERMS="edge_sbo_terms",
    NODE_SPECIES_TYPE="node_species_type",
    NODE_TYPE="node_type",
)

VALID_STRATIFY_BY = list(STRATIFY_BY.__dict__.values())

STRATIFY_BY_TO_ARTIFACT_NAMES = {
    STRATIFY_BY.EDGE_SBO_TERMS: DEFAULT_ARTIFACTS_NAMES.EDGE_STRATA_BY_EDGE_SBO_TERMS,
    STRATIFY_BY.NODE_SPECIES_TYPE: DEFAULT_ARTIFACTS_NAMES.EDGE_STRATA_BY_NODE_SPECIES_TYPE,
    STRATIFY_BY.NODE_TYPE: DEFAULT_ARTIFACTS_NAMES.EDGE_STRATA_BY_NODE_TYPE,
}

STRATIFICATION_DEFS = SimpleNamespace(
    EDGE_STRATA="edge_strata",
    FROM_TO_SEPARATOR=" -> ",
)

# toss these attributes during augmentation

IGNORED_EDGE_ATTRIBUTES = [
    "string_wt",  # defined in graph_attrs_spec.yaml, same pattern of missingness as other STRING vars. Should be uppercase to be consistent with them so a readable prefix is generated during deduplication.
    "IntAct_interaction_method_unknown",
    "OmniPath_is_directed",
    "OmniPath_is_inhibition",
    "OmniPath_is_stimulation",
    "sbo_term_downstream_SBO:0000336",  # interactors will always be identical between upstream and downstream vertex
]

IGNORED_VERTEX_ATTRIBUTES = [
    "ontology_reactome",  # identical to the Reactome source assignments
    "ontology_intact",  # identical to the IntAct source assignments
    "ontology_kegg.drug",  # currently these are the only species types for drug
    "ontology_smiles",  # currently the same as OmniPath small molecule
    "ontology_other",  # currently the same as the unknown species type
]

IGNORED_IF_CONSTANT_EDGE_ATTRIBUTES = {
    "STRING_database_transferred": 0,
    "STRING_neighborhood": 0,
}

IGNORED_IF_CONSTANT_VERTEX_ATTRIBUTES = {}

# checkpoints

CHECKPOINT_STRUCTURE = SimpleNamespace(
    STATE_DICT="state_dict",
    HYPER_PARAMETERS="hyper_parameters",
    EPOCH="epoch",
    GLOBAL_STEP="global_step",
    PYTORCH_LIGHTNING_VERSION="pytorch_lightning_version",
)

CHECKPOINT_HYPERPARAMETERS = SimpleNamespace(
    CONFIG="config",
    MODEL="model",
    DATA="data",
    ENVIRONMENT="environment",
)

# foundation models

FOUNDATION_MODEL_NAMES = SimpleNamespace(
    AIDOCELL="AIDOCell",
    SCPRINT="scPRINT",
    SCGPT="scGPT",
)

AIDOCELL_CLASSES = SimpleNamespace(
    THREE_M="aido_cell_3m",
    TEN_M="aido_cell_10m",
    ONE_HUNDRED_M="aido_cell_100m",
)
AIDOCELL_CLASSES_LIST = list(AIDOCELL_CLASSES.__dict__.values())

ALL_MODEL_FULL_NAMES = {
    FOUNDATION_MODEL_NAMES.SCGPT,
    FOUNDATION_MODEL_NAMES.SCPRINT,
} | {FOUNDATION_MODEL_NAMES.AIDOCELL + "_" + x for x in AIDOCELL_CLASSES_LIST}

FM_CLASSES = SimpleNamespace(
    FOUNDATION_MODEL="FoundationModel",
    FOUNDATION_MODEL_WEIGHTS="FoundationModelWeights",
    ATTENTION_LAYER="AttentionLayer",
)

FM_DEFS = SimpleNamespace(
    # class-specific fields
    MODELS="models",
    ATTENTION_LAYERS="attention_layers",
    # model summaries
    WEIGHTS_DICT="weights_dict",
    GENE_EMBEDDING="gene_embedding",
    ATTENTION_WEIGHTS="attention_weights",
    LAYER_NAME_TEMPLATE="layer_{layer_idx}",
    W_Q="W_q",
    W_K="W_k",
    W_V="W_v",
    W_O="W_o",
    # gene metadata
    GENE_ANNOTATIONS="gene_annotations",
    VOCAB_NAME="vocab_name",
    # model metadata
    MODEL_METADATA="model_metadata",
    MODEL_NAME="model_name",
    MODEL_VARIANT="model_variant",
    N_GENES="n_genes",
    N_VOCAB="n_vocab",
    ORDERED_VOCABULARY="ordered_vocabulary",
    EMBED_DIM="embed_dim",
    N_LAYERS="n_layers",
    N_HEADS="n_heads",
    # filename/variable name templates
    WEIGHTS_TEMPLATE="{prefix}_weights.npz",
    METADATA_TEMPLATE="{prefix}_metadata.json",
)
