"""Constants for virtual cell foundation models and embedding comparisons."""

from types import SimpleNamespace

FOUNDATION_MODEL_NAMES = SimpleNamespace(
    AIDOCELL="AIDOCell",
    SCFOUNDATION="scFoundation",
    SCGPT="scGPT",
    SCPRINT="scPRINT",
)

VALID_FOUNDATION_MODEL_NAMES = list(FOUNDATION_MODEL_NAMES.__dict__.values())

AIDOCELL_CLASSES = SimpleNamespace(
    THREE_M="aido_cell_3m",
    TEN_M="aido_cell_10m",
    ONE_HUNDRED_M="aido_cell_100m",
)
AIDOCELL_CLASSES_LIST = list(AIDOCELL_CLASSES.__dict__.values())

SCPRINT_VERSIONS = SimpleNamespace(
    SMALL="small",
    MEDIUM="medium",
    LARGE="large",
)
SCPRINT_VERSIONS_LIST = list(SCPRINT_VERSIONS.__dict__.values())

MODEL_NICE_NAMES = {
    (FOUNDATION_MODEL_NAMES.AIDOCELL, AIDOCELL_CLASSES.THREE_M): "AIDO.Cell (3M)",
    (FOUNDATION_MODEL_NAMES.AIDOCELL, AIDOCELL_CLASSES.TEN_M): "AIDO.Cell (10M)",
    (
        FOUNDATION_MODEL_NAMES.AIDOCELL,
        AIDOCELL_CLASSES.ONE_HUNDRED_M,
    ): "AIDO.Cell (100M)",
    (FOUNDATION_MODEL_NAMES.SCGPT, None): "scGPT",
    (FOUNDATION_MODEL_NAMES.SCFOUNDATION, None): "scFoundation",
    (FOUNDATION_MODEL_NAMES.SCPRINT, SCPRINT_VERSIONS.SMALL): "scPRINT (small)",
    (FOUNDATION_MODEL_NAMES.SCPRINT, SCPRINT_VERSIONS.MEDIUM): "scPRINT (medium)",
    (FOUNDATION_MODEL_NAMES.SCPRINT, SCPRINT_VERSIONS.LARGE): "scPRINT (large)",
}

FM_CLASSES = SimpleNamespace(
    FOUNDATION_MODEL="FoundationModel",
    FOUNDATION_MODEL_WEIGHTS="FoundationModelWeights",
    ATTENTION_LAYER="AttentionLayer",
    GENE_EMBEDDINGS="GeneEmbeddings",
    GENE_EMBEDDINGS_SET="GeneEmbeddingsSet",
    DATASET_GENE_EMBEDDINGS="DatasetGeneEmbeddings",
)

FM_DEFS = SimpleNamespace(
    # class-specific fields
    MODELS="models",
    ATTENTION_LAYERS="attention_layers",
    # model summaries
    WEIGHTS_DICT="weights_dict",
    STATIC_GENE_EMBEDDINGS="static_gene_embeddings",
    ATTENTION_WEIGHTS="attention_weights",
    LAYER_IDX="layer_idx",
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
    WEIGHTS_FILENAME="weights.npz",
    METADATA_FILENAME="metadata.json",
    RESIDUALS_INDEX_FILENAME="residuals_index.yaml",
    RESIDUALS_SUBDIR="residuals",
    # gene embeddings
    EMBEDDINGS="embeddings",
    ORDERED_GENES="ordered_genes",
    CATEGORY_DICT="category_dict",
    DATASET_NAME="dataset_name",
    DATASET_URI="dataset_uri",
    CATEGORY="category",
    DATASET_GENE_EMBEDDINGS="dataset_gene_embeddings",  # optional DatasetGeneEmbeddings
)

CELLXGENE_DEFS = SimpleNamespace(
    CELL_TYPE="cell_type",
    LEIDEN_SCVI="leiden_scVI",
)

FM_DEFAULTS = SimpleNamespace(
    MIN_CLUSTER_CELLS=10,
    CELLS_PER_CLUSTER=100,
    # Residual-stream capture defaults differ by model family:
    # Memory-heavy encoders (AIDOCell, scFoundation, similar): one cell per forward pass.
    RESIDUAL_STREAM_BATCH_SIZE_MEM_SAFE=1,
    # scGPT residual capture default.
    RESIDUAL_STREAM_BATCH_SIZE_SCGPT=64,
)

EMBEDDING_METADATA_FIELDS = SimpleNamespace(
    MODEL_NAME=FM_DEFS.MODEL_NAME,
    MODEL_VARIANT=FM_DEFS.MODEL_VARIANT,
    LAYER_IDX=FM_DEFS.LAYER_IDX,
    DATASET_NAME=FM_DEFS.DATASET_NAME,
    CATEGORY=FM_DEFS.CATEGORY,
    MODEL_LABEL="model_label",
    SOURCE_LABEL="source_label",
    SCOPED_KEY="scoped_key",
)

# Ordered list of fields used for scoping. model_name and model_variant
# are combined into model_label and treated as a single unit.
SCOPING_FIELDS = [
    EMBEDDING_METADATA_FIELDS.MODEL_LABEL,
    EMBEDDING_METADATA_FIELDS.LAYER_IDX,
    EMBEDDING_METADATA_FIELDS.DATASET_NAME,
    EMBEDDING_METADATA_FIELDS.CATEGORY,
]

FM_EDGELIST = SimpleNamespace(
    FROM_GENE="from_gene",
    TO_GENE="to_gene",
    FROM_IDX="from_idx",
    TO_IDX="to_idx",
    LAYER="layer",
    ATTENTION="attention",
    ATTENTION_RANK="attention_rank",
    MODEL="model",
)

FM_LAYER_CONSENSUS_METHODS = SimpleNamespace(
    ABSOLUTE_ARGMAX="absolute-argmax",
    MAX="max",
    SUM="sum",
)

VALID_FM_LAYER_CONSENSUS_METHODS = list(FM_LAYER_CONSENSUS_METHODS.__dict__.values())

COMPARE_EMBEDDINGS_COMPARISONS = SimpleNamespace(
    RESIDUAL_STREAM_CORRELATIONS="residual_stream_correlations",
    MODEL_LAYER_CORRELATIONS="model_layer_correlations",
    MODEL_LAYER_RANK_AGREEMENT="model_layer_rank_agreement",
    CROSS_MODEL_X_LAYER_TOP_ATTENTIONS="cross_model_x_layer_top_attentions",
    CROSS_MODEL_X_LAYER_RANK_AGREEMENT="cross_model_x_layer_rank_agreement",
    CROSS_MODEL_CONSENSUS_TOP_ATTENTIONS="cross_model_consensus_top_attentions",
    CROSS_MODEL_CONSENSUS_TOP_ATTENTIONS_RANK_AGREEMENT="cross_model_consensus_top_attentions_rank_agreement",
    SETTINGS="settings",
)

VALID_COMPARE_EMBEDDINGS_COMPARISONS = list(
    COMPARE_EMBEDDINGS_COMPARISONS.__dict__.values()
)

COMPARE_EMBEDDINGS_SETTINGS = SimpleNamespace(
    TOP_K="top_k",
    IGNORE_SELF_ATTENTION="ignore_self_attention",
    BY_ABSOLUTE_VALUE="by_absolute_value",
    CONSENSUS_METHOD="consensus_method",
    EMBEDDING_KEYS="embedding_keys",
    N_GENES="n_genes",
)

# scFoundation constants
SCFOUNDATION_DEFS = SimpleNamespace(
    MODEL_NAME=FOUNDATION_MODEL_NAMES.SCFOUNDATION,
    REPO_ID="genbio-ai/scFoundation",
    CHECKPOINT_FILE="models.ckpt",
    GENE_LIST_URL="https://raw.githubusercontent.com/biomap-research/scFoundation/main/OS_scRNA_gene_index.19264.tsv",
    # Expected values (will be extracted from checkpoint and validated)
    N_GENES=19264,
    EMBED_DIM=768,
    N_ENCODER_LAYERS=12,
    N_HEADS=12,
    GENE_ENCODER="gene",
)

# scPRINT constants
SCPRINT_CHECKPOINTS = {
    SCPRINT_VERSIONS.SMALL: "small-v1.ckpt",
    SCPRINT_VERSIONS.MEDIUM: "medium-v1.5.ckpt",
    SCPRINT_VERSIONS.LARGE: "large-v1.ckpt",
}

SCPRINT_DEFS = SimpleNamespace(
    MODEL_NAME=FOUNDATION_MODEL_NAMES.SCPRINT,
    VERSIONS=SCPRINT_VERSIONS,
    CHECKPOINTS=SCPRINT_CHECKPOINTS,
    REPO_ID="jkobject/scPRINT",
    # Expected values (will be extracted from model and validated)
    N_HEADS=4,  # Fixed architecture parameter
)

# AIDOCell constants
AIDOCELL_DEFS = SimpleNamespace(
    MODEL_NAME=FOUNDATION_MODEL_NAMES.AIDOCELL,
    CLASSES=AIDOCELL_CLASSES,
    # files
    GENE_FILE="gene_lists/OS_scRNA_gene_index.19264.tsv",
    PREFIX_TEMPLATE="{model_name}_{model_class_name}",
    # parameters
    EMBED_DIM="embed_dim",
    N_LAYERS="n_layers",
    N_HEADS="n_heads",
    HIDDEN_DIM="hidden_dim",
)

# scGPT constants
SCGPT_DEFS = SimpleNamespace(
    MODEL_NAME=FOUNDATION_MODEL_NAMES.SCGPT,
    # urls
    GENE_IDENTIFIERS_URL="https://github.com/bowang-lab/scGPT/files/13243634/gene_info.csv",
    # files
    CONFIG_FILENAME="args.json",
    MODEL_FILENAME="best_model.pt",
    VOCAB_FILENAME="vocab.json",
    # parameters
    EMBSIZE="embsize",
    NHEAD="nheads",
    D_HID="d_hid",
    NLAYERS="nlayers",
    N_LAYERS_CLS="n_layers_cls",
    # constants
    PAD_TOKEN="<pad>",
    SPECIAL_TOKENS=["<pad>", "<cls>", "<eoc>"],
    N_HVG=1200,
    N_BINS=51,
    MASK_VALUE=-1,
    PAD_VALUE=-2,
    N_INPUT_BINS=51,
)
