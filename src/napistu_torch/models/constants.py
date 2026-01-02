from types import SimpleNamespace

MODEL_DEFS = SimpleNamespace(
    ENCODER="encoder",
    ENCODER_TYPE="encoder_type",
    EDGE_ENCODER="edge_encoder",
    GCN="gcn",
    HEAD="head",
    HEAD_TYPE="head_type",
    # structural properties defined based on data
    IN_CHANNELS="in_channels",
    EDGE_IN_CHANNELS="edge_in_channels",
    NUM_RELATIONS="num_relations",
    NUM_CLASSES="num_classes",
    SYMMETRIC_RELATION_INDICES="symmetric_relation_indices",
    # generally applicable parameters
    DROPOUT="dropout",
    HIDDEN_CHANNELS="hidden_channels",
    NUM_LAYERS="num_layers",
)

ENCODERS = SimpleNamespace(
    GAT="gat",
    GCN="gcn",
    GRAPH_CONV="graph_conv",
    SAGE="sage",
)

VALID_ENCODERS = list(ENCODERS.__dict__.values())

ENCODER_SPECIFIC_ARGS = SimpleNamespace(
    GAT_HEADS="gat_heads",
    GAT_CONCAT="gat_concat",
    GRAPH_CONV_AGGREGATOR="graph_conv_aggregator",
    SAGE_AGGREGATOR="sage_aggregator",
)

VALID_ENCODER_NAMED_ARGS = list(ENCODER_SPECIFIC_ARGS.__dict__.values())

# defaults and other miscellaneous encoder definitions
ENCODER_DEFS = SimpleNamespace(
    STATIC_EDGE_WEIGHTS="static_edge_weights",
    GRAPH_CONV_DEFAULT_AGGREGATOR="mean",
    SAGE_DEFAULT_AGGREGATOR="mean",
    # derived encoder attributes
    EDGE_WEIGHTING_TYPE="edge_weighting_type",
    EDGE_WEIGHTING_VALUE="edge_weighting_value",
    # defaults
    DEFAULT_NUM_LAYERS=3,
    DEFAULT_DROPOUT=0.1,
)

# select the relevant arguments and convert from the {encoder}_{arg} convention back to just arg
ENCODER_NATIVE_ARGNAMES_MAPS = {
    ENCODERS.GAT: {
        ENCODER_SPECIFIC_ARGS.GAT_HEADS: "heads",
        MODEL_DEFS.DROPOUT: "dropout",
        ENCODER_SPECIFIC_ARGS.GAT_CONCAT: "concat",
    },
    ENCODERS.GRAPH_CONV: {
        ENCODER_SPECIFIC_ARGS.GRAPH_CONV_AGGREGATOR: "aggr",
    },
    ENCODERS.SAGE: {ENCODER_SPECIFIC_ARGS.SAGE_AGGREGATOR: "aggr"},
}

HEADS = SimpleNamespace(
    # standard node-level heads
    NODE_CLASSIFICATION="node_classification",
    # standard symmetric edge prediction heads
    DOT_PRODUCT="dot_product",
    # expressive edge prediction heads
    ATTENTION="attention",
    MLP="mlp",
    # relation prediction heads
    ROTATE="rotate",
    CONDITIONAL_ROTATE="conditional_rotate",
    TRANSE="transe",
    DISTMULT="distmult",
    RELATION_ATTENTION="relation_attention",
    RELATION_GATED_MLP="relation_gated_mlp",
    RELATION_ATTENTION_MLP="relation_attention_mlp",
)

VALID_HEADS = list(HEADS.__dict__.values())

RELATION_AWARE_HEADS = {
    HEADS.ROTATE,
    HEADS.CONDITIONAL_ROTATE,
    HEADS.TRANSE,
    HEADS.DISTMULT,
    HEADS.RELATION_ATTENTION,
    HEADS.RELATION_GATED_MLP,
    HEADS.RELATION_ATTENTION_MLP,
}
EDGE_PREDICTION_HEADS = {
    HEADS.DOT_PRODUCT,
    HEADS.MLP,
    HEADS.ATTENTION,
} | RELATION_AWARE_HEADS

HEADS_W_SPECIAL_SYMMETRY_HANDLING = {
    HEADS.CONDITIONAL_ROTATE,
}

# Head-specific parameter names
HEAD_SPECIFIC_ARGS = SimpleNamespace(
    INIT_HEAD_AS_IDENTITY="init_head_as_identity",
    MLP_HIDDEN_DIM="mlp_hidden_dim",
    MLP_NUM_LAYERS="mlp_num_layers",
    MLP_DROPOUT="mlp_dropout",
    NC_DROPOUT="nc_dropout",
    ROTATE_MARGIN="rotate_margin",
    TRANSE_MARGIN="transe_margin",
    RELATION_EMB_DIM="relation_emb_dim",
    RELATION_ATTENTION_HEADS="relation_attention_heads",
)

HEAD_DEFS = SimpleNamespace(
    DEFAULT_MLP_HIDDEN_DIM=128,
    DEFAULT_MLP_NUM_LAYERS=2,
    DEFAULT_MLP_DROPOUT=0.1,
    DEFAULT_NC_DROPOUT=0.1,
    DEFAULT_ROTATE_MARGIN=0.1,
    DEFAULT_TRANSE_MARGIN=0.1,
    DEFAULT_RELATION_EMB_DIM=64,
    DEFAULT_RELATION_ATTENTION_HEADS=4,
    DEFAULT_INIT_HEAD_AS_IDENTITY=False,
    # attributes
    LOSS_TYPE="loss_type",
    MARGIN="margin",
    SCORE_TO_PROBS="score_to_probs",
)

HEAD_DESCRIPTIONS = {
    HEADS.DOT_PRODUCT: {
        "label": "Dot product",
        "category": "edge_prediction",
        "description": "Simple dot product of source and target embeddings",
    },
    HEADS.MLP: {
        "label": "MLP",
        "category": "edge_prediction",
        "description": "Multi-layer perceptron head that concatenates source and target embeddings, then applies 2-layer MLP with ReLU and dropout.",
    },
    HEADS.ATTENTION: {
        "label": "Attention",
        "category": "edge_prediction",
        "description": "Attention head that projects nodes to query/key spaces and computes scaled dot-product attention. Learns separate transformations for source (query) and target (key) embeddings.",
    },
    HEADS.ROTATE: {
        "label": "RotatE",
        "category": "knowledge_graph_embedding",
        "description": "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space",
    },
    HEADS.TRANSE: {
        "label": "TransE",
        "category": "knowledge_graph_embedding",
        "description": "TransE: Translating Embeddings for Modeling Multi-relational Data",
    },
    HEADS.DISTMULT: {
        "label": "DistMult",
        "category": "knowledge_graph_embedding",
        "description": "DistMult: Embedding Entities and Relations for Learning and Inference in Knowledge Bases",
    },
    HEADS.RELATION_ATTENTION: {
        "label": "Relation-aware attention head",
        "category": "relation_prediction",
        "description": "Relation-aware multi-head attention head. Uses relation embeddings as queries to attend to edge features (concatenated source/target).",
    },
    HEADS.RELATION_ATTENTION_MLP: {
        "label": "Relation-aware attention head with MLP",
        "category": "relation_prediction",
        "description": "Processes edge features through MLP, then uses relation embeddings as queries in multi-head attention to select relevant features. Includes residual connection and output MLP for final prediction.",
    },
    HEADS.RELATION_GATED_MLP: {
        "label": "Relation-gated MLP head",
        "category": "relation_prediction",
        "description": "Edge features are processed through MLP, then modulated by relation-specific gates (element-wise multiplication), and passed through output MLP.",
    },
    HEADS.NODE_CLASSIFICATION: {
        "label": "Node classification head",
        "category": "node_classification",
        "description": "Node classification head that projects nodes to a hidden space and applies a single-layer MLP with ReLU and dropout.",
    },
}

EDGE_ENCODER_ARGS = SimpleNamespace(
    HIDDEN_DIM="hidden_dim",
    DROPOUT="dropout",
    INIT_BIAS="init_bias",
    # names used in the ModelConfig
    EDGE_ENCODER_DIM="edge_encoder_dim",
    EDGE_ENCODER_DROPOUT="edge_encoder_dropout",
    EDGE_ENCODER_INIT_BIAS="edge_encoder_init_bias",
)

EDGE_ENCODER_DEFS = SimpleNamespace(
    DEFAULT_HIDDEN_DIM=32,
    DEFAULT_DROPOUT=0.1,
    DEFAULT_INIT_BIAS=None,
)

EDGE_ENCODER_ARGS_TO_MODEL_CONFIG_NAMES = {
    EDGE_ENCODER_ARGS.HIDDEN_DIM: EDGE_ENCODER_ARGS.EDGE_ENCODER_DIM,
    EDGE_ENCODER_ARGS.DROPOUT: EDGE_ENCODER_ARGS.EDGE_ENCODER_DROPOUT,
    EDGE_ENCODER_ARGS.INIT_BIAS: EDGE_ENCODER_ARGS.EDGE_ENCODER_INIT_BIAS,
}

EDGE_WEIGHTING_TYPE = SimpleNamespace(
    NONE="none",
    STATIC_WEIGHTS="static_weights",
    LEARNED_ENCODER="learned_encoder",
)

ENCODERS_SUPPORTING_EDGE_WEIGHTING = {
    ENCODERS.GCN,
    ENCODERS.GRAPH_CONV,
}
