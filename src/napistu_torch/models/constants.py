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
    GRAPH_CONV_DEFAULT_AGGREGATOR="mean",
    SAGE_DEFAULT_AGGREGATOR="mean",
    # derived encoder attributes
    EDGE_WEIGHTING_TYPE="edge_weighting_type",
    EDGE_WEIGHTING_VALUE="edge_weighting_value",
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
    MLP="mlp",
    BILINEAR="bilinear",
    # relation prediction heads
    ROTATE="rotate",
    TRANSE="transe",
    DISTMULT="distmult",
)

VALID_HEADS = list(HEADS.__dict__.values())

RELATION_AWARE_HEADS = {HEADS.ROTATE, HEADS.TRANSE, HEADS.DISTMULT}
EDGE_PREDICTION_HEADS = {
    HEADS.DOT_PRODUCT,
    HEADS.MLP,
    HEADS.BILINEAR,
} | RELATION_AWARE_HEADS

# Head-specific parameter names
HEAD_SPECIFIC_ARGS = SimpleNamespace(
    MLP_HIDDEN_DIM="mlp_hidden_dim",
    MLP_NUM_LAYERS="mlp_num_layers",
    MLP_DROPOUT="mlp_dropout",
    BILINEAR_BIAS="bilinear_bias",
    NC_DROPOUT="nc_dropout",
    ROTATE_MARGIN="rotate_margin",
    TRANSE_MARGIN="transe_margin",
)

EDGE_ENCODER_ARGS = SimpleNamespace(
    HIDDEN_DIM="hidden_dim",
    DROPOUT="dropout",
    INIT_BIAS="init_bias",
    # names used in the ModelConfig
    EDGE_ENCODER_DIM="edge_encoder_dim",
    EDGE_ENCODER_DROPOUT="edge_encoder_dropout",
    EDGE_ENCODER_INIT_BIAS="edge_encoder_init_bias",
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
