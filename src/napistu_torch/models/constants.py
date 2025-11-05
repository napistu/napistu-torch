from types import SimpleNamespace

MODEL_DEFS = SimpleNamespace(
    ENCODER="encoder",
    ENCODER_TYPE="encoder_type",
    GCN="gcn",
    HEAD="head",
    HEAD_TYPE="head_type",
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
    DROPOUT="dropout",
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
        ENCODER_SPECIFIC_ARGS.DROPOUT: "dropout",
        ENCODER_SPECIFIC_ARGS.GAT_CONCAT: "concat",
    },
    ENCODERS.GRAPH_CONV: {
        ENCODER_SPECIFIC_ARGS.GRAPH_CONV_AGGREGATOR: "aggr",
    },
    ENCODERS.SAGE: {ENCODER_SPECIFIC_ARGS.SAGE_AGGREGATOR: "aggr"},
}

HEADS = SimpleNamespace(
    DOT_PRODUCT="dot_product",
    MLP="mlp",
    BILINEAR="bilinear",
    NODE_CLASSIFICATION="node_classification",
)

VALID_HEADS = list(HEADS.__dict__.values())

# Head-specific parameter names
HEAD_SPECIFIC_ARGS = SimpleNamespace(
    MLP_HIDDEN_DIM="mlp_hidden_dim",
    MLP_NUM_LAYERS="mlp_num_layers",
    MLP_DROPOUT="mlp_dropout",
    BILINEAR_BIAS="bilinear_bias",
    NC_NUM_CLASSES="nc_num_classes",
    NC_DROPOUT="nc_dropout",
)

EDGE_ENCODER_ARGS = SimpleNamespace(
    EDGE_IN_CHANNELS="edge_in_channels",
    EDGE_ENCODER_DIM="edge_encoder_dim",
    EDGE_ENCODER_DROPOUT="edge_encoder_dropout",
)

EDGE_WEIGHTING_TYPE = SimpleNamespace(
    NONE="none",
    STATIC_WEIGHTS="static_weights",
    LEARNED_ENCODER="learned_encoder",
)

ENCODERS_SUPPORTING_EDGE_WEIGHTING = {
    ENCODERS.GCN,
    ENCODERS.GRAPH_CONV,
}
