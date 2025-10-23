from types import SimpleNamespace

ENCODERS = SimpleNamespace(
    SAGE="sage",
    GCN="gcn",
    GAT="gat",
)

VALID_ENCODERS = list(ENCODERS.__dict__.values())

ENCODER_SPECIFIC_ARGS = SimpleNamespace(
    DROPOUT="dropout",
    GAT_HEADS="gat_heads",
    GAT_CONCAT="gat_concat",
    SAGE_AGGREGATOR="sage_aggregator",
)

VALID_ENCODER_NAMED_ARGS = list(ENCODER_SPECIFIC_ARGS.__dict__.values())

# defaults and other miscellaneous encoder definitions
ENCODER_DEFS = SimpleNamespace(
    SAGE_DEFAULT_AGGREGATOR="mean",
)

# select the relevant arguments and convert from the {encoder}_{arg} convention back to just arg
ENCODER_NATIVE_ARGNAMES_MAPS = {
    ENCODERS.SAGE: {ENCODER_SPECIFIC_ARGS.SAGE_AGGREGATOR: "aggr"},
    ENCODERS.GAT: {
        ENCODER_SPECIFIC_ARGS.GAT_HEADS: "heads",
        ENCODER_SPECIFIC_ARGS.DROPOUT: "dropout",
        ENCODER_SPECIFIC_ARGS.GAT_CONCAT: "concat",
    },
}

HEADS = SimpleNamespace(
    DOT_PRODUCT="dot_product",
    MLP="mlp",
    BILINEAR="bilinear",
)

VALID_HEADS = list(HEADS.__dict__.values())
