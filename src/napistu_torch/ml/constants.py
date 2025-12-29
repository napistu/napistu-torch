from types import SimpleNamespace

# this shouldn't declare imports since it is imported into the top-level constants.py

TRAINING = SimpleNamespace(
    TRAIN="train",
    TEST="test",
    VALIDATION="validation",
)

# Mapping from split names to mask attribute names
SPLIT_TO_MASK = {
    TRAINING.TRAIN: "train_mask",
    TRAINING.TEST: "test_mask",
    TRAINING.VALIDATION: "val_mask",
}

VALID_SPLITS = list(SPLIT_TO_MASK.keys())

DEVICE = SimpleNamespace(
    CPU="cpu",
    GPU="gpu",
    MPS="mps",
)

VALID_DEVICES = list(DEVICE.__dict__.values())

# metrics

METRICS = SimpleNamespace(
    AUC="auc",
    AP="ap",
)

VALID_METRICS = list(METRICS.__dict__.values())

RELATION_WEIGHTED_AUC_DEFS = SimpleNamespace(
    RELATION_WEIGHTED_AUC="relation_weighted_auc",
    RELATION_AUC_TEMPLATE="auc_{relation_name}",
)

METRIC_SUMMARIES = SimpleNamespace(
    VAL_RELATION_WEIGHTED_AUC="val_relation_weighted_auc",
    TEST_RELATION_WEIGHTED_AUC="test_relation_weighted_auc",
    VAL_AUC="val_auc",
    TEST_AUC="test_auc",
    VAL_AP="val_ap",
    TEST_AP="test_ap",
    TRAIN_LOSS="train_loss",
    BEST_EPOCH="epoch",
)

# Lookup table for nice display names
METRIC_DISPLAY_NAMES = {
    METRIC_SUMMARIES.VAL_RELATION_WEIGHTED_AUC: "Validation relation-weighted AUC",
    METRIC_SUMMARIES.TEST_RELATION_WEIGHTED_AUC: "Test relation-weighted AUC",
    METRIC_SUMMARIES.VAL_AUC: "Validation AUC",
    METRIC_SUMMARIES.TEST_AUC: "Test AUC",
    METRIC_SUMMARIES.VAL_AP: "Validation AP",
    METRIC_SUMMARIES.TEST_AP: "Test AP",
    METRIC_SUMMARIES.TRAIN_LOSS: "Training Loss",
    METRIC_SUMMARIES.BEST_EPOCH: "Best Epoch",
}

# Default metrics to include in model cards
DEFAULT_MODEL_CARD_METRICS = [
    METRIC_SUMMARIES.VAL_RELATION_WEIGHTED_AUC,
    METRIC_SUMMARIES.TEST_RELATION_WEIGHTED_AUC,
    METRIC_SUMMARIES.VAL_AUC,
    METRIC_SUMMARIES.TEST_AUC,
    METRIC_SUMMARIES.VAL_AP,
    METRIC_SUMMARIES.TEST_AP,
    METRIC_SUMMARIES.BEST_EPOCH,
]

# Score distribution statistics keys
SCORE_DISTRIBUTION_STATS = SimpleNamespace(
    HEAD_TYPE="head_type",
    SPLIT="split",
    POS_SCORE_MEAN="pos_score_mean",
    POS_SCORE_STD="pos_score_std",
    POS_SCORE_MIN="pos_score_min",
    POS_SCORE_MAX="pos_score_max",
    NEG_SCORE_MEAN="neg_score_mean",
    NEG_SCORE_STD="neg_score_std",
    NEG_SCORE_MIN="neg_score_min",
    NEG_SCORE_MAX="neg_score_max",
    SEPARATION_COHENS_D="separation_cohens_d",
    POS_SATURATED_PCT="pos_saturated_pct",
    NEG_SATURATED_PCT="neg_saturated_pct",
    RANK_CORR_WITH_DOTPROD="rank_corr_with_dotprod",
)

LOSSES = SimpleNamespace(
    BCE="bce",
    MARGIN="margin",
)

VALID_LOSSES = list(LOSSES.__dict__.values())

# hugging face

DEFAULT_HUGGING_FACE_TAGS = [
    "napistu",
    "napistu-torch",
    "graph-neural-networks",
    "biological-networks",
    "pytorch",
]

HUGGING_FACE_REPOS = SimpleNamespace(
    MODEL="model",
    DATASET="dataset",
)

VALID_HUGGING_FACE_REPOS = list(HUGGING_FACE_REPOS.__dict__.values())

# wandb

WANDB_INFO = SimpleNamespace(
    RUN_SUMMARIES="run_summaries",
    WANDB_ENTITY="wandb_entity",
    WANDB_PROJECT="wandb_project",
    WANDB_RUN_ID="wandb_run_id",
    RUN_PATH="run_path",
)
