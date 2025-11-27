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

METRIC_SUMMARIES = SimpleNamespace(
    VAL_AUC="val_auc",
    TEST_AUC="test_auc",
    VAL_AP="val_ap",
    TEST_AP="test_ap",
    TRAIN_LOSS="train_loss",
    BEST_EPOCH="epoch",
)

# Lookup table for nice display names
METRIC_DISPLAY_NAMES = {
    METRIC_SUMMARIES.VAL_AUC: "Validation AUC",
    METRIC_SUMMARIES.TEST_AUC: "Test AUC",
    METRIC_SUMMARIES.VAL_AP: "Validation AP",
    METRIC_SUMMARIES.TEST_AP: "Test AP",
    METRIC_SUMMARIES.TRAIN_LOSS: "Training Loss",
    METRIC_SUMMARIES.BEST_EPOCH: "Best Epoch",
}

# Default metrics to include in model cards
DEFAULT_MODEL_CARD_METRICS = [
    METRIC_SUMMARIES.VAL_AUC,
    METRIC_SUMMARIES.TEST_AUC,
    METRIC_SUMMARIES.VAL_AP,
    METRIC_SUMMARIES.TEST_AP,
    METRIC_SUMMARIES.BEST_EPOCH,
]
