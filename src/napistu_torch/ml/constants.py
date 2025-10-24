from types import SimpleNamespace

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
