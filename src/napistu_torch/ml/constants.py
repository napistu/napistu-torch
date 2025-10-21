from types import SimpleNamespace

TRAINING = SimpleNamespace(
    TRAIN="train",
    TEST="test",
    VALIDATION="validation",
    SPLIT_MASK_TEMPLATE="{split_name}_mask",
)

DEVICE = SimpleNamespace(
    CPU="cpu",
    GPU="gpu",
    MPS="mps",
)

VALID_DEVICES = list(DEVICE.__dict__.values())
