"""Lightning-specific constants."""

from types import SimpleNamespace

EXPERIMENT_DICT = SimpleNamespace(
    DATA_MODULE="data_module",
    MODEL="model",
    TRAINER="trainer",
    RUN_MANIFEST="run_manifest",
    WANDB_LOGGER="wandb_logger",
)

TRAINER_MODES = SimpleNamespace(
    TRAIN="train",
    EVAL="eval",
)

VALID_TRAINER_MODES = list(TRAINER_MODES.__dict__.values())

NAPISTU_DATA_MODULE = SimpleNamespace(
    NAPISTU_DATA="napistu_data",
    TRAIN_DATA="train_data",
    VAL_DATA="val_data",
    TEST_DATA="test_data",
    DATA="data",
)
