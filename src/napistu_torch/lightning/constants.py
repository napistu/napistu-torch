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
