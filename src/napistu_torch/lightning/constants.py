"""Lightning-specific constants."""

from types import SimpleNamespace

EXPERIMENT_DICT = SimpleNamespace(
    DATA_MODULE="data_module",
    MODEL="model",
    TRAINER="trainer",
    WANDB_LOGGER="wandb_logger",
)
