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

# callbacks

# Embedding norm statistics keys
EMBEDDING_NORM_STATS = SimpleNamespace(
    EMBEDDING_NORM_MEAN="embedding_norm_mean",
    EMBEDDING_NORM_MEDIAN="embedding_norm_median",
    EMBEDDING_NORM_STD="embedding_norm_std",
    EMBEDDING_NORM_MAX="embedding_norm_max",
)

# Experiment timing statistics keys
EXPERIMENT_TIMING_STATS = SimpleNamespace(
    EPOCH_DURATION_SECONDS="epoch_duration_seconds",
    AVG_EPOCH_DURATION="avg_epoch_duration",
    TOTAL_TRAIN_TIME_MINUTES="total_train_time_minutes",
    TOTAL_EPOCHS_COMPLETED="total_epochs_completed",
    TIME_PER_EPOCH_AVG="time_per_epoch_avg",
    TIME_PER_EPOCH_STD="time_per_epoch_std",
    START_TIME="start_time",
    EPOCH_TIMES="epoch_times",
    EPOCH_START="epoch_start",
)
