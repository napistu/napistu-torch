from __future__ import annotations

from types import SimpleNamespace

from napistu_torch.ml.constants import SPLIT_TO_MASK, TRAINING

ARTIFACT_TYPES = SimpleNamespace(
    NAPISTU_DATA="napistu_data",
    VERTEX_TENSOR="vertex_tensor",
    PANDAS_DFS="pandas_dfs",
)

VALID_ARTIFACT_TYPES = list(ARTIFACT_TYPES.__dict__.values())

NAPISTU_DATA = SimpleNamespace(
    EDGE_ATTR="edge_attr",
    EDGE_FEATURE_NAMES="edge_feature_names",
    EDGE_FEATURE_NAME_ALIASES="edge_feature_name_aliases",
    EDGE_INDEX="edge_index",
    EDGE_WEIGHT="edge_weight",
    NG_EDGE_NAMES="ng_edge_names",
    NG_VERTEX_NAMES="ng_vertex_names",
    VERTEX_FEATURE_NAMES="vertex_feature_names",
    VERTEX_FEATURE_NAME_ALIASES="vertex_feature_name_aliases",
    X="x",
    Y="y",
    NAME="name",
    SPLITTING_STRATEGY="splitting_strategy",
    LABELING_MANAGER="labeling_manager",
    TRAIN_MASK=SPLIT_TO_MASK[TRAINING.TRAIN],
    TEST_MASK=SPLIT_TO_MASK[TRAINING.TEST],
    VAL_MASK=SPLIT_TO_MASK[TRAINING.VALIDATION],
)

NAPISTU_DATA_DEFAULT_NAME = "default"

VERTEX_TENSOR = SimpleNamespace(
    DATA="data",
    FEATURE_NAMES="feature_names",
    VERTEX_NAMES="vertex_names",
    NAME="name",
    DESCRIPTION="description",
)

# defs in the json/config
NAPISTU_DATA_STORE = SimpleNamespace(
    # top-level categories
    NAPISTU_RAW="napistu_raw",
    NAPISTU_DATA="napistu_data",
    VERTEX_TENSORS="vertex_tensors",
    PANDAS_DFS="pandas_dfs",
    # attributes
    SBML_DFS="sbml_dfs",
    NAPISTU_GRAPH="napistu_graph",
    OVERWRITE="overwrite",
    # metadata
    LAST_MODIFIED="last_modified",
    CREATED="created",
    FILENAME="filename",
    PT_TEMPLATE="{name}.pt",
    PARQUET_TEMPLATE="{name}.parquet",
)

NAPISTU_DATA_STORE_STRUCTURE = SimpleNamespace(
    REGISTRY_FILE="registry.json",
    # file directories
    NAPISTU_RAW=NAPISTU_DATA_STORE.NAPISTU_RAW,
    NAPISTU_DATA=NAPISTU_DATA_STORE.NAPISTU_DATA,
    VERTEX_TENSORS=NAPISTU_DATA_STORE.VERTEX_TENSORS,
    PANDAS_DFS=NAPISTU_DATA_STORE.PANDAS_DFS,
)

METRICS = SimpleNamespace(
    AUC="auc",
    AP="ap",
)

VALID_METRICS = list(METRICS.__dict__.values())

OPTIMIZERS = SimpleNamespace(
    ADAM="adam",
    ADAMW="adamw",
)

VALID_OPTIMIZERS = list(OPTIMIZERS.__dict__.values())

SCHEDULERS = SimpleNamespace(
    PLATEAU="plateau",
    COSINE="cosine",
)

VALID_SCHEDULERS = list(SCHEDULERS.__dict__.values())

WANDB_MODES = SimpleNamespace(
    ONLINE="online",
    OFFLINE="offline",
    DISABLED="disabled",
)
VALID_WANDB_MODES = list(WANDB_MODES.__dict__.values())

DATA_CONFIG = SimpleNamespace(
    NAME="name",
    STORE_DIR="store_dir",
    SBML_DFS_PATH="sbml_dfs_path",
    NAPISTU_GRAPH_PATH="napistu_graph_path",
    COPY_TO_STORE="copy_to_store",
    OVERWRITE="overwrite",
    NAPISTU_DATA_NAME="napistu_data_name",
    OTHER_ARTIFACTS="other_artifacts",
)

MODEL_CONFIG = SimpleNamespace(
    ENCODER="encoder",  # for brevity, maps to encoder_type in models.constants.ENCODERS
    HEAD="head",  # for brevity, maps to head_type in models.constants.HEADS
    USE_EDGE_ENCODER="use_edge_encoder",
    EDGE_IN_CHANNELS="edge_in_channels",
    EDGE_ENCODER_DIM="edge_encoder_dim",
    EDGE_ENCODER_DROPOUT="edge_encoder_dropout",
)

TASK_CONFIG = SimpleNamespace(
    TASK="task",
    METRICS="metrics",
    EDGE_PREDICTION_NEG_SAMPLING_RATIO="edge_prediction_neg_sampling_ratio",
    EDGE_PREDICTION_NEG_SAMPLING_STRATIFY_BY="edge_prediction_neg_sampling_stratify_by",
    EDGE_PREDICTION_NEG_SAMPLING_STRATEGY="edge_prediction_neg_sampling_strategy",
)

TRAINING_CONFIG = SimpleNamespace(
    LR="lr",
    WEIGHT_DECAY="weight_decay",
    OPTIMIZER="optimizer",
    SCHEDULER="scheduler",
    EPOCHS="epochs",
    BATCH_SIZE="batch_size",
    ACCELERATOR="accelerator",
    DEVICES="devices",
    PRECISION="precision",
    EARLY_STOPPING="early_stopping",
    EARLY_STOPPING_PATIENCE="early_stopping_patience",
    EARLY_STOPPING_METRIC="early_stopping_metric",
    CHECKPOINT_DIR="checkpoint_dir",
    SAVE_CHECKPOINTS="save_checkpoints",
    CHECKPOINT_METRIC="checkpoint_metric",
)

WANDB_CONFIG = SimpleNamespace(
    PROJECT="project",
    ENTITY="entity",
    GROUP="group",
    TAGS="tags",
    SAVE_DIR="save_dir",
    LOG_MODEL="log_model",
    MODE="mode",
)

EXPERIMENT_CONFIG = SimpleNamespace(
    NAME="name",
    SEED="seed",
    DETERMINISTIC="deterministic",
    FAST_DEV_RUN="fast_dev_run",
    LIMIT_TRAIN_BATCHES="limit_train_batches",
    LIMIT_VAL_BATCHES="limit_val_batches",
    MODEL="model",
    DATA="data",
    TASK="task",
    TRAINING="training",
    WANDB="wandb",
)
