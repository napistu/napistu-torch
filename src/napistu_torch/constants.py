from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from napistu_torch.load.constants import STRATIFY_BY
from napistu_torch.ml.constants import SPLIT_TO_MASK, TRAINING
from napistu_torch.models.constants import (
    EDGE_ENCODER_ARGS,
    ENCODER_SPECIFIC_ARGS,
    ENCODERS,
    HEADS,
    MODEL_DEFS,
)
from napistu_torch.tasks.constants import (
    NEGATIVE_SAMPLING_STRATEGIES,
    TASKS,
)

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

NAPISTU_DATA_TRIM_ARGS = SimpleNamespace(
    KEEP_EDGE_ATTR="keep_edge_attr",
    KEEP_LABELS="keep_labels",
    KEEP_MASKS="keep_masks",
)

# VertexTensor

VERTEX_TENSOR = SimpleNamespace(
    DATA="data",
    FEATURE_NAMES="feature_names",
    VERTEX_NAMES="vertex_names",
    NAME="name",
    DESCRIPTION="description",
)

# NapistuDataStore

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

# Configs

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
    STORE_DIR="store_dir",
    SBML_DFS_PATH="sbml_dfs_path",
    NAPISTU_GRAPH_PATH="napistu_graph_path",
    COPY_TO_STORE="copy_to_store",
    OVERWRITE="overwrite",
    NAPISTU_DATA_NAME="napistu_data_name",
    OTHER_ARTIFACTS="other_artifacts",
)

DATA_CONFIG_DEFAULTS = {
    DATA_CONFIG.STORE_DIR: Path("./.store"),
    DATA_CONFIG.NAPISTU_DATA_NAME: "edge_prediction",
}

MODEL_CONFIG = SimpleNamespace(
    ENCODER="encoder",  # for brevity, maps to encoder_type in models.constants.ENCODERS
    HEAD="head",  # for brevity, maps to head_type in models.constants.HEADS
    USE_EDGE_ENCODER="use_edge_encoder",
    HIDDEN_CHANNELS=MODEL_DEFS.HIDDEN_CHANNELS,
    NUM_LAYERS=MODEL_DEFS.NUM_LAYERS,
    DROPOUT=ENCODER_SPECIFIC_ARGS.DROPOUT,
    GAT_HEADS=ENCODER_SPECIFIC_ARGS.GAT_HEADS,
    GAT_CONCAT=ENCODER_SPECIFIC_ARGS.GAT_CONCAT,
    GRAPH_CONV_AGGREGATOR=ENCODER_SPECIFIC_ARGS.GRAPH_CONV_AGGREGATOR,
    SAGE_AGGREGATOR=ENCODER_SPECIFIC_ARGS.SAGE_AGGREGATOR,
    EDGE_IN_CHANNELS=EDGE_ENCODER_ARGS.EDGE_IN_CHANNELS,
    EDGE_ENCODER_DIM=EDGE_ENCODER_ARGS.EDGE_ENCODER_DIM,
    EDGE_ENCODER_DROPOUT=EDGE_ENCODER_ARGS.EDGE_ENCODER_DROPOUT,
)

MODEL_CONFIG_DEFAULTS = {
    MODEL_CONFIG.ENCODER: ENCODERS.SAGE,
    MODEL_CONFIG.HEAD: HEADS.DOT_PRODUCT,
    MODEL_CONFIG.USE_EDGE_ENCODER: False,
}

TASK_CONFIG = SimpleNamespace(
    TASK="task",
    METRICS="metrics",
    EDGE_PREDICTION_NEG_SAMPLING_RATIO="edge_prediction_neg_sampling_ratio",
    EDGE_PREDICTION_NEG_SAMPLING_STRATIFY_BY="edge_prediction_neg_sampling_stratify_by",
    EDGE_PREDICTION_NEG_SAMPLING_STRATEGY="edge_prediction_neg_sampling_strategy",
)

TASK_CONFIG_DEFAULTS = {
    TASK_CONFIG.TASK: TASKS.EDGE_PREDICTION,
    TASK_CONFIG.EDGE_PREDICTION_NEG_SAMPLING_STRATIFY_BY: STRATIFY_BY.NODE_TYPE,
    TASK_CONFIG.EDGE_PREDICTION_NEG_SAMPLING_STRATEGY: NEGATIVE_SAMPLING_STRATEGIES.DEGREE_WEIGHTED,
}

TRAINING_CONFIG = SimpleNamespace(
    LR="lr",
    WEIGHT_DECAY="weight_decay",
    OPTIMIZER="optimizer",
    SCHEDULER="scheduler",
    EPOCHS="epochs",
    BATCHES_PER_EPOCH="batches_per_epoch",
    ACCELERATOR="accelerator",
    DEVICES="devices",
    PRECISION="precision",
    EARLY_STOPPING="early_stopping",
    EARLY_STOPPING_PATIENCE="early_stopping_patience",
    EARLY_STOPPING_METRIC="early_stopping_metric",
    CHECKPOINT_SUBDIR="checkpoint_subdir",
    SAVE_CHECKPOINTS="save_checkpoints",
    CHECKPOINT_METRIC="checkpoint_metric",
)

TRAINING_CONFIG_DEFAULTS = {
    TRAINING_CONFIG.CHECKPOINT_SUBDIR: "checkpoints",
}

WANDB_CONFIG = SimpleNamespace(
    PROJECT="project",
    ENTITY="entity",
    GROUP="group",
    TAGS="tags",
    LOG_MODEL="log_model",
    MODE="mode",
    WANDB_SUBDIR="wandb_subdir",
)

WANDB_CONFIG_DEFAULTS = {
    WANDB_CONFIG.ENTITY: "napistu",
    WANDB_CONFIG.PROJECT: "napistu-experiments",
    WANDB_CONFIG.GROUP: "baseline",
    WANDB_CONFIG.TAGS: [],
    WANDB_CONFIG.LOG_MODEL: False,
    WANDB_CONFIG.MODE: WANDB_MODES.ONLINE,
    WANDB_CONFIG.WANDB_SUBDIR: "logs",
}

EXPERIMENT_CONFIG = SimpleNamespace(
    NAME="name",
    SEED="seed",
    DETERMINISTIC="deterministic",
    FAST_DEV_RUN="fast_dev_run",
    LIMIT_TRAIN_BATCHES="limit_train_batches",
    LIMIT_VAL_BATCHES="limit_val_batches",
    OUTPUT_DIR="output_dir",
    MODEL="model",
    DATA="data",
    TASK="task",
    TRAINING="training",
    WANDB="wandb",
)

EXPERIMENT_CONFIG_DEFAULTS = {
    EXPERIMENT_CONFIG.NAME: None,
    EXPERIMENT_CONFIG.SEED: 42,
    EXPERIMENT_CONFIG.OUTPUT_DIR: Path("./output"),
}

RUN_MANIFEST = SimpleNamespace(
    EXPERIMENT_NAME="experiment_name",
    WANDB_RUN_ID="wandb_run_id",
    WANDB_RUN_URL="wandb_run_url",
    WANDB_PROJECT="wandb_project",
    WANDB_ENTITY="wandb_entity",
    EXPERIMENT_CONFIG="experiment_config",
    MANIFEST_FILENAME="manifest_filename",
)

RUN_MANIFEST_DEFAULTS = {
    RUN_MANIFEST.MANIFEST_FILENAME: "run_manifest.yaml",
}

OPTIONAL_DEPENDENCIES = SimpleNamespace(
    VIZ="viz",
    WANDB="wandb",
    LIGHTNING="lightning",
)

OPTIONAL_DEFS = SimpleNamespace(
    LIGHTNING_PACKAGE="pytorch_lightning",
    LIGHTNING_EXTRA=OPTIONAL_DEPENDENCIES.LIGHTNING,
)
