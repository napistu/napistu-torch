from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from napistu_torch.load.constants import STRATIFY_BY
from napistu_torch.ml.constants import SPLIT_TO_MASK, TRAINING
from napistu_torch.models.constants import (
    EDGE_ENCODER_ARGS,
    ENCODER_SPECIFIC_ARGS,
    ENCODERS,
    HEAD_SPECIFIC_ARGS,
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

PYG = SimpleNamespace(
    X="x",
    Y="y",
    EDGE_INDEX="edge_index",
    EDGE_WEIGHT="edge_weight",
    EDGE_ATTR="edge_attr",
    # properites
    NUM_NODES="num_nodes",
    NUM_EDGES="num_edges",
    NUM_NODE_FEATURES="num_node_features",
    NUM_EDGE_FEATURES="num_edge_features",
)

NAPISTU_DATA = SimpleNamespace(
    EDGE_FEATURE_NAMES="edge_feature_names",
    EDGE_FEATURE_NAME_ALIASES="edge_feature_name_aliases",
    NG_EDGE_NAMES="ng_edge_names",
    NG_VERTEX_NAMES="ng_vertex_names",
    VERTEX_FEATURE_NAMES="vertex_feature_names",
    VERTEX_FEATURE_NAME_ALIASES="vertex_feature_name_aliases",
    NAME="name",
    SPLITTING_STRATEGY="splitting_strategy",
    LABELS="labels",
    LABELING_MANAGER="labeling_manager",
    RELATION_TYPE="relation_type",
    RELATION_MANAGER="relation_manager",
    TRAIN_MASK=SPLIT_TO_MASK[TRAINING.TRAIN],
    TEST_MASK=SPLIT_TO_MASK[TRAINING.TEST],
    VAL_MASK=SPLIT_TO_MASK[TRAINING.VALIDATION],
)

NAPISTU_DATA_DEFAULT_NAME = "default"

NAPISTU_DATA_TRIM_ARGS = SimpleNamespace(
    KEEP_EDGE_ATTR="keep_edge_attr",
    KEEP_LABELS="keep_labels",
    KEEP_MASKS="keep_masks",
    KEEP_RELATION_TYPE="keep_relation_type",
)

NAPISTU_DATA_SUMMARIES = SimpleNamespace(
    HAS_VERTEX_FEATURE_NAMES="has_vertex_feature_names",
    HAS_EDGE_FEATURE_NAMES="has_edge_feature_names",
    HAS_EDGE_WEIGHTS="has_edge_weights",
    HAS_NG_VERTEX_NAMES="has_ng_vertex_names",
    HAS_NG_EDGE_NAMES="has_ng_edge_names",
    HAS_SPLITTING_STRATEGY="has_splitting_strategy",
    HAS_LABELING_MANAGER="has_labeling_manager",
    HAS_RELATION_MANAGER="has_relation_manager",
    NUM_UNIQUE_RELATIONS="num_unique_relations",
    NUM_TRAIN_EDGES="num_train_edges",
    NUM_VAL_EDGES="num_val_edges",
    NUM_TEST_EDGES="num_test_edges",
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
    ENCODER=MODEL_DEFS.ENCODER,  # for brevity, maps to encoder_type in models.constants.ENCODERS
    HEAD=MODEL_DEFS.HEAD,  # for brevity, maps to head_type in models.constants.HEADS
    USE_EDGE_ENCODER="use_edge_encoder",
    # encoders
    HIDDEN_CHANNELS=MODEL_DEFS.HIDDEN_CHANNELS,
    NUM_LAYERS=MODEL_DEFS.NUM_LAYERS,
    DROPOUT=MODEL_DEFS.DROPOUT,
    GAT_HEADS=ENCODER_SPECIFIC_ARGS.GAT_HEADS,
    GAT_CONCAT=ENCODER_SPECIFIC_ARGS.GAT_CONCAT,
    GRAPH_CONV_AGGREGATOR=ENCODER_SPECIFIC_ARGS.GRAPH_CONV_AGGREGATOR,
    SAGE_AGGREGATOR=ENCODER_SPECIFIC_ARGS.SAGE_AGGREGATOR,
    # edge encoders
    EDGE_ENCODER_DIM=EDGE_ENCODER_ARGS.EDGE_ENCODER_DIM,
    EDGE_ENCODER_DROPOUT=EDGE_ENCODER_ARGS.EDGE_ENCODER_DROPOUT,
    EDGE_ENCODER_INIT_BIAS=EDGE_ENCODER_ARGS.EDGE_ENCODER_INIT_BIAS,
    # heads
    MLP_HIDDEN_DIM=HEAD_SPECIFIC_ARGS.MLP_HIDDEN_DIM,
    MLP_NUM_LAYERS=HEAD_SPECIFIC_ARGS.MLP_NUM_LAYERS,
    MLP_DROPOUT=HEAD_SPECIFIC_ARGS.MLP_DROPOUT,
    BILINEAR_BIAS=HEAD_SPECIFIC_ARGS.BILINEAR_BIAS,
    NC_DROPOUT=HEAD_SPECIFIC_ARGS.NC_DROPOUT,
    ROTATE_MARGIN=HEAD_SPECIFIC_ARGS.ROTATE_MARGIN,
    TRANSE_MARGIN=HEAD_SPECIFIC_ARGS.TRANSE_MARGIN,
    # loading
    ENCODER_SOURCE="encoder_source",
    ENCODER_PATH="encoder_path",
    ENCODER_REVISION="encoder_revision",
    HEAD_SOURCE="head_source",
    HEAD_PATH="head_path",
    HEAD_REVISION="head_revision",
)

MODEL_CONFIG_DEFAULTS = {
    MODEL_CONFIG.ENCODER: ENCODERS.SAGE,
    MODEL_CONFIG.HEAD: HEADS.DOT_PRODUCT,
    MODEL_CONFIG.USE_EDGE_ENCODER: False,
}

# split up the architecture specification by component
MODEL_COMPONENTS = {
    MODEL_DEFS.ENCODER: {
        MODEL_CONFIG.ENCODER,
        MODEL_CONFIG.HIDDEN_CHANNELS,
        MODEL_CONFIG.NUM_LAYERS,
        MODEL_CONFIG.DROPOUT,
        MODEL_CONFIG.GAT_HEADS,
        MODEL_CONFIG.GAT_CONCAT,
        MODEL_CONFIG.GRAPH_CONV_AGGREGATOR,
        MODEL_CONFIG.SAGE_AGGREGATOR,
    },
    MODEL_DEFS.HEAD: {
        MODEL_DEFS.HEAD,
        MODEL_CONFIG.MLP_HIDDEN_DIM,
        MODEL_CONFIG.MLP_NUM_LAYERS,
        MODEL_CONFIG.MLP_DROPOUT,
        MODEL_CONFIG.BILINEAR_BIAS,
        MODEL_CONFIG.NC_DROPOUT,
        MODEL_CONFIG.ROTATE_MARGIN,
        MODEL_CONFIG.TRANSE_MARGIN,
    },
    MODEL_DEFS.EDGE_ENCODER: {
        MODEL_CONFIG.EDGE_ENCODER_DIM,
        MODEL_CONFIG.EDGE_ENCODER_DROPOUT,
        MODEL_CONFIG.EDGE_ENCODER_INIT_BIAS,
    },
}

PRETRAINING_DEFS = SimpleNamespace(
    PRETRAINED="pretrained",
)
PRETRAINED_COMPONENT_SOURCES = SimpleNamespace(
    HUGGINGFACE="huggingface",
    LOCAL="local",
)
VALID_PRETRAINED_COMPONENT_SOURCES = list(
    PRETRAINED_COMPONENT_SOURCES.__dict__.values()
)

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

ANONYMIZATION_PLACEHOLDER_DEFAULT = "[REDACTED]"

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

# CLI

TRAIN_NAMED_ARGS = SimpleNamespace(
    SEED="seed",
    FAST_DEV_RUN="fast_dev_run",
    ENCODER="encoder",
    HEAD="head",
    HIDDEN_CHANNELS="hidden_channels",
    DROPOUT="dropout",
    LR="lr",
    OPTIMIZER="optimizer",
    WEIGHT_DECAY="weight_decay",
    EPOCHS="epochs",
    WANDB_GROUP="wandb_group",
    WANDB_MODE="wandb_mode",
)

PARAM_OVERRIDE_MAP = {
    TRAIN_NAMED_ARGS.SEED: EXPERIMENT_CONFIG.SEED,
    TRAIN_NAMED_ARGS.FAST_DEV_RUN: EXPERIMENT_CONFIG.FAST_DEV_RUN,
    TRAIN_NAMED_ARGS.ENCODER: f"{EXPERIMENT_CONFIG.MODEL}.{MODEL_CONFIG.ENCODER}",
    TRAIN_NAMED_ARGS.HEAD: f"{EXPERIMENT_CONFIG.MODEL}.{MODEL_CONFIG.HEAD}",
    TRAIN_NAMED_ARGS.HIDDEN_CHANNELS: f"{EXPERIMENT_CONFIG.MODEL}.{MODEL_CONFIG.HIDDEN_CHANNELS}",
    TRAIN_NAMED_ARGS.DROPOUT: f"{EXPERIMENT_CONFIG.MODEL}.{MODEL_CONFIG.DROPOUT}",
    TRAIN_NAMED_ARGS.LR: f"{EXPERIMENT_CONFIG.TRAINING}.{TRAINING_CONFIG.LR}",
    TRAIN_NAMED_ARGS.OPTIMIZER: f"{EXPERIMENT_CONFIG.TRAINING}.{TRAINING_CONFIG.OPTIMIZER}",
    TRAIN_NAMED_ARGS.WEIGHT_DECAY: f"{EXPERIMENT_CONFIG.TRAINING}.{TRAINING_CONFIG.WEIGHT_DECAY}",
    TRAIN_NAMED_ARGS.EPOCHS: f"{EXPERIMENT_CONFIG.TRAINING}.{TRAINING_CONFIG.EPOCHS}",
    TRAIN_NAMED_ARGS.WANDB_GROUP: f"{EXPERIMENT_CONFIG.WANDB}.{WANDB_CONFIG.GROUP}",
    TRAIN_NAMED_ARGS.WANDB_MODE: f"{EXPERIMENT_CONFIG.WANDB}.{WANDB_CONFIG.MODE}",
}
