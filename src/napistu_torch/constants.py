from __future__ import annotations

from types import SimpleNamespace

NAPISTU_DATA = SimpleNamespace(
    EDGE_ATTR="edge_attr",
    EDGE_FEATURE_NAMES="edge_feature_names",
    EDGE_INDEX="edge_index",
    EDGE_WEIGHT="edge_weight",
    NG_EDGE_NAMES="ng_edge_names",
    NG_VERTEX_NAMES="ng_vertex_names",
    VERTEX_FEATURE_NAMES="vertex_feature_names",
    X="x",
    Y="y",
    NAME="name",
    SPLITTING_STRATEGY="splitting_strategy",
    LABELING_MANAGER="labeling_manager",
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
    # attributes
    SBML_DFS="sbml_dfs",
    NAPISTU_GRAPH="napistu_graph",
    OVERWRITE="overwrite",
    # metadata
    LAST_MODIFIED="last_modified",
    CREATED="created",
    FILENAME="filename",
    PT_TEMPLATE="{name}.pt",
)

NAPISTU_DATA_STORE_STRUCTURE = SimpleNamespace(
    REGISTRY_FILE="registry.json",
    # file directories
    NAPISTU_RAW=NAPISTU_DATA_STORE.NAPISTU_RAW,
    NAPISTU_DATA=NAPISTU_DATA_STORE.NAPISTU_DATA,
    VERTEX_TENSORS=NAPISTU_DATA_STORE.VERTEX_TENSORS,
)

ENCODER_TYPES = SimpleNamespace(
    SAGE="sage",
    GCN="gcn",
    GAT="gat",
)

VALID_ENCODER_TYPES = list(ENCODER_TYPES.__dict__.values())

HEADS = SimpleNamespace(
    DOT_PRODUCT="dot_product",
    MLP="mlp",
    BILINEAR="bilinear",
)

VALID_HEADS = list(HEADS.__dict__.values())

TASKS = SimpleNamespace(
    EDGE_PREDICTION="edge_prediction",
    NETWORK_EMBEDDING="network_embedding",
    NODE_CLASSIFICATION="node_classification",
)

VALID_TASKS = list(TASKS.__dict__.values())

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


MODEL_CONFIG = SimpleNamespace(
    ENCODER_TYPE="encoder_type",
    HIDDEN_CHANNELS="hidden_channels",
    NUM_LAYERS="num_layers",
    DROPOUT="dropout",
    HEAD_TYPE="head_type",
    AGGREGATOR="aggregator",
    HEADS="heads",
    HEAD_HIDDEN_DIM="head_hidden_dim",
)

DATA_CONFIG = SimpleNamespace(
    NAME="name",
    STORE_DIR="store_dir",
    SPLITTING_STRATEGY="splitting_strategy",
    TRAIN_SIZE="train_size",
    VAL_SIZE="val_size",
    TEST_SIZE="test_size",
)

TASK_CONFIG = SimpleNamespace(
    TASK="task",
    NEG_SAMPLING_RATIO="neg_sampling_ratio",
    METRICS="metrics",
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
