import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from napistu_torch.constants import (
    DATA_CONFIG,
    DATA_CONFIG_DEFAULTS,
    EXPERIMENT_CONFIG,
    EXPERIMENT_CONFIG_DEFAULTS,
    METRICS,
    MODEL_CONFIG,
    MODEL_CONFIG_DEFAULTS,
    OPTIMIZERS,
    TASK_CONFIG,
    TASK_CONFIG_DEFAULTS,
    TRAINING_CONFIG,
    TRAINING_CONFIG_DEFAULTS,
    VALID_OPTIMIZERS,
    VALID_SCHEDULERS,
    VALID_WANDB_MODES,
    WANDB_CONFIG,
    WANDB_CONFIG_DEFAULTS,
)
from napistu_torch.load.artifacts import ensure_stratify_by_artifact_name
from napistu_torch.models.constants import (
    ENCODER_DEFS,
    ENCODERS_SUPPORTING_EDGE_WEIGHTING,
    MODEL_DEFS,
    VALID_ENCODERS,
    VALID_HEADS,
)
from napistu_torch.tasks.constants import (
    TASKS,
    VALID_TASKS,
)

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Model architecture configuration"""

    encoder: str = Field(default=MODEL_CONFIG_DEFAULTS[MODEL_CONFIG.ENCODER])
    hidden_channels: int = Field(default=128, gt=0)
    num_layers: int = Field(default=3, ge=1, le=10)
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    head: str = Field(default=MODEL_CONFIG_DEFAULTS[MODEL_CONFIG.HEAD])

    # Model-specific fields (optional, with defaults)
    gat_heads: Optional[int] = Field(default=4, gt=0)  # For GAT
    gat_concat: Optional[bool] = True  # For GAT
    graph_conv_aggregator: Optional[str] = (
        ENCODER_DEFS.GRAPH_CONV_DEFAULT_AGGREGATOR
    )  # For GraphConv
    sage_aggregator: Optional[str] = ENCODER_DEFS.SAGE_DEFAULT_AGGREGATOR  # For SAGE

    # Head-specific fields (optional, with defaults)
    mlp_hidden_dim: Optional[int] = 64  # For MLP head
    mlp_num_layers: Optional[int] = Field(default=2, ge=1)  # For MLP head
    mlp_dropout: Optional[float] = Field(default=0.1, ge=0.0, lt=1.0)  # For MLP head
    bilinear_bias: Optional[bool] = True  # For bilinear head
    nc_num_classes: Optional[int] = Field(
        default=2, ge=2
    )  # For node classification head
    nc_dropout: Optional[float] = Field(
        default=0.1, ge=0.0, lt=1.0
    )  # For node classification head

    # Edge encoder fields (optional, with defaults)
    use_edge_encoder: Optional[bool] = MODEL_CONFIG_DEFAULTS[
        MODEL_CONFIG.USE_EDGE_ENCODER
    ]  # Whether to use edge encoder
    edge_encoder_dim: Optional[int] = Field(default=32, gt=0)  # Edge encoder hidden dim
    edge_encoder_dropout: Optional[float] = Field(
        default=0.1, ge=0.0, lt=1.0
    )  # Edge encoder dropout

    @field_validator(MODEL_DEFS.ENCODER)
    @classmethod
    def validate_encoder(cls, v):
        if v not in VALID_ENCODERS:
            raise ValueError(
                f"Invalid encoder type: {v}. Valid types are: {VALID_ENCODERS}"
            )
        return v

    @field_validator(MODEL_DEFS.HEAD)
    @classmethod
    def validate_head(cls, v):
        if v not in VALID_HEADS:
            raise ValueError(f"Invalid head type: {v}. Valid types are: {VALID_HEADS}")
        return v

    @field_validator(MODEL_DEFS.HIDDEN_CHANNELS)
    @classmethod
    def validate_power_of_2(cls, v):
        """Optionally enforce power of 2 for efficiency"""
        if v & (v - 1) != 0:
            raise ValueError(f"hidden_channels should be power of 2, got {v}")
        return v

    model_config = ConfigDict(extra="forbid")  # Catch typos


class DataConfig(BaseModel):
    """Data loading and splitting configuration. These parameters are used to setup the NapistuDataStore object and construct the NapistuData object."""

    # config for defining the NapistuDataStore
    store_dir: Path = Field(default=DATA_CONFIG_DEFAULTS[DATA_CONFIG.STORE_DIR])
    sbml_dfs_path: Path = Field()
    napistu_graph_path: Path = Field()
    copy_to_store: bool = Field(default=False)
    overwrite: bool = Field(default=False)

    # named artifacts which are needed for the experiment
    napistu_data_name: str = Field(
        default=DATA_CONFIG_DEFAULTS[DATA_CONFIG.NAPISTU_DATA_NAME],
        description="Name of the NapistuData artifact to use for training.",
    )
    other_artifacts: List[str] = Field(
        default_factory=list,
        description="List of additional artifact names that must exist in the store.",
    )

    model_config = ConfigDict(extra="forbid")


class TaskConfig(BaseModel):
    """Task-specific configuration"""

    task: str = Field(default=TASK_CONFIG_DEFAULTS[TASK_CONFIG.TASK])
    metrics: List[str] = Field(default_factory=lambda: [METRICS.AUC, METRICS.AP])

    edge_prediction_neg_sampling_ratio: float = Field(default=1.0, gt=0.0)
    edge_prediction_neg_sampling_stratify_by: str = Field(
        default=TASK_CONFIG_DEFAULTS[
            TASK_CONFIG.EDGE_PREDICTION_NEG_SAMPLING_STRATIFY_BY
        ]
    )
    edge_prediction_neg_sampling_strategy: str = Field(
        default=TASK_CONFIG_DEFAULTS[TASK_CONFIG.EDGE_PREDICTION_NEG_SAMPLING_STRATEGY]
    )

    @field_validator(TASK_CONFIG.TASK)
    @classmethod
    def validate_task(cls, v):
        if v not in VALID_TASKS:
            raise ValueError(f"Invalid task: {v}. Valid tasks are: {VALID_TASKS}")
        return v

    model_config = ConfigDict(extra="forbid")


class TrainingConfig(BaseModel):
    """Training hyperparameters"""

    lr: float = Field(default=0.001, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    optimizer: str = Field(default=OPTIMIZERS.ADAM)
    scheduler: Optional[str] = None

    epochs: int = Field(default=200, gt=0)
    batches_per_epoch: int = Field(default=1, gt=0)

    # Training infrastructure
    accelerator: str = "auto"
    devices: int = 1
    precision: Literal[16, 32, "16-mixed", "32-true"] = 32

    # Callbacks
    early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_metric: str = "val_auc"

    checkpoint_subdir: str = Field(
        default=TRAINING_CONFIG_DEFAULTS[TRAINING_CONFIG.CHECKPOINT_SUBDIR],
        description="Subdirectory for checkpoints within output_dir",
    )
    save_checkpoints: bool = True
    checkpoint_metric: str = "val_auc"

    def get_checkpoint_dir(self, output_dir: Path) -> Path:
        """Get absolute checkpoint directory"""
        return output_dir / self.checkpoint_subdir

    @field_validator(TRAINING_CONFIG.OPTIMIZER)
    @classmethod
    def validate_optimizer(cls, v):
        if v not in VALID_OPTIMIZERS:
            raise ValueError(
                f"Invalid optimizer: {v}. Valid optimizers are: {VALID_OPTIMIZERS}"
            )
        return v

    @field_validator(TRAINING_CONFIG.SCHEDULER)
    @classmethod
    def validate_scheduler(cls, v):
        if v is not None and v not in VALID_SCHEDULERS:
            raise ValueError(
                f"Invalid scheduler: {v}. Valid schedulers are: {VALID_SCHEDULERS}"
            )
        return v

    model_config = ConfigDict(extra="forbid")


class WandBConfig(BaseModel):
    """Weights & Biases configuration"""

    project: str = WANDB_CONFIG_DEFAULTS[WANDB_CONFIG.PROJECT]
    entity: Optional[str] = WANDB_CONFIG_DEFAULTS[WANDB_CONFIG.ENTITY]
    group: Optional[str] = WANDB_CONFIG_DEFAULTS[WANDB_CONFIG.GROUP]
    tags: List[str] = Field(
        default_factory=lambda: WANDB_CONFIG_DEFAULTS[WANDB_CONFIG.TAGS]
    )
    log_model: bool = WANDB_CONFIG_DEFAULTS[WANDB_CONFIG.LOG_MODEL]
    mode: str = WANDB_CONFIG_DEFAULTS[WANDB_CONFIG.MODE]
    wandb_subdir: str = WANDB_CONFIG_DEFAULTS[WANDB_CONFIG.WANDB_SUBDIR]

    @field_validator(WANDB_CONFIG.MODE)
    @classmethod
    def validate_mode(cls, v):
        if v not in VALID_WANDB_MODES:
            raise ValueError(f"Invalid mode: {v}. Valid modes are: {VALID_WANDB_MODES}")
        return v

    def get_enhanced_tags(
        self, model_config: "ModelConfig", task_config: "TaskConfig"
    ) -> List[str]:
        """Get tags with model and task-specific additions"""
        enhanced_tags = self.tags.copy()
        enhanced_tags.extend(
            [
                model_config.encoder,
                task_config.task,
                f"hidden_{model_config.hidden_channels}",
                f"layers_{model_config.num_layers}",
            ]
        )
        return enhanced_tags

    def get_save_dir(self, output_dir: Path) -> Path:
        """Get absolute wandb save directory"""
        # note that wandb automatically creates a "wandb" subdirectory within the output_dir
        return output_dir / self.wandb_subdir

    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration"""

    # Experiment metadata
    name: Optional[str] = EXPERIMENT_CONFIG_DEFAULTS[EXPERIMENT_CONFIG.NAME]
    seed: int = EXPERIMENT_CONFIG_DEFAULTS[EXPERIMENT_CONFIG.SEED]
    deterministic: bool = True

    output_dir: Path = Field(
        default=Path("./output"),
        description="Base output directory for all run artifacts (checkpoints, logs, wandb)",
    )

    # Component configs
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    wandb: WandBConfig = Field(default_factory=WandBConfig)

    # Debug options
    fast_dev_run: bool = False
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0

    model_config = ConfigDict(extra="forbid")  # Catch config typos!

    # Convenience methods
    def to_dict(self):
        """Export to plain dict"""
        return self.model_dump()

    def to_json(self, filepath: Path):
        """Save to JSON"""
        filepath.write_text(self.model_dump_json(indent=2))

    @classmethod
    def from_json(cls, filepath: Path):
        """Load from JSON"""
        return cls.model_validate_json(filepath.read_text())

    def to_yaml(self, filepath: Path):
        """Save to YAML"""
        import yaml

        # Convert Path objects to strings for YAML serialization
        data = self.model_dump()

        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj

        data = convert_paths(data)
        with open(filepath, "w") as f:
            yaml.dump(data, f)

    @classmethod
    def from_yaml(cls, filepath: Path):
        """Load from YAML"""
        import yaml

        with open(filepath) as f:
            data = yaml.safe_load(f)

        # Get config file's directory for resolving relative paths
        config_dir = filepath.parent.resolve()

        # Convert string paths back to Path objects and resolve relative paths to absolute
        def convert_strings_to_paths(obj, key=None):
            if isinstance(obj, dict):
                return {k: convert_strings_to_paths(v, k) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_strings_to_paths(item) for item in obj]
            elif isinstance(obj, str) and key in [
                DATA_CONFIG.STORE_DIR,
                DATA_CONFIG.SBML_DFS_PATH,
                DATA_CONFIG.NAPISTU_GRAPH_PATH,
                EXPERIMENT_CONFIG.OUTPUT_DIR,
            ]:
                path = Path(obj)
                # Resolve relative paths to absolute paths relative to config file directory
                # These paths should always be resolved to absolute paths
                if not path.is_absolute():
                    return (config_dir / path).resolve()
                else:
                    return path.resolve()
            else:
                return obj

        # Apply path conversion
        data = convert_strings_to_paths(data)

        return cls(**data)

    def get_experiment_name(self) -> str:
        """Generate a descriptive experiment name based on model and task configs"""
        return f"{self.model.encoder}_h{self.model.hidden_channels}_l{self.model.num_layers}_{self.task.task}"


class RunManifest(BaseModel):
    """Manifest file containing all information about a training run."""

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.now, description="When this run was created"
    )

    # WandB information
    wandb_run_id: Optional[str] = Field(
        default=None, description="WandB run ID for this experiment"
    )
    wandb_run_url: Optional[str] = Field(
        default=None, description="Direct URL to the WandB run"
    )
    wandb_project: Optional[str] = Field(default=None, description="WandB project name")
    wandb_entity: Optional[str] = Field(
        default=None, description="WandB entity (username/team)"
    )

    # Experiment configuration
    experiment_name: Optional[str] = Field(
        default=None, description="Name of the experiment"
    )

    # Full experiment config (always an ExperimentConfig object)
    experiment_config: ExperimentConfig = Field(
        description="Complete experiment configuration"
    )

    def to_yaml(self, filepath: Path) -> None:
        """
        Save manifest to YAML file.

        Parameters
        ----------
        filepath : Path
            Path where the YAML file will be written
        """
        import yaml

        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Pydantic automatically serializes nested models
        # Use mode="json" to convert Path objects to strings
        data = self.model_dump(mode="json")

        # Write to YAML file
        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, filepath: Path) -> "RunManifest":
        """
        Load manifest from YAML file.

        Parameters
        ----------
        filepath : Path
            Path to the YAML file

        Returns
        -------
        RunManifest
            Loaded manifest object with experiment_config as ExperimentConfig instance
        """
        import yaml

        with open(filepath) as f:
            data = yaml.safe_load(f)

        # Pydantic automatically converts the dict to ExperimentConfig when creating the model
        return cls(**data)


# Public functions for working with configs


def config_to_data_trimming_spec(config: ExperimentConfig) -> Dict[str, bool]:
    """
    Based on the config, return a dictionary of booleans indicating whether each attribute should be kept.

    Parameters
    ----------
    config : ExperimentConfig
        The experiment configuration

    Returns
    -------
    Dict[str, bool]
        A dictionary with keys "keep_edge_attr", "keep_labels", "keep_masks" and values indicating whether each attribute should be kept. These match the arguments to NapistuData.trim().
    """

    # do we need edge attributes?
    if getattr(config.model, MODEL_CONFIG.USE_EDGE_ENCODER):
        edge_encoder = getattr(config.model, MODEL_CONFIG.ENCODER)
        if edge_encoder in ENCODERS_SUPPORTING_EDGE_WEIGHTING:
            keep_edge_attr = True
        else:
            logger.warning(
                f"Edge encoders are not supported by {edge_encoder}, only {ENCODERS_SUPPORTING_EDGE_WEIGHTING} support for edge-weighted message passing. Edge attributes will not be used."
            )
            keep_edge_attr = False
    else:
        keep_edge_attr = False

    # do we need labels?
    tasks = getattr(config.task, TASK_CONFIG.TASK)
    TASKS_WITH_LABELS = {TASKS.NODE_CLASSIFICATION}
    if tasks in TASKS_WITH_LABELS:
        keep_labels = True
    else:
        keep_labels = False

    # do we need masks
    TASKS_WITH_MASKS = {TASKS.EDGE_PREDICTION, TASKS.NODE_CLASSIFICATION}
    if tasks in TASKS_WITH_MASKS:
        keep_masks = True
    else:
        keep_masks = False

    return {
        "keep_edge_attr": keep_edge_attr,
        "keep_labels": keep_labels,
        "keep_masks": keep_masks,
    }


def create_template_yaml(
    output_path: Path,
    sbml_dfs_path: Optional[Path] = None,
    napistu_graph_path: Optional[Path] = None,
    name: Optional[str] = None,
) -> None:
    """
    Create a minimal YAML template file for experiment configuration.

    This creates a clean, minimal YAML file with only:
    - Required data paths (sbml_dfs_path, napistu_graph_path)
    - Experiment metadata (name)
    - Common configuration options (without default values)

    Users can then customize this template without all the default values cluttering the file.

    Parameters
    ----------
    output_path : Path
        Path where the YAML template file will be written
    sbml_dfs_path : Optional[Path], default=None
        Path to the SBML_dfs pickle file. If None, uses a placeholder.
    napistu_graph_path : Optional[Path], default=None
        Path to the NapistuGraph pickle file. If None, uses a placeholder.
    name : Optional[str], default=None
        Experiment name. If None, omits the name field.

    Examples
    --------
    >>> from pathlib import Path
    >>> from napistu_torch.configs import create_template_yaml
    >>>
    >>> # Create template with placeholder paths
    >>> create_template_yaml(
    ...     output_path=Path("config.yaml"),
    ...     sbml_dfs_path=Path("data/sbml_dfs.pkl"),
    ...     napistu_graph_path=Path("data/graph.pkl"),
    ...     name="my_experiment"
    ... )
    """
    import yaml

    # Build minimal template dict - only required fields and commonly customized ones
    template = {}

    template[EXPERIMENT_CONFIG.NAME] = (
        name if name else EXPERIMENT_CONFIG_DEFAULTS[EXPERIMENT_CONFIG.NAME]
    )
    template[EXPERIMENT_CONFIG.SEED] = EXPERIMENT_CONFIG_DEFAULTS[
        EXPERIMENT_CONFIG.SEED
    ]

    template[EXPERIMENT_CONFIG.MODEL] = {
        MODEL_CONFIG.ENCODER: MODEL_CONFIG_DEFAULTS[MODEL_CONFIG.ENCODER],
        MODEL_CONFIG.HEAD: MODEL_CONFIG_DEFAULTS[MODEL_CONFIG.HEAD],
        MODEL_CONFIG.USE_EDGE_ENCODER: MODEL_CONFIG_DEFAULTS[
            MODEL_CONFIG.USE_EDGE_ENCODER
        ],
    }

    template[EXPERIMENT_CONFIG.TASK] = {
        TASK_CONFIG.TASK: TASK_CONFIG_DEFAULTS[TASK_CONFIG.TASK],
    }

    template[EXPERIMENT_CONFIG.DATA] = {
        DATA_CONFIG.SBML_DFS_PATH: (
            str(sbml_dfs_path) if sbml_dfs_path else "path/to/sbml_dfs.pkl"
        ),
        DATA_CONFIG.NAPISTU_GRAPH_PATH: (
            str(napistu_graph_path)
            if napistu_graph_path
            else "path/to/napistu_graph.pkl"
        ),
        DATA_CONFIG.NAPISTU_DATA_NAME: DATA_CONFIG_DEFAULTS[
            DATA_CONFIG.NAPISTU_DATA_NAME
        ],
    }

    template[EXPERIMENT_CONFIG.WANDB] = {
        WANDB_CONFIG.GROUP: WANDB_CONFIG_DEFAULTS[WANDB_CONFIG.GROUP],
        WANDB_CONFIG.TAGS: WANDB_CONFIG_DEFAULTS[WANDB_CONFIG.TAGS],
    }
    # Include empty/minimal sections for training and wandb
    template[EXPERIMENT_CONFIG.TRAINING] = {}

    # Write YAML file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(template, f, default_flow_style=False, sort_keys=False)


def task_config_to_artifact_names(task_config: TaskConfig) -> List[str]:
    """
    Convert a TaskConfig to a list of artifact names required by the task.

    Parameters
    ----------
    task_config : TaskConfig
        Task configuration object

    Returns
    -------
    List[str]
        List of artifact names required by the task

    Examples
    --------
    >>> from napistu_torch.configs import TaskConfig, task_config_to_artifact_names
    >>> task_config = TaskConfig(
    ...     task="edge_prediction",
    ...     edge_prediction_neg_sampling_stratify_by="edge_strata_by_node_type"
    ... )
    >>> artifacts = task_config_to_artifact_names(task_config)
    >>> print(artifacts)
    ['edge_strata_by_node_type']
    """
    if task_config.task == TASKS.EDGE_PREDICTION:
        return _task_config_to_artifact_names_edge_prediction(task_config)
    else:
        return []


# Private functions for working with configs


def _task_config_to_artifact_names_edge_prediction(
    task_config: TaskConfig,
) -> List[str]:
    """Convert a TaskConfig to a list of artifact names for edge prediction."""
    if task_config.edge_prediction_neg_sampling_stratify_by == "none":
        return []
    else:
        # validate the value and return the artifact name
        return [
            ensure_stratify_by_artifact_name(
                task_config.edge_prediction_neg_sampling_stratify_by
            )
        ]
