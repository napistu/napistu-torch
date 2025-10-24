from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from napistu_torch.constants import (
    DATA_CONFIG,
    METRICS,
    OPTIMIZERS,
    TASK_CONFIG,
    TASKS,
    TRAINING_CONFIG,
    VALID_OPTIMIZERS,
    VALID_SCHEDULERS,
    VALID_TASKS,
    VALID_WANDB_MODES,
    WANDB_CONFIG,
)
from napistu_torch.load.constants import (
    SPLITTING_STRATEGIES,
    VALID_SPLITTING_STRATEGIES,
)
from napistu_torch.models.constants import (
    ENCODERS,
    ENCODER_DEFS,
    HEADS,
    MODEL_DEFS,
    VALID_ENCODERS,
    VALID_HEADS,
)


class ModelConfig(BaseModel):
    """Model architecture configuration"""

    encoder: str = Field(default=ENCODERS.SAGE)
    hidden_channels: int = Field(default=128, gt=0)
    num_layers: int = Field(default=3, ge=1, le=10)
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    head: str = Field(default=HEADS.DOT_PRODUCT)

    # Model-specific fields (optional, with defaults)
    sage_aggregator: Optional[str] = ENCODER_DEFS.SAGE_DEFAULT_AGGREGATOR  # For SAGE
    gat_heads: Optional[int] = Field(default=4, gt=0)  # For GAT
    gat_concat: Optional[bool] = True  # For GAT

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

    class Config:
        extra = "forbid"  # Catch typos


class DataConfig(BaseModel):
    """Data loading and splitting configuration. These parameters are used to setup the NapistuDataStore object and construct the NapistuData object."""

    name: str = "default"
    store_dir: Path = Field(default=Path(".store"))
    splitting_strategy: str = Field(default=SPLITTING_STRATEGIES.NO_MASK)
    train_size: float = Field(default=0.7, gt=0.0, lt=1.0)
    val_size: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_size: float = Field(default=0.15, gt=0.0, lt=1.0)

    @field_validator(DATA_CONFIG.SPLITTING_STRATEGY)
    @classmethod
    def validate_splitting_strategy(cls, v):
        if v not in VALID_SPLITTING_STRATEGIES:
            raise ValueError(
                f"Invalid splitting strategy: {v}. Valid strategies are: {VALID_SPLITTING_STRATEGIES}"
            )
        return v

    class Config:
        extra = "forbid"


class TaskConfig(BaseModel):
    """Task-specific configuration"""

    task: str = Field(default=TASKS.EDGE_PREDICTION)
    neg_sampling_ratio: float = Field(default=1.0, gt=0.0)
    metrics: List[str] = Field(default_factory=lambda: [METRICS.AUC, METRICS.AP])

    @field_validator(TASK_CONFIG.TASK)
    @classmethod
    def validate_task(cls, v):
        if v not in VALID_TASKS:
            raise ValueError(f"Invalid task: {v}. Valid tasks are: {VALID_TASKS}")
        return v

    class Config:
        extra = "forbid"


class TrainingConfig(BaseModel):
    """Training hyperparameters"""

    lr: float = Field(default=0.001, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    optimizer: str = Field(default=OPTIMIZERS.ADAM)
    scheduler: Optional[str] = None

    epochs: int = Field(default=200, gt=0)
    batch_size: int = Field(default=32, gt=0)

    # Training infrastructure
    accelerator: str = "auto"
    devices: int = 1
    precision: Literal[16, 32, "16-mixed", "32-true"] = 32

    # Callbacks
    early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_metric: str = "val_auc"

    checkpoint_dir: Path = Field(default=Path("checkpoints"))
    save_checkpoints: bool = True
    checkpoint_metric: str = "val_auc"

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

    class Config:
        extra = "forbid"


class WandBConfig(BaseModel):
    """Weights & Biases configuration"""

    project: str = "napistu-experiments"
    entity: Optional[str] = None
    group: Optional[str] = "baseline"
    tags: List[str] = Field(default_factory=lambda: ["gnn", "pytorch-lightning"])
    save_dir: Path = Field(default=Path("./wandb"))
    log_model: bool = False
    mode: str = Field(default="online")

    @field_validator(WANDB_CONFIG.MODE)
    @classmethod
    def validate_mode(cls, v):
        if v not in VALID_WANDB_MODES:
            raise ValueError(f"Invalid mode: {v}. Valid modes are: {VALID_WANDB_MODES}")
        return v

    def get_run_name(
        self, model_config: "ModelConfig", task_config: "TaskConfig"
    ) -> str:
        """Generate a descriptive run name based on model and task configs"""
        return f"{model_config.encoder}_h{model_config.hidden_channels}_l{model_config.num_layers}_{task_config.task}"

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

    class Config:
        extra = "forbid"


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration"""

    # Experiment metadata
    name: Optional[str] = None
    seed: int = 42
    deterministic: bool = True

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

    class Config:
        extra = "forbid"  # Catch config typos!

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

        # Convert string paths back to Path objects
        def convert_strings_to_paths(obj, key=None):
            if isinstance(obj, dict):
                return {k: convert_strings_to_paths(v, k) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_strings_to_paths(item) for item in obj]
            elif isinstance(obj, str) and key in [
                "store_dir",
                "checkpoint_dir",
                "save_dir",
            ]:
                return Path(obj)
            else:
                return obj

        # Apply path conversion
        data = convert_strings_to_paths(data)

        return cls(**data)
