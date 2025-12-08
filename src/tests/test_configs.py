"""Tests for napistu_torch configs module."""

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from napistu_torch.configs import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    RunManifest,
    TaskConfig,
    TrainingConfig,
    WandBConfig,
    config_to_data_trimming_spec,
    create_template_yaml,
    task_config_to_artifact_names,
)
from napistu_torch.constants import (
    ANONYMIZATION_PLACEHOLDER_DEFAULT,
    DATA_CONFIG,
    EXPERIMENT_CONFIG,
    NAPISTU_DATA_TRIM_ARGS,
    OPTIMIZERS,
    PRETRAINED_COMPONENT_SOURCES,
    SCHEDULERS,
    TASK_CONFIG,
    TRAINING_CONFIG,
    VALID_OPTIMIZERS,
    VALID_SCHEDULERS,
    VALID_WANDB_MODES,
    WANDB_CONFIG,
    WANDB_CONFIG_DEFAULTS,
    WANDB_MODES,
)
from napistu_torch.lightning.constants import EXPERIMENT_DICT
from napistu_torch.load.constants import (
    DEFAULT_ARTIFACTS_NAMES,
    STRATIFY_BY,
    STRATIFY_BY_ARTIFACT_NAMES,
)
from napistu_torch.ml.constants import METRICS
from napistu_torch.models.constants import (
    ENCODER_SPECIFIC_ARGS,
    ENCODERS,
    HEAD_SPECIFIC_ARGS,
    HEADS,
    MODEL_DEFS,
    VALID_ENCODERS,
    VALID_HEADS,
)
from napistu_torch.tasks.constants import (
    NEGATIVE_SAMPLING_STRATEGIES,
    TASKS,
    VALID_TASKS,
)


@pytest.fixture
def stubbed_data_config():
    """Create a stubbed DataConfig for testing."""
    return DataConfig(
        sbml_dfs_path=Path("stub_sbml.pkl"), napistu_graph_path=Path("stub_graph.pkl")
    )


class TestModelConfig:
    """Test ModelConfig class."""

    def test_encoder_validation(self):
        """Test encoder validation with valid and invalid values."""
        # Test valid encoder types
        for encoder in VALID_ENCODERS:
            config = ModelConfig(encoder=encoder)
            assert hasattr(config, MODEL_DEFS.ENCODER)
            assert config.encoder == encoder

        # Test invalid encoder type
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(encoder="invalid_encoder")
        assert "Invalid encoder" in str(exc_info.value)

    def test_head_validation(self):
        """Test head validation with valid and invalid values."""
        # Test valid head types
        for head in VALID_HEADS:
            config = ModelConfig(head=head)
            assert hasattr(config, MODEL_DEFS.HEAD)
            assert config.head == head

        # Test invalid head type
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(head="invalid_head")
        assert "Invalid head type" in str(exc_info.value)

    def test_hidden_channels_validation(self):
        """Test hidden_channels validation with valid and invalid values."""
        # Test valid power of 2 values
        valid_values = [64, 128, 256, 512]
        for value in valid_values:
            config = ModelConfig(hidden_channels=value)
            assert hasattr(config, MODEL_DEFS.HIDDEN_CHANNELS)
            assert config.hidden_channels == value

        # Test invalid values (not power of 2)
        invalid_values = [100, 150, 200, 300]
        for value in invalid_values:
            with pytest.raises(ValidationError) as exc_info:
                ModelConfig(hidden_channels=value)
            assert "power of 2" in str(exc_info.value)

    def test_num_layers_validation(self):
        """Test num_layers validation with valid and invalid values."""
        # Test valid range
        valid_values = [1, 3, 5, 10]
        for value in valid_values:
            config = ModelConfig(num_layers=value)
            assert hasattr(config, MODEL_DEFS.NUM_LAYERS)
            assert config.num_layers == value

        # Test invalid values (out of range)
        with pytest.raises(ValidationError):
            ModelConfig(num_layers=0)  # Below minimum

        with pytest.raises(ValidationError):
            ModelConfig(num_layers=11)  # Above maximum

    def test_dropout_validation(self):
        """Test dropout validation with valid and invalid values."""
        # Test valid range
        valid_values = [0.0, 0.2, 0.5, 0.99]
        for value in valid_values:
            config = ModelConfig(dropout=value)
            assert hasattr(config, MODEL_DEFS.DROPOUT)
            assert config.dropout == value

        # Test invalid values
        with pytest.raises(ValidationError):
            ModelConfig(dropout=-0.1)  # Below minimum

        with pytest.raises(ValidationError):
            ModelConfig(dropout=1.0)  # At maximum (should be < 1.0)

    def test_optional_fields_exist(self):
        """Test that optional fields exist and can be set."""
        config = ModelConfig()

        # Test that optional fields exist
        assert hasattr(config, ENCODER_SPECIFIC_ARGS.SAGE_AGGREGATOR)
        assert hasattr(config, ENCODER_SPECIFIC_ARGS.GAT_HEADS)

        # Test that they can be customized
        config = ModelConfig(sage_aggregator="max", gat_heads=8)
        assert config.sage_aggregator == "max"
        assert config.gat_heads == 8

    def test_head_specific_fields_exist(self):
        """Test that head-specific fields exist and can be set."""
        config = ModelConfig()

        # Test that head-specific fields exist
        assert hasattr(config, HEAD_SPECIFIC_ARGS.MLP_HIDDEN_DIM)
        assert hasattr(config, HEAD_SPECIFIC_ARGS.MLP_NUM_LAYERS)
        assert hasattr(config, HEAD_SPECIFIC_ARGS.MLP_DROPOUT)
        assert hasattr(config, HEAD_SPECIFIC_ARGS.BILINEAR_BIAS)
        assert hasattr(config, HEAD_SPECIFIC_ARGS.NC_DROPOUT)
        assert hasattr(config, HEAD_SPECIFIC_ARGS.ROTATE_MARGIN)
        assert hasattr(config, HEAD_SPECIFIC_ARGS.TRANSE_MARGIN)

        # Test that they can be customized
        config = ModelConfig(
            mlp_hidden_dim=128,
            mlp_num_layers=3,
            mlp_dropout=0.2,
            bilinear_bias=False,
            nc_dropout=0.15,
            rotate_margin=12.0,
            transe_margin=2.0,
        )
        assert config.mlp_hidden_dim == 128
        assert config.mlp_num_layers == 3
        assert config.mlp_dropout == 0.2
        assert config.bilinear_bias is False
        assert config.nc_dropout == 0.15
        assert config.rotate_margin == 12.0
        assert config.transe_margin == 2.0

    def test_mlp_parameters_validation(self):
        """Test MLP head parameter validation."""
        # Test valid mlp_hidden_dim
        config = ModelConfig(mlp_hidden_dim=64)
        assert config.mlp_hidden_dim == 64

        # Test valid mlp_num_layers
        valid_layers = [1, 2, 3, 5]
        for num_layers in valid_layers:
            config = ModelConfig(mlp_num_layers=num_layers)
            assert config.mlp_num_layers == num_layers

        # Test invalid mlp_num_layers (below minimum)
        with pytest.raises(ValidationError):
            ModelConfig(mlp_num_layers=0)

        # Test valid mlp_dropout
        valid_dropouts = [0.0, 0.1, 0.5, 0.99]
        for dropout in valid_dropouts:
            config = ModelConfig(mlp_dropout=dropout)
            assert config.mlp_dropout == dropout

        # Test invalid mlp_dropout
        with pytest.raises(ValidationError):
            ModelConfig(mlp_dropout=-0.1)  # Below minimum

        with pytest.raises(ValidationError):
            ModelConfig(mlp_dropout=1.0)  # At maximum (should be < 1.0)

    def test_node_classification_parameters_validation(self):
        """Test node classification head parameter validation."""
        # Test valid nc_dropout
        valid_dropouts = [0.0, 0.1, 0.5, 0.99]
        for dropout in valid_dropouts:
            config = ModelConfig(nc_dropout=dropout)
            assert config.nc_dropout == dropout

        # Test invalid nc_dropout
        with pytest.raises(ValidationError):
            ModelConfig(nc_dropout=-0.1)  # Below minimum

        with pytest.raises(ValidationError):
            ModelConfig(nc_dropout=1.0)  # At maximum (should be < 1.0)

    def test_relation_aware_head_margins_validation(self):
        """Test relation-aware head margin parameter validation."""
        # Test valid rotate_margin
        valid_margins = [1.0, 5.0, 9.0, 15.0]
        for margin in valid_margins:
            config = ModelConfig(rotate_margin=margin)
            assert config.rotate_margin == margin

        # Test invalid rotate_margin
        with pytest.raises(ValidationError):
            ModelConfig(rotate_margin=0.0)  # At minimum (should be > 0.0)

        with pytest.raises(ValidationError):
            ModelConfig(rotate_margin=-1.0)  # Negative value

        # Test valid transe_margin
        valid_margins = [0.5, 1.0, 2.0, 5.0]
        for margin in valid_margins:
            config = ModelConfig(transe_margin=margin)
            assert config.transe_margin == margin

        # Test invalid transe_margin
        with pytest.raises(ValidationError):
            ModelConfig(transe_margin=0.0)  # At minimum (should be > 0.0)

        with pytest.raises(ValidationError):
            ModelConfig(transe_margin=-1.0)  # Negative value

    def test_get_architecture_string(self):
        """Test get_architecture_string method."""
        # Test with encoder and head
        config = ModelConfig(
            encoder=ENCODERS.SAGE,
            head=HEADS.DOT_PRODUCT,
            hidden_channels=128,
            num_layers=3,
        )
        arch_str = config.get_architecture_string()
        assert arch_str == "sage-dot_product_h128_l3"

        # Test with different encoder and head combinations
        config = ModelConfig(
            encoder=ENCODERS.GRAPH_CONV,
            head=HEADS.MLP,
            hidden_channels=64,
            num_layers=2,
        )
        arch_str = config.get_architecture_string()
        assert arch_str == "graph_conv-mlp_h64_l2"

        # Test with different encoder and head combinations
        config = ModelConfig(
            encoder=ENCODERS.GAT, head=HEADS.BILINEAR, hidden_channels=256, num_layers=5
        )
        arch_str = config.get_architecture_string()
        assert arch_str == "gat-bilinear_h256_l5"

        # Test with default values (should use defaults for hidden_channels and num_layers)
        config = ModelConfig(encoder=ENCODERS.SAGE, head=HEADS.MLP)
        arch_str = config.get_architecture_string()
        # Defaults are hidden_channels=128, num_layers=3
        assert arch_str == "sage-mlp_h128_l3"

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(invalid_field="value")
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_pretrained_model_validation(self):
        """Test validation for use_pretrained_model settings."""
        # Test that use_pretrained_model=False (default) doesn't require fields
        config = ModelConfig(use_pretrained_model=False)
        assert config.use_pretrained_model is False
        assert config.pretrained_model_source is None
        assert config.pretrained_model_path is None

        # Test that use_pretrained_model=True requires source and path
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(use_pretrained_model=True)
        assert "pretrained_model_source must be specified" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                use_pretrained_model=True,
                pretrained_model_source=PRETRAINED_COMPONENT_SOURCES.HUGGINGFACE,
            )
        assert "pretrained_model_path must be specified" in str(exc_info.value)

        # Test valid configuration with all required fields
        config = ModelConfig(
            use_pretrained_model=True,
            pretrained_model_source=PRETRAINED_COMPONENT_SOURCES.HUGGINGFACE,
            pretrained_model_path="org/model-name",
        )
        assert config.use_pretrained_model is True
        assert (
            config.pretrained_model_source == PRETRAINED_COMPONENT_SOURCES.HUGGINGFACE
        )
        assert config.pretrained_model_path == "org/model-name"

        # Test with local source
        config = ModelConfig(
            use_pretrained_model=True,
            pretrained_model_source=PRETRAINED_COMPONENT_SOURCES.LOCAL,
            pretrained_model_path="/path/to/model",
        )
        assert config.pretrained_model_source == PRETRAINED_COMPONENT_SOURCES.LOCAL

        # Test invalid source
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                use_pretrained_model=True,
                pretrained_model_source="invalid_source",
                pretrained_model_path="some/path",
            )
        assert "Invalid pretrained_model_source" in str(exc_info.value)

        # Test optional fields can be set
        config = ModelConfig(
            use_pretrained_model=True,
            pretrained_model_source=PRETRAINED_COMPONENT_SOURCES.HUGGINGFACE,
            pretrained_model_path="org/model-name",
            pretrained_model_revision="main",
            pretrained_model_load_head=False,
            pretrained_model_freeze_encoder_weights=True,
        )
        assert config.pretrained_model_revision == "main"
        assert config.pretrained_model_load_head is False
        assert config.pretrained_model_freeze_encoder_weights is True
        assert config.pretrained_model_freeze_head_weights is False


class TestDataConfig:
    """Test DataConfig class."""

    def test_required_fields_exist(self):
        """Test that all required fields exist and can be set."""
        # Test that required fields exist in the model
        model_fields = DataConfig.model_fields
        assert DATA_CONFIG.STORE_DIR in model_fields
        assert DATA_CONFIG.SBML_DFS_PATH in model_fields
        assert DATA_CONFIG.NAPISTU_GRAPH_PATH in model_fields
        assert DATA_CONFIG.COPY_TO_STORE in model_fields
        assert DATA_CONFIG.OVERWRITE in model_fields
        assert DATA_CONFIG.NAPISTU_DATA_NAME in model_fields
        assert DATA_CONFIG.OTHER_ARTIFACTS in model_fields

    def test_default_values(self):
        """Test that default values are set correctly."""
        # Create config with only required fields
        config = DataConfig(
            sbml_dfs_path=Path("test_sbml.pkl"),
            napistu_graph_path=Path("test_graph.pkl"),
        )

        assert config.store_dir == Path(".store")
        assert config.copy_to_store is False
        assert config.overwrite is False
        assert config.napistu_data_name == DEFAULT_ARTIFACTS_NAMES.EDGE_PREDICTION
        assert config.other_artifacts == []

    def test_custom_values(self):
        """Test that fields can be customized."""
        custom_path = Path("/custom/path")
        config = DataConfig(
            store_dir=custom_path,
            sbml_dfs_path=Path("custom_sbml.pkl"),
            napistu_graph_path=Path("custom_graph.pkl"),
            copy_to_store=True,
            overwrite=True,
            napistu_data_name=DEFAULT_ARTIFACTS_NAMES.UNLABELED,
            other_artifacts=[
                DEFAULT_ARTIFACTS_NAMES.EDGE_PREDICTION,
                DEFAULT_ARTIFACTS_NAMES.EDGE_STRATA_BY_NODE_SPECIES_TYPE,
            ],
        )

        assert config.store_dir == custom_path
        assert config.sbml_dfs_path == Path("custom_sbml.pkl")
        assert config.napistu_graph_path == Path("custom_graph.pkl")
        assert config.copy_to_store is True
        assert config.overwrite is True
        assert config.napistu_data_name == DEFAULT_ARTIFACTS_NAMES.UNLABELED
        assert config.other_artifacts == [
            DEFAULT_ARTIFACTS_NAMES.EDGE_PREDICTION,
            DEFAULT_ARTIFACTS_NAMES.EDGE_STRATA_BY_NODE_SPECIES_TYPE,
        ]

    def test_path_objects(self):
        """Test that Path objects are properly handled."""
        config = DataConfig(
            sbml_dfs_path=Path("test_sbml.pkl"),
            napistu_graph_path=Path("test_graph.pkl"),
        )

        assert isinstance(config.store_dir, Path)
        assert isinstance(config.sbml_dfs_path, Path)
        assert isinstance(config.napistu_graph_path, Path)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            DataConfig(
                sbml_dfs_path=Path("test_sbml.pkl"),
                napistu_graph_path=Path("test_graph.pkl"),
                invalid_field="value",
            )
        assert "Extra inputs are not permitted" in str(exc_info.value)


class TestTaskConfig:
    """Test TaskConfig class."""

    def test_task_validation(self):
        """Test task validation with valid and invalid values."""
        # Test valid task types
        for task in VALID_TASKS:
            config = TaskConfig(task=task)
            assert hasattr(config, TASK_CONFIG.TASK)
            assert config.task == task

        # Test invalid task type
        with pytest.raises(ValidationError) as exc_info:
            TaskConfig(task="invalid_task")
        assert "Invalid task" in str(exc_info.value)

    def test_neg_sampling_ratio_validation(self):
        """Test edge_prediction_neg_sampling_ratio validation with valid and invalid values."""
        # Test valid values
        valid_ratios = [0.1, 1.0, 2.0, 5.0, 10.0]
        for ratio in valid_ratios:
            config = TaskConfig(edge_prediction_neg_sampling_ratio=ratio)
            assert hasattr(config, TASK_CONFIG.EDGE_PREDICTION_NEG_SAMPLING_RATIO)
            assert config.edge_prediction_neg_sampling_ratio == ratio

        # Test invalid values
        with pytest.raises(ValidationError):
            TaskConfig(
                edge_prediction_neg_sampling_ratio=0.0
            )  # At minimum (should be > 0.0)

        with pytest.raises(ValidationError):
            TaskConfig(edge_prediction_neg_sampling_ratio=-1.0)  # Negative value

    def test_metrics_validation(self):
        """Test metrics validation with valid and invalid values."""
        # Test valid metrics lists
        valid_metrics_lists = [
            [METRICS.AUC],
            [METRICS.AP],
            [METRICS.AUC, METRICS.AP],
            [],
        ]
        for metrics in valid_metrics_lists:
            config = TaskConfig(metrics=metrics)
            assert hasattr(config, "metrics")
            assert config.metrics == metrics

    def test_required_fields_exist(self):
        """Test that all required fields exist and can be set."""
        config = TaskConfig()

        # Test that all fields exist
        assert hasattr(config, TASK_CONFIG.TASK)
        assert hasattr(config, TASK_CONFIG.EDGE_PREDICTION_NEG_SAMPLING_RATIO)
        assert hasattr(config, TASK_CONFIG.EDGE_PREDICTION_NEG_SAMPLING_STRATIFY_BY)
        assert hasattr(config, TASK_CONFIG.EDGE_PREDICTION_NEG_SAMPLING_STRATEGY)
        assert hasattr(config, TASK_CONFIG.METRICS)

        # Test that they can be customized
        config = TaskConfig(
            task=TASKS.NODE_CLASSIFICATION,
            edge_prediction_neg_sampling_ratio=3.0,
            edge_prediction_neg_sampling_stratify_by=STRATIFY_BY.NODE_TYPE,
            edge_prediction_neg_sampling_strategy=NEGATIVE_SAMPLING_STRATEGIES.UNIFORM,
            metrics=[METRICS.AUC],
        )
        assert config.task == TASKS.NODE_CLASSIFICATION
        assert config.edge_prediction_neg_sampling_ratio == 3.0
        assert config.edge_prediction_neg_sampling_stratify_by == STRATIFY_BY.NODE_TYPE
        assert (
            config.edge_prediction_neg_sampling_strategy
            == NEGATIVE_SAMPLING_STRATEGIES.UNIFORM
        )
        assert config.metrics == [METRICS.AUC]

    def test_task_config_to_artifact_names(self):
        """Test task_config_to_artifact_names function."""
        # Test edge_prediction with "none" stratify_by -> returns empty list
        task_config = TaskConfig(
            task=TASKS.EDGE_PREDICTION,
            edge_prediction_neg_sampling_stratify_by="none",
        )
        artifacts = task_config_to_artifact_names(task_config)
        assert artifacts == []

        # Test edge_prediction with valid artifact name -> returns list with artifact
        for artifact_name in STRATIFY_BY_ARTIFACT_NAMES:
            task_config = TaskConfig(
                task=TASKS.EDGE_PREDICTION,
                edge_prediction_neg_sampling_stratify_by=artifact_name,
            )
            artifacts = task_config_to_artifact_names(task_config)
            assert artifacts == [artifact_name]

        # Test non-edge_prediction task -> returns empty list
        task_config = TaskConfig(task=TASKS.NODE_CLASSIFICATION)
        artifacts = task_config_to_artifact_names(task_config)
        assert artifacts == []


class TestTrainingConfig:
    """Test TrainingConfig class."""

    def test_optimizer_validation(self):
        """Test optimizer validation with valid and invalid values."""
        # Test valid optimizers
        for optimizer in VALID_OPTIMIZERS:
            config = TrainingConfig(optimizer=optimizer)
            assert hasattr(config, TRAINING_CONFIG.OPTIMIZER)
            assert config.optimizer == optimizer

        # Test invalid optimizer
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(optimizer="invalid_optimizer")
        assert "Invalid optimizer" in str(exc_info.value)

    def test_scheduler_validation(self):
        """Test scheduler validation with valid and invalid values."""
        # Test valid schedulers
        for scheduler in VALID_SCHEDULERS:
            config = TrainingConfig(scheduler=scheduler)
            assert hasattr(config, TRAINING_CONFIG.SCHEDULER)
            assert config.scheduler == scheduler

        # Test None scheduler
        config = TrainingConfig(scheduler=None)
        assert hasattr(config, TRAINING_CONFIG.SCHEDULER)
        assert config.scheduler is None

        # Test invalid scheduler
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(scheduler="invalid_scheduler")
        assert "Invalid scheduler" in str(exc_info.value)

    def test_lr_validation(self):
        """Test learning rate validation with valid and invalid values."""
        # Test valid values
        valid_lrs = [0.0001, 0.001, 0.01, 0.1]
        for lr in valid_lrs:
            config = TrainingConfig(lr=lr)
            assert hasattr(config, TRAINING_CONFIG.LR)
            assert config.lr == lr

        # Test invalid values
        with pytest.raises(ValidationError):
            TrainingConfig(lr=0.0)  # At minimum (should be > 0.0)

        with pytest.raises(ValidationError):
            TrainingConfig(lr=-0.1)  # Negative value

    def test_weight_decay_validation(self):
        """Test weight decay validation with valid and invalid values."""
        # Test valid values
        valid_decays = [0.0, 0.0001, 0.001, 0.01, 0.1]
        for decay in valid_decays:
            config = TrainingConfig(weight_decay=decay)
            assert hasattr(config, TRAINING_CONFIG.WEIGHT_DECAY)
            assert config.weight_decay == decay

        # Test invalid values
        with pytest.raises(ValidationError):
            TrainingConfig(weight_decay=-0.1)  # Negative value

    def test_epochs_validation(self):
        """Test epochs validation with valid and invalid values."""
        # Test valid values
        valid_epochs = [1, 10, 100, 1000]
        for epochs in valid_epochs:
            config = TrainingConfig(epochs=epochs)
            assert hasattr(config, TRAINING_CONFIG.EPOCHS)
            assert config.epochs == epochs

        # Test invalid values
        with pytest.raises(ValidationError):
            TrainingConfig(epochs=0)  # At minimum (should be > 0)

        with pytest.raises(ValidationError):
            TrainingConfig(epochs=-1)  # Negative value

    def test_batches_per_epoch_validation(self):
        """Test batches per epoch validation with valid and invalid values."""
        # Test valid values
        valid_batches_per_epoch = [1, 16, 32, 64, 128]
        for batches_per_epoch in valid_batches_per_epoch:
            config = TrainingConfig(batches_per_epoch=batches_per_epoch)
            assert hasattr(config, TRAINING_CONFIG.BATCHES_PER_EPOCH)
            assert config.batches_per_epoch == batches_per_epoch

        # Test invalid values
        with pytest.raises(ValidationError):
            TrainingConfig(batches_per_epoch=0)  # At minimum (should be > 0)

        with pytest.raises(ValidationError):
            TrainingConfig(batches_per_epoch=-1)  # Negative value

    def test_precision_validation(self):
        """Test precision validation with valid and invalid values."""
        # Test valid precisions
        valid_precisions = [16, 32, "16-mixed", "32-true"]
        for precision in valid_precisions:
            config = TrainingConfig(precision=precision)
            assert hasattr(config, TRAINING_CONFIG.PRECISION)
            assert config.precision == precision

    def test_required_fields_exist(self):
        """Test that all required fields exist and can be set."""
        config = TrainingConfig()

        # Test that all fields exist
        assert hasattr(config, TRAINING_CONFIG.LR)
        assert hasattr(config, TRAINING_CONFIG.WEIGHT_DECAY)
        assert hasattr(config, TRAINING_CONFIG.OPTIMIZER)
        assert hasattr(config, TRAINING_CONFIG.SCHEDULER)
        assert hasattr(config, TRAINING_CONFIG.EPOCHS)
        assert hasattr(config, TRAINING_CONFIG.BATCHES_PER_EPOCH)
        assert hasattr(config, TRAINING_CONFIG.ACCELERATOR)
        assert hasattr(config, TRAINING_CONFIG.DEVICES)
        assert hasattr(config, TRAINING_CONFIG.PRECISION)
        assert hasattr(config, TRAINING_CONFIG.EARLY_STOPPING)
        assert hasattr(config, TRAINING_CONFIG.EARLY_STOPPING_PATIENCE)
        assert hasattr(config, TRAINING_CONFIG.EARLY_STOPPING_METRIC)
        assert hasattr(config, TRAINING_CONFIG.CHECKPOINT_SUBDIR)
        assert hasattr(config, TRAINING_CONFIG.SAVE_CHECKPOINTS)
        assert hasattr(config, TRAINING_CONFIG.CHECKPOINT_METRIC)

        # Test that they can be customized
        config = TrainingConfig(
            lr=0.01,
            weight_decay=0.001,
            optimizer=OPTIMIZERS.ADAMW,
            scheduler=SCHEDULERS.PLATEAU,
            epochs=100,
            batches_per_epoch=64,
            checkpoint_subdir="custom_checkpoints",
        )
        assert config.lr == 0.01
        assert config.weight_decay == 0.001
        assert config.optimizer == OPTIMIZERS.ADAMW
        assert config.scheduler == SCHEDULERS.PLATEAU
        assert config.epochs == 100
        assert config.batches_per_epoch == 64
        assert config.checkpoint_subdir == "custom_checkpoints"

    def test_get_checkpoint_dir(self):
        """Test get_checkpoint_dir method."""
        config = TrainingConfig(checkpoint_subdir="my_checkpoints")
        output_dir = Path("/output")
        checkpoint_dir = config.get_checkpoint_dir(output_dir)
        assert checkpoint_dir == Path("/output/my_checkpoints")
        assert isinstance(checkpoint_dir, Path)

        # Test with default subdir
        config = TrainingConfig()
        checkpoint_dir = config.get_checkpoint_dir(output_dir)
        assert checkpoint_dir == Path("/output/checkpoints")


class TestWandBConfig:
    """Test WandBConfig class."""

    def test_mode_validation(self):
        """Test mode validation with valid and invalid values."""
        # Test valid modes
        for mode in VALID_WANDB_MODES:
            config = WandBConfig(mode=mode)
            assert hasattr(config, WANDB_CONFIG.MODE)
            assert config.mode == mode

        # Test invalid mode
        with pytest.raises(ValidationError) as exc_info:
            WandBConfig(mode="invalid_mode")
        assert "Invalid mode" in str(exc_info.value)

    def test_required_fields_exist(self):
        """Test that all required fields exist and can be set."""
        config = WandBConfig()

        # Test that all fields exist
        assert hasattr(config, WANDB_CONFIG.PROJECT)
        assert hasattr(config, WANDB_CONFIG.ENTITY)
        assert hasattr(config, WANDB_CONFIG.GROUP)
        assert hasattr(config, WANDB_CONFIG.TAGS)
        assert hasattr(config, WANDB_CONFIG.LOG_MODEL)
        assert hasattr(config, WANDB_CONFIG.MODE)
        assert hasattr(config, WANDB_CONFIG.WANDB_SUBDIR)

        # Test that they can be customized
        config = WandBConfig(
            project="my-project",
            entity="my-entity",
            group="my-group",
            tags=["tag1", "tag2"],
            log_model=True,
            mode=WANDB_MODES.OFFLINE,
            wandb_subdir="my_wandb",
        )
        assert config.project == "my-project"
        assert config.entity == "my-entity"
        assert config.group == "my-group"
        assert config.tags == ["tag1", "tag2"]
        assert config.log_model is True
        assert config.mode == WANDB_MODES.OFFLINE
        assert config.wandb_subdir == "my_wandb"

    def test_get_save_dir(self):
        """Test get_save_dir method."""
        config = WandBConfig(wandb_subdir="my_wandb")
        output_dir = Path("/output")
        save_dir = config.get_save_dir(output_dir)
        assert save_dir == Path("/output/my_wandb")
        assert isinstance(save_dir, Path)

        # Test with default subdir (from constants)
        config = WandBConfig()
        save_dir = config.get_save_dir(output_dir)
        expected_default = output_dir / WANDB_CONFIG_DEFAULTS[WANDB_CONFIG.WANDB_SUBDIR]
        assert save_dir == expected_default
        assert isinstance(save_dir, Path)

    def test_get_enhanced_tags(self):
        """Test get_enhanced_tags method."""
        config = WandBConfig(tags=["base_tag"])
        model_config = ModelConfig(
            encoder=ENCODERS.GAT, hidden_channels=256, num_layers=3
        )
        task_config = TaskConfig(task=TASKS.EDGE_PREDICTION)

        enhanced_tags = config.get_enhanced_tags(model_config, task_config)
        assert "base_tag" in enhanced_tags
        assert ENCODERS.GAT in enhanced_tags
        assert TASKS.EDGE_PREDICTION in enhanced_tags
        assert "hidden_256" in enhanced_tags
        assert "layers_3" in enhanced_tags


class TestExperimentConfig:
    """Test ExperimentConfig class."""

    def test_required_fields_exist(self, stubbed_data_config):
        """Test that all required fields exist and can be set."""
        # Create config with required DataConfig fields
        config = ExperimentConfig(data=stubbed_data_config)

        # Test that all fields exist
        assert hasattr(config, EXPERIMENT_CONFIG.NAME)
        assert hasattr(config, EXPERIMENT_CONFIG.SEED)
        assert hasattr(config, EXPERIMENT_CONFIG.DETERMINISTIC)
        assert hasattr(config, EXPERIMENT_CONFIG.FAST_DEV_RUN)
        assert hasattr(config, EXPERIMENT_CONFIG.LIMIT_TRAIN_BATCHES)
        assert hasattr(config, EXPERIMENT_CONFIG.LIMIT_VAL_BATCHES)
        assert hasattr(config, EXPERIMENT_CONFIG.MODEL)
        assert hasattr(config, EXPERIMENT_CONFIG.DATA)
        assert hasattr(config, EXPERIMENT_CONFIG.TASK)
        assert hasattr(config, EXPERIMENT_CONFIG.TRAINING)
        assert hasattr(config, EXPERIMENT_CONFIG.WANDB)

        # Test that component configs are created
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.task, TaskConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.wandb, WandBConfig)

    def test_component_configs_customization(self):
        """Test that component configs can be customized."""
        model_config = ModelConfig(hidden_channels=256, num_layers=5)
        data_config = DataConfig(
            sbml_dfs_path=Path("stub_sbml.pkl"),
            napistu_graph_path=Path("stub_graph.pkl"),
        )

        config = ExperimentConfig(model=model_config, data=data_config)

        assert config.model.hidden_channels == 256
        assert config.model.num_layers == 5
        assert config.data.sbml_dfs_path == Path("stub_sbml.pkl")
        assert config.data.napistu_graph_path == Path("stub_graph.pkl")

    def test_serialization_methods(self, stubbed_data_config):
        """Test serialization and deserialization methods."""
        config = ExperimentConfig(
            name="test_experiment", seed=123, data=stubbed_data_config
        )

        # Test to_dict method
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict[EXPERIMENT_CONFIG.NAME] == "test_experiment"
        assert EXPERIMENT_CONFIG.MODEL in config_dict
        assert EXPERIMENT_CONFIG.DATA in config_dict
        assert EXPERIMENT_CONFIG.TASK in config_dict
        assert EXPERIMENT_CONFIG.TRAINING in config_dict
        assert EXPERIMENT_CONFIG.WANDB in config_dict

        # Test JSON serialization
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config.to_json(Path(f.name))

            # Load back
            loaded_config = ExperimentConfig.from_json(Path(f.name))

            assert loaded_config.name == "test_experiment"
            assert loaded_config.seed == 123
            assert isinstance(loaded_config.model, ModelConfig)
            assert isinstance(loaded_config.data, DataConfig)

        # Test YAML serialization
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config.to_yaml(Path(f.name))

            # Load back
            loaded_config = ExperimentConfig.from_yaml(Path(f.name))

            assert loaded_config.name == "test_experiment"
            assert loaded_config.seed == 123
            assert isinstance(loaded_config.model, ModelConfig)
            assert isinstance(loaded_config.data, DataConfig)

    def test_from_yaml_resolves_relative_paths(self):
        """Test that from_yaml resolves relative paths to absolute paths."""

        # Create a temporary YAML file with complex relative paths
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)
            config_data = {
                "seed": 42,
                "data": {
                    "store_dir": "../../.store",
                    "sbml_dfs_path": "../../data/napistu/sbml_dfs.pkl",
                    "napistu_graph_path": "../data/graph.pkl",
                },
                "output_dir": "./output/experiments",
            }
            yaml.dump(config_data, f)

        try:
            # Load the config
            config = ExperimentConfig.from_yaml(temp_path)

            # Verify paths are resolved to absolute paths
            assert config.data.store_dir.is_absolute()
            assert config.data.sbml_dfs_path.is_absolute()
            assert config.data.napistu_graph_path.is_absolute()
            assert config.output_dir.is_absolute()

            # Verify paths are resolved relative to config file directory
            config_dir = temp_path.parent.resolve()
            assert config.data.store_dir == (config_dir / "../../.store").resolve()
            assert (
                config.data.sbml_dfs_path
                == (config_dir / "../../data/napistu/sbml_dfs.pkl").resolve()
            )
            assert (
                config.data.napistu_graph_path
                == (config_dir / "../data/graph.pkl").resolve()
            )
            assert config.output_dir == (config_dir / "./output/experiments").resolve()

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_get_experiment_name(self, stubbed_data_config):
        """Test get_experiment_name method."""
        config = ExperimentConfig(
            data=stubbed_data_config,
            model=ModelConfig(encoder=ENCODERS.GAT, hidden_channels=128, num_layers=3),
            task=TaskConfig(task=TASKS.EDGE_PREDICTION),
        )

        experiment_name = config.get_experiment_name()
        # get_experiment_name uses get_architecture_string which includes head
        # Default head is dot_product, so format is "encoder-head_h{hidden_channels}_l{num_layers}_{task}"
        assert experiment_name == "gat-dot_product_h128_l3_edge_prediction"

        # Test with different values
        config.model.hidden_channels = 256
        config.model.num_layers = 5
        config.task.task = TASKS.NODE_CLASSIFICATION
        experiment_name = config.get_experiment_name()
        assert experiment_name == "gat-dot_product_h256_l5_node_classification"

        # Test with explicit head
        config.model.head = HEADS.BILINEAR
        experiment_name = config.get_experiment_name()
        assert experiment_name == "gat-bilinear_h256_l5_node_classification"

    @pytest.mark.skip_on_windows
    def test_anonymize(self, stubbed_data_config):
        """Test anonymize method masks all Path-like values."""
        # Create config with absolute paths
        config = ExperimentConfig(
            output_dir=Path("/PATH/TO/EXPERIMENTS/test"),
            data=DataConfig(
                store_dir=Path("/PATH/TO/STORE/.store"),
                sbml_dfs_path=Path("/PATH/TO/DATA/sbml.pkl"),
                napistu_graph_path=Path("/PATH/TO/DATA/graph.pkl"),
                copy_to_store=False,
                overwrite=False,
                napistu_data_name="test",
                other_artifacts=[],
            ),
        )

        # Test non-inplace anonymization
        anonymized = config.anonymize(inplace=False)

        # Original config should be unchanged
        assert str(config.output_dir) == "/PATH/TO/EXPERIMENTS/test"
        assert str(config.data.sbml_dfs_path) == "/PATH/TO/DATA/sbml.pkl"

        # Anonymized config should have masked paths (default placeholder is [REDACTED])
        assert str(anonymized.output_dir) == ANONYMIZATION_PLACEHOLDER_DEFAULT
        assert str(anonymized.data.store_dir) == ANONYMIZATION_PLACEHOLDER_DEFAULT
        assert str(anonymized.data.sbml_dfs_path) == ANONYMIZATION_PLACEHOLDER_DEFAULT
        assert (
            str(anonymized.data.napistu_graph_path) == ANONYMIZATION_PLACEHOLDER_DEFAULT
        )

        # Non-path values should be unchanged
        assert anonymized.data.copy_to_store == config.data.copy_to_store
        assert anonymized.data.napistu_data_name == config.data.napistu_data_name

        # Test inplace anonymization
        config2 = ExperimentConfig(
            output_dir=Path("/tmp/test"),
            data=stubbed_data_config,
        )
        result = config2.anonymize(inplace=True)

        # Should return self
        assert result is config2
        assert str(config2.output_dir) == ANONYMIZATION_PLACEHOLDER_DEFAULT

        # Test custom placeholder
        config3 = ExperimentConfig(
            output_dir=Path("/tmp/test"),
            data=stubbed_data_config,
        )
        anonymized3 = config3.anonymize(placeholder="<<local_path>>")
        assert str(anonymized3.output_dir) == "<<local_path>>"

    def test_extra_fields_forbidden(self, stubbed_data_config):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(invalid_field="value", data=stubbed_data_config)
        assert "Extra inputs are not permitted" in str(exc_info.value)


class TestConfigIntegration:
    """Test integration between different config classes."""

    def test_experiment_config_with_custom_components(self):
        """Test creating experiment config with custom component configs."""
        model_config = ModelConfig(
            encoder=ENCODERS.GAT, hidden_channels=256, head=HEADS.MLP
        )

        data_config = DataConfig(
            sbml_dfs_path=Path("custom_sbml.pkl"),
            napistu_graph_path=Path("custom_graph.pkl"),
            napistu_data_name=DEFAULT_ARTIFACTS_NAMES.UNLABELED,
        )

        task_config = TaskConfig(
            task=TASKS.NODE_CLASSIFICATION,
            edge_prediction_neg_sampling_ratio=2.0,
            edge_prediction_neg_sampling_stratify_by=STRATIFY_BY.NODE_SPECIES_TYPE,
            edge_prediction_neg_sampling_strategy=NEGATIVE_SAMPLING_STRATEGIES.DEGREE_WEIGHTED,
            metrics=[METRICS.AUC],
        )

        training_config = TrainingConfig(
            lr=0.01, optimizer=OPTIMIZERS.ADAMW, scheduler="plateau", epochs=100
        )

        wandb_config = WandBConfig(project="custom-project", mode="offline")

        experiment_config = ExperimentConfig(
            name="integration_test",
            model=model_config,
            data=data_config,
            task=task_config,
            training=training_config,
            wandb=wandb_config,
        )

        # Verify all custom values are preserved
        assert experiment_config.name == "integration_test"
        assert experiment_config.model.encoder == ENCODERS.GAT
        assert experiment_config.model.hidden_channels == 256
        assert experiment_config.data.sbml_dfs_path == Path("custom_sbml.pkl")
        assert experiment_config.data.napistu_graph_path == Path("custom_graph.pkl")
        assert (
            experiment_config.data.napistu_data_name
            == DEFAULT_ARTIFACTS_NAMES.UNLABELED
        )
        assert experiment_config.task.task == TASKS.NODE_CLASSIFICATION
        assert experiment_config.training.optimizer == OPTIMIZERS.ADAMW
        assert experiment_config.wandb.project == "custom-project"

    def test_config_validation_cascades(self, stubbed_data_config):
        """Test that validation errors cascade properly."""
        # Test that invalid values in component configs cause validation errors
        with pytest.raises(ValidationError):
            ExperimentConfig(
                model=ModelConfig(encoder="invalid_encoder"), data=stubbed_data_config
            )

        with pytest.raises(ValidationError):
            ExperimentConfig(
                data=DataConfig(
                    sbml_dfs_path=Path("test.pkl"),
                    napistu_graph_path=Path("test.pkl"),
                    invalid_field="should_fail",
                )
            )

        with pytest.raises(ValidationError):
            ExperimentConfig(
                task=TaskConfig(task="invalid_task"), data=stubbed_data_config
            )

        with pytest.raises(ValidationError):
            ExperimentConfig(
                training=TrainingConfig(optimizer="invalid_optimizer"),
                data=stubbed_data_config,
            )

        with pytest.raises(ValidationError):
            ExperimentConfig(
                wandb=WandBConfig(mode="invalid_mode"), data=stubbed_data_config
            )

    def test_config_roundtrip_serialization(self, stubbed_data_config):
        """Test that configs can be serialized and deserialized without data loss."""
        original_config = ExperimentConfig(
            name="roundtrip_test",
            seed=999,
            deterministic=False,
            fast_dev_run=True,
            data=stubbed_data_config,
        )

        # Customize component configs
        original_config.model.hidden_channels = 512
        original_config.data.napistu_data_name = DEFAULT_ARTIFACTS_NAMES.UNLABELED
        original_config.data.other_artifacts = [DEFAULT_ARTIFACTS_NAMES.EDGE_PREDICTION]
        original_config.task.edge_prediction_neg_sampling_ratio = 3.0
        original_config.training.lr = 0.005
        original_config.wandb.project = "roundtrip-project"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # Serialize to JSON
            original_config.to_json(Path(f.name))

            # Deserialize from JSON
            loaded_config = ExperimentConfig.from_json(Path(f.name))

            # Verify all values are preserved
            assert loaded_config.name == "roundtrip_test"
            assert loaded_config.seed == 999
            assert loaded_config.deterministic is False
            assert loaded_config.fast_dev_run is True
            assert loaded_config.model.hidden_channels == 512
            assert (
                loaded_config.data.napistu_data_name
                == DEFAULT_ARTIFACTS_NAMES.UNLABELED
            )
            assert loaded_config.data.other_artifacts == [
                DEFAULT_ARTIFACTS_NAMES.EDGE_PREDICTION
            ]
            assert loaded_config.task.edge_prediction_neg_sampling_ratio == 3.0
            assert loaded_config.training.lr == 0.005
            assert loaded_config.wandb.project == "roundtrip-project"

    def test_create_template_yaml(self):
        """Test that create_template_yaml creates a valid YAML that can be loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            template_path = Path(f.name)

        # Create template with specific paths and name
        sbml_path = Path("test_data/sbml_dfs.pkl")
        graph_path = Path("test_data/napistu_graph.pkl")
        experiment_name = "test_experiment"

        create_template_yaml(
            output_path=template_path,
            sbml_dfs_path=sbml_path,
            napistu_graph_path=graph_path,
            name=experiment_name,
        )

        # Verify file was created
        assert template_path.exists()

        # Load the template using ExperimentConfig.from_yaml
        loaded_config = ExperimentConfig.from_yaml(template_path)

        # Verify the loaded config has the expected values
        assert loaded_config.name == experiment_name
        # Paths are resolved to absolute paths relative to config file directory
        config_dir = template_path.parent.resolve()
        assert loaded_config.data.sbml_dfs_path == (config_dir / sbml_path).resolve()
        assert (
            loaded_config.data.napistu_graph_path == (config_dir / graph_path).resolve()
        )

        # Verify defaults are applied (not in template)
        assert isinstance(loaded_config.model, ModelConfig)
        assert isinstance(loaded_config.task, TaskConfig)
        assert isinstance(loaded_config.training, TrainingConfig)
        assert isinstance(loaded_config.wandb, WandBConfig)

        # Verify wandb fields from template
        assert loaded_config.wandb.group == WANDB_CONFIG_DEFAULTS[WANDB_CONFIG.GROUP]
        assert loaded_config.wandb.tags == WANDB_CONFIG_DEFAULTS[WANDB_CONFIG.TAGS]

        # Clean up
        template_path.unlink()


def test_config_to_data_trimming_spec(stubbed_data_config):
    """Test config_to_data_trimming_spec returns correct trimming flags."""
    # Test edge_prediction task without edge encoder
    config1 = ExperimentConfig(
        data=stubbed_data_config,
        task=TaskConfig(task=TASKS.EDGE_PREDICTION),
        model=ModelConfig(use_edge_encoder=False),
    )
    spec1 = config_to_data_trimming_spec(config1)
    assert spec1[NAPISTU_DATA_TRIM_ARGS.KEEP_EDGE_ATTR] is False
    assert spec1[NAPISTU_DATA_TRIM_ARGS.KEEP_LABELS] is False
    assert spec1[NAPISTU_DATA_TRIM_ARGS.KEEP_MASKS] is True

    # Test node_classification task
    config2 = ExperimentConfig(
        data=stubbed_data_config,
        task=TaskConfig(task=TASKS.NODE_CLASSIFICATION),
        model=ModelConfig(use_edge_encoder=False),
    )
    spec2 = config_to_data_trimming_spec(config2)
    assert spec2[NAPISTU_DATA_TRIM_ARGS.KEEP_EDGE_ATTR] is False
    assert spec2[NAPISTU_DATA_TRIM_ARGS.KEEP_LABELS] is True
    assert spec2[NAPISTU_DATA_TRIM_ARGS.KEEP_MASKS] is True

    # Test edge_prediction with edge encoder that supports weighting
    config3 = ExperimentConfig(
        data=stubbed_data_config,
        task=TaskConfig(task=TASKS.EDGE_PREDICTION),
        model=ModelConfig(use_edge_encoder=True, encoder=ENCODERS.GRAPH_CONV),
    )
    spec3 = config_to_data_trimming_spec(config3)
    assert spec3[NAPISTU_DATA_TRIM_ARGS.KEEP_EDGE_ATTR] is True
    assert spec3[NAPISTU_DATA_TRIM_ARGS.KEEP_LABELS] is False
    assert spec3[NAPISTU_DATA_TRIM_ARGS.KEEP_MASKS] is True


@pytest.mark.skip_on_windows
def test_run_manifest_round_trip(experiment_dict):
    """Test that RunManifest can be saved to YAML and loaded back with all fields preserved."""
    # Get the run_manifest from experiment_dict
    original_manifest = experiment_dict[EXPERIMENT_DICT.RUN_MANIFEST]

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        temp_path = Path(f.name)

    try:
        # Save the manifest
        original_manifest.to_yaml(temp_path)

        # Verify file was created
        assert temp_path.exists()

        # Load it back
        loaded_manifest = RunManifest.from_yaml(temp_path)

        # Verify all fields are preserved
        assert loaded_manifest.experiment_name == original_manifest.experiment_name
        assert loaded_manifest.created_at == original_manifest.created_at
        assert loaded_manifest.wandb_run_id == original_manifest.wandb_run_id
        assert loaded_manifest.wandb_run_url == original_manifest.wandb_run_url
        assert loaded_manifest.wandb_project == original_manifest.wandb_project
        assert loaded_manifest.wandb_entity == original_manifest.wandb_entity

        # Verify experiment_config is preserved
        # Both should be ExperimentConfig objects now
        assert isinstance(original_manifest.experiment_config, ExperimentConfig)
        assert isinstance(loaded_manifest.experiment_config, ExperimentConfig)

        # Compare as dicts (Path objects are serialized as strings via model_dump(mode='json'))
        original_config_dict = original_manifest.experiment_config.model_dump(
            mode="json"
        )
        loaded_config_dict = loaded_manifest.experiment_config.model_dump(mode="json")
        assert loaded_config_dict == original_config_dict

        # Verify all expected attributes are present in experiment_config
        assert hasattr(loaded_manifest.experiment_config, EXPERIMENT_CONFIG.NAME)
        assert hasattr(loaded_manifest.experiment_config, EXPERIMENT_CONFIG.SEED)
        assert hasattr(loaded_manifest.experiment_config, EXPERIMENT_CONFIG.MODEL)
        assert hasattr(loaded_manifest.experiment_config, EXPERIMENT_CONFIG.DATA)
        assert hasattr(loaded_manifest.experiment_config, EXPERIMENT_CONFIG.TASK)
        assert hasattr(loaded_manifest.experiment_config, EXPERIMENT_CONFIG.TRAINING)
        assert hasattr(loaded_manifest.experiment_config, EXPERIMENT_CONFIG.WANDB)

    finally:
        # Clean up temporary file
        if temp_path.exists():
            temp_path.unlink()
