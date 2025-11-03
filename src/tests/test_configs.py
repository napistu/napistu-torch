"""Tests for napistu_torch configs module."""

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from napistu_torch.configs import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TaskConfig,
    TrainingConfig,
    WandBConfig,
)
from napistu_torch.constants import (
    DATA_CONFIG,
    EXPERIMENT_CONFIG,
    METRICS,
    OPTIMIZERS,
    SCHEDULERS,
    TASK_CONFIG,
    TRAINING_CONFIG,
    VALID_OPTIMIZERS,
    VALID_SCHEDULERS,
    VALID_WANDB_MODES,
    WANDB_CONFIG,
    WANDB_MODES,
)
from napistu_torch.load.constants import DEFAULT_ARTIFACTS_NAMES, STRATIFY_BY
from napistu_torch.models.constants import (
    ENCODER_SPECIFIC_ARGS,
    ENCODERS,
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
            assert hasattr(config, ENCODER_SPECIFIC_ARGS.DROPOUT)
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

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(invalid_field="value")
        assert "Extra inputs are not permitted" in str(exc_info.value)


class TestDataConfig:
    """Test DataConfig class."""

    def test_required_fields_exist(self):
        """Test that all required fields exist and can be set."""
        # Test that required fields exist in the model
        model_fields = DataConfig.model_fields
        assert DATA_CONFIG.NAME in model_fields
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

        assert config.name == "default"
        assert config.store_dir == Path(".store")
        assert config.copy_to_store is False
        assert config.overwrite is False
        assert config.napistu_data_name == DEFAULT_ARTIFACTS_NAMES.EDGE_PREDICTION
        assert config.other_artifacts == []

    def test_custom_values(self):
        """Test that fields can be customized."""
        custom_path = Path("/custom/path")
        config = DataConfig(
            name="custom_name",
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

        assert config.name == "custom_name"
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

    def test_batch_size_validation(self):
        """Test batch size validation with valid and invalid values."""
        # Test valid values
        valid_batch_sizes = [1, 16, 32, 64, 128]
        for batch_size in valid_batch_sizes:
            config = TrainingConfig(batch_size=batch_size)
            assert hasattr(config, TRAINING_CONFIG.BATCH_SIZE)
            assert config.batch_size == batch_size

        # Test invalid values
        with pytest.raises(ValidationError):
            TrainingConfig(batch_size=0)  # At minimum (should be > 0)

        with pytest.raises(ValidationError):
            TrainingConfig(batch_size=-1)  # Negative value

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
        assert hasattr(config, TRAINING_CONFIG.BATCH_SIZE)
        assert hasattr(config, TRAINING_CONFIG.ACCELERATOR)
        assert hasattr(config, TRAINING_CONFIG.DEVICES)
        assert hasattr(config, TRAINING_CONFIG.PRECISION)
        assert hasattr(config, TRAINING_CONFIG.EARLY_STOPPING)
        assert hasattr(config, TRAINING_CONFIG.EARLY_STOPPING_PATIENCE)
        assert hasattr(config, TRAINING_CONFIG.EARLY_STOPPING_METRIC)
        assert hasattr(config, TRAINING_CONFIG.CHECKPOINT_DIR)
        assert hasattr(config, TRAINING_CONFIG.SAVE_CHECKPOINTS)
        assert hasattr(config, TRAINING_CONFIG.CHECKPOINT_METRIC)

        # Test that they can be customized
        custom_path = Path("/custom/checkpoints")
        config = TrainingConfig(
            lr=0.01,
            weight_decay=0.001,
            optimizer=OPTIMIZERS.ADAMW,
            scheduler=SCHEDULERS.PLATEAU,
            epochs=100,
            batch_size=64,
            checkpoint_dir=custom_path,
        )
        assert config.lr == 0.01
        assert config.weight_decay == 0.001
        assert config.optimizer == OPTIMIZERS.ADAMW
        assert config.scheduler == SCHEDULERS.PLATEAU
        assert config.epochs == 100
        assert config.batch_size == 64
        assert config.checkpoint_dir == custom_path
        assert isinstance(config.checkpoint_dir, Path)


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
        assert hasattr(config, WANDB_CONFIG.SAVE_DIR)
        assert hasattr(config, WANDB_CONFIG.LOG_MODEL)
        assert hasattr(config, WANDB_CONFIG.MODE)

        # Test that they can be customized
        custom_path = Path("/custom/wandb")
        config = WandBConfig(
            project="my-project",
            entity="my-entity",
            group="my-group",
            tags=["tag1", "tag2"],
            save_dir=custom_path,
            log_model=True,
            mode=WANDB_MODES.OFFLINE,
        )
        assert config.project == "my-project"
        assert config.entity == "my-entity"
        assert config.group == "my-group"
        assert config.tags == ["tag1", "tag2"]
        assert config.save_dir == custom_path
        assert isinstance(config.save_dir, Path)
        assert config.log_model is True
        assert config.mode == WANDB_MODES.OFFLINE


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
            name="custom_data",
            sbml_dfs_path=Path("stub_sbml.pkl"),
            napistu_graph_path=Path("stub_graph.pkl"),
        )

        config = ExperimentConfig(model=model_config, data=data_config)

        assert config.model.hidden_channels == 256
        assert config.model.num_layers == 5
        assert config.data.name == "custom_data"
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
            name="custom_dataset",
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
        assert experiment_config.data.name == "custom_dataset"
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
        original_config.data.name = "roundtrip_data"
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
            assert loaded_config.data.name == "roundtrip_data"
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
