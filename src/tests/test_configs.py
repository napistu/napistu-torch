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
    ENCODER_TYPES,
    EXPERIMENT_CONFIG,
    HEADS,
    METRICS,
    MODEL_CONFIG,
    OPTIMIZERS,
    SCHEDULERS,
    TASK_CONFIG,
    TASKS,
    TRAINING_CONFIG,
    VALID_ENCODER_TYPES,
    VALID_HEADS,
    VALID_OPTIMIZERS,
    VALID_SCHEDULERS,
    VALID_TASKS,
    VALID_WANDB_MODES,
    WANDB_CONFIG,
    WANDB_MODES,
)
from napistu_torch.load.constants import (
    SPLITTING_STRATEGIES,
    VALID_SPLITTING_STRATEGIES,
)


class TestModelConfig:
    """Test ModelConfig class."""

    def test_encoder_type_validation(self):
        """Test encoder_type validation with valid and invalid values."""
        # Test valid encoder types
        for encoder_type in VALID_ENCODER_TYPES:
            config = ModelConfig(encoder_type=encoder_type)
            assert hasattr(config, "encoder_type")
            assert config.encoder_type == encoder_type

        # Test invalid encoder type
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(encoder_type="invalid_encoder")
        assert "Invalid encoder type" in str(exc_info.value)

    def test_head_type_validation(self):
        """Test head_type validation with valid and invalid values."""
        # Test valid head types
        for head_type in VALID_HEADS:
            config = ModelConfig(head_type=head_type)
            assert hasattr(config, "head_type")
            assert config.head_type == head_type

        # Test invalid head type
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(head_type="invalid_head")
        assert "Invalid head type" in str(exc_info.value)

    def test_hidden_channels_validation(self):
        """Test hidden_channels validation with valid and invalid values."""
        # Test valid power of 2 values
        valid_values = [64, 128, 256, 512]
        for value in valid_values:
            config = ModelConfig(hidden_channels=value)
            assert hasattr(config, "hidden_channels")
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
            assert hasattr(config, "num_layers")
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
            assert hasattr(config, "dropout")
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
        assert hasattr(config, MODEL_CONFIG.AGGREGATOR)
        assert hasattr(config, MODEL_CONFIG.HEADS)
        assert hasattr(config, MODEL_CONFIG.HEAD_HIDDEN_DIM)

        # Test that they can be customized
        config = ModelConfig(aggregator="max", heads=8, head_hidden_dim=128)
        assert config.aggregator == "max"
        assert config.heads == 8
        assert config.head_hidden_dim == 128

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(invalid_field="value")
        assert "Extra inputs are not permitted" in str(exc_info.value)


class TestDataConfig:
    """Test DataConfig class."""

    def test_splitting_strategy_validation(self):
        """Test splitting_strategy validation with valid and invalid values."""
        # Test valid splitting strategies
        for strategy in VALID_SPLITTING_STRATEGIES:
            config = DataConfig(splitting_strategy=strategy)
            assert hasattr(config, DATA_CONFIG.SPLITTING_STRATEGY)
            assert config.splitting_strategy == strategy

        # Test invalid splitting strategy
        with pytest.raises(ValidationError) as exc_info:
            DataConfig(splitting_strategy="invalid_strategy")
        assert "Invalid splitting strategy" in str(exc_info.value)

    def test_size_validation(self):
        """Test train/val/test size validation with valid and invalid values."""
        # Test valid sizes
        valid_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
        for size in valid_sizes:
            config = DataConfig(train_size=size, val_size=0.1, test_size=0.1)
            assert hasattr(config, DATA_CONFIG.TRAIN_SIZE)
            assert config.train_size == size

            config = DataConfig(train_size=0.7, val_size=size, test_size=0.1)
            assert hasattr(config, DATA_CONFIG.VAL_SIZE)
            assert config.val_size == size

            config = DataConfig(train_size=0.7, val_size=0.1, test_size=size)
            assert hasattr(config, DATA_CONFIG.TEST_SIZE)
            assert config.test_size == size

        # Test invalid values (at boundaries)
        with pytest.raises(ValidationError):
            DataConfig(train_size=0.0)  # At minimum (should be > 0.0)

        with pytest.raises(ValidationError):
            DataConfig(val_size=0.0)  # At minimum (should be > 0.0)

        with pytest.raises(ValidationError):
            DataConfig(test_size=0.0)  # At minimum (should be > 0.0)

        with pytest.raises(ValidationError):
            DataConfig(train_size=1.0)  # At maximum (should be < 1.0)

        with pytest.raises(ValidationError):
            DataConfig(val_size=1.0)  # At maximum (should be < 1.0)

        with pytest.raises(ValidationError):
            DataConfig(test_size=1.0)  # At maximum (should be < 1.0)

    def test_required_fields_exist(self):
        """Test that all required fields exist and can be set."""
        config = DataConfig()

        # Test that all fields exist
        assert hasattr(config, DATA_CONFIG.NAME)
        assert hasattr(config, DATA_CONFIG.STORE_DIR)
        assert hasattr(config, DATA_CONFIG.SPLITTING_STRATEGY)
        assert hasattr(config, DATA_CONFIG.TRAIN_SIZE)
        assert hasattr(config, DATA_CONFIG.VAL_SIZE)
        assert hasattr(config, DATA_CONFIG.TEST_SIZE)

        # Test that they can be customized
        custom_path = Path("/custom/path")
        config = DataConfig(
            name="custom_name",
            store_dir=custom_path,
            train_size=0.8,
            val_size=0.1,
            test_size=0.1,
        )
        assert config.name == "custom_name"
        assert config.store_dir == custom_path
        assert isinstance(config.store_dir, Path)
        assert config.train_size == 0.8
        assert config.val_size == 0.1
        assert config.test_size == 0.1


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
        """Test neg_sampling_ratio validation with valid and invalid values."""
        # Test valid values
        valid_ratios = [0.1, 1.0, 2.0, 5.0, 10.0]
        for ratio in valid_ratios:
            config = TaskConfig(neg_sampling_ratio=ratio)
            assert hasattr(config, TASK_CONFIG.NEG_SAMPLING_RATIO)
            assert config.neg_sampling_ratio == ratio

        # Test invalid values
        with pytest.raises(ValidationError):
            TaskConfig(neg_sampling_ratio=0.0)  # At minimum (should be > 0.0)

        with pytest.raises(ValidationError):
            TaskConfig(neg_sampling_ratio=-1.0)  # Negative value

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
        assert hasattr(config, TASK_CONFIG.NEG_SAMPLING_RATIO)
        assert hasattr(config, TASK_CONFIG.METRICS)

        # Test that they can be customized
        config = TaskConfig(
            task=TASKS.NODE_CLASSIFICATION,
            neg_sampling_ratio=3.0,
            metrics=[METRICS.AUC],
        )
        assert config.task == TASKS.NODE_CLASSIFICATION
        assert config.neg_sampling_ratio == 3.0
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

    def test_required_fields_exist(self):
        """Test that all required fields exist and can be set."""
        config = ExperimentConfig()

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
        data_config = DataConfig(name="custom_data", train_size=0.8)

        config = ExperimentConfig(model=model_config, data=data_config)

        assert config.model.hidden_channels == 256
        assert config.model.num_layers == 5
        assert config.data.name == "custom_data"
        assert config.data.train_size == 0.8

    def test_serialization_methods(self):
        """Test serialization and deserialization methods."""
        config = ExperimentConfig(name="test_experiment", seed=123)

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

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(invalid_field="value")
        assert "Extra inputs are not permitted" in str(exc_info.value)


class TestConfigIntegration:
    """Test integration between different config classes."""

    def test_experiment_config_with_custom_components(self):
        """Test creating experiment config with custom component configs."""
        model_config = ModelConfig(
            encoder_type=ENCODER_TYPES.GAT, hidden_channels=256, head_type=HEADS.MLP
        )

        data_config = DataConfig(
            name="custom_dataset",
            splitting_strategy=SPLITTING_STRATEGIES.VERTEX_MASK,
            train_size=0.8,
        )

        task_config = TaskConfig(
            task=TASKS.NODE_CLASSIFICATION,
            neg_sampling_ratio=2.0,
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
        assert experiment_config.model.encoder_type == ENCODER_TYPES.GAT
        assert experiment_config.model.hidden_channels == 256
        assert experiment_config.data.name == "custom_dataset"
        assert experiment_config.task.task == TASKS.NODE_CLASSIFICATION
        assert experiment_config.training.optimizer == OPTIMIZERS.ADAMW
        assert experiment_config.wandb.project == "custom-project"

    def test_config_validation_cascades(self):
        """Test that validation errors cascade properly."""
        # Test that invalid values in component configs cause validation errors
        with pytest.raises(ValidationError):
            ExperimentConfig(model=ModelConfig(encoder_type="invalid_encoder"))

        with pytest.raises(ValidationError):
            ExperimentConfig(data=DataConfig(splitting_strategy="invalid_strategy"))

        with pytest.raises(ValidationError):
            ExperimentConfig(task=TaskConfig(task="invalid_task"))

        with pytest.raises(ValidationError):
            ExperimentConfig(training=TrainingConfig(optimizer="invalid_optimizer"))

        with pytest.raises(ValidationError):
            ExperimentConfig(wandb=WandBConfig(mode="invalid_mode"))

    def test_config_roundtrip_serialization(self):
        """Test that configs can be serialized and deserialized without data loss."""
        original_config = ExperimentConfig(
            name="roundtrip_test", seed=999, deterministic=False, fast_dev_run=True
        )

        # Customize component configs
        original_config.model.hidden_channels = 512
        original_config.data.train_size = 0.9
        original_config.task.neg_sampling_ratio = 3.0
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
            assert loaded_config.data.train_size == 0.9
            assert loaded_config.task.neg_sampling_ratio == 3.0
            assert loaded_config.training.lr == 0.005
            assert loaded_config.wandb.project == "roundtrip-project"
