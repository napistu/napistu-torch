"""Tests for EdgeEncoder."""

from napistu_torch.models.constants import (
    EDGE_ENCODER_ARGS,
    MODEL_DEFS,
)
from napistu_torch.models.edge_encoder import EdgeEncoder


class TestEdgeEncoderConfig:
    """Tests for EdgeEncoder.config method."""

    def test_config_without_to_model_config_names(self):
        """Test config method returns internal arg names when to_model_config_names=False."""
        edge_encoder = EdgeEncoder(
            edge_dim=10,
            hidden_dim=32,
            dropout=0.1,
            init_bias=0.5,
        )

        config = edge_encoder.get_summary(to_model_config_names=False)

        # Should return internal arg names
        assert isinstance(config, dict)
        assert config[MODEL_DEFS.EDGE_IN_CHANNELS] == 10
        assert config[EDGE_ENCODER_ARGS.HIDDEN_DIM] == 32
        assert config[EDGE_ENCODER_ARGS.DROPOUT] == 0.1
        assert config[EDGE_ENCODER_ARGS.INIT_BIAS] == 0.5

    def test_config_with_to_model_config_names(self):
        """Test config method converts to model config names when to_model_config_names=True."""
        edge_encoder = EdgeEncoder(
            edge_dim=10,
            hidden_dim=32,
            dropout=0.1,
            init_bias=0.5,
        )

        config = edge_encoder.get_summary(to_model_config_names=True)

        # Should return model config names for mapped args
        assert isinstance(config, dict)
        assert config[MODEL_DEFS.EDGE_IN_CHANNELS] == 10  # Not mapped, stays the same
        assert (
            config[EDGE_ENCODER_ARGS.EDGE_ENCODER_DIM] == 32
        )  # Mapped from HIDDEN_DIM
        assert (
            config[EDGE_ENCODER_ARGS.EDGE_ENCODER_DROPOUT] == 0.1
        )  # Mapped from DROPOUT
        assert (
            config[EDGE_ENCODER_ARGS.EDGE_ENCODER_INIT_BIAS] == 0.5
        )  # Mapped from INIT_BIAS

        # Should not contain internal arg names that were mapped
        assert EDGE_ENCODER_ARGS.HIDDEN_DIM not in config
        assert EDGE_ENCODER_ARGS.DROPOUT not in config
        assert EDGE_ENCODER_ARGS.INIT_BIAS not in config

    def test_config_default_parameter(self):
        """Test config method defaults to to_model_config_names=False."""
        edge_encoder = EdgeEncoder(
            edge_dim=10,
            hidden_dim=32,
            dropout=0.1,
            init_bias=0.0,
        )

        config_default = edge_encoder.get_summary()
        config_explicit = edge_encoder.get_summary(to_model_config_names=False)

        # Should be the same
        assert config_default == config_explicit
        assert config_default[EDGE_ENCODER_ARGS.HIDDEN_DIM] == 32

    def test_config_preserves_all_parameters(self):
        """Test that config method preserves all initialization parameters."""
        edge_encoder = EdgeEncoder(
            edge_dim=20,
            hidden_dim=64,
            dropout=0.2,
            init_bias=1.0,
        )

        config = edge_encoder.get_summary(to_model_config_names=False)

        # Should contain all initialization parameters
        assert len(config) == 4
        assert config[MODEL_DEFS.EDGE_IN_CHANNELS] == 20
        assert config[EDGE_ENCODER_ARGS.HIDDEN_DIM] == 64
        assert config[EDGE_ENCODER_ARGS.DROPOUT] == 0.2
        assert config[EDGE_ENCODER_ARGS.INIT_BIAS] == 1.0

    def test_config_mapping_correctness(self):
        """Test that the mapping between internal and model config names is correct."""
        edge_encoder = EdgeEncoder(
            edge_dim=10,
            hidden_dim=32,
            dropout=0.1,
            init_bias=0.5,
        )

        config_internal = edge_encoder.get_summary(to_model_config_names=False)
        config_model = edge_encoder.get_summary(to_model_config_names=True)

        # Verify mapping is correct
        assert (
            config_model[EDGE_ENCODER_ARGS.EDGE_ENCODER_DIM]
            == config_internal[EDGE_ENCODER_ARGS.HIDDEN_DIM]
        )
        assert (
            config_model[EDGE_ENCODER_ARGS.EDGE_ENCODER_DROPOUT]
            == config_internal[EDGE_ENCODER_ARGS.DROPOUT]
        )
        assert (
            config_model[EDGE_ENCODER_ARGS.EDGE_ENCODER_INIT_BIAS]
            == config_internal[EDGE_ENCODER_ARGS.INIT_BIAS]
        )
