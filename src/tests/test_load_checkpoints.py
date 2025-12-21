"""Tests for Checkpoint loading and validation utilities."""

import pytest
from pydantic import ValidationError

from napistu_torch.configs import ModelConfig, TrainingConfig
from napistu_torch.constants import NAPISTU_DATA_SUMMARY_TYPES
from napistu_torch.load.checkpoints import (
    Checkpoint,
    DataMetadata,
    EdgeEncoderMetadata,
    EncoderMetadata,
    HeadMetadata,
    ModelMetadata,
)
from napistu_torch.load.constants import (
    CHECKPOINT_HYPERPARAMETERS,
    CHECKPOINT_STRUCTURE,
)
from napistu_torch.models.constants import ENCODERS, HEADS, MODEL_DEFS
from napistu_torch.napistu_data import NapistuData


def create_minimal_checkpoint_dict(
    napistu_data: NapistuData,
    encoder_metadata: EncoderMetadata = None,
    head_metadata: HeadMetadata = None,
    edge_encoder_metadata: EdgeEncoderMetadata = None,
    training_config: TrainingConfig = None,
) -> dict:
    """
    Create a minimal valid checkpoint dictionary for testing.

    Parameters
    ----------
    napistu_data : NapistuData
        NapistuData fixture to generate data metadata from
    encoder_metadata : EncoderMetadata, optional
        Encoder metadata. If None, creates minimal SAGE encoder.
    head_metadata : HeadMetadata, optional
        Head metadata. If None, creates minimal DOT_PRODUCT head.
    edge_encoder_metadata : EdgeEncoderMetadata, optional
        Edge encoder metadata. If None, no edge encoder is included.
    training_config : TrainingConfig, optional
        Training configuration. If None, creates minimal TrainingConfig.

    Returns
    -------
    dict
        Minimal checkpoint dictionary that passes CheckpointStructure validation
    """
    if training_config is None:
        training_config = TrainingConfig()

    # Create minimal encoder metadata if not provided
    if encoder_metadata is None:
        encoder_metadata = EncoderMetadata(
            encoder=ENCODERS.SAGE,
            in_channels=napistu_data.num_node_features,
            hidden_channels=128,
            num_layers=3,
        )

    # Create minimal head metadata if not provided
    if head_metadata is None:
        head_metadata = HeadMetadata(
            head=HEADS.DOT_PRODUCT,
            hidden_channels=128,
        )

    # Create model metadata
    model_metadata = ModelMetadata(
        encoder=encoder_metadata.model_dump(),
        head=head_metadata.model_dump(),
        edge_encoder=(
            edge_encoder_metadata.model_dump() if edge_encoder_metadata else None
        ),
    )

    # Create data metadata from NapistuData.get_summary()
    data_summary = napistu_data.get_summary(NAPISTU_DATA_SUMMARY_TYPES.VALIDATION)
    data_metadata = DataMetadata.model_validate(data_summary)

    # Create hyperparameters dict
    hyper_parameters = {
        CHECKPOINT_HYPERPARAMETERS.CONFIG: training_config.model_dump(),
        CHECKPOINT_HYPERPARAMETERS.MODEL: model_metadata.model_dump(),
        CHECKPOINT_HYPERPARAMETERS.DATA: data_metadata.model_dump(),
    }

    # Create checkpoint dict
    checkpoint_dict = {
        CHECKPOINT_STRUCTURE.STATE_DICT: {
            "encoder.layer_0.weight": None,  # Minimal non-empty state_dict
        },
        CHECKPOINT_STRUCTURE.HYPER_PARAMETERS: hyper_parameters,
        CHECKPOINT_STRUCTURE.EPOCH: 10,
        CHECKPOINT_STRUCTURE.GLOBAL_STEP: 100,
    }

    return checkpoint_dict


class TestCheckpoint:
    """Test Checkpoint class."""

    def test_checkpoint_initialization(self, napistu_data):
        """Test that Checkpoint can be initialized with a valid checkpoint dict."""
        checkpoint_dict = create_minimal_checkpoint_dict(napistu_data)
        checkpoint = Checkpoint(checkpoint_dict)

        assert checkpoint.encoder_metadata.encoder == ENCODERS.SAGE
        assert checkpoint.head_metadata.head == HEADS.DOT_PRODUCT
        assert checkpoint.data_metadata.name == napistu_data.name

    def test_checkpoint_validation_fails_invalid_structure(self):
        """Test that Checkpoint validation fails for invalid structures."""
        # Missing state_dict
        with pytest.raises(ValidationError):
            Checkpoint({CHECKPOINT_STRUCTURE.HYPER_PARAMETERS: {}})

        # Empty state_dict
        with pytest.raises(ValidationError):
            Checkpoint(
                {
                    CHECKPOINT_STRUCTURE.STATE_DICT: {},
                    CHECKPOINT_STRUCTURE.HYPER_PARAMETERS: {},
                }
            )

        # Missing hyper_parameters
        with pytest.raises(ValidationError):
            Checkpoint({CHECKPOINT_STRUCTURE.STATE_DICT: {"weight": None}})

    def test_update_model_config_with_encoder(self, napistu_data):
        """Test _update_model_config_with_encoder updates ModelConfig correctly."""
        encoder_metadata = EncoderMetadata(
            encoder=ENCODERS.GAT,
            in_channels=napistu_data.num_node_features,
            hidden_channels=128,
            num_layers=3,
            dropout=0.2,
            gat_heads=4,
            gat_concat=True,
        )
        checkpoint_dict = create_minimal_checkpoint_dict(
            napistu_data,
            encoder_metadata=encoder_metadata,
        )
        checkpoint = Checkpoint(checkpoint_dict)

        # Create a fresh ModelConfig with defaults
        model_config = ModelConfig()

        # Update with encoder from checkpoint
        checkpoint._update_model_config_with_encoder(model_config, inplace=True)

        # Verify encoder fields are updated
        assert model_config.encoder == ENCODERS.GAT
        assert model_config.hidden_channels == 128
        assert model_config.num_layers == 3
        assert model_config.dropout == 0.2
        assert model_config.gat_heads == 4
        assert model_config.gat_concat is True

        # Verify edge encoder is not set
        assert model_config.use_edge_encoder is False

    def test_update_model_config_with_encoder_with_edge_encoder(self, napistu_data):
        """Test _update_model_config_with_encoder with edge encoder."""
        encoder_metadata = EncoderMetadata(
            encoder=ENCODERS.GRAPH_CONV,
            in_channels=napistu_data.num_node_features,
            hidden_channels=128,
            num_layers=3,
            graph_conv_aggregator="mean",
        )
        edge_encoder_metadata = EdgeEncoderMetadata(
            edge_in_channels=napistu_data.num_edge_features,
            edge_encoder_dim=32,
            edge_encoder_dropout=0.1,
            edge_encoder_init_bias=None,
        )
        checkpoint_dict = create_minimal_checkpoint_dict(
            napistu_data,
            encoder_metadata=encoder_metadata,
            edge_encoder_metadata=edge_encoder_metadata,
        )
        checkpoint = Checkpoint(checkpoint_dict)

        model_config = ModelConfig()
        updated_config = checkpoint._update_model_config_with_encoder(
            model_config, inplace=False
        )

        # Verify encoder fields
        assert updated_config.encoder == ENCODERS.GRAPH_CONV
        assert updated_config.graph_conv_aggregator == "mean"

        # Verify edge encoder fields
        assert updated_config.use_edge_encoder is True
        assert updated_config.edge_encoder_dim == 32
        assert updated_config.edge_encoder_dropout == 0.1

    def test_update_model_config_with_head(self, napistu_data):
        """Test _update_model_config_with_head updates ModelConfig correctly."""
        head_metadata = HeadMetadata(
            head=HEADS.MLP,
            hidden_channels=128,
            mlp_hidden_dim=64,
            mlp_num_layers=2,
            mlp_dropout=0.1,
        )
        checkpoint_dict = create_minimal_checkpoint_dict(
            napistu_data,
            head_metadata=head_metadata,
        )
        checkpoint = Checkpoint(checkpoint_dict)

        model_config = ModelConfig()
        updated_config = checkpoint._update_model_config_with_head(
            model_config, inplace=False
        )

        # Verify head fields are updated
        assert updated_config.head == HEADS.MLP
        assert updated_config.mlp_hidden_dim == 64
        assert updated_config.mlp_num_layers == 2
        assert updated_config.mlp_dropout == 0.1

    def test_update_model_config_with_head_relation_aware(
        self, edge_prediction_with_sbo_relations
    ):
        """Test _update_model_config_with_head with relation-aware heads."""
        num_relations = 10
        head_metadata = HeadMetadata(
            head=HEADS.ROTATE,
            hidden_channels=128,
            rotate_margin=9.0,
            num_relations=num_relations,
        )
        checkpoint_dict = create_minimal_checkpoint_dict(
            edge_prediction_with_sbo_relations,
            head_metadata=head_metadata,
        )
        checkpoint = Checkpoint(checkpoint_dict)

        model_config = ModelConfig()
        checkpoint._update_model_config_with_head(model_config, inplace=True)

        # Verify head fields are updated
        assert model_config.head == HEADS.ROTATE
        assert model_config.rotate_margin == 9.0

    def test_get_encoder_config(self, napistu_data):
        """Test get_encoder_config returns encoder metadata as dict."""
        encoder_metadata = EncoderMetadata(
            encoder=ENCODERS.SAGE,
            in_channels=napistu_data.num_node_features,
            hidden_channels=128,
            num_layers=3,
        )
        checkpoint_dict = create_minimal_checkpoint_dict(
            napistu_data,
            encoder_metadata=encoder_metadata,
        )
        checkpoint = Checkpoint(checkpoint_dict)

        encoder_config = checkpoint.get_encoder_config()
        assert isinstance(encoder_config, dict)
        assert encoder_config[MODEL_DEFS.ENCODER] == ENCODERS.SAGE
        assert encoder_config[MODEL_DEFS.HIDDEN_CHANNELS] == 128
        assert encoder_config[MODEL_DEFS.NUM_LAYERS] == 3

    def test_get_head_config(self, napistu_data):
        """Test get_head_config returns head metadata as dict."""
        head_metadata = HeadMetadata(
            head=HEADS.ATTENTION,
            hidden_channels=128,
        )
        checkpoint_dict = create_minimal_checkpoint_dict(
            napistu_data,
            head_metadata=head_metadata,
        )
        checkpoint = Checkpoint(checkpoint_dict)

        head_config = checkpoint.get_head_config()
        assert isinstance(head_config, dict)
        assert head_config[MODEL_DEFS.HEAD] == HEADS.ATTENTION
        assert head_config[MODEL_DEFS.HIDDEN_CHANNELS] == 128

    def test_get_data_summary(self, napistu_data):
        """Test get_data_summary returns data metadata as dict."""
        checkpoint_dict = create_minimal_checkpoint_dict(napistu_data)
        checkpoint = Checkpoint(checkpoint_dict)

        data_summary = checkpoint.get_data_summary()
        assert isinstance(data_summary, dict)
        assert data_summary["name"] == napistu_data.name
        assert data_summary["num_nodes"] == napistu_data.num_nodes
        assert data_summary["num_edges"] == napistu_data.num_edges

    def test_repr(self, napistu_data):
        """Test __repr__ returns a string representation."""
        encoder_metadata = EncoderMetadata(
            encoder=ENCODERS.GAT,
            in_channels=napistu_data.num_node_features,
            hidden_channels=128,
            num_layers=3,
        )
        head_metadata = HeadMetadata(
            head=HEADS.MLP,
            hidden_channels=128,
        )
        checkpoint_dict = create_minimal_checkpoint_dict(
            napistu_data,
            encoder_metadata=encoder_metadata,
            head_metadata=head_metadata,
        )
        checkpoint = Checkpoint(checkpoint_dict)

        repr_str = repr(checkpoint)
        assert isinstance(repr_str, str)
        assert ENCODERS.GAT in repr_str
        assert HEADS.MLP in repr_str
        assert napistu_data.name in repr_str

    def test_assert_same_napistu_data(self, napistu_data):
        """Test assert_same_napistu_data passes when data matches."""
        checkpoint_dict = create_minimal_checkpoint_dict(napistu_data)
        checkpoint = Checkpoint(checkpoint_dict)

        # Should not raise an error when data matches
        checkpoint.assert_same_napistu_data(napistu_data)
