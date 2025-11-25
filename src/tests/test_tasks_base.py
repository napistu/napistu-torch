"""Tests for BaseTask functionality."""

import torch

from napistu_torch.configs import ModelConfig
from napistu_torch.models.constants import (
    EDGE_ENCODER_ARGS,
    EDGE_WEIGHTING_TYPE,
    ENCODER_DEFS,
    ENCODERS,
    HEADS,
    MODEL_DEFS,
)
from napistu_torch.models.heads import Decoder
from napistu_torch.models.message_passing_encoder import MessagePassingEncoder
from napistu_torch.tasks.edge_prediction import EdgePredictionTask


def test_to_model_config_dict_with_edge_encoder(edge_masked_napistu_data):
    """Test that to_model_config_dict includes edge encoder config when present."""
    # Create model config with edge encoder enabled
    model_config = ModelConfig(
        encoder=ENCODERS.GCN,  # GCN supports edge weights
        hidden_channels=32,
        num_layers=2,
        use_edge_encoder=True,
        edge_encoder_dim=16,
        edge_encoder_dropout=0.1,
    )

    # Add dummy edge attributes to test data
    edge_masked_napistu_data.edge_attr = torch.randn(
        edge_masked_napistu_data.edge_index.size(1), 10
    )

    # Create encoder and head with edge encoder
    encoder = MessagePassingEncoder.from_config(
        model_config,
        in_channels=edge_masked_napistu_data.num_node_features,
        edge_in_channels=edge_masked_napistu_data.num_edge_features,
    )
    # Use Decoder which has the config property
    head = Decoder(hidden_channels=32, head_type=HEADS.DOT_PRODUCT)

    # Create task
    task = EdgePredictionTask(encoder=encoder, head=head)

    # Get config dictionary
    config_dict = task.to_model_config_dict()

    # Verify encoder config is present
    assert isinstance(config_dict, dict)
    assert MODEL_DEFS.ENCODER_TYPE in config_dict
    assert MODEL_DEFS.HIDDEN_CHANNELS in config_dict
    assert MODEL_DEFS.NUM_LAYERS in config_dict
    assert config_dict[MODEL_DEFS.ENCODER_TYPE] == ENCODERS.GCN
    assert config_dict[MODEL_DEFS.HIDDEN_CHANNELS] == 32
    assert config_dict[MODEL_DEFS.NUM_LAYERS] == 2

    # Verify edge encoder config is present with model config names
    assert EDGE_ENCODER_ARGS.EDGE_ENCODER_DIM in config_dict
    assert EDGE_ENCODER_ARGS.EDGE_ENCODER_DROPOUT in config_dict
    assert config_dict[EDGE_ENCODER_ARGS.EDGE_ENCODER_DIM] == 16
    assert config_dict[EDGE_ENCODER_ARGS.EDGE_ENCODER_DROPOUT] == 0.1

    # Verify head config is present (Decoder instances have config property)
    assert isinstance(head, Decoder)
    head_config = head.config
    for key, value in head_config.items():
        assert key in config_dict
        assert config_dict[key] == value

    # Verify edge encoder is actually present in encoder
    assert hasattr(encoder, ENCODER_DEFS.EDGE_WEIGHTING_TYPE)
    assert (
        getattr(encoder, ENCODER_DEFS.EDGE_WEIGHTING_TYPE)
        == EDGE_WEIGHTING_TYPE.LEARNED_ENCODER
    )
    edge_encoder = getattr(encoder, ENCODER_DEFS.EDGE_WEIGHTING_VALUE)
    assert edge_encoder is not None
    assert edge_encoder.hidden_dim == 16
    assert edge_encoder.edge_dim == 10
