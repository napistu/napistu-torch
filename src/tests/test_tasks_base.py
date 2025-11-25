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


def test_get_summary_with_edge_encoder(edge_masked_napistu_data):
    """Test that get_summary includes edge encoder summary when present."""
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

    # Get summary dictionary (nested structure)
    summary_dict = task.get_summary()

    # Verify top-level structure
    assert isinstance(summary_dict, dict)
    assert MODEL_DEFS.ENCODER in summary_dict
    assert MODEL_DEFS.EDGE_ENCODER in summary_dict
    assert MODEL_DEFS.HEAD in summary_dict

    # Verify encoder config is present in nested structure
    encoder_summary = summary_dict[MODEL_DEFS.ENCODER]
    assert isinstance(encoder_summary, dict)
    assert MODEL_DEFS.ENCODER_TYPE in encoder_summary
    assert MODEL_DEFS.HIDDEN_CHANNELS in encoder_summary
    assert MODEL_DEFS.NUM_LAYERS in encoder_summary
    assert encoder_summary[MODEL_DEFS.ENCODER_TYPE] == ENCODERS.GCN
    assert encoder_summary[MODEL_DEFS.HIDDEN_CHANNELS] == 32
    assert encoder_summary[MODEL_DEFS.NUM_LAYERS] == 2

    # Verify edge encoder config is present with model config names in nested structure
    edge_encoder_summary = summary_dict[MODEL_DEFS.EDGE_ENCODER]
    assert isinstance(edge_encoder_summary, dict)
    assert EDGE_ENCODER_ARGS.EDGE_ENCODER_DIM in edge_encoder_summary
    assert EDGE_ENCODER_ARGS.EDGE_ENCODER_DROPOUT in edge_encoder_summary
    assert edge_encoder_summary[EDGE_ENCODER_ARGS.EDGE_ENCODER_DIM] == 16
    assert edge_encoder_summary[EDGE_ENCODER_ARGS.EDGE_ENCODER_DROPOUT] == 0.1

    # Verify head config is present in nested structure (Decoder instances have get_summary method)
    assert isinstance(head, Decoder)
    head_summary = summary_dict[MODEL_DEFS.HEAD]
    assert isinstance(head_summary, dict)
    head_config = head.get_summary()
    for key, value in head_config.items():
        assert key in head_summary
        assert head_summary[key] == value

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
