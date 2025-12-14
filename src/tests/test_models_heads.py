"""Tests for decoders / heads."""

import pytest
import torch

from napistu_torch.configs import ModelConfig
from napistu_torch.models.constants import (
    EDGE_PREDICTION_HEADS,
    HEADS,
    RELATION_AWARE_HEADS,
)
from napistu_torch.models.heads import (
    Decoder,
    DistMultHead,
    DotProductHead,
    EdgeMLPHead,
    NodeClassificationHead,
    RotatEHead,
    TransEHead,
)


class TestRotatEHead:
    """Tests for RotatE head."""

    def test_initialization(self):
        """Test RotatE head initializes correctly."""
        head = RotatEHead(embedding_dim=256, num_relations=4, margin=9.0)
        assert head.embedding_dim == 256
        assert head.num_relations == 4
        assert head.margin == 9.0

    def test_odd_embedding_dim_raises(self):
        """Test that odd embedding_dim raises error."""
        with pytest.raises(ValueError, match="even"):
            RotatEHead(embedding_dim=255, num_relations=4)

    def test_forward(self):
        """Test forward pass produces correct output shape."""
        head = RotatEHead(embedding_dim=256, num_relations=4)

        # Create dummy inputs
        node_embeddings = torch.randn(100, 256)
        edge_index = torch.randint(0, 100, (2, 50))
        relation_type = torch.randint(0, 4, (50,))

        # Forward pass
        scores = head(node_embeddings, edge_index, relation_type)

        assert scores.shape == (50,)
        assert not torch.isnan(scores).any()

    def test_relation_embeddings_shape(self):
        """Test relation embeddings have correct shape."""
        head = RotatEHead(embedding_dim=256, num_relations=4)

        # Relation embeddings should be half the node embedding size (complex space)
        assert head.relation_emb.weight.shape == (4, 128)


class TestTransEHead:
    """Tests for TransE head."""

    def test_initialization(self):
        """Test TransE head initializes correctly."""
        head = TransEHead(embedding_dim=256, num_relations=4, margin=1.0)
        assert head.embedding_dim == 256
        assert head.num_relations == 4
        assert head.margin == 1.0

    def test_forward(self):
        """Test forward pass."""
        head = TransEHead(embedding_dim=256, num_relations=4)

        node_embeddings = torch.randn(100, 256)
        edge_index = torch.randint(0, 100, (2, 50))
        relation_type = torch.randint(0, 4, (50,))

        scores = head(node_embeddings, edge_index, relation_type)

        assert scores.shape == (50,)
        assert not torch.isnan(scores).any()

    def test_different_norms(self):
        """Test that different norm values work."""
        head_l1 = TransEHead(embedding_dim=256, num_relations=4, norm=1)
        head_l2 = TransEHead(embedding_dim=256, num_relations=4, norm=2)

        node_embeddings = torch.randn(100, 256)
        edge_index = torch.randint(0, 100, (2, 50))
        relation_type = torch.randint(0, 4, (50,))

        scores_l1 = head_l1(node_embeddings, edge_index, relation_type)
        scores_l2 = head_l2(node_embeddings, edge_index, relation_type)

        # Scores should be different for different norms
        assert not torch.allclose(scores_l1, scores_l2)


class TestDistMultHead:
    """Tests for DistMult head."""

    def test_initialization(self):
        """Test DistMult head initializes correctly."""
        head = DistMultHead(embedding_dim=256, num_relations=4)
        assert head.embedding_dim == 256
        assert head.num_relations == 4

    def test_forward(self):
        """Test forward pass."""
        head = DistMultHead(embedding_dim=256, num_relations=4)

        node_embeddings = torch.randn(100, 256)
        edge_index = torch.randint(0, 100, (2, 50))
        relation_type = torch.randint(0, 4, (50,))

        scores = head(node_embeddings, edge_index, relation_type)

        assert scores.shape == (50,)

    def test_symmetry(self):
        """Test that DistMult is symmetric (score(h,r,t) = score(t,r,h))."""
        head = DistMultHead(embedding_dim=256, num_relations=4)

        node_embeddings = torch.randn(100, 256)

        # Create edge (A -> B)
        edge_forward = torch.tensor([[10], [20]])
        edge_backward = torch.tensor([[20], [10]])
        relation_type = torch.tensor([0])

        score_forward = head(node_embeddings, edge_forward, relation_type)
        score_backward = head(node_embeddings, edge_backward, relation_type)

        # Should be equal (symmetric)
        assert torch.allclose(score_forward, score_backward, atol=1e-6)


class TestDecoderWithRelationAware:
    """Tests for Decoder with relation-aware heads."""

    def test_decoder_requires_num_relations(self):
        """Test that relation-aware heads require num_relations."""
        with pytest.raises(ValueError, match="num_relations is required"):
            Decoder(hidden_channels=256, head_type=HEADS.ROTATE)

    def test_decoder_rotate_requires_even_hidden(self):
        """Test that RotatE requires even hidden_channels."""
        with pytest.raises(ValueError, match="even hidden_channels"):
            Decoder(hidden_channels=255, head_type=HEADS.ROTATE, num_relations=4)

    def test_decoder_forward_without_relation_type_raises(self):
        """Test that forward without relation_type raises error."""
        decoder = Decoder(hidden_channels=256, head_type=HEADS.ROTATE, num_relations=4)

        node_embeddings = torch.randn(100, 256)
        edge_index = torch.randint(0, 100, (2, 50))

        with pytest.raises(ValueError, match="requires relation_type parameter"):
            decoder(node_embeddings, edge_index)

    def test_decoder_forward_with_relation_type(self):
        """Test that forward with relation_type works."""
        decoder = Decoder(hidden_channels=256, head_type=HEADS.ROTATE, num_relations=4)

        node_embeddings = torch.randn(100, 256)
        edge_index = torch.randint(0, 100, (2, 50))
        relation_type = torch.randint(0, 4, (50,))

        scores = decoder(node_embeddings, edge_index, relation_type)

        assert scores.shape == (50,)

    def test_all_relation_aware_heads(self):
        """Test that all relation-aware heads work through Decoder."""
        for head_type in RELATION_AWARE_HEADS:
            # Skip RotatE with odd dimensions
            hidden_channels = 256 if head_type == HEADS.ROTATE else 255

            if head_type == HEADS.ROTATE and hidden_channels % 2 != 0:
                continue

            decoder = Decoder(
                hidden_channels=hidden_channels, head_type=head_type, num_relations=4
            )

            node_embeddings = torch.randn(100, hidden_channels)
            edge_index = torch.randint(0, 100, (2, 50))
            relation_type = torch.randint(0, 4, (50,))

            scores = decoder(node_embeddings, edge_index, relation_type)

            assert scores.shape == (50,)
            assert not torch.isnan(scores).any()


class TestDotProductHead:
    """Tests for DotProduct head."""

    def test_initialization(self):
        """Test DotProduct head initializes correctly."""
        head = DotProductHead()
        assert head is not None

    def test_forward(self):
        """Test forward pass produces correct output shape."""
        head = DotProductHead()

        node_embeddings = torch.randn(100, 256)
        edge_index = torch.randint(0, 100, (2, 50))

        scores = head(node_embeddings, edge_index)

        assert scores.shape == (50,)
        assert not torch.isnan(scores).any()


class TestEdgeMLPHead:
    """Tests for EdgeMLP head."""

    def test_initialization(self):
        """Test EdgeMLP head initializes correctly."""
        head = EdgeMLPHead(embedding_dim=256, hidden_dim=64, num_layers=2, dropout=0.1)
        assert head.embedding_dim == 256
        assert head.hidden_dim == 64
        assert head.num_layers == 2
        assert head.dropout == 0.1

    def test_forward(self):
        """Test forward pass produces correct output shape."""
        head = EdgeMLPHead(embedding_dim=256)

        node_embeddings = torch.randn(100, 256)
        edge_index = torch.randint(0, 100, (2, 50))

        scores = head(node_embeddings, edge_index)

        assert scores.shape == (50,)
        assert not torch.isnan(scores).any()

    def test_different_configurations(self):
        """Test that different MLP configurations work."""
        head1 = EdgeMLPHead(embedding_dim=256, hidden_dim=32, num_layers=1)
        head2 = EdgeMLPHead(embedding_dim=256, hidden_dim=128, num_layers=3)

        node_embeddings = torch.randn(100, 256)
        edge_index = torch.randint(0, 100, (2, 50))

        scores1 = head1(node_embeddings, edge_index)
        scores2 = head2(node_embeddings, edge_index)

        assert scores1.shape == (50,)
        assert scores2.shape == (50,)


class TestNodeClassificationHead:
    """Tests for NodeClassification head."""

    def test_initialization(self):
        """Test NodeClassification head initializes correctly."""
        head = NodeClassificationHead(embedding_dim=256, num_classes=10, dropout=0.1)
        assert head.classifier.in_features == 256
        assert head.classifier.out_features == 10

    def test_forward(self):
        """Test forward pass produces correct output shape."""
        head = NodeClassificationHead(embedding_dim=256, num_classes=5)

        node_embeddings = torch.randn(100, 256)

        logits = head(node_embeddings)

        assert logits.shape == (100, 5)
        assert not torch.isnan(logits).any()

    def test_different_num_classes(self):
        """Test that different numbers of classes work."""
        head = NodeClassificationHead(embedding_dim=256, num_classes=2)

        node_embeddings = torch.randn(50, 256)
        logits = head(node_embeddings)

        assert logits.shape == (50, 2)


class TestDecoderWithEdgePrediction:
    """Tests for Decoder with edge prediction heads."""

    def test_decoder_forward_without_edge_index_raises(self):
        """Test that forward without edge_index raises error for edge prediction heads."""
        for head_type in [HEADS.DOT_PRODUCT, HEADS.MLP]:
            decoder = Decoder(hidden_channels=256, head_type=head_type)

            node_embeddings = torch.randn(100, 256)

            with pytest.raises(ValueError, match="edge_index is required"):
                decoder(node_embeddings)

    def test_decoder_forward_with_edge_index(self):
        """Test that forward with edge_index works for edge prediction heads."""
        for head_type in [HEADS.DOT_PRODUCT, HEADS.MLP]:
            decoder = Decoder(hidden_channels=256, head_type=head_type)

            node_embeddings = torch.randn(100, 256)
            edge_index = torch.randint(0, 100, (2, 50))

            scores = decoder(node_embeddings, edge_index)

            assert scores.shape == (50,)
            assert not torch.isnan(scores).any()

    def test_all_edge_prediction_heads(self):
        """Test that all edge prediction heads work through Decoder."""
        for head_type in EDGE_PREDICTION_HEADS:
            if head_type in RELATION_AWARE_HEADS:
                # Skip relation-aware heads (tested separately)
                continue

            decoder = Decoder(hidden_channels=256, head_type=head_type)

            node_embeddings = torch.randn(100, 256)
            edge_index = torch.randint(0, 100, (2, 50))

            scores = decoder(node_embeddings, edge_index)

            assert scores.shape == (50,)
            assert not torch.isnan(scores).any()


class TestDecoderWithNodeClassification:
    """Tests for Decoder with node classification head."""

    def test_decoder_forward_node_classification(self):
        """Test that forward works for node classification head."""
        decoder = Decoder(
            hidden_channels=256, head_type=HEADS.NODE_CLASSIFICATION, num_classes=10
        )

        node_embeddings = torch.randn(100, 256)

        logits = decoder(node_embeddings)

        assert logits.shape == (100, 10)
        assert not torch.isnan(logits).any()

    def test_decoder_node_classification_different_classes(self):
        """Test node classification with different numbers of classes."""
        for num_classes in [2, 5, 10]:
            decoder = Decoder(
                hidden_channels=256,
                head_type=HEADS.NODE_CLASSIFICATION,
                num_classes=num_classes,
            )

            node_embeddings = torch.randn(50, 256)
            logits = decoder(node_embeddings)

            assert logits.shape == (50, num_classes)


class TestConfigIntegration:
    """Test config-based instantiation."""

    def test_from_config_rotate(self):
        """Test creating RotatE head from config."""
        config = ModelConfig(
            encoder="gcn",
            head="rotate",
            hidden_channels=256,
            rotate_margin=9.0,
        )

        decoder = Decoder.from_config(config, num_relations=4)

        assert decoder.head_type == HEADS.ROTATE
        assert decoder.hidden_channels == 256

    def test_from_config_transe(self):
        """Test creating TransE head from config."""
        config = ModelConfig(
            encoder="gcn",
            head="transe",
            hidden_channels=256,
            transe_margin=1.0,
        )

        decoder = Decoder.from_config(config, num_relations=4)

        assert decoder.head_type == HEADS.TRANSE
        assert decoder.hidden_channels == 256

    def test_from_config_dot_product(self):
        """Test creating DotProduct head from config."""
        config = ModelConfig(
            encoder="gcn",
            head="dot_product",
            hidden_channels=256,
        )

        decoder = Decoder.from_config(config)

        assert decoder.head_type == HEADS.DOT_PRODUCT
        assert decoder.hidden_channels == 256

    def test_from_config_mlp(self):
        """Test creating MLP head from config."""
        config = ModelConfig(
            encoder="gcn",
            head="mlp",
            hidden_channels=256,
            mlp_hidden_dim=64,
            mlp_num_layers=2,
            mlp_dropout=0.1,
        )

        decoder = Decoder.from_config(config)

        assert decoder.head_type == HEADS.MLP
        assert decoder.hidden_channels == 256

        assert decoder.hidden_channels == 256

    def test_from_config_node_classification(self):
        """Test creating NodeClassification head from config."""
        config = ModelConfig(
            encoder="gcn",
            head="node_classification",
            hidden_channels=256,
            nc_dropout=0.1,
        )

        decoder = Decoder.from_config(config, num_classes=5)

        assert decoder.head_type == HEADS.NODE_CLASSIFICATION
        assert decoder.hidden_channels == 256
