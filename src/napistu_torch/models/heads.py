"""
Prediction heads for Napistu-Torch.

This module provides implementations of different prediction heads for various tasks
like edge prediction, node classification, etc. All heads follow a consistent interface.
"""

from typing import Optional

import torch
import torch.nn as nn

from napistu_torch.constants import MODEL_CONFIG
from napistu_torch.models.constants import (
    HEAD_SPECIFIC_ARGS,
    HEADS,
    MODEL_DEFS,
    VALID_HEADS,
)


class DotProductHead(nn.Module):
    """
    Dot product head for edge prediction.

    Computes edge scores as the dot product of source and target node embeddings.
    This is the simplest and most efficient head for edge prediction tasks.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, node_embeddings: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute edge scores using dot product.

        Parameters
        ----------
        node_embeddings : torch.Tensor
            Node embeddings [num_nodes, embedding_dim]
        edge_index : torch.Tensor
            Edge connectivity [2, num_edges]

        Returns
        -------
        torch.Tensor
            Edge scores [num_edges]
        """
        # Get source and target node embeddings
        src_embeddings = node_embeddings[edge_index[0]]  # [num_edges, embedding_dim]
        tgt_embeddings = node_embeddings[edge_index[1]]  # [num_edges, embedding_dim]

        # Compute dot product
        edge_scores = torch.sum(src_embeddings * tgt_embeddings, dim=1)  # [num_edges]

        return edge_scores


class EdgeMLPHead(nn.Module):
    """
    Multi-layer perceptron head for edge prediction.

    Uses an MLP to predict edge scores from concatenated source and target embeddings.
    More expressive than dot product but requires more parameters.

    Parameters
    ----------
    embedding_dim : int
        Dimension of input node embeddings
    hidden_dim : int, optional
        Hidden layer dimension, by default 64
    num_layers : int, optional
        Number of hidden layers, by default 2
    dropout : float, optional
        Dropout probability, by default 0.1
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Build MLP layers
        layers = []
        input_dim = 2 * embedding_dim  # Concatenated source and target embeddings

        # Hidden layers
        for i in range(num_layers):
            output_dim = hidden_dim if i < num_layers - 1 else 1
            layers.append(nn.Linear(input_dim, output_dim))
            if i < num_layers - 1:  # Don't add activation to last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

    def forward(
        self, node_embeddings: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute edge scores using MLP.

        Parameters
        ----------
        node_embeddings : torch.Tensor
            Node embeddings [num_nodes, embedding_dim]
        edge_index : torch.Tensor
            Edge connectivity [2, num_edges]

        Returns
        -------
        torch.Tensor
            Edge scores [num_edges]
        """
        # Get source and target node embeddings
        src_embeddings = node_embeddings[edge_index[0]]  # [num_edges, embedding_dim]
        tgt_embeddings = node_embeddings[edge_index[1]]  # [num_edges, embedding_dim]

        # Concatenate embeddings
        edge_features = torch.cat(
            [src_embeddings, tgt_embeddings], dim=1
        )  # [num_edges, 2*embedding_dim]

        # Apply MLP
        edge_scores = self.mlp(edge_features).squeeze(-1)  # [num_edges]

        return edge_scores


class BilinearHead(nn.Module):
    """
    Bilinear head for edge prediction.

    Uses a bilinear transformation to compute edge scores:
    score = src_emb^T * W * tgt_emb

    More expressive than dot product but more efficient than MLP.

    Parameters
    ----------
    embedding_dim : int
        Dimension of input node embeddings
    bias : bool, optional
        Whether to add bias term, by default True
    """

    def __init__(self, embedding_dim: int, bias: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.bilinear = nn.Bilinear(embedding_dim, embedding_dim, 1, bias=bias)

    def forward(
        self, node_embeddings: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute edge scores using bilinear transformation.

        Parameters
        ----------
        node_embeddings : torch.Tensor
            Node embeddings [num_nodes, embedding_dim]
        edge_index : torch.Tensor
            Edge connectivity [2, num_edges]

        Returns
        -------
        torch.Tensor
            Edge scores [num_edges]
        """
        # Get source and target node embeddings
        src_embeddings = node_embeddings[edge_index[0]]  # [num_edges, embedding_dim]
        tgt_embeddings = node_embeddings[edge_index[1]]  # [num_edges, embedding_dim]

        # Apply bilinear transformation
        edge_scores = self.bilinear(src_embeddings, tgt_embeddings).squeeze(
            -1
        )  # [num_edges]

        return edge_scores


class NodeClassificationHead(nn.Module):
    """
    Simple linear head for node classification tasks.

    Parameters
    ----------
    embedding_dim : int
        Dimension of input node embeddings
    num_classes : int
        Number of output classes
    dropout : float, optional
        Dropout probability, by default 0.1
    """

    def __init__(self, embedding_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute node class predictions.

        Parameters
        ----------
        node_embeddings : torch.Tensor
            Node embeddings [num_nodes, embedding_dim]

        Returns
        -------
        torch.Tensor
            Node class logits [num_nodes, num_classes]
        """
        x = self.dropout(node_embeddings)
        logits = self.classifier(x)
        return logits


class Decoder(nn.Module):
    """
    Unified head decoder that can create different types of prediction heads.

    This class provides a single interface for creating various head types
    (dot product, MLP, bilinear, node classification) with a from_config
    classmethod for easy integration with configuration systems.

    Parameters
    ----------
    hidden_channels : int
        Dimension of input node embeddings (should match GNN encoder output)
    head_type : str
        Type of head to create (dot_product, mlp, bilinear, node_classification)
    mlp_hidden_dim : int, optional
        Hidden layer dimension for MLP head, by default 64
    mlp_num_layers : int, optional
        Number of hidden layers for MLP head, by default 2
    mlp_dropout : float, optional
        Dropout probability for MLP head, by default 0.1
    bilinear_bias : bool, optional
        Whether to add bias term for bilinear head, by default True
    nc_num_classes : int, optional
        Number of output classes for node classification head, by default 2
    nc_dropout : float, optional
        Dropout probability for node classification head, by default 0.1
    """

    def __init__(
        self,
        hidden_channels: int,
        head_type: str = HEADS.DOT_PRODUCT,
        mlp_hidden_dim: int = 64,
        mlp_num_layers: int = 2,
        mlp_dropout: float = 0.1,
        bilinear_bias: bool = True,
        nc_num_classes: int = 2,
        nc_dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.head_type = head_type

        if head_type not in VALID_HEADS:
            raise ValueError(f"Unknown head: {head_type}. Must be one of {VALID_HEADS}")

        # Create the appropriate head based on type
        if head_type == HEADS.DOT_PRODUCT:
            self.head = DotProductHead()
        elif head_type == HEADS.MLP:
            self.head = EdgeMLPHead(
                self.hidden_channels, mlp_hidden_dim, mlp_num_layers, mlp_dropout
            )
        elif head_type == HEADS.BILINEAR:
            self.head = BilinearHead(self.hidden_channels, bilinear_bias)
        elif head_type == HEADS.NODE_CLASSIFICATION:
            self.head = NodeClassificationHead(
                self.hidden_channels, nc_num_classes, nc_dropout
            )
        else:
            raise ValueError(f"Unsupported head type: {head_type}")

    def forward(
        self, node_embeddings: torch.Tensor, edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the head.

        Parameters
        ----------
        node_embeddings : torch.Tensor
            Node embeddings [num_nodes, embedding_dim]
        edge_index : torch.Tensor, optional
            Edge connectivity [2, num_edges] (required for edge prediction heads)

        Returns
        -------
        torch.Tensor
            Head output (edge scores or node predictions)
        """
        if self.head_type in [HEADS.DOT_PRODUCT, HEADS.MLP, HEADS.BILINEAR]:
            if edge_index is None:
                raise ValueError(f"edge_index is required for {self.head_type} head")
            return self.head(node_embeddings, edge_index)
        elif self.head_type == HEADS.NODE_CLASSIFICATION:
            # Node classification head doesn't need edge_index
            return self.head(node_embeddings)
        else:
            raise ValueError(f"Unsupported head type: {self.head_type}")

    @classmethod
    def from_config(cls, config):
        """
        Create a Decoder from a configuration object.

        Parameters
        ----------
        config : ModelConfig
            Configuration object containing head parameters

        Returns
        -------
        Decoder
            Configured head decoder
        """
        # Extract head-specific parameters from config
        head_kwargs = {
            MODEL_DEFS.HIDDEN_CHANNELS: getattr(config, MODEL_DEFS.HIDDEN_CHANNELS),
            MODEL_DEFS.HEAD_TYPE: getattr(config, MODEL_CONFIG.HEAD),
            HEAD_SPECIFIC_ARGS.MLP_HIDDEN_DIM: getattr(
                config, HEAD_SPECIFIC_ARGS.MLP_HIDDEN_DIM
            ),
            HEAD_SPECIFIC_ARGS.MLP_NUM_LAYERS: getattr(
                config, HEAD_SPECIFIC_ARGS.MLP_NUM_LAYERS
            ),
            HEAD_SPECIFIC_ARGS.MLP_DROPOUT: getattr(
                config, HEAD_SPECIFIC_ARGS.MLP_DROPOUT
            ),
            HEAD_SPECIFIC_ARGS.BILINEAR_BIAS: getattr(
                config, HEAD_SPECIFIC_ARGS.BILINEAR_BIAS
            ),
            HEAD_SPECIFIC_ARGS.NC_NUM_CLASSES: getattr(
                config, HEAD_SPECIFIC_ARGS.NC_NUM_CLASSES
            ),
            HEAD_SPECIFIC_ARGS.NC_DROPOUT: getattr(
                config, HEAD_SPECIFIC_ARGS.NC_DROPOUT
            ),
        }

        return cls(**head_kwargs)
