"""
Prediction heads for Napistu-Torch.

This module provides implementations of different prediction heads for various tasks
like edge prediction, node classification, etc. All heads follow a consistent interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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
