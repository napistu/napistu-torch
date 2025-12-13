"""
Prediction heads for Napistu-Torch.

This module provides implementations of different prediction heads for various tasks
like edge prediction, node classification, etc. All heads follow a consistent interface.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from napistu_torch.configs import ModelConfig
from napistu_torch.constants import MODEL_CONFIG
from napistu_torch.models.constants import (
    EDGE_PREDICTION_HEADS,
    HEAD_SPECIFIC_ARGS,
    HEADS,
    MODEL_DEFS,
    RELATION_AWARE_HEADS,
    VALID_HEADS,
)
from napistu_torch.utils.tensor_utils import validate_tensor_for_nan_inf


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

    def __init__(
        self, embedding_dim: int, bias: bool = True, init_as_identity: bool = False
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.bilinear = nn.Bilinear(embedding_dim, embedding_dim, 1, bias=bias)

        if init_as_identity:
            with torch.no_grad():
                # Start with small random weights
                nn.init.xavier_uniform_(self.bilinear.weight, gain=0.01)
                # Add small diagonal component to bias toward dot product
                for i in range(embedding_dim):
                    self.bilinear.weight[0, i, i] += 0.1
        else:
            # Standard initialization
            nn.init.xavier_uniform_(self.bilinear.weight)
        if bias:
            nn.init.zeros_(self.bilinear.bias)

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
        validate_tensor_for_nan_inf(src_embeddings, name="src_embeddings")
        validate_tensor_for_nan_inf(tgt_embeddings, name="tgt_embeddings")

        # Apply bilinear transformation
        edge_scores = self.bilinear(src_embeddings, tgt_embeddings).squeeze(
            -1
        )  # [num_edges]

        # Normalize by embedding dimension
        edge_scores = edge_scores / float(self.embedding_dim)

        # Smooth saturation with tanh, scaled to [-10, 10]
        edge_scores = torch.tanh(edge_scores) * 10.0
        validate_tensor_for_nan_inf(edge_scores, name="edge_scores_final")

        return edge_scores


class DistMultHead(nn.Module):
    """
    DistMult decoder for relation-aware edge prediction.

    Models relations as diagonal matrices: score = <h, r, t> = Σ(h_i * r_i * t_i)

    Parameters
    ----------
    embedding_dim : int
        Dimension of node embeddings from GNN
    num_relations : int
        Number of distinct relation types

    Notes
    -----
    - Simpler than RotatE/TransE (bilinear scoring)
    - WARNING: DistMult is SYMMETRIC (score(h,r,t) = score(t,r,h))
    - Cannot distinguish directed relations (substrate→reaction vs reaction→substrate)
    - Good for undirected or symmetric relations only
    - Included for completeness but may not be ideal for Napistu

    References
    ----------
    Yang et al. "Embedding Entities and Relations for Learning and Inference in
    Knowledge Bases" ICLR 2015.

    Examples
    --------
    >>> # Use only if relations are symmetric
    >>> head = DistMultHead(embedding_dim=256, num_relations=4)
    >>> scores = head(z, edge_index, relation_type)
    """

    def __init__(
        self, embedding_dim: int, num_relations: int, init_as_identity: bool = False
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_relations = num_relations

        self.relation_emb = nn.Embedding(num_relations, embedding_dim)

        if init_as_identity:
            # Initialize to ones: h * 1 * t = h · t
            nn.init.ones_(self.relation_emb.weight)
        else:
            # Initialize around 1 with small noise
            # This gives score ≈ dot product initially
            nn.init.normal_(self.relation_emb.weight, mean=1.0, std=0.1)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        relation_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute edge scores using DistMult.

        Parameters
        ----------
        node_embeddings : torch.Tensor
            Node embeddings from GNN [num_nodes, embedding_dim]
        edge_index : torch.Tensor
            Edge connectivity [2, num_edges]
        relation_type : torch.Tensor
            Relation type for each edge [num_edges]

        Returns
        -------
        torch.Tensor
            Edge scores [num_edges]
        """
        # Get head and tail embeddings
        head = node_embeddings[edge_index[0]]
        tail = node_embeddings[edge_index[1]]

        # Get relation embeddings
        rel = self.relation_emb(relation_type)

        # DistMult scoring: trilinear dot product (use mean for dimension-agnostic)
        score = (head * rel * tail).mean(dim=-1)

        return score


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

        # Compute mean dot product (dimension-agnostic)
        edge_scores = (src_embeddings * tgt_embeddings).mean(dim=1)  # [num_edges]

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


class RelationGatedMLPHead(nn.Module):
    """
    Relation-gated MLP head for edge prediction.

    Uses relation embeddings to modulate edge features via element-wise gating.
    The relation type controls HOW the MLP processes each edge pair.

    Architecture:
    1. Process [src || tgt] through edge MLP → hidden features
    2. Relation embedding → gate values (via Tanh)
    3. Modulate: gated_features = edge_features * relation_gates
    4. Final MLP → edge score

    Parameters
    ----------
    embedding_dim : int
        Dimension of input node embeddings
    num_relations : int
        Number of distinct relation types
    relation_emb_dim : int, optional
        Dimension of relation embeddings, by default 64
    hidden_dim : int, optional
        Hidden layer dimension, by default 128
    num_layers : int, optional
        Number of layers in output MLP, by default 2
    dropout : float, optional
        Dropout probability, by default 0.1
    init_as_identity : bool, optional
        Initialize relation gates near identity (all 1s), by default False

    Notes
    -----
    - Handles imbalanced relation frequencies well (rare relations share parameters)
    - More parameter-efficient than separate MLPs per relation
    - Tanh gating allows both suppression (negative) and amplification (positive)
    - Reuses MLP hyperparameters from EdgeMLPHead for consistency

    Examples
    --------
    >>> head = RelationGatedMLPHead(
    ...     embedding_dim=256,
    ...     num_relations=10,
    ...     relation_emb_dim=64,
    ...     hidden_dim=128
    ... )
    >>> scores = head(node_embeddings, edge_index, relation_type)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_relations: int,
        relation_emb_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        init_as_identity: bool = False,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_relations = num_relations
        self.relation_emb_dim = relation_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Relation embeddings
        self.relation_emb = nn.Embedding(num_relations, relation_emb_dim)

        # Edge processing MLP: [src || tgt] → hidden
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Relation gating: relation_emb → gate values
        self.relation_gate = nn.Sequential(
            nn.Linear(relation_emb_dim, hidden_dim),
            nn.Tanh(),  # [-1, 1] for symmetric modulation
        )

        # Output MLP after gating
        layers = []
        for i in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(hidden_dim, 1))
        self.output_mlp = nn.Sequential(*layers)

        # Initialize
        self._initialize_weights(init_as_identity)

    def _initialize_weights(self, init_as_identity: bool):
        """Initialize weights, optionally starting near identity."""
        if init_as_identity:
            # Initialize relation embeddings to produce gate values near 1.0
            # Tanh(0) = 0, but we want Tanh(x) ≈ 1, so init with positive bias
            nn.init.normal_(self.relation_emb.weight, mean=0.0, std=0.01)
            # Initialize gate network to map to positive values
            with torch.no_grad():
                # Bias the linear layer to output ~0.5, so Tanh(0.5) ≈ 0.46
                self.relation_gate[0].bias.fill_(0.5)
        else:
            # Standard initialization
            nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        relation_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute edge scores using relation-gated MLP.

        Parameters
        ----------
        node_embeddings : torch.Tensor
            Node embeddings [num_nodes, embedding_dim]
        edge_index : torch.Tensor
            Edge connectivity [2, num_edges]
        relation_type : torch.Tensor
            Relation type for each edge [num_edges]

        Returns
        -------
        torch.Tensor
            Edge scores [num_edges]
        """
        # Get source and target embeddings
        src = node_embeddings[edge_index[0]]  # [num_edges, embedding_dim]
        tgt = node_embeddings[edge_index[1]]  # [num_edges, embedding_dim]

        # Process edge through MLP
        edge_features = torch.cat([src, tgt], dim=1)  # [num_edges, 2*embedding_dim]
        edge_hidden = self.edge_mlp(edge_features)  # [num_edges, hidden_dim]

        # Get relation-specific gates
        rel_emb = self.relation_emb(relation_type)  # [num_edges, relation_emb_dim]
        gates = self.relation_gate(
            rel_emb
        )  # [num_edges, hidden_dim], values in [-1, 1]

        # Modulate via element-wise multiplication
        modulated = edge_hidden * gates  # [num_edges, hidden_dim]

        # Final prediction
        score = self.output_mlp(modulated).squeeze(-1)  # [num_edges]

        return score


class RelationAttentionMLPHead(nn.Module):
    """
    Relation-attention MLP head for edge prediction.

    Uses relation embeddings to query edge features via multi-head attention.
    The relation type determines WHICH aspects of edge features to attend to.

    Architecture:
    1. Process [src || tgt] through edge MLP → hidden features
    2. Relation embedding → Query
    3. Edge features → Key, Value
    4. Multi-head attention: relation queries edge features
    5. Residual connection + output MLP → edge score

    Parameters
    ----------
    embedding_dim : int
        Dimension of input node embeddings
    num_relations : int
        Number of distinct relation types
    relation_emb_dim : int, optional
        Dimension of relation embeddings, by default 64
    hidden_dim : int, optional
        Hidden layer dimension (must be divisible by num_attention_heads), by default 128
    num_layers : int, optional
        Number of layers in output MLP, by default 2
    dropout : float, optional
        Dropout probability, by default 0.1
    num_attention_heads : int, optional
        Number of attention heads, by default 4
    init_as_identity : bool, optional
        Initialize to approximate identity (uniform attention), by default False

    Notes
    -----
    - More expressive than gating (can learn complex feature selection)
    - Different heads can specialize (e.g., one for catalysis, one for inhibition)
    - Attention is per-edge over hidden dimensions (not graph-level like GAT)
    - More parameters than gating but potentially better for diverse relation semantics
    - Reuses MLP hyperparameters from EdgeMLPHead for consistency

    Examples
    --------
    >>> head = RelationAttentionMLPHead(
    ...     embedding_dim=256,
    ...     num_relations=10,
    ...     relation_emb_dim=64,
    ...     hidden_dim=128,
    ...     num_attention_heads=4
    ... )
    >>> scores = head(node_embeddings, edge_index, relation_type)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_relations: int,
        relation_emb_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_attention_heads: int = 4,
        init_as_identity: bool = False,
    ):
        super().__init__()

        if hidden_dim % num_attention_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )

        self.embedding_dim = embedding_dim
        self.num_relations = num_relations
        self.relation_emb_dim = relation_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_dim // num_attention_heads

        # Relation embeddings
        self.relation_emb = nn.Embedding(num_relations, relation_emb_dim)

        # Edge processing MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Attention projections
        self.query_proj = nn.Linear(relation_emb_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output MLP after attention
        layers = []
        for i in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(hidden_dim, 1))
        self.output_mlp = nn.Sequential(*layers)

        # Initialize
        self._initialize_weights(init_as_identity)

    def _initialize_weights(self, init_as_identity: bool):
        """Initialize weights, optionally starting near identity."""
        if init_as_identity:
            # Initialize to approximate uniform attention
            nn.init.normal_(self.relation_emb.weight, mean=0.0, std=0.01)
            nn.init.xavier_uniform_(self.query_proj.weight, gain=0.01)
            nn.init.xavier_uniform_(self.key_proj.weight, gain=0.01)
            nn.init.xavier_uniform_(self.value_proj.weight, gain=1.0)
        else:
            # Standard initialization
            nn.init.xavier_uniform_(self.relation_emb.weight)
            nn.init.xavier_uniform_(self.query_proj.weight)
            nn.init.xavier_uniform_(self.key_proj.weight)
            nn.init.xavier_uniform_(self.value_proj.weight)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        relation_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute edge scores using relation-attention MLP.

        Parameters
        ----------
        node_embeddings : torch.Tensor
            Node embeddings [num_nodes, embedding_dim]
        edge_index : torch.Tensor
            Edge connectivity [2, num_edges]
        relation_type : torch.Tensor
            Relation type for each edge [num_edges]

        Returns
        -------
        torch.Tensor
            Edge scores [num_edges]
        """
        # Get source and target embeddings
        src = node_embeddings[edge_index[0]]
        tgt = node_embeddings[edge_index[1]]

        # Process edge
        edge_features = torch.cat([src, tgt], dim=1)
        edge_hidden = self.edge_mlp(edge_features)  # [num_edges, hidden_dim]

        # Get relation embeddings
        rel_emb = self.relation_emb(relation_type)  # [num_edges, relation_emb_dim]

        # Multi-head attention
        batch_size = edge_hidden.size(0)

        # Project to Q, K, V and reshape for multi-head
        Q = self.query_proj(rel_emb).view(
            batch_size, self.num_attention_heads, self.head_dim
        )
        K = self.key_proj(edge_hidden).view(
            batch_size, self.num_attention_heads, self.head_dim
        )
        V = self.value_proj(edge_hidden).view(
            batch_size, self.num_attention_heads, self.head_dim
        )

        # Scaled dot-product attention (per edge, across heads)
        # Q, K, V: [num_edges, num_heads, head_dim]
        scale = torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32, device=Q.device)
        )
        scores = torch.sum(Q * K, dim=-1) / scale  # [num_edges, num_heads]
        attention_weights = torch.softmax(scores, dim=-1)  # [num_edges, num_heads]

        # Apply attention to values
        attended = (
            attention_weights.unsqueeze(-1) * V
        )  # [num_edges, num_heads, head_dim]
        attended = attended.view(batch_size, -1)  # [num_edges, hidden_dim]

        # Residual connection
        output = attended + edge_hidden

        # Final prediction
        score = self.output_mlp(output).squeeze(-1)  # [num_edges]

        return score


class RotatEHead(nn.Module):
    """
    RotatE decoder for relation-aware edge prediction.

    Models relations as rotations in complex space. Given node embeddings from
    a GNN encoder, this head learns relation-specific transformations.

    Scoring function: score = -||h ∘ r - t|| where ∘ is complex multiplication

    Parameters
    ----------
    embedding_dim : int
        Dimension of node embeddings from GNN (must be even for complex space)
    num_relations : int
        Number of distinct relation types in the graph
    margin : float, optional
        Margin for ranking loss, by default 9.0

    Notes
    -----
    - Embedding dimension must be even (split into real/imaginary components)
    - Relation embeddings represent rotation angles in complex space
    - Naturally handles asymmetric relations: r(h→t) ≠ r(t→h)
    - More expressive than TransE but more parameters

    References
    ----------
    Sun et al. "RotatE: Knowledge Graph Embedding by Relational Rotation in
    Complex Space" ICLR 2019.

    Examples
    --------
    >>> # After GNN encoding
    >>> z = gnn_encoder(x, edge_index, edge_attr)  # [num_nodes, 256]
    >>>
    >>> # Score edges with relation types
    >>> head = RotatEHead(embedding_dim=256, num_relations=4)
    >>> scores = head(z, edge_index, relation_type)  # [num_edges]
    """

    def __init__(
        self,
        embedding_dim: int,
        num_relations: int,
        margin: float = 9.0,
        init_as_identity: bool = False,
    ):
        super().__init__()

        if embedding_dim % 2 != 0:
            raise ValueError(
                f"RotatE requires even embedding_dim for complex space, "
                f"got {embedding_dim}"
            )

        self.embedding_dim = embedding_dim
        self.num_relations = num_relations
        self.margin = margin

        # Relation embeddings
        self.relation_emb = nn.Embedding(num_relations, embedding_dim // 2)
        if init_as_identity:
            # Initialize to zero phase (no rotation)
            nn.init.zeros_(self.relation_emb.weight)
        else:
            # Initialize with small random phases
            nn.init.uniform_(self.relation_emb.weight, -0.1, 0.1)

        # Learnable centering parameters (initialize to reasonable defaults)
        self.log_temperature = nn.Parameter(
            torch.tensor(0.0)
        )  # Instead of direct temperature
        self.bias = nn.Parameter(
            torch.tensor(1.0)
        )  # Expected distance for random pairs if unit norm embeddings
        self.eps = 1e-8

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        relation_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute edge scores using RotatE.

        Parameters
        ----------
        node_embeddings : torch.Tensor
            Node embeddings from GNN [num_nodes, embedding_dim]
        edge_index : torch.Tensor
            Edge connectivity [2, num_edges]
        relation_type : torch.Tensor
            Relation type for each edge [num_edges]

        Returns
        -------
        torch.Tensor
            Edge scores [num_edges] (higher = more likely to exist)
        """

        node_embeddings = F.normalize(node_embeddings, p=2, dim=1)

        # Get head and tail embeddings
        head = node_embeddings[edge_index[0]]  # [num_edges, embedding_dim]
        tail = node_embeddings[edge_index[1]]  # [num_edges, embedding_dim]
        validate_tensor_for_nan_inf(head, name="head_embeddings")
        validate_tensor_for_nan_inf(tail, name="tail_embeddings")

        # Get relation phases
        phase_rel = self.relation_emb(relation_type)  # [num_edges, embedding_dim//2]
        validate_tensor_for_nan_inf(phase_rel, name="phase_rel")

        # Split embeddings into real and imaginary components
        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=-1)

        # Normalize phases to [-π, π]
        phase_rel = phase_rel / (self.margin / torch.pi)

        # Convert phase to complex rotation
        re_rel = torch.cos(phase_rel)
        im_rel = torch.sin(phase_rel)

        # Complex multiplication: (h_re + i*h_im) * (r_re + i*r_im)
        re_score = re_head * re_rel - im_head * im_rel
        im_score = re_head * im_rel + im_head * re_rel
        validate_tensor_for_nan_inf(re_score, name="re_score")
        validate_tensor_for_nan_inf(im_score, name="im_score")

        # Compute distance to tail in complex space
        re_diff = re_score - re_tail
        im_diff = im_score - im_tail

        # L2 distance: sqrt(re_diff**2 + im_diff**2)
        # MEAN distance (dimension-agnostic)
        distance = torch.sqrt(re_diff**2 + im_diff**2).mean(dim=-1)
        validate_tensor_for_nan_inf(distance, name="distance")

        # CENTER around zero with learnable parameters
        temp = self.temperature  # Uses property, gets [0.1, 10.0]
        score = -(distance / (temp + self.eps)) + self.bias
        score = score.clamp(-50, 50)  # Prevent BCE overflow
        validate_tensor_for_nan_inf(score, name="score_final")

        return score

    @property
    def temperature(self):
        """Temperature constrained to [0.1, 10.0] range."""
        return torch.exp(self.log_temperature).clamp(0.1, 10.0)


class TransEHead(nn.Module):
    """
    TransE decoder for relation-aware edge prediction.

    Models relations as translations in embedding space: h + r ≈ t
    Simpler than RotatE and often easier to interpret.

    Scoring function: score = -||h + r - t||

    Parameters
    ----------
    embedding_dim : int
        Dimension of node embeddings from GNN
    num_relations : int
        Number of distinct relation types
    margin : float, optional
        Margin for ranking loss, by default 1.0
    norm : int, optional
        Norm to use for distance (1 or 2), by default 2

    Notes
    -----
    - Simpler than RotatE (fewer parameters, easier optimization)
    - Naturally handles asymmetric relations: h+r₁ vs h+r₂
    - May struggle with 1-to-N relations (e.g., one reaction → many products)
    - Good baseline before trying more complex heads

    References
    ----------
    Bordes et al. "Translating Embeddings for Modeling Multi-relational Data"
    NeurIPS 2013.

    Examples
    --------
    >>> head = TransEHead(embedding_dim=256, num_relations=4)
    >>> scores = head(z, edge_index, relation_type)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_relations: int,
        margin: float = 1.0,
        norm: int = 2,
        init_as_identity: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_relations = num_relations
        self.margin = margin
        self.norm = norm

        self.relation_emb = nn.Embedding(num_relations, embedding_dim)

        if init_as_identity:
            # Initialize to zero: h + 0 - t = h - t
            nn.init.zeros_(self.relation_emb.weight)
        else:
            # Standard initialization
            nn.init.xavier_uniform_(self.relation_emb.weight)

        # Learnable centering parameters (initialize to reasonable defaults)
        self.log_temperature = nn.Parameter(
            torch.tensor(0.0)
        )  # Instead of direct temperature
        self.bias = nn.Parameter(
            torch.tensor(1.0)
        )  # Expected distance for random pairs if unit norm embeddings
        self.eps = 1e-8

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        relation_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute edge scores using TransE.

        Parameters
        ----------
        node_embeddings : torch.Tensor
            Node embeddings from GNN [num_nodes, embedding_dim]
        edge_index : torch.Tensor
            Edge connectivity [2, num_edges]
        relation_type : torch.Tensor
            Relation type for each edge [num_edges]

        Returns
        -------
        torch.Tensor
            Edge scores [num_edges] (higher = more likely)
        """
        head = node_embeddings[edge_index[0]]
        tail = node_embeddings[edge_index[1]]
        rel = self.relation_emb(relation_type)

        # TransE scoring: h + r should be close to t
        diff = head + rel - tail

        # Compute norm and normalize by sqrt(dim) for dimension-agnostic scaling
        if self.norm == 1:
            # L1 norm: just take mean of absolute values
            distance = diff.abs().mean(dim=-1)
        else:
            # L2 norm: ||diff|| / sqrt(dim) = sqrt(mean(diff²))
            distance = torch.norm(diff, p=2, dim=-1) / torch.sqrt(
                torch.tensor(float(self.embedding_dim))
            )

        # CENTER around zero with learnable parameters
        temp = self.temperature  # Uses property, gets [0.1, 10.0]
        score = -(distance / (temp + self.eps)) + self.bias
        score = score.clamp(-50, 50)  # Prevent BCE overflow

        return score

    @property
    def temperature(self):
        """Temperature constrained to [0.1, 10.0] range."""
        return torch.exp(self.log_temperature).clamp(0.1, 10.0)


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
    num_relations : int, optional
        Number of relation types (required for relation-aware heads)
    num_classes : int, optional
        Number of output classes for node classification head
    init_head_as_identity : bool, optional
        Whether to initialize the head to approximate an identity transformation, by default False
    mlp_hidden_dim : int, optional
        Hidden layer dimension for MLP head, by default 64
    mlp_num_layers : int, optional
        Number of hidden layers for MLP head, by default 2
    mlp_dropout : float, optional
        Dropout probability for MLP head, by default 0.1
    bilinear_bias : bool, optional
        Whether to add bias term for bilinear head, by default True
    nc_dropout : float, optional
        Dropout probability for node classification head, by default 0.1
    rotate_margin : float, optional
        Margin for RotatE head, by default 9.0
    transe_margin : float, optional
        Margin for TransE head, by default 1.0
    relation_emb_dim: int,
        Dimension of relation embeddings for relation-aware MLP heads, by default 64
    relation_attention_heads: int,
        Number of attention heads for RelationAttentionMLP, by default 4

    Public Methods
    --------------
    config(self) -> Dict[str, Any]:
        Get the configuration dictionary for this decoder.
    from_config(config: ModelConfig, num_relations: Optional[int] = None, num_classes: Optional[int] = None) -> Decoder:
        Create a Decoder from a ModelConfig instance.
    forward(node_embeddings: torch.Tensor, edge_index: Optional[torch.Tensor] = None, relation_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        Forward pass through the head.
    supports_relations(self) -> bool:
        Check if this decoder supports relation-aware heads.
    """

    def __init__(
        self,
        hidden_channels: int,
        head_type: str = HEADS.DOT_PRODUCT,
        num_relations: Optional[int] = None,
        num_classes: Optional[int] = None,
        init_head_as_identity: bool = False,
        mlp_hidden_dim: int = 64,
        mlp_num_layers: int = 2,
        mlp_dropout: float = 0.1,
        bilinear_bias: bool = True,
        nc_dropout: float = 0.1,
        rotate_margin: float = 9.0,
        transe_margin: float = 1.0,
        relation_emb_dim: int = 64,
        relation_attention_heads: int = 4,
    ):
        super().__init__()

        # Store all initialization parameters FIRST (before any validation)
        self._init_args = {
            MODEL_DEFS.HIDDEN_CHANNELS: hidden_channels,
            MODEL_DEFS.HEAD_TYPE: head_type,
            MODEL_DEFS.NUM_RELATIONS: num_relations,
            MODEL_DEFS.NUM_CLASSES: num_classes,
            HEAD_SPECIFIC_ARGS.INIT_HEAD_AS_IDENTITY: init_head_as_identity,
            HEAD_SPECIFIC_ARGS.MLP_HIDDEN_DIM: mlp_hidden_dim,
            HEAD_SPECIFIC_ARGS.MLP_NUM_LAYERS: mlp_num_layers,
            HEAD_SPECIFIC_ARGS.MLP_DROPOUT: mlp_dropout,
            HEAD_SPECIFIC_ARGS.BILINEAR_BIAS: bilinear_bias,
            HEAD_SPECIFIC_ARGS.NC_DROPOUT: nc_dropout,
            HEAD_SPECIFIC_ARGS.ROTATE_MARGIN: rotate_margin,
            HEAD_SPECIFIC_ARGS.TRANSE_MARGIN: transe_margin,
            HEAD_SPECIFIC_ARGS.RELATION_EMB_DIM: relation_emb_dim,
            HEAD_SPECIFIC_ARGS.RELATION_ATTENTION_HEADS: relation_attention_heads,
        }

        self.head_type = head_type
        self.hidden_channels = hidden_channels
        self.num_relations = num_relations

        if head_type not in VALID_HEADS:
            raise ValueError(f"Unknown head: {head_type}. Must be one of {VALID_HEADS}")

        # Validate relation-aware head requirements
        if head_type in RELATION_AWARE_HEADS:
            if num_relations is None:
                raise ValueError(
                    f"num_relations is required for {head_type} head. "
                    f"This should be inferred from edge_strata."
                )
            if head_type == HEADS.ROTATE and hidden_channels % 2 != 0:
                raise ValueError(
                    f"RotatE requires even hidden_channels for complex space, "
                    f"got {hidden_channels}"
                )

        if head_type == HEADS.NODE_CLASSIFICATION:
            if num_classes is None:
                raise ValueError(
                    f"num_classes is required for {head_type} head. "
                    f"This should be inferred from the data."
                )

        # Create the appropriate head based on type
        if head_type == HEADS.BILINEAR:
            self.head = BilinearHead(
                self.hidden_channels, bilinear_bias, init_head_as_identity
            )
        elif head_type == HEADS.DISTMULT:
            self.head = DistMultHead(
                self.hidden_channels, num_relations, init_head_as_identity
            )
        elif head_type == HEADS.DOT_PRODUCT:
            self.head = DotProductHead()
        elif head_type == HEADS.MLP:
            self.head = EdgeMLPHead(
                self.hidden_channels, mlp_hidden_dim, mlp_num_layers, mlp_dropout
            )
        elif head_type == HEADS.NODE_CLASSIFICATION:
            self.head = NodeClassificationHead(
                self.hidden_channels, num_classes, nc_dropout
            )
        elif head_type == HEADS.RELATION_ATTENTION_MLP:
            self.head = RelationAttentionMLPHead(
                self.hidden_channels,
                num_relations,
                relation_emb_dim,
                mlp_hidden_dim,
                mlp_num_layers,
                mlp_dropout,
                relation_attention_heads,
                init_head_as_identity,
            )
        elif head_type == HEADS.RELATION_GATED_MLP:
            self.head = RelationGatedMLPHead(
                self.hidden_channels,
                num_relations,
                relation_emb_dim,
                mlp_hidden_dim,
                mlp_num_layers,
                mlp_dropout,
                init_head_as_identity,
            )
        elif head_type == HEADS.ROTATE:
            self.head = RotatEHead(
                self.hidden_channels,
                num_relations,
                rotate_margin,
                init_head_as_identity,
            )
        elif head_type == HEADS.TRANSE:
            self.head = TransEHead(
                self.hidden_channels,
                num_relations,
                transe_margin,
                init_head_as_identity,
            )
        else:
            raise ValueError(f"Unsupported head type: {head_type}")

    @property
    def config(self) -> Dict[str, Any]:
        """
        Get the configuration dictionary for this decoder.

        Returns a dict containing all initialization parameters needed
        to reconstruct this decoder instance.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary with all __init__ parameters
        """
        return self._init_args.copy()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get decoder metadata summary for checkpointing.

        Returns essential metadata needed to reconstruct the decoder
        from a checkpoint, including ALL parameters that were used.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all initialization parameters,
            with None values filtered out for head-type-specific params
        """
        summary = {}

        # Always include these
        summary[MODEL_DEFS.HEAD] = self._init_args[MODEL_DEFS.HEAD_TYPE]
        summary[MODEL_DEFS.HIDDEN_CHANNELS] = self._init_args[
            MODEL_DEFS.HIDDEN_CHANNELS
        ]

        # Add parameters based on head type
        if self.head_type in RELATION_AWARE_HEADS:
            summary[MODEL_DEFS.NUM_RELATIONS] = self._init_args[
                MODEL_DEFS.NUM_RELATIONS
            ]

        if self.head_type == HEADS.NODE_CLASSIFICATION:
            summary[HEAD_SPECIFIC_ARGS.NUM_CLASSES] = self._init_args[
                HEAD_SPECIFIC_ARGS.NUM_CLASSES
            ]
            summary[HEAD_SPECIFIC_ARGS.NC_DROPOUT] = self._init_args[
                HEAD_SPECIFIC_ARGS.NC_DROPOUT
            ]

        # Head-specific parameters
        if self.head_type == HEADS.MLP:
            summary[HEAD_SPECIFIC_ARGS.MLP_HIDDEN_DIM] = self._init_args[
                HEAD_SPECIFIC_ARGS.MLP_HIDDEN_DIM
            ]
            summary[HEAD_SPECIFIC_ARGS.MLP_NUM_LAYERS] = self._init_args[
                HEAD_SPECIFIC_ARGS.MLP_NUM_LAYERS
            ]
            summary[HEAD_SPECIFIC_ARGS.MLP_DROPOUT] = self._init_args[
                HEAD_SPECIFIC_ARGS.MLP_DROPOUT
            ]
        elif self.head_type == HEADS.BILINEAR:
            summary[HEAD_SPECIFIC_ARGS.BILINEAR_BIAS] = self._init_args[
                HEAD_SPECIFIC_ARGS.BILINEAR_BIAS
            ]
        elif self.head_type == HEADS.ROTATE:
            summary[HEAD_SPECIFIC_ARGS.ROTATE_MARGIN] = self._init_args[
                HEAD_SPECIFIC_ARGS.ROTATE_MARGIN
            ]
        elif self.head_type == HEADS.TRANSE:
            summary[HEAD_SPECIFIC_ARGS.TRANSE_MARGIN] = self._init_args[
                HEAD_SPECIFIC_ARGS.TRANSE_MARGIN
            ]
        if self.head_type in {HEADS.RELATION_GATED_MLP, HEADS.RELATION_ATTENTION_MLP}:
            summary[HEAD_SPECIFIC_ARGS.RELATION_EMB_DIM] = self._init_args[
                HEAD_SPECIFIC_ARGS.RELATION_EMB_DIM
            ]
        if self.head_type == HEADS.RELATION_ATTENTION_MLP:
            summary[HEAD_SPECIFIC_ARGS.RELATION_ATTENTION_HEADS] = self._init_args[
                HEAD_SPECIFIC_ARGS.RELATION_ATTENTION_HEADS
            ]

        return summary

    @property
    def supports_relations(self) -> bool:
        """
        Check if this decoder supports relation-aware heads.

        Returns
        -------
        bool
            True if the head type is in RELATION_AWARE_HEADS, False otherwise
        """
        return self.head_type in RELATION_AWARE_HEADS

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        relation_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the head.

        Parameters
        ----------
        node_embeddings : torch.Tensor
            Node embeddings [num_nodes, embedding_dim]
        edge_index : torch.Tensor, optional
            Edge connectivity [2, num_edges] (required for edge prediction heads)
        relation_type : torch.Tensor, optional
            Relation type for each edge [num_edges] (required for relation-aware heads)

        Returns
        -------
        torch.Tensor
            Head output (edge scores or node predictions)
        """

        # Relation-aware heads require relation_type
        if self.head_type in RELATION_AWARE_HEADS:
            if relation_type is None:
                raise ValueError(
                    f"{self.head_type} head requires relation_type parameter. "
                    f"Make sure relation types are passed to prepare_batch."
                )
            return self.head(node_embeddings, edge_index, relation_type)

        # Edge prediction heads require edge_index
        elif self.head_type in EDGE_PREDICTION_HEADS:
            if edge_index is None:
                raise ValueError(f"edge_index is required for {self.head_type} head")
            return self.head(node_embeddings, edge_index)

        elif self.head_type == HEADS.NODE_CLASSIFICATION:
            # Node classification head doesn't need edge_index
            return self.head(node_embeddings)
        else:
            raise ValueError(f"Unsupported head type: {self.head_type}")

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        num_relations: Optional[int] = None,
        num_classes: Optional[int] = None,
    ):
        """
        Create a Decoder from a configuration object.

        Parameters
        ----------
        config : ModelConfig
            Configuration object containing head parameters
        num_relations : int, optional
            Number of relation types (required for relation-aware heads).
            This should be inferred from edge_strata.
        num_classes : int, optional
            Number of output classes for node classification head (required for node classification head).
            This should be inferred from the data.

        Returns
        -------
        Decoder
            Configured head decoder
        """
        # Extract head-specific parameters from config
        head_kwargs = {
            MODEL_DEFS.HIDDEN_CHANNELS: getattr(config, MODEL_DEFS.HIDDEN_CHANNELS),
            MODEL_DEFS.HEAD_TYPE: getattr(config, MODEL_CONFIG.HEAD),
            MODEL_DEFS.NUM_RELATIONS: num_relations,
            MODEL_DEFS.NUM_CLASSES: num_classes,
            HEAD_SPECIFIC_ARGS.INIT_HEAD_AS_IDENTITY: getattr(
                config,
                HEAD_SPECIFIC_ARGS.INIT_HEAD_AS_IDENTITY,
            ),
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
            HEAD_SPECIFIC_ARGS.NC_DROPOUT: getattr(
                config, HEAD_SPECIFIC_ARGS.NC_DROPOUT
            ),
            HEAD_SPECIFIC_ARGS.ROTATE_MARGIN: getattr(
                config, HEAD_SPECIFIC_ARGS.ROTATE_MARGIN
            ),
            HEAD_SPECIFIC_ARGS.TRANSE_MARGIN: getattr(
                config, HEAD_SPECIFIC_ARGS.TRANSE_MARGIN
            ),
            HEAD_SPECIFIC_ARGS.RELATION_EMB_DIM: getattr(
                config, HEAD_SPECIFIC_ARGS.RELATION_EMB_DIM
            ),
            HEAD_SPECIFIC_ARGS.RELATION_ATTENTION_HEADS: getattr(
                config, HEAD_SPECIFIC_ARGS.RELATION_ATTENTION_HEADS
            ),
        }

        return cls(**head_kwargs)
