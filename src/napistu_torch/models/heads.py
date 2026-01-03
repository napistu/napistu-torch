"""
Prediction heads for Napistu-Torch.

This module provides implementations of different prediction heads for various tasks
like edge prediction, node classification, etc. All heads follow a consistent interface.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from napistu_torch.configs import ModelConfig
from napistu_torch.constants import MODEL_CONFIG
from napistu_torch.ml.constants import LOSSES
from napistu_torch.models.constants import (
    EDGE_PREDICTION_HEADS,
    HEAD_DEFS,
    HEAD_SPECIFIC_ARGS,
    HEADS,
    MODEL_DEFS,
    RELATION_AWARE_HEADS,
    VALID_HEADS,
)
from napistu_torch.models.head_utils import (
    compute_rotate_distance,
    normalized_distances_to_probs,
    validate_symmetric_relation_indices,
)


class AttentionHead(nn.Module):
    """
    Lightweight attention head for edge prediction.

    Projects nodes to query/key spaces and computes scaled dot-product attention.
    More expressive than dot product but much lighter than full MLP with attention.

    Architecture:
    1. Project source nodes → query space
    2. Project target nodes → key space
    3. Compute scaled dot product: (W_q @ src)^T @ (W_k @ tgt) / sqrt(d)

    Parameters
    ----------
    embedding_dim : int
        Dimension of input node embeddings
    attention_dim : int, optional
        Dimension of attention space (lower = more compression), by default 64
    init_as_identity : bool, optional
        Initialize projections to approximate identity (dot product), by default False

    Notes
    -----
    - Projects to lower dimension for efficiency and regularization
    - Learns separate query/key transformations (more flexible than dot product)
    - Scaled dot product prevents gradient vanishing in high dimensions
    - Normalizes embeddings for numerical stability
    - ~2 * embedding_dim * attention_dim parameters (e.g., ~16K for 128→64)

    Comparison to other heads:
    - vs DotProduct: Learns transformations (more expressive)
    - vs MLP: Much fewer parameters, easier to interpret
    - vs RelationAttention: No relation-specific modulation

    Examples
    --------
    >>> head = AttentionHead(embedding_dim=128, attention_dim=64)
    >>> scores = head(node_embeddings, edge_index)
    >>> # scores ∈ ℝ^{num_edges}, approximately normalized by scaling
    """

    loss_type = LOSSES.BCE

    def __init__(
        self,
        embedding_dim: int,
        attention_dim: int = 64,
        init_as_identity: bool = False,
    ):
        super().__init__()

        if attention_dim > embedding_dim:
            raise ValueError(
                f"attention_dim ({attention_dim}) should not exceed "
                f"embedding_dim ({embedding_dim}) for regularization"
            )

        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim

        # Query and key projections (no bias for simplicity)
        self.query_proj = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.key_proj = nn.Linear(embedding_dim, attention_dim, bias=False)

        # Scaling factor for numerical stability (standard attention scaling)
        self.scale = 1.0 / (attention_dim**0.5)

        # Initialize weights
        self._initialize_weights(init_as_identity)

    def _initialize_weights(self, init_as_identity: bool):
        """Initialize projection weights."""
        if init_as_identity:
            # Initialize to approximate identity transformation
            # For rectangular matrices, use pseudo-identity
            with torch.no_grad():
                # Query and Key start as identity-like (first attention_dim dims)
                nn.init.eye_(self.query_proj.weight)
                nn.init.eye_(self.key_proj.weight)

                # Add small noise for symmetry breaking
                self.query_proj.weight.add_(
                    torch.randn_like(self.query_proj.weight) * 0.01
                )
                self.key_proj.weight.add_(torch.randn_like(self.key_proj.weight) * 0.01)
        else:
            # Standard Xavier initialization with small gain
            nn.init.xavier_uniform_(self.query_proj.weight, gain=0.1)
            nn.init.xavier_uniform_(self.key_proj.weight, gain=0.1)

    def forward(
        self, node_embeddings: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention-based edge scores.

        Parameters
        ----------
        node_embeddings : torch.Tensor
            Node embeddings from encoder [num_nodes, embedding_dim]
        edge_index : torch.Tensor
            Edge connectivity [2, num_edges]

        Returns
        -------
        torch.Tensor
            Edge scores [num_edges]

        Notes
        -----
        Scores are computed as:
            score = (W_q @ normalize(src))^T @ (W_k @ normalize(tgt)) / sqrt(d_attn)

        Normalization ensures embeddings have bounded norms, preventing
        score explosion with pretrained encoders.
        """
        # Normalize embeddings to unit norm (critical for stability)
        node_embeddings = F.normalize(node_embeddings, p=2, dim=1)

        # Project all nodes to query and key spaces
        # More efficient than projecting per edge
        queries = self.query_proj(node_embeddings)  # [num_nodes, attention_dim]
        keys = self.key_proj(node_embeddings)  # [num_nodes, attention_dim]

        # Get edge-specific queries and keys
        src_queries = queries[edge_index[0]]  # [num_edges, attention_dim]
        tgt_keys = keys[edge_index[1]]  # [num_edges, attention_dim]

        # Scaled dot-product attention (per edge)
        # Element-wise multiply then sum over attention dimension
        scores = (src_queries * tgt_keys).sum(dim=1) * self.scale  # [num_edges]

        return scores


class ConditionalRotatEHead(nn.Module):
    """
    Conditional decoder: DotProduct for symmetric relations, RotatE for asymmetric.

    Automatically routes different relation types to appropriate scoring functions:
    - Symmetric relations (e.g., "protein->protein"): DotProduct distance
    - Asymmetric relations (e.g., "catalyst->modified"): RotatE rotation distance

    Both heads produce distance-based scores in [-2, 0] for margin loss.

    Parameters
    ----------
    embedding_dim : int
        Dimension of node embeddings (must be even for RotatE complex embeddings)
    num_relations : int
        Total number of relation types
    symmetric_relation_indices : List[int]
        Indices of relations that should use dot product (symmetric).
        All other relations use RotatE (asymmetric).
        Obtained from NapistuData.analyze_relation_symmetry()
    init_asymmetric_as_identity : bool, optional
        Initialize RotatE phases to 0 (identity rotation), by default False
    margin : float, optional
        Margin for ranking loss (applied to both heads), by default 9.0

    Notes
    -----
    **Score Ranges (with normalized embeddings):**
    Both heads produce scores in [-2, 0]:
    - DotProduct: distance = 1 - similarity, score = -distance
    - RotatE: distance = ||h⊙r - t||, score = -distance

    **When to Use:**
    - Graph has mix of symmetric (A↔B) and asymmetric (A→B) relations
    - Example: protein-protein interactions (symmetric) + catalysis (asymmetric)

    **When NOT to Use:**
    - All relations symmetric → Use DotProduct or DistMult instead
    - All relations asymmetric → Use RotatE or TransE instead
    """

    loss_type = LOSSES.MARGIN

    def __init__(
        self,
        embedding_dim: int,
        num_relations: int,
        symmetric_relation_indices: List[int],
        init_asymmetric_as_identity: bool = HEAD_DEFS.DEFAULT_INIT_HEAD_AS_IDENTITY,
        margin: float = HEAD_DEFS.DEFAULT_ROTATE_MARGIN,
    ):
        super().__init__()

        if embedding_dim % 2 != 0:
            raise ValueError(
                f"embedding_dim must be even for RotatE component, got {embedding_dim}"
            )

        # Validate symmetric relation indices
        validate_symmetric_relation_indices(symmetric_relation_indices, num_relations)

        self.embedding_dim = embedding_dim
        self.num_relations = num_relations
        self.margin = margin
        self.eps = 1e-8

        # Partition relations into symmetric vs asymmetric
        self.symmetric_relations = set(symmetric_relation_indices)
        self.asymmetric_relations = [
            i for i in range(num_relations) if i not in self.symmetric_relations
        ]

        # Create relation routing (as buffer so it's device-aware and saved in state_dict)
        self.register_buffer(
            "is_symmetric", torch.zeros(num_relations, dtype=torch.bool)
        )
        for idx in symmetric_relation_indices:
            self.is_symmetric[idx] = True

        # Create full embedding table for all relations
        # Symmetric relations will be initialized to NaN (unused)
        # Asymmetric relations will be initialized for RotatE
        self.relation_emb = nn.Embedding(
            num_relations,
            embedding_dim // 2,  # Complex space: half dimension for re, half for im
        )

        # Initialize embeddings
        with torch.no_grad():
            # Initialize symmetric relations to -1 (unused)
            for idx in symmetric_relation_indices:
                self.relation_emb.weight[idx] = float(-1.0)

            # Initialize asymmetric relations for RotatE
            if init_asymmetric_as_identity:
                # Phase = 0 means identity rotation (no transformation)
                for idx in self.asymmetric_relations:
                    self.relation_emb.weight[idx].zero_()
            else:
                # Random phases in [-π, π] for diverse rotations
                for idx in self.asymmetric_relations:
                    self.relation_emb.weight[idx].uniform_(-torch.pi, torch.pi)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        relation_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute edge scores using conditional head selection.

        Routes edges to DotProduct (symmetric) or RotatE (asymmetric) based on
        relation type, then returns unified distance-based scores.

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
            Edge scores in [-2, 0] for margin loss [num_edges]
            Higher score = more likely edge (closer to 0)
        """
        # CRITICAL: Normalize embeddings to unit norm
        # This ensures distances are bounded in [0, 2] for both heads
        node_embeddings = F.normalize(node_embeddings, p=2, dim=1)

        num_edges = edge_index.size(1)
        scores = torch.zeros(num_edges, device=node_embeddings.device)

        # Process symmetric relations (dot product → distance)
        sym_mask = self.is_symmetric[relation_type]
        if sym_mask.any():
            scores[sym_mask] = self._compute_dot_scores(
                node_embeddings, edge_index[:, sym_mask]
            )

        # Process asymmetric relations (RotatE)
        asym_mask = ~sym_mask
        if asym_mask.any():
            scores[asym_mask] = self._compute_rotate_scores(
                node_embeddings,
                edge_index[:, asym_mask],
                relation_type[asym_mask],  # Direct indexing now!
            )

        return scores

    def scores_to_probs(self, scores: torch.Tensor) -> torch.Tensor:
        return normalized_distances_to_probs(scores)

    def _compute_dot_scores(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dot product-based distance scores."""
        src = node_embeddings[edge_index[0]]
        tgt = node_embeddings[edge_index[1]]

        # Similarity in [-1, 1] for normalized embeddings
        similarity = (src * tgt).mean(dim=1)

        # Convert to distance in [0, 2], then negate for score in [-2, 0]
        distance = 1.0 - similarity
        score = -distance

        return score

    def _compute_rotate_scores(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        relation_type: torch.Tensor,  # Now uses global indices directly!
    ) -> torch.Tensor:
        """Compute RotatE rotation-based distance scores using shared utility."""
        # Get head and tail embeddings
        head_embeddings = node_embeddings[edge_index[0]]
        tail_embeddings = node_embeddings[edge_index[1]]

        # Get relation phase angles (direct indexing, NaNs for symmetric relations)
        phase = self.relation_emb(relation_type)

        # Compute RotatE distance using shared utility
        distance = compute_rotate_distance(
            head_embeddings, tail_embeddings, phase, self.eps
        )

        # Negate for score in [-2, 0]
        score = -distance

        return score


class DistMultHead(nn.Module):
    """
    DistMult-style relation scoring for graph neural networks.

    Adapted from knowledge graph DistMult (Yang et al. 2015) to GNN setting
    where nodes share embedding space instead of having separate entity embeddings.

    Score = mean(h ⊙ r ⊙ t) where h,t ∈ same embedding space

    Parameters
    ----------
    embedding_dim : int
        Dimension of node embeddings from GNN
    num_relations : int
        Number of distinct relation types

    Notes
    -----
    **Key Difference from Original DistMult:**
    - Original: Separate embeddings per entity (h_aspirin, t_headache)
    - This version: Shared node embedding space (all nodes use same encoder)

    **Symmetry Warning:**
    Like original DistMult, this is symmetric: score(h,r,t) = score(t,r,h)
    Cannot distinguish directed relations without combining with asymmetric encoder
    or relation-specific directionality in the GNN.


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

    loss_type = LOSSES.BCE

    def __init__(
        self,
        embedding_dim: int,
        num_relations: int,
        init_as_identity: bool = HEAD_DEFS.DEFAULT_INIT_HEAD_AS_IDENTITY,
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

    loss_type = LOSSES.BCE

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

    loss_type = LOSSES.BCE

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


class RelationAttentionHead(nn.Module):
    """
    Lightweight relation-aware multi-head attention for edge prediction.

    Simplified version of RelationAttentionMLPHead:
    - Directly projects nodes instead of edge MLP
    - Multi-head attention with relation queries
    - No residual connection or output MLP (lighter)
    - Still captures relation-specific feature selection

    Architecture:
    1. Project nodes → attention space (replaces edge MLP)
    2. Relation embeddings → queries (multi-head)
    3. Node projections → keys, values (multi-head)
    4. Attention → weighted combination → score

    Parameters
    ----------
    embedding_dim : int
        Dimension of input node embeddings
    num_relations : int
        Number of distinct relation types
    relation_emb_dim : int, optional
        Dimension of relation embeddings, by default 64
    hidden_dim : int, optional
        Hidden dimension (must be divisible by num_attention_heads), by default 128
    num_attention_heads : int, optional
        Number of attention heads, by default 4

    Notes
    -----
    Comparison to RelationAttentionMLPHead:
    - Much lighter: no edge MLP, no output MLP, no residual
    - Same core idea: relation queries edge features via attention
    - Parameters: ~50K vs ~100K (half the size)
    - More interpretable: fewer non-linearities

    Comparison to AttentionHead:
    - Adds relation-specific attention (like RelationAttentionMLPHead)
    - Multi-head for richer feature selection
    - More parameters but more expressive
    """

    loss_type = LOSSES.BCE

    def __init__(
        self,
        embedding_dim: int,
        num_relations: int,
        relation_emb_dim: int = HEAD_DEFS.DEFAULT_RELATION_EMB_DIM,
        hidden_dim: int = HEAD_DEFS.DEFAULT_MLP_HIDDEN_DIM,
        num_attention_heads: int = HEAD_DEFS.DEFAULT_RELATION_ATTENTION_HEADS,
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
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_dim // num_attention_heads

        # Relation embeddings
        self.relation_emb = nn.Embedding(num_relations, relation_emb_dim)

        # Project concatenated [src || tgt] to hidden space
        # (This replaces the edge_mlp from RelationAttentionMLPHead)
        self.edge_proj = nn.Linear(2 * embedding_dim, hidden_dim)

        # Attention projections (like RelationAttentionMLPHead)
        self.query_proj = nn.Linear(relation_emb_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection (simpler than full MLP)
        self.output_proj = nn.Linear(hidden_dim, 1)

        # Initialize
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights with sensible defaults.

        Strategy:
        - Relation embeddings: Small random (std=0.1) to allow learning
        - Edge projection: Xavier with moderate gain to preserve info
        - Attention Q/K/V: Standard Xavier to balance stability/expressiveness
        - Output: Small gain to avoid initial saturation
        """
        # Relation embeddings: small but learnable
        nn.init.normal_(self.relation_emb.weight, mean=0.0, std=0.1)

        # Edge projection: preserve magnitude
        nn.init.xavier_uniform_(self.edge_proj.weight, gain=1.0)

        # Attention projections: standard
        nn.init.xavier_uniform_(self.query_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.key_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.value_proj.weight, gain=1.0)

        # Output: small to avoid saturation
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)

        # Zero biases
        nn.init.zeros_(self.edge_proj.bias)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        relation_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute relation-aware multi-head attention scores.

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
        # Normalize embeddings
        node_embeddings = F.normalize(node_embeddings, p=2, dim=1)

        # Get source and target embeddings
        src = node_embeddings[edge_index[0]]
        tgt = node_embeddings[edge_index[1]]

        # Project edge (src || tgt) to hidden space
        edge_features = torch.cat([src, tgt], dim=1)
        edge_hidden = self.edge_proj(edge_features)  # [num_edges, hidden_dim]

        # Get relation embeddings
        rel_emb = self.relation_emb(relation_type)  # [num_edges, relation_emb_dim]

        # Multi-head attention
        batch_size = edge_hidden.size(0)

        # Project to Q, K, V and reshape for multi-head
        Q = self.query_proj(rel_emb).view(
            batch_size, self.num_attention_heads, self.head_dim
        )  # [num_edges, num_heads, head_dim]
        K = self.key_proj(edge_hidden).view(
            batch_size, self.num_attention_heads, self.head_dim
        )
        V = self.value_proj(edge_hidden).view(
            batch_size, self.num_attention_heads, self.head_dim
        )

        # Scaled dot-product attention (per edge, across heads)
        scale = torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32, device=Q.device)
        )
        attn_scores = (Q * K).sum(dim=-1) / scale  # [num_edges, num_heads]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [num_edges, num_heads]

        # Apply attention to values
        attended = attn_weights.unsqueeze(-1) * V  # [num_edges, num_heads, head_dim]
        attended = attended.view(batch_size, -1)  # [num_edges, hidden_dim]

        # Output projection (no residual, simpler than full MLP)
        score = self.output_proj(attended).squeeze(-1)  # [num_edges]

        return score


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

    loss_type = LOSSES.BCE

    def __init__(
        self,
        embedding_dim: int,
        num_relations: int,
        relation_emb_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
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

    loss_type = LOSSES.BCE

    def __init__(
        self,
        embedding_dim: int,
        num_relations: int,
        relation_emb_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_attention_heads: int = 4,
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
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights, optionally starting near identity."""
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

    Models relations as rotations in complex space: h ⊙ r ≈ t
    where ⊙ is complex multiplication (Hadamard product in re/im components).

    Scoring function: score = -||h ⊙ r - t||

    Parameters
    ----------
    embedding_dim : int
        Dimension of node embeddings from GNN (must be even for complex embeddings)
    num_relations : int
        Number of distinct relation types
    margin : float, optional
        Margin for ranking loss, by default 1.0
    init_as_identity : bool, optional
        Initialize relations as identity rotations (angle=0), by default False

    Notes
    -----
    - Embeddings are split into real/imaginary parts: [embedding_dim/2, embedding_dim/2]
    - Relations are phase angles that rotate head embeddings
    - Handles symmetric relations (r₁ = -r₂) and composition (r₃ = r₁ + r₂)
    - **Requires normalized embeddings** (||h|| = ||t|| = 1) for bounded distances
    - Distance range with unit norm: [0, 2]
    - Score range: [-2, 0]

    References
    ----------
    Sun et al. "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space"
    ICLR 2019.

    Examples
    --------
    >>> head = RotatEHead(embedding_dim=256, num_relations=4)
    >>> scores = head(z, edge_index, relation_type)
    """

    loss_type = LOSSES.MARGIN

    def __init__(
        self,
        embedding_dim: int,
        num_relations: int,
        margin: float = HEAD_DEFS.DEFAULT_ROTATE_MARGIN,
        init_as_identity: bool = HEAD_DEFS.DEFAULT_INIT_HEAD_AS_IDENTITY,
    ):
        super().__init__()

        if embedding_dim % 2 != 0:
            raise ValueError(
                "embedding_dim must be even for RotatE (complex embeddings)"
            )

        self.embedding_dim = embedding_dim
        self.num_relations = num_relations
        self.margin = margin
        self.eps = 1e-8

        # Relation embeddings as phase angles
        # Shape: [num_relations, embedding_dim/2]
        self.relation_emb = nn.Embedding(num_relations, embedding_dim // 2)

        if init_as_identity:
            # Initialize to zero phase (identity rotation)
            nn.init.zeros_(self.relation_emb.weight)
        else:
            # Initialize uniformly in [-π, π]
            nn.init.uniform_(
                self.relation_emb.weight,
                a=-torch.pi,
                b=torch.pi,
            )

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
            Edge scores [num_edges] (higher = more likely, range [-2, 0])
        """
        # CRITICAL: Normalize embeddings to unit norm (RotatE requirement)
        node_embeddings = F.normalize(node_embeddings, p=2, dim=1)

        # Split into real and imaginary parts
        head_embeddings = node_embeddings[edge_index[0]]
        tail_embeddings = node_embeddings[edge_index[1]]

        head_re, head_im = torch.chunk(head_embeddings, 2, dim=-1)
        tail_re, tail_im = torch.chunk(tail_embeddings, 2, dim=-1)

        # Get relation phase angles
        phase = self.relation_emb(relation_type)

        # Convert phase to rotation (cos + i*sin)
        rel_re = torch.cos(phase)
        rel_im = torch.sin(phase)

        # Complex multiplication: (h_re + i*h_im) * (r_re + i*r_im)
        re_score = head_re * rel_re - head_im * rel_im
        im_score = head_re * rel_im + head_im * rel_re

        # Distance between rotated head and tail
        re_diff = re_score - tail_re
        im_diff = im_score - tail_im

        # L2 distance with numerical stability
        distance = torch.sqrt(re_diff**2 + im_diff**2 + self.eps).mean(dim=-1)

        # Score = negative distance (range [-2, 0])
        # Higher score = better edge
        score = -distance

        return score

    def scores_to_probs(self, scores: torch.Tensor) -> torch.Tensor:
        return normalized_distances_to_probs(scores)


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
    - **Requires normalized embeddings** (||h|| = ||t|| = 1) for bounded distances

    References
    ----------
    Bordes et al. "Translating Embeddings for Modeling Multi-relational Data"
    NeurIPS 2013.

    Examples
    --------
    >>> head = TransEHead(embedding_dim=256, num_relations=4)
    >>> scores = head(z, edge_index, relation_type)
    """

    loss_type = LOSSES.MARGIN

    def __init__(
        self,
        embedding_dim: int,
        num_relations: int,
        margin: float = HEAD_DEFS.DEFAULT_TRANSE_MARGIN,
        norm: int = 2,
        init_as_identity: bool = HEAD_DEFS.DEFAULT_INIT_HEAD_AS_IDENTITY,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_relations = num_relations
        self.margin = margin
        self.norm = norm
        self.eps = 1e-8

        self.relation_emb = nn.Embedding(num_relations, embedding_dim)
        if init_as_identity:
            # Initialize to zero: h + 0 - t = h - t
            nn.init.zeros_(self.relation_emb.weight)
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
        # CRITICAL: Normalize embeddings to unit norm (TransE requirement)
        # This ensures distances are bounded in [0, 2] for L2 norm
        node_embeddings = F.normalize(node_embeddings, p=2, dim=1)

        head = node_embeddings[edge_index[0]]
        tail = node_embeddings[edge_index[1]]
        rel = self.relation_emb(relation_type)

        # TransE scoring: h + r should be close to t
        diff = head + rel - tail

        # Compute distance
        if self.norm == 1:
            # L1 norm with normalized embeddings
            # Max L1 distance between unit vectors ≈ 2 (when opposite)
            distance = diff.abs().mean(dim=-1)
        else:
            # L2 norm with normalized embeddings
            # Max L2 distance = sqrt(2) when vectors are orthogonal
            distance = torch.sqrt((diff**2).sum(dim=-1) + self.eps)

        # Center around zero with learnable parameters
        score = -distance

        return score

    def scores_to_probs(self, scores: torch.Tensor) -> torch.Tensor:
        return normalized_distances_to_probs(scores)


class Decoder(nn.Module):
    """
    Unified head decoder that can create different types of prediction heads.

    This class provides a single interface for creating various head types
    (e.g., dot product, MLP, attention, node classification) with a from_config
    classmethod for easy integration with configuration systems.

    Parameters
    ----------
    hidden_channels : int
        Dimension of input node embeddings (should match GNN encoder output)
    head_type : str
        Type of head to create (dot_product, mlp, attention, node_classification)
    num_relations : int, optional
        Number of relation types (required for relation-aware heads)
    symmetric_relation_indices : List[int], optional
        List of relation type indices that are symmetric.
        This is required for heads that support special symmetry handling.
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
        symmetric_relation_indices: Optional[List[int]] = None,
        num_classes: Optional[int] = None,
        init_head_as_identity: bool = HEAD_DEFS.DEFAULT_INIT_HEAD_AS_IDENTITY,
        mlp_hidden_dim: int = HEAD_DEFS.DEFAULT_MLP_HIDDEN_DIM,
        mlp_num_layers: int = HEAD_DEFS.DEFAULT_MLP_NUM_LAYERS,
        mlp_dropout: float = HEAD_DEFS.DEFAULT_MLP_DROPOUT,
        nc_dropout: float = HEAD_DEFS.DEFAULT_NC_DROPOUT,
        rotate_margin: float = HEAD_DEFS.DEFAULT_ROTATE_MARGIN,
        transe_margin: float = HEAD_DEFS.DEFAULT_TRANSE_MARGIN,
        relation_emb_dim: int = HEAD_DEFS.DEFAULT_RELATION_EMB_DIM,
        relation_attention_heads: int = HEAD_DEFS.DEFAULT_RELATION_ATTENTION_HEADS,
    ):
        super().__init__()

        # Store all initialization parameters FIRST (before any validation)
        self._init_args = {
            MODEL_DEFS.HIDDEN_CHANNELS: hidden_channels,
            MODEL_DEFS.HEAD_TYPE: head_type,
            MODEL_DEFS.NUM_RELATIONS: num_relations,
            MODEL_DEFS.SYMMETRIC_RELATION_INDICES: symmetric_relation_indices,
            MODEL_DEFS.NUM_CLASSES: num_classes,
            HEAD_SPECIFIC_ARGS.INIT_HEAD_AS_IDENTITY: init_head_as_identity,
            HEAD_SPECIFIC_ARGS.MLP_HIDDEN_DIM: mlp_hidden_dim,
            HEAD_SPECIFIC_ARGS.MLP_NUM_LAYERS: mlp_num_layers,
            HEAD_SPECIFIC_ARGS.MLP_DROPOUT: mlp_dropout,
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
            rotate_heads = {HEADS.ROTATE, HEADS.CONDITIONAL_ROTATE}
            if head_type in rotate_heads and hidden_channels % 2 != 0:
                raise ValueError(
                    f"{head_type} requires even hidden_channels for complex space, "
                    f"got {hidden_channels}"
                )

        if head_type == HEADS.NODE_CLASSIFICATION:
            if num_classes is None:
                raise ValueError(
                    f"num_classes is required for {head_type} head. "
                    f"This should be inferred from the data."
                )

        # Create the appropriate head based on type

        if head_type == HEADS.ATTENTION:
            self.head = AttentionHead(
                self.hidden_channels, mlp_hidden_dim, init_head_as_identity
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
        elif head_type == HEADS.RELATION_ATTENTION:
            self.head = RelationAttentionHead(
                embedding_dim=self.hidden_channels,
                num_relations=num_relations,
                relation_emb_dim=relation_emb_dim,
                hidden_dim=mlp_hidden_dim,
                num_attention_heads=relation_attention_heads,
            )
        elif head_type == HEADS.RELATION_ATTENTION_MLP:
            self.head = RelationAttentionMLPHead(
                embedding_dim=self.hidden_channels,
                num_relations=num_relations,
                relation_emb_dim=relation_emb_dim,
                hidden_dim=mlp_hidden_dim,
                num_layers=mlp_num_layers,
                dropout=mlp_dropout,
                num_attention_heads=relation_attention_heads,
            )
        elif head_type == HEADS.RELATION_GATED_MLP:
            self.head = RelationGatedMLPHead(
                embedding_dim=self.hidden_channels,
                num_relations=num_relations,
                relation_emb_dim=relation_emb_dim,
                hidden_dim=mlp_hidden_dim,
                num_layers=mlp_num_layers,
                dropout=mlp_dropout,
            )
        elif head_type == HEADS.ROTATE:
            self.head = RotatEHead(
                embedding_dim=self.hidden_channels,
                num_relations=num_relations,
                margin=rotate_margin,
                init_as_identity=init_head_as_identity,
            )
        elif head_type == HEADS.CONDITIONAL_ROTATE:
            self.head = ConditionalRotatEHead(
                embedding_dim=self.hidden_channels,
                num_relations=num_relations,
                margin=rotate_margin,
                init_asymmetric_as_identity=init_head_as_identity,
                symmetric_relation_indices=symmetric_relation_indices,
            )
        elif head_type == HEADS.TRANSE:
            self.head = TransEHead(
                embedding_dim=self.hidden_channels,
                num_relations=num_relations,
                margin=transe_margin,
                init_as_identity=init_head_as_identity,
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
        if self.head_type in {
            HEADS.MLP,
            HEADS.RELATION_GATED_MLP,
            HEADS.RELATION_ATTENTION_MLP,
        }:
            summary[HEAD_SPECIFIC_ARGS.MLP_HIDDEN_DIM] = self._init_args[
                HEAD_SPECIFIC_ARGS.MLP_HIDDEN_DIM
            ]
            summary[HEAD_SPECIFIC_ARGS.MLP_NUM_LAYERS] = self._init_args[
                HEAD_SPECIFIC_ARGS.MLP_NUM_LAYERS
            ]
            summary[HEAD_SPECIFIC_ARGS.MLP_DROPOUT] = self._init_args[
                HEAD_SPECIFIC_ARGS.MLP_DROPOUT
            ]
        elif self.head_type == HEADS.ROTATE:
            summary[HEAD_SPECIFIC_ARGS.ROTATE_MARGIN] = self._init_args[
                HEAD_SPECIFIC_ARGS.ROTATE_MARGIN
            ]
        elif self.head_type == HEADS.TRANSE:
            summary[HEAD_SPECIFIC_ARGS.TRANSE_MARGIN] = self._init_args[
                HEAD_SPECIFIC_ARGS.TRANSE_MARGIN
            ]
        if self.head_type in {
            HEADS.RELATION_GATED_MLP,
            HEADS.RELATION_ATTENTION_MLP,
            HEADS.RELATION_ATTENTION,
        }:
            summary[HEAD_SPECIFIC_ARGS.RELATION_EMB_DIM] = self._init_args[
                HEAD_SPECIFIC_ARGS.RELATION_EMB_DIM
            ]
        if self.head_type in {HEADS.RELATION_ATTENTION_MLP, HEADS.RELATION_ATTENTION}:
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

    @property
    def loss_type(self) -> str:
        """
        Get the loss type required by the underlying head.

        Returns
        -------
        str
            Loss type (e.g., LOSSES.BCE, LOSSES.MARGIN)
        """
        return type(self.head).loss_type

    @property
    def margin(self) -> float:
        """
        Get the margin value for heads that support margin loss (RotatE, TransE).

        Returns
        -------
        float
            Margin value for ranking loss

        Raises
        ------
        AttributeError
            If the underlying head does not have a margin attribute
        """
        return self.head.margin

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
        symmetric_relation_indices: Optional[List[int]] = None,
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
        symmetric_relation_indices : List[int], optional
            List of relation type indices that are symmetric.
            This is required for heads that support special symmetry handling.

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
            MODEL_DEFS.SYMMETRIC_RELATION_INDICES: symmetric_relation_indices,
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
