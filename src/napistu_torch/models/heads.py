"""
Prediction heads for Napistu-Torch.

This module provides implementations of different prediction heads for various tasks
like edge prediction, node classification, etc. All heads follow a consistent interface.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

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

        # Initialize bilinear weights properly to avoid NaN
        # Use Xavier uniform initialization for stability
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

        # Apply bilinear transformation
        edge_scores = self.bilinear(src_embeddings, tgt_embeddings).squeeze(
            -1
        )  # [num_edges]

        # Clamp to prevent overflow (values becoming too large/too negative) that could lead to NaN
        # This is a safety measure - if scores are extremely large, clamp them
        # The range [-50, 50] is safe for sigmoid (sigmoid(50) ≈ 1, sigmoid(-50) ≈ 0)
        edge_scores = torch.clamp(edge_scores, min=-50.0, max=50.0)

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

    def __init__(self, embedding_dim: int, num_relations: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_relations = num_relations

        # Relation embeddings (diagonal of relation matrix)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)

        nn.init.xavier_uniform_(self.relation_emb.weight)

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

        # DistMult scoring: trilinear dot product
        score = (head * rel * tail).sum(dim=-1)

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

    def __init__(self, embedding_dim: int, num_relations: int, margin: float = 9.0):
        super().__init__()

        if embedding_dim % 2 != 0:
            raise ValueError(
                f"RotatE requires even embedding_dim for complex space, "
                f"got {embedding_dim}"
            )

        self.embedding_dim = embedding_dim
        self.num_relations = num_relations
        self.margin = margin

        # Relation embeddings (phases in complex space)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim // 2)

        # Initialize relation phases uniformly in [-π, π]
        nn.init.uniform_(self.relation_emb.weight, -self.margin, self.margin)

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
        # Get head and tail embeddings
        head = node_embeddings[edge_index[0]]  # [num_edges, embedding_dim]
        tail = node_embeddings[edge_index[1]]  # [num_edges, embedding_dim]

        # Get relation phases
        phase_rel = self.relation_emb(relation_type)  # [num_edges, embedding_dim//2]

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

        # Compute distance to tail in complex space
        re_diff = re_score - re_tail
        im_diff = im_score - im_tail

        # L2 distance (negative because lower distance = higher score)
        score = torch.sqrt(re_diff**2 + im_diff**2).sum(dim=-1)
        score = -score  # Negate so higher score = better match

        return score


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
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_relations = num_relations
        self.margin = margin
        self.norm = norm

        # Relation embeddings (translations in embedding space)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)

        # Initialize with Xavier uniform
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
        # Get head and tail embeddings
        head = node_embeddings[edge_index[0]]  # [num_edges, embedding_dim]
        tail = node_embeddings[edge_index[1]]  # [num_edges, embedding_dim]

        # Get relation embeddings
        rel = self.relation_emb(relation_type)  # [num_edges, embedding_dim]

        # TransE scoring: h + r should be close to t
        # Distance metric: ||h + r - t||
        distance = (head + rel - tail).norm(p=self.norm, dim=-1)

        # Negate so higher score = better match
        score = -distance

        return score


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
        mlp_hidden_dim: int = 64,
        mlp_num_layers: int = 2,
        mlp_dropout: float = 0.1,
        bilinear_bias: bool = True,
        nc_dropout: float = 0.1,
        rotate_margin: float = 9.0,
        transe_margin: float = 1.0,
    ):
        super().__init__()

        # Store all initialization parameters FIRST (before any validation)
        self._init_args = {
            MODEL_DEFS.HIDDEN_CHANNELS: hidden_channels,
            MODEL_DEFS.HEAD_TYPE: head_type,
            MODEL_DEFS.NUM_RELATIONS: num_relations,
            MODEL_DEFS.NUM_CLASSES: num_classes,
            HEAD_SPECIFIC_ARGS.MLP_HIDDEN_DIM: mlp_hidden_dim,
            HEAD_SPECIFIC_ARGS.MLP_NUM_LAYERS: mlp_num_layers,
            HEAD_SPECIFIC_ARGS.MLP_DROPOUT: mlp_dropout,
            HEAD_SPECIFIC_ARGS.BILINEAR_BIAS: bilinear_bias,
            HEAD_SPECIFIC_ARGS.NC_DROPOUT: nc_dropout,
            HEAD_SPECIFIC_ARGS.ROTATE_MARGIN: rotate_margin,
            HEAD_SPECIFIC_ARGS.TRANSE_MARGIN: transe_margin,
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
            self.head = BilinearHead(self.hidden_channels, bilinear_bias)
        elif head_type == HEADS.DISTMULT:
            self.head = DistMultHead(self.hidden_channels, num_relations)
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
        elif head_type == HEADS.ROTATE:
            self.head = RotatEHead(self.hidden_channels, num_relations, rotate_margin)
        elif head_type == HEADS.TRANSE:
            self.head = TransEHead(self.hidden_channels, num_relations, transe_margin)
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
        }

        return cls(**head_kwargs)
