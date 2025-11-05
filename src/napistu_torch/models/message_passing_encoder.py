"""
Graph Neural Network models for Napistu-Torch - CLEAN VERSION

Removed edge_weight parameter since it's not used for message passing.
Edge weights are stored in edge attributes for supervision, not encoding.
"""

import logging
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GraphConv, SAGEConv

from napistu_torch.configs import ModelConfig
from napistu_torch.constants import MODEL_CONFIG
from napistu_torch.models.constants import (
    EDGE_WEIGHTING_TYPE,
    ENCODER_DEFS,
    ENCODER_NATIVE_ARGNAMES_MAPS,
    ENCODER_SPECIFIC_ARGS,
    ENCODERS,
    ENCODERS_SUPPORTING_EDGE_WEIGHTING,
    MODEL_DEFS,
    VALID_ENCODERS,
)
from napistu_torch.models.edge_encoder import EdgeEncoder

logger = logging.getLogger(__name__)


ENCODER_CLASSES = {
    ENCODERS.GAT: GATConv,
    ENCODERS.GCN: GCNConv,
    ENCODERS.GRAPH_CONV: GraphConv,
    ENCODERS.SAGE: SAGEConv,
}


class MessagePassingEncoder(nn.Module):
    """
    Unified Graph Neural Network encoder supporting multiple architectures.

    This class eliminates boilerplate by providing a single interface for
    SAGE, GCN, and GAT models with consistent behavior and configuration.

    Edge Weight Support
    -------------------
    - GCN: ✅ Supports edge_weight parameter
    - GraphConv: ✅ Supports edge_weight parameter (SAGE-like with edge weights)
    - SAGE: ❌ Does not support edge_weight (gracefully ignored)
    - GAT: ❌ Uses learned attention (edge_weight not needed)

    Parameters
    ----------
    in_channels : int
        Number of input node features
    hidden_channels : int
        Number of hidden channels in each layer
    num_layers : int
        Number of GNN layers
    dropout : float, optional
        Dropout probability, by default 0.0
    encoder_type : str, optional
        Type of encoder ('sage', 'gcn', 'gat'), by default 'sage'
    sage_aggregator : str, optional
        Aggregation method for SAGE ('mean', 'max', 'lstm'), by default 'mean'
    gat_heads : int, optional
        Number of attention heads for GAT, by default 1
    gat_concat : bool, optional
        Whether to concatenate attention heads in GAT, by default True
    graph_conv_aggregator : str, optional
        Aggregation method for GraphConv, by default 'add'

    Notes
    -----
    This encoder does NOT use edge weights for message passing. If you need
    weighted message passing, you would need to:
    1. Use GCNConv (only encoder that natively supports edge weights)
    2. Implement custom message passing with edge attributes

    Edge weights and attributes in your NapistuData are still available for
    supervision and evaluation - they just aren't used during encoding.

    Examples
    --------
    >>> # Direct instantiation
    >>> encoder = MessagePassingEncoder(128, 256, 3, encoder_type='sage', sage_aggregator='mean')
    >>>
    >>> # From config
    >>> config = ModelConfig(encoder='sage', hidden_channels=256, num_layers=3)
    >>> encoder = MessagePassingEncoder.from_config(config, in_channels=128)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float = 0.0,
        encoder_type: str = ENCODERS.SAGE,
        # Edge weighting
        weight_edges_by: Optional[Union[torch.Tensor, nn.Module]] = None,
        # GAT-specific parameters
        gat_heads: int = 1,
        gat_concat: bool = True,
        # GraphConv-specific parameters
        graph_conv_aggregator: str = ENCODER_DEFS.GRAPH_CONV_DEFAULT_AGGREGATOR,
        # SAGE-specific parameters
        sage_aggregator: str = ENCODER_DEFS.SAGE_DEFAULT_AGGREGATOR,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.encoder_type = encoder_type

        # Map encoder types to classes
        if encoder_type not in VALID_ENCODERS:
            raise ValueError(
                f"Unknown encoder: {encoder_type}. Must be one of {VALID_ENCODERS}"
            )

        encoder = ENCODER_CLASSES[encoder_type]
        self.convs = nn.ModuleList()
        self.supports_edge_weight = encoder_type in ENCODERS_SUPPORTING_EDGE_WEIGHTING

        # Parse edge weighting specification into type indicator and value
        self.edge_weighting_type, self.edge_weighting_value = _parse_edge_weighting(
            weight_edges_by, self.supports_edge_weight, encoder_type
        )
        logger.debug(
            f"Initialized MessagePassingEncoder with edge_weighting_type={self.edge_weighting_type}"
        )

        # Build encoder_kwargs based on encoder using dict comprehension
        param_mapping = ENCODER_NATIVE_ARGNAMES_MAPS.get(encoder_type, {})
        local_vars = locals()
        encoder_kwargs = {
            native_param: local_vars[encoder_param]
            for encoder_param, native_param in param_mapping.items()
        }

        # Build layers
        for i in range(num_layers):
            if i == 0:
                # First layer: in_channels -> hidden_channels
                self.convs.append(
                    encoder(in_channels, hidden_channels, **encoder_kwargs)
                )
            else:
                # Hidden/output layers: handle GAT's head concatenation
                if encoder_type == ENCODERS.GAT:
                    # For GAT, calculate input dimension based on previous layer's concat setting
                    if i == 1:
                        # Second layer: input comes from first layer
                        in_dim = (
                            hidden_channels * gat_heads
                            if gat_concat
                            else hidden_channels
                        )
                    else:
                        # Subsequent layers: input comes from previous layer
                        # If previous layer concatenated, we need to account for that
                        in_dim = (
                            hidden_channels * gat_heads
                            if gat_concat
                            else hidden_channels
                        )

                    # For the final layer, we might want to not concatenate to get clean output
                    layer_kwargs = encoder_kwargs.copy()
                    if i == num_layers - 1 and not gat_concat:
                        # Final layer: don't concatenate heads for clean output
                        layer_kwargs["concat"] = False
                        in_dim = (
                            hidden_channels * gat_heads
                        )  # Previous layer was concatenated

                    self.convs.append(encoder(in_dim, hidden_channels, **layer_kwargs))
                else:
                    # SAGE/GCN/GraphConv: hidden_channels -> hidden_channels
                    self.convs.append(
                        encoder(hidden_channels, hidden_channels, **encoder_kwargs)
                    )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_data: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the GNN encoder.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix [num_nodes, in_channels]
        edge_index : torch.Tensor
            Edge connectivity [2, num_edges]
        edge_data : torch.Tensor, optional
            Edge data [num_edges, edge_dim] for edge weighting.
            Can be edge attributes (for learnable encoder) or edge weights (for static weighting).

        Returns
        -------
        torch.Tensor
            Node embeddings [num_nodes, hidden_channels]

        Notes
        -----
        Edge weighting is handled based on the edge_weighting_type attribute:
        - EDGE_WEIGHTING_TYPE.NONE: No edge weighting (uniform message passing)
        - EDGE_WEIGHTING_TYPE.STATIC_WEIGHTS: Static edge weights (edge_data contains pre-computed weights)
        - EDGE_WEIGHTING_TYPE.LEARNED_ENCODER: Learnable edge encoder (edge_data contains edge attributes)
        """
        # Compute edge weights based on edge_weighting_type
        if (
            getattr(self, ENCODER_DEFS.EDGE_WEIGHTING_TYPE)
            == EDGE_WEIGHTING_TYPE.LEARNED_ENCODER
        ):
            # Learnable edge encoder - requires edge_data (edge attributes)
            if edge_data is None:
                raise ValueError("edge_data required when using learnable edge encoder")
            edge_weight = getattr(self, ENCODER_DEFS.EDGE_WEIGHTING_VALUE)(edge_data)
        elif (
            getattr(self, ENCODER_DEFS.EDGE_WEIGHTING_TYPE)
            == EDGE_WEIGHTING_TYPE.STATIC_WEIGHTS
        ):
            # Static edge weights - use pre-masked weights from edge_data
            if edge_data is None:
                raise ValueError("edge_data required when using static edge weights")
            edge_weight = edge_data
        else:  # EDGE_WEIGHTING_TYPE.NONE
            edge_weight = None

        # Warn if edge weights are provided but not supported
        if not self.supports_edge_weight and edge_weight is not None:
            logger.warning(
                f"Edge weights are not supported for {self.encoder_type}. Ignoring edge_weight."
            )

        for i, conv in enumerate(self.convs):
            if edge_weight is not None and self.supports_edge_weight:
                x = conv(x, edge_index, edge_weight=edge_weight)
            else:
                x = conv(x, edge_index)

            # Apply activation and dropout (except on last layer)
            if i < len(self.convs) - 1:
                # GAT uses ELU, others use ReLU
                if self.encoder_type == ENCODERS.GAT:
                    x = F.elu(x)
                else:
                    x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_data: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Alias for forward method for consistency with other models.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix [num_nodes, in_channels]
        edge_index : torch.Tensor
            Edge connectivity [2, num_edges]
        edge_data : Optional[torch.Tensor]
            Edge data [num_edges, edge_dim] for edge weighting.
            Can be edge attributes (for learnable encoder) or edge weights (for static weighting).

        Returns
        -------
        torch.Tensor
            Node embeddings [num_nodes, hidden_channels]
        """
        return self.forward(x, edge_index, edge_data)

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        in_channels: int,
        edge_in_channels: Optional[int] = None,
    ) -> "MessagePassingEncoder":
        """
        Create MessagePassingEncoder from ModelConfig.

        Parameters
        ----------
        config : ModelConfig
            Model configuration containing encoder, hidden_channels, etc.
        in_channels : int
            Number of input node features (not in config as it depends on data)
        edge_in_channels : int, optional
            Number of input edge features. Required if use_edge_encoder=True.

        Returns
        -------
        MessagePassingEncoder
            Configured encoder instance

        Examples
        --------
        >>> config = ModelConfig(encoder='sage', hidden_channels=256, num_layers=3)
        >>> encoder = MessagePassingEncoder.from_config(config, in_channels=128)
        >>>
        >>> # With edge encoder
        >>> config = ModelConfig(encoder='gcn', use_edge_encoder=True, edge_encoder_dim=32)
        >>> encoder = MessagePassingEncoder.from_config(config, in_channels=128, edge_in_channels=10)
        """

        encoder_type = getattr(config, MODEL_CONFIG.ENCODER)
        if encoder_type not in VALID_ENCODERS:
            raise ValueError(
                f"Unknown encoder: {encoder_type}. Must be one of {VALID_ENCODERS}"
            )

        # Build model-specific parameters
        model_kwargs = {}

        if encoder_type == ENCODERS.SAGE and config.sage_aggregator is not None:
            model_kwargs[ENCODER_SPECIFIC_ARGS.SAGE_AGGREGATOR] = config.sage_aggregator
        if encoder_type == ENCODERS.GAT and config.gat_heads is not None:
            model_kwargs[ENCODER_SPECIFIC_ARGS.GAT_HEADS] = config.gat_heads
        if encoder_type == ENCODERS.GAT and config.gat_concat is not None:
            model_kwargs[ENCODER_SPECIFIC_ARGS.GAT_CONCAT] = config.gat_concat
        if (
            encoder_type == ENCODERS.GRAPH_CONV
            and config.graph_conv_aggregator is not None
        ):
            model_kwargs[ENCODER_SPECIFIC_ARGS.GRAPH_CONV_AGGREGATOR] = (
                config.graph_conv_aggregator
            )

        # Handle edge encoder creation
        if getattr(config, MODEL_CONFIG.USE_EDGE_ENCODER, False):
            if edge_in_channels is None:
                raise ValueError(
                    "edge_in_channels must be provided when use_edge_encoder=True"
                )

            logger.debug(
                f"Creating EdgeEncoder: edge_in_channels={edge_in_channels}, "
                f"hidden_dim={config.edge_encoder_dim}, dropout={config.edge_encoder_dropout}"
            )
            edge_encoder = EdgeEncoder(
                edge_dim=edge_in_channels,
                hidden_dim=config.edge_encoder_dim,
                dropout=config.edge_encoder_dropout,
            )
            weight_edges_by = edge_encoder
        else:
            weight_edges_by = None

        result = cls(
            in_channels=in_channels,
            hidden_channels=getattr(config, MODEL_DEFS.HIDDEN_CHANNELS),
            num_layers=getattr(config, MODEL_DEFS.NUM_LAYERS),
            dropout=getattr(config, ENCODER_SPECIFIC_ARGS.DROPOUT),
            encoder_type=encoder_type,
            weight_edges_by=weight_edges_by,
            **model_kwargs,
        )
        logger.debug(
            f"Created MessagePassingEncoder with edge_weighting_type={getattr(result, ENCODER_DEFS.EDGE_WEIGHTING_TYPE, 'N/A')}"
        )
        return result


# private utils


def _parse_edge_weighting(
    weight_edges_by: Optional[Union[torch.Tensor, nn.Module]],
    supports_edge_weight: bool,
    encoder_type: str,
) -> tuple[str, Optional[Union[torch.Tensor, nn.Module]]]:
    """
    Parse weight_edges_by parameter into type indicator and value.

    This utility function handles the polyschematicity of edge weighting options
    by explicitly separating the type indicator from the value.

    Parameters
    ----------
    weight_edges_by : Optional[Union[torch.Tensor, nn.Module]]
        Edge weighting specification:
        - None: No edge weighting
        - torch.Tensor: Static edge weights
        - nn.Module: Learnable edge encoder
    supports_edge_weight : bool
        Whether the encoder type supports edge weighting
    encoder_type : str
        Name of encoder type (for logging)

    Returns
    -------
    tuple[str, Optional[Union[torch.Tensor, nn.Module]]]
        Tuple of (edge_weighting_type, edge_weighting_value)
        - edge_weighting_type: String constant from EDGE_WEIGHTING_TYPE indicating the type of weighting
        - edge_weighting_value: The actual value (None, Tensor, or Module)

    Raises
    ------
    ValueError
        If weight_edges_by is not None, Tensor, or Module
    """

    if weight_edges_by is None:
        return EDGE_WEIGHTING_TYPE.NONE, None

    if not supports_edge_weight:
        logger.warning(
            f"Edge weighting is not supported for {encoder_type}, ignoring weight_edges_by"
        )
        return EDGE_WEIGHTING_TYPE.NONE, None

    if isinstance(weight_edges_by, nn.Module):
        return EDGE_WEIGHTING_TYPE.LEARNED_ENCODER, weight_edges_by
    elif isinstance(weight_edges_by, torch.Tensor):
        return EDGE_WEIGHTING_TYPE.STATIC_WEIGHTS, weight_edges_by
    else:
        raise ValueError(
            f"weight_edges_by must be None, nn.Module, or torch.Tensor, got {type(weight_edges_by)}"
        )
