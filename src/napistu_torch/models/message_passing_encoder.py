"""
Graph Neural Network models for Napistu-Torch - CLEAN VERSION

Removed edge_weight parameter since it's not used for message passing.
Edge weights are stored in edge attributes for supervision, not encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from napistu_torch.configs import ModelConfig
from napistu_torch.constants import MODEL_CONFIG
from napistu_torch.models.constants import (
    ENCODER_DEFS,
    ENCODER_NATIVE_ARGNAMES_MAPS,
    ENCODER_SPECIFIC_ARGS,
    ENCODERS,
    MODEL_DEFS,
    VALID_ENCODERS,
)

ENCODER_CLASSES = {
    ENCODERS.SAGE: SAGEConv,
    ENCODERS.GCN: GCNConv,
    ENCODERS.GAT: GATConv,
}


class MessagePassingEncoder(nn.Module):
    """
    Unified Graph Neural Network encoder supporting multiple architectures.

    This class eliminates boilerplate by providing a single interface for
    SAGE, GCN, and GAT models with consistent behavior and configuration.

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
        # SAGE-specific parameters
        sage_aggregator: str = ENCODER_DEFS.SAGE_DEFAULT_AGGREGATOR,
        # GAT-specific parameters
        gat_heads: int = 1,
        gat_concat: bool = True,
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
                    # SAGE/GCN: hidden_channels -> hidden_channels
                    self.convs.append(
                        encoder(hidden_channels, hidden_channels, **encoder_kwargs)
                    )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the GNN encoder.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix [num_nodes, in_channels]
        edge_index : torch.Tensor
            Edge connectivity [2, num_edges]

        Returns
        -------
        torch.Tensor
            Node embeddings [num_nodes, hidden_channels]

        Notes
        -----
        This method does NOT use edge weights or edge attributes for message passing.
        All encoders use unweighted/uniform message passing (or attention in GAT's case).

        If you need edge-weighted message passing in the future:
        - GCN: Add edge_weight parameter and pass to GCNConv
        - SAGE: Implement custom message passing (not natively supported)
        - GAT: Already uses learned attention (different from edge weights)
        """
        for i, conv in enumerate(self.convs):
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
    ) -> torch.Tensor:
        """
        Alias for forward method for consistency with other models.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix [num_nodes, in_channels]
        edge_index : torch.Tensor
            Edge connectivity [2, num_edges]

        Returns
        -------
        torch.Tensor
            Node embeddings [num_nodes, hidden_channels]
        """
        return self.forward(x, edge_index)

    @classmethod
    def from_config(
        cls, config: ModelConfig, in_channels: int
    ) -> "MessagePassingEncoder":
        """
        Create MessagePassingEncoder from ModelConfig.

        Parameters
        ----------
        config : ModelConfig
            Model configuration containing encoder, hidden_channels, etc.
        in_channels : int
            Number of input node features (not in config as it depends on data)

        Returns
        -------
        MessagePassingEncoder
            Configured encoder instance

        Examples
        --------
        >>> config = ModelConfig(encoder='sage', hidden_channels=256, num_layers=3)
        >>> encoder = MessagePassingEncoder.from_config(config, in_channels=128)
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

        return cls(
            in_channels=in_channels,
            hidden_channels=getattr(config, MODEL_DEFS.HIDDEN_CHANNELS),
            num_layers=getattr(config, MODEL_DEFS.NUM_LAYERS),
            dropout=getattr(config, ENCODER_SPECIFIC_ARGS.DROPOUT),
            encoder_type=encoder_type,
            **model_kwargs,
        )
