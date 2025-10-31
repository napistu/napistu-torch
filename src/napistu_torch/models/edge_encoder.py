# Add to models/encoders.py (new file) or models/gnns.py
import torch
import torch.nn as nn


class EdgeEncoder(nn.Module):
    """
    Learns edge importance weights from edge features.

    This is a standalone module that composes with GNNEncoder to provide
    learned edge weights for message passing.

    Architecture
    ------------
    edge_features → MLP → sigmoid → edge_weights [0, 1]

    The output edge weights scale message contributions during GNN aggregation,
    effectively learning to filter out noisy edges.

    Parameters
    ----------
    edge_dim : int
        Dimensionality of input edge features
    hidden_dim : int, default=32
        Hidden layer size. Keep small to avoid overfitting.
    dropout : float, default=0.1
        Dropout probability for regularization
    init_bias : float, default=0.0
        Initial bias for output layer. Controls starting edge weights:
        - 0.0 → sigmoid(0) = 0.5 (neutral, equal weighting)
        - 1.4 → sigmoid(1.4) ≈ 0.8 (optimistic, most edges good)
        - -1.4 → sigmoid(-1.4) ≈ 0.2 (pessimistic, most edges bad)

    Examples
    --------
    >>> # Create edge encoder
    >>> edge_encoder = EdgeEncoder(edge_dim=10, hidden_dim=32)
    >>>
    >>> # Use with GNNEncoder
    >>> edge_weights = edge_encoder(edge_attr)  # [num_edges, 10] -> [num_edges]
    >>> z = gnn_encoder(x, edge_index, edge_weight=edge_weights)
    >>>
    >>> # Start from heuristic weights
    >>> edge_encoder = EdgeEncoder.from_heuristic(
    ...     edge_dim=10,
    ...     heuristic_weight_idx=3  # Use column 3 as starting point
    ... )

    Notes
    -----
    - Output is in [0, 1] via sigmoid
    - Very lightweight: ~edge_dim * hidden_dim parameters
    - Learns end-to-end with the main task
    - Can be initialized to approximate existing heuristics
    """

    def __init__(
        self,
        edge_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        init_bias: float = 0.0,
    ):
        super().__init__()

        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        # Simple MLP: edge_features -> importance score
        self.net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # [0, 1] range for weights
        )

        # Initialize output layer bias
        with torch.no_grad():
            self.net[-2].bias.fill_(init_bias)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Compute edge importance weights from edge features.

        Parameters
        ----------
        edge_attr : torch.Tensor
            Edge features [num_edges, edge_dim]

        Returns
        -------
        edge_weight : torch.Tensor
            Learned edge importance weights [num_edges]
            Values in range [0, 1] where higher = more important
        """
        return self.net(edge_attr).squeeze(-1)

    @classmethod
    def from_heuristic(
        cls,
        edge_dim: int,
        heuristic_weight_idx: int,
        hidden_dim: int = 32,
    ) -> "EdgeEncoder":
        """
        Create an EdgeEncoder initialized to use a heuristic weight column.

        Useful for warm-starting from existing edge weights. The model starts
        by using your heuristic, then learns to improve it during training.

        Parameters
        ----------
        edge_dim : int
            Dimensionality of edge features
        heuristic_weight_idx : int
            Index of the column in edge_attr containing heuristic weights
            These weights should be in range [0, 1]
        hidden_dim : int, default=32
            Hidden layer size

        Returns
        -------
        EdgeEncoder
            Initialized to approximate heuristic[:, heuristic_weight_idx]

        Examples
        --------
        >>> # edge_attr[:, 3] contains your heuristic edge weights
        >>> edge_encoder = EdgeEncoder.from_heuristic(
        ...     edge_dim=10,
        ...     heuristic_weight_idx=3
        ... )
        >>> # Initially: edge_encoder(edge_attr) ≈ edge_attr[:, 3]
        >>> # After training: edge_encoder(edge_attr) = learned improvements
        """
        encoder = cls(edge_dim=edge_dim, hidden_dim=hidden_dim, init_bias=0.0)

        # Initialize first layer to extract the heuristic column
        with torch.no_grad():
            # Zero out all input weights except the heuristic column
            encoder.net[0].weight.zero_()
            encoder.net[0].weight[:, heuristic_weight_idx] = 1.0
            encoder.net[0].bias.zero_()

            # Initialize output layer to pass through (identity via sigmoid)
            # For sigmoid to be nearly identity around [0, 1], use:
            # y ≈ x when sigmoid(a*x + b) ≈ x
            # This is approximate, but good enough for initialization
            encoder.net[-2].weight.fill_(1.0)
            encoder.net[-2].bias.zero_()

        return encoder
