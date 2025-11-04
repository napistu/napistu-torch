import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch
import torch.nn as nn

from napistu_torch.constants import NAPISTU_DATA
from napistu_torch.labels.create import _prepare_discrete_labels
from napistu_torch.load.artifacts import ensure_stratify_by_artifact_name
from napistu_torch.load.constants import (
    STRATIFY_BY_ARTIFACT_NAMES,
    VALID_STRATIFY_BY,
)
from napistu_torch.load.stratification import (
    ensure_strata_series,
    validate_edge_strata_alignment,
)
from napistu_torch.ml.constants import SPLIT_TO_MASK, TRAINING
from napistu_torch.models.constants import (
    EDGE_WEIGHTING_TYPE,
    ENCODER_DEFS,
)
from napistu_torch.napistu_data import NapistuData
from napistu_torch.tasks.base import BaseTask
from napistu_torch.tasks.constants import (
    EDGE_PREDICTION_BATCH,
    NEGATIVE_SAMPLING_STRATEGIES,
)
from napistu_torch.tasks.negative_sampler import NegativeSampler

logger = logging.getLogger(__name__)


class EdgePredictionTask(BaseTask):
    """
    Edge prediction (link prediction) task.

    Predicts whether edges exist between node pairs using:
    1. Node embeddings from encoder
    2. Edge scores from head (dot product, MLP, etc.)
    3. Category-constrained negative sampling (optional)

    This class is Lightning-free - pure PyTorch logic.
    Use EdgePredictionLightning (in napistu_torch.lightning) for training.

    Parameters
    ----------
    encoder : nn.Module
        Graph encoder (SAGE, GCN, GAT, etc.)
    head : nn.Module
        Edge decoder (DotProduct, MLP, Bilinear, etc.)
    neg_sampling_ratio : float
        Ratio of negative to positive samples
    edge_strata : pd.Series or pd.DataFrame, optional
        Edge categories for stratified negative sampling.
        If DataFrame, must have single column named "edge_strata".
        If None, uses single category (still gets degree-weighted sampling).
    neg_sampling_strategy : str
        'uniform' or 'degree_weighted'
    metrics : List[str]
        Metrics to compute ('auc', 'ap', etc.)

    Examples
    --------
    >>> # Create task with stratified sampling
    >>> task = EdgePredictionTask(
    ...     encoder, head,
    ...     edge_strata=edge_strata_series,
    ...     neg_sampling_strategy='degree_weighted'
    ... )
    >>>
    >>> # Create task without strata (still uses degree-weighted sampling)
    >>> task = EdgePredictionTask(
    ...     encoder, head,
    ...     neg_sampling_strategy='degree_weighted'
    ... )
    """

    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        neg_sampling_ratio: float = 1.0,
        edge_strata: Optional[Union[pd.Series, pd.DataFrame]] = None,
        neg_sampling_strategy: str = NEGATIVE_SAMPLING_STRATEGIES.DEGREE_WEIGHTED,
        metrics: List[str] = None,
    ):
        super().__init__(encoder, head)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.neg_sampling_ratio = neg_sampling_ratio
        self.edge_strata = edge_strata
        self.neg_sampling_strategy = neg_sampling_strategy
        self.metrics = metrics or ["auc", "ap"]

        # Negative sampler (initialized lazily on first prepare_batch call)
        self.negative_sampler = None
        self._sampler_initialized = False

    def prepare_batch(
        self,
        data: NapistuData,
        split: str = TRAINING.TRAIN,
        edge_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare batch for edge prediction.

        For transductive (mask-based) splits, this:
        1. Gets positive edges from the split mask (or edge_indices subset)
        2. Gets supervision edges (for message passing)
        3. Samples negative edges

        Parameters
        ----------
        data : NapistuData
            Full graph data
        split : str
            Which split ('train', 'val', 'test')
        edge_indices : torch.Tensor, optional
            Indices into data.edge_index for this mini-batch.
            If None, uses all edges from the split mask (full-batch mode).
            If provided, uses only these edges (mini-batch mode).

        Returns
        -------
        Dict with keys:
            - x: Node features
            - supervision_edges: Edges for message passing (always full training graph)
            - pos_edges: Positive edges to predict (from edge_indices or split mask)
            - neg_edges: Negative edges to predict (sampled)
            - edge_data: Edge data for supervision edges (attributes for learnable encoders,
                        weights for static weighting)
        """
        # Lazy initialization on first call
        self._ensure_negative_sampler(data)

        # Get positive edges for this batch
        if edge_indices is not None:
            # Mini-batch mode: use provided edge indices and make sure they are on the data's device
            edge_indices = edge_indices.to(data.edge_index.device)
            pos_edge_index = data.edge_index[:, edge_indices]
        else:
            # Full-batch mode: use all edges from split mask
            mask_attr = SPLIT_TO_MASK[split]
            mask = getattr(data, mask_attr)
            pos_edge_index = data.edge_index[:, mask]

        # Always use training edges for message passing (prevents data leakage)
        train_indices = torch.where(data.train_mask)[0]
        supervision_edges = data.edge_index[:, train_indices]

        # Sample negative edges proportional to batch size
        num_neg = int(pos_edge_index.size(1) * self.neg_sampling_ratio)
        neg_edge_index, _ = self.negative_sampler.sample(
            num_neg=num_neg, device=str(pos_edge_index.device)
        )

        # Handle edge data based on encoder edge weighting type
        edge_data = None
        if hasattr(self.encoder, ENCODER_DEFS.EDGE_WEIGHTING_TYPE):

            if self.encoder.edge_weighting_type == EDGE_WEIGHTING_TYPE.LEARNED_ENCODER:
                # Learnable edge encoder - pass edge attributes for supervision edges
                edge_attr = getattr(data, NAPISTU_DATA.EDGE_ATTR, None)
                if edge_attr is not None:
                    edge_data = edge_attr[train_indices]
                else:
                    logger.warning(
                        f"No edge attributes present in NapistuData despite edge weighting type being {EDGE_WEIGHTING_TYPE.LEARNED_ENCODER}. Falling back to uniform message passing."
                    )
            elif self.encoder.edge_weighting_type == EDGE_WEIGHTING_TYPE.STATIC_WEIGHTS:
                # Static edge weights - pass weights for supervision edges
                if (
                    hasattr(self.encoder, ENCODER_DEFS.EDGE_WEIGHTING_VALUE)
                    and getattr(self.encoder, ENCODER_DEFS.EDGE_WEIGHTING_VALUE)
                    is not None
                ):
                    edge_data = getattr(
                        self.encoder, ENCODER_DEFS.EDGE_WEIGHTING_VALUE
                    )[train_indices]
                else:
                    logger.warning(
                        f"No edge weighting value present in encoder despite edge weighting type being {EDGE_WEIGHTING_TYPE.STATIC_WEIGHTS}. Falling back to uniform message passing."
                    )

        batch_dict = {
            EDGE_PREDICTION_BATCH.X: data.x,
            EDGE_PREDICTION_BATCH.SUPERVISION_EDGES: supervision_edges,
            EDGE_PREDICTION_BATCH.POS_EDGES: pos_edge_index,
            EDGE_PREDICTION_BATCH.NEG_EDGES: neg_edge_index,
            EDGE_PREDICTION_BATCH.EDGE_DATA: edge_data,
        }

        self._validate_data_dims(batch_dict, data)

        return batch_dict

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute BCE loss for edge prediction.

        Steps:
        1. Encode nodes using supervision edges
        2. Score positive and negative edges
        3. Compute binary cross-entropy loss
        """
        # Encode nodes with edge data
        z = self.encoder.encode(
            batch[EDGE_PREDICTION_BATCH.X],
            batch[EDGE_PREDICTION_BATCH.SUPERVISION_EDGES],
            batch.get(EDGE_PREDICTION_BATCH.EDGE_DATA, None),
        )

        # Score positive and negative edges
        pos_scores = self.head(z, batch[EDGE_PREDICTION_BATCH.POS_EDGES])
        neg_scores = self.head(z, batch[EDGE_PREDICTION_BATCH.NEG_EDGES])

        # Binary classification loss
        pos_loss = self.loss_fn(pos_scores, torch.ones_like(pos_scores))
        neg_loss = self.loss_fn(neg_scores, torch.zeros_like(neg_scores))

        return pos_loss + neg_loss

    def compute_metrics(
        self,
        data: NapistuData,
        split: str = TRAINING.VALIDATION,
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics (AUC, AP, etc.).

        This runs in eval mode (no gradients).
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        self.eval()
        with torch.no_grad():
            # Prepare batch
            batch = self.prepare_batch(data, split=split)

            # Encode nodes
            z = self.encoder.encode(
                batch[EDGE_PREDICTION_BATCH.X],
                batch[EDGE_PREDICTION_BATCH.SUPERVISION_EDGES],
                batch.get(EDGE_PREDICTION_BATCH.EDGE_DATA, None),
            )

            # Score positive and negative edges
            pos_scores = torch.sigmoid(
                self.head(z, batch[EDGE_PREDICTION_BATCH.POS_EDGES])
            )
            neg_scores = torch.sigmoid(
                self.head(z, batch[EDGE_PREDICTION_BATCH.NEG_EDGES])
            )

            # Combine predictions and labels
            y_pred = torch.cat([pos_scores, neg_scores]).cpu().numpy()
            y_true = torch.cat(
                [
                    torch.ones(pos_scores.size(0)),
                    torch.zeros(neg_scores.size(0)),
                ]
            ).numpy()

            # Compute metrics
            results = {}
            if "auc" in self.metrics:
                results["auc"] = roc_auc_score(y_true, y_pred)
            if "ap" in self.metrics:
                results["ap"] = average_precision_score(y_true, y_pred)

            return results

    def predict_edge_scores(
        self,
        data: NapistuData,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict scores for specific edge pairs.

        Useful for predicting on new/unseen edges.

        Parameters
        ----------
        data : NapistuData
            Graph data (for node features and structure)
        edge_index : torch.Tensor
            Edge pairs to score [2, num_edges]

        Returns
        -------
        torch.Tensor
            Edge scores [num_edges]

        Examples
        --------
        >>> # Predict on new edge candidates
        >>> task = EdgePredictionTask(encoder, head)
        >>> new_edges = torch.tensor([[0, 1], [2, 3], [4, 5]]).T
        >>> scores = task.predict_edge_scores(data, new_edges)
        """
        self.eval()
        with torch.no_grad():
            # load a fixed tensor for weights if one exists
            if (
                hasattr(self.encoder, "static_edge_weights")
                and self.encoder.static_edge_weights is not None
            ):
                edge_data = self.encoder.static_edge_weights
            else:
                edge_data = getattr(data, "edge_attr", None)

            # Encode nodes
            z = self.encoder.encode(
                data.x,
                data.edge_index,
                edge_data,
            )

            # Score the specified edges
            scores = torch.sigmoid(self.head(z, edge_index))

            return scores

    # private methods

    def _ensure_negative_sampler(self, data: NapistuData):
        """
        Lazy initialization of negative sampler on first call.

        Always initializes sampler (even without strata) to get degree-weighted sampling.
        """
        if self._sampler_initialized:
            return

        logger.info(
            f"Initializing negative sampler with strategy: {self.neg_sampling_strategy}"
        )

        # Get encoded edge strata (or single strata if None)
        encoded_edge_strata = _get_encoded_edge_strata(data, self.edge_strata)

        # Get training edges and their strata
        if hasattr(data, "train_mask"):
            # Transductive
            train_mask_cpu = data.train_mask.cpu()
            train_edges = data.edge_index[:, train_mask_cpu].cpu()
            edge_strata = encoded_edge_strata[train_mask_cpu]
        else:
            # Inductive (data is already train split)
            train_edges = data.edge_index.cpu()
            edge_strata = encoded_edge_strata

        # Initialize sampler (always, even without strata)
        self.negative_sampler = NegativeSampler(
            edge_index=train_edges,
            edge_strata=edge_strata,
            sampling_strategy=self.neg_sampling_strategy,
            oversample_ratio=1.2,
            max_oversample_ratio=2.0,
        )

        num_strata = self.negative_sampler.strata.numel()
        if num_strata == 1:
            logger.info(
                f"Initialized negative sampler: single strata, "
                f"{self.neg_sampling_strategy} strategy"
            )
        else:
            logger.info(
                f"Initialized strata-constrained negative sampler: "
                f"{num_strata} strata, {self.neg_sampling_strategy} strategy"
            )

        self._sampler_initialized = True

    def _predict_impl(self, data: NapistuData) -> torch.Tensor:
        """
        Predict scores for all edges in data.

        This is for inference - no training, no negative sampling.
        """
        if (
            hasattr(self.encoder, "static_edge_weights")
            and self.encoder.static_edge_weights is not None
        ):
            edge_data = self.encoder.static_edge_weights
        else:
            edge_data = getattr(data, "edge_attr", None)

        # Encode nodes using all edges
        z = self.encoder.encode(
            data.x,
            data.edge_index,
            edge_data,
        )

        # Score all edges
        scores = torch.sigmoid(self.head(z, data.edge_index))

        return scores

    def _validate_data_dims(
        self, batch_dict: Dict[str, Optional[torch.Tensor]], data: NapistuData
    ) -> None:
        """Check that the dimensions of the tensors in the batch_dict are compatible."""

        if batch_dict[EDGE_PREDICTION_BATCH.X].shape != (
            data.num_nodes,
            data.num_node_features,
        ):
            raise ValueError(
                f"x shape mismatch: {batch_dict[EDGE_PREDICTION_BATCH.X].shape} != ({data.num_nodes}, {data.num_node_features})"
            )

        n_supervision_edges = data.train_mask.sum().item()
        if batch_dict[EDGE_PREDICTION_BATCH.SUPERVISION_EDGES].shape != (
            2,
            n_supervision_edges,
        ):
            raise ValueError(
                f"supervision_edges shape mismatch: {batch_dict[EDGE_PREDICTION_BATCH.SUPERVISION_EDGES].shape} != (2, {n_supervision_edges})"
            )

        n_pos_edges = batch_dict[EDGE_PREDICTION_BATCH.POS_EDGES].shape[1]
        if batch_dict[EDGE_PREDICTION_BATCH.POS_EDGES].shape != (2, n_pos_edges):
            raise ValueError(
                f"pos_edges shape mismatch: {batch_dict[EDGE_PREDICTION_BATCH.POS_EDGES].shape} != (2, {n_pos_edges})"
            )

        n_neg_edges = batch_dict[EDGE_PREDICTION_BATCH.NEG_EDGES].shape[1]
        if batch_dict[EDGE_PREDICTION_BATCH.NEG_EDGES].shape != (2, n_neg_edges):
            raise ValueError(
                f"neg_edges shape mismatch: {batch_dict[EDGE_PREDICTION_BATCH.NEG_EDGES].shape} != (2, {n_neg_edges})"
            )

        if n_pos_edges != n_neg_edges:
            logger.warning(
                f"pos_edges and neg_edges have different number of edges: {n_pos_edges} != {n_neg_edges}"
            )

        if batch_dict[EDGE_PREDICTION_BATCH.EDGE_DATA] is not None:
            if (
                batch_dict[EDGE_PREDICTION_BATCH.EDGE_DATA].shape[0]
                != n_supervision_edges
            ):
                raise ValueError(
                    f"edge_data shape mismatch: {batch_dict[EDGE_PREDICTION_BATCH.EDGE_DATA].shape[0]} edge_data entries versus {n_supervision_edges} supervision edges"
                )
        return None


def get_edge_strata_from_artifacts(
    stratify_by: str,
    artifacts: Dict[str, Any],
) -> Optional[pd.Series]:
    """
    Extract edge_strata from loaded artifacts dictionary.

    Parameters
    ----------
    stratify_by : str
        Name of the stratification artifact (e.g., "edge_strata_by_species_type")
        or "none" for no stratification.
    artifacts : Dict[str, Any]
        Dictionary of loaded artifacts (e.g., from DataModule.other_artifacts)

    Returns
    -------
    pd.Series or None
        Edge strata series if available, None otherwise

    Examples
    --------
    >>> artifacts = {"edge_strata_by_species_type": edge_strata_df}
    >>> edge_strata = get_edge_strata_from_artifacts(
    ...     stratify_by="edge_strata_by_species_type",
    ...     artifacts=artifacts
    ... )
    """
    if stratify_by == "none":
        return None

    try:
        stratify_by = ensure_stratify_by_artifact_name(stratify_by)
    except ValueError:
        logger.warning(
            f"Invalid stratify_by value: {stratify_by}. Must be one of: {VALID_STRATIFY_BY} | {STRATIFY_BY_ARTIFACT_NAMES}"
        )
        return None

    if stratify_by in artifacts:
        logger.info(f"Loaded edge_strata artifact: {stratify_by}")
        edge_strata_df = artifacts[stratify_by]
        return ensure_strata_series(edge_strata_df)
    else:
        logger.warning(
            f"Stratify by '{stratify_by}' specified but artifact not found. "
            f"Available artifacts: {list(artifacts.keys())}. "
            f"Proceeding with single category."
        )
        return None


def _get_encoded_edge_strata(
    napistu_data: NapistuData,
    edge_strata: Optional[Union[pd.Series, pd.DataFrame]] = None,
) -> torch.Tensor:
    """
    Encode edge strata into integer categories.

    If edge_strata is None, returns all ones (single category).
    This still enables degree-weighted sampling.

    Parameters
    ----------
    napistu_data : NapistuData
        Graph data
    edge_strata : pd.Series or pd.DataFrame, optional
        Edge categories. If DataFrame, must have single column named "edge_strata".
        If None, uses single category.

    Returns
    -------
    torch.Tensor
        Integer-encoded categories [num_edges]
    """
    if edge_strata is None:
        # Single category - still enables degree-weighted sampling
        encoded_strata = torch.zeros(len(napistu_data.edge_index[0]), dtype=torch.long)
        logger.info("No edge_strata provided - using single category")
    else:
        # Ensure edge_strata is a Series (handles both Series and DataFrame cases)
        strata_series = ensure_strata_series(edge_strata)
        validate_edge_strata_alignment(napistu_data, strata_series)
        encoded_strata, _ = _prepare_discrete_labels(
            strata_series, missing_value="other"
        )
        unique_categories = torch.unique(encoded_strata)
        logger.info(f"Encoded {len(unique_categories)} unique edge strata")

    return encoded_strata
