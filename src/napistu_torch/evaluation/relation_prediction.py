"""
Evaluation functions for relation prediction and relation-stratified loss.

This module provides functions for evaluating relation prediction performance,
including confusion matrices, correlation analysis, and comparison with
external truth datasets like PerturbSeq.

Public Functions
----------------
calculate_relation_type_confusion_and_correlation(model, napistu_data, normalize='true')
    Calculate the confusion matrix for the relation types in the test set.
compare_relation_type_predictions_to_perturbseq_truth(model, relation_type_indices, perturbseq_edgelist_tensor, napistu_data, distinct_perturbseq_pairs, distinct_harmonizome_perturbseq_interactions, perturbseq_order, relation_type_order, normalize=None, device=None)
    Compare model predictions to PerturbSeq truth data.
get_perturbseq_edgelist_tensor(distinct_perturbseq_pairs, napistu_data)
    Convert PerturbSeq edge list to tensor format aligned with NapistuData.
summarize_relation_type_aucs(relation_type_aucs, relation_type_order)
    Summarize relation type AUCs as a DataFrame.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningModule
from scipy.stats import chi2_contingency

from napistu_torch.ml.constants import METRIC_SUMMARIES
from napistu_torch.napistu_data import NapistuData
from napistu_torch.utils.tensor_utils import (
    compute_confusion_matrix,
    compute_correlation_matrix,
)


def calculate_relation_type_confusion_and_correlation(
    model: LightningModule, napistu_data: NapistuData, normalize: Optional[str] = "true"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ "
    Calculate the confusion matrix for the relation types in the test set.

    Parameters
    ----------
    model: LightningModule
        The model to evaluate.
    napistu_data: NapistuData
        The NapistuData instance to evaluate on.
    normalize: str
        The normalization method to use for the confusion matrix. Defaults to 'true'.
        Normalization mode for confusion matrix:
        - 'true': normalize over true labels (rows sum to 1)
          Shows recall-like metrics: proportion of each true class predicted as each class
        - 'pred': normalize over predicted labels (columns sum to 1)
          Shows precision-like metrics: proportion of each predicted class from each true class
        - 'all': normalize over all samples (entire matrix sums to 1)
          Shows overall proportion of samples in each true/pred combination
        - None: no normalization, returns raw counts

    Returns
    -------
    confusion_matrix: torch.Tensor
        A numpy array of shape [num_relation_types, num_relation_types] containing the confusion matrix.
    correlation_matrix: torch.Tensor
        A numpy array of shape [num_relation_types, num_relation_types] containing the correlation matrix.
    """

    test_edges = napistu_data.edge_index[:, napistu_data.test_mask]
    test_relation_types = napistu_data.relation_type[napistu_data.test_mask]
    relation_type_indices = {
        napistu_data.relation_manager.label_names[i]: i
        for i in range(len(napistu_data.relation_manager.label_names))
    }

    relation_type_predictions = _score_relation_types(
        model, relation_type_indices, test_edges, napistu_data
    )

    relation_type_labels = test_relation_types

    # 1. Compute confusion matrix (raw counts)
    confusion_matrix = compute_confusion_matrix(
        relation_type_predictions,
        relation_type_labels,
        normalize=normalize,  # normalize by true labels
    )

    correlation_matrix, _ = compute_correlation_matrix(relation_type_predictions)

    return confusion_matrix, correlation_matrix


def compare_relation_type_predictions_to_perturbseq_truth(
    model: LightningModule,
    relation_type_indices: dict[str, int],
    perturbseq_edgelist_tensor: torch.Tensor,
    napistu_data: NapistuData,
    distinct_perturbseq_pairs: pd.DataFrame,
    distinct_harmonizome_perturbseq_interactions: pd.DataFrame,
    perturbseq_order: list[str],
    relation_type_order: list[str],
    normalize: Optional[str] = "true",
) -> tuple[pd.DataFrame, float]:
    """
    Compare the relation-type predictions of a model to the ground-truth PerturbSeq data.

    Parameters
    ----------
    model: LightningModule
        The model to use for scoring.
    relation_type_indices: dict[str, int]
        A dictionary mapping relation_type names to their indices in the relation_type embedding.
    perturbseq_edgelist_tensor: torch.Tensor
        A tensor of shape (2, num_edges) containing the source and target species_ids for each edge.
    napistu_data: NapistuData
        The NapistuData instance containing the graph structure.
    distinct_perturbseq_pairs: pd.DataFrame
        A dataframe containing the unique source and target species_ids for each edge.
    distinct_harmonizome_perturbseq_interactions: pd.DataFrame
        A dataframe containing the source and target species_ids for each edge, dataset, and perturbation type.
    perturbseq_order: list[str]
        The order of the perturbseq relation_types.
    relation_type_order: list[str]
        The order of the relation_types.
    normalize: str
        The normalization method to use for the counts matrix. One of 'true', 'pred', or None. Defaults to 'true'.

    Returns
    -------
    normalized_predictions_w_realized_regulation_counts: pd.DataFrame
        A dataframe of shape (num_relation_types, 2) containing the normalized counts of each relation_type x predicted_direction pair.
    log10_p_value: float
        The log10 of the p-value for the chi-square test of independence between the relation_type predictions and the ground-truth PerturbSeq data.
    """

    relation_type_predictions = _score_relation_types(
        model, relation_type_indices, perturbseq_edgelist_tensor, napistu_data
    )

    predictions_w_realized_regulation = (
        # all unique source-target pairs
        distinct_perturbseq_pairs.join(
            # merge relation_type-specific scores for the types in PERTURBSEQ_RELATION_TYPES via aligned rows
            pd.DataFrame(
                relation_type_predictions.numpy(), columns=relation_type_indices.keys()
            )
        )
        # merge relation-type-specific source-target pairs with realized PerturbSeq regulation (1-to-many)
        .merge(
            distinct_harmonizome_perturbseq_interactions,
            on=["perturbed_species_id", "target_species_id"],
        ).assign(
            # identify the relation_type with the highest predicted score
            predicted_relation_type=lambda df: df[relation_type_order].idxmax(axis=1)
        )
    )

    predictions_w_realized_regulation_counts = (
        predictions_w_realized_regulation
        # tabulate counts of each relation_type x predicted_direction pair and pivot to wide form
        .pivot_table(
            index="predicted_relation_type",
            columns="perturbseq_prediction",
            values="target_species_id",  # or any other column
            aggfunc="count",
            fill_value=0,
        )
        # reorder columns to match PERTURBSEQ_RELATION_TYPES
        [perturbseq_order]
        # reorder rows to match PERTURBSEQ_RELATION_TYPES
        .reindex(relation_type_order)
    )

    if normalize is not None:
        if normalize == "true":
            normalized_predictions_w_realized_regulation_counts = (
                predictions_w_realized_regulation_counts.apply(
                    lambda col: col / col.sum(), axis=0
                )
            )
        elif normalize == "pred":
            normalized_predictions_w_realized_regulation_counts = (
                predictions_w_realized_regulation_counts.apply(
                    lambda row: row / row.sum(), axis=1
                )
            )
        else:
            raise ValueError(f"Invalid normalization method: {normalize}")
    else:
        normalized_predictions_w_realized_regulation_counts = (
            predictions_w_realized_regulation_counts
        )

    chi2_stat, _, _, _ = chi2_contingency(
        predictions_w_realized_regulation_counts.values
    )

    # For very large chi-square values, use asymptotic approximation
    # P-value ≈ exp(-chi2/2) / sqrt(2*pi*chi2) for large chi2 with dof=1
    # log10(P-value) ≈ -chi2/(2*ln(10)) - 0.5*log10(2*pi*chi2)

    log10_p_value = -(chi2_stat / (2 * np.log(10))) - 0.5 * np.log10(
        2 * np.pi * chi2_stat
    )

    return normalized_predictions_w_realized_regulation_counts, log10_p_value


def get_perturbseq_edgelist_tensor(
    distinct_perturbseq_pairs: pd.DataFrame, name_to_sid_map: pd.DataFrame
) -> torch.Tensor:
    """
    Get a tensor of shape (2, num_edges) containing the source and target species_ids for each edge.

    Parameters
    ----------
    distinct_perturbseq_pairs: pd.DataFrame
        A dataframe containing the unique source and target species_ids for each edge.
    name_to_sid_map: pd.DataFrame
        A dataframe mapping species_ids to integer_ids.

    Returns
    -------
    perturbseq_edgelist_tensor: torch.Tensor
        A tensor of shape (2, num_edges) containing the source and target species_ids for each edge.
    """

    perturbseq_edgelist_tensor = torch.tensor(
        distinct_perturbseq_pairs.merge(
            name_to_sid_map.rename(
                columns={
                    "s_id": "perturbed_species_id",
                    "integer_id": "perturbed_integer_id",
                }
            )[["perturbed_species_id", "perturbed_integer_id"]],
            on=["perturbed_species_id"],
            how="left",
        )
        .merge(
            name_to_sid_map.rename(
                columns={"s_id": "target_species_id", "integer_id": "target_integer_id"}
            )[["target_species_id", "target_integer_id"]],
            on=["target_species_id"],
            how="left",
        )[["perturbed_integer_id", "target_integer_id"]]
        .to_numpy()
        .T
    )

    return perturbseq_edgelist_tensor


def summarize_relation_type_aucs(
    run_summaries: Dict[str, Dict[str, Any]], relation_types: List[str]
) -> pd.DataFrame:
    """
    Summarize the AUCs for each relation type for each experiment.

    Parameters
    ----------
    run_summaries : Dict[str, Dict[str, Any]]
        The run summaries to summarize. As produced by `LocalEvaluationManager.get_run_summary()`.
    relation_types : List[str]
        The relation types to summarize.

    Returns
    -------
    pd.DataFrame
        A dataframe with the AUCs for each relation type for each experiment.
    """

    relation_type_aucs = {}
    for k, v in run_summaries.items():
        relation_type_aucs[k] = {}
        for relation_type in relation_types:
            relation_type_aucs[k][relation_type] = {
                METRIC_SUMMARIES.VAL_AUC: None,
                METRIC_SUMMARIES.TEST_AUC: None,
            }
            val_key = METRIC_SUMMARIES.VAL_AUC + "_" + relation_type
            if val_key in v:
                relation_type_aucs[k][relation_type][METRIC_SUMMARIES.VAL_AUC] = v[
                    val_key
                ]
            test_key = METRIC_SUMMARIES.TEST_AUC + "_" + relation_type
            if test_key in v:
                relation_type_aucs[k][relation_type][METRIC_SUMMARIES.TEST_AUC] = v[
                    test_key
                ]

    # Flatten the nested dictionary into a list of records
    records = []
    for experiment, relation_dict in relation_type_aucs.items():
        for relation_type, metrics in relation_dict.items():
            records.append(
                {
                    "experiment": experiment,
                    "relation_type": relation_type,
                    METRIC_SUMMARIES.VAL_AUC: metrics[METRIC_SUMMARIES.VAL_AUC],
                    METRIC_SUMMARIES.TEST_AUC: metrics[METRIC_SUMMARIES.TEST_AUC],
                }
            )

    # Create DataFrame
    return pd.DataFrame(records)


# private functions


@torch.no_grad()
def _score_relation_types(
    model: torch.nn.Module,
    relation_type_indices: dict[str, int],
    edge_index: torch.Tensor,
    napistu_data: NapistuData,
):
    """
    Predict the scores of each relation_type for the given edge list.

    Parameters
    ----------
    model: LightningModule
        The model to use for scoring.
    relation_type_indices: dict[str, int]
        A dictionary mapping relation_type names to their indices in the relation_type embedding.
    edge_index: torch.Tensor
        The edge index of the graph.
    napistu_data: NapistuData
        The NapistuData instance to use for scoring.

    Returns
    -------
    relation_type_predictions: torch.Tensor
        A tensor of shape (num_edges, num_relation_types) containing the predicted scores for each relation_type.
    """

    task = model.task
    num_edges = edge_index.shape[1]
    num_relation_types = len(relation_type_indices)

    # Pre-allocate output tensor: [num_edges, num_relation_types]
    relation_type_predictions = torch.zeros(
        num_edges, num_relation_types, dtype=torch.float32
    )

    for col_idx, relation_type in enumerate(relation_type_indices):
        relation_type_array = torch.tensor(
            [relation_type_indices[relation_type]] * num_edges, dtype=torch.long
        )
        predictions = task.predict_edge_scores(
            data=napistu_data, edge_index=edge_index, relation_type=relation_type_array
        )
        relation_type_predictions[:, col_idx] = (
            predictions.cpu()
        )  # Move to CPU immediately
        del relation_type_array, predictions  # Explicit cleanup

    return relation_type_predictions
