"""Evaluation functions for relation prediction and relation-stratified loss."""

from typing import Any, Dict, List

import pandas as pd

from napistu_torch.ml.constants import METRIC_SUMMARIES


def summarize_relation_type_aucs(
    run_summaries: Dict[str, Dict[str, Any]], relation_types: List[str]
) -> pd.DataFrame:
    """
    Summarize the AUCs for each relation type for each experiment.

    Parameters
    ----------
    run_summaries : Dict[str, Dict[str, Any]]
        The run summaries to summarize. As produced by `EvaluationManager.get_run_summary()`.
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
