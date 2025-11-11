from types import SimpleNamespace

EVALUATION_TENSORS = SimpleNamespace(
    COMPREHENSIVE_PATHWAY_MEMBERSHIPS="comprehensive_pathway_memberships",
)

EVALUATION_TENSOR_DESCRIPTIONS = {
    EVALUATION_TENSORS.COMPREHENSIVE_PATHWAY_MEMBERSHIPS: "Comprehensive source membership from SBML_dfs",
}

PATHWAY_SIMILARITY_DEFS = SimpleNamespace(
    OVERALL="overall",
    OTHER="other",
)

EVALUATION_MANAGER = SimpleNamespace(
    MANIFEST="manifest",
    EXPERIMENT_CONFIG="experiment_config",
)

EDGE_PREDICTION_BY_STRATA_DEFS = SimpleNamespace(
    FROM_ATTRIBUTE="from_attribute",
    TO_ATTRIBUTE="to_attribute",
    FROM_ATTRIBUTE_COUNT="from_attribute_count",
    TO_ATTRIBUTE_COUNT="to_attribute_count",
    EXPECTED_COUNT="expected_count",
    OBSERVED_OVER_EXPECTED="observed_over_expected",
    AVERAGE_PREDICTION_PROBABILITY="average_prediction_probability",
    COUNT="count",
)
