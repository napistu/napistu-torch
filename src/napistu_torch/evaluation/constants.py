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
