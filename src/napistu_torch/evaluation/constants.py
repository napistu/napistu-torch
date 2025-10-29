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

# stratification

STRATIFY_BY = SimpleNamespace(
    NODE_SPECIES_TYPE="node_species_type",
    NODE_TYPE="node_type",
)

VALID_STRATIFY_BY = list(STRATIFY_BY.__dict__.values())
