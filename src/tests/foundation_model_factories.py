"""Shared factories and utilities for foundation model tests."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from napistu.ontologies.constants import ONTOLOGIES

from napistu_torch.load.constants import FM_DEFS
from napistu_torch.load.foundation_models import (
    AttentionLayer,
    AttentionPatternsInputs,
    DatasetGeneEmbeddings,
    FoundationModel,
    FoundationModels,
    FoundationModelWeights,
    GeneEmbeddings,
    GeneEmbeddingsSet,
    LayerwiseAttentionInputs,
    ModelMetadata,
)

# ---------------------------------------------------------------------------
# Primitive utilities  (no dependencies on other factories)
# ---------------------------------------------------------------------------

SCOPING_TEST_GENES = [
    "ENSG00000000001",
    "ENSG00000000002",
    "ENSG00000000003",
]


def make_gene_ids(n: int, prefix: str = "ENSG", start: int = 0) -> List[str]:
    return [f"{prefix}{str(i).zfill(5)}" for i in range(start, start + n)]


def make_gene_annotations(
    gene_ids: List[str],
    vocab_names: Optional[List[str]] = None,
    symbols: Optional[List[str]] = None,
) -> pd.DataFrame:
    if vocab_names is None:
        vocab_names = list(gene_ids)
    if symbols is None:
        symbols = [f"SYM_{str(i).zfill(5)}" for i in range(len(gene_ids))]
    return pd.DataFrame(
        {
            FM_DEFS.VOCAB_NAME: vocab_names,
            ONTOLOGIES.ENSEMBL_GENE: gene_ids,
            "symbol": symbols,
        }
    )


def _make_gene_annotations(genes: List[str]) -> pd.DataFrame:
    """Minimal annotations for scoping tests (vocab_name + ensembl_gene only)."""
    return pd.DataFrame(
        {
            FM_DEFS.VOCAB_NAME: list(genes),
            ONTOLOGIES.ENSEMBL_GENE: list(genes),
        }
    )


def _make_gene_emb(
    genes: List[str],
    embed_dim: int = 4,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    category: Optional[str] = None,
    layer_idx: Optional[int] = None,
) -> GeneEmbeddings:
    """Minimal GeneEmbeddings for scoping and unit tests."""
    n = len(genes)
    return GeneEmbeddings(
        embedding=np.arange(n * embed_dim, dtype=np.float64).reshape(n, embed_dim),
        ordered_gene_ids=list(genes),
        gene_annotations=_make_gene_annotations(genes),
        model_name=model_name,
        layer_idx=layer_idx,
        dataset_name=dataset_name,
        category=category,
    )


# ---------------------------------------------------------------------------
# DGE utilities  (depend only on primitives above)
# ---------------------------------------------------------------------------


def _make_layer_grid_dge(
    *,
    n_layers: int,
    n_categories: int,
    gene_ids: List[str],
    model_name: str,
    embed_dim: int = 8,
    layers_per_emb: Optional[List[int]] = None,
    model_variant: Optional[str] = None,
    dataset_name: str = "ds1",
    gene_annotations: Optional[pd.DataFrame] = None,
) -> DatasetGeneEmbeddings:
    """Expression embeddings: one matrix per (category, layer) pair."""
    annotations = (
        gene_annotations
        if gene_annotations is not None
        else make_gene_annotations(gene_ids)
    )
    rng = np.random.default_rng(42)
    mats: List[GeneEmbeddings] = []
    layers_cycle = (
        layers_per_emb if layers_per_emb is not None else list(range(n_layers))
    )

    for ci in range(n_categories):
        for layer in layers_cycle:
            emb_mx = rng.standard_normal((len(gene_ids), embed_dim)).astype(np.float32)
            mats.append(
                GeneEmbeddings(
                    embedding=emb_mx,
                    ordered_gene_ids=annotations[FM_DEFS.VOCAB_NAME].tolist(),
                    gene_annotations=annotations,
                    model_name=model_name,
                    model_variant=model_variant,
                    layer_idx=layer,
                    dataset_name=dataset_name,
                    category=f"cluster_{ci}",
                )
            )

    ges = GeneEmbeddingsSet.from_gene_embeddings(mats)
    return DatasetGeneEmbeddings({dataset_name: ges})


def _make_all_layer_idx_none_dge(
    *,
    gene_ids: List[str],
    n_embeddings: int = 2,
    embed_dim: int = 8,
    dataset_key: str = "ds1",
) -> DatasetGeneEmbeddings:
    """Legacy-style embeddings with layer_idx=None (for validation failure tests)."""
    annotations = make_gene_annotations(gene_ids)
    rng = np.random.default_rng(0)
    mats: List[GeneEmbeddings] = []
    for i in range(n_embeddings):
        emb_mx = rng.standard_normal((len(gene_ids), embed_dim)).astype(np.float32)
        mats.append(
            GeneEmbeddings(
                embedding=emb_mx,
                ordered_gene_ids=list(gene_ids),
                gene_annotations=annotations,
                model_name="TestModel",
                layer_idx=None,
                dataset_name=dataset_key,
                category=f"cluster_{i}",
            )
        )
    ges = GeneEmbeddingsSet.from_gene_embeddings(mats)
    return DatasetGeneEmbeddings({dataset_key: ges})


def _metadata_dict_from_model(fm: FoundationModel) -> dict:
    return {
        FM_DEFS.MODEL_NAME: fm.model_name,
        FM_DEFS.MODEL_VARIANT: fm.model_variant,
        FM_DEFS.N_GENES: fm.n_genes,
        FM_DEFS.N_VOCAB: fm.n_vocab,
        FM_DEFS.ORDERED_VOCABULARY: fm.ordered_vocabulary,
        FM_DEFS.EMBED_DIM: fm.embed_dim,
        FM_DEFS.N_LAYERS: fm.n_layers,
        FM_DEFS.N_HEADS: fm.n_heads,
    }


def _clone_fm_with_dge(
    fm: FoundationModel, dge: DatasetGeneEmbeddings
) -> FoundationModel:
    return FoundationModel(
        weights=fm.weights,
        gene_annotations=fm.gene_annotations,
        model_metadata=_metadata_dict_from_model(fm),
        dataset_gene_embeddings=dge,
    )


# ---------------------------------------------------------------------------
# FoundationModel factories  (depend on DGE utilities)
# ---------------------------------------------------------------------------


def make_attention_layer(
    embed_dim: int = 16,
    layer_idx: int = 0,
    seed: int = 42,
) -> AttentionLayer:
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(embed_dim)
    return AttentionLayer(
        layer_idx=layer_idx,
        W_q=rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * scale,
        W_k=rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * scale,
        W_v=rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * scale,
        W_o=rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * scale,
    )


def make_foundation_model(
    n_genes: int = 20,
    embed_dim: int = 16,
    n_layers: int = 2,
    n_heads: int = 2,
    model_name: str = "TestModel",
    model_variant: Optional[str] = None,
    gene_ids: Optional[List[str]] = None,
    vocab_names: Optional[List[str]] = None,
    extra_vocab: int = 2,
    seed: int = 42,
) -> FoundationModel:
    rng = np.random.default_rng(seed)

    if gene_ids is None:
        gene_ids = make_gene_ids(n_genes)
    else:
        n_genes = len(gene_ids)

    if vocab_names is None:
        vocab_names = list(gene_ids)

    special_tokens = [f"<special_{i}>" for i in range(extra_vocab)]
    full_vocab = vocab_names + special_tokens
    n_vocab = len(full_vocab)

    annotations = make_gene_annotations(gene_ids=gene_ids, vocab_names=vocab_names)
    full_embedding = rng.standard_normal((n_vocab, embed_dim)).astype(np.float32)

    static_ge = GeneEmbeddings(
        embedding=full_embedding[:n_genes],
        ordered_gene_ids=vocab_names,
        gene_annotations=annotations,
        model_name=model_name,
        model_variant=model_variant,
    )

    attention_layers = [
        make_attention_layer(embed_dim=embed_dim, layer_idx=i, seed=seed + i + 1)
        for i in range(n_layers)
    ]

    weights = FoundationModelWeights(
        static_gene_embeddings=static_ge,
        attention_layers=attention_layers,
    )

    metadata = ModelMetadata(
        model_name=model_name,
        model_variant=model_variant,
        n_genes=n_genes,
        n_vocab=n_vocab,
        ordered_vocabulary=full_vocab,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_heads=n_heads,
    )

    return FoundationModel(
        weights=weights,
        gene_annotations=annotations,
        model_metadata=metadata,
    )


def make_foundation_model_pair(
    n_shared: int = 15,
    n_unique_a: int = 5,
    n_unique_b: int = 5,
    embed_dim: int = 16,
    n_layers_a: int = 2,
    n_layers_b: int = 2,
    n_heads: int = 2,
    model_name_a: str = "ModelA",
    model_name_b: str = "ModelB",
    use_different_vocab: bool = False,
    seed: int = 42,
) -> Tuple[FoundationModel, FoundationModel, List[str]]:
    shared_ids = make_gene_ids(n_shared, prefix="ENSG", start=0)
    unique_a_ids = make_gene_ids(n_unique_a, prefix="ENSG", start=n_shared)
    unique_b_ids = make_gene_ids(n_unique_b, prefix="ENSG", start=n_shared + n_unique_a)

    gene_ids_a = unique_a_ids + shared_ids
    gene_ids_b = shared_ids + unique_b_ids

    vocab_names_b = None
    if use_different_vocab:
        vocab_names_b = [f"SYM_{str(i).zfill(5)}" for i in range(len(gene_ids_b))]

    model_a = make_foundation_model(
        gene_ids=gene_ids_a,
        embed_dim=embed_dim,
        n_layers=n_layers_a,
        n_heads=n_heads,
        model_name=model_name_a,
        seed=seed,
    )
    model_b = make_foundation_model(
        gene_ids=gene_ids_b,
        vocab_names=vocab_names_b,
        embed_dim=embed_dim,
        n_layers=n_layers_b,
        n_heads=n_heads,
        model_name=model_name_b,
        seed=seed + 1000,
    )

    return model_a, model_b, shared_ids


def make_foundation_models(
    n_models: int = 3,
    n_shared: int = 15,
    n_unique_per_model: int = 5,
    embed_dim: int = 16,
    n_layers: int = 2,
    n_heads: int = 2,
    seed: int = 42,
) -> Tuple[FoundationModels, List[str]]:
    shared_ids = make_gene_ids(n_shared, prefix="ENSG", start=0)
    models = []
    for i in range(n_models):
        unique_start = n_shared + i * n_unique_per_model
        unique_ids = make_gene_ids(
            n_unique_per_model, prefix="ENSG", start=unique_start
        )
        gene_ids = unique_ids + shared_ids
        models.append(
            make_foundation_model(
                gene_ids=gene_ids,
                embed_dim=embed_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                model_name=f"TestModel{i}",
                seed=seed + i * 1000,
            )
        )
    return FoundationModels(models=models), shared_ids


# ---------------------------------------------------------------------------
# LayerwiseAttentionInputs factories  (depend on FoundationModel factories)
# ---------------------------------------------------------------------------


def make_attended_embeddings(
    n_genes: int = 10,
    embed_dim: int = 16,
    n_layers: int = 2,
    n_heads: int = 2,
    model_name: str = "TestModel",
    category: str = "cluster_0",
    dataset_name: str = "ds1",
    seed: int = 42,
) -> LayerwiseAttentionInputs:
    gene_ids = make_gene_ids(n_genes)
    model = make_foundation_model(
        n_genes=n_genes,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        model_name=model_name,
        gene_ids=gene_ids,
        seed=seed,
    )
    dge = _make_layer_grid_dge(
        n_layers=n_layers,
        n_categories=1,
        gene_ids=gene_ids,
        embed_dim=embed_dim,
        model_name=model.model_name,
        model_variant=model.model_variant,
        dataset_name=dataset_name,
    )
    ge_set = dge[dataset_name]
    residual_map = {
        ge.layer_idx: ge for ge in ge_set.values() if ge.category == category
    }
    return LayerwiseAttentionInputs(
        residual_stream_embeddings=residual_map,
        foundation_model=model,
    )


def make_attended_embeddings_set(
    n_models: int = 3,
    n_shared: int = 15,
    n_unique_per_model: int = 5,
    embed_dim: int = 16,
    n_layers: int = 2,
    n_heads: int = 2,
    seed: int = 42,
) -> Tuple[AttentionPatternsInputs, List[str]]:
    fm, shared_ids = make_foundation_models(
        n_models=n_models,
        n_shared=n_shared,
        n_unique_per_model=n_unique_per_model,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        seed=seed,
    )
    models_with_dge: List[FoundationModel] = []
    for model in fm.models:
        gene_ids = model.gene_annotations[ONTOLOGIES.ENSEMBL_GENE].tolist()
        dge = _make_layer_grid_dge(
            n_layers=model.n_layers,
            n_categories=2,
            gene_ids=gene_ids,
            embed_dim=model.embed_dim,
            model_name=model.model_name,
            model_variant=model.model_variant,
            dataset_name="ds1",
        )
        models_with_dge.append(_clone_fm_with_dge(model, dge))
    fm_with_dge = FoundationModels(models=models_with_dge)
    attended_set = AttentionPatternsInputs.from_expression(
        fm_with_dge,
        dataset_name="ds1",
        category="cluster_0",
        verbose=False,
    )
    return attended_set, shared_ids
