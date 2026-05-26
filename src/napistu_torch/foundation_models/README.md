1. FM-centric / ETL view (serialization, single-model directory)

Purpose: isolate environments, freeze weights, stash whatever the forward pass emitted —
no requirement that different categories/clusters share row order or gene set.

For **scFoundation**, residuals use a **category-specific vocabulary**: only genes whose cluster-mean residual is finite after ``min_cells_per_gene_embedding`` gating.

```text
FoundationModelStore  (one directory per model variant: disk layout + index only)
    weights.npz              # static embeddings + attention weights
    metadata.json            # annotations + ModelMetadata payloads
    residuals_index.yaml     # datasets → categories → stem
    residuals/
        {stem}.npz           # layer_0, layer_1, … arrays for one (dataset, category)
        {stem}_metadata.json # parallel list of per-layer GeneEmbeddings-sidecar dicts

ETL/runtime contract (conceptual):
    FoundationModel ← load(store)          # weights + gene table + dims; optional .store handle
    Residual writes: List[GeneEmbeddings] → store.save_residuals(...)
        • each GeneEmbeddings has dataset_name, category, layer_idx, embedding, annotations
        • categories are saved independently — no enforced common vocabulary across categories
```

During ETL you emit GeneEmbeddings one per transformer layer: each artifact on disk corresponds to (dataset_name, category, layer_idx). The helper API may take a flat List[GeneEmbeddings], but FoundationModelStore.save_residuals groups by (dataset_name, category) and writes residuals/{stem}.npz with keys like layer_0, layer_1, … plus a sidecar {stem}_metadata.json — so every dataset × category × layer is persisted, not “one blob per dataset.

2. Analysis-oriented view (comparison, aligned geometry)

Purpose: chosen context (usually same dataset_name + category, optionally multiple models) → explicit intersection/reorder → structures that assume one gene order across members.

```text
GeneEmbeddingsSet   # built here: aligns common genes / row order across supplied GeneEmbeddings
    GeneEmbeddings  # still one 2D matrix + its GeneAnnotations slice (now co-registered)

FoundationModels
    FoundationModel  # each may load_category_residuals(dataset, category) → Dict[int, GeneEmbeddings]
                     # aligned set is assembled from those dicts (+ align_on), not read from disk pre-aligned

AttentionPatternsInputs
    attended_embeddings : keyed (model_full_name × category-style group)
        LayerwiseAttentionInputs
            residual_stream_embeddings: Dict[int, GeneEmbeddings]
            foundation_model : FoundationModel  # attention machinery for this group only
```

Anything that assumes shared indices (GeneEmbeddingsSet, attention comparison, correlations) belongs in this lane; omit it from the ETL diagram so nobody thinks clusters were aligned at save time.