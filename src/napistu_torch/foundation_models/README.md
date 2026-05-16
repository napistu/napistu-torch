```text
FoundationModels
    FoundationModel / FoundationModelStore (foundation model, and its on-disk assets)
        FoundationModelWeights
            GeneEmbeddings (static)
                GeneAnnotations
            List[AttentionLayer]
        ModelMetadata
    GeneEmbeddingsSet (all embeddings for a given dataset)
        GeneEmbeddings (a single 2D embedding matrix, e.g., of a cell type, cluster, or individual sample)
            GeneAnnotations

AttentionPatternsInputs
    LayerwiseAttentionInputs
        Dict[int, GeneEmbeddings] (per-layer residual stream embeddings)
        FoundationModel (per group; supplies attention weights for that group's layers)
```