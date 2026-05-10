```text
FoundationModels
    FoundationModel
        FoundationModelWeights
            GeneEmbeddings (static)
                GeneAnnotations
            List[AttentionLayer]
        DatasetGeneEmbeddings
            GeneEmbeddingsSet (all embeddings for a given dataset)
                GeneEmbeddings (a single 2D embedding matrix, e.g., of a cell type, cluster, or individual sample)
                    GeneAnnotations
        ModelMetadata

AttentionPatternsInputs
    LayerwiseAttentionInputs
        Dict[int, GeneEmbeddings] (per-layer residual stream embeddings)
        FoundationModel (per group; supplies attention weights for that group's layers)
```