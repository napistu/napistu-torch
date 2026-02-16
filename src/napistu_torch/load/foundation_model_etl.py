"""
Functions for ETLing specific virtual cell foundation models.

Public Functions
----------------
populate_lamin_db:
    Populate the lamin database.
process_aidocell:
    Process an AIDOCell model and save the results.
process_scfoundation:
    Process a scFoundation model and save the results.
process_scgpt:
    Process a scGPT model and save the results.
process_scprint:
    Process a scPRINT model and save the results.

Each model has its own public process function and a similar set of private functions:
*_extract_attention_weights:
    Extract the attention weights from the model returning a list of AttentionLayer instances. Called by *_extract_weights.
*_extract_weights:
    Extract the weights from the model returning a FoundationModelWeights instance.
*_get_expression_embeddings:
    Extract the expression embeddings from the model returning a GeneEmbeddingsSet instance.
*_get_gene_embedding_by_cell_type:
    Extract the gene embedding by cell type from the model returning a GeneEmbeddings instance. Called by *_get_expression_embeddings.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from napistu.genomics.scverse_loading import DatasetsConfig
from napistu.ontologies.constants import ONTOLOGIES
from napistu.utils.io_utils import download_wget
from scipy.sparse import issparse

from napistu_torch.load.constants import (
    AIDOCELL_CLASSES,
    AIDOCELL_DEFS,
    CELLXGENE_DEFS,
    FM_DEFAULTS,
    FM_DEFS,
    FOUNDATION_MODEL_NAMES,
    SCFOUNDATION_DEFS,
    SCGPT_DEFS,
    SCPRINT_DEFS,
    VALID_FOUNDATION_MODEL_NAMES,
)
from napistu_torch.load.foundation_models import (
    AttentionLayer,
    DatasetGeneEmbeddings,
    FoundationModel,
    FoundationModelWeights,
    GeneEmbeddings,
    GeneEmbeddingsSet,
)
from napistu_torch.ml.constants import DEVICE
from napistu_torch.utils.optional import (
    require_bionty,
    require_modelgenerator,
    require_scdataloader,
    require_scgpt,
    require_scprint,
    require_torchtext,
)
from napistu_torch.utils.pd_utils import filter_and_reorder_df
from napistu_torch.utils.torch_utils import (
    empty_cache,
    ensure_device,
    memory_manager,
    select_device,
)

# Set up warnings for scGPT
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore")

if TYPE_CHECKING:
    from anndata import AnnData
    from modelgenerator.huggingface_models.scfoundation.pretrainmodels.mae_autobin import (
        MaeAutobin,
    )
    from scgpt.model.transformer_model import TransformerModel
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from scprint.model import scPrint

    pass

logger = logging.getLogger(__name__)


@require_bionty
@require_scdataloader
def populate_lamin_db() -> None:
    """Populate the lamin database.

    Add species, identifiers, and other metadata to the lamin database

    Returns
    -------
    None
    """
    import bionty as bt
    from scdataloader.utils import populate_my_ontology

    # quick check to see if the lamin database is already configured
    organisms = bt.Organism.filter().df()
    human_defined = (
        "NCBITaxon:9606" in organisms["ontology_id"].values
        if len(organisms) > 0
        else False
    )
    if not human_defined:
        logger.info(
            "Populating the full metadata catalog recommended by the scPRINT developers"
        )
        # populate the full metadata catalog recommended by the scPRINT developers
        populate_my_ontology()
    else:
        logger.info("Lamin database already configured")


@require_modelgenerator
def process_aidocell(
    model_class: Any,
    output_dir: str,
    datasets_config: Optional[DatasetsConfig] = None,
    min_cluster_cells: int = FM_DEFAULTS.MIN_CLUSTER_CELLS,
) -> None:
    """Process a given AIDOCell model class and save the results.

    Parameters
    ----------
    model_class : Any
        AIDOCell model class to load. Can be a class or a class name string.
        If string, uses AIDOCELL_CLASSES to look up the backbone.
    output_dir : str
        Output directory to save the results

    Returns
    -------
    None
    """
    # Handle both class and string inputs
    if isinstance(model_class, str):
        model_class = _aidocell_get_backbone(model_class)

    model_class_name = model_class.__name__
    file_prefix = AIDOCELL_DEFS.PREFIX_TEMPLATE.format(
        model_name=AIDOCELL_DEFS.MODEL_NAME, model_class_name=model_class_name
    )

    logger.info(f"Extracting: {model_class_name}")

    # 1. Load model and data
    logger.info("\n1. Loading model and data...")
    model, gene_annotations, model_metadata = _aidocell_load_model_full(model_class)
    logger.info(
        f"   {len(gene_annotations)} genes, {model_metadata[FM_DEFS.N_LAYERS]} layers"
    )

    # 2. Extract weights
    logger.info("2. Extracting weights...")
    weights = _aidocell_extract_weights(model, gene_annotations)
    logger.info(f"   Embeddings: {weights.static_gene_embedding.embed_dim}")
    logger.info(
        f"   Attention weights: {model_metadata[FM_DEFS.N_LAYERS]} layers × 4 matrices (Q,K,V,O)"
    )

    # 3. Extract dataset expression embeddings
    logger.info("3. Extracting dataset expression embeddings...")
    dataset_sets: Dict[str, GeneEmbeddingsSet] = {}
    for config in datasets_config.data.values():
        dataset_sets[config.name] = _aidocell_get_expression_embeddings(
            model,
            config.load_h5ad(),
            gene_annotations,
            dataset_name=config.name,
            dataset_uri=config.uri,
            min_cluster_cells=min_cluster_cells,
        )

    dataset_gene_embeddings = DatasetGeneEmbeddings(dataset_sets)

    # 4. Create FoundationModel and save
    _create_and_save_foundation_model(
        weights,
        gene_annotations,
        model_metadata,
        dataset_gene_embeddings,
        output_dir,
        file_prefix,
    )

    return None


@require_modelgenerator
def process_scfoundation(
    output_dir: str,
    checkpoint_path: Optional[str] = None,
    output_prefix: Optional[str] = None,
    cache_dir: Optional[str] = None,
    datasets_config: Optional[DatasetsConfig] = None,
    min_cluster_cells: int = FM_DEFAULTS.MIN_CLUSTER_CELLS,
) -> None:
    """Process scFoundation checkpoint and save to disk.

    Parameters
    ----------
    output_dir : str
        Directory to save processed model
    checkpoint_path : str, optional
        Path to local checkpoint. If None, downloads from HuggingFace.
    output_prefix : str, optional
        Prefix for output files (default: "scFoundation")
    cache_dir : str, optional
        Cache directory for HuggingFace downloads
    datasets_config : DatasetsConfig, optional
        Datasets configuration
    min_cluster_cells : int, optional
        Minimum number of cells per cluster to include. Clusters smaller than this
        are excluded. Default from FM_DEFAULTS.MIN_CLUSTER_CELLS (10).

    Returns
    -------
    None

    Examples
    --------
    >>> # Download and save
    >>> process_scfoundation(output_dir="./models")
    >>>
    >>> # Process local file
    >>> process_scfoundation(
    ...     output_dir="./models",
    ...     checkpoint_path="./models.ckpt"
    ... )
    """
    # Gene annotations are shared between AIDOCell and scFoundation
    from huggingface_hub import hf_hub_download
    from modelgenerator.huggingface_models.scfoundation.load_scfoundation import (
        convertconfig,
        select_model,
    )

    logger.info("Extracting: scFoundation")

    # Download checkpoint if needed
    if checkpoint_path is None:
        logger.info("\n1. Downloading checkpoint from HuggingFace...")
        checkpoint_path = hf_hub_download(
            repo_id=SCFOUNDATION_DEFS.REPO_ID,
            filename=SCFOUNDATION_DEFS.CHECKPOINT_FILE,
            cache_dir=cache_dir,
        )
    else:
        logger.info("\n1. Loading checkpoint...")

    # Load checkpoint
    logger.info(
        f"Loading scFoundation checkpoint (gene encoder: {SCFOUNDATION_DEFS.GENE_ENCODER})"
    )
    full_checkpoint = torch.load(checkpoint_path, map_location=DEVICE.CPU)

    if SCFOUNDATION_DEFS.GENE_ENCODER not in full_checkpoint:
        raise ValueError(
            f"Gene encoder '{SCFOUNDATION_DEFS.GENE_ENCODER}' not found in checkpoint. "
            f"Available: {list(full_checkpoint.keys())}"
        )

    checkpoint = full_checkpoint[SCFOUNDATION_DEFS.GENE_ENCODER]
    model_data = convertconfig(checkpoint)
    config = model_data["config"]
    model = select_model(config)
    model.load_state_dict(model_data["model_state_dict"])
    model.eval()

    # Extract components
    logger.info("2. Extracting weights...")
    # Gene annotations are shared between AIDOCell and scFoundation
    gene_annotations = _aidocell_load_gene_annotations()
    weights = _scfoundation_extract_weights(model, checkpoint, gene_annotations)
    metadata = _scfoundation_extract_metadata(checkpoint, gene_annotations)
    logger.info(
        f"   {len(gene_annotations)} genes, {metadata[FM_DEFS.N_LAYERS]} layers"
    )
    logger.info(f"   Embeddings: {weights.static_gene_embedding.embed_dim}")
    logger.info(
        f"   Attention weights: {metadata[FM_DEFS.N_LAYERS]} layers × 4 matrices (Q,K,V,O)"
    )

    logger.info("3. Extracting dataset expression embeddings...")
    dataset_sets: Dict[str, GeneEmbeddingsSet] = {}
    for config in datasets_config.data.values():
        dataset_sets[config.name] = _scfoundation_get_expression_embeddings(
            model,
            config.load_h5ad(),
            gene_annotations,
            dataset_name=config.name,
            dataset_uri=config.uri,
            min_cluster_cells=min_cluster_cells,
        )

    dataset_gene_embeddings = DatasetGeneEmbeddings(dataset_sets)

    # Set default prefix if not provided
    if output_prefix is None:
        output_prefix = SCFOUNDATION_DEFS.MODEL_NAME

    # Build model and save
    _create_and_save_foundation_model(
        weights,
        gene_annotations,
        metadata,
        dataset_gene_embeddings,
        output_dir,
        output_prefix,
    )

    return None


@require_scgpt
def process_scgpt(
    model_dir: str,
    output_dir: str,
    annotations_path: Optional[str] = None,
    datasets_config: Optional[DatasetsConfig] = None,
    min_cluster_cells: int = FM_DEFAULTS.MIN_CLUSTER_CELLS,
) -> None:
    """Process the scGPT model and save the results to the output directory.

    Parameters
    ----------
    model_dir : str
        Directory containing the scGPT model files (args.json, best_model.pt, vocab.json)
    output_dir : str
        Output directory to save the results
    annotations_path : str, optional
        Path to gene annotations file. If None, downloads from GENE_IDENTIFIERS_URL
    datasets_config : DatasetsConfig, optional
        Datasets configuration
    min_cluster_cells : int, optional
        Minimum number of cells per cluster to include. Clusters smaller than this
        are excluded. Default from FM_DEFAULTS.MIN_CLUSTER_CELLS (10).

    Returns
    -------
    None
    """
    file_prefix = SCGPT_DEFS.MODEL_NAME

    logger.info("Extracting: scGPT")

    # 1. Download and load gene annotations
    logger.info("\n1. Downloading/loading gene annotations...")
    if annotations_path is None:
        # Default to same directory as model_dir (typically "data" folder)
        data_dir = os.path.dirname(model_dir)
        os.makedirs(data_dir, exist_ok=True)
        annotations_path = os.path.join(data_dir, "scgpt_gene_info.csv")

    if not os.path.isfile(annotations_path):
        logger.info(
            f"   Downloading gene annotations from {SCGPT_DEFS.GENE_IDENTIFIERS_URL}"
        )
        download_wget(SCGPT_DEFS.GENE_IDENTIFIERS_URL, annotations_path)

    gene_annotations = _scgpt_load_gene_annotations(annotations_path)
    logger.info(f"   Loaded {len(gene_annotations)} gene annotations")

    # 2. Load model
    logger.info("2. Loading scGPT model...")
    model, vocab, model_metadata, checkpoint_path = _scgpt_load_model(model_dir)
    logger.info(
        f"   {len(gene_annotations)} genes, {model_metadata[FM_DEFS.N_LAYERS]} layers"
    )

    # 3. Extract weights
    logger.info("3. Extracting weights...")
    weights = _scgpt_extract_weights(
        model=model,
        vocab=vocab,
        model_metadata=model_metadata,
        checkpoint_path=checkpoint_path,
        gene_annotations=gene_annotations,
    )
    logger.info(f"   Embeddings: {weights.static_gene_embedding.embed_dim}")
    logger.info(
        f"   Attention weights: {model_metadata[FM_DEFS.N_LAYERS]} layers × 4 matrices (Q,K,V,O)"
    )

    # 4. Extract dataset expression embeddings
    logger.info("4. Extracting dataset expression embeddings...")
    dataset_sets: Dict[str, GeneEmbeddingsSet] = {}
    for config in datasets_config.data.values():
        dataset_sets[config.name] = _scgpt_get_expression_embeddings(
            model=model,
            adata=config.load_h5ad(),
            gene_annotations=gene_annotations,
            vocab=vocab,
            dataset_name=config.name,
            dataset_uri=config.uri,
            min_cluster_cells=min_cluster_cells,
        )

    dataset_gene_embeddings = DatasetGeneEmbeddings(dataset_sets)

    # 5. Create FoundationModel and save
    _create_and_save_foundation_model(
        weights,
        gene_annotations,
        model_metadata,
        dataset_gene_embeddings,
        output_dir,
        file_prefix,
    )

    return None


@require_scprint
def process_scprint(
    version_key: str,
    output_dir: str,
    model_path: Optional[str] = None,
    datasets_config: Optional[DatasetsConfig] = None,
    min_cluster_cells: int = FM_DEFAULTS.MIN_CLUSTER_CELLS,
) -> None:
    """Process a given scPRINT model version and save the results to the output directory.

    Parameters
    ----------
    version_key : str
        scPRINT version key (e.g., "SMALL", "MEDIUM", "LARGE")
    output_dir : str
        Output directory to save the results
    model_path : str, optional
        Path to directory where models are cached. If None, uses default "data/scPRINT"
    datasets_config : DatasetsConfig, optional
        Datasets configuration
    min_cluster_cells : int, optional
        Minimum number of cells per cluster to include. Clusters smaller than this
        are excluded. Default from FM_DEFAULTS.MIN_CLUSTER_CELLS (10).

    Returns
    -------
    None
    """

    # Get version ID and checkpoint filename from the version key
    version_id = getattr(SCPRINT_DEFS.VERSIONS, version_key)
    file_prefix = f"{SCPRINT_DEFS.MODEL_NAME}_{version_id}"

    # 1. Download and load model
    logger.info(f"Extracting: scPRINT {version_id} ({version_key})")
    checkpoint_file = _scprint_get_checkpoint(version_key, model_path)

    logger.info("Loading scPRINT model")
    model, gene_annotations, model_metadata = _scprint_load_model(
        checkpoint_file, version=version_id
    )
    logger.info(
        f"   {len(gene_annotations)} genes, {model_metadata[FM_DEFS.N_LAYERS]} layers"
    )

    # 2. Extract weights
    logger.info("2. Extracting weights...")
    weights = _scprint_extract_weights(model=model, gene_annotations=gene_annotations)
    logger.info(f"   Embeddings: {weights.static_gene_embedding.embed_dim}")
    logger.info(
        f"   Attention weights: {model_metadata[FM_DEFS.N_LAYERS]} layers × 4 matrices (Q,K,V,O)"
    )

    # 3. Extract dataset expression embeddings
    logger.info("3. Extracting dataset expression embeddings...")
    dataset_sets: Dict[str, GeneEmbeddingsSet] = {}
    for config in datasets_config.data.values():
        dataset_sets[config.name] = _scprint_get_expression_embeddings(
            model,
            config.load_h5ad(),
            gene_annotations,
            dataset_name=config.name,
            dataset_uri=config.uri,
            min_cluster_cells=min_cluster_cells,
        )

    dataset_gene_embeddings = DatasetGeneEmbeddings(dataset_sets)

    # 4. Create FoundationModel and save
    _create_and_save_foundation_model(
        weights,
        gene_annotations,
        model_metadata,
        dataset_gene_embeddings,
        output_dir,
        file_prefix,
    )

    return None


@require_modelgenerator
def _aidocell_extract_attention_weights(model: Any) -> List[AttentionLayer]:
    """Extract core attention weights (Q, K, V, O) from all layers.

    Parameters
    ----------
    model : Any
        The AIDOCell model

    Returns
    -------
    List[AttentionLayer]
        List of AttentionLayer instances
    """
    attention_layers = []
    encoder = model.encoder
    transformer_layers = encoder.encoder.layer
    n_layers = model.get_num_layer()

    for layer_idx in range(n_layers):
        layer = transformer_layers[layer_idx]
        attention_self = layer.attention.self
        attention_output = layer.attention.output

        attention_layers.append(
            AttentionLayer(
                layer_idx=layer_idx,
                W_q=attention_self.query.weight.detach().cpu().numpy(),
                W_k=attention_self.key.weight.detach().cpu().numpy(),
                W_v=attention_self.value.weight.detach().cpu().numpy(),
                W_o=attention_output.dense.weight.detach().cpu().numpy(),
            )
        )

    return attention_layers


@require_modelgenerator
def _aidocell_extract_weights(
    model: Any, gene_annotations: pd.DataFrame
) -> FoundationModelWeights:
    """Extract model weights in the standardized format.

    Parameters
    ----------
    model : Any
        The AIDOCell model
    gene_annotations : pd.DataFrame
        Gene annotations DataFrame

    Returns
    -------
    FoundationModelWeights
        FoundationModelWeights instance containing static_gene_embedding and attention_layers
    """
    encoder = model.encoder
    n_genes = len(gene_annotations)

    with torch.no_grad():
        gene_positions = torch.arange(n_genes)
        gene_embedding_array = encoder.position_embedding(gene_positions).cpu().numpy()

    attention_layers = _aidocell_extract_attention_weights(model)

    gene_embedding = GeneEmbeddings(
        embedding=gene_embedding_array,
        ordered_gene_ids=gene_annotations[FM_DEFS.VOCAB_NAME].tolist(),
        gene_annotations=gene_annotations,
    )

    return FoundationModelWeights(
        static_gene_embedding=gene_embedding, attention_layers=attention_layers
    )


@require_modelgenerator
def _aidocell_format_metadata(model: Any, model_class_name: str) -> Dict:
    """Extract model architecture metadata.

    Parameters
    ----------
    model : Any
        The AIDOCell model
    model_class_name : str
        Name of the model class

    Returns
    -------
    Dict
        Dictionary with model metadata
    """
    encoder = model.encoder
    gene_annotations = _aidocell_load_gene_annotations()
    n_genes = len(gene_annotations)

    # Get vocabulary as list of gene symbols (AIDOCell doesn't have special tokens)
    vocab_list = gene_annotations[FM_DEFS.VOCAB_NAME].tolist()

    return _format_base_metadata(
        model_name=AIDOCELL_DEFS.MODEL_NAME,
        n_genes=n_genes,
        n_vocab=n_genes,  # Same as n_genes for AIDOCell (no special tokens)
        vocab_list=vocab_list,
        embed_dim=int(model.get_embedding_size()),
        n_layers=int(model.get_num_layer()),
        n_heads=int(encoder.config.num_attention_heads),
        model_variant=model_class_name,
        # Additional AIDOCell-specific metadata
        **{AIDOCELL_DEFS.HIDDEN_DIM: int(encoder.config.hidden_size)},
    )


@require_modelgenerator
def _aidocell_get_backbone(class_name: str) -> Any:
    """Get AIDOCell backbone class by name.

    Parameters
    ----------
    class_name : str
        AIDOCell class name (e.g., "aido_cell_3m", "aido_cell_10m", "aido_cell_100m")

    Returns
    -------
    Any
        The AIDOCell backbone class

    Examples
    --------
    >>> backbone = _aidocell_get_backbone("aido_cell_3m")
    >>> model = backbone(...)
    """
    from modelgenerator.backbones import (
        aido_cell_3m,
        aido_cell_10m,
        aido_cell_100m,
    )

    backbone_map = {
        AIDOCELL_CLASSES.THREE_M: aido_cell_3m,
        AIDOCELL_CLASSES.TEN_M: aido_cell_10m,
        AIDOCELL_CLASSES.ONE_HUNDRED_M: aido_cell_100m,
    }

    if class_name not in backbone_map:
        raise ValueError(
            f"Unknown AIDOCell class name: {class_name}. "
            f"Must be one of: {list(backbone_map.keys())}"
        )

    return backbone_map[class_name]


@require_modelgenerator
def _aidocell_get_expression_embeddings(
    model: Any,
    adata: AnnData,
    gene_annotations: pd.DataFrame,
    dataset_name: Optional[str] = None,
    dataset_uri: Optional[str] = None,
    min_cluster_cells: int = None,
) -> GeneEmbeddingsSet:
    """Embed each cell type in an AnnData object as a tensor of shape (n_cells, embed_dim).

    Parameters
    ----------
    model : Any
        The AIDOCell model
    adata : anndata.AnnData
        The AnnData object containing the cells to embed
    gene_annotations : pd.DataFrame
        The gene annotations
    dataset_name : Optional[str] = None
        The name of the dataset
    dataset_uri : Optional[str] = None
        The URI of the dataset
    min_cluster_cells : int, optional
        Minimum number of cells per cluster to include. Clusters smaller than this
        are excluded. If None, then all clusters are included.

    Returns
    -------
    GeneEmbeddingsSet
        Expression embeddings for the dataset
    """

    cluster_embeddings, selected_genes, cell_cluster_dict = (
        _aidocell_get_gene_embedding_by_cell_type(
            model, adata, gene_annotations, min_cluster_cells=min_cluster_cells
        )
    )

    expression_embeddings = _expression_tensor_to_gene_embeddings_set(
        embeddings_3d=cluster_embeddings,
        ordered_genes=selected_genes,
        gene_annotations=gene_annotations,
        category_dict=cell_cluster_dict,
        dataset_name=dataset_name,
        dataset_uri=dataset_uri,
    )

    return expression_embeddings


@require_modelgenerator
def _aidocell_get_gene_embedding_by_cell_type(
    model: Any,
    adata: AnnData,
    gene_annotations: pd.DataFrame,
    min_cluster_cells: Optional[int] = None,
) -> Tuple[torch.Tensor, List[str], Dict[int, str]]:
    """Get the gene embeddings for each cell type in an AnnData object.

    Parameters
    ----------
    model : Any
        The AIDOCell model
    adata : anndata.AnnData
        The AnnData object containing the cells to embed
    gene_annotations : List[str]
        The common genes to use for the embeddings
    min_cluster_cells : int, optional
        Minimum number of cells per cluster to include. Clusters smaller than this
        are excluded. If None, then all clusters are included.

    Returns
    -------
    Tuple[torch.Tensor, List[str], Dict[int, str]]
        Tuple of (cluster_embeddings, selected_genes, cell_cluster_dict)
        - cluster_embeddings : torch.Tensor
            The gene embeddings for each cell type
        - selected_genes : List[str]
            The selected genes (ensembl gene ids) which are shared between the model and the adata object.
        - cell_cluster_dict : Dict[int, str]
            A dictionary mapping cell type indices to cell type names
    """

    # 1. Get indices into model vocabulary
    adata_subset, selected_genes = _aidocell_preprocess_dataset(adata, gene_annotations)

    # 2. Get the position-based embedding for the common genes
    annotations_w_index = gene_annotations.assign(
        position=range(len(gene_annotations))
    ).query("ensembl_gene in @selected_genes")
    annotations_name_to_index = annotations_w_index.set_index(ONTOLOGIES.ENSEMBL_GENE)[
        "position"
    ].to_dict()

    with torch.no_grad():
        gene_positions = [annotations_name_to_index[g] for g in selected_genes]
        pos_embedding = model.encoder.position_embedding(torch.tensor(gene_positions))

    # 3. Track cell types to setup creating cell-type masks
    cell_clusters, cell_cluster_dict = _get_cell_clusters_and_category_dict(
        adata_subset.obs, min_cluster_cells=min_cluster_cells
    )

    # Allocate output
    cluster_embeddings = torch.zeros(
        len(cell_clusters),
        len(selected_genes),
        model.get_embedding_size(),
    )

    # 4. Embed each cell type
    with torch.no_grad():
        for i, cluster in enumerate(cell_clusters["leiden_scVI"]):
            cluster_mask = adata_subset.obs["leiden_scVI"] == cluster
            cluster_adata = adata_subset[cluster_mask]

            logger.info(f"Cluster {i}: {cluster_adata.shape[0]} cells")

            if issparse(cluster_adata.X):
                cluster_expr = torch.from_numpy(cluster_adata.X.toarray())
            else:
                cluster_expr = torch.from_numpy(cluster_adata.X)

            # Use batched processing
            cluster_embeddings[i] = _embed_expression_batch(
                model=model,
                model_type=FOUNDATION_MODEL_NAMES.AIDOCELL,
                expression=cluster_expr,
                gene_emb=pos_embedding,  # Position embeddings serve as "static" gene identity
                batch_size=64,
            )

            logger.info(f"  ✓ Completed cluster {i}")

    return cluster_embeddings, selected_genes, cell_cluster_dict


@require_modelgenerator
def _aidocell_load_gene_annotations() -> pd.DataFrame:
    """Load gene annotations from AIDOCell model.

    This is a flat file which is bundled with the package

    Returns
    -------
    pd.DataFrame
        DataFrame with gene annotations
    """
    import modelgenerator.cell.utils as cell_utils

    load_base = os.path.dirname(os.path.abspath(cell_utils.__file__))
    gene_file = os.path.join(load_base, AIDOCELL_DEFS.GENE_FILE)

    # Load gene symbols
    gene_symbols = pd.read_csv(gene_file, sep="\t")["gene_name"].values

    # Build the mapping from symbols to Ensembl IDs
    gene_map = cell_utils.build_map(gene_symbols)

    # Create the mapping table
    gene_table = pd.DataFrame(
        {
            FM_DEFS.VOCAB_NAME: gene_symbols,
            ONTOLOGIES.SYMBOL: gene_symbols,
            ONTOLOGIES.ENSEMBL_GENE: [
                gene_map.get(x, f"{x}_unknown_ensg") for x in gene_symbols
            ],
        }
    )

    return gene_table


@require_modelgenerator
def _aidocell_load_model(model_class: Any) -> Any:
    """Load AIDOCell model in eval mode.

    Parameters
    ----------
    model_class : Any
        AIDOCell model class to load

    Returns
    -------
    Any
        The AIDOCell model in eval mode
    """
    model = model_class(
        legacy_adapter_type=None, default_config=None, from_scratch=False
    )
    model.eval()
    return model


@require_modelgenerator
def _aidocell_load_model_full(model_class: Any) -> Tuple[Any, pd.DataFrame, Dict]:
    """Load the AIDOCell model and return model, gene annotations, and metadata.

    Parameters
    ----------
    model_class : Any
        AIDOCell model class to load

    Returns
    -------
    Tuple[Any, pd.DataFrame, Dict]
        Tuple of (model, gene_annotations, model_metadata)
    """
    logger.info("Loading AIDOCell model")
    model = _aidocell_load_model(model_class)

    logger.info("Loading gene annotations")
    gene_annotations = _aidocell_load_gene_annotations()

    logger.info("Formatting model metadata")
    model_metadata = _aidocell_format_metadata(
        model, model_class_name=model_class.__name__
    )

    return model, gene_annotations, model_metadata


@require_modelgenerator
def _aidocell_preprocess_dataset(
    adata: AnnData, gene_annotations: pd.DataFrame
) -> Tuple[AnnData, List[str]]:
    """Preprocess an annData object for an AIDOCell model.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to preprocess.
    gene_annotations : pd.DataFrame
        The gene annotations for the AIDOCell model.

    Returns
    -------
    adata_subset : AnnData
        The preprocessed AnnData object.
    selected_genes : List[str]
        The list of genes (ensembl genes) which are shared between the model and the adata object.
    """
    import scanpy as sc

    # preprocess the adata object
    adata_subset = adata.copy()

    # standard AIDOCell preprocessing
    sc.pp.filter_cells(adata_subset, min_genes=500)
    sc.pp.filter_genes(adata_subset, min_cells=3)

    # retain genes which are shared between the model and the adata object
    model_genes = gene_annotations[ONTOLOGIES.ENSEMBL_GENE]  # The 44,756 genes
    selected_genes = [g for g in model_genes if g in adata_subset.var_names]

    adata_subset = adata_subset[:, selected_genes]

    return adata_subset, selected_genes


def _create_and_save_foundation_model(
    weights: FoundationModelWeights,
    gene_annotations: pd.DataFrame,
    model_metadata: Dict,
    dataset_gene_embeddings: Optional[DatasetGeneEmbeddings],
    output_dir: str,
    file_prefix: str,
) -> FoundationModel:
    """Create FoundationModel instance and save to disk.

    Parameters
    ----------
    weights : FoundationModelWeights
        Model weights
    gene_annotations : pd.DataFrame
        Gene annotations DataFrame
    model_metadata : Dict
        Model metadata dictionary
    dataset_gene_embeddings : DatasetGeneEmbeddings
        Contexutalized gene embeddings for 0+ datasets
    output_dir : str
        Output directory for saving
    file_prefix : str
        Prefix for output files

    Returns
    -------
    FoundationModel
        Created FoundationModel instance

    Examples
    --------
    >>> model = _create_and_save_foundation_model(
    ...     weights, annotations, metadata, "./output", "scGPT"
    ... )
    """
    logger.info("Creating FoundationModel and saving...")
    foundation_model = FoundationModel(
        weights=weights,
        gene_annotations=gene_annotations,
        model_metadata=model_metadata,
        dataset_gene_embeddings=dataset_gene_embeddings,
    )
    foundation_model.save(output_dir, file_prefix)
    logger.info("Successfully saved all results!")
    return foundation_model


def _embed_expression_batch(
    model: Any,
    model_type: str,
    expression: torch.Tensor,
    gene_emb: torch.Tensor,
    gene_indices: Optional[torch.Tensor] = None,  # For scGPT
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Create expression-aware embeddings by processing cells in batches.

    Universal pattern for all models:
    1. Batch cells through model-specific encoder
    2. Accumulate expression embeddings across cells
    3. Average and add to static gene embeddings

    Parameters
    ----------
    model : Any
        The foundation model
    model_type : str
        One of: "scPRINT", "scGPT", "AIDOCell", "scFoundation"
    expression : torch.Tensor
        Expression matrix (n_cells, n_genes)
    gene_emb : torch.Tensor
        Static gene embeddings (n_genes, embed_dim)
    gene_indices : torch.Tensor, optional
        Gene vocabulary indices (required for scGPT)
    batch_size : int
        Number of cells to process at once
    device : torch.device, optional
        Device for computation
    verbose : bool, optional
        Whether to print progress

    Returns
    -------
    torch.Tensor
        Contextualized gene embeddings (n_genes, embed_dim)
    """
    device = ensure_device(device, allow_autoselect=True)

    _validate_embed_expression_inputs(expression, gene_emb, model_type, gene_indices)

    t_expression = expression.to(device)
    t_gene_emb = gene_emb.to(device)
    t_model = model.to(device)
    t_gene_indices = None
    if gene_indices is not None:
        t_gene_indices = gene_indices.to(device)

    n_cells = t_expression.shape[0]
    n_genes = t_gene_emb.shape[0]
    embed_dim = t_gene_emb.shape[1]

    # Accumulate expression embeddings
    expr_emb_sum = torch.zeros(n_genes, embed_dim, device=device)

    with memory_manager(device):
        for i in range(0, n_cells, batch_size):
            if verbose:
                logger.info(
                    f"Processing batch {i // batch_size + 1} of {n_cells // batch_size}"
                )

            batch_expr = t_expression[i : i + batch_size]
            batch_size_actual = batch_expr.shape[0]

            # Model-specific encoding
            if model_type == FOUNDATION_MODEL_NAMES.SCPRINT:
                batch_input = batch_expr.unsqueeze(-1)  # (batch, n_genes, 1)
                batch_expr_emb = t_model.expr_encoder(
                    batch_input
                )  # (batch, n_genes, 1, 256)
                batch_expr_emb = batch_expr_emb.squeeze(2)  # (batch, n_genes, 256)

                del batch_input
            elif model_type == FOUNDATION_MODEL_NAMES.SCGPT:
                # Prepare inputs for scGPT
                src = t_gene_indices.unsqueeze(0).expand(batch_size_actual, -1)
                src_key_padding_mask = torch.zeros(
                    batch_size_actual, n_genes, dtype=torch.bool, device=device
                )

                # Get contextualized embeddings from transformer
                batch_expr_emb = t_model._encode(
                    src, batch_expr, src_key_padding_mask, batch_labels=None
                )

                del src, src_key_padding_mask

            elif model_type == FOUNDATION_MODEL_NAMES.AIDOCELL:
                # batch_expr: (batch_size, n_genes) with raw counts
                batch_input = batch_expr.unsqueeze(-1)  # (batch_size, n_genes, 1)
                # Auto-discretization (expression component)
                batch_expr_emb = t_model.encoder.gene_embedding(batch_input)
                del batch_input

            elif model_type == FOUNDATION_MODEL_NAMES.SCFOUNDATION:
                batch_input = batch_expr.unsqueeze(-1).float()
                batch_expr_emb = t_model.token_emb(batch_input, output_weight=0)
                del batch_input
            else:
                raise ValueError(
                    f"Unknown model_type: {model_type}. Defined types: {VALID_FOUNDATION_MODEL_NAMES}"
                )

            # Accumulate
            expr_emb_sum += batch_expr_emb.sum(dim=0)

            # Cleanup
            del batch_expr_emb
            empty_cache(device)

    # Average expression embeddings across cells
    mean_expr_emb = expr_emb_sum / n_cells

    # Combine with static gene embeddings (universal pattern)
    contextual_gene_emb = t_gene_emb + mean_expr_emb

    del t_expression, t_gene_emb, t_model, t_gene_indices, mean_expr_emb, expr_emb_sum

    return contextual_gene_emb.cpu()


def _expression_tensor_to_gene_embeddings_set(
    embeddings_3d: Union[np.ndarray, torch.Tensor],
    ordered_genes: List[str],
    gene_annotations: pd.DataFrame,
    category_dict: Dict[int, str],
    gene_id_column: str = ONTOLOGIES.ENSEMBL_GENE,
    align_on: str = ONTOLOGIES.ENSEMBL_GENE,
    model_name: Optional[str] = None,
    model_variant: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_uri: Optional[str] = None,
) -> "GeneEmbeddingsSet":
    """Convert a 3D expression embedding tensor into a GeneEmbeddingsSet.

    Slices a (n_categories, n_genes, embed_dim) tensor along the category axis
    and creates one ``GeneEmbeddings`` per category, each with properly filtered
    and aligned gene annotations. The resulting list is wrapped in a
    ``GeneEmbeddingsSet`` since all categories within a dataset share the same
    gene vocabulary.

    Each category becomes a ``GeneEmbeddings`` that owns its full gene metadata,
    and the set provides dict-like access by source_label.

    Parameters
    ----------
    embeddings_3d : np.ndarray or torch.Tensor
        Expression-contextualized gene embeddings of shape
        (n_categories, n_genes, embed_dim).
    ordered_genes : List[str]
        Gene identifiers matching dimension 1 of ``embeddings_3d``.
        These are looked up in ``gene_annotations[gene_id_column]``
        to filter and reorder the annotations.
    gene_annotations : pd.DataFrame
        Full gene annotations table (may contain more genes than
        ``ordered_genes``). Must contain ``vocab_name`` and ``ensembl_gene``
        columns at minimum.
    category_dict : Dict[int, str]
        Maps category index (0-based) to category name (e.g., cell type).
        Keys must be ``{0, 1, ..., n_categories - 1}``.
    gene_id_column : str, optional
        Column in ``gene_annotations`` to match ``ordered_genes`` against
        (default: 'ensembl_gene').
    align_on : str, optional
        Column in gene_annotations used for alignment validation in the
        resulting GeneEmbeddingsSet (default: 'ensembl_gene').
    model_name : Optional[str]
        Source model name.
    model_variant : Optional[str]
        Source model variant.
    dataset_name : Optional[str]
        Source dataset name.
    dataset_uri : Optional[str]
        Source dataset URI.

    Returns
    -------
    GeneEmbeddingsSet
        Set of GeneEmbeddings, one per category, all sharing the same gene
        vocabulary. Keyed by ``source_label`` (which encodes
        model/variant/dataset/category).

    Raises
    ------
    ValueError
        If ``embeddings_3d`` is not 3-dimensional.
        If ``ordered_genes`` length doesn't match dimension 1.
        If ``category_dict`` keys don't match ``{0, ..., n_categories - 1}``.
        If gene annotations cannot be aligned to ``ordered_genes``.
    """
    # Convert torch to numpy
    if isinstance(embeddings_3d, torch.Tensor):
        embeddings_3d = embeddings_3d.numpy()

    if not isinstance(embeddings_3d, np.ndarray):
        raise ValueError("embeddings_3d must be a numpy array or torch.Tensor")

    if embeddings_3d.ndim != 3:
        raise ValueError(
            f"embeddings_3d must be 3-dimensional (n_categories, n_genes, embed_dim), "
            f"got shape {embeddings_3d.shape}"
        )

    n_categories, n_genes, _ = embeddings_3d.shape

    # Validate ordered_genes length
    if len(ordered_genes) != n_genes:
        raise ValueError(
            f"ordered_genes has {len(ordered_genes)} entries but "
            f"embeddings_3d has {n_genes} genes (dim 1)"
        )

    # Validate category_dict keys
    category_dict = {int(k): v for k, v in category_dict.items()}
    expected_keys = set(range(n_categories))
    if set(category_dict.keys()) != expected_keys:
        raise ValueError(
            f"category_dict must have keys {{0, ..., {n_categories - 1}}}, "
            f"got {sorted(category_dict.keys())}"
        )

    # Filter and reorder gene annotations to match ordered_genes
    aligned_annotations = filter_and_reorder_df(
        df=gene_annotations,
        target_ids=ordered_genes,
        id_column=gene_id_column,
    )

    # Validate that alignment succeeded for all genes
    if len(aligned_annotations) != n_genes:
        n_found = len(aligned_annotations)
        raise ValueError(
            f"Only {n_found} of {n_genes} ordered_genes found in "
            f"gene_annotations['{gene_id_column}']. "
            f"All genes must be present."
        )

    # Slice along category axis and create GeneEmbeddings per category
    result = []
    for cat_idx in range(n_categories):
        category_name = category_dict[cat_idx]

        ge = GeneEmbeddings(
            embedding=embeddings_3d[cat_idx],
            ordered_gene_ids=list(ordered_genes),
            gene_annotations=aligned_annotations.copy(),
            model_name=model_name,
            model_variant=model_variant,
            dataset_name=dataset_name,
            dataset_uri=dataset_uri,
            category=str(category_name),
        )
        result.append(ge)

    logger.info(
        f"Created {len(result)} GeneEmbeddings from expression tensor "
        f"({n_genes} genes, {n_categories} categories)"
    )

    return GeneEmbeddingsSet.from_gene_embeddings(result, align_on=align_on)


def _format_base_metadata(
    model_name: str,
    n_genes: int,
    n_vocab: int,
    vocab_list: List[str],
    embed_dim: int,
    n_layers: int,
    n_heads: int,
    model_variant: Optional[str] = None,
    **extra_metadata,
) -> Dict:
    """Format base model metadata dictionary with standard keys.

    Parameters
    ----------
    model_name : str
        Model name (e.g., "scGPT", "scFoundation")
    n_genes : int
        Number of genes
    n_vocab : int
        Vocabulary size (may include special tokens)
    vocab_list : List[str]
        Ordered vocabulary list
    embed_dim : int
        Embedding dimension
    n_layers : int
        Number of transformer layers
    n_heads : int
        Number of attention heads
    model_variant : str, optional
        Model variant identifier (e.g., "small", "medium")
    **extra_metadata : Dict, optional
        Additional metadata to include

    Returns
    -------
    Dict
        Metadata dictionary with standard FM_DEFS keys

    Examples
    --------
    >>> metadata = format_base_metadata(
    ...     "scGPT", 1000, 1003, ["gene1", "gene2", ...], 512, 12, 8
    ... )
    """
    metadata = {
        FM_DEFS.MODEL_NAME: model_name,
        FM_DEFS.N_GENES: n_genes,
        FM_DEFS.N_VOCAB: n_vocab,
        FM_DEFS.ORDERED_VOCABULARY: vocab_list,
        FM_DEFS.EMBED_DIM: embed_dim,
        FM_DEFS.N_LAYERS: n_layers,
        FM_DEFS.N_HEADS: n_heads,
    }

    if model_variant is not None:
        metadata[FM_DEFS.MODEL_VARIANT] = model_variant

    # Add any extra metadata
    metadata.update(extra_metadata)

    return metadata


def _get_cell_clusters_and_category_dict(
    obs: pd.DataFrame,
    min_cluster_cells: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """Build cell clusters and category dict from obs with cell_type and leiden_scVI.

    Parameters
    ----------
    obs : pd.DataFrame
        AnnData.obs (or subset) with columns cell_type, leiden_scVI.
    min_cluster_cells : int, optional
        Minimum cells per cluster. Clusters smaller than this are excluded.
        If None, no filtering.

    Returns
    -------
    cell_clusters : pd.DataFrame
        Filtered clusters with columns cell_type, leiden_scVI, n_cells.
    cell_cluster_dict : Dict[int, str]
        Maps index 0..n-1 to "{cell_type} ({leiden})".
    """
    # observed=True: only count observed (cell_type, leiden_scVI) pairs.
    # With categorical columns, observed=False creates the full Cartesian product
    # of all category levels, yielding n_cells=0 for unobserved pairs.
    cell_clusters = (
        obs.groupby(
            [CELLXGENE_DEFS.CELL_TYPE, CELLXGENE_DEFS.LEIDEN_SCVI], observed=True
        )
        .size()
        .reset_index(name="n_cells")
        .sort_values(CELLXGENE_DEFS.LEIDEN_SCVI)
    )

    # Each leiden cluster must map to exactly one cell_type
    leiden_to_cell_types = cell_clusters.groupby(CELLXGENE_DEFS.LEIDEN_SCVI)[
        CELLXGENE_DEFS.CELL_TYPE
    ].nunique()
    multi_cell_type_clusters = leiden_to_cell_types[leiden_to_cell_types > 1]
    if len(multi_cell_type_clusters) > 0:
        raise ValueError(
            f"Each leiden_scVI cluster must have exactly one cell_type. "
            f"Found {len(multi_cell_type_clusters)} cluster(s) with multiple cell_types: "
            f"{multi_cell_type_clusters.index.tolist()}. "
            "Check obs for inconsistent cell_type annotations within leiden clusters."
        )

    if min_cluster_cells is not None and min_cluster_cells > 0:
        n_before = len(cell_clusters)
        cell_clusters = cell_clusters[
            cell_clusters["n_cells"] >= min_cluster_cells
        ].reset_index(drop=True)
        if len(cell_clusters) == 0:
            raise ValueError(
                f"No clusters have >= {min_cluster_cells} cells. "
                "Try lowering min_cluster_cells or check obs for cell_type/leiden_scVI."
            )
        if n_before > len(cell_clusters):
            logger.info(
                f"Excluded {n_before - len(cell_clusters)} cluster(s) "
                f"with < {min_cluster_cells} cells"
            )

    # create unique category labels
    cell_cluster_dict = {
        i: f"{row[CELLXGENE_DEFS.CELL_TYPE]} ({row[CELLXGENE_DEFS.LEIDEN_SCVI]})"
        for i, row in cell_clusters.iterrows()
    }
    return cell_clusters, cell_cluster_dict


@require_modelgenerator
def _scfoundation_extract_attention_weights(
    model: "MaeAutobin",
) -> List[AttentionLayer]:
    """Extract core attention weights (Q, K, V, O) from all scFoundation encoder layers.

    Parameters
    ----------
    model : MaeAutobin
        The scFoundation model (MaeAutobin from modelgenerator)

    Returns
    -------
    List[AttentionLayer]
        List of AttentionLayer instances
    """
    attention_layers = []
    encoder = model.encoder
    transformer_encoder = encoder.transformer_encoder
    n_layers = len(transformer_encoder)

    for layer_idx in range(n_layers):
        layer = transformer_encoder[layer_idx]
        self_attn = layer.self_attn
        in_proj = self_attn.in_proj_weight.detach().cpu().numpy()
        out_proj = self_attn.out_proj.weight.detach().cpu().numpy()

        embed_dim = in_proj.shape[1]
        w_q, w_k, w_v = _split_qkv_weights(in_proj, embed_dim)

        attention_layers.append(
            AttentionLayer(
                layer_idx=layer_idx,
                W_q=w_q,
                W_k=w_k,
                W_v=w_v,
                W_o=out_proj,
            )
        )

    return attention_layers


@require_modelgenerator
def _scfoundation_extract_metadata(
    checkpoint: dict, gene_annotations: pd.DataFrame
) -> Dict:
    """Extract model metadata from scFoundation checkpoint config.

    Parameters
    ----------
    checkpoint : dict
        Loaded checkpoint for gene encoder
    gene_annotations : pd.DataFrame
        Gene annotations table

    Returns
    -------
    Dict
        Metadata dictionary for FoundationModel
    """
    logger.info("Extracting metadata...")

    config = checkpoint["config"]
    encoder_config = config["model_config"]["mae_autobin"]["encoder"]

    n_genes = len(gene_annotations)
    vocab_list = gene_annotations[FM_DEFS.VOCAB_NAME].tolist()

    return _format_base_metadata(
        model_name=SCFOUNDATION_DEFS.MODEL_NAME,
        n_genes=n_genes,
        n_vocab=n_genes,
        vocab_list=vocab_list,
        embed_dim=encoder_config["hidden_dim"],
        n_layers=encoder_config["depth"],
        n_heads=encoder_config["heads"],
    )


@require_modelgenerator
def _scfoundation_extract_weights(
    model: "MaeAutobin",
    checkpoint: dict,
    gene_annotations: pd.DataFrame,
) -> FoundationModelWeights:
    """Extract gene embeddings and attention weights from scFoundation model and checkpoint.

    Parameters
    ----------
    model : MaeAutobin
        Loaded scFoundation model (used for attention weights)
    checkpoint : dict
        Loaded checkpoint for gene embeddings and validation
    gene_annotations : pd.DataFrame
        Gene annotations table (used to determine N_GENES)

    Returns
    -------
    FoundationModelWeights
        Extracted weights in standard format
    """
    logger.info("Extracting model weights...")
    state_dict = checkpoint["state_dict"]

    # Extract architecture parameters from checkpoint config
    config = checkpoint["config"]
    encoder_config = config["model_config"]["mae_autobin"]["encoder"]
    embed_dim = encoder_config["hidden_dim"]
    n_encoder_layers = encoder_config["depth"]
    n_heads = encoder_config["heads"]

    # Validate against constants (warn if mismatch)
    if embed_dim != SCFOUNDATION_DEFS.EMBED_DIM:
        logger.warning(
            f"EMBED_DIM mismatch: checkpoint has {embed_dim}, "
            f"expected {SCFOUNDATION_DEFS.EMBED_DIM}"
        )
    if n_encoder_layers != SCFOUNDATION_DEFS.N_ENCODER_LAYERS:
        logger.warning(
            f"N_ENCODER_LAYERS mismatch: checkpoint has {n_encoder_layers}, "
            f"expected {SCFOUNDATION_DEFS.N_ENCODER_LAYERS}"
        )
    if n_heads != SCFOUNDATION_DEFS.N_HEADS:
        logger.warning(
            f"N_HEADS mismatch: checkpoint has {n_heads}, "
            f"expected {SCFOUNDATION_DEFS.N_HEADS}"
        )

    # Gene embeddings from checkpoint (pos_emb)
    gene_emb_full = state_dict["model.pos_emb.weight"].cpu().numpy()
    n_genes = len(gene_annotations)

    # Validate N_GENES
    if n_genes != SCFOUNDATION_DEFS.N_GENES:
        logger.warning(
            f"N_GENES mismatch: annotations have {n_genes}, "
            f"expected {SCFOUNDATION_DEFS.N_GENES}"
        )

    gene_emb_array = gene_emb_full[:n_genes, :]
    logger.info(f"  ✓ Extracted gene embeddings: {gene_emb_array.shape}")

    # Attention layers from model
    attention_layers = _scfoundation_extract_attention_weights(model)
    logger.info(f"  ✓ Extracted {len(attention_layers)} attention layers")

    gene_embedding = GeneEmbeddings(
        embedding=gene_emb_array,
        ordered_gene_ids=gene_annotations[FM_DEFS.VOCAB_NAME].tolist(),
        gene_annotations=gene_annotations,
    )

    return FoundationModelWeights(
        static_gene_embedding=gene_embedding, attention_layers=attention_layers
    )


@require_modelgenerator
def _scfoundation_get_expression_embeddings(
    model: MaeAutobin,
    adata: AnnData,
    gene_annotations: pd.DataFrame,
    dataset_name: Optional[str] = None,
    dataset_uri: Optional[str] = None,
    min_cluster_cells: int = FM_DEFAULTS.MIN_CLUSTER_CELLS,
) -> GeneEmbeddingsSet:
    """Embed each cell type in an AnnData object as a tensor of shape (n_cells, embed_dim).

    Parameters
    ----------
    model : MaeAutobin
        The scFoundation model
    adata : anndata.AnnData
        The AnnData object containing the cells to embed
    gene_annotations : pd.DataFrame
        The gene annotations
    vocab : GeneVocab
        The vocabulary
    dataset_name : Optional[str] = None
        The name of the dataset
    dataset_uri : Optional[str] = None
        The URI of the dataset

    Returns
    -------
    GeneEmbeddingsSet
        Expression embeddings for the dataset
    """

    cluster_embeddings, selected_genes, cell_cluster_dict = (
        _scfoundation_get_gene_embedding_by_cell_type(
            model, adata, gene_annotations, min_cluster_cells=min_cluster_cells
        )
    )

    expression_embeddings = _expression_tensor_to_gene_embeddings_set(
        embeddings_3d=cluster_embeddings,
        ordered_genes=selected_genes,
        gene_annotations=gene_annotations,
        category_dict=cell_cluster_dict,
        dataset_name=dataset_name,
        dataset_uri=dataset_uri,
    )

    return expression_embeddings


@require_modelgenerator
def _scfoundation_get_gene_embedding_by_cell_type(
    model: MaeAutobin,
    adata: AnnData,
    gene_annotations: pd.DataFrame,
    min_cluster_cells: Optional[int] = None,
) -> Tuple[torch.Tensor, List[str], Dict[int, str]]:
    """Get the gene embeddings for each cell type in an AnnData object.

    Parameters
    ----------
    model : scprint.model.model.scPrint
        The scPRINT model
    adata : anndata.AnnData
        The AnnData object containing the cells to embed
    gene_annotations : List[str]
        The common genes to use for the embeddings
    min_cluster_cells : Optional[int], optional
        Minimum number of cells per cluster to include. Clusters smaller than this
        are excluded. If None, then all clusters are included.

    Returns
    -------
    Tuple[torch.Tensor, List[str], Dict[int, str]]
        Tuple of (cluster_embeddings, selected_genes, cell_cluster_dict)
        - cluster_embeddings : torch.Tensor
            The gene embeddings for each cell type
        - selected_genes : List[str]
            The selected genes (ensembl gene ids) which are shared between the model and the adata object.
        - cell_cluster_dict : Dict[int, str]
            A dictionary mapping cell type indices to cell type names
    """

    # 1. Get indices into model vocabulary
    adata_subset, selected_genes = _scfoundation_preprocess_dataset(
        adata, gene_annotations
    )

    # 2. Get the gene embeddings and filter to selected genes
    annotations_w_index = gene_annotations.assign(
        position=range(len(gene_annotations))
    ).query(f"{ONTOLOGIES.ENSEMBL_GENE} in @selected_genes")
    annotations_name_to_index = annotations_w_index.set_index(ONTOLOGIES.ENSEMBL_GENE)[
        "position"
    ].to_dict()

    with torch.no_grad():
        gene_positions = [annotations_name_to_index[g] for g in selected_genes]
        pos_embedding = model.pos_emb.weight[(torch.tensor(gene_positions)), :]

    # 3. Track cell types to setup creating cell-type masks (use adata_subset: cells after QC)
    cell_clusters, cell_cluster_dict = _get_cell_clusters_and_category_dict(
        adata_subset.obs, min_cluster_cells=min_cluster_cells
    )

    # Allocate output
    cluster_embeddings = torch.zeros(
        len(cell_clusters),
        len(selected_genes),
        pos_embedding.shape[1],
    )

    # 4. Embed each cell type
    with torch.no_grad():
        for i, cluster in enumerate(cell_clusters["leiden_scVI"]):
            cluster_mask = adata_subset.obs["leiden_scVI"] == cluster
            cluster_adata = adata_subset[cluster_mask]

            logger.info(f"Cluster {i}: {cluster_adata.shape[0]} cells")

            if issparse(cluster_adata.X):
                cluster_expr = torch.from_numpy(cluster_adata.X.toarray())
            else:
                cluster_expr = torch.from_numpy(cluster_adata.X)

            # Use batched processing
            cluster_embeddings[i] = _embed_expression_batch(
                model=model,
                model_type=FOUNDATION_MODEL_NAMES.SCFOUNDATION,
                expression=cluster_expr,
                gene_emb=pos_embedding,  # Position embeddings serve as "static" gene identity
                batch_size=64,
            )

            logger.info(f"  ✓ Completed cluster {i}")

    return cluster_embeddings, selected_genes, cell_cluster_dict


@require_modelgenerator
def _scfoundation_preprocess_dataset(
    adata: AnnData, gene_annotations: pd.DataFrame
) -> Tuple[AnnData, List[str]]:
    """Preprocess an annData object for scFoundation model.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to preprocess.
    gene_annotations : pd.DataFrame
        The gene annotations for the scFoundation model.

    Returns
    -------
    adata_subset : AnnData
        The preprocessed AnnData object.
    selected_genes : List[str]
        The list of genes (ensembl genes) which are shared between the model and the adata object.
    """
    import scanpy as sc

    adata_subset = adata.copy()

    # scFoundation QC filtering
    sc.pp.filter_cells(adata_subset, min_genes=200)
    # No gene filtering (min_cells=0)

    # Normalize: 1e4 total counts + log1p
    sc.pp.normalize_total(adata_subset, target_sum=1e4)
    sc.pp.log1p(adata_subset)

    # Retain genes shared between model and adata
    model_genes = gene_annotations[ONTOLOGIES.ENSEMBL_GENE]  # 19,264 genes
    selected_genes = [g for g in model_genes if g in adata_subset.var_names]

    adata_subset = adata_subset[:, selected_genes]

    return adata_subset, selected_genes


@require_scgpt
def _scgpt_extract_attention_weights(
    state_dict: dict, n_layers: int, embed_dim: int
) -> List[AttentionLayer]:
    """Extract core attention weights (Q, K, V, O) from scGPT checkpoint state_dict.

    Note: scGPT model.state_dict() returns incorrect/shared weights, so we extract
    from the checkpoint file's state_dict directly.

    Parameters
    ----------
    state_dict : dict
        Model state dict loaded from checkpoint file
    n_layers : int
        Number of transformer layers
    embed_dim : int
        Embedding dimension (d_model)

    Returns
    -------
    List[AttentionLayer]
        List of AttentionLayer instances
    """
    attention_layers = []
    for layer_idx in range(n_layers):
        wqkv_key = f"transformer_encoder.layers.{layer_idx}.self_attn.Wqkv.weight"
        out_proj_key = (
            f"transformer_encoder.layers.{layer_idx}.self_attn.out_proj.weight"
        )

        if wqkv_key not in state_dict:
            raise KeyError(f"Could not find {wqkv_key} in state_dict")
        if out_proj_key not in state_dict:
            raise KeyError(f"Could not find {out_proj_key} in state_dict")

        in_proj = state_dict[wqkv_key].clone()
        out_proj = state_dict[out_proj_key].clone()

        if in_proj.shape[0] != 3 * embed_dim:
            raise ValueError(
                f"Expected in_proj.shape[0] to be 3*d ({3*embed_dim}), "
                f"but got {in_proj.shape[0]}"
            )

        in_proj_np = in_proj.cpu().detach().numpy()
        w_q, w_k, w_v = _split_qkv_weights(in_proj_np, embed_dim)
        w_o = out_proj.cpu().detach().numpy()

        attention_layers.append(
            AttentionLayer(
                layer_idx=layer_idx,
                W_q=w_q,
                W_k=w_k,
                W_v=w_v,
                W_o=w_o,
            )
        )

    return attention_layers


@require_scgpt
@require_torchtext
def _scgpt_extract_weights(
    model: "TransformerModel",
    vocab: GeneVocab,
    model_metadata: dict,
    checkpoint_path: str,
    gene_annotations: pd.DataFrame,
    device: Optional[Union[str, torch.device]] = None,
) -> FoundationModelWeights:
    """Extract weights from scGPT model.

    Note: Weights must be loaded directly from the checkpoint file because
    model.state_dict() returns incorrect/shared weights across layers.

    Parameters
    ----------
    model : TransformerModel
        scGPT TransformerModel
    vocab : GeneVocab
        scGPT vocabulary object
    model_metadata : dict
        Model metadata dictionary
    checkpoint_path : str
        Path to checkpoint file
    gene_annotations : pd.DataFrame
        Gene annotations DataFrame. Used to identify gene-only vocabulary
        entries and strip special tokens.
    device : Optional[Union[str, torch.device]], optional
        Device to use for computation

    Returns
    -------
    FoundationModelWeights
        Extracted model weights
    """
    device = ensure_device(device, allow_autoselect=True)
    model = model.to(device)

    # Get full vocabulary embeddings (includes special tokens)
    gene_ids = torch.arange(len(vocab), device=device)
    full_embeddings = model.encoder(gene_ids).detach().cpu().numpy()

    # Strip special tokens: keep only genes present in gene_annotations
    gene_vocab_names = gene_annotations[FM_DEFS.VOCAB_NAME].tolist()
    gene_indices = [vocab[name] for name in gene_vocab_names]
    gene_embedding_array = full_embeddings[gene_indices, :]

    # Load weights directly from checkpoint file (model.state_dict() is unreliable)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE.CPU)

    # Extract attention weights via dedicated function (like AIDOCell, scPRINT)
    n_layers = model_metadata[FM_DEFS.N_LAYERS]
    d = model_metadata[FM_DEFS.EMBED_DIM]
    attention_layers = _scgpt_extract_attention_weights(
        state_dict=state_dict, n_layers=n_layers, embed_dim=d
    )

    gene_embedding = GeneEmbeddings(
        embedding=gene_embedding_array,
        ordered_gene_ids=gene_annotations[FM_DEFS.VOCAB_NAME].tolist(),
        gene_annotations=gene_annotations,
    )

    return FoundationModelWeights(
        static_gene_embedding=gene_embedding, attention_layers=attention_layers
    )


@require_torchtext
def _scgpt_format_metadata(model_configs: dict, vocab: Any) -> Dict:
    """Format scGPT model metadata.

    Parameters
    ----------
    model_configs : dict
        Model configuration dictionary
    vocab : Any
        scGPT vocabulary object

    Returns
    -------
    Dict
        Metadata dictionary with standard FM_DEFS keys
    """
    # Get vocabulary as list of tokens in order
    vocab_list = vocab.get_itos()

    # Count actual genes (excluding special tokens)
    n_genes = len(
        [
            token
            for token in vocab_list
            if not token.startswith("<") and token != SCGPT_DEFS.PAD_TOKEN
        ]
    )

    return _format_base_metadata(
        model_name=SCGPT_DEFS.MODEL_NAME,
        n_genes=n_genes,
        n_vocab=len(vocab),
        vocab_list=vocab_list,
        embed_dim=model_configs[SCGPT_DEFS.D_HID],
        n_layers=model_configs[SCGPT_DEFS.NLAYERS],
        n_heads=model_configs[SCGPT_DEFS.NHEAD],
    )


@require_scgpt
def _scgpt_get_expression_embeddings(
    model: TransformerModel,
    adata: AnnData,
    gene_annotations: pd.DataFrame,
    vocab: GeneVocab,
    dataset_name: Optional[str] = None,
    dataset_uri: Optional[str] = None,
    min_cluster_cells: Optional[int] = None,
) -> GeneEmbeddingsSet:
    """Embed each cell type in an AnnData object as a tensor of shape (n_cells, embed_dim).

    Parameters
    ----------
    model : scprint.model.model.scPrint
        The scPRINT model
    adata : anndata.AnnData
        The AnnData object containing the cells to embed
    gene_annotations : pd.DataFrame
        The gene annotations
    vocab : GeneVocab
        The vocabulary
    dataset_name : Optional[str] = None
        The name of the dataset
    dataset_uri : Optional[str] = None
        The URI of the dataset

    Returns
    -------
    GeneEmbeddingsSet
        Expression embeddings for the dataset
    """

    cluster_embeddings, selected_genes, cell_cluster_dict = (
        _scgpt_get_gene_embedding_by_cell_type(
            model, adata, gene_annotations, vocab, min_cluster_cells=min_cluster_cells
        )
    )

    expression_embeddings = _expression_tensor_to_gene_embeddings_set(
        embeddings_3d=cluster_embeddings,
        ordered_genes=selected_genes,
        gene_annotations=gene_annotations,
        category_dict=cell_cluster_dict,
        dataset_name=dataset_name,
        dataset_uri=dataset_uri,
    )

    return expression_embeddings


@require_scgpt
def _scgpt_get_gene_embedding_by_cell_type(
    model: TransformerModel,
    adata: AnnData,
    gene_annotations: pd.DataFrame,
    vocab: GeneVocab,
    device: Optional[Union[torch.device, str]] = None,
    min_cluster_cells: Optional[int] = None,
) -> Tuple[torch.Tensor, List[str], Dict[str, str]]:
    """Get the gene embeddings for each cell type in an AnnData object.

    Parameters
    ----------
    model : scprint.model.model.scPrint
        The scPRINT model
    adata : anndata.AnnData
        The AnnData object containing the cells to embed
    gene_annotations : List[str]
        The common genes to use for the embeddings
    vocab : GeneVocab
        The vocabulary
    device : Optional[Union[torch.device, str]]
        The device to use for the embeddings
    min_cluster_cells: Optional[int] = None,

    Returns
    -------
    Tuple[torch.Tensor, Dict[str, str]]
        Tuple of (cluster_embeddings, cell_cluster_dict)
        - cluster_embeddings : torch.Tensor
            The gene embeddings for each cell type
        - selected_genes : List[str]
            The selected genes (ensembl gene ids)
        - cell_cluster_dict : Dict[str, str]
            A dictionary mapping cell type indices to cell type names
    """

    device = ensure_device(device, allow_autoselect=True)

    # 1. Get indices into model vocabulary
    # Get common genes between model and data
    common_genes_df = gene_annotations.query(
        f"{ONTOLOGIES.ENSEMBL_GENE} in @adata.var_names"
    ).assign(index=lambda x: x["vocab_name"].apply(lambda v: vocab[v]))
    gene_indices = common_genes_df["index"].values

    # 2. Track cell types to setup creating cell-type masks
    # Use adata.obs here; adata_subset created below has same cells, subset of genes
    cell_clusters, cell_cluster_dict = _get_cell_clusters_and_category_dict(
        adata.obs, min_cluster_cells=min_cluster_cells
    )

    # filter to highly variable genes and bin expression values (as per the training data)
    adata_subset, selected_genes, gene_indices = _scgpt_preprocess_dataset(
        adata, model, vocab, gene_annotations
    )

    # static embeddings
    d_gene_indices = gene_indices.to(device)
    gene_embeddings = model.encoder(d_gene_indices)
    del d_gene_indices

    # Allocate output
    cluster_embeddings = torch.zeros(
        len(cell_clusters),
        len(selected_genes),
        model.d_model,
    )

    # 3. Embed each cell type
    model.eval()
    with torch.no_grad():
        for i, cluster in enumerate(cell_clusters["leiden_scVI"]):
            cluster_mask = adata_subset.obs["leiden_scVI"] == cluster
            cluster_adata = adata_subset[cluster_mask]

            logger.info(f"Cluster {i}: {cluster_adata.shape[0]} cells")

            if model.input_emb_style == "continuous":
                cluster_expr = torch.tensor(cluster_adata.X, dtype=torch.float32)
            else:
                cluster_expr = torch.tensor(cluster_adata.X)

            # Use batched processing
            cluster_embeddings[i] = _embed_expression_batch(
                model=model,
                model_type=FOUNDATION_MODEL_NAMES.SCGPT,
                expression=cluster_expr,
                gene_emb=gene_embeddings,
                gene_indices=gene_indices,
                batch_size=64,
                device=device,
            )

            logger.info(f"  ✓ Completed cluster {i}")

    return cluster_embeddings, selected_genes, cell_cluster_dict


@require_scgpt
def _scgpt_load_gene_annotations(annotations_path: str) -> pd.DataFrame:
    """Load gene annotations for scGPT.

    Parameters
    ----------
    annotations_path : str
        Path to gene annotations CSV file

    Returns
    -------
    pd.DataFrame
        DataFrame with gene annotations
    """
    return (
        pd.read_csv(annotations_path, index_col=0)
        .rename(
            columns={
                "feature_id": ONTOLOGIES.ENSEMBL_GENE,
                "feature_name": ONTOLOGIES.SYMBOL,
            }
        )
        .assign(**{FM_DEFS.VOCAB_NAME: lambda x: x[ONTOLOGIES.SYMBOL]})
        .drop(columns=["feature_length", "soma_joinid"])
    )


@require_scgpt
@require_torchtext
def _scgpt_load_model(model_dir: str) -> Tuple[Any, Any, dict, str]:
    """Load scGPT model from directory.

    Parameters
    ----------
    model_dir : str
        Directory containing scGPT model files

    Returns
    -------
    Tuple[Any, Any, dict, str]
        Tuple of (model, vocab, model_metadata, checkpoint_path)
    """
    from scgpt.tokenizer.gene_tokenizer import GeneVocab

    model_config_file = os.path.join(model_dir, SCGPT_DEFS.CONFIG_FILENAME)
    model_file = os.path.join(model_dir, SCGPT_DEFS.MODEL_FILENAME)
    vocab_file = os.path.join(model_dir, SCGPT_DEFS.VOCAB_FILENAME)

    vocab = GeneVocab.from_file(vocab_file)
    for s in SCGPT_DEFS.SPECIAL_TOKENS:
        if s not in vocab:
            vocab.append_token(s)

    # Retrieve model parameters from config files
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    print(
        f"Resume model from {model_file}, the model args will override the "
        f"config {model_config_file}."
    )

    model = _scgpt_load_model_from_file(model_file, vocab, model_configs)

    model_metadata = _scgpt_format_metadata(model_configs, vocab)

    return model, vocab, model_metadata, model_file


@require_scgpt
@require_torchtext
def _scgpt_load_model_from_file(
    model_file: str, vocab: Any, model_configs: dict
) -> Any:
    """Load scGPT model from checkpoint file.

    Parameters
    ----------
    model_file : str
        Path to model checkpoint file
    vocab : Any
        scGPT vocabulary object
    model_configs : dict
        Model configuration dictionary

    Returns
    -------
    Any
        Loaded scGPT TransformerModel
    """
    from scgpt.model import TransformerModel

    device = select_device()

    ntokens = len(vocab)  # size of vocabulary
    model = TransformerModel(
        ntokens,
        model_configs[SCGPT_DEFS.EMBSIZE],
        model_configs[SCGPT_DEFS.NHEAD],
        model_configs[SCGPT_DEFS.D_HID],
        model_configs[SCGPT_DEFS.NLAYERS],
        vocab=vocab,
        pad_value=SCGPT_DEFS.PAD_VALUE,
        n_input_bins=SCGPT_DEFS.N_INPUT_BINS,
    )

    try:
        model.load_state_dict(
            torch.load(model_file, map_location=torch.device(DEVICE.CPU))
        )
        print(f"Loading all model params from {model_file}")
    except Exception as e:
        logger.warning(
            f"Error loading model: {e}; recovering by extracting specific parameters."
        )
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file, map_location=torch.device(DEVICE.CPU))
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model.to(device)

    return model


def _scgpt_preprocess_dataset(
    adata: AnnData,
    model: Any,
    vocab: Any,
    gene_annotations: pd.DataFrame,
    input_style: str = "binned",
    max_seq_len: int = 1200,
    n_bins: int = 51,
) -> Tuple[AnnData, List[str], torch.Tensor, torch.Tensor]:
    """Preprocess entire dataset for scGPT model according to model's training config.

    Performs:
    1. Filter to common genes between adata and model vocab
    2. HVG selection (to max_seq_len)
    3. Normalization and binning

    Parameters
    ----------
    adata : AnnData
        The full dataset to preprocess
    model : Any
        The scGPT model
    vocab : Any
        The scGPT vocabulary
    gene_annotations : pd.DataFrame
        Gene annotation mappings
    input_style : str
        The input style to use (from the model's args.json file).
    max_seq_len : int
        Maximum sequence length (from model's args.json)
    n_bins : int
        Number of bins (from model's args.json)

    Returns
    -------
    adata_processed : AnnData
        Preprocessed AnnData with HVG-selected genes as the X attribute
    selected_genes : List[str]
        List of selected gene names (Ensembl IDs)
    gene_indices : torch.Tensor
        Vocabulary indices for selected genes, shape (n_genes,)
    """
    from scgpt.preprocess import Preprocessor

    # 1. Get common genes between adata and vocab
    common_genes_df = gene_annotations.query(
        f"{ONTOLOGIES.ENSEMBL_GENE} in @adata.var_names"
    ).assign(index=lambda x: x["vocab_name"].apply(lambda v: vocab[v]))
    common_genes = common_genes_df[ONTOLOGIES.ENSEMBL_GENE].values.tolist()

    logger.info(f"Found {len(common_genes)} common genes between data and model vocab")

    # 2. Subset to common genes
    adata_subset = adata[:, common_genes].copy()

    # 3. Determine preprocessing based on model config
    if not hasattr(model, "input_emb_style"):
        raise ValueError("Model does not have input_emb_style attribute")

    input_style = model.input_emb_style

    logger.info(f"Model input_style: {input_style}")

    # 4. Run preprocessing pipeline
    if input_style == "binned":
        # Model expects binned integer values
        n_bins = (
            model.value_encoder.num_embeddings
            if hasattr(model.value_encoder, "num_embeddings")
            else n_bins
        )

        preprocessor = Preprocessor(
            use_key="X",
            filter_cell_by_counts=1,  # filter_cell_by_counts=1: remove all-zero rows before binning (avoids scGPT warning)
            normalize_total=1e4,
            log1p=False,  # log1p=False: CellxGene and most h5ad files have .X already log1p-transformed
            subset_hvg=max_seq_len,  # Select top N HVGs
            hvg_flavor="seurat_v3",
            binning=n_bins,
            result_binned_key="X_binned",
        )
        preprocessor(adata_subset)

        # Store binned values in .X for easy access
        if issparse(adata_subset.layers["X_binned"]):
            adata_subset.X = adata_subset.layers["X_binned"].toarray()
        else:
            adata_subset.X = adata_subset.layers["X_binned"]

    elif input_style == "continuous":
        # Model expects continuous normalized values
        preprocessor = Preprocessor(
            use_key="X",
            filter_cell_by_counts=1,  # filter_cell_by_counts=1: remove all-zero rows before binning (avoids scGPT warning)
            normalize_total=1e4,
            log1p=False,  # log1p=False: CellxGene and most h5ad files have .X already log1p-transformed
            subset_hvg=max_seq_len,  # Select top N HVGs
            hvg_flavor="seurat_v3",
            binning=n_bins,  # Still bin! Model treats bins as continuous
            result_binned_key="X_binned",
        )
        preprocessor(adata_subset)

        # Store binned values as float (ContinuousValueEncoder treats them as continuous)
        if issparse(adata_subset.layers["X_binned"]):
            adata_subset.X = adata_subset.layers["X_binned"].toarray().astype(float)
        else:
            adata_subset.X = adata_subset.layers["X_binned"].astype(float)
    else:
        raise ValueError(f"Unknown input_style: {input_style}")

    # 5. Get selected genes after HVG filtering
    selected_genes = adata_subset.var_names.tolist()
    logger.info(f"Selected {len(selected_genes)} HVGs for processing")

    # 6. Get gene indices and embeddings for selected genes
    selected_genes_df = gene_annotations[
        gene_annotations[ONTOLOGIES.ENSEMBL_GENE].isin(selected_genes)
    ].copy()
    selected_genes_df["vocab_index"] = selected_genes_df["vocab_name"].apply(
        lambda x: vocab[x]
    )

    # Ensure order matches adata_subset.var_names
    selected_genes_df = (
        selected_genes_df.set_index(ONTOLOGIES.ENSEMBL_GENE)
        .loc[selected_genes]
        .reset_index()
    )

    gene_indices = torch.tensor(
        selected_genes_df["vocab_index"].values, dtype=torch.long
    )

    return adata_subset, selected_genes, gene_indices


@require_scprint
def _scprint_extract_attention_weights(model: scPrint) -> List[AttentionLayer]:
    """Extract attention weights (Q, K, V, O) from all layers.

    Parameters
    ----------
    model : scPrint
        The scPRINT model

    Returns
    -------
    List[AttentionLayer]
        List of AttentionLayer instances
    """
    attention_layers = []
    d_model = model.d_model
    n_layers = model.nlayers

    # Validate n_heads (scPRINT has fixed n_heads=4)
    # Note: scPRINT doesn't expose n_heads directly, so we validate against constant
    # The actual n_heads is embedded in the QKV weight shape, but we use the constant

    for layer_idx in range(n_layers):
        block = model.transformer.blocks[layer_idx]
        mixer = block.mixer

        # Get combined QKV weight: (3 * d_model, d_model)
        qkv_weight = mixer.Wqkv.weight.detach().cpu().numpy()
        w_q, w_k, w_v = _split_qkv_weights(qkv_weight, d_model)

        attention_layers.append(
            AttentionLayer(
                layer_idx=layer_idx,
                W_q=w_q,
                W_k=w_k,
                W_v=w_v,
                W_o=mixer.out_proj.weight.detach().cpu().numpy(),
            )
        )

    return attention_layers


@require_scprint
def _scprint_extract_weights(
    model: scPrint, gene_annotations: pd.DataFrame
) -> FoundationModelWeights:
    """Extract model weights in the standardized format.

    Parameters
    ----------
    model : scPrint
        The scPRINT model
    gene_annotations : pd.DataFrame
        Gene annotations DataFrame

    Returns
    -------
    FoundationModelWeights
        FoundationModelWeights instance containing static_gene_embedding and attention_layers
    """
    gene_embedding_array = model.gene_encoder.embeddings.weight.detach().cpu().numpy()
    gene_embedding = GeneEmbeddings(
        embedding=gene_embedding_array,
        ordered_gene_ids=gene_annotations[FM_DEFS.VOCAB_NAME].tolist(),
        gene_annotations=gene_annotations,
    )

    attention_layers = _scprint_extract_attention_weights(model)

    return FoundationModelWeights(
        static_gene_embedding=gene_embedding, attention_layers=attention_layers
    )


@require_scprint
def _scprint_format_metadata(model: scPrint, version: Optional[str] = None) -> Dict:
    """Extract model architecture metadata.

    Parameters
    ----------
    model : scPrint
        The scPRINT model
    version : str, optional
        Version string (e.g., "small-v1", "medium-v1.5", "large-v1")

    Returns
    -------
    Dict
        Dictionary with model metadata
    """
    # Extract architecture parameters from model
    d_model = int(model.d_model)
    n_layers = int(model.nlayers)
    # Note: scPRINT models have n_heads as a fixed architecture parameter
    # We'll validate against the constant
    n_heads = SCPRINT_DEFS.N_HEADS  # Fixed at 4 for all scPRINT models

    # Validate n_heads (though it's fixed, good to check if model structure changes)
    # Note: scPRINT doesn't expose n_heads directly, so we validate against constant

    # Get vocabulary as list of genes (scPRINT doesn't have special tokens)
    vocab_list = list(model.genes)
    n_genes = len(vocab_list)

    return _format_base_metadata(
        model_name=SCPRINT_DEFS.MODEL_NAME,
        n_genes=n_genes,
        n_vocab=n_genes,  # Same as n_genes for scPRINT (no special tokens)
        vocab_list=vocab_list,
        embed_dim=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        model_variant=version,
    )


@require_scprint
def _scprint_get_checkpoint(version_key: str, model_path: str) -> str:
    """Get the checkpoint file for a given version key.

    Parameters
    ----------
    version_key : str
        Version key
    model_path : str
        Model path

    Returns
    -------
    str
        Path to checkpoint file
    """
    from huggingface_hub import hf_hub_download

    # Get version ID and checkpoint filename from the version key
    version_id = getattr(SCPRINT_DEFS.VERSIONS, version_key)
    checkpoint_filename = getattr(SCPRINT_DEFS.CHECKPOINTS, version_key)

    logger.info(
        f"\n1. Loading {version_id} ({version_key}) model from {checkpoint_filename} and downloading if needed..."
    )
    return hf_hub_download(
        repo_id=SCPRINT_DEFS.REPO_ID, filename=checkpoint_filename, cache_dir=model_path
    )


@require_scprint
def _scprint_get_expression_embeddings(
    model: scPrint,
    adata: AnnData,
    gene_annotations: pd.DataFrame,
    dataset_name: Optional[str] = None,
    dataset_uri: Optional[str] = None,
) -> Tuple[torch.Tensor, Dict[str, str], List[str]]:
    """Embed each cell type in an AnnData object as a tensor of shape (n_cells, embed_dim).

    Parameters
    ----------
    model : scprint.model.model.scPrint
        The scPRINT model
    adata : anndata.AnnData
        The AnnData object containing the cells to embed
    dataset_name : Optional[str] = None
        The name of the dataset
    dataset_uri : Optional[str] = None
        The URI of the dataset

    Returns
    -------
    Tuple[torch.Tensor, Dict[str, str], List[str]]
        Tuple of (cluster_embeddings, cell_cluster_dict, common_genes)
    """

    model_genes = model.genes  # The 44,756 genes
    common_genes = [g for g in model_genes if g in adata.var_names]

    cluster_embeddings, cell_cluster_dict = _scprint_get_gene_embedding_by_cell_type(
        model, adata, common_genes
    )

    expression_embeddings = _expression_tensor_to_gene_embeddings_set(
        embeddings_3d=cluster_embeddings,
        ordered_genes=common_genes,
        gene_annotations=gene_annotations,
        category_dict=cell_cluster_dict,
        dataset_name=dataset_name,
        dataset_uri=dataset_uri,
    )

    return expression_embeddings


@require_scprint
def _scprint_get_gene_embedding_by_cell_type(
    model: scPrint,
    adata: AnnData,
    common_genes: List[str],
    min_cluster_cells: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, str]]:
    """Get the gene embeddings for each cell type in an AnnData object.

    Parameters
    ----------
    model : scprint.model.model.scPrint
        The scPRINT model
    adata : anndata.AnnData
        The AnnData object containing the cells to embed
    min_cluster_cells: Optional[int] = None,
        Minimum number of cells per cluster to include. Clusters smaller than this
        are excluded. If None, then all clusters are included.

    Returns
    -------
    Tuple[torch.Tensor, Dict[str, str]]
        Tuple of (cluster_embeddings, cell_cluster_dict)
        - cluster_embeddings : torch.Tensor
            The gene embeddings for each cell type
        - cell_cluster_dict : Dict[str, str]
            A dictionary mapping cell type indices to cell type names
    """

    # 1. Get indices into model vocabulary
    gene_indices = [model.genes.index(g) for g in common_genes]
    gene_emb_full = model.gene_encoder.embeddings.weight.data  # (44756, 256)
    gene_emb = gene_emb_full[gene_indices]  # (n_common, 256)

    # 2. Track cell types to setup creating cell-type masks
    cell_clusters, cell_cluster_dict = _get_cell_clusters_and_category_dict(
        adata.obs, min_cluster_cells=min_cluster_cells
    )

    # Allocate output
    cluster_embeddings = torch.zeros(
        len(cell_clusters),
        len(common_genes),
        model.d_model,
    )

    # 3. Embed each cell type
    with torch.no_grad():
        for i, cluster in enumerate(cell_clusters["leiden_scVI"]):
            cluster_mask = adata.obs["leiden_scVI"] == cluster
            cluster_adata = adata[cluster_mask]

            logger.info(f"Cluster {i}: {cluster_adata.shape[0]} cells")

            cluster_expr = _scprint_normalize_expression(cluster_adata, common_genes)

            # Use batched processing
            cluster_embeddings[i] = _embed_expression_batch(
                model,
                FOUNDATION_MODEL_NAMES.SCPRINT,
                cluster_expr,
                gene_emb,
                batch_size=64,
            )

            logger.info(f"  ✓ Completed cluster {i}")

    return cluster_embeddings, cell_cluster_dict


@require_scprint
def _scprint_load_gene_annotations(model: scPrint) -> pd.DataFrame:
    """Load gene annotations from scPRINT model.

    Parameters
    ----------
    model : Any
        The scPRINT model

    Returns
    -------
    pd.DataFrame
        DataFrame with gene annotations
    """
    gene_table = pd.DataFrame(
        {
            FM_DEFS.VOCAB_NAME: model.genes,
            ONTOLOGIES.ENSEMBL_GENE: model.genes,
        }
    )

    # Optionally add gene symbols from lamindb
    try:
        import bionty as bt

        all_genes_df = bt.Gene.filter().df()
        ensembl_to_symbol = all_genes_df.set_index("ensembl_gene_id")[
            "symbol"
        ].to_dict()
        gene_table[ONTOLOGIES.SYMBOL] = gene_table[ONTOLOGIES.ENSEMBL_GENE].map(
            ensembl_to_symbol
        )
    except Exception as e:
        logger.warning(f"Error loading gene symbols from lamin database: {e}")
        gene_table[ONTOLOGIES.SYMBOL] = gene_table[ONTOLOGIES.ENSEMBL_GENE]

    return gene_table


@require_scprint
def _scprint_load_model(
    checkpoint_path: str, transformer: str = "normal", version: Optional[str] = None
) -> Tuple[Any, pd.DataFrame, Dict]:
    """Load the scPRINT model and return model, gene annotations, and metadata.

    Parameters
    ----------
    checkpoint_path : str
        Path to the scPRINT checkpoint file
    transformer : str, optional
        Transformer type, by default "normal"
    version : str, optional
        Version string (e.g., "small-v1", "medium-v1.5", "large-v1")

    Returns
    -------
    Tuple[Any, pd.DataFrame, Dict]
        Tuple of (model, gene_annotations, model_metadata)
    """
    logger.info("Loading scPRINT model")
    model = _scprint_load_model_from_file(checkpoint_path, transformer)

    logger.info("Loading gene annotations")
    gene_annotations = _scprint_load_gene_annotations(model)

    logger.info("Formatting model metadata")
    model_metadata = _scprint_format_metadata(model, version=version)

    return model, gene_annotations, model_metadata


@require_scprint
def _scprint_load_model_from_file(
    checkpoint_path: str, transformer: str = "normal"
) -> Any:
    """Load scPRINT model from checkpoint file.

    Parameters
    ----------
    checkpoint_path : str
        Path to the scPRINT checkpoint file
    transformer : str, optional
        Transformer type, by default "normal"

    Returns
    -------
    Any
        The scPRINT model
    """
    from scprint import scPrint

    m = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    if "prenorm" in m["hyper_parameters"]:
        m["hyper_parameters"].pop("prenorm")

    if "label_counts" in m["hyper_parameters"]:
        model = scPrint.load_from_checkpoint(
            checkpoint_path,
            precpt_gene_emb=None,
            classes=m["hyper_parameters"]["label_counts"],
            transformer=transformer,
        )
    else:
        model = scPrint.load_from_checkpoint(
            checkpoint_path, precpt_gene_emb=None, transformer=transformer
        )

    model.eval()

    return model


@require_scprint
def _scprint_normalize_expression(
    adata: AnnData, common_genes: List[str]
) -> torch.Tensor:
    """Normalize the expression matrix of an AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the cells to normalize
    common_genes : List[str]
        The common genes to subset the AnnData object to

    Returns
    -------
    torch.Tensor
        The normalized expression matrix
    """
    # 1. Subset adata to just the "common_genes" across the model and adata
    adata_subset = adata[:, common_genes]

    # 2. Get expression matrix (raw counts)
    if issparse(adata_subset.X):
        expression = torch.from_numpy(adata_subset.X.toarray())
    else:
        expression = torch.from_numpy(adata_subset.X)

    # 3. Normalize (sum method)
    return expression / expression.sum(1, keepdim=True)  # Per-cell normalization


def _split_qkv_weights(
    qkv_weight: np.ndarray, embed_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split combined QKV weight matrix into separate Q, K, V matrices.

    Parameters
    ----------
    qkv_weight : np.ndarray
        Combined QKV weight matrix of shape (3 * embed_dim, embed_dim)
    embed_dim : int
        Embedding dimension

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of (W_q, W_k, W_v) matrices, each of shape (embed_dim, embed_dim)

    Examples
    --------
    >>> qkv = np.random.randn(768, 256)  # 3*256 = 768
    >>> w_q, w_k, w_v = _split_qkv_weights(qkv, embed_dim=256)
    >>> assert w_q.shape == (256, 256)
    """
    if qkv_weight.shape[0] != 3 * embed_dim:
        raise ValueError(
            f"Expected qkv_weight.shape[0] to be 3*embed_dim ({3*embed_dim}), "
            f"but got {qkv_weight.shape[0]}"
        )

    w_q = qkv_weight[:embed_dim, :]
    w_k = qkv_weight[embed_dim : 2 * embed_dim, :]
    w_v = qkv_weight[2 * embed_dim :, :]

    return w_q, w_k, w_v


def _validate_embed_expression_inputs(
    expression: torch.Tensor,
    gene_emb: torch.Tensor,
    model_type: str,
    gene_indices: Optional[torch.Tensor] = None,
) -> None:
    """Validate inputs for _embed_expression_batch.

    Checks:
    - Tensor dimensions are correct
    - Gene counts match across inputs
    - Model-specific requirements are met

    Parameters
    ----------
    expression : torch.Tensor
        Expression matrix (n_cells, n_genes)
    gene_emb : torch.Tensor
        Static gene embeddings (n_genes, embed_dim)
    model_type : str
        One of: "scPRINT", "scGPT", "AIDOCell", "scFoundation"
    gene_indices : torch.Tensor, optional
        Gene vocabulary indices (required for scGPT), shape (n_genes,)

    Raises
    ------
    ValueError
        If any dimension checks fail
    """
    # Check expression tensor
    if expression.ndim != 2:
        raise ValueError(
            f"expression must be 2D (n_cells, n_genes), got {expression.ndim}D "
            f"with shape {expression.shape}"
        )

    n_cells, n_genes_expr = expression.shape

    # Check gene_emb tensor
    if gene_emb.ndim != 2:
        raise ValueError(
            f"gene_emb must be 2D (n_genes, embed_dim), got {gene_emb.ndim}D "
            f"with shape {gene_emb.shape}"
        )

    n_genes_emb, embed_dim = gene_emb.shape

    # Check gene dimension compatibility
    if n_genes_expr != n_genes_emb:
        raise ValueError(
            f"Gene count mismatch:\n"
            f"  expression: {n_genes_expr} genes (shape {expression.shape})\n"
            f"  gene_emb:   {n_genes_emb} genes (shape {gene_emb.shape})\n"
            f"These must match!"
        )

    n_genes = n_genes_emb

    # Model-specific validation
    if model_type == FOUNDATION_MODEL_NAMES.SCGPT:
        if gene_indices is None:
            raise ValueError(
                "scGPT requires gene_indices parameter. "
                "Pass the vocabulary indices for the genes."
            )

        if gene_indices.ndim != 1:
            raise ValueError(
                f"gene_indices must be 1D (n_genes,), got {gene_indices.ndim}D "
                f"with shape {gene_indices.shape}"
            )

        n_genes_indices = len(gene_indices)
        if n_genes_indices != n_genes:
            raise ValueError(
                f"Gene count mismatch:\n"
                f"  expression:    {n_genes_expr} genes\n"
                f"  gene_emb:      {n_genes_emb} genes\n"
                f"  gene_indices:  {n_genes_indices} genes\n"
                f"All must match!"
            )

    elif model_type in [
        FOUNDATION_MODEL_NAMES.SCPRINT,
        FOUNDATION_MODEL_NAMES.AIDOCELL,
        FOUNDATION_MODEL_NAMES.SCFOUNDATION,
    ]:
        # These models don't need gene_indices
        pass

    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Must be one of: {VALID_FOUNDATION_MODEL_NAMES}"
        )

    # Success message with summary
    logger.debug(
        f"Input validation passed for {model_type}:\n"
        f"  {n_cells} cells x {n_genes} genes\n"
        f"  Embedding dimension: {embed_dim}"
    )
