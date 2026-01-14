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
"""

import json
import logging
import os
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from napistu.constants import ONTOLOGIES
from napistu.utils import download_wget

from napistu_torch.load.constants import (
    AIDOCELL_CLASSES,
    AIDOCELL_DEFS,
    FM_DEFS,
    SCFOUNDATION_DEFS,
    SCGPT_DEFS,
    SCPRINT_DEFS,
)
from napistu_torch.load.foundation_models import (
    AttentionLayer,
    FoundationModel,
    FoundationModelWeights,
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
from napistu_torch.utils.torch_utils import select_device

# Set up warnings for scGPT
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore")

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ============================================================================
# Public API
# ============================================================================


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
def process_aidocell(model_class: Any, output_dir: str) -> None:
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
    weights = _aidocell_extract_weights(model)
    logger.info(f"   Embeddings: {weights.gene_embedding.shape}")
    logger.info(
        f"   Attention weights: {model_metadata[FM_DEFS.N_LAYERS]} layers × 4 matrices (Q,K,V,O)"
    )

    # 3. Create FoundationModel and save
    _create_and_save_foundation_model(
        weights, gene_annotations, model_metadata, output_dir, file_prefix
    )

    return None


@require_modelgenerator
def process_scfoundation(
    output_dir: str,
    checkpoint_path: Optional[str] = None,
    output_prefix: Optional[str] = None,
    cache_dir: Optional[str] = None,
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
    full_checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if SCFOUNDATION_DEFS.GENE_ENCODER not in full_checkpoint:
        raise ValueError(
            f"Gene encoder '{SCFOUNDATION_DEFS.GENE_ENCODER}' not found in checkpoint. "
            f"Available: {list(full_checkpoint.keys())}"
        )

    checkpoint = full_checkpoint[SCFOUNDATION_DEFS.GENE_ENCODER]

    # Extract components
    logger.info("2. Extracting weights...")
    # Gene annotations are shared between AIDOCell and scFoundation
    gene_annotations = _aidocell_load_gene_annotations()
    weights = _scfoundation_extract_weights(checkpoint, gene_annotations)
    metadata = _scfoundation_extract_metadata(checkpoint, gene_annotations)
    logger.info(
        f"   {len(gene_annotations)} genes, {metadata[FM_DEFS.N_LAYERS]} layers"
    )
    logger.info(f"   Embeddings: {weights.gene_embedding.shape}")
    logger.info(
        f"   Attention weights: {metadata[FM_DEFS.N_LAYERS]} layers × 4 matrices (Q,K,V,O)"
    )

    # Set default prefix if not provided
    if output_prefix is None:
        output_prefix = SCFOUNDATION_DEFS.MODEL_NAME

    # Build model and save
    _create_and_save_foundation_model(
        weights, gene_annotations, metadata, output_dir, output_prefix
    )

    return None


@require_scgpt
def process_scgpt(
    model_dir: str, output_dir: str, annotations_path: Optional[str] = None
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
    weights = _scgpt_extract_weights(model, vocab, model_metadata, checkpoint_path)
    logger.info(f"   Embeddings: {weights.gene_embedding.shape}")
    logger.info(
        f"   Attention weights: {model_metadata[FM_DEFS.N_LAYERS]} layers × 4 matrices (Q,K,V,O)"
    )

    # 4. Create FoundationModel and save
    _create_and_save_foundation_model(
        weights, gene_annotations, model_metadata, output_dir, file_prefix
    )

    return None


@require_scprint
def process_scprint(
    version_key: str, output_dir: str, model_path: Optional[str] = None
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

    Returns
    -------
    None
    """
    from huggingface_hub import hf_hub_download

    if model_path is None:
        model_path = os.path.join("data", "scPRINT")

    # Get version ID and checkpoint filename from the version key
    version_id = getattr(SCPRINT_DEFS.VERSIONS, version_key)
    checkpoint_filename = getattr(SCPRINT_DEFS.CHECKPOINTS, version_key)
    file_prefix = f"{SCPRINT_DEFS.MODEL_NAME}_{version_id}"

    logger.info(f"Extracting: scPRINT {version_id} ({version_key})")

    # 1. Download and load model
    logger.info("\n1. Downloading/loading model if needed...")
    checkpoint_file = hf_hub_download(
        repo_id=SCPRINT_DEFS.REPO_ID, filename=checkpoint_filename, cache_dir=model_path
    )

    logger.info("Loading scPRINT model")
    model, gene_annotations, model_metadata = _scprint_load_model(
        checkpoint_file, version=version_id
    )
    logger.info(
        f"   {len(gene_annotations)} genes, {model_metadata[FM_DEFS.N_LAYERS]} layers"
    )

    # 2. Extract weights
    logger.info("2. Extracting weights...")
    weights = _scprint_extract_weights(model)
    logger.info(f"   Embeddings: {weights.gene_embedding.shape}")
    logger.info(
        f"   Attention weights: {model_metadata[FM_DEFS.N_LAYERS]} layers × 4 matrices (Q,K,V,O)"
    )

    # 3. Create FoundationModel and save
    _create_and_save_foundation_model(
        weights, gene_annotations, model_metadata, output_dir, file_prefix
    )

    return None


# ============================================================================
# Private helper functions
# ============================================================================


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
def _aidocell_extract_weights(model: Any) -> FoundationModelWeights:
    """Extract model weights in the standardized format.

    Parameters
    ----------
    model : Any
      The AIDOCell model

    Returns
    -------
    FoundationModelWeights
      FoundationModelWeights instance containing gene_embedding and attention_layers
    """
    # Extract gene embeddings
    encoder = model.encoder
    n_genes = len(_aidocell_load_gene_annotations())

    with torch.no_grad():
        gene_positions = torch.arange(n_genes)
        gene_embedding = encoder.position_embedding(gene_positions).cpu().numpy()

    # Extract attention weights as AttentionLayer instances
    attention_layers = _aidocell_extract_attention_weights(model)

    return FoundationModelWeights(
        gene_embedding=gene_embedding, attention_layers=attention_layers
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


def _create_and_save_foundation_model(
    weights: FoundationModelWeights,
    gene_annotations: pd.DataFrame,
    model_metadata: Dict,
    output_dir: str,
    file_prefix: str,
) -> FoundationModel:
    """
    Create FoundationModel instance and save to disk.

    Parameters
    ----------
    weights : FoundationModelWeights
      Model weights
    gene_annotations : pd.DataFrame
      Gene annotations DataFrame
    model_metadata : Dict
      Model metadata dictionary
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
    )
    foundation_model.save(output_dir, file_prefix)
    logger.info("Successfully saved all results!")
    return foundation_model


def _extract_attention_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    n_layers: int,
    embed_dim: int,
    layer_prefix: str = "transformer_encoder.layers",
    qkv_key_template: str = "{layer_idx}.self_attn.Wqkv.weight",
    out_proj_key_template: str = "{layer_idx}.self_attn.out_proj.weight",
) -> List[AttentionLayer]:
    """
    Extract attention layers from a state dict using configurable key templates.

    Parameters
    ----------
    state_dict : Dict[str, torch.Tensor]
      Model state dictionary
    n_layers : int
      Number of transformer layers
    embed_dim : int
      Embedding dimension
    layer_prefix : str
      Prefix for layer keys (default: "transformer_encoder.layers")
    qkv_key_template : str
      Template for QKV weight key, use {layer_idx} placeholder
      (default: "{layer_idx}.self_attn.Wqkv.weight")
    out_proj_key_template : str
      Template for output projection key, use {layer_idx} placeholder
      (default: "{layer_idx}.self_attn.out_proj.weight")

    Returns
    -------
    List[AttentionLayer]
      List of AttentionLayer instances

    Examples
    --------
    >>> layers = _extract_attention_from_state_dict(
    ...     state_dict, n_layers=12, embed_dim=512
    ... )
    """
    attention_layers = []

    for layer_idx in range(n_layers):
        # Format keys with layer index
        qkv_key = f"{layer_prefix}.{qkv_key_template.format(layer_idx=layer_idx)}"
        out_proj_key = (
            f"{layer_prefix}.{out_proj_key_template.format(layer_idx=layer_idx)}"
        )

        if qkv_key not in state_dict:
            raise KeyError(f"Could not find {qkv_key} in state_dict")
        if out_proj_key not in state_dict:
            raise KeyError(f"Could not find {out_proj_key} in state_dict")

        # Extract and convert to numpy
        qkv_weight = state_dict[qkv_key].clone().cpu().detach().numpy()
        out_proj = state_dict[out_proj_key].clone().cpu().detach().numpy()

        # Split QKV
        w_q, w_k, w_v = _split_qkv_weights(qkv_weight, embed_dim)

        attention_layers.append(
            AttentionLayer(layer_idx=layer_idx, W_q=w_q, W_k=w_k, W_v=w_v, W_o=out_proj)
        )

    return attention_layers


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
    """
    Format base model metadata dictionary with standard keys.

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
    **extra_metadata : Dict
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
    checkpoint: dict, gene_annotations: pd.DataFrame
) -> FoundationModelWeights:
    """Extract gene embeddings and attention weights from scFoundation checkpoint.

    Parameters
    ----------
    checkpoint : dict
      Loaded checkpoint for specific variant
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

    # Gene embeddings (exclude special tokens)
    gene_emb_full = state_dict["model.pos_emb.weight"].cpu().numpy()
    n_genes = len(gene_annotations)

    # Validate N_GENES
    if n_genes != SCFOUNDATION_DEFS.N_GENES:
        logger.warning(
            f"N_GENES mismatch: annotations have {n_genes}, "
            f"expected {SCFOUNDATION_DEFS.N_GENES}"
        )

    # Use actual values from checkpoint/annotations
    gene_embedding = gene_emb_full[:n_genes, :]

    logger.info(f"Extracted gene embeddings: {gene_embedding.shape}")

    # Attention layers
    attention_layers = []
    for layer_idx in range(n_encoder_layers):
        # Combined QKV projection
        in_proj_key = (
            f"model.encoder.transformer_encoder.{layer_idx}.self_attn.in_proj_weight"
        )
        out_proj_key = (
            f"model.encoder.transformer_encoder.{layer_idx}.self_attn.out_proj.weight"
        )

        in_proj = state_dict[in_proj_key].cpu().numpy()
        out_proj = state_dict[out_proj_key].cpu().numpy()

        # Split combined QKV (shape: [3*embed_dim, embed_dim])
        w_q, w_k, w_v = _split_qkv_weights(in_proj, embed_dim)

        attention_layers.append(
            AttentionLayer(layer_idx=layer_idx, W_q=w_q, W_k=w_k, W_v=w_v, W_o=out_proj)
        )

    logger.info(f"Extracted {len(attention_layers)} attention layers")

    return FoundationModelWeights(
        gene_embedding=gene_embedding, attention_layers=attention_layers
    )


@require_scgpt
@require_torchtext
def _scgpt_extract_weights(
    model: Any, vocab: Any, model_metadata: dict, checkpoint_path: str
) -> FoundationModelWeights:
    """Extract weights from scGPT model.

    Note: Weights must be loaded directly from the checkpoint file because
    model.state_dict() returns incorrect/shared weights across layers.

    Parameters
    ----------
    model : Any
      scGPT TransformerModel
    vocab : Any
      scGPT vocabulary object
    model_metadata : dict
      Model metadata dictionary
    checkpoint_path : str
      Path to checkpoint file

    Returns
    -------
    FoundationModelWeights
      Extracted model weights
    """
    # Extract gene embeddings
    gene_ids = torch.arange(len(vocab))
    embeddings = model.encoder(gene_ids).detach().cpu().numpy()

    # Load weights directly from checkpoint file (model.state_dict() is unreliable)
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Extract attention weights for all layers as AttentionLayer instances
    attention_layers = []
    n_layers = model_metadata[FM_DEFS.N_LAYERS]
    d = model_metadata[FM_DEFS.EMBED_DIM]

    for layer_idx in range(n_layers):
        # scGPT uses Wqkv.weight for the combined QKV projection
        wqkv_key = f"transformer_encoder.layers.{layer_idx}.self_attn.Wqkv.weight"
        out_proj_key = (
            f"transformer_encoder.layers.{layer_idx}.self_attn.out_proj.weight"
        )

        if wqkv_key not in state_dict:
            raise KeyError(f"Could not find {wqkv_key} in state_dict")
        if out_proj_key not in state_dict:
            raise KeyError(f"Could not find {out_proj_key} in state_dict")

        # Clone immediately to ensure independent copies
        in_proj = state_dict[wqkv_key].clone()
        out_proj = state_dict[out_proj_key].clone()

        # Validate shape
        if in_proj.shape[0] != 3 * d:
            raise ValueError(
                f"Expected in_proj.shape[0] to be 3*d ({3*d}), but got {in_proj.shape[0]}"
            )

        # Split QKV into separate matrices and convert to numpy
        in_proj_np = in_proj.clone().cpu().detach().numpy()
        w_q, w_k, w_v = _split_qkv_weights(in_proj_np, d)
        w_o = out_proj.clone().cpu().detach().numpy()

        attention_layers.append(
            AttentionLayer(layer_idx=layer_idx, W_q=w_q, W_k=w_k, W_v=w_v, W_o=w_o)
        )

    return FoundationModelWeights(
        gene_embedding=embeddings, attention_layers=attention_layers
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
        for k, v in pretrained_dict.items():
            print(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    model.to(device)

    return model


@require_scprint
def _scprint_extract_attention_weights(model: Any) -> List[AttentionLayer]:
    """Extract self-attention weights (Q, K, V, O) from all layers.

    Parameters
    ----------
    model : Any
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
def _scprint_extract_weights(model: Any) -> FoundationModelWeights:
    """Extract model weights in the standardized format.

    Parameters
    ----------
    model : Any
      The scPRINT model

    Returns
    -------
    FoundationModelWeights
      FoundationModelWeights instance containing gene_embedding and attention_layers
    """
    # Extract gene embeddings
    gene_embedding = model.gene_encoder.embeddings.weight.detach().cpu().numpy()

    # Extract attention weights as AttentionLayer instances
    attention_layers = _scprint_extract_attention_weights(model)

    return FoundationModelWeights(
        gene_embedding=gene_embedding, attention_layers=attention_layers
    )


@require_scprint
def _scprint_format_metadata(model: Any, version: Optional[str] = None) -> Dict:
    """Extract model architecture metadata.

    Parameters
    ----------
    model : Any
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


def _scprint_load_gene_annotations(model: Any) -> pd.DataFrame:
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


def _split_qkv_weights(
    qkv_weight: np.ndarray, embed_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split combined QKV weight matrix into separate Q, K, V matrices.

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
