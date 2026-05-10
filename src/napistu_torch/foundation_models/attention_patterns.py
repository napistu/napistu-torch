"""
Residual-stream embeddings paired with foundation-model attention for analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from napistu.ontologies.constants import ONTOLOGIES
from torch import Tensor

from napistu_torch.foundation_models.constants import (
    COMPARE_EMBEDDINGS_COMPARISONS,
    COMPARE_EMBEDDINGS_SETTINGS,
    FM_EDGELIST,
    FM_LAYER_CONSENSUS_METHODS,
    VALID_COMPARE_EMBEDDINGS_COMPARISONS,
    VALID_FM_LAYER_CONSENSUS_METHODS,
)
from napistu_torch.foundation_models.foundation_models import (
    AttentionLayer,
    FoundationModel,
    FoundationModels,
)
from napistu_torch.foundation_models.gene_embeddings import (
    GeneEmbeddings,
    GeneEmbeddingsSet,
    _get_model_label,
)
from napistu_torch.utils.base_utils import normalize_and_validate_indices
from napistu_torch.utils.pd_utils import calculate_ranks
from napistu_torch.utils.statistics import compare_top_k_union_ranks
from napistu_torch.utils.tensor_utils import (
    compute_correlation_matrix,
    compute_max_abs_over_z,
    compute_max_over_z,
    compute_tensor_ranks,
    compute_tensor_ranks_for_indices,
    find_top_k,
)
from napistu_torch.utils.torch_utils import (
    cleanup_tensors,
    ensure_device,
    memory_manager,
)

logger = logging.getLogger(__name__)


class LayerwiseAttentionInputs:
    """Per-layer residual stream embeddings paired with a model's attention machinery.

    Maps each transformer layer index to a :class:`GeneEmbeddings` for that
    layer's residual (e.g. static or expression-contextualized activations)
    and references the :class:`FoundationModel` that supplies attention weights.

    Typically created by :class:`AttentionPatternsInputs` rather than directly.

    Attributes
    ----------
    residual_stream_embeddings : Dict[int, GeneEmbeddings]
        Residual embedding matrix and metadata per layer index (0 .. n_layers - 1).
    foundation_model : FoundationModel
        Source model (attention layers, n_heads, vocabulary, etc.).

    Properties
    ----------
    ordered_gene_ids : List[str]
        Gene IDs in row order (same for every layer).
    n_genes : int
        Number of genes.
    embed_dim : int
        Embedding dimensionality.
    n_layers : int
        Number of attention layers (from the model).
    n_heads : int
        Number of attention heads (from the model).
    attention_layers : List[AttentionLayer]
        Attention layers (from the model).
    model_name : str
        Model display name (from the model).

    Examples
    --------
    >>> ae = LayerwiseAttentionInputs(residual_stream_embeddings=per_layer, foundation_model=model)
    >>> attn = ae.compute_attention(layer_idx=0)
    >>> consensus = ae.compute_consensus_attention()
    """

    def __init__(
        self,
        residual_stream_embeddings: Dict[int, GeneEmbeddings],
        foundation_model: FoundationModel,
    ):
        if not isinstance(residual_stream_embeddings, dict):
            raise TypeError(
                f"residual_stream_embeddings must be a Dict[int, GeneEmbeddings], "
                f"got {type(residual_stream_embeddings)}"
            )
        if not residual_stream_embeddings:
            raise ValueError("residual_stream_embeddings must not be empty")
        for layer_idx, ge in residual_stream_embeddings.items():
            if not isinstance(layer_idx, int):
                raise TypeError(
                    f"residual_stream_embeddings keys must be int layer indices, "
                    f"got {type(layer_idx)}"
                )
            if not isinstance(ge, GeneEmbeddings):
                raise TypeError(
                    f"residual_stream_embeddings values must be GeneEmbeddings, "
                    f"got {type(ge)} at layer {layer_idx}"
                )
        if not isinstance(foundation_model, FoundationModel):
            raise TypeError(
                f"foundation_model must be a FoundationModel, "
                f"got {type(foundation_model)}"
            )

        expected_layers = set(range(foundation_model.n_layers))
        provided_layers = set(residual_stream_embeddings.keys())
        if provided_layers != expected_layers:
            missing = sorted(expected_layers - provided_layers)
            extra = sorted(provided_layers - expected_layers)
            msg_parts = []
            if missing:
                msg_parts.append(f"missing layers {missing}")
            if extra:
                msg_parts.append(f"unexpected layers {extra}")
            raise ValueError(
                f"residual_stream_embeddings layers do not match model. "
                f"{'; '.join(msg_parts)}. "
                f"Expected {{0..{foundation_model.n_layers - 1}}}."
            )

        any_ge = next(iter(residual_stream_embeddings.values()))
        emb_label = _get_model_label(any_ge.model_name, any_ge.model_variant)
        if emb_label != foundation_model.full_name:
            raise ValueError(
                f"Embedding model label '{emb_label}' does not match "
                f"foundation model '{foundation_model.full_name}'"
            )

        if len(foundation_model.weights.attention_layers) == 0:
            raise ValueError(
                f"Model '{foundation_model.full_name}' has no attention layers."
            )

        self.residual_stream_embeddings = residual_stream_embeddings
        self.foundation_model = foundation_model

    # --- Properties (shortcuts) ---

    @property
    def ordered_gene_ids(self) -> List[str]:
        return self.residual_stream_embeddings[0].ordered_gene_ids

    @property
    def n_genes(self) -> int:
        return self.residual_stream_embeddings[0].n_genes

    @property
    def embed_dim(self) -> int:
        return self.residual_stream_embeddings[0].embed_dim

    @property
    def n_layers(self) -> int:
        """Number of layers present in residual_stream_embeddings."""
        return len(self.residual_stream_embeddings)

    @property
    def n_heads(self) -> int:
        """Number of attention heads."""
        return self.foundation_model.n_heads

    @property
    def attention_layers(self) -> List[AttentionLayer]:
        """Attention layers from the model."""
        return self.foundation_model.weights.attention_layers

    @property
    def model_name(self) -> str:
        """Model display name."""
        return self.foundation_model.full_name

    def compare_layer_attention_consistency(
        self,
        top_k: int,
        by_absolute_value: bool,
        ignore_self_attention: bool,
        target_ids: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:

        _, gene_ids = self._get_gene_mask(target_ids)

        top_k_attention_edges = self.get_top_attentions(
            k=top_k,
            target_ids=gene_ids,
            by_absolute_value=by_absolute_value,
            ignore_self_attention=ignore_self_attention,
            verbose=verbose,
        )
        # re-extract the top-k edges across all layers
        distinct_top_edges = top_k_attention_edges[
            ["from_gene", "to_gene"]
        ].drop_duplicates()

        # extract the attention scores for the distinct top-k edges across all layers
        top_k_union = self.get_specific_attentions(
            distinct_top_edges,
            target_ids=gene_ids,
            compute_ranks=True,
            by_absolute_value=by_absolute_value,
            verbose=verbose,
        )

        wide_top_k_union = top_k_union.pivot(
            index=["from_gene", "to_gene"], columns="layer", values="attention"
        )

        corr = compute_correlation_matrix(wide_top_k_union.to_numpy())
        top_k_ranks = compare_top_k_union_ranks(
            top_k_union,
            grouping_vars=["layer"],
            defining_vars=["from_gene", "to_gene"],
            max_rank=len(gene_ids) ** 2,
            top_k=top_k,
            rank_col="attention_rank",
        )

        return corr, top_k_ranks

    def compute_attention(
        self,
        layer_idx: int,
        target_ids: Optional[List[str]] = None,
        apply_softmax: bool = True,
        return_tensor: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Union[Tensor, np.ndarray]:
        """Compute attention scores for a specific layer.

        Uses this instance's embedding matrix with the model's attention
        weight matrices to compute the attention pattern.

        Parameters
        ----------
        layer_idx : int
            Index of the layer to compute attention for.
        target_ids : List[str], optional
            Subset of gene IDs to compute attention for. Results will be
            ordered to match target_ids. Must be a subset of this embedding's
            gene IDs. If None, uses all genes in embedding order.
        apply_softmax : bool, optional
            If True, apply softmax to get attention probabilities (default: True).
            If False, return raw scores (Q @ K.T / sqrt(d_k)).
        return_tensor : bool, optional
            If True, return torch.Tensor (default: False).
        device : str or torch.device, optional
            Device for computation (default: None to auto-select).

        Returns
        -------
        torch.Tensor or np.ndarray
            Attention matrix of shape (n_target, n_target) if target_ids
            is provided, otherwise (n_genes, n_genes). Rows and columns
            are ordered to match target_ids when provided.

        Raises
        ------
        ValueError
            If layer_idx is out of range or target_ids contains unknown genes.
        """
        if layer_idx not in self.residual_stream_embeddings:
            raise ValueError(
                f"Layer index {layer_idx} not found in residual_stream_embeddings. "
                f"Available layers: {sorted(self.residual_stream_embeddings.keys())}"
            )

        gene_mask, _ = self._get_gene_mask(target_ids)

        embeddings = self._embedding_for_layer(layer_idx)
        if gene_mask is not None:
            embeddings = embeddings[gene_mask]

        layer = self.attention_layers[layer_idx]

        attention = layer.compute_attention_pattern(
            embeddings=embeddings,
            n_heads=self.n_heads,
            apply_softmax=apply_softmax,
            return_tensor=True,
            device=device,
        )

        # Reorder from embedding order to target_ids order if needed
        if target_ids is not None:
            masked_gene_ids = [
                gid for gid, m in zip(self.ordered_gene_ids, gene_mask) if m
            ]
            masked_id_to_idx = {gid: i for i, gid in enumerate(masked_gene_ids)}
            reorder_indices = [masked_id_to_idx[gid] for gid in target_ids]
            attention = attention[reorder_indices, :][:, reorder_indices]

        if return_tensor:
            return attention
        else:
            return attention.cpu().numpy()

    def compute_consensus_attention(
        self,
        target_ids: Optional[List[str]] = None,
        consensus_method: str = FM_LAYER_CONSENSUS_METHODS.ABSOLUTE_ARGMAX,
        apply_softmax: bool = False,
        return_layer_indices: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Compute consensus attention across all layers.

        For each gene pair, aggregates attention values across layers using the
        specified consensus method. The default method ("absolute-argmax") finds
        the layer with the strongest attention (by absolute value) and returns
        that value with its original sign preserved.

        Parameters
        ----------
        target_ids : List[str], optional
            Subset of gene IDs to compute consensus attention for. Results will be
            ordered to match target_ids. Must be a subset of this embedding's
            gene IDs. If None, uses all genes in embedding order.
        consensus_method : str, optional
            Method for aggregating across layers:
            - "absolute-argmax" (default): Layer with max absolute value, sign preserved
            - "max": Layer with maximum value
            - "sum": Sum across all layers
        apply_softmax : bool, optional
            If True, apply softmax per layer (default: False).
        return_layer_indices : bool, optional
            If True, also return which layer had max attention (default: False).
        device : str or torch.device, optional
            Device for computation (default: None to auto-select).

        Returns
        -------
        torch.Tensor
            Consensus attention of shape (n_genes, n_genes).
        torch.Tensor (optional)
            If return_layer_indices=True, layer indices of shape (n_genes, n_genes).
        """

        n_genes = len(target_ids) if target_ids is not None else self.n_genes
        layer_indices = sorted(self.residual_stream_embeddings.keys())

        all_attention = torch.zeros(
            (n_genes, n_genes, self.n_layers), dtype=torch.float32
        )

        for stack_pos, layer_idx in enumerate(layer_indices):
            attention = self.compute_attention(
                layer_idx=layer_idx,
                target_ids=target_ids,
                apply_softmax=apply_softmax,
                return_tensor=True,
                device=device,
            )
            all_attention[:, :, stack_pos] = attention

        if consensus_method == FM_LAYER_CONSENSUS_METHODS.ABSOLUTE_ARGMAX:
            return compute_max_abs_over_z(
                all_attention, return_indices=return_layer_indices
            )
        elif consensus_method == FM_LAYER_CONSENSUS_METHODS.MAX:
            return compute_max_over_z(
                all_attention, return_indices=return_layer_indices
            )
        elif consensus_method == FM_LAYER_CONSENSUS_METHODS.SUM:
            if return_layer_indices:
                return all_attention.sum(dim=2), torch.zeros(
                    (self.n_genes, self.n_genes), dtype=torch.long
                )
            return all_attention.sum(dim=2)
        else:
            raise ValueError(
                f"Unknown consensus_method '{consensus_method}'. "
                f"Supported methods: {VALID_FM_LAYER_CONSENSUS_METHODS}"
            )

    def get_specific_attentions(
        self,
        edge_list: pd.DataFrame,
        layer_indices: Optional[List[int]] = None,
        target_ids: Optional[List[str]] = None,
        apply_softmax: bool = False,
        compute_ranks: bool = False,
        by_absolute_value: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Extract specific attention values across layers for given edges.

        Extracts the exact attention values for specific gene pairs across
        specified layers. Useful for analyzing how specific relationships
        vary across layers.

        Parameters
        ----------
        edge_list : pd.DataFrame
            DataFrame with at minimum 'from_gene' and 'to_gene' columns.
        layer_indices : List[int], optional
            Layers to extract from. If None, uses all layers.
        target_ids : List[str], optional
            Subset of gene IDs to use for attention computation. Must be a
            subset of this embedding's gene IDs. If None, uses all genes.
        apply_softmax : bool, optional
            If True, use softmax-normalized attention (default: False).
        compute_ranks : bool, optional
            If True, add attention ranks to output (default: False).
        by_absolute_value : bool, optional
            If True, rank by absolute value (default: True).
            Only used if compute_ranks=True.
        device : str or torch.device, optional
            Device for computation (default: None to auto-select).
        verbose : bool, optional
            Print progress (default: False).

        Returns
        -------
        pd.DataFrame
            Columns: from_gene, to_gene, layer, attention,
            and optionally attention_rank.
        """
        device = ensure_device(device, allow_autoselect=True)

        _, gene_ids = self._get_gene_mask(target_ids)

        # Convert edge list to indices ONCE
        edge_df = _edgelist_to_indices(
            edge_list=edge_list,
            gene_ids=gene_ids,
            verbose=verbose,
        )

        if layer_indices is None:
            layer_indices = sorted(self.residual_stream_embeddings.keys())
        else:
            layer_indices = normalize_and_validate_indices(
                indices=layer_indices,
                max_value=self.n_layers,
                param_name="layer_indices",
            )

        results = []

        with memory_manager(device):
            from_idx_tensor = (
                torch.from_numpy(edge_df[FM_EDGELIST.FROM_IDX].values).long().to(device)
            )
            to_idx_tensor = (
                torch.from_numpy(edge_df[FM_EDGELIST.TO_IDX].values).long().to(device)
            )

            for layer_idx in layer_indices:
                if verbose:
                    logger.info(f"Extracting attentions from layer {layer_idx}...")

                attention = self.compute_attention(
                    layer_idx=layer_idx,
                    target_ids=target_ids,
                    apply_softmax=apply_softmax,
                    return_tensor=True,
                    device=device,
                )

                # Extract edges using tensor indexing
                edge_attentions = attention[from_idx_tensor, to_idx_tensor]

                layer_df = edge_df[[FM_EDGELIST.FROM_GENE, FM_EDGELIST.TO_GENE]].copy()
                layer_df[FM_EDGELIST.LAYER] = layer_idx
                layer_df[FM_EDGELIST.ATTENTION] = edge_attentions.cpu().numpy()

                if compute_ranks:
                    if verbose:
                        logger.info(f"Calculating ranks for layer {layer_idx}...")

                    edge_ranks = compute_tensor_ranks_for_indices(
                        attention,
                        (from_idx_tensor, to_idx_tensor),
                        by_absolute_value=by_absolute_value,
                    )
                    layer_df[FM_EDGELIST.ATTENTION_RANK] = edge_ranks.cpu().numpy()

                results.append(layer_df)
                cleanup_tensors(attention, edge_attentions)

        all_attentions = pd.concat(results, ignore_index=True)

        if verbose:
            logger.info(
                f"Extracted {len(all_attentions)} total attention values "
                f"({len(edge_df)} edges × {len(layer_indices)} layers)"
            )

        return all_attentions

    def get_top_attentions(
        self,
        k: int,
        layer_indices: Optional[List[int]] = None,
        target_ids: Optional[List[str]] = None,
        apply_softmax: bool = False,
        by_absolute_value: bool = True,
        compute_ranks: bool = False,
        ignore_self_attention: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Extract top-k strongest attention edges across layers.

        For each layer, identifies the k gene pairs with highest attention
        values and returns them as a DataFrame.

        Parameters
        ----------
        k : int
            Number of top edges to extract per layer.
        layer_indices : List[int], optional
            Layers to analyze. If None, uses all layers.
        target_ids : List[str], optional
            Subset of gene identifiers to analyze. Must be a subset of
            this embedding's gene IDs. If None, uses all genes.
        apply_softmax : bool, optional
            If True, use softmax-normalized attention (default: False).
        by_absolute_value : bool, optional
            If True, rank by absolute attention value (default: True).
        compute_ranks : bool, optional
            If True, add attention ranks to output (default: False).
        ignore_self_attention : bool, optional
            If True, exclude self-attention edges (default: False).
        device : str or torch.device, optional
            Device for computation (default: None to auto-select).
        verbose : bool, optional
            Print progress (default: False).

        Returns
        -------
        pd.DataFrame
            Columns: layer, from_idx, to_idx, from_gene, to_gene, attention,
            and optionally attention_rank.
        """

        device = ensure_device(device, allow_autoselect=True)

        _, gene_ids = self._get_gene_mask(target_ids)

        if layer_indices is None:
            layer_indices = sorted(self.residual_stream_embeddings.keys())
        else:
            layer_indices = normalize_and_validate_indices(
                indices=layer_indices,
                max_value=self.n_layers,
                param_name="layer_indices",
            )

        results = []

        with memory_manager(device):
            for layer_idx in layer_indices:
                if verbose:
                    value_type = "absolute value" if by_absolute_value else "raw value"
                    logger.info(
                        f"Extracting top-{k} edges from layer {layer_idx} "
                        f"by {value_type}..."
                    )

                attention = self.compute_attention(
                    layer_idx=layer_idx,
                    target_ids=target_ids,
                    apply_softmax=apply_softmax,
                    return_tensor=True,
                    device=device,
                )

                layer_df = _find_top_k_edges_in_attention_layer(
                    attention=attention,
                    k=k,
                    layer_idx=layer_idx,
                    gene_ids=gene_ids,
                    by_absolute_value=by_absolute_value,
                    ignore_self_attention=ignore_self_attention,
                )

                results.append(layer_df)
                cleanup_tensors(attention)

        all_edges = pd.concat(results, ignore_index=True)

        if compute_ranks:
            all_edges[FM_EDGELIST.ATTENTION_RANK] = calculate_ranks(
                df=all_edges,
                value_col=FM_EDGELIST.ATTENTION,
                by_absolute_value=by_absolute_value,
                grouping_vars=FM_EDGELIST.LAYER,
            )

        if verbose:
            logger.info(
                f"Extracted {len(all_edges)} total edges "
                f"across {len(layer_indices)} layers"
            )

        return all_edges

    def _embedding_for_layer(self, layer_idx: int) -> np.ndarray:
        if layer_idx not in self.residual_stream_embeddings:
            raise ValueError(
                f"Layer index {layer_idx} not found in residual_stream_embeddings. "
                f"Available layers: {sorted(self.residual_stream_embeddings.keys())}"
            )
        return self.residual_stream_embeddings[layer_idx].embedding

    def _get_gene_mask(self, target_ids=None):
        return self.residual_stream_embeddings[0].get_gene_mask(target_ids)

    def __repr__(self) -> str:
        layers = sorted(self.residual_stream_embeddings.keys())
        return (
            f"LayerwiseAttentionInputs("
            f"model={self.model_name}, "
            f"layers={layers}, "
            f"n_genes={self.n_genes}, "
            f"embed_dim={self.embed_dim}"
            f")"
        )


class AttentionPatternsInputs:
    """Per-layer residual stream embeddings paired with attention machinery for cross-model analysis.

    Takes a :class:`GeneEmbeddingsSet` whose entries are residual-stream
    :class:`GeneEmbeddings` (one matrix per transformer layer, per model and category),
    aligned to common genes. Constructor arguments are validated against a
    :class:`FoundationModels` container; each group is wrapped as a
    :class:`LayerwiseAttentionInputs` (per-layer dict plus that group's
    :class:`FoundationModel` for attention weights).

    Attention computation needs:
    1. Residual stream embeddings per layer (here, as ``GeneEmbeddings`` in ``embeddings_set``)
    2. Attention weight matrices (W_q, W_k, W_v, W_o from :class:`AttentionLayer`)
    3. ``n_heads`` (from model metadata)

    Attention weights and head counts (items 2 and 3) are fixed per foundation model;
    the per-layer residual streams (item 1) vary by context (e.g. expression category).
    This class groups per-layer residual streams into ``(model, category)`` units and
    exposes them keyed by ``"{full_name}/{category}"``.

    Parameters
    ----------
    embeddings_set : GeneEmbeddingsSet
        Aligned residual-stream embeddings. All entries must share the same common genes
        in the same row order; layer indices and model metadata must allow grouping
        by model and category.
    foundation_models : FoundationModels
        Container of :class:`FoundationModel` instances. Each embedding in ``embeddings_set``
        must map to exactly one model (via ``model_name`` + ``model_variant`` → ``full_name``).

    Attributes
    ----------
    embeddings_set : GeneEmbeddingsSet
        The aligned per-layer embeddings backing this set.
    attended_embeddings : Dict[str, LayerwiseAttentionInputs]
        One :class:`LayerwiseAttentionInputs` per ``(model, category)`` group, keyed by
        ``"{full_name}/{category}"``.

    Properties
    ----------
    n_embeddings : int
        Number of ``(model × category)`` groups (length of ``attended_embeddings``),
        not the count of individual layer tensors.
    n_common_genes : int
        Number of common genes (delegates to ``embeddings_set``).
    common_gene_ids : List[str]
        Gene IDs shared across all embeddings (delegates to ``embeddings_set``).
    embedding_keys : List[str]
        Keys for each group (same as ``attended_embeddings`` keys).
    model_names : List[str]
        Unique model names referenced by groups (order preserved).

    Public Methods
    --------------
    from_expression(foundation_models: FoundationModels, dataset_name: str, category: str, align_on: str = ONTOLOGIES.ENSEMBL_GENE, verbose: bool = True) -> "AttentionPatternsInputs":
        Build expression-contextualized per-layer residual streams for a single category.
    get_consensus_attention(k: int = 10000, target_ids: Optional[List[str]] = None, consensus_method: str = FM_LAYER_CONSENSUS_METHODS.ABSOLUTE_ARGMAX, by_absolute_value: bool = True, reextract_union: bool = False, apply_softmax: bool = False, compute_ranks: bool = False, ignore_self_attention: bool = False, return_original_and_reextracted: bool = False, device: Optional[Union[str, torch.device]] = None, verbose: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        Compute consensus attention across all layers for each ``(model × category)`` group.
    get_consensus_top_attentions(k: int = 10000, target_ids: Optional[List[str]] = None, consensus_method: str = FM_LAYER_CONSENSUS_METHODS.ABSOLUTE_ARGMAX, by_absolute_value: bool = True, reextract_union: bool = False, apply_softmax: bool = False, compute_ranks: bool = False, ignore_self_attention: bool = False, return_original_and_reextracted: bool = False, device: Optional[Union[str, torch.device]] = None, verbose: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        Extract top-k consensus attention edges across groups.
    get_specific_attentions(edges: pd.DataFrame, target_ids: Optional[List[str]] = None, apply_softmax: bool = False, compute_ranks: bool = False, by_absolute_value: bool = True, verbose: bool = False) -> pd.DataFrame:
        Extract specific attention edges across layers.
    get_top_attentions(k: int, layer_indices: Optional[List[int]] = None, target_ids: Optional[List[str]] = None, apply_softmax: bool = False, by_absolute_value: bool = True, compute_ranks: bool = False, ignore_self_attention: bool = False, device: Optional[Union[str, torch.device]] = None, verbose: bool = False) -> pd.DataFrame:
        Extract top-k strongest attention edges across layers.

    Examples
    --------
    >>> # One (model × category) group per model from expression residual streams
    >>> attended = AttentionPatternsInputs.from_expression(
    ...     foundation_models, dataset_name="efthymiou2025", category="adipocyte (0)"
    ... )

    >>> # From a pre-built GeneEmbeddingsSet (per-layer residual streams already aligned)
    >>> attended = AttentionPatternsInputs(embeddings_set, foundation_models)
    """

    def __init__(
        self,
        embeddings_set: GeneEmbeddingsSet,
        foundation_models: FoundationModels,
    ):
        # Validate types
        if not isinstance(embeddings_set, GeneEmbeddingsSet):
            raise TypeError(
                f"embeddings_set must be a GeneEmbeddingsSet, "
                f"got {type(embeddings_set)}"
            )
        if not isinstance(foundation_models, FoundationModels):
            raise TypeError(
                f"foundation_models must be a FoundationModels, "
                f"got {type(foundation_models)}"
            )

        # Build mapping: embedding key -> model full_name
        # Each GeneEmbeddings carries model_name and model_variant;
        # combine them the same way FoundationModel.full_name does.
        embedding_to_model: Dict[str, str] = {}
        available_model_names = set(foundation_models.model_names)

        for key, emb in embeddings_set.items():
            if emb.model_name is None:
                raise ValueError(
                    f"Embedding '{key}' has no model_name set. "
                    f"Cannot map to a FoundationModel."
                )

            full_name = _get_model_label(emb.model_name, emb.model_variant)

            if full_name not in available_model_names:
                raise ValueError(
                    f"Embedding '{key}' maps to model '{full_name}', "
                    f"but no matching FoundationModel was found. "
                    f"Available models: {sorted(available_model_names)}"
                )

            embedding_to_model[key] = full_name

        # Validate that every referenced model has attention layers
        referenced_models = set(embedding_to_model.values())
        for model_name in referenced_models:
            model = foundation_models.get_model(model_name)
            if len(model.weights.attention_layers) == 0:
                raise ValueError(
                    f"Model '{model_name}' has no attention layers. "
                    f"LayerwiseAttentionInputs requires models with attention weights."
                )

        # Build LayerwiseAttentionInputs instances
        groups = _group_embeddings_by_model_and_category(
            embeddings_set, embedding_to_model
        )

        attended: Dict[str, LayerwiseAttentionInputs] = {}
        for group_key, layer_embeddings in groups.items():
            full_name, category = group_key
            model = foundation_models.get_model(full_name)
            attended[f"{full_name}/{category}"] = LayerwiseAttentionInputs(
                residual_stream_embeddings=layer_embeddings,
                foundation_model=model,
            )

        self.embeddings_set = embeddings_set
        self.attended_embeddings = attended

    @property
    def common_gene_ids(self) -> List[str]:
        """Gene IDs shared across all embeddings."""
        return self.embeddings_set.common_gene_ids

    @property
    def n_embeddings(self) -> int:
        """Number of (model × category) groups in ``attended_embeddings`` (not per-layer count)."""
        return len(self.attended_embeddings)

    @property
    def n_common_genes(self) -> int:
        """Number of common genes."""
        return self.embeddings_set.n_common_genes

    @property
    def embedding_keys(self) -> List[str]:
        """Keys for each ``(model × category)`` group (``attended_embeddings`` keys)."""
        return list(self.attended_embeddings.keys())

    @property
    def model_names(self) -> List[str]:
        """Unique model names referenced by groups (preserves order)."""
        seen = set()
        result = []
        for ae in self.attended_embeddings.values():
            if ae.model_name not in seen:
                seen.add(ae.model_name)
                result.append(ae.model_name)
        return result

    def compare(
        self,
        comparison_types: List[str] = VALID_COMPARE_EMBEDDINGS_COMPARISONS,
        top_k: int = 10000,
        consensus_method: str = FM_LAYER_CONSENSUS_METHODS.SUM,
        by_absolute_value: bool = False,
        ignore_self_attention: bool = True,
        verbose: bool = False,
    ) -> Dict[str, Any]:

        invalid_comparison_types = set(comparison_types) - set(
            VALID_COMPARE_EMBEDDINGS_COMPARISONS
        )
        if invalid_comparison_types:
            raise ValueError(
                f"The following requested comparison types are not valid: {invalid_comparison_types}. Valid comparison types: {VALID_COMPARE_EMBEDDINGS_COMPARISONS}."
            )

        if consensus_method not in VALID_FM_LAYER_CONSENSUS_METHODS:
            raise ValueError(
                f"The requested consensus method is not valid: {consensus_method}. Valid consensus methods: {VALID_FM_LAYER_CONSENSUS_METHODS}."
            )

        # precalculate and cache operations that take more than a minute or so
        comparisons = dict()
        n_genes = self.n_common_genes

        # gene embedding correlations
        if (
            COMPARE_EMBEDDINGS_COMPARISONS.RESIDUAL_STREAM_CORRELATIONS
            in comparison_types
        ):
            logger.info("Calculating residual stream correlations...")
            comparisons[COMPARE_EMBEDDINGS_COMPARISONS.RESIDUAL_STREAM_CORRELATIONS] = (
                self.embeddings_set.compare_embeddings(verbose=verbose)
            )
        else:
            comparisons[COMPARE_EMBEDDINGS_COMPARISONS.RESIDUAL_STREAM_CORRELATIONS] = (
                None
            )

        # within model layer x layer comparisons (correlations and rank agreement)
        if COMPARE_EMBEDDINGS_COMPARISONS.MODEL_LAYER_CORRELATIONS in comparison_types:
            logger.info("Calculating within model layer x layer comparisons...")
            (
                comparisons[COMPARE_EMBEDDINGS_COMPARISONS.MODEL_LAYER_CORRELATIONS],
                comparisons[COMPARE_EMBEDDINGS_COMPARISONS.MODEL_LAYER_RANK_AGREEMENT],
            ) = self.get_within_embeddings_layer_comparisons(
                top_k, ignore_self_attention, by_absolute_value, verbose
            )
        else:
            comparisons[COMPARE_EMBEDDINGS_COMPARISONS.MODEL_LAYER_CORRELATIONS] = None
            comparisons[COMPARE_EMBEDDINGS_COMPARISONS.MODEL_LAYER_RANK_AGREEMENT] = (
                None
            )

        # cross model x layer top attentions
        if (
            COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_X_LAYER_TOP_ATTENTIONS
            in comparison_types
        ):
            logger.info("Calculating cross model x layer top attentions...")
            comparisons[
                COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_X_LAYER_TOP_ATTENTIONS
            ] = self.get_top_attentions(
                k=top_k,
                by_absolute_value=by_absolute_value,
                reextract_union=True,
                compute_ranks=True,
                ignore_self_attention=ignore_self_attention,
                verbose=verbose,
            )
        else:
            comparisons[
                COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_X_LAYER_TOP_ATTENTIONS
            ] = None

        if (
            COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_X_LAYER_RANK_AGREEMENT
            in comparison_types
        ):
            logger.info(
                "Comparing ranks of topK attentions in one model/layer to all other models/layers..."
            )
            comparisons[
                COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_X_LAYER_RANK_AGREEMENT
            ] = compare_top_k_union_ranks(
                comparisons[
                    COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_X_LAYER_TOP_ATTENTIONS
                ],
                grouping_vars=[FM_EDGELIST.MODEL, FM_EDGELIST.LAYER],
                defining_vars=[FM_EDGELIST.FROM_GENE, FM_EDGELIST.TO_GENE],
                max_rank=n_genes**2,
                top_k=top_k,
                rank_col=FM_EDGELIST.ATTENTION_RANK,
            )
        else:
            comparisons[
                COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_X_LAYER_RANK_AGREEMENT
            ] = None

        if (
            COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_CONSENSUS_TOP_ATTENTIONS
            in comparison_types
        ):
            logger.info("Calculating cross model consensus top attentions...")
            comparisons[
                COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_CONSENSUS_TOP_ATTENTIONS
            ] = self.get_consensus_top_attentions(
                k=top_k,
                consensus_method=consensus_method,
                by_absolute_value=by_absolute_value,
                reextract_union=True,
                compute_ranks=True,
                ignore_self_attention=ignore_self_attention,
                verbose=verbose,
            )
        else:
            comparisons[
                COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_CONSENSUS_TOP_ATTENTIONS
            ] = None

        if (
            COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_CONSENSUS_TOP_ATTENTIONS_RANK_AGREEMENT
            in comparison_types
        ):
            comparisons[
                COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_CONSENSUS_TOP_ATTENTIONS_RANK_AGREEMENT
            ] = compare_top_k_union_ranks(
                comparisons[
                    COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_CONSENSUS_TOP_ATTENTIONS
                ],
                grouping_vars=[FM_EDGELIST.MODEL],
                defining_vars=[FM_EDGELIST.FROM_GENE, FM_EDGELIST.TO_GENE],
                max_rank=n_genes**2,
                top_k=top_k,
                rank_col=FM_EDGELIST.ATTENTION_RANK,
            )
        else:
            comparisons[
                COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_CONSENSUS_TOP_ATTENTIONS_RANK_AGREEMENT
            ] = None

        comparisons[COMPARE_EMBEDDINGS_COMPARISONS.SETTINGS] = {
            COMPARE_EMBEDDINGS_SETTINGS.TOP_K: top_k,
            COMPARE_EMBEDDINGS_SETTINGS.CONSENSUS_METHOD: consensus_method,
            COMPARE_EMBEDDINGS_SETTINGS.BY_ABSOLUTE_VALUE: by_absolute_value,
            COMPARE_EMBEDDINGS_SETTINGS.IGNORE_SELF_ATTENTION: ignore_self_attention,
            COMPARE_EMBEDDINGS_SETTINGS.EMBEDDING_KEYS: self.embedding_keys,
            COMPARE_EMBEDDINGS_SETTINGS.N_GENES: n_genes,
        }

        return comparisons

    @classmethod
    def from_expression(
        cls,
        foundation_models: FoundationModels,
        dataset_name: str,
        category: str,
        align_on: str = ONTOLOGIES.ENSEMBL_GENE,
        verbose: bool = True,
    ) -> "AttentionPatternsInputs":
        """Build an :class:`AttentionPatternsInputs` from expression-contextualized per-layer residual streams.

        For each model, collects per-layer :class:`GeneEmbeddings` for the specified
        dataset and category, aligns them to common genes, and constructs
        :class:`LayerwiseAttentionInputs` grouped by model and category.

        Parameters
        ----------
        foundation_models : FoundationModels
            Container with one or more loaded foundation models. Each model must have
            dataset_gene_embeddings containing the specified dataset and category.
        dataset_name : str
            Name of the expression dataset (e.g., 'efthymiou2025').
        category : str
            Category within the dataset (e.g., 'adipocyte (0)', 'T_cell').
        align_on : str, optional
            Ontology column for gene alignment (default: 'ensembl_gene').
        verbose : bool, optional
            Extra reporting (default: True)

        Returns
        -------
        AttentionPatternsInputs
            Container with one ``(model × category)`` group per model, each holding
            aligned residual streams and attention machinery.

        Raises
        ------
        ValueError
            If any model lacks dataset_gene_embeddings.
            If the specified dataset is not found in any model.
            If the specified category is not found in any model's dataset.

        Examples
        --------
        >>> models = FoundationModels.load_multiple(dir, ['scGPT', 'scPRINT'])
        >>> attended = AttentionPatternsInputs.from_expression(
        ...     models, dataset_name="efthymiou2025", category="adipocyte (0)"
        ... )
        >>> attended.n_embeddings
        2
        """
        if not isinstance(foundation_models, FoundationModels):
            raise TypeError(
                f"foundation_models must be a FoundationModels, "
                f"got {type(foundation_models)}"
            )

        expression_embeddings: List[GeneEmbeddings] = []

        for model in foundation_models.models:
            if model.dataset_gene_embeddings is not None:
                ge_set = model.dataset_gene_embeddings[dataset_name]
                layer_embeddings = _get_category_layer_embeddings(
                    ge_set, category, model, dataset_name
                )
            else:
                layer_embeddings = model.load_category_residuals(
                    dataset_name, category
                ).values()

        expression_embeddings.extend(layer_embeddings)

        embeddings_set = GeneEmbeddingsSet.from_gene_embeddings(
            expression_embeddings, align_on=align_on, verbose=verbose
        )

        return cls(
            embeddings_set=embeddings_set,
            foundation_models=foundation_models,
        )

    def get_consensus_attentions(
        self,
        target_ids: Optional[List[str]] = None,
        consensus_method: str = FM_LAYER_CONSENSUS_METHODS.ABSOLUTE_ARGMAX,
        apply_softmax: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tensor:
        """Compute consensus attention for each embedding in the set.

        For each embedding, computes consensus attention across all layers
        using the specified method.

        Parameters
        ----------
        target_ids : List[str], optional
            Subset of gene IDs. If None, uses all common genes.
        consensus_method : str, optional
            Aggregation method across layers (default: 'absolute-argmax').
        apply_softmax : bool, optional
            If True, apply softmax per layer (default: False).
        device : str or torch.device, optional
            Device for computation (default: None to auto-select).

        Returns
        -------
        Tensor
            3D tensor of shape (n_embeddings, n_genes, n_genes).
        """
        gene_ids = target_ids if target_ids is not None else self.common_gene_ids
        n_genes = len(gene_ids)

        cross_embedding_attention = torch.zeros(
            (self.n_embeddings, n_genes, n_genes), dtype=torch.float32
        )

        for i, (key, ae) in enumerate(self.attended_embeddings.items()):
            logger.info(f"Computing consensus attention for {key}...")

            attention = ae.compute_consensus_attention(
                target_ids=target_ids,
                consensus_method=consensus_method,
                apply_softmax=apply_softmax,
                device=device,
            )

            cross_embedding_attention[i] = attention

        return cross_embedding_attention

    def get_consensus_top_attentions(
        self,
        k: int = 10000,
        target_ids: Optional[List[str]] = None,
        consensus_method: str = FM_LAYER_CONSENSUS_METHODS.ABSOLUTE_ARGMAX,
        by_absolute_value: bool = True,
        reextract_union: bool = False,
        apply_softmax: bool = False,
        compute_ranks: bool = False,
        ignore_self_attention: bool = False,
        return_original_and_reextracted: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Extract top-k consensus attention edges across embeddings.

        For each embedding:
        1. Compute consensus attention across all layers
        2. Extract top-k strongest edges

        Optionally re-extract the union of all top edges from every
        embedding's consensus.

        Parameters
        ----------
        k : int, optional
            Number of top edges per embedding (default: 10000).
        target_ids : List[str], optional
            Subset of gene IDs. If None, uses all common genes.
        consensus_method : str, optional
            Aggregation method across layers (default: 'absolute-argmax').
        by_absolute_value : bool, optional
            If True, rank by absolute value (default: True).
        reextract_union : bool, optional
            If True, re-extract union from all embeddings (default: False).
        apply_softmax : bool, optional
            If True, use softmax attention (default: False).
        compute_ranks : bool, optional
            If True, add ranks to output (default: False).
        ignore_self_attention : bool, optional
            If True, exclude self-attention edges (default: False).
        return_original_and_reextracted : bool, optional
            If True and reextract_union=True, return tuple (default: False).
        device : str or torch.device, optional
            Device for computation (default: None to auto-select).
        verbose : bool, optional
            Print progress (default: False).

        Returns
        -------
        pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]
            DataFrame with columns: from_idx, to_idx, from_gene, to_gene,
            attention, model, and optionally attention_rank.
        """
        gene_ids = target_ids if target_ids is not None else self.common_gene_ids

        if verbose:
            logger.info(
                f"Computing consensus attention across {self.n_embeddings} embeddings "
                f"for {len(gene_ids)} genes..."
            )

        # Phase 1: Compute consensus for all embeddings
        all_consensus = self.get_consensus_attentions(
            target_ids=target_ids,
            consensus_method=consensus_method,
            apply_softmax=apply_softmax,
            device=device,
        )

        # Extract top-k from each embedding's consensus
        top_edges_list = []
        keys = list(self.attended_embeddings.keys())

        for i, key in enumerate(keys):
            ae = self.attended_embeddings[key]
            if verbose:
                logger.info(f"Extracting top-{k} edges from {key}...")

            model_top_k = _find_top_k_edges_in_attention_layer(
                attention=all_consensus[i],
                k=k,
                layer_idx=None,
                gene_ids=gene_ids,
                by_absolute_value=by_absolute_value,
                ignore_self_attention=ignore_self_attention,
            )
            model_top_k[FM_EDGELIST.MODEL] = ae.model_name

            top_edges_list.append(model_top_k)

        all_top_edges = pd.concat(top_edges_list, ignore_index=True)

        if verbose:
            logger.info(
                f"Extracted {len(all_top_edges)} total edges "
                f"({k} per embedding × {self.n_embeddings} embeddings)"
            )

        if not reextract_union:
            if return_original_and_reextracted:
                logger.warning(
                    "return_original_and_reextracted=True but reextract_union=False, "
                    "returning original top-k edges only"
                )
            return all_top_edges

        # Phase 2: Union re-extraction
        unique_edges = all_top_edges[
            [FM_EDGELIST.FROM_GENE, FM_EDGELIST.TO_GENE]
        ].drop_duplicates()

        if verbose:
            logger.info(
                f"Re-extracting {len(unique_edges)} unique edges from all embeddings..."
            )

        edge_df = _edgelist_to_indices(
            edge_list=unique_edges,
            gene_ids=gene_ids,
            verbose=verbose,
        )

        # Reuse consensus tensors from Phase 1
        attention_tensors = [all_consensus[i] for i in range(self.n_embeddings)]
        metadata = [
            {FM_EDGELIST.MODEL: self.attended_embeddings[key].model_name}
            for key in keys
        ]

        reextracted_union = _extract_edges_from_attention_tensors(
            edge_df=edge_df,
            attention_tensors=attention_tensors,
            metadata=metadata,
            compute_ranks=compute_ranks,
            by_absolute_value=by_absolute_value,
            device=device,
            verbose=verbose,
        )

        if verbose:
            logger.info(
                f"Extracted {len(reextracted_union)} total attention values "
                f"({len(unique_edges)} edges × {self.n_embeddings} embeddings)"
            )

        if return_original_and_reextracted:
            return all_top_edges, reextracted_union
        return reextracted_union

    def get_model_for_embedding(self, embedding_key: str) -> FoundationModel:
        if embedding_key not in self.attended_embeddings:
            raise KeyError(
                f"Embedding '{embedding_key}' not found. "
                f"Available keys: {self.embedding_keys}"
            )
        return self.attended_embeddings[embedding_key].foundation_model

    def get_specific_attentions(
        self,
        edge_list: pd.DataFrame,
        target_ids: Optional[List[str]] = None,
        apply_softmax: bool = False,
        compute_ranks: bool = False,
        by_absolute_value: bool = True,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Extract specific attention values across all embeddings and layers.

        Parameters
        ----------
        edge_list : pd.DataFrame
            DataFrame with 'from_gene' and 'to_gene' columns.
        target_ids : List[str], optional
            Subset of gene IDs. If None, uses all common genes.
        apply_softmax : bool, optional
            If True, use softmax attention (default: False).
        compute_ranks : bool, optional
            If True, add ranks to output (default: False).
        by_absolute_value : bool, optional
            If True, rank by absolute value (default: True).
        verbose : bool, optional
            Print progress (default: False).

        Returns
        -------
        pd.DataFrame
            Columns: from_gene, to_gene, model, layer, attention,
            and optionally attention_rank.
        """
        results = []

        for key, ae in self.attended_embeddings.items():
            if verbose:
                logger.info(f"Extracting attentions from {key}...")

            model_attentions = ae.get_specific_attentions(
                edge_list=edge_list,
                target_ids=target_ids,
                apply_softmax=apply_softmax,
                compute_ranks=compute_ranks,
                by_absolute_value=by_absolute_value,
                verbose=False,
            )

            model_attentions[FM_EDGELIST.MODEL] = ae.model_name

            results.append(model_attentions)

        all_attentions = pd.concat(results, ignore_index=True)

        if verbose:
            n_edges = len(
                edge_list[
                    [FM_EDGELIST.FROM_GENE, FM_EDGELIST.TO_GENE]
                ].drop_duplicates()
            )
            logger.info(
                f"Extracted {len(all_attentions)} total attention values "
                f"({n_edges} edges × {self.n_embeddings} embeddings)"
            )

        return all_attentions

    def get_top_attentions(
        self,
        k: int = 10000,
        target_ids: Optional[List[str]] = None,
        by_absolute_value: bool = True,
        reextract_union: bool = False,
        apply_softmax: bool = False,
        compute_ranks: bool = False,
        ignore_self_attention: bool = False,
        return_original_and_reextracted: bool = False,
        verbose: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Extract top-k attention edges across all embeddings.

        For each embedding, identifies the k strongest attention relationships
        per layer. Enables cross-embedding comparison of attention patterns.

        Parameters
        ----------
        k : int, optional
            Number of top edges per layer per embedding (default: 10000).
        target_ids : List[str], optional
            Subset of gene IDs. If None, uses all common genes.
        by_absolute_value : bool, optional
            If True, rank by absolute value (default: True).
        reextract_union : bool, optional
            If True, re-extract union of top edges from all embeddings (default: False).
        apply_softmax : bool, optional
            If True, use softmax attention (default: False).
        compute_ranks : bool, optional
            If True, add ranks to output (default: False).
        ignore_self_attention : bool, optional
            If True, exclude self-attention edges (default: False).
        return_original_and_reextracted : bool, optional
            If True and reextract_union=True, return tuple (default: False).
        verbose : bool, optional
            Print progress (default: False).

        Returns
        -------
        pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]
            DataFrame with columns: layer, from_idx, to_idx, from_gene,
            to_gene, attention, model, and optionally attention_rank.
        """
        top_attention_edges = []

        for key, ae in self.attended_embeddings.items():
            logger.info(f"Computing top-k attention for {key}...")

            model_top_k = ae.get_top_attentions(
                k=k,
                target_ids=target_ids,
                apply_softmax=apply_softmax,
                by_absolute_value=by_absolute_value,
                compute_ranks=compute_ranks,
                ignore_self_attention=ignore_self_attention,
                verbose=verbose,
            ).assign(**{FM_EDGELIST.MODEL: ae.model_name})

            top_attention_edges.append(model_top_k)

        all_top_edges = pd.concat(top_attention_edges, ignore_index=True)

        if reextract_union:
            logger.info("Re-extracting top edges from every embedding and layer...")

            reextracted = self.get_specific_attentions(
                all_top_edges,
                target_ids=target_ids,
                apply_softmax=apply_softmax,
                compute_ranks=compute_ranks,
                by_absolute_value=by_absolute_value,
                verbose=verbose,
            )

            if return_original_and_reextracted:
                return all_top_edges, reextracted
            return reextracted

        if return_original_and_reextracted:
            logger.warning(
                "return_original_and_reextracted=True but reextract_union=False, "
                "returning original top-k edges only"
            )
        return all_top_edges

    def get_within_embeddings_layer_comparisons(
        self,
        top_k: int,
        ignore_self_attention: bool = True,
        by_absolute_value: bool = False,
        verbose: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get within model layer x layer comparisons.

        For each of the embeddings in the set calculate cross-layer attention consistency.

        Parameters
        ----------
        top_k : int
            The number of attention pairs to summarize for "top" summaries.
        ignore_self_attention : bool
            Should self-attention (i.e., gene A - gene A) be ignored when summarizing top attention pairs. Default is True.
        by_absolute_value : bool
            Should the absolute value of the attention be used when summarizing top attention pairs. If False (default) then top-attention is selected from the most positive attention values. Default is False.
        verbose : bool
            Should verbose output be printed. Default is False.

        Returns
        -------
        tuple
            A tuple of two dictionaries. The first contains the model x layer correlations and the second contains the model x layer rank agreement.

        """

        model_layer_correlations = {}
        model_layer_rank_agreement = {}

        for an_attended_embeddings in self.attended_embeddings.values():

            model_name = an_attended_embeddings.model_name
            logger.info(
                f"Summarizing cross-layer attention consistency for {model_name}..."
            )

            (
                model_layer_correlations[model_name],
                model_layer_rank_agreement[model_name],
            ) = an_attended_embeddings.compare_layer_attention_consistency(
                top_k=top_k,
                ignore_self_attention=ignore_self_attention,
                by_absolute_value=by_absolute_value,
                verbose=verbose,
            )

        return model_layer_correlations, model_layer_rank_agreement

    def __getitem__(self, key: str) -> LayerwiseAttentionInputs:
        if key not in self.attended_embeddings:
            raise KeyError(
                f"Embedding '{key}' not found. "
                f"Available keys: {self.embedding_keys}"
            )
        return self.attended_embeddings[key]

    def __len__(self) -> int:
        return self.n_embeddings

    def __repr__(self) -> str:
        return (
            f"AttentionPatternsInputs("
            f"n_embeddings={self.n_embeddings}, "
            f"n_common_genes={self.n_common_genes}, "
            f"models={self.model_names}, "
            f"keys={self.embedding_keys}"
            f")"
        )


# Public functions


def aggregate_embedding_comparisons_over_categories(
    embedding_comparisons: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Combine a dictionary of foundation model comparison dicts into a single dict of summaries.

    For matrix-based summaries, take the median across categories.
    For DataFrame-based summaries, concatenate the DataFrames across categories and add the category as a column.
    Each comparison field is either present (dict/DataFrame) or None for a category; only whole-field None is checked.
    If all categories have None for a field, that field is omitted.

    Parameters
    ----------
    embedding_comparisons : dict
        A dictionary of embedding comparison dicts (e.g. from AttentionPatternsInputs.compare() per category).

    Returns
    -------
    dict
        A dictionary of aggregated embedding comparisons (no SETTINGS roll-up; callers have per-category settings).
    """
    comparisons = dict()
    categories = list(embedding_comparisons.keys())
    if not categories:
        return comparisons

    def _non_none(key: str):
        return [
            embedding_comparisons[cat][key]
            for cat in categories
            if embedding_comparisons[cat].get(key) is not None
        ]

    def _concat_categories(key: str):
        parts = [
            embedding_comparisons[cat][key].assign(category=cat)
            for cat in categories
            if embedding_comparisons[cat].get(key) is not None
        ]
        if not parts:
            return None
        return pd.concat(parts, ignore_index=True)

    # gene_embedding_correlations: median over categories (structure from first non-None)
    vals = _non_none(COMPARE_EMBEDDINGS_COMPARISONS.RESIDUAL_STREAM_CORRELATIONS)
    if vals:
        comparisons[COMPARE_EMBEDDINGS_COMPARISONS.RESIDUAL_STREAM_CORRELATIONS] = {
            key: np.median([cat[key] for cat in vals]) for key in vals[0]
        }
    else:
        logger.debug(
            "Omitting %s: all categories had None",
            COMPARE_EMBEDDINGS_COMPARISONS.RESIDUAL_STREAM_CORRELATIONS,
        )

    # model_layer_correlations: median over categories per model (structure from first non-None)
    vals = _non_none(COMPARE_EMBEDDINGS_COMPARISONS.MODEL_LAYER_CORRELATIONS)
    if vals:
        comparisons[COMPARE_EMBEDDINGS_COMPARISONS.MODEL_LAYER_CORRELATIONS] = {
            model: np.median([v[model] for v in vals], axis=0) for model in vals[0]
        }
    else:
        logger.debug(
            "Omitting %s: all categories had None",
            COMPARE_EMBEDDINGS_COMPARISONS.MODEL_LAYER_CORRELATIONS,
        )

    # model_layer_rank_agreement: concat DataFrames per model (structure from first non-None)
    vals = _non_none(COMPARE_EMBEDDINGS_COMPARISONS.MODEL_LAYER_RANK_AGREEMENT)
    if vals:
        key_mlra = COMPARE_EMBEDDINGS_COMPARISONS.MODEL_LAYER_RANK_AGREEMENT
        comparisons[key_mlra] = {
            model: pd.concat(
                [
                    embedding_comparisons[cat][key_mlra][model].assign(category=cat)
                    for cat in categories
                    if embedding_comparisons[cat].get(key_mlra) is not None
                ],
                ignore_index=True,
            )
            for model in vals[0]
        }
    else:
        logger.debug(
            "Omitting %s: all categories had None",
            COMPARE_EMBEDDINGS_COMPARISONS.MODEL_LAYER_RANK_AGREEMENT,
        )

    # DataFrame fields
    for key in (
        COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_X_LAYER_TOP_ATTENTIONS,
        COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_X_LAYER_RANK_AGREEMENT,
        COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_CONSENSUS_TOP_ATTENTIONS,
        COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_CONSENSUS_TOP_ATTENTIONS_RANK_AGREEMENT,
    ):
        result = _concat_categories(key)
        if result is not None:
            comparisons[key] = result
        else:
            logger.debug("Omitting %s: all categories had None", key)

    return comparisons


def validate_embedding_comparisons_settings(
    embedding_comparisons: Dict[str, Any],
    top_k: int,
    consensus_method: str,
    by_absolute_value: bool,
    ignore_self_attention: bool,
    embeddings_keys: Optional[List[str]] = None,
) -> None:
    """
    Validate embedding comparisons to ensure that their recorded settings agree with the provided settings.

    Parameters
    ----------
    embedding_comparisons : Dict[str, Any]
        The comparisons to validate. Created by AttentionPatternsInputs.compare().
    top_k : int
        The number of top-k attention pairs to summarize.
    consensus_method : str
        The consensus method to use. Valid options are: {VALID_FM_LAYER_CONSENSUS_METHODS}.
    by_absolute_value : bool
        Whether to use the absolute value of the attention.
    ignore_self_attention : bool
        Whether to ignore self-attention.
    embeddings_keys : Optional[List[str]]
        The embeddings keys to validate. If None, all embeddings keys will be validated.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the comparisons do not match the provided settings.
    """

    # check compatibility between the loaded results and the current settings
    if (
        top_k
        != embedding_comparisons[COMPARE_EMBEDDINGS_COMPARISONS.SETTINGS][
            COMPARE_EMBEDDINGS_SETTINGS.TOP_K
        ]
    ):
        logger.warning(
            f"TOP_K mismatch: {top_k} != {embedding_comparisons[COMPARE_EMBEDDINGS_COMPARISONS.SETTINGS][COMPARE_EMBEDDINGS_SETTINGS.TOP_K]}"
        )
    if (
        consensus_method
        != embedding_comparisons[COMPARE_EMBEDDINGS_COMPARISONS.SETTINGS][
            COMPARE_EMBEDDINGS_SETTINGS.CONSENSUS_METHOD
        ]
    ):
        logger.warning(
            f"CONSENSUS_METHOD mismatch: {consensus_method} != {embedding_comparisons[COMPARE_EMBEDDINGS_COMPARISONS.SETTINGS][COMPARE_EMBEDDINGS_SETTINGS.CONSENSUS_METHOD]}"
        )
    if (
        by_absolute_value
        != embedding_comparisons[COMPARE_EMBEDDINGS_COMPARISONS.SETTINGS][
            COMPARE_EMBEDDINGS_SETTINGS.BY_ABSOLUTE_VALUE
        ]
    ):
        logger.warning(
            f"BY_ABSOLUTE_VALUE mismatch: {by_absolute_value} != {embedding_comparisons[COMPARE_EMBEDDINGS_COMPARISONS.SETTINGS][COMPARE_EMBEDDINGS_SETTINGS.BY_ABSOLUTE_VALUE]}"
        )
    if (
        ignore_self_attention
        != embedding_comparisons[COMPARE_EMBEDDINGS_COMPARISONS.SETTINGS][
            COMPARE_EMBEDDINGS_SETTINGS.IGNORE_SELF_ATTENTION
        ]
    ):
        logger.warning(
            f"IGNORE_SELF_ATTENTION mismatch: {ignore_self_attention} != {embedding_comparisons[COMPARE_EMBEDDINGS_COMPARISONS.SETTINGS][COMPARE_EMBEDDINGS_SETTINGS.IGNORE_SELF_ATTENTION]}"
        )
    if embeddings_keys is not None:
        if set(embeddings_keys) != set(
            embedding_comparisons[COMPARE_EMBEDDINGS_COMPARISONS.SETTINGS][
                COMPARE_EMBEDDINGS_SETTINGS.EMBEDDING_KEYS
            ]
        ):
            extra_models = set(
                embedding_comparisons[COMPARE_EMBEDDINGS_COMPARISONS.SETTINGS][
                    COMPARE_EMBEDDINGS_SETTINGS.EMBEDDING_KEYS
                ]
            ) - set(embeddings_keys)
            missing_models = set(embeddings_keys) - set(
                embedding_comparisons[COMPARE_EMBEDDINGS_COMPARISONS.SETTINGS][
                    COMPARE_EMBEDDINGS_SETTINGS.EMBEDDING_KEYS
                ]
            )
            logger.warning(f"Extra models: {extra_models}")
            logger.warning(f"Missing models: {missing_models}")


def _edgelist_to_indices(
    edge_list: pd.DataFrame,
    gene_ids: List[str],
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Convert edge list with gene identifiers to indices.

    Parameters
    ----------
    edge_list : pd.DataFrame
        DataFrame with 'from_gene' and 'to_gene' columns
    gene_ids : List[str]
        Ordered list of gene identifiers (defines index mapping)
    verbose : bool, optional
        Whether to print warnings about filtered edges

    Returns
    -------
    pd.DataFrame
        DataFrame with 'from_gene', 'to_gene', 'from_idx', 'to_idx' columns,
        filtered to only edges where both genes are in gene_ids
    """
    # Validate input
    required_cols = [FM_EDGELIST.FROM_GENE, FM_EDGELIST.TO_GENE]
    missing = [col for col in required_cols if col not in edge_list.columns]
    if missing:
        raise ValueError(f"edge_list must contain columns: {missing}")

    # Get unique edges
    unique_edges = edge_list[required_cols].drop_duplicates()
    n_edges = len(unique_edges)

    # Filter edges to only those in gene_ids
    edges_in_common = unique_edges[
        unique_edges[FM_EDGELIST.FROM_GENE].isin(gene_ids)
        & unique_edges[FM_EDGELIST.TO_GENE].isin(gene_ids)
    ].copy()

    if len(edges_in_common) < n_edges and verbose:
        logger.warning(
            f"Filtered from {n_edges} to {len(edges_in_common)} edges "
            f"(some genes not in gene_ids)"
        )

    # Create index mappings
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_ids)}
    edges_in_common[FM_EDGELIST.FROM_IDX] = edges_in_common[FM_EDGELIST.FROM_GENE].map(
        gene_to_idx
    )
    edges_in_common[FM_EDGELIST.TO_IDX] = edges_in_common[FM_EDGELIST.TO_GENE].map(
        gene_to_idx
    )

    return edges_in_common


def _extract_edges_from_attention_tensor(
    edge_df: pd.DataFrame,
    attention: Tensor,
    from_idx_tensor: Tensor,
    to_idx_tensor: Tensor,
    metadata: Optional[Dict[str, Any]] = None,
    rank_tensor: Optional[Tensor] = None,
) -> pd.DataFrame:
    """
    Extract specific edge values from a single attention tensor.

    Core utility for extracting edge attention values from an attention matrix
    using pre-computed index tensors. This function performs the actual GPU
    extraction and DataFrame construction.

    Parameters
    ----------
    edge_df : pd.DataFrame
        Must contain 'from_gene' and 'to_gene' columns.
        Typically output from _edgelist_to_indices().
    attention : Tensor
        Attention matrix of shape (n_genes, n_genes) on device.
    from_idx_tensor : Tensor
        Pre-computed source indices tensor on device.
    to_idx_tensor : Tensor
        Pre-computed target indices tensor on device.
    metadata : Dict[str, Any], optional
        Metadata dict whose keys become DataFrame columns (default: None).
        Example: {'model': 'scGPT', 'layer': 0}
    rank_tensor : Tensor, optional
        Pre-computed rank tensor from compute_tensor_ranks().
        If provided, adds 'attention_rank' column with integer ranks (rank 1 = highest).
        Uses fast O(n_edges) indexing (default: None)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - from_gene : str
        - to_gene : str
        - attention : float
        - attention_rank : int (if rank_tensor is provided)
            Integer rank compared to all attention values (rank 1 = highest)
        - <metadata columns> : any keys from metadata dict

    Examples
    --------
    >>> # Extract edges from a single attention tensor
    >>> from_idx = torch.from_numpy(edge_df['from_idx'].values).long().to(device)
    >>> to_idx = torch.from_numpy(edge_df['to_idx'].values).long().to(device)
    >>> result = _extract_edges_from_attention_tensor(
    ...     edge_df, attention, from_idx, to_idx, {'layer': 0}
    ... )
    >>>
    >>> # Extract with ranks: pre-compute rank tensor once, then use for multiple extractions
    >>> rank_tensor = compute_tensor_ranks(attention)
    >>> result = _extract_edges_from_attention_tensor(
    ...     edge_df, attention, from_idx, to_idx, {'layer': 0},
    ...     rank_tensor=rank_tensor
    ... )
    >>> # Filter to top 100 strongest relationships
    >>> top_100 = result[result['attention_rank'] <= 100]
    """
    # Extract edges ON GPU using pre-computed indices
    edge_attentions = attention[from_idx_tensor, to_idx_tensor]

    # Move only the extracted values to CPU
    result_df = edge_df[[FM_EDGELIST.FROM_GENE, FM_EDGELIST.TO_GENE]].copy()
    result_df[FM_EDGELIST.ATTENTION] = edge_attentions.cpu().numpy()

    # Compute ranks if rank_tensor is provided
    if rank_tensor is not None:
        ranks = rank_tensor[from_idx_tensor, to_idx_tensor].cpu().numpy()
        result_df[FM_EDGELIST.ATTENTION_RANK] = ranks

    # Add metadata columns
    if metadata:
        for key, value in metadata.items():
            result_df[key] = value

    return result_df


def _extract_edges_from_attention_tensors(
    edge_df: pd.DataFrame,
    attention_tensors: List[Tensor],
    metadata: List[Dict[str, Any]],
    delete_as_processed: bool = False,
    compute_ranks: bool = False,
    by_absolute_value: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Extract specific edge values from multiple attention tensors.

    Generic utility for extracting edges from a sequence of attention matrices,
    whether they represent different layers, models, or other dimensions.

    Parameters
    ----------
    edge_df : pd.DataFrame
        Must contain 'from_idx', 'to_idx', 'from_gene', 'to_gene' columns.
        Typically output from _edgelist_to_indices().
    attention_tensors : List[Tensor]
        List of attention matrices, each of shape (n_genes, n_genes).
        All tensors must have the same shape.
    metadata : List[Dict[str, Any]]
        Metadata for each tensor. Must have same length as attention_tensors.
        Each dict's keys become DataFrame columns.
        Example: [{'model': 'scGPT', 'layer': 0}, ...]
    delete_as_processed : bool, optional
        If True, delete each attention tensor from the list immediately after processing
        to free GPU memory. Useful when processing many large tensors (default: True)
    compute_ranks : bool, optional
        If True, compute ranks of attention values and add them to the output table.
    by_absolute_value : bool, optional
        If True, rank by absolute value when calculating ranks (default: True).
        Only used if compute_ranks=True.
    device : str or torch.device, optional
        Device for computation (default: None to auto-select)
    verbose : bool, optional
        Print progress information (default: False)

    Returns
    -------
    pd.DataFrame
        Combined results with columns:
        - from_gene : str
        - to_gene : str
        - attention : float
        - <metadata columns> : any keys from metadata dicts

    Raises
    ------
    ValueError
        If attention_tensors and metadata have different lengths

    Examples
    --------
    >>> # Extract from multiple layers
    >>> attention_tensors = [layer0_attn, layer1_attn, layer2_attn]
    >>> metadata = [{'layer': 0}, {'layer': 1}, {'layer': 2}]
    >>> results = _extract_edges_from_attention_tensors(
    ...     edge_df, attention_tensors, metadata
    ... )

    >>> # Extract from multiple models
    >>> attention_tensors = [model1_consensus, model2_consensus]
    >>> metadata = [{'model': 'scGPT'}, {'model': 'scPRINT'}]
    >>> results = _extract_edges_from_attention_tensors(
    ...     edge_df, attention_tensors, metadata
    ... )
    """
    if len(attention_tensors) != len(metadata):
        raise ValueError(
            f"attention_tensors and metadata must have same length, "
            f"got {len(attention_tensors)} and {len(metadata)}"
        )

    device = ensure_device(device, allow_autoselect=True)
    results = []

    with memory_manager(device):
        # Create index tensors ONCE - stays on device for all iterations
        from_idx_tensor = (
            torch.from_numpy(edge_df[FM_EDGELIST.FROM_IDX].values).long().to(device)
        )
        to_idx_tensor = (
            torch.from_numpy(edge_df[FM_EDGELIST.TO_IDX].values).long().to(device)
        )

        # Iterate backwards to allow safe deletion from list while iterating
        # When delete_as_processed=False, this is equivalent to forward iteration
        for i in range(len(attention_tensors) - 1, -1, -1):
            attention = attention_tensors[i].to(device).clone()
            meta = metadata[i]

            if verbose:
                meta_str = ", ".join(f"{k}={v}" for k, v in meta.items())
                logger.info(
                    f"Extracting edges from tensor {len(attention_tensors) - i}/{len(attention_tensors)} "
                    f"({meta_str})..."
                )

            rank_tensor = None
            if compute_ranks:
                rank_tensor = compute_tensor_ranks(
                    attention, by_absolute_value=by_absolute_value
                )

            if delete_as_processed:
                if verbose:
                    logger.info(f"Deleting attention tensor {i} from list...")
                del attention_tensors[i]

            # Extract edges using utility function
            result_df = _extract_edges_from_attention_tensor(
                edge_df=edge_df,
                attention=attention,
                from_idx_tensor=from_idx_tensor,
                to_idx_tensor=to_idx_tensor,
                metadata=meta,
                rank_tensor=rank_tensor,
            )

            # Prepend to maintain order (since we're iterating backwards)
            results.insert(0, result_df)

            # Cleanup
            cleanup_tensors(attention)

    return pd.concat(results, ignore_index=True)


def _find_top_k_edges_in_attention_layer(
    attention: Tensor,
    k: int,
    layer_idx: Optional[int] = None,
    gene_ids: Optional[List[str]] = None,
    by_absolute_value: bool = True,
    ignore_self_attention: bool = False,
) -> pd.DataFrame:
    """
    Extract top-k edges from an attention matrix.

    Identifies the k gene pairs with highest attention values (by absolute value
    or raw value depending on by_absolute_value parameter) and returns them as a
    DataFrame with gene indices and identifiers.

    Parameters
    ----------
    attention : torch.Tensor
        Attention matrix of shape (n_genes, n_genes)
    k : int
        Number of top edges to extract
    layer_idx : int, optional
        Layer index to include in output (default: None)
    gene_ids : List[str], optional
        Gene identifiers corresponding to attention matrix rows/cols.
        If provided, includes 'from_gene' and 'to_gene' columns in output.
        (default: None)
    by_absolute_value : bool, optional
        If True, rank edges by absolute attention value (default: True).
        If False, rank edges by raw attention value.
    ignore_self_attention : bool, optional
        If True, exclude self-attention edges (where from_gene == to_gene) from
        top-k selection (default: False).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - layer : int (if layer_idx provided)
            Layer index
        - from_idx : int
            Source gene index
        - to_idx : int
            Target gene index
        - from_gene : str (if gene_ids provided)
            Source gene identifier
        - to_gene : str (if gene_ids provided)
            Target gene identifier
        - attention : float
            Attention value (preserves sign)
        Sorted by descending absolute attention value (if by_absolute_value=True)
        or descending raw attention value (if by_absolute_value=False).

    Examples
    --------
    >>> # Basic usage
    >>> attention = model.compute_reordered_attention(0, common_genes, return_tensor=True)
    >>> top_edges = find_top_k_edges(attention, k=1000, layer_idx=0, gene_ids=common_genes)
    >>>
    >>> # Without gene IDs
    >>> top_edges = find_top_k_edges(attention, k=100)
    >>>
    >>> # Rank by raw value instead of absolute value
    >>> top_edges = find_top_k_edges(attention, k=100, by_absolute_value=False)
    >>>
    >>> # Exclude self-attention edges
    >>> top_edges = find_top_k_edges(attention, k=1000, ignore_self_attention=True)
    """

    # Get device from attention tensor for memory management
    cleanup_device = attention.device

    # Extract top-k indices and values (stays on device)
    # Use memory_manager to ensure intermediate tensors are cleaned up
    with memory_manager(cleanup_device):
        from_indices, to_indices, top_values = find_top_k(
            attention,
            k=k,
            by_absolute_value=by_absolute_value,
            ignore_self_attention=ignore_self_attention,
        )

        # Build base DataFrame with indices and attention
        df = pd.DataFrame(
            {
                FM_EDGELIST.FROM_IDX: from_indices.cpu().numpy(),
                FM_EDGELIST.TO_IDX: to_indices.cpu().numpy(),
                FM_EDGELIST.ATTENTION: top_values.cpu().numpy(),
            }
        )

        # Clean up GPU tensors immediately
        del from_indices, to_indices, top_values

    # Add layer if provided
    if layer_idx is not None:
        df[FM_EDGELIST.LAYER] = layer_idx

    # Add gene IDs via merge if provided
    if gene_ids is not None:
        # Create lookup table mapping index to gene_id
        gene_lookup = pd.DataFrame({"idx": range(len(gene_ids)), "gene_id": gene_ids})

        # Merge for from_gene
        df = df.merge(
            gene_lookup.rename(
                columns={"idx": FM_EDGELIST.FROM_IDX, "gene_id": FM_EDGELIST.FROM_GENE}
            ),
            on=FM_EDGELIST.FROM_IDX,
            how="left",
        )

        # Merge for to_gene
        df = df.merge(
            gene_lookup.rename(
                columns={"idx": FM_EDGELIST.TO_IDX, "gene_id": FM_EDGELIST.TO_GENE}
            ),
            on=FM_EDGELIST.TO_IDX,
            how="left",
        )

    # Order columns
    col_order = []
    if FM_EDGELIST.LAYER in df.columns:
        col_order.append(FM_EDGELIST.LAYER)
    col_order.extend([FM_EDGELIST.FROM_IDX, FM_EDGELIST.TO_IDX])
    if FM_EDGELIST.FROM_GENE in df.columns:
        col_order.extend([FM_EDGELIST.FROM_GENE, FM_EDGELIST.TO_GENE])
    col_order.append(FM_EDGELIST.ATTENTION)

    return df[col_order]


def _get_category_layer_embeddings(
    ge_set: GeneEmbeddingsSet,
    category: str,
    model: FoundationModel,
    dataset_name: str,
) -> List[GeneEmbeddings]:
    """Extract all per-layer residual stream embeddings for one category.

    Parameters
    ----------
    ge_set : GeneEmbeddingsSet
        The full set of embeddings for a dataset (all categories × layers).
    category : str
        The category to extract (e.g., 'adipocyte (0)').
    model : FoundationModel
        The model these embeddings belong to (used to validate layer coverage).
    dataset_name : str
        Dataset name, used only for error messages.

    Returns
    -------
    List[GeneEmbeddings]
        One GeneEmbeddings per layer, ordered by layer_idx.

    Raises
    ------
    ValueError
        If the category is absent or any layer is missing.
    """
    category_embeddings = [ge for ge in ge_set.values() if ge.category == category]

    if not category_embeddings:
        available_categories = sorted(
            {ge.category for ge in ge_set.values() if ge.category is not None}
        )
        raise ValueError(
            f"Category '{category}' not found in dataset '{dataset_name}' "
            f"for model '{model.full_name}'. "
            f"Available categories: {available_categories}"
        )

    found_layers = {ge.layer_idx for ge in category_embeddings}
    expected_layers = set(range(model.n_layers))
    if found_layers != expected_layers:
        missing = sorted(expected_layers - found_layers)
        raise ValueError(
            f"Category '{category}' in dataset '{dataset_name}' "
            f"for model '{model.full_name}' is missing layers {missing}. "
            f"Expected layers {{0..{model.n_layers - 1}}}."
        )

    return sorted(category_embeddings, key=lambda ge: ge.layer_idx)


def _group_embeddings_by_model_and_category(
    embeddings_set: GeneEmbeddingsSet,
    embedding_to_model: Dict[str, str],
) -> Dict[Tuple[str, str], Dict[int, GeneEmbeddings]]:
    """Group a flat GeneEmbeddingsSet into per-(model, category) layer dicts.

    Parameters
    ----------
    embeddings_set : GeneEmbeddingsSet
        Flat set of embeddings, one per (model, category, layer) combination.
    embedding_to_model : Dict[str, str]
        Mapping from scoped key to model full_name, pre-validated in __init__.

    Returns
    -------
    Dict[Tuple[str, str], Dict[int, GeneEmbeddings]]
        Outer key: (model_full_name, category).
        Inner key: layer_idx.
        Inner value: the GeneEmbeddings for that layer.

    Raises
    ------
    ValueError
        If any embedding has layer_idx=None.
        If the model derived from the group key does not match the embedding's
        own model_name (indicates a bug in embedding_to_model).
    """
    groups: Dict[Tuple[str, str], Dict[int, GeneEmbeddings]] = {}

    for key, emb in embeddings_set.items():
        full_name = embedding_to_model[key]
        group_key = (full_name, emb.category)

        if emb.layer_idx is None:
            raise ValueError(
                f"Embedding '{key}' has layer_idx=None. "
                f"AttentionPatternsInputs requires per-layer residual streams."
            )

        # Verify the embedding's own model_name matches the group it's being
        # assigned to — catches any mismatch between scoped keys and metadata
        emb_label = _get_model_label(emb.model_name, emb.model_variant)
        if emb_label != full_name:
            raise ValueError(
                f"Embedding '{key}' has model_name '{emb_label}' but "
                f"embedding_to_model maps it to '{full_name}'. "
                f"This indicates a bug in the embedding_to_model mapping."
            )

        if group_key not in groups:
            groups[group_key] = {}

        groups[group_key][emb.layer_idx] = emb

    return groups
