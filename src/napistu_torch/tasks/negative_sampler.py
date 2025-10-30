import logging
from typing import Literal, Optional, Tuple

import numpy as np
import torch

from napistu_torch.tasks.constants import NEGATIVE_SAMPLING_STRATEGIES

logger = logging.getLogger(__name__)


class NegativeSampler:
    """
    Efficient negative edge sampler using vectorized collision detection.

    Uses strata-constrained sampling with fast np.isin() for collision detection.
    Inspired by PyTorch Geometric's negative_sampling implementation.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_strata: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        sampling_strategy: Literal[
            NEGATIVE_SAMPLING_STRATEGIES.UNIFORM,
            NEGATIVE_SAMPLING_STRATEGIES.DEGREE_WEIGHTED,
        ] = NEGATIVE_SAMPLING_STRATEGIES.UNIFORM,
        oversample_ratio: float = 1.2,
        max_oversample_ratio: float = 2.0,
    ):
        """
        Initialize sampler with vectorized collision detection.

        Parameters
        ----------
        edge_index : torch.Tensor
            Training edges [2, num_edges]
        edge_strata : torch.Tensor
            Strata label for each edge [num_edges]
        edge_attr : torch.Tensor, optional
            Edge attributes [num_edges, num_features]
            This is unnecessary when message passing is just on positive edges but may be useful for other tasks.
        sampling_strategy : {'uniform', 'degree_weighted'}
            How to sample nodes within each strata. Either:
            - 'uniform': Sample nodes uniformly within each strata
            - 'degree_weighted': Sample nodes according to their out- and in-degree within each strata
        oversample_ratio : float
            Initial over-sampling factor (1.2 = 20% extra to account for collisions).
            Will be adaptively increased if needed and maintained across calls.
        max_oversample_ratio : float
            Maximum over-sampling factor (caps adaptive increases)
        """
        self.num_nodes = edge_index.max().item() + 1
        self.sampling_strategy = sampling_strategy
        self.oversample_ratio = oversample_ratio
        self.max_oversample_ratio = max_oversample_ratio

        # Store edge attributes if provided
        if edge_attr is not None:
            self.edge_attr = edge_attr.cpu()
            self.has_edge_attr = True
        else:
            self.edge_attr = None
            self.has_edge_attr = False

        # Move to CPU for initialization
        edge_index_cpu = edge_index.cpu()
        edge_strata_cpu = edge_strata.cpu()

        # Build structures
        self._build_strata_structure(edge_index_cpu, edge_strata_cpu)

        if sampling_strategy == "degree_weighted":
            self._build_degree_distributions(edge_index_cpu)

        self._build_edge_hash(edge_index_cpu)

        logger.info(
            f"NegativeSampler initialized: "
            f"{len(self.strata_structure)} strata, "
            f"{self.num_nodes} nodes, "
            f"{edge_index.size(1)} edges, "
            f"strategy={sampling_strategy}"
        )

    def _build_edge_hash(self, edge_index):
        """Build sorted edge index array for vectorized collision detection."""
        self.edge_linear_idx = (edge_index[0] * self.num_nodes + edge_index[1]).numpy()
        self.edge_linear_idx.sort()
        logger.debug(f"Built edge hash with {len(self.edge_linear_idx)} edges")

    def _build_strata_structure(self, edge_index, edge_strata):
        """Extract valid (from_nodes, to_nodes) pairs for each strata."""
        self.strata_structure = {}
        unique_strata = torch.unique(edge_strata)

        for strata in unique_strata:
            strata_mask = edge_strata == strata
            strata_edges = edge_index[:, strata_mask]

            self.strata_structure[strata.item()] = {
                "from_nodes": torch.unique(strata_edges[0]),
                "to_nodes": torch.unique(strata_edges[1]),
                "count": strata_mask.sum().item(),
                "edge_indices": torch.where(strata_mask)[0],
            }

        # Strata sampling probabilities (proportional to frequency)
        total = sum(s["count"] for s in self.strata_structure.values())
        self.strata_probs = torch.tensor(
            [
                self.strata_structure[strata]["count"] / total
                for strata in sorted(self.strata_structure.keys())
            ]
        )
        self.strata = torch.tensor(sorted(self.strata_structure.keys()))

    def _build_degree_distributions(self, edge_index):
        """Build degree-weighted sampling distributions per strata."""
        # Compute OUT-degree (as source) and IN-degree (as destination)
        out_degrees = torch.bincount(edge_index[0], minlength=self.num_nodes).float()
        in_degrees = torch.bincount(edge_index[1], minlength=self.num_nodes).float()

        for structure in self.strata_structure.values():
            # Use OUT-degree for source nodes
            from_nodes = structure["from_nodes"]
            from_degrees = out_degrees[from_nodes]
            structure["from_probs"] = from_degrees / from_degrees.sum()

            # Use IN-degree for destination nodes
            to_nodes = structure["to_nodes"]
            to_degrees = in_degrees[to_nodes]  # Now using correct degrees!
            structure["to_probs"] = to_degrees / to_degrees.sum()

    def sample(
        self, num_neg: int, device: Optional[str] = None, return_edge_attr: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample negative edges with fast vectorized collision detection.

        Parameters
        ----------
        num_neg : int
            Number of negative edges to sample
        device : str or torch.device, optional
            Device to return results on. If None, returns on CPU.
        return_edge_attr : bool
            Whether to return edge attributes for the sampled edges. Default is False.

        Returns
        -------
        tuple[torch.Tensor, Optional[torch.Tensor]]
            Tuple containing:
            - Negative edges [2, num_neg]
            - Edge attributes [num_neg, num_features] if return_edge_attr is True, otherwise None
        """
        collected_src = []
        collected_dst = []
        collected_strata = []

        # Allow up to 3 attempts - if we need more, something is wrong
        max_attempts = 3

        for _ in range(1, max_attempts + 1):
            # Calculate how many we still need
            total_collected = sum(len(s) for s in collected_src)
            remaining = num_neg - total_collected

            if remaining <= 0:
                break

            # Sample and filter a batch
            valid_src, valid_dst, valid_strata = self._sample_and_filter_batch(
                remaining
            )

            if len(valid_src) > 0:
                collected_src.append(valid_src)
                collected_dst.append(valid_dst)
                collected_strata.append(valid_strata)

        # Check if we got enough samples
        total_collected = sum(len(s) for s in collected_src)
        if total_collected < num_neg:
            logger.warning(
                f"Collected {total_collected}/{num_neg} negatives "
                f"({(num_neg - total_collected)/num_neg:.1%} shortfall) "
                f"after {max_attempts} attempts"
            )

        # Concatenate and trim to exact size
        all_src = torch.cat(collected_src)[:num_neg]
        all_dst = torch.cat(collected_dst)[:num_neg]
        all_strata = torch.cat(collected_strata)[:num_neg]
        result = torch.stack([all_src, all_dst])

        # Generate edge attributes if requested
        edge_attr = None
        if return_edge_attr:
            edge_attr = self._generate_edge_attributes(all_strata)

        if device is not None:
            result = result.to(device)
            if edge_attr is not None:
                edge_attr = edge_attr.to(device)

        return result, edge_attr

    # private methods

    def _check_collisions_vectorized(
        self, src: torch.Tensor, dst: torch.Tensor
    ) -> torch.Tensor:
        """Fast vectorized collision detection using np.isin."""
        candidate_idx = src.numpy() * self.num_nodes + dst.numpy()
        is_positive = np.isin(candidate_idx, self.edge_linear_idx)
        is_self_loop = (src == dst).numpy()
        valid_mask = ~is_positive & ~is_self_loop
        return torch.from_numpy(valid_mask)

    def _generate_edge_attributes(self, sampled_strata: torch.Tensor) -> torch.Tensor:
        """
        Generate plausible edge attributes for negative samples.

        For each negative edge, sample attributes from a real edge
        in the same strata.

        Parameters
        ----------
        sampled_strata : torch.Tensor
            Strata assignment for each sampled negative edge [num_neg]

        Returns
        -------
        torch.Tensor
            Edge attributes [num_neg, num_features]
        """
        num_neg = len(sampled_strata)
        num_features = self.edge_attr.shape[1]

        # Pre-allocate result
        neg_edge_attr = torch.empty(
            (num_neg, num_features),
            dtype=self.edge_attr.dtype,
            device=self.edge_attr.device,
        )

        # For each strata, sample attributes from real edges in that strata
        for strata in self.strata:
            # Find negative samples in this strata
            mask = sampled_strata == strata
            count = mask.sum().item()

            if count == 0:
                continue

            # Get indices of real edges in this strata
            structure = self.strata_structure[strata.item()]
            edge_indices = structure["edge_indices"]

            # Randomly sample from real edges in this strata
            random_indices = torch.randint(
                0, len(edge_indices), (count,), device=self.edge_attr.device
            )
            sampled_edge_indices = edge_indices[random_indices]

            # Copy attributes
            neg_edge_attr[mask] = self.edge_attr[sampled_edge_indices]

        return neg_edge_attr

    def _sample_candidates(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample candidate edges respecting strata structure."""
        sampled_strata = self.strata[
            torch.multinomial(self.strata_probs, batch_size, replacement=True)
        ]

        src_nodes = torch.empty(batch_size, dtype=torch.long)
        dst_nodes = torch.empty(batch_size, dtype=torch.long)

        for strata in self.strata:
            mask = sampled_strata == strata
            count = mask.sum().item()

            if count == 0:
                continue

            structure = self.strata_structure[strata.item()]
            from_nodes = structure["from_nodes"]
            to_nodes = structure["to_nodes"]

            if self.sampling_strategy == NEGATIVE_SAMPLING_STRATEGIES.UNIFORM:
                src_idx = torch.randint(0, len(from_nodes), (count,))
                dst_idx = torch.randint(0, len(to_nodes), (count,))
            elif self.sampling_strategy == NEGATIVE_SAMPLING_STRATEGIES.DEGREE_WEIGHTED:
                from_probs = structure["from_probs"]
                to_probs = structure["to_probs"]
                src_idx = torch.multinomial(from_probs, count, replacement=True)
                dst_idx = torch.multinomial(to_probs, count, replacement=True)
            else:
                raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

            src_nodes[mask] = from_nodes[src_idx]
            dst_nodes[mask] = to_nodes[dst_idx]

        return src_nodes, dst_nodes, sampled_strata

    def _sample_and_filter_batch(
        self, num_needed: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of candidates and filter for valid negatives.

        Adaptively increases oversample ratio if collision rate is high.
        The increased ratio is maintained across future sample() calls.

        Parameters
        ----------
        num_needed : int
            Number of valid negatives still needed

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Valid source and destination nodes
        """
        batch_size = int(num_needed * self.oversample_ratio)

        # Generate candidates
        src, dst, strata = self._sample_candidates(batch_size)

        # Filter collisions
        valid_mask = self._check_collisions_vectorized(src, dst)
        valid_src = src[valid_mask]
        valid_dst = dst[valid_mask]
        valid_strata = strata[valid_mask]

        num_valid = len(valid_src)
        collision_rate = 1 - (num_valid / batch_size)

        # If collision rate is high (>30%), increase oversample ratio
        # This adapts to graph saturation and persists across calls
        if collision_rate > 0.3 and self.oversample_ratio < self.max_oversample_ratio:
            old_ratio = self.oversample_ratio
            self.oversample_ratio = min(
                self.oversample_ratio * 1.5, self.max_oversample_ratio
            )
            logger.info(
                f"High collision rate ({collision_rate:.1%}). "
                f"Increased oversample ratio: {old_ratio:.2f} → {self.oversample_ratio:.2f} "
                f"(will persist for future calls)"
            )

        return valid_src, valid_dst, valid_strata
