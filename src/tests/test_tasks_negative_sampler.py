import pytest
import torch

from napistu_torch.constants import NAPISTU_DATA
from napistu_torch.tasks.negative_sampler import NegativeSampler


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed before each test for reproducibility."""
    torch.manual_seed(42)
    yield


def test_basic_sampling():
    """Test that sampler returns correct shape and respects basic constraints"""
    # Create simple graph: 10 nodes, 2 categories
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [5, 6, 7, 8, 9, 6, 7, 8],
        ]
    )
    edge_categories = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    sampler = NegativeSampler(edge_index, edge_categories)

    # Sample negatives
    num_neg = 100
    neg_edges, _, neg_edge_attr = sampler.sample(num_neg)

    # Check shape
    assert neg_edges.shape == (2, num_neg)
    assert neg_edges.dtype == torch.long
    assert neg_edge_attr is None  # No edge_attr provided

    # Check no self-loops
    assert (neg_edges[0] == neg_edges[1]).sum() == 0


def test_no_collision_with_positives():
    """Test that sampled negatives don't overlap with positive edges"""
    # Create graph
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [5, 6, 7, 8, 9, 6],
        ]
    )
    edge_categories = torch.zeros(6, dtype=torch.long)

    sampler = NegativeSampler(edge_index, edge_categories)
    neg_edges, _, _ = sampler.sample(500)

    # Convert to sets for comparison
    pos_edges = set(tuple(e) for e in edge_index.t().tolist())
    neg_edges_set = set(tuple(e) for e in neg_edges.t().tolist())

    # Check no overlap
    overlap = pos_edges & neg_edges_set
    assert len(overlap) == 0, f"Found {len(overlap)} positive edges in negatives"


def test_category_constraints():
    """Test that negatives respect edge category structure"""
    # Create graph with distinct categories
    # Category 0: nodes 0-2 -> nodes 5-7
    # Category 1: nodes 3-4 -> nodes 8-9
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
        ]
    )
    edge_categories = torch.tensor([0, 0, 0, 1, 1])

    sampler = NegativeSampler(edge_index, edge_categories)
    neg_edges, _, neg_edge_attr = sampler.sample(200)

    # Check each negative belongs to a valid category
    for src, dst in neg_edges.t():
        src_item, dst_item = src.item(), dst.item()

        # Should be in category 0 (0-2 -> 5-7) or category 1 (3-4 -> 8-9)
        valid_cat0 = (src_item in [0, 1, 2]) and (dst_item in [5, 6, 7])
        valid_cat1 = (src_item in [3, 4]) and (dst_item in [8, 9])

        assert (
            valid_cat0 or valid_cat1
        ), f"Edge ({src_item}, {dst_item}) doesn't match any category"


def test_degree_weighted_sampling():
    """Test that degree-weighted strategy differs from uniform"""
    # Create a larger, sparser graph with clear degree differences
    # Nodes 0-4 are sources, nodes 10-19 are destinations
    # Node 0 has high out-degree (5 edges), nodes 1-4 have low out-degree (1 edge each)
    edge_index = torch.tensor(
        [
            [0, 0, 0, 0, 0, 1, 2, 3, 4],  # node 0: degree 5  # nodes 1-4: degree 1 each
            [10, 11, 12, 13, 14, 15, 16, 17, 18],
        ]
    )
    edge_categories = torch.zeros(9, dtype=torch.long)

    # Uniform sampling
    sampler_uniform = NegativeSampler(
        edge_index, edge_categories, sampling_strategy="uniform"
    )

    # Degree-weighted sampling
    sampler_degree = NegativeSampler(
        edge_index, edge_categories, sampling_strategy="degree_weighted"
    )

    # Sample large batch (graph has plenty of room: 5 sources × 10 destinations = 50 possible edges, only 9 exist)
    neg_uniform, _, _ = sampler_uniform.sample(500)
    neg_degree, _, _ = sampler_degree.sample(500)

    # Count how often node 0 appears as source
    count_uniform = (neg_uniform[0] == 0).sum().item()
    count_degree = (neg_degree[0] == 0).sum().item()

    # Node 0 has 5/9 ≈ 56% of edges, so in degree-weighted sampling it should appear
    # much more frequently than in uniform (where it's 1/5 = 20%)
    # Uniform: expect ~100 out of 500 (20%)
    # Degree-weighted: expect ~280 out of 500 (56%)
    assert count_degree > count_uniform * 1.5, (
        f"Degree-weighted should sample high-degree nodes more frequently: "
        f"uniform={count_uniform} ({count_uniform/500:.1%}), "
        f"degree={count_degree} ({count_degree/500:.1%})"
    )


def test_oversample_ratio_adaptation():
    """Test that oversample ratio adapts and persists across calls"""
    # Create a DENSE graph to force high collision rate
    # 20 nodes, but densely connected (50% of possible edges)

    num_cat0_nodes_src = 10
    num_cat0_nodes_dst = 10

    # Create dense category 0: most edges between two groups exist
    src_nodes = []
    dst_nodes = []
    for s in range(num_cat0_nodes_src):
        for d in range(num_cat0_nodes_src, num_cat0_nodes_src + num_cat0_nodes_dst):
            # Include 70% of possible edges to create high saturation
            if torch.rand(1).item() < 0.7:
                src_nodes.append(s)
                dst_nodes.append(d)

    edge_index = torch.tensor([src_nodes, dst_nodes])
    edge_categories = torch.zeros(len(src_nodes), dtype=torch.long)

    # Start with low oversample ratio
    sampler = NegativeSampler(
        edge_index,
        edge_categories,
        oversample_ratio=1.1,  # Very low to force adaptation
        max_oversample_ratio=2.0,
    )

    initial_ratio = sampler.oversample_ratio
    assert initial_ratio == 1.1

    # Sample large batch - should trigger adaptation due to high collision rate
    sampler.sample(500)
    ratio_after_first = sampler.oversample_ratio

    # Ratio MUST have increased due to dense graph
    assert (
        ratio_after_first > initial_ratio
    ), f"Dense graph should trigger adaptation: {initial_ratio} -> {ratio_after_first}"

    # Sample again - ratio should persist (not reset)
    sampler.sample(500)
    ratio_after_second = sampler.oversample_ratio

    # Should stay elevated or increase further
    assert ratio_after_second >= ratio_after_first

    # Should respect max
    assert sampler.oversample_ratio <= 2.0


def test_edge_attr_sampling(napistu_data):
    """Test that sampler can generate edge attributes for negative samples."""
    # Extract edge data from napistu_data fixture
    edge_index = napistu_data.edge_index
    edge_attr = napistu_data.edge_attr

    # Create simple edge categories (all same category for simplicity)
    edge_categories = torch.zeros(edge_index.size(1), dtype=torch.long)

    # Create sampler with edge attributes
    sampler = NegativeSampler(edge_index, edge_categories, edge_attr=edge_attr)

    # Sample negatives with edge attributes
    num_neg = 50
    neg_edges, _, neg_edge_attr = sampler.sample(num_neg, return_edge_attr=True)

    # Check basic properties
    assert neg_edges.shape == (2, num_neg)
    assert neg_edges.dtype == torch.long
    assert neg_edge_attr is not None
    assert neg_edge_attr.shape == (num_neg, edge_attr.shape[1])
    assert neg_edge_attr.dtype == edge_attr.dtype

    # Check no self-loops
    assert (neg_edges[0] == neg_edges[1]).sum() == 0

    # Check that edge attributes are reasonable (not all zeros or identical)
    assert not torch.all(neg_edge_attr == 0), "Edge attributes should not be all zeros"
    assert not torch.all(
        neg_edge_attr == neg_edge_attr[0]
    ), "Edge attributes should vary"


def test_relations_sampling(edge_prediction_with_sbo_relations):
    """Test that sampler can generate relation types for negative samples."""

    # Extract data from fixture
    data = edge_prediction_with_sbo_relations
    edge_index = data.edge_index
    relation_type = getattr(data, NAPISTU_DATA.RELATION_TYPE, None)

    # Verify relation types are available
    assert relation_type is not None, "Fixture should have relation_type"
    assert relation_type.shape[0] == edge_index.size(
        1
    ), "Relation types should match edge count"

    # Get training edges and their relation types/strata
    train_mask = data.train_mask
    train_edges = edge_index[:, train_mask]
    train_relation_type = relation_type[train_mask]

    # Use relation types as strata (they encode the same grouping information)
    # Since relation types are derived from edge_strata, we can use them as strata
    edge_strata = train_relation_type

    # Create sampler with relation types
    sampler = NegativeSampler(
        edge_index=train_edges,
        edge_strata=edge_strata,
        relation_type=train_relation_type,
    )

    # Sample negatives with relation types
    num_neg = 50
    neg_edges, neg_relation_type, _ = sampler.sample(num_neg, return_relations=True)

    # Check basic properties
    assert neg_edges.shape == (2, num_neg)
    assert neg_edges.dtype == torch.long
    assert neg_relation_type is not None
    assert neg_relation_type.shape == (num_neg,)
    assert neg_relation_type.dtype == train_relation_type.dtype

    # Check no self-loops
    assert (neg_edges[0] == neg_edges[1]).sum() == 0

    # Check that relation types are valid (within the range of training relation types)
    unique_train_relation_type = torch.unique(train_relation_type)
    assert torch.all(
        torch.isin(neg_relation_type, unique_train_relation_type)
    ), "Negative relation types should be sampled from training relation types"

    # Check that relation types vary (not all the same)
    unique_neg_relation_type = torch.unique(neg_relation_type)
    assert (
        len(unique_neg_relation_type) > 1 or num_neg == 1
    ), "Relation types should vary when sampling multiple negatives"
