"""Tests for foundation model data structures and validation."""

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from napistu_torch.load.foundation_models import (
    DatasetExpressionEmbeddings,
    ExpressionEmbeddings,
)


def test_embeddings_field_validation():
    """Test embeddings field validation: shape, type, and conversion."""
    # Valid 3D numpy array (with category_dict for multi-category)
    valid_embeddings = np.random.randn(2, 10, 32)
    expr_emb = ExpressionEmbeddings(
        embeddings=valid_embeddings,
        ordered_genes=[f"gene_{i}" for i in range(10)],
        category_dict={0: "type1", 1: "type2"},
    )
    assert expr_emb.embeddings.shape == (2, 10, 32)

    # Valid torch.Tensor
    torch_embeddings = torch.randn(2, 10, 32)
    expr_emb = ExpressionEmbeddings(
        embeddings=torch_embeddings,
        ordered_genes=[f"gene_{i}" for i in range(10)],
        category_dict={0: "type1", 1: "type2"},
    )
    assert isinstance(expr_emb.embeddings, np.ndarray)
    assert expr_emb.embeddings.shape == (2, 10, 32)

    # Numpy converted to torch before validation (single category, no ordered_genes needed)
    numpy_embeddings = np.random.randn(1, 5, 16)
    expr_emb = ExpressionEmbeddings(embeddings=numpy_embeddings)
    assert isinstance(expr_emb.embeddings, np.ndarray)
    assert expr_emb.embeddings.shape == (1, 5, 16)

    # Invalid: 2D array (validation runs in __init__ before accessing shape[0])
    with pytest.raises(ValueError) as exc_info:
        ExpressionEmbeddings(embeddings=np.random.randn(10, 32))
    assert "3-dimensional" in str(exc_info.value)
    assert "got shape (10, 32)" in str(exc_info.value)

    # Invalid: 1D array
    with pytest.raises(ValueError) as exc_info:
        ExpressionEmbeddings(embeddings=np.random.randn(32))
    assert "3-dimensional" in str(exc_info.value)

    # Invalid: 4D array
    with pytest.raises(ValueError) as exc_info:
        ExpressionEmbeddings(embeddings=np.random.randn(2, 10, 32, 1))
    assert "3-dimensional" in str(exc_info.value)


def test_model_validation_and_defaults():
    """Test model-level validation: category_dict defaults and consistency checks."""
    # Single category: defaults to {0: "category_0"}
    embeddings = np.random.randn(1, 10, 32)
    expr_emb = ExpressionEmbeddings(
        embeddings=embeddings, ordered_genes=[f"gene_{i}" for i in range(10)]
    )
    assert expr_emb.category_dict == {0: "category_0"}
    assert expr_emb.n_categories == 1
    assert expr_emb.n_genes == 10
    assert expr_emb.embed_dim == 32

    # Multi-category: defaults to {0: "category_0", 1: "category_1", ...}
    embeddings = np.random.randn(3, 10, 32)
    expr_emb = ExpressionEmbeddings(
        embeddings=embeddings, ordered_genes=[f"gene_{i}" for i in range(10)]
    )
    assert expr_emb.category_dict == {0: "category_0", 1: "category_1", 2: "category_2"}

    # Valid multi-category with correct category_dict
    category_dict = {0: "type1", 1: "type2", 2: "type3"}
    expr_emb = ExpressionEmbeddings(
        embeddings=embeddings,
        ordered_genes=[f"gene_{i}" for i in range(10)],
        category_dict=category_dict,
    )
    assert expr_emb.category_dict == category_dict

    # Invalid: category_dict missing key
    with pytest.raises(ValidationError) as exc_info:
        ExpressionEmbeddings(
            embeddings=embeddings,
            ordered_genes=[f"gene_{i}" for i in range(10)],
            category_dict={0: "type1", 1: "type2"},  # Missing key 2
        )
    assert "category_dict must have keys 0 to 2" in str(exc_info.value)

    # Invalid: category_dict extra key
    with pytest.raises(ValidationError) as exc_info:
        ExpressionEmbeddings(
            embeddings=embeddings,
            ordered_genes=[f"gene_{i}" for i in range(10)],
            category_dict={0: "type1", 1: "type2", 2: "type3", 3: "type4"},  # Extra key
        )
    assert "category_dict must have keys 0 to 2" in str(exc_info.value)

    # Invalid: ordered_genes length mismatch
    with pytest.raises(ValidationError) as exc_info:
        ExpressionEmbeddings(
            embeddings=embeddings,
            ordered_genes=[f"gene_{i}" for i in range(5)],  # Wrong length
            category_dict=category_dict,
        )
    assert "ordered_genes has 5 entries but embeddings has 10 genes" in str(
        exc_info.value
    )


def test_dataset_expression_embeddings():
    """Test DatasetExpressionEmbeddings: get, keys, values, items, dict and list init."""
    emb1 = ExpressionEmbeddings(
        embeddings=np.random.randn(2, 10, 32),
        ordered_genes=[f"gene_{i}" for i in range(10)],
        category_dict={0: "type1", 1: "type2"},
        dataset_name="efthymiou",
    )
    emb2 = ExpressionEmbeddings(
        embeddings=np.random.randn(3, 10, 32),
        ordered_genes=[f"gene_{i}" for i in range(10)],
        category_dict={0: "a", 1: "b", 2: "c"},
        dataset_name="tabula_sapiens",
    )

    # Init from dict
    container = DatasetExpressionEmbeddings({"efthymiou": emb1, "tabula_sapiens": emb2})
    assert container.get("efthymiou") is emb1
    assert container["efthymiou"] is emb1
    assert container.get("tabula_sapiens") is emb2
    assert "efthymiou" in container
    assert "missing" not in container
    assert set(container.keys()) == {"efthymiou", "tabula_sapiens"}
    assert list(container.values()) == [emb1, emb2]
    assert len(list(container.items())) == 2
    assert "DatasetExpressionEmbeddings" in repr(container)

    # Init from list (keys from dataset_name)
    container2 = DatasetExpressionEmbeddings([emb1, emb2])
    assert container2.get("efthymiou") is emb1
    assert container2.get("tabula_sapiens") is emb2

    # KeyError when not found
    with pytest.raises(KeyError) as exc_info:
        container.get("nonexistent")
    assert "nonexistent" in str(exc_info.value)
    assert "Available datasets" in str(exc_info.value)

    # Duplicate dataset_name in list raises
    emb_dup = ExpressionEmbeddings(
        embeddings=np.random.randn(1, 10, 32),
        dataset_name="efthymiou",
    )
    with pytest.raises(ValueError) as exc_info:
        DatasetExpressionEmbeddings([emb1, emb_dup])
    assert "Duplicate dataset name" in str(exc_info.value)
