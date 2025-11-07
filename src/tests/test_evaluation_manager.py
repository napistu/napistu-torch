"""Tests for evaluation_manager module."""

from pathlib import Path

from napistu_torch.evaluation.evaluation_manager import (
    _parse_checkpoint_filename,
    find_best_checkpoint,
)


def test_parse_checkpoint_filename_valid():
    """Test parsing valid checkpoint filenames."""
    # Test with string
    result = _parse_checkpoint_filename("best-epoch=120-val_auc=0.7604.ckpt")
    assert result == (120, 0.7604)

    # Test with Path object
    result = _parse_checkpoint_filename(Path("best-epoch=50-val_auc=0.9000.ckpt"))
    assert result == (50, 0.9000)


def test_parse_checkpoint_filename_invalid():
    """Test parsing invalid checkpoint filenames."""
    # Invalid filename
    assert _parse_checkpoint_filename("invalid_filename.ckpt") is None

    # Missing val_auc
    assert _parse_checkpoint_filename("epoch=120.ckpt") is None


def test_find_best_checkpoint_valid(tmp_path):
    """Test finding best checkpoint with valid files."""
    # Single file
    checkpoint_file = tmp_path / "best-epoch=10-val_auc=0.7500.ckpt"
    checkpoint_file.touch()
    result = find_best_checkpoint(tmp_path)
    assert result is not None
    assert result[0] == checkpoint_file
    assert result[1] == 0.7500

    # Multiple files - should return highest val_auc
    (tmp_path / "best-epoch=10-val_auc=0.7000.ckpt").touch()
    (tmp_path / "best-epoch=20-val_auc=0.8500.ckpt").touch()
    (tmp_path / "best-epoch=30-val_auc=0.8000.ckpt").touch()
    result = find_best_checkpoint(tmp_path)
    assert result is not None
    assert result[1] == 0.8500


def test_find_best_checkpoint_invalid(tmp_path):
    """Test finding best checkpoint with invalid scenarios."""
    # Empty directory
    assert find_best_checkpoint(tmp_path) is None

    # Files that don't match pattern
    (tmp_path / "invalid1.ckpt").touch()
    (tmp_path / "invalid2.ckpt").touch()
    assert find_best_checkpoint(tmp_path) is None
