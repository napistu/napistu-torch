"""Tests for NapistuData utility functions."""

from napistu_torch.constants import NAPISTU_DATA
from napistu_torch.utils.nd_utils import compute_mask_hashes


def test_compute_mask_hashes(edge_prediction_with_sbo_relations):
    """Test compute_mask_hashes with relation prediction fixture."""
    data = edge_prediction_with_sbo_relations

    hashes = compute_mask_hashes(
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
    )

    # Check all keys present and hashes are SHA256 hex strings (64 chars)
    for mask_name in [
        NAPISTU_DATA.TRAIN_MASK,
        NAPISTU_DATA.VAL_MASK,
        NAPISTU_DATA.TEST_MASK,
    ]:
        hash_key = f"{mask_name}_hash"
        assert hash_key in hashes
        assert hashes[hash_key] is not None
        assert len(hashes[hash_key]) == 64

    # Identical masks produce identical hashes
    hashes2 = compute_mask_hashes(
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
    )
    assert hashes == hashes2

    # None masks return None
    hashes_none = compute_mask_hashes()
    assert all(v is None for v in hashes_none.values())
