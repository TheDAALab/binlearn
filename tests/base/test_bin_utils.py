"""
Comprehensive tests for simplified _bin_utils.py.
"""

import numpy as np
import pytest

from binning.base._bin_utils import (
    ensure_bin_dict,
    validate_bins,
    default_representatives,
    create_bin_masks,
)
from binning.base._constants import MISSING_VALUE, ABOVE_RANGE, BELOW_RANGE


class TestEnsureBinDict:
    """Test ensure_bin_dict function."""

    def test_ensure_bin_dict_none(self):
        """Test ensure_bin_dict with None input."""
        result = ensure_bin_dict(None)
        assert result == {}

    def test_ensure_bin_dict_dict_input(self):
        """Test ensure_bin_dict with dictionary input."""
        input_dict = {0: [1, 2, 3], "col": [0.5, 1.5, 2.5]}
        result = ensure_bin_dict(input_dict)

        assert result == {0: [1.0, 2.0, 3.0], "col": [0.5, 1.5, 2.5]}
        assert all(isinstance(v, list) for v in result.values())
        assert all(isinstance(x, float) for v in result.values() for x in v)

    def test_ensure_bin_dict_scalar(self):
        """Test ensure_bin_dict with scalar input."""
        result = ensure_bin_dict(42)
        assert result == {0: [42.0]}

    def test_ensure_bin_dict_1d_array(self):
        """Test ensure_bin_dict with 1D array."""
        arr = [1, 2, 3, 4]
        result = ensure_bin_dict(arr)
        assert result == {0: [1.0, 2.0, 3.0, 4.0]}

    def test_ensure_bin_dict_2d_array(self):
        """Test ensure_bin_dict with 2D array."""
        arr = [[1, 2, 3], [4, 5, 6]]
        result = ensure_bin_dict(arr)
        assert result == {0: [1.0, 2.0, 3.0], 1: [4.0, 5.0, 6.0]}

    def test_ensure_bin_dict_numpy_arrays(self):
        """Test ensure_bin_dict with numpy arrays."""
        # 1D numpy array
        arr_1d = np.array([1, 2, 3])
        result = ensure_bin_dict(arr_1d)
        assert result == {0: [1.0, 2.0, 3.0]}

        # 2D numpy array
        arr_2d = np.array([[1, 2], [3, 4]])
        result = ensure_bin_dict(arr_2d)
        assert result == {0: [1.0, 2.0], 1: [3.0, 4.0]}

    def test_ensure_bin_dict_mixed_types(self):
        """Test ensure_bin_dict with mixed data types."""
        input_dict = {0: [1, 2.5, 3], "col": np.array([4, 5, 6])}
        result = ensure_bin_dict(input_dict)

        assert result == {0: [1.0, 2.5, 3.0], "col": [4.0, 5.0, 6.0]}


class TestValidateBins:
    """Test validate_bins function."""

    def test_validate_bins_valid(self):
        """Test validate_bins with valid inputs."""
        bin_spec = {0: [0, 1, 2, 3], "col": [0.0, 0.5, 1.0]}
        bin_reps = {0: [0.5, 1.5, 2.5], "col": [0.25, 0.75]}

        # Should not raise any exception
        validate_bins(bin_spec, bin_reps)

    def test_validate_bins_too_few_edges(self):
        """Test validate_bins with too few bin edges."""
        bin_spec = {0: [1]}  # Only one edge
        bin_reps = {}

        with pytest.raises(ValueError, match="Column 0 needs at least 2 bin edges"):
            validate_bins(bin_spec, bin_reps)

    def test_validate_bins_unsorted_edges(self):
        """Test validate_bins with unsorted bin edges."""
        bin_spec = {0: [1, 3, 2, 4]}  # Not sorted
        bin_reps = {}

        with pytest.raises(ValueError, match="Bin edges for column 0 must be non-decreasing"):
            validate_bins(bin_spec, bin_reps)

    def test_validate_bins_mismatch_representatives(self):
        """Test validate_bins with mismatched representatives."""
        bin_spec = {0: [0, 1, 2, 3]}  # 3 bins
        bin_reps = {0: [0.5, 1.5]}  # Only 2 representatives

        with pytest.raises(ValueError, match="Column 0: 2 representatives for 3 bins"):
            validate_bins(bin_spec, bin_reps)

    def test_validate_bins_equal_edges_allowed(self):
        """Test validate_bins allows equal consecutive edges."""
        bin_spec = {0: [0, 1, 1, 2]}  # Equal edges allowed
        bin_reps = {0: [0.5, 1.0, 1.5]}

        # Should not raise exception
        validate_bins(bin_spec, bin_reps)

    def test_validate_bins_missing_representatives(self):
        """Test validate_bins when some columns don't have representatives."""
        bin_spec = {0: [0, 1, 2], 1: [0, 0.5, 1]}
        bin_reps = {0: [0.5, 1.5]}  # Missing representatives for column 1

        # Should not raise exception (representatives are optional)
        validate_bins(bin_spec, bin_reps)


class TestDefaultRepresentatives:
    """Test default_representatives function."""

    def test_default_representatives_finite_edges(self):
        """Test default_representatives with finite edges."""
        edges = [0, 1, 2, 3]
        result = default_representatives(edges)
        assert result == [0.5, 1.5, 2.5]

    def test_default_representatives_infinite_left(self):
        """Test default_representatives with infinite left edge."""
        edges = [-np.inf, 0, 1]
        result = default_representatives(edges)
        assert result == [-1.0, 0.5]  # 0 - 1 = -1 for left infinite

    def test_default_representatives_infinite_right(self):
        """Test default_representatives with infinite right edge."""
        edges = [0, 1, np.inf]
        result = default_representatives(edges)
        assert result == [0.5, 2.0]  # 1 + 1 = 2 for right infinite

    def test_default_representatives_both_infinite(self):
        """Test default_representatives with both edges infinite."""
        edges = [-np.inf, np.inf]
        result = default_representatives(edges)
        assert result == [0.0]  # Default to 0 when both are infinite

    def test_default_representatives_mixed_infinite(self):
        """Test default_representatives with mixed finite and infinite edges."""
        edges = [-np.inf, 0, 1, np.inf]
        result = default_representatives(edges)
        assert result == [-1.0, 0.5, 2.0]

    def test_default_representatives_single_finite_bin(self):
        """Test default_representatives with single finite bin."""
        edges = [2, 5]
        result = default_representatives(edges)
        assert result == [3.5]  # (2 + 5) / 2


class TestCreateBinMasks:
    """Test create_bin_masks function."""

    def test_create_bin_masks_valid_indices(self):
        """Test create_bin_masks with valid bin indices."""
        bin_indices = np.array([0, 1, 2, 0, 1])
        n_bins = 3

        valid, nan_mask, below_mask, above_mask = create_bin_masks(bin_indices, n_bins)

        assert np.array_equal(valid, [True, True, True, True, True])
        assert np.array_equal(nan_mask, [False, False, False, False, False])
        assert np.array_equal(below_mask, [False, False, False, False, False])
        assert np.array_equal(above_mask, [False, False, False, False, False])

    def test_create_bin_masks_special_values(self):
        """Test create_bin_masks with special values."""
        bin_indices = np.array([0, MISSING_VALUE, BELOW_RANGE, ABOVE_RANGE, 1])
        n_bins = 3

        valid, nan_mask, below_mask, above_mask = create_bin_masks(bin_indices, n_bins)

        assert np.array_equal(valid, [True, False, False, False, True])
        assert np.array_equal(nan_mask, [False, True, False, False, False])
        assert np.array_equal(below_mask, [False, False, True, False, False])
        assert np.array_equal(above_mask, [False, False, False, True, False])

    def test_create_bin_masks_out_of_range_positive(self):
        """Test create_bin_masks with out-of-range positive indices."""
        bin_indices = np.array([0, 1, 3, 4])  # 3 and 4 are >= n_bins
        n_bins = 3

        valid, nan_mask, below_mask, above_mask = create_bin_masks(bin_indices, n_bins)

        assert np.array_equal(valid, [True, True, False, False])
        assert np.array_equal(nan_mask, [False, False, False, False])
        assert np.array_equal(below_mask, [False, False, False, False])
        assert np.array_equal(above_mask, [False, False, False, False])

    def test_create_bin_masks_negative_indices(self):
        """Test create_bin_masks with negative indices."""
        # Assuming MISSING_VALUE = -1, BELOW_RANGE = -2, ABOVE_RANGE = -3
        from binning.base._constants import MISSING_VALUE, BELOW_RANGE, ABOVE_RANGE

        # Use different negative values that are NOT the special constants
        negative_val_1 = -10  # Some negative that's not a special value
        negative_val_2 = -20  # Some negative that's not a special value
        bin_indices = np.array([negative_val_1, negative_val_2, 0, 1])
        n_bins = 3

        valid, nan_mask, below_mask, above_mask = create_bin_masks(bin_indices, n_bins)

        assert np.array_equal(valid, [False, False, True, True])
        assert np.array_equal(nan_mask, [False, False, False, False])
        assert np.array_equal(below_mask, [False, False, False, False])
        assert np.array_equal(above_mask, [False, False, False, False])

    def test_create_bin_masks_actual_special_values(self):
        """Test create_bin_masks with actual special constant values."""
        bin_indices = np.array([MISSING_VALUE, BELOW_RANGE, ABOVE_RANGE, 0, 1])
        n_bins = 3

        valid, nan_mask, below_mask, above_mask = create_bin_masks(bin_indices, n_bins)

        assert np.array_equal(valid, [False, False, False, True, True])
        assert np.array_equal(nan_mask, [True, False, False, False, False])
        assert np.array_equal(below_mask, [False, True, False, False, False])
        assert np.array_equal(above_mask, [False, False, True, False, False])

    def test_create_bin_masks_2d_array(self):
        """Test create_bin_masks with 2D array."""
        bin_indices = np.array([[0, 1], [MISSING_VALUE, 2]])
        n_bins = 3

        valid, nan_mask, below_mask, above_mask = create_bin_masks(bin_indices, n_bins)

        expected_valid = [[True, True], [False, True]]
        expected_nan = [[False, False], [True, False]]

        assert np.array_equal(valid, expected_valid)
        assert np.array_equal(nan_mask, expected_nan)

    def test_create_bin_masks_empty_array(self):
        """Test create_bin_masks with empty array."""
        bin_indices = np.array([])
        n_bins = 3

        valid, nan_mask, below_mask, above_mask = create_bin_masks(bin_indices, n_bins)

        assert valid.shape == (0,)
        assert nan_mask.shape == (0,)
        assert below_mask.shape == (0,)
        assert above_mask.shape == (0,)

    def test_create_bin_masks_edge_case_zero_bins(self):
        """Test create_bin_masks with zero bins."""
        bin_indices = np.array([0, 1, -1])
        n_bins = 0

        valid, nan_mask, below_mask, above_mask = create_bin_masks(bin_indices, n_bins)

        assert np.array_equal(valid, [False, False, False])  # No valid bins when n_bins=0
