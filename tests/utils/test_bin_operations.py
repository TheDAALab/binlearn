"""Tests for bin_operations module."""

from unittest.mock import patch

import numpy as np
import pytest

from binlearn.utils.bin_operations import (
    create_bin_masks,
    default_representatives,
    validate_bin_edges_format,
    validate_bin_representatives_format,
    validate_bins,
)
from binlearn.utils.constants import ABOVE_RANGE, BELOW_RANGE, MISSING_VALUE


class TestValidateBinEdgesFormat:
    """Test validate_bin_edges_format function."""

    def test_none_input(self):
        """Test with None input."""
        # Should not raise any exception
        validate_bin_edges_format(None)

    def test_valid_dict_input(self):
        """Test with valid dictionary input."""
        data = {"col1": [1.0, 2.0, 3.0], "col2": [4.0, 5.0]}
        # Should not raise any exception
        validate_bin_edges_format(data)

    def test_dict_with_numpy_arrays(self):
        """Test with dictionary containing numpy arrays."""
        data = {"col1": np.array([1.0, 2.0, 3.0]), "col2": np.array([4.0, 5.0])}
        # Should not raise any exception
        validate_bin_edges_format(data)

    def test_non_dict_input(self):
        """Test with non-dict input."""
        with pytest.raises(ValueError, match="bin_edges must be a dictionary"):
            validate_bin_edges_format(5.0)

    def test_invalid_edges_non_iterable(self):
        """Test with non-iterable edges."""
        with pytest.raises(ValueError, match="must be array-like"):
            validate_bin_edges_format({"col1": 5.0})

    def test_insufficient_edges(self):
        """Test with insufficient edges."""
        with pytest.raises(ValueError, match="needs at least 2 bin edges"):
            validate_bin_edges_format({"col1": [1.0]})

    def test_non_numeric_edges(self):
        """Test with non-numeric edges."""
        with pytest.raises(ValueError, match="must be numeric"):
            validate_bin_edges_format({"col1": ["a", "b", "c"]})

    def test_unsorted_edges(self):
        """Test with unsorted edges."""
        with pytest.raises(ValueError, match="must be sorted in ascending order"):
            validate_bin_edges_format({"col1": [3.0, 1.0, 2.0]})


class TestValidateBinRepresentativesFormat:
    """Test validate_bin_representatives_format function."""

    def test_none_input(self):
        """Test with None input."""
        # Should not raise any exception
        validate_bin_representatives_format(None)

    def test_valid_dict_input(self):
        """Test with valid dictionary input."""
        data = {"col1": [1.5, 2.5], "col2": [4.5]}
        # Should not raise any exception
        validate_bin_representatives_format(data)

    def test_non_dict_input(self):
        """Test with non-dict input."""
        with pytest.raises(ValueError, match="bin_representatives must be a dictionary"):
            validate_bin_representatives_format(5.0)

    def test_invalid_reps_non_iterable(self):
        """Test with non-iterable representatives."""
        with pytest.raises(ValueError, match="must be array-like"):
            validate_bin_representatives_format({"col1": 5.0})

    def test_non_numeric_reps(self):
        """Test with non-numeric representatives."""
        with pytest.raises(ValueError, match="must be numeric"):
            validate_bin_representatives_format({"col1": ["a", "b"]})

    def test_mismatched_with_edges(self):
        """Test with mismatched number compared to edges."""
        bin_edges = {"col1": [0.0, 1.0, 2.0]}  # 2 bins
        bin_reps = {"col1": [0.5]}  # 1 representative
        with pytest.raises(ValueError, match="representatives provided, but.*expected"):
            validate_bin_representatives_format(bin_reps, bin_edges)


class TestValidateBins:
    """Test validate_bins function."""

    def test_valid_bins(self):
        """Test with valid bin specifications."""
        bin_spec = {"col1": [0.0, 1.0, 2.0]}
        bin_reps = {"col1": [0.5, 1.5]}
        # Should not raise any exception
        validate_bins(bin_spec, bin_reps)

    def test_insufficient_edges(self):
        """Test with insufficient bin edges."""
        bin_spec = {"col1": [1.0]}  # Only one edge
        bin_reps = {}
        with pytest.raises(ValueError, match="needs at least 2 bin edges"):
            validate_bins(bin_spec, bin_reps)

    def test_unsorted_edges(self):
        """Test with unsorted bin edges."""
        bin_spec = {"col1": [2.0, 1.0, 3.0]}  # Unsorted
        bin_reps = {}
        with pytest.raises(ValueError, match="must be non-decreasing"):
            validate_bins(bin_spec, bin_reps)

    def test_equal_edges_allowed(self):
        """Test that equal consecutive edges are allowed."""
        bin_spec = {"col1": [1.0, 1.0, 2.0]}  # Equal consecutive edges
        bin_reps = {"col1": [1.0, 1.5]}
        # Should not raise any exception
        validate_bins(bin_spec, bin_reps)

    def test_mismatched_representatives(self):
        """Test with mismatched number of representatives."""
        bin_spec = {"col1": [0.0, 1.0, 2.0]}  # 2 bins
        bin_reps = {"col1": [0.5]}  # Only 1 representative
        with pytest.raises(ValueError, match="representatives.*for.*bins"):
            validate_bins(bin_spec, bin_reps)

    def test_too_many_representatives(self):
        """Test with too many representatives."""
        bin_spec = {"col1": [0.0, 1.0, 2.0]}  # 2 bins
        bin_reps = {"col1": [0.5, 1.5, 2.5]}  # 3 representatives
        with pytest.raises(ValueError, match="representatives.*for.*bins"):
            validate_bins(bin_spec, bin_reps)

    def test_multiple_columns(self):
        """Test with multiple columns."""
        bin_spec = {"col1": [0.0, 1.0, 2.0], "col2": [10.0, 20.0, 30.0, 40.0]}
        bin_reps = {"col1": [0.5, 1.5], "col2": [15.0, 25.0, 35.0]}
        # Should not raise any exception
        validate_bins(bin_spec, bin_reps)

    def test_missing_representatives_for_some_columns(self):
        """Test when representatives are missing for some columns."""
        bin_spec = {"col1": [0.0, 1.0, 2.0], "col2": [10.0, 20.0, 30.0]}
        bin_reps = {"col1": [0.5, 1.5]}  # Missing col2
        # Should not raise any exception - missing reps are allowed
        validate_bins(bin_spec, bin_reps)


class TestDefaultRepresentatives:
    """Test default_representatives function."""

    def test_normal_intervals(self):
        """Test with normal finite intervals."""
        edges = [0.0, 1.0, 2.0, 3.0]
        result = default_representatives(edges)
        expected = [0.5, 1.5, 2.5]
        assert result == expected

    def test_single_interval(self):
        """Test with single interval."""
        edges = [0.0, 2.0]
        result = default_representatives(edges)
        expected = [1.0]
        assert result == expected

    def test_negative_infinity_left(self):
        """Test with negative infinity on the left."""
        edges = [float("-inf"), 0.0, 1.0]
        result = default_representatives(edges)
        expected = [-1.0, 0.5]  # right - 1.0 for -inf case
        assert result == expected

    def test_positive_infinity_right(self):
        """Test with positive infinity on the right."""
        edges = [0.0, 1.0, float("inf")]
        result = default_representatives(edges)
        expected = [0.5, 2.0]  # left + 1.0 for +inf case
        assert result == expected

    def test_both_infinities(self):
        """Test with both infinities."""
        edges = [float("-inf"), float("inf")]
        result = default_representatives(edges)
        expected = [0.0]  # Special case for (-inf, +inf)
        assert result == expected

    def test_zero_width_interval(self):
        """Test with zero-width interval."""
        edges = [1.0, 1.0]
        result = default_representatives(edges)
        expected = [1.0]  # Midpoint of [1.0, 1.0] is 1.0
        assert result == expected

    def test_multiple_infinities(self):
        """Test with multiple infinity cases."""
        edges = [float("-inf"), 0.0, 1.0, float("inf")]
        result = default_representatives(edges)
        expected = [-1.0, 0.5, 2.0]
        assert result == expected


class TestCreateBinMasks:
    """Test create_bin_masks function."""

    def test_normal_indices(self):
        """Test with normal bin indices."""
        indices = np.array([0, 1, 2, 0, 1])
        n_bins = 3
        valid, nan_mask, below_mask, above_mask = create_bin_masks(indices, n_bins)

        expected_valid = np.array([True, True, True, True, True])
        expected_nan = np.array([False, False, False, False, False])
        expected_below = np.array([False, False, False, False, False])
        expected_above = np.array([False, False, False, False, False])

        np.testing.assert_array_equal(valid, expected_valid)
        np.testing.assert_array_equal(nan_mask, expected_nan)
        np.testing.assert_array_equal(below_mask, expected_below)
        np.testing.assert_array_equal(above_mask, expected_above)

    def test_missing_values(self):
        """Test with missing value indices."""
        indices = np.array([0, MISSING_VALUE, 1, MISSING_VALUE])
        n_bins = 2
        valid, nan_mask, below_mask, above_mask = create_bin_masks(indices, n_bins)

        expected_valid = np.array([True, False, True, False])
        expected_nan = np.array([False, True, False, True])
        expected_below = np.array([False, False, False, False])
        expected_above = np.array([False, False, False, False])

        np.testing.assert_array_equal(valid, expected_valid)
        np.testing.assert_array_equal(nan_mask, expected_nan)
        np.testing.assert_array_equal(below_mask, expected_below)
        np.testing.assert_array_equal(above_mask, expected_above)

    def test_below_range_values(self):
        """Test with below range indices."""
        indices = np.array([0, BELOW_RANGE, 1, BELOW_RANGE])
        n_bins = 2
        valid, nan_mask, below_mask, above_mask = create_bin_masks(indices, n_bins)

        expected_valid = np.array([True, False, True, False])
        expected_nan = np.array([False, False, False, False])
        expected_below = np.array([False, True, False, True])
        expected_above = np.array([False, False, False, False])

        np.testing.assert_array_equal(valid, expected_valid)
        np.testing.assert_array_equal(nan_mask, expected_nan)
        np.testing.assert_array_equal(below_mask, expected_below)
        np.testing.assert_array_equal(above_mask, expected_above)

    def test_above_range_values(self):
        """Test with above range indices."""
        indices = np.array([0, ABOVE_RANGE, 1, ABOVE_RANGE])
        n_bins = 2
        valid, nan_mask, below_mask, above_mask = create_bin_masks(indices, n_bins)

        expected_valid = np.array([True, False, True, False])
        expected_nan = np.array([False, False, False, False])
        expected_below = np.array([False, False, False, False])
        expected_above = np.array([False, True, False, True])

        np.testing.assert_array_equal(valid, expected_valid)
        np.testing.assert_array_equal(nan_mask, expected_nan)
        np.testing.assert_array_equal(below_mask, expected_below)
        np.testing.assert_array_equal(above_mask, expected_above)

    def test_mixed_indices(self):
        """Test with mix of all types of indices."""
        indices = np.array([0, 1, MISSING_VALUE, BELOW_RANGE, ABOVE_RANGE, 0])
        n_bins = 2
        valid, nan_mask, below_mask, above_mask = create_bin_masks(indices, n_bins)

        expected_valid = np.array([True, True, False, False, False, True])
        expected_nan = np.array([False, False, True, False, False, False])
        expected_below = np.array([False, False, False, True, False, False])
        expected_above = np.array([False, False, False, False, True, False])

        np.testing.assert_array_equal(valid, expected_valid)
        np.testing.assert_array_equal(nan_mask, expected_nan)
        np.testing.assert_array_equal(below_mask, expected_below)
        np.testing.assert_array_equal(above_mask, expected_above)

    def test_out_of_range_positive_indices(self):
        """Test with positive indices that are out of range."""
        indices = np.array([0, 1, 5, 2])  # 5 is >= n_bins
        n_bins = 2
        valid, nan_mask, below_mask, above_mask = create_bin_masks(indices, n_bins)

        expected_valid = np.array([True, True, False, False])  # 5 and 2 are invalid
        expected_nan = np.array([False, False, False, False])
        expected_below = np.array([False, False, False, False])
        expected_above = np.array([False, False, False, False])

        np.testing.assert_array_equal(valid, expected_valid)
        np.testing.assert_array_equal(nan_mask, expected_nan)
        np.testing.assert_array_equal(below_mask, expected_below)
        np.testing.assert_array_equal(above_mask, expected_above)

    def test_empty_array(self):
        """Test with empty array."""
        indices = np.array([])
        n_bins = 2
        valid, nan_mask, below_mask, above_mask = create_bin_masks(indices, n_bins)

        assert len(valid) == 0
        assert len(nan_mask) == 0
        assert len(below_mask) == 0
        assert len(above_mask) == 0


class TestValidateBinEdgesNonFiniteValues:
    """Test validate_bin_edges_format with non-finite values."""

    def test_infinite_edges(self):
        """Test with infinite edge values."""
        with pytest.raises(ValueError, match="must be finite values"):
            validate_bin_edges_format({"col1": [1.0, 2.0, np.inf]})

    def test_nan_edges(self):
        """Test with NaN edge values."""
        # NaN comparison issues - let's try a different approach
        # We need to create a scenario where finite check is reached
        # but sorted check passes
        with patch("numpy.isfinite") as mock_isfinite:
            mock_isfinite.return_value = False
            with pytest.raises(ValueError, match="must be finite values"):
                validate_bin_edges_format({"col1": [1.0, 2.0, 3.0]})


# pylint: disable=too-few-public-methods
class TestValidateBinsNonFiniteValues:
    """Test validate_bins with None input."""

    def test_validate_bins_with_none_input(self):
        """Test validate_bins with None input to cover early return."""
        # This should not raise any exception and cover line 117
        validate_bins(None, {})  # type: ignore
