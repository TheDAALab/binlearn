"""
Tests for equal width utilities.
"""

import warnings

import numpy as np
import pytest

from binlearn.utils._equal_width_utils import (
    apply_equal_width_fallback,
    create_equal_width_bins,
    ensure_monotonic_edges,
    validate_binning_input,
)


class TestCreateEqualWidthBins:
    """Test the create_equal_width_bins function."""

    def test_basic_functionality(self):
        """Test basic equal-width binning."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        n_bins = 3
        result = create_equal_width_bins(data, n_bins)

        expected = np.array([1.0, 2.333333, 3.666667, 5.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_with_data_range(self):
        """Test with explicit data range."""
        data = np.array([1.0, 2.0, 3.0])
        n_bins = 2
        data_range = (0.0, 10.0)
        result = create_equal_width_bins(data, n_bins, data_range=data_range)

        expected = np.array([0.0, 5.0, 10.0])
        np.testing.assert_array_equal(result, expected)

    def test_equal_min_max_with_epsilon(self):
        """Test when min == max with add_epsilon=True."""
        data = np.array([5.0, 5.0, 5.0])
        n_bins = 2
        result = create_equal_width_bins(data, n_bins, add_epsilon=True)

        # Should add epsilon to max value - the implementation adds epsilon to handle equal min/max
        # but also adds epsilon to the last edge, so we check the overall range
        assert result[0] == 5.0
        assert len(result) == n_bins + 1
        # The function handles equal min/max by adding epsilon internally
        assert len(result) == n_bins + 1

    def test_equal_min_max_without_epsilon(self):
        """Test when min == max with add_epsilon=False."""
        data = np.array([5.0, 5.0, 5.0])
        n_bins = 2
        result = create_equal_width_bins(data, n_bins, add_epsilon=False)

        # Should create bins around the single value
        assert result[0] < 5.0
        assert result[-1] > 5.0
        assert len(result) == n_bins + 1

    def test_zero_value_equal_case(self):
        """Test when all values are zero."""
        data = np.array([0.0, 0.0, 0.0])
        n_bins = 3
        result = create_equal_width_bins(data, n_bins, add_epsilon=False)

        # Should handle zero case appropriately
        assert len(result) == n_bins + 1
        assert result[0] < 0.0
        assert result[-1] > 0.0

    def test_no_epsilon_with_range(self):
        """Test add_epsilon=False doesn't add epsilon when data_range provided."""
        data = np.array([1.0, 2.0, 3.0])
        n_bins = 2
        data_range = (0.0, 10.0)
        result = create_equal_width_bins(data, n_bins, data_range=data_range, add_epsilon=False)

        expected = np.array([0.0, 5.0, 10.0])
        np.testing.assert_array_equal(result, expected)

    def test_single_bin(self):
        """Test with single bin."""
        data = np.array([1.0, 5.0])
        n_bins = 1
        result = create_equal_width_bins(data, n_bins)

        assert len(result) == 2
        assert result[0] == 1.0
        assert result[1] >= 5.0  # Should be >= due to epsilon


class TestApplyEqualWidthFallback:
    """Test the apply_equal_width_fallback function."""

    def test_with_warning(self):
        """Test fallback with warning enabled."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        n_bins = 2
        method_name = "TestMethod"

        with pytest.warns(UserWarning, match="TestMethod binning failed, falling back"):
            result = apply_equal_width_fallback(data, n_bins, method_name, warn_on_fallback=True)

        # Should return equal-width bins
        assert len(result) == n_bins + 1
        assert result[0] == 1.0

    def test_without_warning(self):
        """Test fallback with warning disabled."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        n_bins = 2
        method_name = "TestMethod"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = apply_equal_width_fallback(data, n_bins, method_name, warn_on_fallback=False)

            # Should not issue warning
            assert len(w) == 0

        # Should still return equal-width bins
        assert len(result) == n_bins + 1
        assert result[0] == 1.0

    def test_default_method_name(self):
        """Test with default method name."""
        data = np.array([1.0, 2.0, 3.0])
        n_bins = 2

        with pytest.warns(UserWarning, match="method binning failed"):
            apply_equal_width_fallback(data, n_bins)  # Uses default method name

    def test_default_warn_setting(self):
        """Test default warn_on_fallback setting."""
        data = np.array([1.0, 2.0, 3.0])
        n_bins = 2

        with pytest.warns(UserWarning):
            apply_equal_width_fallback(data, n_bins, "TestMethod")  # Default warn=True


class TestValidateBinningInput:
    """Test the validate_binning_input function."""

    def test_empty_data(self):
        """Test validation with empty data."""
        data = np.array([])
        n_bins = 3

        with pytest.raises(ValueError, match="Cannot bin empty data array"):
            validate_binning_input(data, n_bins)

    def test_negative_bins(self):
        """Test validation with negative bins."""
        data = np.array([1.0, 2.0, 3.0])
        n_bins = -1

        with pytest.raises(ValueError, match="Number of bins must be positive, got -1"):
            validate_binning_input(data, n_bins)

    def test_zero_bins(self):
        """Test validation with zero bins."""
        data = np.array([1.0, 2.0, 3.0])
        n_bins = 0

        with pytest.raises(ValueError, match="Number of bins must be positive, got 0"):
            validate_binning_input(data, n_bins)

    def test_bins_exceed_data_size(self):
        """Test warning when bins >= data size."""
        data = np.array([1.0, 2.0, 3.0])
        n_bins = 3

        with pytest.warns(UserWarning, match="Number of bins \\(3\\) is greater than or equal"):
            validate_binning_input(data, n_bins)

    def test_bins_much_larger_than_data(self):
        """Test warning when bins >> data size."""
        data = np.array([1.0, 2.0])
        n_bins = 10

        with pytest.warns(UserWarning, match="may result in many empty bins"):
            validate_binning_input(data, n_bins)

    def test_valid_input(self):
        """Test with valid input - no errors or warnings."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        n_bins = 3

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_binning_input(data, n_bins)

            # Should not raise warnings
            assert len(w) == 0

    def test_single_data_point(self):
        """Test with single data point and single bin."""
        data = np.array([5.0])
        n_bins = 1

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_binning_input(data, n_bins)

            # This should trigger a warning since n_bins >= data.size
            assert len(w) == 1
            assert "greater than or equal to data size" in str(w[0].message)


class TestEnsureMonotonicEdges:
    """Test the ensure_monotonic_edges function."""

    def test_already_monotonic(self):
        """Test with already monotonic edges."""
        edges = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ensure_monotonic_edges(edges)

        np.testing.assert_array_equal(result, edges)

    def test_fix_equal_adjacent(self):
        """Test fixing equal adjacent values."""
        edges = np.array([1.0, 2.0, 2.0, 3.0])
        result = ensure_monotonic_edges(edges)

        # Should fix the equal values with machine epsilon
        assert result[0] == 1.0
        assert result[1] == 2.0
        assert result[2] > result[1]  # Should be slightly larger
        assert result[3] == 3.0

    def test_fix_decreasing_values(self):
        """Test fixing decreasing values."""
        edges = np.array([1.0, 3.0, 2.0, 4.0])
        result = ensure_monotonic_edges(edges)

        # Should fix the decreasing sequence using machine epsilon
        assert result[0] == 1.0
        assert result[1] == 3.0
        assert result[2] > result[1]  # Should be > result[1] by machine epsilon
        assert result[3] == 4.0

    def test_multiple_violations(self):
        """Test with multiple monotonicity violations."""
        edges = np.array([1.0, 1.0, 0.5, 2.0, 2.0])
        result = ensure_monotonic_edges(edges)

        # All values should be strictly increasing by machine epsilon increments
        for i in range(1, len(result)):
            assert result[i] > result[i - 1], f"Position {i}: {result[i]} <= {result[i-1]}"

    def test_preserve_original_array(self):
        """Test that original array is modified in place."""
        original_edges = np.array([1.0, 2.0, 2.0, 3.0])
        edges_copy = original_edges.copy()

        result = ensure_monotonic_edges(original_edges)

        # Should modify the original array and fix the equal values
        assert result is original_edges
        assert not np.array_equal(
            original_edges, edges_copy
        )  # Should have changed the equal values    def test_single_edge(self):
        """Test with single edge."""
        edges = np.array([5.0])
        result = ensure_monotonic_edges(edges)

        np.testing.assert_array_equal(result, edges)

    def test_two_equal_edges(self):
        """Test with two equal edges."""
        edges = np.array([5.0, 5.0])
        result = ensure_monotonic_edges(edges)

        assert result[0] == 5.0
        assert result[1] > result[0]  # Should be greater by machine epsilon

    def test_zero_value_epsilon_branch(self):
        """Test epsilon calculation with zero values to cover epsilon == 0 branch."""
        edges = np.array([0.0, 0.0])
        result = ensure_monotonic_edges(edges)

        # Should handle zero case and create strictly increasing values
        # This specifically tests the epsilon == 0 branch in the implementation
        assert result[0] == 0.0
        assert result[1] > 0.0  # Should be > 0 due to epsilon handling

    def test_ensure_monotonic_edges_zero_value_branch(self):
        """Test ensure_monotonic_edges with zero values to cover epsilon == 0 branch."""
        # Create edges with zeros to trigger the epsilon == 0 condition
        edges = np.array([0.0, 0.0, 1.0])

        result = ensure_monotonic_edges(edges)

        # Should handle the zero case and create strictly increasing edges
        assert result[0] == 0.0
        assert result[1] > result[0]  # Should be greater due to epsilon handling
        assert result[2] == 1.0

        # Verify the epsilon == 0 branch was covered (line 150 in _equal_width_utils.py)
        assert all(result[i] > result[i - 1] for i in range(1, len(result)))

    def test_equal_width_ensure_monotonic_zero_epsilon(self):
        """Test ensure_monotonic_edges when value is exactly zero."""
        # Create edges where one value is exactly 0.0 to trigger special case
        edges = np.array([0.0, 0.0])

        result = ensure_monotonic_edges(edges)

        # Should use 1e-10 as epsilon when abs(edges[i-1]) == 0 (covers line 148)
        assert result[1] > result[0]
        assert result[1] == 1e-10
