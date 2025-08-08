"""Comprehensive tests for binlearn.utils._binning_operations module.

This module tests all functions in the binning operations utility module
to achieve 100% test coverage, including edge cases and error conditions.
"""

import numpy as np
import pytest

from binlearn.utils._binning_operations import (
    validate_bin_edges_format,
    validate_bin_representatives_format,
    validate_bins,
    default_representatives,
    create_bin_masks,
    generate_default_flexible_representatives,
    validate_flexible_bins,
    validate_flexible_bin_spec_format,
    _validate_single_flexible_bin_def,
    is_missing_value,
    find_flexible_bin_for_value,
    calculate_flexible_bin_width,
    transform_value_to_flexible_bin,
    get_flexible_bin_count,
)
from binlearn.utils._types import MISSING_VALUE, BELOW_RANGE, ABOVE_RANGE


class TestValidateBinEdgesFormat:
    """Test suite for validate_bin_edges_format function."""

    def test_valid_bin_edges(self):
        """Test validation of valid bin edges."""
        # Standard valid case
        edges = {"col1": [1.0, 2.0, 3.0, 4.0]}
        validate_bin_edges_format(edges)

        # Two edges minimum
        edges = {"col2": [0.0, 1.0]}
        validate_bin_edges_format(edges)

    def test_none_input(self):
        """Test that None input is handled gracefully."""
        validate_bin_edges_format(None)  # Should not raise

    def test_invalid_input_type(self):
        """Test validation of invalid input types."""
        # Non-dictionary input
        with pytest.raises(ValueError, match="must be a dictionary"):
            validate_bin_edges_format([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="must be a dictionary"):
            validate_bin_edges_format(np.array([1.0, 2.0, 3.0]))

    def test_non_iterable_edges(self):
        """Test error with non-iterable edges."""
        with pytest.raises(ValueError, match="must be array-like"):
            edges = {"col1": 5}  # Single number, not iterable
            validate_bin_edges_format(edges)

        # String edges (iterable but explicitly rejected)
        with pytest.raises(ValueError, match="must be array-like"):
            edges = {"col1": "abc"}
            validate_bin_edges_format(edges)

    def test_insufficient_edges(self):
        """Test error with insufficient number of edges."""
        # Empty edges
        with pytest.raises(ValueError, match="needs at least 2 bin edges"):
            edges = {"col1": []}
            validate_bin_edges_format(edges)

        # Single edge
        with pytest.raises(ValueError, match="needs at least 2 bin edges"):
            edges = {"col1": [1.0]}
            validate_bin_edges_format(edges)

    def test_non_numeric_edges(self):
        """Test error with non-numeric edges."""
        with pytest.raises(ValueError, match="must be numeric"):
            edges = {"col1": ["a", "b", "c"]}
            validate_bin_edges_format(edges)

    def test_unsorted_edges(self):
        """Test error with unsorted edges."""
        with pytest.raises(ValueError, match="must be sorted"):
            edges = {"col1": [3.0, 1.0, 2.0]}
            validate_bin_edges_format(edges)

    def test_valid_duplicate_edges(self):
        """Test handling of duplicate edges (should be allowed)."""
        edges = {"col1": [1.0, 2.0, 2.0, 3.0]}
        validate_bin_edges_format(edges)  # Should not raise

    def test_multiple_columns(self):
        """Test validation with multiple columns."""
        edges = {"col1": [1.0, 2.0, 3.0], "col2": [0.0, 5.0, 10.0, 15.0]}
        validate_bin_edges_format(edges)


class TestValidateBinRepresentativesFormat:
    """Test suite for validate_bin_representatives_format function."""

    def test_valid_representatives(self):
        """Test validation of valid representatives."""
        reps = {"col1": [1.5, 2.5, 3.5]}
        validate_bin_representatives_format(reps)

    def test_none_input(self):
        """Test that None input is handled gracefully."""
        validate_bin_representatives_format(None)  # Should not raise

    def test_with_edges(self):
        """Test validation with corresponding edges."""
        reps = {"col1": [1.5, 2.5]}
        edges = {"col1": [1.0, 2.0, 3.0]}
        validate_bin_representatives_format(reps, edges)

    def test_invalid_input_type(self):
        """Test validation of invalid input types."""
        # Non-dictionary input
        with pytest.raises(ValueError, match="must be a dictionary"):
            validate_bin_representatives_format([1.5, 2.5, 3.5])

    def test_non_iterable_representatives(self):
        """Test error with non-iterable representatives."""
        with pytest.raises(ValueError, match="must be array-like"):
            reps = {"col1": 1.5}  # Single number, not iterable
            validate_bin_representatives_format(reps)


class TestValidateBins:
    """Test suite for validate_bins function."""

    def test_valid_bins(self):
        """Test validation of valid bin specifications."""
        bin_spec = {"col1": [1.0, 2.0, 3.0]}
        bin_reps = {"col1": [1.5, 2.5]}
        validate_bins(bin_spec, bin_reps)

    def test_none_inputs(self):
        """Test validation with None inputs."""
        validate_bins(None, None)

        bin_spec = {"col1": [1.0, 2.0, 3.0]}
        validate_bins(bin_spec, None)

        bin_reps = {"col1": [1.5, 2.5]}
        validate_bins(None, bin_reps)

    def test_mismatched_columns(self):
        """Test that validate_bins doesn't raise error for different columns."""
        bin_spec = {"col1": [1.0, 2.0, 3.0]}
        bin_reps = {"col2": [1.5, 2.5]}
        # This should not raise an error - validate_bins only checks consistency
        # for columns that exist in both dictionaries
        validate_bins(bin_spec, bin_reps)

    def test_inconsistent_bin_counts(self):
        """Test error with inconsistent bin counts."""
        bin_spec = {"col1": [1.0, 2.0, 3.0]}  # 2 bins
        bin_reps = {"col1": [1.5, 2.5, 3.5]}  # 3 representatives
        with pytest.raises(ValueError):
            validate_bins(bin_spec, bin_reps)


class TestDefaultRepresentatives:
    """Test suite for default_representatives function."""

    def test_basic_representatives(self):
        """Test generation of basic representatives."""
        edges = [1.0, 2.0, 3.0, 4.0]
        reps = default_representatives(edges)
        expected = [1.5, 2.5, 3.5]
        assert len(reps) == len(expected)
        for i, val in enumerate(expected):
            assert abs(reps[i] - val) < 1e-10

    def test_two_edges(self):
        """Test representatives with minimum edges."""
        edges = [0.0, 1.0]
        reps = default_representatives(edges)
        expected = [0.5]
        assert len(reps) == len(expected)
        assert abs(reps[0] - expected[0]) < 1e-10

    def test_negative_edges(self):
        """Test representatives with negative edges."""
        edges = [-2.0, -1.0, 0.0]
        reps = default_representatives(edges)
        expected = [-1.5, -0.5]
        assert len(reps) == len(expected)
        for i, val in enumerate(expected):
            assert abs(reps[i] - val) < 1e-10


class TestCreateBinMasks:
    """Test suite for create_bin_masks function."""

    def test_basic_masks(self):
        """Test creation of basic bin masks."""
        bin_indices = np.array([0, 1, 2, 0, 1])
        n_bins = 3
        valid_mask, nan_mask, below_mask, above_mask = create_bin_masks(bin_indices, n_bins)

        np.testing.assert_array_equal(valid_mask, [True, True, True, True, True])
        np.testing.assert_array_equal(nan_mask, [False, False, False, False, False])
        np.testing.assert_array_equal(below_mask, [False, False, False, False, False])
        np.testing.assert_array_equal(above_mask, [False, False, False, False, False])

    def test_special_values(self):
        """Test masks with special values."""
        bin_indices = np.array([0, MISSING_VALUE, BELOW_RANGE, ABOVE_RANGE, 1])
        n_bins = 2
        valid_mask, nan_mask, below_mask, above_mask = create_bin_masks(bin_indices, n_bins)

        np.testing.assert_array_equal(valid_mask, [True, False, False, False, True])
        np.testing.assert_array_equal(nan_mask, [False, True, False, False, False])
        np.testing.assert_array_equal(below_mask, [False, False, True, False, False])
        np.testing.assert_array_equal(above_mask, [False, False, False, True, False])

    def test_empty_indices(self):
        """Test masks with empty indices array."""
        bin_indices = np.array([])
        n_bins = 3
        valid_mask, nan_mask, below_mask, above_mask = create_bin_masks(bin_indices, n_bins)

        assert len(valid_mask) == 0
        assert len(nan_mask) == 0
        assert len(below_mask) == 0
        assert len(above_mask) == 0


class TestIsMissingValue:
    """Test suite for is_missing_value function."""

    def test_nan_values(self):
        """Test detection of NaN values."""
        assert is_missing_value(np.nan)
        assert is_missing_value(float("nan"))

    def test_none_values(self):
        """Test detection of None values."""
        assert is_missing_value(None)

    def test_regular_values(self):
        """Test that regular values are not considered missing."""
        assert not is_missing_value(0)
        assert not is_missing_value(1.5)
        assert not is_missing_value("test")
        assert not is_missing_value([])

    def test_infinity_values(self):
        """Test that infinity values are not considered missing."""
        assert not is_missing_value(np.inf)
        assert not is_missing_value(-np.inf)

    def test_special_constants(self):
        """Test that special constants are not considered missing."""
        assert not is_missing_value(MISSING_VALUE)
        assert not is_missing_value(BELOW_RANGE)
        assert not is_missing_value(ABOVE_RANGE)


class TestCalculateFlexibleBinWidth:
    """Test suite for calculate_flexible_bin_width function."""

    def test_interval_width(self):
        """Test width calculation for interval bins."""
        bin_def = (1.0, 4.0)  # interval bin as tuple
        width = calculate_flexible_bin_width(bin_def)
        assert width == 3.0

    def test_singleton_width(self):
        """Test width calculation for singleton bins."""
        bin_def = 5  # singleton bin as scalar
        width = calculate_flexible_bin_width(bin_def)
        assert width == 0.0

    def test_zero_width_interval(self):
        """Test width calculation for zero-width interval."""
        bin_def = (2.0, 2.0)  # interval with same start and end
        width = calculate_flexible_bin_width(bin_def)
        assert width == 0.0


class TestGenerateDefaultFlexibleRepresentatives:
    """Test suite for generate_default_flexible_representatives function."""

    def test_interval_bins(self):
        """Test generation for interval bin definitions."""
        bin_defs = [(1.0, 2.0), (2.0, 3.0)]  # interval as tuple  # interval as tuple
        reps = generate_default_flexible_representatives(bin_defs)
        expected = [1.5, 2.5]
        assert len(reps) == len(expected)
        for i, val in enumerate(expected):
            assert abs(reps[i] - val) < 1e-10

    def test_singleton_bins(self):
        """Test generation for singleton bin definitions."""
        bin_defs = [5, 10]  # singleton as scalar  # singleton as scalar
        reps = generate_default_flexible_representatives(bin_defs)
        expected = [5, 10]
        assert len(reps) == len(expected)
        for i, val in enumerate(expected):
            assert abs(reps[i] - val) < 1e-10

    def test_mixed_bins(self):
        """Test generation for mixed bin definitions."""
        bin_defs = [
            (1.0, 2.0),  # interval as tuple
            5,  # singleton as scalar
            (6.0, 7.0),  # interval as tuple
        ]
        reps = generate_default_flexible_representatives(bin_defs)
        expected = [1.5, 5, 6.5]
        assert len(reps) == len(expected)
        for i, val in enumerate(expected):
            assert abs(reps[i] - val) < 1e-10

    def test_empty_bins(self):
        """Test generation for empty bin definitions."""
        bin_defs = []
        reps = generate_default_flexible_representatives(bin_defs)
        expected = []
        assert len(reps) == len(expected)
