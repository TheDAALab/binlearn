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
        edges = np.array([1.0, 2.0, 3.0, 4.0])
        validate_bin_edges_format(edges, "test_column", 0)

        # Two edges minimum
        edges = np.array([0.0, 1.0])
        validate_bin_edges_format(edges, "col", 1)

    def test_empty_edges_error(self):
        """Test error when edges array is empty."""
        edges = np.array([])
        with pytest.raises(ValueError, match="Column test_col, spec 0: Need at least 2 edges"):
            validate_bin_edges_format(edges, "test_col", 0)

    def test_single_edge_error(self):
        """Test error when only one edge is provided."""
        edges = np.array([1.0])
        with pytest.raises(ValueError, match="Column col1, spec 1: Need at least 2 edges"):
            validate_bin_edges_format(edges, "col1", 1)

    def test_non_finite_edges_error(self):
        """Test error when edges contain non-finite values."""
        # Test with NaN
        edges = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="Column test, spec 2: All edges must be finite"):
            validate_bin_edges_format(edges, "test", 2)

        # Test with infinity
        edges = np.array([1.0, 2.0, np.inf])
        with pytest.raises(ValueError, match="Column col, spec 0: All edges must be finite"):
            validate_bin_edges_format(edges, "col", 0)

        # Test with negative infinity
        edges = np.array([-np.inf, 1.0, 2.0])
        with pytest.raises(ValueError, match="Column x, spec 1: All edges must be finite"):
            validate_bin_edges_format(edges, "x", 1)

    def test_non_monotonic_edges_error(self):
        """Test error when edges are not strictly increasing."""
        # Decreasing edges
        edges = np.array([3.0, 2.0, 1.0])
        with pytest.raises(
            ValueError, match="Column test, spec 0: Edges must be strictly increasing"
        ):
            validate_bin_edges_format(edges, "test", 0)

        # Equal consecutive edges
        edges = np.array([1.0, 2.0, 2.0, 3.0])
        with pytest.raises(
            ValueError, match="Column col, spec 1: Edges must be strictly increasing"
        ):
            validate_bin_edges_format(edges, "col", 1)

    def test_edge_case_values(self):
        """Test with edge case numeric values."""
        # Very small differences (but still increasing)
        edges = np.array([0.0, 1e-10, 2e-10])
        validate_bin_edges_format(edges, "small", 0)

        # Large values
        edges = np.array([1e6, 2e6, 3e6])
        validate_bin_edges_format(edges, "large", 0)

        # Negative values
        edges = np.array([-3.0, -2.0, -1.0, 0.0, 1.0])
        validate_bin_edges_format(edges, "negative", 0)


class TestValidateBinRepresentativesFormat:
    """Test suite for validate_bin_representatives_format function."""

    def test_valid_representatives(self):
        """Test validation of valid representatives."""
        reps = np.array([1.5, 2.5, 3.5])
        edges = np.array([1.0, 2.0, 3.0, 4.0])
        validate_bin_representatives_format(reps, edges, "test_col", 0)

    def test_wrong_length_error(self):
        """Test error when representatives length doesn't match number of bins."""
        reps = np.array([1.5, 2.5])  # 2 representatives
        edges = np.array([1.0, 2.0, 3.0, 4.0])  # 3 bins
        with pytest.raises(
            ValueError, match="Column test, spec 0: Expected 3 representatives, got 2"
        ):
            validate_bin_representatives_format(reps, edges, "test", 0)

    def test_non_finite_representatives_error(self):
        """Test error when representatives contain non-finite values."""
        # Test with NaN
        reps = np.array([1.5, np.nan, 3.5])
        edges = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(
            ValueError, match="Column col, spec 1: All representatives must be finite"
        ):
            validate_bin_representatives_format(reps, edges, "col", 1)

        # Test with infinity
        reps = np.array([1.5, np.inf, 3.5])
        edges = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(
            ValueError, match="Column x, spec 2: All representatives must be finite"
        ):
            validate_bin_representatives_format(reps, edges, "x", 2)

    def test_edge_cases(self):
        """Test edge cases for representatives validation."""
        # Single bin
        reps = np.array([1.5])
        edges = np.array([1.0, 2.0])
        validate_bin_representatives_format(reps, edges, "single", 0)

        # Large number of bins
        n_bins = 100
        reps = np.arange(n_bins, dtype=float)
        edges = np.arange(n_bins + 1, dtype=float)
        validate_bin_representatives_format(reps, edges, "many", 0)


class TestValidateBins:
    """Test suite for validate_bins function."""

    def test_valid_bins_dict(self):
        """Test validation of valid bins dictionary."""
        bins = {"col1": np.array([1.0, 2.0, 3.0]), "col2": np.array([0.0, 1.0, 2.0, 3.0])}
        representatives = {"col1": np.array([1.5, 2.5]), "col2": np.array([0.5, 1.5, 2.5])}
        validate_bins(bins, representatives)

    def test_bins_without_representatives(self):
        """Test validation when representatives is None."""
        bins = {"col1": np.array([1.0, 2.0, 3.0]), "col2": np.array([0.0, 1.0])}
        validate_bins(bins, None)

    def test_mismatched_columns_error(self):
        """Test error when bins and representatives have different columns."""
        bins = {"col1": np.array([1.0, 2.0, 3.0])}
        representatives = {"col2": np.array([1.5, 2.5])}
        with pytest.raises(ValueError, match="Bins and representatives must have same columns"):
            validate_bins(bins, representatives)

    def test_empty_bins_dict(self):
        """Test validation with empty bins dictionary."""
        validate_bins({}, None)
        validate_bins({}, {})

    def test_invalid_bin_edges_propagation(self):
        """Test that invalid bin edges errors are propagated."""
        bins = {"col1": np.array([3.0, 2.0, 1.0])}  # Non-monotonic
        with pytest.raises(ValueError, match="Edges must be strictly increasing"):
            validate_bins(bins, None)

    def test_invalid_representatives_propagation(self):
        """Test that invalid representatives errors are propagated."""
        bins = {"col1": np.array([1.0, 2.0, 3.0])}
        representatives = {"col1": np.array([1.5])}  # Wrong length
        with pytest.raises(ValueError, match="Expected 2 representatives, got 1"):
            validate_bins(bins, representatives)


class TestDefaultRepresentatives:
    """Test suite for default_representatives function."""

    def test_single_column_representatives(self):
        """Test generation of default representatives for single column."""
        edges = np.array([1.0, 2.0, 3.0, 4.0])
        expected = np.array([1.5, 2.5, 3.5])
        result = default_representatives(edges)
        np.testing.assert_array_equal(result, expected)

    def test_multiple_columns_representatives(self):
        """Test generation of default representatives for multiple columns."""
        bins = {"col1": np.array([0.0, 1.0, 2.0]), "col2": np.array([10.0, 20.0, 30.0, 40.0])}
        expected = {"col1": np.array([0.5]), "col2": np.array([15.0, 25.0, 35.0])}
        result = default_representatives(bins)

        assert set(result.keys()) == set(expected.keys())
        for col in expected:
            np.testing.assert_array_equal(result[col], expected[col])

    def test_single_bin_representatives(self):
        """Test representatives for single bin."""
        edges = np.array([0.0, 1.0])
        expected = np.array([0.5])
        result = default_representatives(edges)
        np.testing.assert_array_equal(result, expected)

    def test_negative_edges_representatives(self):
        """Test representatives with negative edges."""
        edges = np.array([-2.0, -1.0, 0.0, 1.0])
        expected = np.array([-1.5, -0.5, 0.5])
        result = default_representatives(edges)
        np.testing.assert_array_equal(result, expected)

    def test_floating_point_precision(self):
        """Test representatives with floating point precision."""
        edges = np.array([0.1, 0.2, 0.3])
        expected = np.array([0.15, 0.25])
        result = default_representatives(edges)
        np.testing.assert_allclose(result, expected)


class TestCreateBinMasks:
    """Test suite for create_bin_masks function."""

    def test_basic_bin_masks(self):
        """Test basic bin mask creation."""
        X = np.array([[1.5, 2.5], [2.5, 3.5], [0.5, 1.5]])
        edges = np.array([1.0, 2.0, 3.0])
        expected_masks = [
            np.array([True, False, True]),  # bin 0: values in [1.0, 2.0)
            np.array([False, True, False]),  # bin 1: values in [2.0, 3.0)
        ]

        result = create_bin_masks(X, edges)
        assert len(result) == len(expected_masks)
        for i, expected in enumerate(expected_masks):
            np.testing.assert_array_equal(result[i], expected)

    def test_single_column_masks(self):
        """Test bin masks for single column data."""
        X = np.array([[1.5], [2.5], [3.5], [0.5]])
        edges = np.array([1.0, 2.0, 3.0, 4.0])

        result = create_bin_masks(X, edges)
        assert len(result) == 3  # 4 edges = 3 bins

        # Check specific assignments
        assert result[0][0] == True  # 1.5 in [1.0, 2.0)
        assert result[1][1] == True  # 2.5 in [2.0, 3.0)
        assert result[2][2] == True  # 3.5 in [3.0, 4.0]
        assert not any(mask[3] for mask in result)  # 0.5 not in any bin

    def test_edge_boundary_behavior(self):
        """Test behavior at bin edges."""
        X = np.array([[1.0], [2.0], [3.0]])
        edges = np.array([1.0, 2.0, 3.0])

        result = create_bin_masks(X, edges)

        # Left edges should be included, right edges excluded (except last)
        assert result[0][0] == True  # 1.0 in [1.0, 2.0)
        assert result[1][1] == True  # 2.0 in [2.0, 3.0]
        assert result[1][2] == True  # 3.0 in [2.0, 3.0] (last bin includes right edge)

    def test_out_of_range_values(self):
        """Test handling of values outside the bin range."""
        X = np.array([[0.5], [1.5], [3.5]])
        edges = np.array([1.0, 2.0, 3.0])

        result = create_bin_masks(X, edges)

        # Only middle value should be in a bin
        assert not any(mask[0] for mask in result)  # 0.5 not in any bin
        assert result[0][1] == True  # 1.5 in bin 0
        assert not any(mask[2] for mask in result)  # 3.5 not in any bin

    def test_empty_bins(self):
        """Test handling when some bins are empty."""
        X = np.array([[1.1], [1.9]])  # All values in first bin
        edges = np.array([1.0, 2.0, 3.0, 4.0])

        result = create_bin_masks(X, edges)

        # First bin should have both values
        np.testing.assert_array_equal(result[0], [True, True])
        # Other bins should be empty
        np.testing.assert_array_equal(result[1], [False, False])
        np.testing.assert_array_equal(result[2], [False, False])

    def test_multiple_columns_masks(self):
        """Test bin masks with multiple columns."""
        X = np.array([[1.5, 10.5], [2.5, 20.5], [1.2, 15.0]])
        edges_col1 = np.array([1.0, 2.0, 3.0])
        edges_col2 = np.array([10.0, 15.0, 20.0, 25.0])

        # Test first column
        result1 = create_bin_masks(X[:, 0:1], edges_col1)
        assert result1[0][0] == True  # 1.5 in bin 0
        assert result1[1][1] == True  # 2.5 in bin 1
        assert result1[0][2] == True  # 1.2 in bin 0

        # Test second column
        result2 = create_bin_masks(X[:, 1:2], edges_col2)
        assert result2[0][0] == True  # 10.5 in bin 0
        assert result2[2][1] == True  # 20.5 in bin 2
        assert result2[1][2] == True  # 15.0 in bin 1 (right edge of bin 1)


class TestGenerateDefaultFlexibleRepresentatives:
    """Test suite for generate_default_flexible_representatives function."""

    def test_singleton_bins_representatives(self):
        """Test representatives for singleton bins."""
        bin_defs = [1, 2, 3]
        expected = [1, 2, 3]
        result = generate_default_flexible_representatives(bin_defs)
        assert result == expected

    def test_interval_bins_representatives(self):
        """Test representatives for interval bins."""
        bin_defs = [(1.0, 2.0), (3.0, 4.0)]
        expected = [1.5, 3.5]
        result = generate_default_flexible_representatives(bin_defs)
        assert result == expected

    def test_mixed_bins_representatives(self):
        """Test representatives for mixed singleton and interval bins."""
        bin_defs = [1, (2.0, 4.0), 5, (6.0, 8.0)]
        expected = [1, 3.0, 5, 7.0]
        result = generate_default_flexible_representatives(bin_defs)
        assert result == expected

    def test_float_singleton_representatives(self):
        """Test representatives for float singleton bins."""
        bin_defs = [1.5, 2.7, 3.14]
        expected = [1.5, 2.7, 3.14]
        result = generate_default_flexible_representatives(bin_defs)
        assert result == expected

    def test_zero_width_intervals(self):
        """Test representatives for zero-width intervals."""
        bin_defs = [(1.0, 1.0), (2.0, 2.0)]
        expected = [1.0, 2.0]
        result = generate_default_flexible_representatives(bin_defs)
        assert result == expected

    def test_negative_intervals(self):
        """Test representatives for negative intervals."""
        bin_defs = [(-2.0, -1.0), (-0.5, 0.5)]
        expected = [-1.5, 0.0]
        result = generate_default_flexible_representatives(bin_defs)
        assert result == expected

    def test_large_intervals(self):
        """Test representatives for large intervals."""
        bin_defs = [(0.0, 1000.0), (1000.0, 2000.0)]
        expected = [500.0, 1500.0]
        result = generate_default_flexible_representatives(bin_defs)
        assert result == expected


class TestValidateFlexibleBins:
    """Test suite for validate_flexible_bins function."""

    def test_valid_flexible_bins(self):
        """Test validation of valid flexible bins."""
        bin_spec = {"col1": [1, (2.0, 3.0), 4], "col2": [(0.0, 1.0), (2.0, 3.0)]}
        representatives = {"col1": [1, 2.5, 4], "col2": [0.5, 2.5]}
        validate_flexible_bins(bin_spec, representatives)

    def test_flexible_bins_without_representatives(self):
        """Test validation when representatives is None."""
        bin_spec = {"col1": [1, 2, 3], "col2": [(0.0, 1.0)]}
        validate_flexible_bins(bin_spec, None)

    def test_mismatched_columns_flexible_error(self):
        """Test error when bin_spec and representatives have different columns."""
        bin_spec = {"col1": [1, 2]}
        representatives = {"col2": [1, 2]}
        with pytest.raises(
            ValueError, match="Bin specification and representatives must have same columns"
        ):
            validate_flexible_bins(bin_spec, representatives)

    def test_mismatched_lengths_flexible_error(self):
        """Test error when bin definitions and representatives have different lengths."""
        bin_spec = {"col1": [1, 2, 3]}
        representatives = {"col1": [1, 2]}  # Only 2 representatives for 3 bins
        with pytest.raises(ValueError, match="Column col1: Expected 3 representatives, got 2"):
            validate_flexible_bins(bin_spec, representatives)

    def test_invalid_bin_definition_error(self):
        """Test error propagation from invalid bin definitions."""
        bin_spec = {"col1": ["invalid"]}  # Invalid bin definition
        with pytest.raises(ValueError, match="Bin must be either a numeric scalar"):
            validate_flexible_bins(bin_spec, None)

    def test_non_finite_representatives_flexible_error(self):
        """Test error when representatives contain non-finite values."""
        bin_spec = {"col1": [1, 2]}
        representatives = {"col1": [1, np.nan]}
        with pytest.raises(ValueError, match="All representatives must be finite"):
            validate_flexible_bins(bin_spec, representatives)

    def test_empty_flexible_bins(self):
        """Test validation with empty bin specification."""
        validate_flexible_bins({}, None)
        validate_flexible_bins({}, {})


class TestValidateFlexibleBinSpecFormat:
    """Test suite for validate_flexible_bin_spec_format function."""

    def test_valid_flexible_bin_spec(self):
        """Test validation of valid flexible bin specification."""
        bin_spec = {"col1": [1, (2.0, 3.0), 4], "col2": [(0.0, 1.0), 2]}
        validate_flexible_bin_spec_format(bin_spec)

    def test_empty_bin_list_error(self):
        """Test error when a column has empty bin list."""
        bin_spec = {"col1": []}
        with pytest.raises(ValueError, match="Column col1: Must have at least one bin"):
            validate_flexible_bin_spec_format(bin_spec)

    def test_invalid_singleton_type_error(self):
        """Test error when singleton bin has invalid type."""
        bin_spec = {"col1": ["invalid"]}
        with pytest.raises(
            ValueError, match="Column col1, bin 0: Bin must be either a numeric scalar"
        ):
            validate_flexible_bin_spec_format(bin_spec)

    def test_invalid_interval_length_error(self):
        """Test error when interval has wrong length."""
        bin_spec = {"col1": [(1.0, 2.0, 3.0)]}  # 3-tuple instead of 2-tuple
        with pytest.raises(ValueError, match="Column col1, bin 0: Interval must be \\(min, max\\)"):
            validate_flexible_bin_spec_format(bin_spec)

    def test_invalid_interval_values_error(self):
        """Test error when interval contains non-numeric values."""
        bin_spec = {"col1": [("a", "b")]}
        with pytest.raises(ValueError, match="Column col1, bin 0: Interval values must be numeric"):
            validate_flexible_bin_spec_format(bin_spec)

    def test_invalid_ordering_error(self):
        """Test error when interval has invalid ordering."""
        bin_spec = {"col1": [(3.0, 2.0)]}  # min > max
        with pytest.raises(
            ValueError, match="Column col1, bin 0: Interval min \\(3.0\\) must be < max \\(2.0\\)"
        ):
            validate_flexible_bin_spec_format(bin_spec)

    def test_check_finite_bounds_option(self):
        """Test check_finite_bounds parameter."""
        # Should pass without finite bounds check
        bin_spec = {"col1": [np.inf, (-np.inf, np.inf)]}
        validate_flexible_bin_spec_format(bin_spec, check_finite_bounds=False)

        # Should fail with finite bounds check
        with pytest.raises(ValueError, match="Singleton value must be finite"):
            validate_flexible_bin_spec_format(bin_spec, check_finite_bounds=True)

    def test_strict_mode_option(self):
        """Test strict parameter for interval validation."""
        # Equal bounds should fail in strict mode
        bin_spec = {"col1": [(1.0, 1.0)]}
        with pytest.raises(ValueError, match="Interval min \\(1.0\\) must be < max \\(1.0\\)"):
            validate_flexible_bin_spec_format(bin_spec, strict=True)

        # Equal bounds should pass in non-strict mode
        validate_flexible_bin_spec_format(bin_spec, strict=False)


class TestValidateSingleFlexibleBinDef:
    """Test suite for _validate_single_flexible_bin_def function."""

    def test_valid_singleton_bin(self):
        """Test validation of valid singleton bin."""
        _validate_single_flexible_bin_def(1, "col", 0)
        _validate_single_flexible_bin_def(1.5, "col", 0)
        _validate_single_flexible_bin_def(0, "col", 0)
        _validate_single_flexible_bin_def(-1.5, "col", 0)

    def test_valid_interval_bin(self):
        """Test validation of valid interval bin."""
        _validate_single_flexible_bin_def((1.0, 2.0), "col", 0)
        _validate_single_flexible_bin_def((-1.0, 1.0), "col", 0)
        _validate_single_flexible_bin_def((0, 1), "col", 0)

    def test_finite_bounds_check_singleton(self):
        """Test finite bounds checking for singleton bins."""
        # Should pass without check
        _validate_single_flexible_bin_def(np.inf, "col", 0, check_finite_bounds=False)

        # Should fail with check
        with pytest.raises(ValueError, match="Column col, bin 0: Singleton value must be finite"):
            _validate_single_flexible_bin_def(np.inf, "col", 0, check_finite_bounds=True)

        with pytest.raises(ValueError, match="Column col, bin 0: Singleton value must be finite"):
            _validate_single_flexible_bin_def(np.nan, "col", 0, check_finite_bounds=True)

    def test_finite_bounds_check_interval(self):
        """Test finite bounds checking for interval bins."""
        # Should pass without check
        _validate_single_flexible_bin_def((-np.inf, np.inf), "col", 0, check_finite_bounds=False)

        # Should fail with check
        with pytest.raises(ValueError, match="Column col, bin 0: Interval bounds must be finite"):
            _validate_single_flexible_bin_def((-np.inf, 1.0), "col", 0, check_finite_bounds=True)

        with pytest.raises(ValueError, match="Column col, bin 0: Interval bounds must be finite"):
            _validate_single_flexible_bin_def((1.0, np.inf), "col", 0, check_finite_bounds=True)

    def test_strict_mode_interval_validation(self):
        """Test strict mode for interval validation."""
        # Equal bounds should fail in strict mode
        with pytest.raises(ValueError, match="Interval min \\(1.0\\) must be < max \\(1.0\\)"):
            _validate_single_flexible_bin_def((1.0, 1.0), "col", 0, strict=True)

        # Equal bounds should pass in non-strict mode
        _validate_single_flexible_bin_def((1.0, 1.0), "col", 0, strict=False)

        # Reversed bounds should fail in both modes
        with pytest.raises(ValueError, match="Interval min \\(2.0\\) must be < max \\(1.0\\)"):
            _validate_single_flexible_bin_def((2.0, 1.0), "col", 0, strict=True)

        with pytest.raises(ValueError, match="Interval min \\(2.0\\) must be <= max \\(1.0\\)"):
            _validate_single_flexible_bin_def((2.0, 1.0), "col", 0, strict=False)

    def test_invalid_bin_types(self):
        """Test error handling for invalid bin definition types."""
        # String
        with pytest.raises(ValueError, match="Bin must be either a numeric scalar"):
            _validate_single_flexible_bin_def("invalid", "col", 0)

        # List
        with pytest.raises(ValueError, match="Bin must be either a numeric scalar"):
            _validate_single_flexible_bin_def([1, 2], "col", 0)

        # Wrong tuple length
        with pytest.raises(ValueError, match="Interval must be \\(min, max\\)"):
            _validate_single_flexible_bin_def((1, 2, 3), "col", 0)

        # Non-numeric interval values
        with pytest.raises(ValueError, match="Interval values must be numeric"):
            _validate_single_flexible_bin_def(("a", "b"), "col", 0)


class TestIsMissingValue:
    """Test suite for is_missing_value function."""

    def test_none_values(self):
        """Test detection of None values."""
        assert is_missing_value(None) == True

    def test_nan_values(self):
        """Test detection of NaN values."""
        assert is_missing_value(np.nan) == True
        assert is_missing_value(float("nan")) == True

    def test_valid_numeric_values(self):
        """Test that valid numeric values are not considered missing."""
        assert is_missing_value(0) == False
        assert is_missing_value(0.0) == False
        assert is_missing_value(1) == False
        assert is_missing_value(-1) == False
        assert is_missing_value(1.5) == False
        assert is_missing_value(-1.5) == False

    def test_infinite_values(self):
        """Test that infinite values are not considered missing."""
        assert is_missing_value(np.inf) == False
        assert is_missing_value(-np.inf) == False

    def test_non_numeric_values(self):
        """Test that non-numeric values are not considered missing."""
        assert is_missing_value("string") == False
        assert is_missing_value([1, 2, 3]) == False
        assert is_missing_value({"key": "value"}) == False
        assert is_missing_value(True) == False
        assert is_missing_value(False) == False


class TestFindFlexibleBinForValue:
    """Test suite for find_flexible_bin_for_value function."""

    def test_singleton_bin_matching(self):
        """Test finding bins for singleton bin definitions."""
        bin_defs = [1, 2, 3]

        assert find_flexible_bin_for_value(1, bin_defs) == 0
        assert find_flexible_bin_for_value(2, bin_defs) == 1
        assert find_flexible_bin_for_value(3, bin_defs) == 2

    def test_interval_bin_matching(self):
        """Test finding bins for interval bin definitions."""
        bin_defs = [(1.0, 2.0), (3.0, 4.0)]

        assert find_flexible_bin_for_value(1.0, bin_defs) == 0
        assert find_flexible_bin_for_value(1.5, bin_defs) == 0
        assert find_flexible_bin_for_value(2.0, bin_defs) == 0
        assert find_flexible_bin_for_value(3.0, bin_defs) == 1
        assert find_flexible_bin_for_value(3.5, bin_defs) == 1
        assert find_flexible_bin_for_value(4.0, bin_defs) == 1

    def test_mixed_bin_matching(self):
        """Test finding bins for mixed singleton and interval definitions."""
        bin_defs = [1, (2.0, 3.0), 4]

        assert find_flexible_bin_for_value(1, bin_defs) == 0
        assert find_flexible_bin_for_value(2.0, bin_defs) == 1
        assert find_flexible_bin_for_value(2.5, bin_defs) == 1
        assert find_flexible_bin_for_value(3.0, bin_defs) == 1
        assert find_flexible_bin_for_value(4, bin_defs) == 2

    def test_no_match_values(self):
        """Test values that don't match any bin."""
        bin_defs = [1, (2.0, 3.0)]

        assert find_flexible_bin_for_value(0, bin_defs) == MISSING_VALUE
        assert find_flexible_bin_for_value(1.5, bin_defs) == MISSING_VALUE
        assert find_flexible_bin_for_value(4, bin_defs) == MISSING_VALUE

    def test_float_precision_matching(self):
        """Test matching with floating point precision."""
        bin_defs = [1.1, (2.2, 3.3)]

        assert find_flexible_bin_for_value(1.1, bin_defs) == 0
        assert find_flexible_bin_for_value(2.2, bin_defs) == 1
        assert find_flexible_bin_for_value(3.3, bin_defs) == 1

    def test_first_match_priority(self):
        """Test that first matching bin is returned."""
        # Overlapping intervals - first one should win
        bin_defs = [(1.0, 3.0), (2.0, 4.0)]

        assert find_flexible_bin_for_value(2.5, bin_defs) == 0  # Matches first interval

    def test_edge_case_values(self):
        """Test edge case values like zero and negative numbers."""
        bin_defs = [0, (-1.0, 1.0)]

        assert find_flexible_bin_for_value(0, bin_defs) == 0
        assert find_flexible_bin_for_value(-1.0, bin_defs) == 1
        assert find_flexible_bin_for_value(-0.5, bin_defs) == 1
        assert find_flexible_bin_for_value(1.0, bin_defs) == 1


class TestCalculateFlexibleBinWidth:
    """Test suite for calculate_flexible_bin_width function."""

    def test_singleton_bin_width(self):
        """Test width calculation for singleton bins."""
        assert calculate_flexible_bin_width(1) == 0.0
        assert calculate_flexible_bin_width(1.5) == 0.0
        assert calculate_flexible_bin_width(-1) == 0.0
        assert calculate_flexible_bin_width(0) == 0.0

    def test_interval_bin_width(self):
        """Test width calculation for interval bins."""
        assert calculate_flexible_bin_width((1.0, 2.0)) == 1.0
        assert calculate_flexible_bin_width((0.0, 1.0)) == 1.0
        assert calculate_flexible_bin_width((-1.0, 1.0)) == 2.0
        assert calculate_flexible_bin_width((1.5, 2.5)) == 1.0

    def test_zero_width_intervals(self):
        """Test width calculation for zero-width intervals."""
        assert calculate_flexible_bin_width((1.0, 1.0)) == 0.0
        assert calculate_flexible_bin_width((0.0, 0.0)) == 0.0

    def test_negative_width_intervals(self):
        """Test width calculation for negative-width intervals."""
        # Note: This should technically be invalid, but the function calculates the difference
        assert calculate_flexible_bin_width((2.0, 1.0)) == -1.0

    def test_large_intervals(self):
        """Test width calculation for large intervals."""
        assert calculate_flexible_bin_width((0.0, 1000.0)) == 1000.0
        assert calculate_flexible_bin_width((-500.0, 500.0)) == 1000.0

    def test_small_intervals(self):
        """Test width calculation for small intervals."""
        assert calculate_flexible_bin_width((0.0, 1e-6)) == 1e-6
        np.testing.assert_allclose(calculate_flexible_bin_width((0.1, 0.2)), 0.1)

    def test_invalid_bin_definition_error(self):
        """Test error for invalid bin definitions."""
        with pytest.raises(ValueError, match="Unknown bin definition"):
            calculate_flexible_bin_width("invalid")

        with pytest.raises(ValueError, match="Unknown bin definition"):
            calculate_flexible_bin_width([1, 2])

        with pytest.raises(ValueError, match="Unknown bin definition"):
            calculate_flexible_bin_width((1, 2, 3))


class TestTransformValueToFlexibleBin:
    """Test suite for transform_value_to_flexible_bin function."""

    def test_missing_value_handling(self):
        """Test handling of missing values."""
        bin_defs = [1, (2.0, 3.0)]

        assert transform_value_to_flexible_bin(None, bin_defs) == MISSING_VALUE
        assert transform_value_to_flexible_bin(np.nan, bin_defs) == MISSING_VALUE

    def test_valid_value_transformation(self):
        """Test transformation of valid values."""
        bin_defs = [1, (2.0, 3.0), 4]

        assert transform_value_to_flexible_bin(1, bin_defs) == 0
        assert transform_value_to_flexible_bin(2.5, bin_defs) == 1
        assert transform_value_to_flexible_bin(4, bin_defs) == 2

    def test_unmatched_value_handling(self):
        """Test handling of values that don't match any bin."""
        bin_defs = [1, (2.0, 3.0)]

        assert transform_value_to_flexible_bin(0, bin_defs) == MISSING_VALUE
        assert transform_value_to_flexible_bin(1.5, bin_defs) == MISSING_VALUE
        assert transform_value_to_flexible_bin(5, bin_defs) == MISSING_VALUE

    def test_edge_case_transformations(self):
        """Test transformation of edge case values."""
        bin_defs = [0, (-1.0, 1.0)]

        assert transform_value_to_flexible_bin(0, bin_defs) == 0
        assert transform_value_to_flexible_bin(-1.0, bin_defs) == 1
        assert transform_value_to_flexible_bin(1.0, bin_defs) == 1
        assert transform_value_to_flexible_bin(-0.5, bin_defs) == 1


class TestGetFlexibleBinCount:
    """Test suite for get_flexible_bin_count function."""

    def test_single_column_bin_count(self):
        """Test bin count calculation for single column."""
        bin_spec = {"col1": [1, 2, 3]}
        expected = {"col1": 3}
        result = get_flexible_bin_count(bin_spec)
        assert result == expected

    def test_multiple_columns_bin_count(self):
        """Test bin count calculation for multiple columns."""
        bin_spec = {"col1": [1, 2], "col2": [(1.0, 2.0), (3.0, 4.0), 5], "col3": [1]}
        expected = {"col1": 2, "col2": 3, "col3": 1}
        result = get_flexible_bin_count(bin_spec)
        assert result == expected

    def test_empty_bin_spec(self):
        """Test bin count calculation for empty specification."""
        result = get_flexible_bin_count({})
        assert result == {}

    def test_mixed_bin_types_count(self):
        """Test bin count calculation for mixed bin types."""
        bin_spec = {
            "singletons": [1, 2, 3, 4, 5],
            "intervals": [(1.0, 2.0), (3.0, 4.0)],
            "mixed": [1, (2.0, 3.0), 4, (5.0, 6.0)],
        }
        expected = {"singletons": 5, "intervals": 2, "mixed": 4}
        result = get_flexible_bin_count(bin_spec)
        assert result == expected


class TestIntegrationScenarios:
    """Integration tests combining multiple binning operations functions."""

    def test_complete_flexible_binning_workflow(self):
        """Test complete workflow with flexible binning operations."""
        # Define bin specification
        bin_spec = {"col1": [1, (2.0, 4.0), 5], "col2": [(0.0, 1.0), 2]}

        # Validate bin specification
        validate_flexible_bin_spec_format(bin_spec)

        # Generate default representatives
        representatives = {}
        for col, bin_defs in bin_spec.items():
            representatives[col] = generate_default_flexible_representatives(bin_defs)

        # Validate bins with representatives
        validate_flexible_bins(bin_spec, representatives)

        # Get bin counts
        bin_counts = get_flexible_bin_count(bin_spec)

        # Test value transformations
        test_values = [1, 3.0, 5, 0.5, 2, None, 10]
        col1_results = []
        col2_results = []

        for value in test_values:
            col1_results.append(transform_value_to_flexible_bin(value, bin_spec["col1"]))
            col2_results.append(transform_value_to_flexible_bin(value, bin_spec["col2"]))

        # Verify expected results
        assert representatives["col1"] == [1, 3.0, 5]
        assert representatives["col2"] == [0.5, 2]
        assert bin_counts == {"col1": 3, "col2": 2}

        # Check specific transformations
        assert col1_results[0] == 0  # 1 -> bin 0
        assert col1_results[1] == 1  # 3.0 -> bin 1
        assert col1_results[2] == 2  # 5 -> bin 2
        assert col1_results[5] == MISSING_VALUE  # None -> MISSING_VALUE

        assert col2_results[3] == 0  # 0.5 -> bin 0
        assert col2_results[4] == 1  # 2 -> bin 1
        assert col2_results[6] == MISSING_VALUE  # 10 -> MISSING_VALUE

    def test_interval_binning_workflow(self):
        """Test complete workflow with interval binning operations."""
        # Define bins
        bins = {
            "feature1": np.array([0.0, 1.0, 2.0, 3.0]),
            "feature2": np.array([10.0, 20.0, 30.0]),
        }

        # Generate representatives
        representatives = default_representatives(bins)

        # Validate bins
        validate_bins(bins, representatives)

        # Create test data
        X_col1 = np.array([[0.5], [1.5], [2.5], [0.2], [2.8]])
        X_col2 = np.array([[15.0], [25.0], [35.0], [5.0], [25.5]])

        # Create bin masks
        masks_col1 = create_bin_masks(X_col1, bins["feature1"])
        masks_col2 = create_bin_masks(X_col2, bins["feature2"])

        # Verify results
        expected_reps = {"feature1": np.array([0.5, 1.5, 2.5]), "feature2": np.array([15.0, 25.0])}

        for col in expected_reps:
            np.testing.assert_array_equal(representatives[col], expected_reps[col])

        # Check bin assignments
        assert len(masks_col1) == 3  # 4 edges = 3 bins
        assert len(masks_col2) == 2  # 3 edges = 2 bins

        # Verify specific assignments
        assert masks_col1[0][0] == True  # 0.5 in bin 0
        assert masks_col1[1][1] == True  # 1.5 in bin 1
        assert masks_col1[2][2] == True  # 2.5 in bin 2

    def test_error_propagation_workflow(self):
        """Test error propagation through validation workflow."""
        # Test with invalid bin edges
        invalid_bins = {"col1": np.array([3.0, 2.0, 1.0])}  # Non-monotonic

        with pytest.raises(ValueError, match="Edges must be strictly increasing"):
            validate_bins(invalid_bins, None)

        # Test with invalid flexible bins
        invalid_flex_bins = {"col1": [(3.0, 2.0)]}  # min > max

        with pytest.raises(ValueError, match="Interval min \\(3.0\\) must be < max \\(2.0\\)"):
            validate_flexible_bin_spec_format(invalid_flex_bins)

        # Test with mismatched representatives
        valid_bins = {"col1": np.array([1.0, 2.0, 3.0])}
        wrong_reps = {"col1": np.array([1.5])}  # Wrong length

        with pytest.raises(ValueError, match="Expected 2 representatives, got 1"):
            validate_bins(valid_bins, wrong_reps)


if __name__ == "__main__":
    pytest.main([__file__])
