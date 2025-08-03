"""Tests for flexible_binning module."""

import numpy as np
import pytest

from binlearn.utils.constants import MISSING_VALUE
from binlearn.utils.flexible_bin_operations import (
    calculate_flexible_bin_width,
    find_flexible_bin_for_value,
    generate_default_flexible_representatives,
    get_flexible_bin_count,
    is_missing_value,
    transform_value_to_flexible_bin,
    validate_flexible_bin_spec_format,
    validate_flexible_bins,
)


class TestValidateFlexibleBinSpecFormat:
    """Test validate_flexible_bin_spec_format function."""

    def test_validbin_spec_(self):
        """Test that valid bin specifications pass validation."""
        bin_spec = {
            "col1": [1, 2, 3],  # Singleton bins
            "col2": [(0, 1), (1, 2), (2, 3)],  # Interval bins
            "col3": [1, (2, 3), 4],  # Mixed bins
        }
        # Should not raise any exception
        validate_flexible_bin_spec_format(bin_spec)

    def test_invalidbin_spec__not_dict(self):
        """Test that non-dict bin specs raise ValueError."""
        with pytest.raises(ValueError, match="bin_spec must be a dictionary"):
            validate_flexible_bin_spec_format([1, 2, 3])  # type: ignore

    def test_invalid_bin_defs_not_list(self):
        """Test that non-list bin definitions raise ValueError."""
        bin_spec = {"col1": 123}  # Should be list/tuple
        with pytest.raises(ValueError, match="must be a list or tuple"):
            validate_flexible_bin_spec_format(bin_spec)  # type: ignore

    def test_empty_bin_defs(self):
        """Test that empty bin definitions raise ValueError."""
        bin_spec = {"col1": []}
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_flexible_bin_spec_format(bin_spec)

    def test_invalid_interval_length(self):
        """Test that intervals with wrong length raise ValueError."""
        bin_spec = {"col1": [(1, 2, 3)]}  # Should be (min, max)
        with pytest.raises(ValueError, match="Interval must be \\(min, max\\)"):
            validate_flexible_bin_spec_format(bin_spec)

    def test_invalid_interval_types(self):
        """Test that intervals with non-numeric values raise ValueError."""
        bin_spec = {"col1": [("a", "b")]}
        with pytest.raises(ValueError, match="Interval values must be numeric"):
            validate_flexible_bin_spec_format(bin_spec)

    def test_invalid_interval_ordering(self):
        """Test that intervals with min >= max raise ValueError."""
        bin_spec = {"col1": [(2, 1)]}  # min > max
        with pytest.raises(ValueError, match="must be < max"):
            validate_flexible_bin_spec_format(bin_spec)

    def test_invalid_interval_equal_bounds(self):
        """Test that intervals with min == max raise ValueError."""
        bin_spec = {"col1": [(2, 2)]}  # min == max
        with pytest.raises(ValueError, match="must be < max"):
            validate_flexible_bin_spec_format(bin_spec)

    def test_non_strict_mode_allows_equal_bounds(self):
        """Test that non-strict mode allows equal bounds but not reversed."""
        # Equal bounds should be allowed in non-strict mode
        bin_spec = {"col1": [(2, 2)]}  # min == max
        # Should not raise any exception in non-strict mode
        validate_flexible_bin_spec_format(bin_spec, strict=False)

        # But reversed bounds (min > max) should still raise error in non-strict mode
        bin_spec_reversed = {"col1": [(3, 1)]}  # min > max
        with pytest.raises(ValueError, match="must be <= max"):
            validate_flexible_bin_spec_format(bin_spec_reversed, strict=False)

    def test_invalid_bin_type(self):
        """Test that invalid bin types raise ValueError."""
        bin_spec = {"col1": ["invalid"]}
        with pytest.raises(ValueError, match="must be either a numeric scalar"):
            validate_flexible_bin_spec_format(bin_spec)

    def test_finite_bounds_check_disabled(self):
        """Test that infinite bounds are allowed when check_finite_bounds=False."""
        bin_spec = {"col1": [float("inf"), (-float("inf"), 0), (0, float("inf"))]}
        # Should not raise exception with default check_finite_bounds=False
        validate_flexible_bin_spec_format(bin_spec, check_finite_bounds=False)

    def test_finite_bounds_check_enabled_singleton(self):
        """Test that infinite singleton values raise ValueError when check_finite_bounds=True."""
        bin_spec = {"col1": [float("inf")]}
        with pytest.raises(ValueError, match="Singleton value must be finite"):
            validate_flexible_bin_spec_format(bin_spec, check_finite_bounds=True)

    def test_finite_bounds_check_enabled_interval(self):
        """Test that infinite interval bounds raise ValueError when check_finite_bounds=True."""
        bin_spec = {"col1": [(0, float("inf"))]}
        with pytest.raises(ValueError, match="Interval bounds must be finite"):
            validate_flexible_bin_spec_format(bin_spec, check_finite_bounds=True)

    def test_finite_bounds_valid(self):
        """Test that finite bounds pass validation when check_finite_bounds=True."""
        bin_spec = {
            "col1": [1.5, (0.0, 2.0), 3.0],
        }
        # Should not raise exception
        validate_flexible_bin_spec_format(bin_spec, check_finite_bounds=True)


class TestGenerateDefaultFlexibleRepresentatives:
    """Test generate_default_flexible_representatives function."""

    def test_singleton_bins(self):
        """Test with singleton bins."""
        bin_defs = [1, 2.5, 10]  # Simplified format: just the values
        result = generate_default_flexible_representatives(bin_defs)
        expected = [1.0, 2.5, 10.0]
        assert result == expected

    def test_interval_bins(self):
        """Test with interval bins."""
        bin_defs = [(0, 2), (3, 5), (-1, 1)]  # Simplified format: tuples
        result = generate_default_flexible_representatives(bin_defs)
        expected = [1.0, 4.0, 0.0]  # Midpoints
        assert result == expected

    def test_mixed_bins(self):
        """Test with mixed singleton and interval bins."""
        bin_defs = [1, (2, 4), 5, (6, 8)]  # Mixed format
        result = generate_default_flexible_representatives(bin_defs)
        expected = [1.0, 3.0, 5.0, 7.0]
        assert result == expected

    def test_empty_bin_defs(self):
        """Test with empty bin definitions."""
        result = generate_default_flexible_representatives([])
        assert not result

    def test_invalid_bin_def(self):
        """Test with invalid bin definition."""
        bin_defs = [{"unknown_key": 1}]  # Old format should fail
        with pytest.raises(ValueError, match="Unknown bin definition"):
            generate_default_flexible_representatives(bin_defs)

    def test_negative_interval(self):
        """Test with negative interval values."""
        bin_defs = [(-5, -2)]  # Simplified format
        result = generate_default_flexible_representatives(bin_defs)
        expected = [-3.5]
        assert result == expected


class TestValidateFlexibleBins:
    """Test validate_flexible_bins function."""

    def test_valid_bins(self):
        """Test with valid bin specifications."""
        bin_spec = {"col1": [1, (2, 3)], "col2": [5]}  # New simplified format
        bin_reps = {"col1": [1.0, 2.5], "col2": [5.0]}
        # Should not raise any exception
        validate_flexible_bins(bin_spec, bin_reps)

    def test_mismatched_lengths(self):
        """Test with mismatched number of bins and representatives."""
        bin_spec = {"col1": [1, (2, 3)]}  # New format
        bin_reps = {"col1": [1.0]}  # Only one representative for two bins
        with pytest.raises(ValueError, match="Number of bin definitions.*must match"):
            validate_flexible_bins(bin_spec, bin_reps)

    def test_missing_column_in_reps(self):
        """Test with missing column in representatives."""
        bin_spec = {"col1": [1]}  # New format
        bin_reps = {}  # Empty representatives
        with pytest.raises(ValueError, match="Number of bin definitions.*must match"):
            validate_flexible_bins(bin_spec, bin_reps)

    def test_invalid_bin_definition_format(self):
        """Test with invalid bin definition format."""
        bin_spec = {"col1": [1, "invalid"]}  # Invalid string bin
        bin_reps = {"col1": [1.0, 2.0]}
        with pytest.raises(
            ValueError,
            match="Bin must be either a numeric scalar \\(singleton\\) or tuple \\(interval\\)",
        ):
            validate_flexible_bins(bin_spec, bin_reps)

    def test_invalid_interval_format(self):
        """Test with invalid interval format."""
        bin_spec = {"col1": [(1,)]}  # Single value tuple instead of (min, max)
        bin_reps = {"col1": [1.0]}
        with pytest.raises(ValueError, match="Interval must be \\(min, max\\)"):
            validate_flexible_bins(bin_spec, bin_reps)

    def test_invalid_interval_order(self):
        """Test with invalid interval order (min > max)."""
        bin_spec = {"col1": [(3, 1)]}  # min > max
        bin_reps = {"col1": [2.0]}
        with pytest.raises(ValueError, match="Interval min \\(3\\) must be < max \\(1\\)"):
            validate_flexible_bins(bin_spec, bin_reps)

    def test_invalid_interval_values(self):
        """Test with non-numeric interval values."""
        bin_spec = {"col1": [("a", "b")]}  # Non-numeric values
        bin_reps = {"col1": [1.0]}
        with pytest.raises(ValueError, match="Interval values must be numeric"):
            validate_flexible_bins(bin_spec, bin_reps)


class TestIsMissingValue:
    """Test is_missing_value function."""

    def test_nan_values(self):
        """Test with NaN values."""
        assert is_missing_value(float("nan")) is True
        assert is_missing_value(np.nan) is True

    def test_numeric_values(self):
        """Test with valid numeric values."""
        assert is_missing_value(0) is False
        assert is_missing_value(1.5) is False
        assert is_missing_value(-10) is False
        assert is_missing_value(float("inf")) is False
        assert is_missing_value(float("-inf")) is False

    def test_none_value(self):
        """Test with None value."""
        assert is_missing_value(None) is True

    def test_string_values(self):
        """Test with string values - now considered non-missing (not supported in
        flexible binning)."""
        assert is_missing_value("string") is False
        assert is_missing_value("1.5") is False
        assert is_missing_value("") is False

    def test_non_convertible_types(self):
        """Test with non-convertible types - now considered non-missing (not supported in
        flexible binning)."""
        assert is_missing_value([1, 2, 3]) is False
        assert is_missing_value({"key": "value"}) is False
        assert is_missing_value(object()) is False

    def test_boolean_values(self):
        """Test with boolean values - now considered non-missing (not supported in
        flexible binning)."""
        assert is_missing_value(True) is False
        assert is_missing_value(False) is False


class TestFindFlexibleBinForValue:
    """Test find_flexible_bin_for_value function."""

    def test_singleton_match(self):
        """Test finding value in singleton bins."""
        bin_defs = [1, 2, 3]  # New simplified format
        assert find_flexible_bin_for_value(1, bin_defs) == 0
        assert find_flexible_bin_for_value(2, bin_defs) == 1
        assert find_flexible_bin_for_value(3, bin_defs) == 2

    def test_interval_match(self):
        """Test finding value in interval bins."""
        bin_defs = [(0, 2), (2, 4), (4, 6)]  # New simplified format
        assert find_flexible_bin_for_value(1.0, bin_defs) == 0
        assert find_flexible_bin_for_value(2.0, bin_defs) == 0  # First interval at boundary
        assert find_flexible_bin_for_value(3.0, bin_defs) == 1
        assert find_flexible_bin_for_value(4.0, bin_defs) == 1  # Second interval at boundary

    def test_mixed_bins(self):
        """Test with mixed singleton and interval bins."""
        bin_defs = [1, (2, 4), 5]  # New simplified format
        assert find_flexible_bin_for_value(1, bin_defs) == 0
        assert find_flexible_bin_for_value(3, bin_defs) == 1
        assert find_flexible_bin_for_value(5, bin_defs) == 2

    def test_no_match(self):
        """Test when value doesn't match any bin."""
        bin_defs = [1, (2, 4)]  # New simplified format
        assert find_flexible_bin_for_value(0, bin_defs) == MISSING_VALUE
        assert find_flexible_bin_for_value(1.5, bin_defs) == MISSING_VALUE
        assert find_flexible_bin_for_value(5, bin_defs) == MISSING_VALUE

    def test_empty_bin_defs(self):
        """Test with empty bin definitions."""
        assert find_flexible_bin_for_value(1, []) == MISSING_VALUE

    def test_interval_boundaries(self):
        """Test interval boundary conditions."""
        bin_defs = [(1, 3)]  # New simplified format
        assert find_flexible_bin_for_value(1.0, bin_defs) == 0  # Left boundary
        assert find_flexible_bin_for_value(3.0, bin_defs) == 0  # Right boundary
        assert find_flexible_bin_for_value(0.9, bin_defs) == MISSING_VALUE  # Just outside left
        assert find_flexible_bin_for_value(3.1, bin_defs) == MISSING_VALUE  # Just outside right

    def test_edge_cases_branch_coverage(self):
        """Test edge cases to cover remaining branches."""
        # Test tuple with wrong length (not 2) - should be skipped
        bin_defs = [(1, 2, 3)]  # Wrong length tuple
        assert find_flexible_bin_for_value(2, bin_defs) == MISSING_VALUE

        # Test non-numeric value against interval (should not match)
        bin_defs = [(1, 3)]
        assert find_flexible_bin_for_value("string", bin_defs) == MISSING_VALUE

        # Test mixed bin types to ensure all branches are covered
        bin_defs = [5, (1, 3), "invalid_bin_def"]  # Mix of valid and invalid
        assert find_flexible_bin_for_value(5, bin_defs) == 0  # Matches singleton
        assert find_flexible_bin_for_value(2, bin_defs) == 1  # Matches interval
        assert find_flexible_bin_for_value(10, bin_defs) == MISSING_VALUE  # No match


class TestCalculateFlexibleBinWidth:
    """Test calculate_flexible_bin_width function."""

    def test_singleton_width(self):
        """Test width of singleton bins."""
        bin_def = 5  # New simplified format: just the scalar
        assert calculate_flexible_bin_width(bin_def) == 0.0

    def test_interval_width(self):
        """Test width of interval bins."""
        bin_def = (2, 5)  # New simplified format: tuple
        assert calculate_flexible_bin_width(bin_def) == 3.0

    def test_zero_width_interval(self):
        """Test zero-width interval."""
        bin_def = (3, 3)  # New simplified format: tuple
        assert calculate_flexible_bin_width(bin_def) == 0.0

    def test_negative_interval(self):
        """Test interval with negative values."""
        bin_def = (-5, -2)  # New simplified format: tuple
        assert calculate_flexible_bin_width(bin_def) == 3.0

    def test_invalid_bin_def(self):
        """Test with invalid bin definition."""
        bin_def = {"unknown": 1}  # Old dict format should fail
        with pytest.raises(ValueError, match="Unknown bin definition"):
            calculate_flexible_bin_width(bin_def)

    def test_large_interval(self):
        """Test with large interval."""
        bin_def = (0, 1000)  # New simplified format: tuple
        assert calculate_flexible_bin_width(bin_def) == 1000.0


class TestTransformValueToFlexibleBin:
    """Test transform_value_to_flexible_bin function."""

    def test_valid_numeric_values(self):
        """Test with valid numeric values."""
        bin_defs = [1, (2, 4)]  # New simplified format
        assert transform_value_to_flexible_bin(1, bin_defs) == 0
        assert transform_value_to_flexible_bin(3, bin_defs) == 1

    def test_missing_values(self):
        """Test with missing values."""
        bin_defs = [1]  # New simplified format
        assert transform_value_to_flexible_bin(float("nan"), bin_defs) == MISSING_VALUE
        assert transform_value_to_flexible_bin(None, bin_defs) == MISSING_VALUE
        assert transform_value_to_flexible_bin("string", bin_defs) == MISSING_VALUE

    def test_no_matching_bin(self):
        """Test when value doesn't match any bin."""
        bin_defs = [1]  # New simplified format
        assert transform_value_to_flexible_bin(2, bin_defs) == MISSING_VALUE

    def test_string_numeric_conversion(self):
        """Test that string numbers are treated as missing."""
        bin_defs = [1]  # New simplified format
        assert transform_value_to_flexible_bin("1", bin_defs) == MISSING_VALUE

    def test_boolean_conversion(self):
        """Test boolean value conversion."""
        bin_defs = [1, 0]  # New simplified format
        assert transform_value_to_flexible_bin(True, bin_defs) == 0  # True -> 1.0
        assert transform_value_to_flexible_bin(False, bin_defs) == 1  # False -> 0.0


class TestGetFlexibleBinCount:
    """Test get_flexible_bin_count function."""

    def test_single_column(self):
        """Test with single column."""
        bin_spec = {"col1": [1, (2, 4)]}  # New simplified format
        result = get_flexible_bin_count(bin_spec)
        assert result == {"col1": 2}

    def test_multiple_columns(self):
        """Test with multiple columns."""
        bin_spec = {
            "col1": [1],  # New simplified format
            "col2": [(2, 4), 5, (6, 8)],  # New simplified format
            "col3": [],
        }
        result = get_flexible_bin_count(bin_spec)
        expected = {"col1": 1, "col2": 3, "col3": 0}
        assert result == expected

    def test_empty_spec(self):
        """Test with empty specification."""
        result = get_flexible_bin_count({})
        assert result == {}

    def test_empty_columns(self):
        """Test with columns having no bins."""
        bin_spec = {"col1": [], "col2": []}
        result = get_flexible_bin_count(bin_spec)
        assert result == {"col1": 0, "col2": 0}
