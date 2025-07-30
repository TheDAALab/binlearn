"""Tests for flexible_binning module."""

import pytest
import numpy as np
from binning.utils.flexible_binning import (
    ensure_flexible_bin_spec,
    generate_default_flexible_representatives,
    validate_flexible_bins,
    is_missing_value,
    find_flexible_bin_for_value,
    calculate_flexible_bin_width,
    transform_value_to_flexible_bin,
    get_flexible_bin_count,
)
from binning.utils.constants import MISSING_VALUE


class TestEnsureFlexibleBinSpec:
    """Test ensure_flexible_bin_spec function."""

    def test_none_input(self):
        """Test with None input."""
        result = ensure_flexible_bin_spec(None)
        assert result == {}

    def test_dict_input(self):
        """Test with valid dictionary input."""
        bin_spec = {"col1": [{"singleton": 1}, {"interval": [2, 3]}], "col2": [{"singleton": 5}]}
        result = ensure_flexible_bin_spec(bin_spec)
        assert result == bin_spec

    def test_invalid_input_type(self):
        """Test with invalid input type."""
        with pytest.raises(ValueError, match="bin_spec must be a dictionary"):
            ensure_flexible_bin_spec([1, 2, 3])

    def test_string_input(self):
        """Test with string input."""
        with pytest.raises(ValueError, match="bin_spec must be a dictionary"):
            ensure_flexible_bin_spec("invalid")

    def test_empty_dict(self):
        """Test with empty dictionary."""
        result = ensure_flexible_bin_spec({})
        assert not result


class TestGenerateDefaultFlexibleRepresentatives:
    """Test generate_default_flexible_representatives function."""

    def test_singleton_bins(self):
        """Test with singleton bins."""
        bin_defs = [{"singleton": 1}, {"singleton": 2.5}, {"singleton": 10}]
        result = generate_default_flexible_representatives(bin_defs)
        expected = [1.0, 2.5, 10.0]
        assert result == expected

    def test_interval_bins(self):
        """Test with interval bins."""
        bin_defs = [{"interval": [0, 2]}, {"interval": [3, 5]}, {"interval": [-1, 1]}]
        result = generate_default_flexible_representatives(bin_defs)
        expected = [1.0, 4.0, 0.0]  # Midpoints
        assert result == expected

    def test_mixed_bins(self):
        """Test with mixed singleton and interval bins."""
        bin_defs = [{"singleton": 1}, {"interval": [2, 4]}, {"singleton": 5}, {"interval": [6, 8]}]
        result = generate_default_flexible_representatives(bin_defs)
        expected = [1.0, 3.0, 5.0, 7.0]
        assert result == expected

    def test_empty_bin_defs(self):
        """Test with empty bin definitions."""
        result = generate_default_flexible_representatives([])
        assert not result

    def test_invalid_bin_def(self):
        """Test with invalid bin definition."""
        bin_defs = [{"unknown_key": 1}]
        with pytest.raises(ValueError, match="Unknown bin definition"):
            generate_default_flexible_representatives(bin_defs)

    def test_negative_interval(self):
        """Test with negative interval values."""
        bin_defs = [{"interval": [-5, -2]}]
        result = generate_default_flexible_representatives(bin_defs)
        expected = [-3.5]
        assert result == expected


class TestValidateFlexibleBins:
    """Test validate_flexible_bins function."""

    def test_valid_bins(self):
        """Test with valid bin specifications."""
        bin_spec = {"col1": [{"singleton": 1}, {"interval": [2, 3]}], "col2": [{"singleton": 5}]}
        bin_reps = {"col1": [1.0, 2.5], "col2": [5.0]}
        # Should not raise any exception
        validate_flexible_bins(bin_spec, bin_reps)

    def test_mismatched_lengths(self):
        """Test with mismatched number of bins and representatives."""
        bin_spec = {"col1": [{"singleton": 1}, {"interval": [2, 3]}]}
        bin_reps = {"col1": [1.0]}  # Only one representative for two bins
        with pytest.raises(ValueError, match="Number of bin definitions.*must match"):
            validate_flexible_bins(bin_spec, bin_reps)

    def test_missing_column_in_reps(self):
        """Test with missing column in representatives."""
        bin_spec = {"col1": [{"singleton": 1}]}
        bin_reps = {}  # Empty representatives
        with pytest.raises(ValueError, match="Number of bin definitions.*must match"):
            validate_flexible_bins(bin_spec, bin_reps)

    def test_invalid_bin_definition_format(self):
        """Test with invalid bin definition format."""
        bin_spec = {"col1": [{"singleton": 1}, "invalid"]}
        bin_reps = {"col1": [1.0, 2.0]}
        with pytest.raises(ValueError, match="Bin definition must be a dictionary"):
            validate_flexible_bins(bin_spec, bin_reps)

    def test_singleton_with_extra_keys(self):
        """Test singleton bin with extra keys."""
        bin_spec = {"col1": [{"singleton": 1, "extra": "key"}]}
        bin_reps = {"col1": [1.0]}
        with pytest.raises(ValueError, match="Singleton bin must have only 'singleton' key"):
            validate_flexible_bins(bin_spec, bin_reps)

    def test_interval_with_extra_keys(self):
        """Test interval bin with extra keys."""
        bin_spec = {"col1": [{"interval": [1, 2], "extra": "key"}]}
        bin_reps = {"col1": [1.5]}
        with pytest.raises(ValueError, match="Interval bin must have only 'interval' key"):
            validate_flexible_bins(bin_spec, bin_reps)

    def test_invalid_interval_format(self):
        """Test with invalid interval format."""
        bin_spec = {"col1": [{"interval": [1]}]}  # Single value instead of [min, max]
        bin_reps = {"col1": [1.0]}
        with pytest.raises(ValueError, match="Interval must be \\[min, max\\]"):
            validate_flexible_bins(bin_spec, bin_reps)

    def test_invalid_interval_order(self):
        """Test with invalid interval order (min > max)."""
        bin_spec = {"col1": [{"interval": [3, 1]}]}  # min > max
        bin_reps = {"col1": [2.0]}
        with pytest.raises(ValueError, match="Interval min must be <= max"):
            validate_flexible_bins(bin_spec, bin_reps)

    def test_unknown_bin_type(self):
        """Test with unknown bin type."""
        bin_spec = {"col1": [{"unknown_type": 1}]}
        bin_reps = {"col1": [1.0]}
        with pytest.raises(ValueError, match="Bin must have 'singleton' or 'interval' key"):
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
        """Test with string values."""
        assert is_missing_value("string") is True
        assert is_missing_value("1.5") is False  # String numbers can be converted to float
        assert is_missing_value("") is True

    def test_non_convertible_types(self):
        """Test with non-convertible types."""
        assert is_missing_value([1, 2, 3]) is True
        assert is_missing_value({"key": "value"}) is True
        assert is_missing_value(object()) is True

    def test_boolean_values(self):
        """Test with boolean values."""
        assert is_missing_value(True) is False  # True converts to 1.0
        assert is_missing_value(False) is False  # False converts to 0.0


class TestFindFlexibleBinForValue:
    """Test find_flexible_bin_for_value function."""

    def test_singleton_match(self):
        """Test finding value in singleton bins."""
        bin_defs = [{"singleton": 1}, {"singleton": 2}, {"singleton": 3}]
        assert find_flexible_bin_for_value(1, bin_defs) == 0
        assert find_flexible_bin_for_value(2, bin_defs) == 1
        assert find_flexible_bin_for_value(3, bin_defs) == 2

    def test_interval_match(self):
        """Test finding value in interval bins."""
        bin_defs = [{"interval": [0, 2]}, {"interval": [2, 4]}, {"interval": [4, 6]}]
        assert find_flexible_bin_for_value(1.0, bin_defs) == 0
        assert find_flexible_bin_for_value(2.0, bin_defs) == 0  # First interval at boundary
        assert find_flexible_bin_for_value(3.0, bin_defs) == 1
        assert find_flexible_bin_for_value(4.0, bin_defs) == 1  # Second interval at boundary

    def test_mixed_bins(self):
        """Test with mixed singleton and interval bins."""
        bin_defs = [{"singleton": 1}, {"interval": [2, 4]}, {"singleton": 5}]
        assert find_flexible_bin_for_value(1, bin_defs) == 0
        assert find_flexible_bin_for_value(3, bin_defs) == 1
        assert find_flexible_bin_for_value(5, bin_defs) == 2

    def test_no_match(self):
        """Test when value doesn't match any bin."""
        bin_defs = [{"singleton": 1}, {"interval": [2, 4]}]
        assert find_flexible_bin_for_value(0, bin_defs) == MISSING_VALUE
        assert find_flexible_bin_for_value(1.5, bin_defs) == MISSING_VALUE
        assert find_flexible_bin_for_value(5, bin_defs) == MISSING_VALUE

    def test_empty_bin_defs(self):
        """Test with empty bin definitions."""
        assert find_flexible_bin_for_value(1, []) == MISSING_VALUE

    def test_interval_boundaries(self):
        """Test interval boundary conditions."""
        bin_defs = [{"interval": [1, 3]}]
        assert find_flexible_bin_for_value(1.0, bin_defs) == 0  # Left boundary
        assert find_flexible_bin_for_value(3.0, bin_defs) == 0  # Right boundary
        assert find_flexible_bin_for_value(0.9, bin_defs) == MISSING_VALUE  # Just outside left
        assert find_flexible_bin_for_value(3.1, bin_defs) == MISSING_VALUE  # Just outside right


class TestCalculateFlexibleBinWidth:
    """Test calculate_flexible_bin_width function."""

    def test_singleton_width(self):
        """Test width of singleton bins."""
        bin_def = {"singleton": 5}
        assert calculate_flexible_bin_width(bin_def) == 0.0

    def test_interval_width(self):
        """Test width of interval bins."""
        bin_def = {"interval": [2, 5]}
        assert calculate_flexible_bin_width(bin_def) == 3.0

    def test_zero_width_interval(self):
        """Test zero-width interval."""
        bin_def = {"interval": [3, 3]}
        assert calculate_flexible_bin_width(bin_def) == 0.0

    def test_negative_interval(self):
        """Test interval with negative values."""
        bin_def = {"interval": [-5, -2]}
        assert calculate_flexible_bin_width(bin_def) == 3.0

    def test_invalid_bin_def(self):
        """Test with invalid bin definition."""
        bin_def = {"unknown": 1}
        with pytest.raises(ValueError, match="Unknown bin definition"):
            calculate_flexible_bin_width(bin_def)

    def test_large_interval(self):
        """Test with large interval."""
        bin_def = {"interval": [0, 1000]}
        assert calculate_flexible_bin_width(bin_def) == 1000.0


class TestTransformValueToFlexibleBin:
    """Test transform_value_to_flexible_bin function."""

    def test_valid_numeric_values(self):
        """Test with valid numeric values."""
        bin_defs = [{"singleton": 1}, {"interval": [2, 4]}]
        assert transform_value_to_flexible_bin(1, bin_defs) == 0
        assert transform_value_to_flexible_bin(3, bin_defs) == 1

    def test_missing_values(self):
        """Test with missing values."""
        bin_defs = [{"singleton": 1}]
        assert transform_value_to_flexible_bin(float("nan"), bin_defs) == MISSING_VALUE
        assert transform_value_to_flexible_bin(None, bin_defs) == MISSING_VALUE
        assert transform_value_to_flexible_bin("string", bin_defs) == MISSING_VALUE

    def test_no_matching_bin(self):
        """Test when value doesn't match any bin."""
        bin_defs = [{"singleton": 1}]
        assert transform_value_to_flexible_bin(2, bin_defs) == MISSING_VALUE

    def test_string_numeric_conversion(self):
        """Test that string numbers are treated as missing."""
        bin_defs = [{"singleton": 1}]
        assert transform_value_to_flexible_bin("1", bin_defs) == MISSING_VALUE

    def test_boolean_conversion(self):
        """Test boolean value conversion."""
        bin_defs = [{"singleton": 1}, {"singleton": 0}]
        assert transform_value_to_flexible_bin(True, bin_defs) == 0  # True -> 1.0
        assert transform_value_to_flexible_bin(False, bin_defs) == 1  # False -> 0.0


class TestGetFlexibleBinCount:
    """Test get_flexible_bin_count function."""

    def test_single_column(self):
        """Test with single column."""
        bin_spec = {"col1": [{"singleton": 1}, {"interval": [2, 4]}]}
        result = get_flexible_bin_count(bin_spec)
        assert result == {"col1": 2}

    def test_multiple_columns(self):
        """Test with multiple columns."""
        bin_spec = {
            "col1": [{"singleton": 1}],
            "col2": [{"interval": [2, 4]}, {"singleton": 5}, {"interval": [6, 8]}],
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
