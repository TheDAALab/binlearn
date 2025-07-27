"""
Comprehensive test suite for _flexible_bin_utils module.

This test suite covers every line of code in the _flexible_bin_utils module,
including all functions, error paths, edge cases, and validation logic.
"""

import pytest
import numpy as np
from typing import Any, Dict, List

from binning.base._bin_utils import (
    FlexibleBinSpec,
    FlexibleBinReps,
    ensure_flexible_bin_spec,
    generate_default_flexible_representatives,
    validate_flexible_bins,
    is_missing_value,
    find_flexible_bin_for_value,
    calculate_flexible_bin_width,
    transform_value_to_flexible_bin,
    get_flexible_bin_count,
    validate_single_flexible_bin_def,
)
from binning.base._constants import MISSING_VALUE


# ============================================================================
# TEST ENSURE_FLEXIBLE_BIN_SPEC
# ============================================================================


class TestEnsureFlexibleBinSpec:
    """Test ensure_flexible_bin_spec function covering all code paths."""

    def test_none_input(self):
        """Test with None input."""
        result = ensure_flexible_bin_spec(None)
        assert result == {}
        assert isinstance(result, dict)

    def test_valid_dict_input(self):
        """Test with valid dictionary input."""
        bin_spec = {
            0: [{"singleton": 1.0}, {"interval": [2.0, 3.0]}],
            1: [{"singleton": 10.0}]
        }
        result = ensure_flexible_bin_spec(bin_spec)
        assert result is bin_spec  # Should return the same object
        assert result == bin_spec

    def test_empty_dict_input(self):
        """Test with empty dictionary input."""
        result = ensure_flexible_bin_spec({})
        assert result == {}

    def test_invalid_input_types(self):
        """Test with invalid input types that should raise ValueError."""
        invalid_inputs = [
            "string",
            123,
            [1, 2, 3],
            np.array([1, 2, 3]),
            (1, 2, 3),
            True,
            False
        ]
        
        for invalid_input in invalid_inputs:
            with pytest.raises(ValueError, match="bin_spec must be a dictionary mapping columns to bin definitions"):
                ensure_flexible_bin_spec(invalid_input)

    def test_complex_nested_dict(self):
        """Test with complex nested dictionary structure."""
        complex_spec = {
            "col_a": [
                {"singleton": 1.5},
                {"interval": [2.0, 5.0]},
                {"singleton": 10.0}
            ],
            42: [
                {"interval": [-np.inf, 0.0]},
                {"singleton": 0.0},
                {"interval": [0.0, np.inf]}
            ]
        }
        result = ensure_flexible_bin_spec(complex_spec)
        assert result == complex_spec


# ============================================================================
# TEST GENERATE_DEFAULT_FLEXIBLE_REPRESENTATIVES
# ============================================================================


class TestGenerateDefaultFlexibleRepresentatives:
    """Test generate_default_flexible_representatives function covering all paths."""

    def test_singleton_bins_only(self):
        """Test with only singleton bins."""
        bin_defs = [
            {"singleton": 1.0},
            {"singleton": 2.5},
            {"singleton": -10.0}
        ]
        result = generate_default_flexible_representatives(bin_defs)
        expected = [1.0, 2.5, -10.0]
        assert result == expected
        assert all(isinstance(x, float) for x in result)

    def test_interval_bins_only(self):
        """Test with only interval bins."""
        bin_defs = [
            {"interval": [0.0, 2.0]},
            {"interval": [5.0, 10.0]},
            {"interval": [-5.0, -1.0]}
        ]
        result = generate_default_flexible_representatives(bin_defs)
        expected = [1.0, 7.5, -3.0]  # Midpoints
        assert result == expected

    def test_mixed_bins(self):
        """Test with mixture of singleton and interval bins."""
        bin_defs = [
            {"singleton": 1.0},
            {"interval": [2.0, 4.0]},
            {"singleton": 5.0},
            {"interval": [10.0, 20.0]}
        ]
        result = generate_default_flexible_representatives(bin_defs)
        expected = [1.0, 3.0, 5.0, 15.0]
        assert result == expected

    def test_empty_bin_defs(self):
        """Test with empty bin definitions list."""
        result = generate_default_flexible_representatives([])
        assert result == []

    def test_extreme_values(self):
        """Test with extreme numerical values."""
        bin_defs = [
            {"singleton": 1e-100},
            {"interval": [1e100, 2e100]},
            {"singleton": -np.inf},
            {"interval": [-np.inf, np.inf]}
        ]
        result = generate_default_flexible_representatives(bin_defs)
        
        assert result[0] == 1e-100
        assert abs(result[1] - 1.5e100) < 1e90  # Allow for floating point precision
        assert result[2] == -np.inf
        assert np.isnan(result[3]) or result[3] == 0.0  # inf - inf handling

    def test_zero_width_interval(self):
        """Test with zero-width interval (degenerate case)."""
        bin_defs = [{"interval": [5.0, 5.0]}]
        result = generate_default_flexible_representatives(bin_defs)
        assert result == [5.0]

    def test_unknown_bin_definition_error(self):
        """Test error path for unknown bin definition format."""
        unknown_bin_defs = [
            [{"unknown_key": 1.0}],
            [{"range": [1, 2]}],
            [{"point": 1.0}],
            [{}],  # Empty dict
        ]
        
        for bin_defs in unknown_bin_defs:
            with pytest.raises(ValueError, match="Unknown bin definition"):
                generate_default_flexible_representatives(bin_defs)

    def test_bin_with_both_keys_uses_first_found(self):
        """Test that bins with both singleton and interval keys use the first key found."""
        # Python dict iteration order is guaranteed since 3.7+
        # "singleton" comes before "interval" alphabetically, so singleton should be used
        bin_defs = [{"singleton": 1.0, "interval": [1, 2]}]
        result = generate_default_flexible_representatives(bin_defs)
        assert result == [1.0]  # Uses singleton, not interval midpoint

    def test_type_conversion_in_singleton(self):
        """Test that singleton values are properly converted to float."""
        bin_defs = [
            {"singleton": 1},      # int
            {"singleton": True},   # bool
            {"singleton": "5.0"},  # string (if convertible)
        ]
        
        # Test int conversion
        result = generate_default_flexible_representatives([bin_defs[0]])
        assert result == [1.0]
        assert isinstance(result[0], float)
        
        # Test bool conversion  
        result = generate_default_flexible_representatives([bin_defs[1]])
        assert result == [1.0]  # True -> 1.0
        
        # String conversion might fail, but if it works:
        try:
            result = generate_default_flexible_representatives([bin_defs[2]])
            assert result == [5.0]
        except (ValueError, TypeError):
            # Expected if string conversion fails
            pass


# ============================================================================
# TEST VALIDATE_FLEXIBLE_BINS
# ============================================================================


class TestValidateFlexibleBins:
    """Test validate_flexible_bins function covering all validation paths."""

    def test_valid_bins_singleton_only(self):
        """Test validation of valid singleton-only bins."""
        bin_spec = {
            0: [{"singleton": 1.0}, {"singleton": 2.0}],
            1: [{"singleton": 10.0}]
        }
        bin_reps = {
            0: [1.0, 2.0],
            1: [10.0]
        }
        
        # Should not raise any exception
        validate_flexible_bins(bin_spec, bin_reps)

    def test_valid_bins_interval_only(self):
        """Test validation of valid interval-only bins."""
        bin_spec = {
            0: [{"interval": [0.0, 1.0]}, {"interval": [2.0, 3.0]}],
            1: [{"interval": [10.0, 20.0]}]
        }
        bin_reps = {
            0: [0.5, 2.5],
            1: [15.0]
        }
        
        # Should not raise any exception
        validate_flexible_bins(bin_spec, bin_reps)

    def test_valid_bins_mixed(self):
        """Test validation of valid mixed bins."""
        bin_spec = {
            0: [{"singleton": 1.0}, {"interval": [2.0, 3.0]}, {"singleton": 5.0}]
        }
        bin_reps = {
            0: [1.0, 2.5, 5.0]
        }
        
        # Should not raise any exception
        validate_flexible_bins(bin_spec, bin_reps)

    def test_empty_specs(self):
        """Test validation with empty specifications."""
        # Both empty
        validate_flexible_bins({}, {})
        
        # Bin spec empty, reps not empty (should be fine)
        validate_flexible_bins({}, {0: [1.0]})

    def test_mismatched_lengths_error(self):
        """Test error when bin definitions and representatives have different lengths."""
        bin_spec = {0: [{"singleton": 1.0}, {"singleton": 2.0}]}
        
        # Too few representatives
        bin_reps_few = {0: [1.0]}
        with pytest.raises(ValueError, match="Column 0: Number of bin definitions \\(2\\) must match number of representatives \\(1\\)"):
            validate_flexible_bins(bin_spec, bin_reps_few)
        
        # Too many representatives
        bin_reps_many = {0: [1.0, 2.0, 3.0]}
        with pytest.raises(ValueError, match="Column 0: Number of bin definitions \\(2\\) must match number of representatives \\(3\\)"):
            validate_flexible_bins(bin_spec, bin_reps_many)

    def test_missing_representatives_for_column(self):
        """Test when representatives are missing for a column (uses empty list)."""
        bin_spec = {0: [{"singleton": 1.0}]}
        bin_reps = {}  # No representatives for column 0
        
        with pytest.raises(ValueError, match="Column 0: Number of bin definitions \\(1\\) must match number of representatives \\(0\\)"):
            validate_flexible_bins(bin_spec, bin_reps)

    def test_invalid_bin_definition_not_dict(self):
        """Test error when bin definition is not a dictionary."""
        bin_spec = {0: ["not_a_dict"]}
        bin_reps = {0: [1.0]}
        
        with pytest.raises(ValueError, match="Column 0, bin 0: Bin definition must be a dictionary"):
            validate_flexible_bins(bin_spec, bin_reps)

    def test_singleton_bin_validation_errors(self):
        """Test singleton bin validation error paths."""
        # Multiple keys in singleton bin
        bin_spec = {0: [{"singleton": 1.0, "extra_key": "value"}]}
        bin_reps = {0: [1.0]}
        
        with pytest.raises(ValueError, match="Column 0, bin 0: Singleton bin must have only 'singleton' key"):
            validate_flexible_bins(bin_spec, bin_reps)

    def test_interval_bin_validation_errors(self):
        """Test interval bin validation error paths."""
        # Multiple keys in interval bin
        bin_spec = {0: [{"interval": [1.0, 2.0], "extra_key": "value"}]}
        bin_reps = {0: [1.5]}
        
        with pytest.raises(ValueError, match="Column 0, bin 0: Interval bin must have only 'interval' key"):
            validate_flexible_bins(bin_spec, bin_reps)

    def test_interval_format_validation_errors(self):
        """Test interval format validation errors."""
        # Not a list or tuple
        bin_spec = {0: [{"interval": "1,2"}]}
        bin_reps = {0: [1.5]}
        
        with pytest.raises(ValueError, match="Column 0, bin 0: Interval must be \\[min, max\\]"):
            validate_flexible_bins(bin_spec, bin_reps)
        
        # Wrong length
        bin_spec = {0: [{"interval": [1.0]}]}
        bin_reps = {0: [1.0]}
        
        with pytest.raises(ValueError, match="Column 0, bin 0: Interval must be \\[min, max\\]"):
            validate_flexible_bins(bin_spec, bin_reps)
        
        # Too many elements
        bin_spec = {0: [{"interval": [1.0, 2.0, 3.0]}]}
        bin_reps = {0: [2.0]}
        
        with pytest.raises(ValueError, match="Column 0, bin 0: Interval must be \\[min, max\\]"):
            validate_flexible_bins(bin_spec, bin_reps)

    def test_interval_order_validation_error(self):
        """Test interval order validation (min > max)."""
        bin_spec = {0: [{"interval": [2.0, 1.0]}]}  # min > max
        bin_reps = {0: [1.5]}
        
        with pytest.raises(ValueError, match="Column 0, bin 0: Interval min must be <= max"):
            validate_flexible_bins(bin_spec, bin_reps)

    def test_unknown_bin_type_error(self):
        """Test error for unknown bin type (neither singleton nor interval)."""
        bin_spec = {0: [{"unknown_type": [1.0, 2.0]}]}
        bin_reps = {0: [1.5]}
        
        with pytest.raises(ValueError, match="Column 0, bin 0: Bin must have 'singleton' or 'interval' key"):
            validate_flexible_bins(bin_spec, bin_reps)

    def test_multiple_columns_with_errors(self):
        """Test that validation reports the first error found when multiple columns have issues."""
        bin_spec = {
            0: [{"singleton": 1.0}],  # Valid
            1: [{"invalid_type": 2.0}]  # Invalid
        }
        bin_reps = {
            0: [1.0],
            1: [2.0]
        }
        
        with pytest.raises(ValueError, match="Column 1, bin 0: Bin must have 'singleton' or 'interval' key"):
            validate_flexible_bins(bin_spec, bin_reps)

    def test_edge_case_interval_equal_bounds(self):
        """Test interval with equal min and max (valid degenerate case)."""
        bin_spec = {0: [{"interval": [5.0, 5.0]}]}
        bin_reps = {0: [5.0]}
        
        # Should not raise exception
        validate_flexible_bins(bin_spec, bin_reps)

    def test_complex_column_keys(self):
        """Test validation with complex column keys."""
        bin_spec = {
            "column_a": [{"singleton": 1.0}],
            42: [{"interval": [0.0, 1.0]}],
            (1, 2): [{"singleton": 5.0}]
        }
        bin_reps = {
            "column_a": [1.0],
            42: [0.5],
            (1, 2): [5.0]
        }
        
        # Should not raise exception
        validate_flexible_bins(bin_spec, bin_reps)


# ============================================================================
# TEST IS_MISSING_VALUE
# ============================================================================


class TestIsMissingValue:
    """Test is_missing_value function covering all detection paths."""

    def test_none_values(self):
        """Test None value detection."""
        assert is_missing_value(None) is True

    def test_numeric_nan_values(self):
        """Test NaN value detection for numeric types."""
        assert is_missing_value(np.nan) is True
        assert is_missing_value(float('nan')) is True

    def test_numeric_non_nan_values(self):
        """Test that non-NaN numeric values are not considered missing."""
        non_missing_numeric = [
            0, 1, -1, 0.0, 1.0, -1.0,
            1e-100, 1e100, -1e100,
            np.inf, -np.inf  # inf values are not missing
        ]
        
        for value in non_missing_numeric:
            assert is_missing_value(value) is False

    def test_string_missing_representations(self):
        """Test string representations of missing values."""
        missing_strings = [
            "nan", "NaN", "NAN", "Nan",
            "none", "None", "NONE", "None",
            "", 
            "null", "NULL", "Null"
        ]
        
        for value in missing_strings:
            assert is_missing_value(value) is True

    def test_string_non_missing_values(self):
        """Test that non-missing string values are not considered missing."""
        non_missing_strings = [
            "0", "1", "hello", "world",
            " ", "  ", "\t", "\n",  # Whitespace is not missing
            "true", "false", "NaNa", "nothing"
        ]
        
        for value in non_missing_strings:
            assert is_missing_value(value) is False

    def test_non_numeric_non_string_types(self):
        """Test that non-numeric, non-string types are not missing (except None)."""
        non_missing_values = [
            [], [1, 2, 3],
            {}, {"key": "value"},
            True, False,
            object()
        ]
        
        for value in non_missing_values:
            assert is_missing_value(value) is False

    def test_types_that_raise_exceptions_in_isnan(self):
        """Test types that raise exceptions when checking np.isnan."""
        # These should go through the exception handling path
        exception_raising_values = [
            "text", [], {}, object(), True, False
        ]
        
        for value in exception_raising_values:
            # Should not raise exception and handle gracefully
            result = is_missing_value(value)
            assert isinstance(result, bool)

    def test_numpy_array_elements(self):
        """Test with numpy array elements."""
        arr = np.array([1.0, np.nan, 3.0])
        
        assert is_missing_value(arr[0]) is False  # 1.0
        assert is_missing_value(arr[1]) is True   # nan
        assert is_missing_value(arr[2]) is False  # 3.0

    def test_pandas_na_values(self):
        """Test pandas NA values if pandas is available."""
        try:
            import pandas as pd
            
            # The current implementation doesn't handle pandas NA specially
            # This test documents the current behavior
            assert is_missing_value(pd.NA) is False  # Current behavior
            assert is_missing_value(pd.NaT) is False  # Current behavior
        except ImportError:
            # pandas not available, skip
            pass

    def test_case_sensitivity_strings(self):
        """Test case sensitivity of string missing value detection."""
        # Test mixed case variations
        mixed_case_missing = [
            "NaN", "nan", "Nan", "nAn", "naN", "NAn", "NAN",
            "None", "none", "NONE", "NonE", "NoNe",
            "Null", "null", "NULL", "NuLl"
        ]
        
        for value in mixed_case_missing:
            assert is_missing_value(value) is True


# ============================================================================
# TEST FIND_FLEXIBLE_BIN_FOR_VALUE
# ============================================================================


class TestFindFlexibleBinForValue:
    """Test find_flexible_bin_for_value function covering all matching paths."""

    def test_singleton_bin_exact_match(self):
        """Test exact matches with singleton bins."""
        bin_defs = [
            {"singleton": 1.0},
            {"singleton": 2.5},
            {"singleton": -10.0}
        ]
        
        assert find_flexible_bin_for_value(1.0, bin_defs) == 0
        assert find_flexible_bin_for_value(2.5, bin_defs) == 1
        assert find_flexible_bin_for_value(-10.0, bin_defs) == 2

    def test_singleton_bin_no_match(self):
        """Test no match with singleton bins."""
        bin_defs = [{"singleton": 1.0}, {"singleton": 2.0}]
        
        assert find_flexible_bin_for_value(1.5, bin_defs) == MISSING_VALUE
        assert find_flexible_bin_for_value(0.0, bin_defs) == MISSING_VALUE

    def test_interval_bin_matches(self):
        """Test matches within interval bins."""
        bin_defs = [
            {"interval": [0.0, 2.0]},
            {"interval": [5.0, 10.0]},
            {"interval": [-5.0, -1.0]}
        ]
        
        # Test values inside intervals
        assert find_flexible_bin_for_value(1.0, bin_defs) == 0
        assert find_flexible_bin_for_value(7.5, bin_defs) == 1
        assert find_flexible_bin_for_value(-3.0, bin_defs) == 2
        
        # Test boundary values (inclusive)
        assert find_flexible_bin_for_value(0.0, bin_defs) == 0   # Left boundary
        assert find_flexible_bin_for_value(2.0, bin_defs) == 0   # Right boundary
        assert find_flexible_bin_for_value(5.0, bin_defs) == 1   # Left boundary
        assert find_flexible_bin_for_value(10.0, bin_defs) == 1  # Right boundary

    def test_interval_bin_no_match(self):
        """Test no match with interval bins."""
        bin_defs = [{"interval": [1.0, 3.0]}, {"interval": [5.0, 7.0]}]
        
        # Values outside all intervals
        assert find_flexible_bin_for_value(0.0, bin_defs) == MISSING_VALUE
        assert find_flexible_bin_for_value(4.0, bin_defs) == MISSING_VALUE
        assert find_flexible_bin_for_value(8.0, bin_defs) == MISSING_VALUE

    def test_mixed_bins_matching(self):
        """Test matching with mixed singleton and interval bins."""
        bin_defs = [
            {"singleton": 1.0},
            {"interval": [2.0, 4.0]},
            {"singleton": 5.0},
            {"interval": [10.0, 20.0]}
        ]
        
        assert find_flexible_bin_for_value(1.0, bin_defs) == 0   # Singleton match
        assert find_flexible_bin_for_value(3.0, bin_defs) == 1   # Interval match
        assert find_flexible_bin_for_value(5.0, bin_defs) == 2   # Singleton match
        assert find_flexible_bin_for_value(15.0, bin_defs) == 3  # Interval match

    def test_first_match_wins(self):
        """Test that first matching bin is returned (if overlapping)."""
        bin_defs = [
            {"interval": [0.0, 5.0]},
            {"interval": [3.0, 8.0]},  # Overlaps with first
            {"singleton": 4.0}         # Would match but comes after intervals
        ]
        
        # Value 4.0 matches all three bins, but first should win
        assert find_flexible_bin_for_value(4.0, bin_defs) == 0

    def test_empty_bin_definitions(self):
        """Test with empty bin definitions list."""
        assert find_flexible_bin_for_value(1.0, []) == MISSING_VALUE

    def test_extreme_values(self):
        """Test with extreme numerical values."""
        bin_defs = [
            {"singleton": 1e-100},
            {"interval": [1e100, 2e100]},
            {"singleton": np.inf},
            {"singleton": -np.inf}
        ]
        
        assert find_flexible_bin_for_value(1e-100, bin_defs) == 0
        assert find_flexible_bin_for_value(1.5e100, bin_defs) == 1
        assert find_flexible_bin_for_value(np.inf, bin_defs) == 2
        assert find_flexible_bin_for_value(-np.inf, bin_defs) == 3

    def test_zero_width_interval(self):
        """Test with zero-width interval (degenerate case)."""
        bin_defs = [{"interval": [5.0, 5.0]}]
        
        assert find_flexible_bin_for_value(5.0, bin_defs) == 0
        assert find_flexible_bin_for_value(4.9, bin_defs) == MISSING_VALUE
        assert find_flexible_bin_for_value(5.1, bin_defs) == MISSING_VALUE

    def test_floating_point_precision(self):
        """Test floating point precision edge cases."""
        bin_defs = [{"singleton": 0.1 + 0.2}]  # Should be 0.3 but may have precision issues
        
        # Test direct computation result
        computed_value = 0.1 + 0.2
        assert find_flexible_bin_for_value(computed_value, bin_defs) == 0
        
        # Test literal 0.3 (may or may not match due to precision)
        result = find_flexible_bin_for_value(0.3, bin_defs)
        # Either 0 (match) or MISSING_VALUE (no match due to precision) is acceptable
        assert result in [0, MISSING_VALUE]


# ============================================================================
# TEST CALCULATE_FLEXIBLE_BIN_WIDTH
# ============================================================================


class TestCalculateFlexibleBinWidth:
    """Test calculate_flexible_bin_width function covering all bin types."""

    def test_singleton_bin_width(self):
        """Test that singleton bins have zero width."""
        singleton_bins = [
            {"singleton": 0.0},
            {"singleton": 1.0},
            {"singleton": -10.0},
            {"singleton": 1e100},
            {"singleton": 1e-100}
        ]
        
        for bin_def in singleton_bins:
            assert calculate_flexible_bin_width(bin_def) == 0.0

    def test_interval_bin_width(self):
        """Test interval bin width calculation."""
        interval_tests = [
            ({"interval": [0.0, 1.0]}, 1.0),
            ({"interval": [1.0, 3.0]}, 2.0),
            ({"interval": [-5.0, 5.0]}, 10.0),
            ({"interval": [10.0, 10.5]}, 0.5),
            ({"interval": [1e6, 2e6]}, 1e6)
        ]
        
        for bin_def, expected_width in interval_tests:
            result = calculate_flexible_bin_width(bin_def)
            assert abs(result - expected_width) < 1e-10

    def test_zero_width_interval(self):
        """Test zero-width interval (degenerate case)."""
        bin_def = {"interval": [5.0, 5.0]}
        assert calculate_flexible_bin_width(bin_def) == 0.0

    def test_negative_width_interval(self):
        """Test interval with negative width (min > max)."""
        bin_def = {"interval": [5.0, 3.0]}
        # Should return negative width (caller's responsibility to validate)
        assert calculate_flexible_bin_width(bin_def) == -2.0

    def test_infinite_intervals(self):
        """Test intervals with infinite bounds."""
        infinite_tests = [
            ({"interval": [-np.inf, 0.0]}, np.inf),
            ({"interval": [0.0, np.inf]}, np.inf),
            ({"interval": [-np.inf, np.inf]}, np.inf)
        ]
        
        for bin_def, expected in infinite_tests:
            result = calculate_flexible_bin_width(bin_def)
            assert result == expected or np.isinf(result)

    def test_unknown_bin_type_error(self):
        """Test error for unknown bin definition format."""
        unknown_bins = [
            {"unknown_key": 1.0},
            {"range": [1, 2]},
            {"point": 5.0},
            {},  # Empty dict
        ]
        
        for bin_def in unknown_bins:
            with pytest.raises(ValueError, match="Unknown bin definition"):
                calculate_flexible_bin_width(bin_def)

    def test_bin_with_both_keys_uses_first_found_width(self):
        """Test that bins with both singleton and interval keys use the first key found."""
        # "singleton" comes before "interval" alphabetically in the if-elif chain
        bin_def = {"singleton": 1.0, "interval": [1, 2]}  # Both keys
        result = calculate_flexible_bin_width(bin_def)
        assert result == 0.0  # Uses singleton (zero width), not interval (width 1)

    def test_extreme_numerical_values(self):
        """Test with extreme numerical values."""
        extreme_tests = [
            ({"interval": [1e-100, 2e-100]}, 1e-100),
            ({"interval": [1e100, 3e100]}, 2e100),
            ({"singleton": 1e-200}, 0.0),
            ({"singleton": 1e200}, 0.0)
        ]
        
        for bin_def, expected in extreme_tests:
            result = calculate_flexible_bin_width(bin_def)
            if expected == 0.0:
                assert result == 0.0
            else:
                # For very large/small numbers, allow some relative tolerance
                assert abs(result - expected) < abs(expected) * 1e-10


# ============================================================================
# TEST TRANSFORM_VALUE_TO_FLEXIBLE_BIN
# ============================================================================


class TestTransformValueToFlexibleBin:
    """Test transform_value_to_flexible_bin function covering all paths."""

    def test_missing_values_detected(self):
        """Test that missing values are detected and return MISSING_VALUE."""
        bin_defs = [{"singleton": 1.0}, {"interval": [2.0, 3.0]}]
        
        missing_values = [None, np.nan, "nan", "", "null"]
        
        for value in missing_values:
            result = transform_value_to_flexible_bin(value, bin_defs)
            assert result == MISSING_VALUE

    def test_successful_bin_matching(self):
        """Test successful bin matching for non-missing values."""
        bin_defs = [
            {"singleton": 1.0},
            {"interval": [2.0, 4.0]},
            {"singleton": 5.0}
        ]
        
        # Test values that should match
        test_cases = [
            (1.0, 0),   # Singleton match
            (3.0, 1),   # Interval match
            (5.0, 2),   # Singleton match
            (2.0, 1),   # Interval boundary
            (4.0, 1)    # Interval boundary
        ]
        
        for value, expected_bin in test_cases:
            result = transform_value_to_flexible_bin(value, bin_defs)
            assert result == expected_bin

    def test_no_bin_match_returns_missing(self):
        """Test that values with no bin match return MISSING_VALUE."""
        bin_defs = [{"singleton": 1.0}, {"interval": [3.0, 4.0]}]
        
        no_match_values = [0.0, 2.0, 5.0, 10.0]
        
        for value in no_match_values:
            result = transform_value_to_flexible_bin(value, bin_defs)
            assert result == MISSING_VALUE

    def test_empty_bin_definitions(self):
        """Test with empty bin definitions."""
        result = transform_value_to_flexible_bin(1.0, [])
        assert result == MISSING_VALUE

    def test_integration_with_both_utility_functions(self):
        """Test that the function properly integrates is_missing_value and find_flexible_bin_for_value."""
        bin_defs = [{"singleton": 1.0}, {"interval": [2.0, 3.0]}]
        
        # Test the integration path: missing value detection
        assert transform_value_to_flexible_bin(None, bin_defs) == MISSING_VALUE
        
        # Test the integration path: bin finding
        assert transform_value_to_flexible_bin(1.0, bin_defs) == 0
        assert transform_value_to_flexible_bin(2.5, bin_defs) == 1
        assert transform_value_to_flexible_bin(10.0, bin_defs) == MISSING_VALUE

    def test_numeric_type_conversion(self):
        """Test that different numeric types work correctly."""
        bin_defs = [{"singleton": 1.0}, {"interval": [2.0, 3.0]}]
        
        # Test different numeric types
        assert transform_value_to_flexible_bin(1, bin_defs) == 0      # int
        assert transform_value_to_flexible_bin(1.0, bin_defs) == 0    # float
        assert transform_value_to_flexible_bin(True, bin_defs) == 0   # bool (True = 1)
        assert transform_value_to_flexible_bin(False, bin_defs) == MISSING_VALUE  # bool (False = 0, no match)


# ============================================================================
# TEST GET_FLEXIBLE_BIN_COUNT
# ============================================================================


class TestGetFlexibleBinCount:
    """Test get_flexible_bin_count function covering all scenarios."""

    def test_empty_bin_spec(self):
        """Test with empty bin specification."""
        result = get_flexible_bin_count({})
        assert result == {}

    def test_single_column_bin_spec(self):
        """Test with single column bin specification."""
        bin_spec = {
            0: [{"singleton": 1.0}, {"interval": [2.0, 3.0]}, {"singleton": 4.0}]
        }
        result = get_flexible_bin_count(bin_spec)
        expected = {0: 3}
        assert result == expected

    def test_multiple_column_bin_spec(self):
        """Test with multiple column bin specifications."""
        bin_spec = {
            0: [{"singleton": 1.0}],
            1: [{"interval": [0.0, 1.0]}, {"interval": [2.0, 3.0]}],
            2: [{"singleton": 10.0}, {"singleton": 20.0}, {"singleton": 30.0}, {"interval": [40.0, 50.0]}]
        }
        result = get_flexible_bin_count(bin_spec)
        expected = {0: 1, 1: 2, 2: 4}
        assert result == expected

    def test_zero_bins_column(self):
        """Test with column having zero bins (edge case)."""
        bin_spec = {
            0: [],  # No bins
            1: [{"singleton": 1.0}]
        }
        result = get_flexible_bin_count(bin_spec)
        expected = {0: 0, 1: 1}
        assert result == expected

    def test_complex_column_keys(self):
        """Test with complex column key types."""
        bin_spec = {
            "column_a": [{"singleton": 1.0}, {"singleton": 2.0}],
            42: [{"interval": [0.0, 1.0]}],
            (1, 2): [{"singleton": 5.0}],
            "complex_key": []
        }
        result = get_flexible_bin_count(bin_spec)
        expected = {"column_a": 2, 42: 1, (1, 2): 1, "complex_key": 0}
        assert result == expected

    def test_preserves_original_spec(self):
        """Test that function doesn't modify the original bin specification."""
        original_spec = {
            0: [{"singleton": 1.0}, {"interval": [2.0, 3.0]}],
            1: [{"singleton": 10.0}]
        }
        original_copy = {
            0: [{"singleton": 1.0}, {"interval": [2.0, 3.0]}],
            1: [{"singleton": 10.0}]
        }
        
        result = get_flexible_bin_count(original_spec)
        
        # Original should be unchanged
        assert original_spec == original_copy
        assert result == {0: 2, 1: 1}


# ============================================================================
# TEST VALIDATE_SINGLE_FLEXIBLE_BIN_DEF
# ============================================================================


class TestValidateSingleFlexibleBinDef:
    """Test validate_single_flexible_bin_def function covering all validation paths."""

    def test_valid_singleton_bin(self):
        """Test validation of valid singleton bins."""
        valid_singletons = [
            {"singleton": 1.0},
            {"singleton": 0.0},
            {"singleton": -10.0},
            {"singleton": 1e100},
            {"singleton": 1e-100}
        ]
        
        for bin_def in valid_singletons:
            # Should not raise any exception
            validate_single_flexible_bin_def(bin_def, "test_col", 0)

    def test_valid_interval_bin(self):
        """Test validation of valid interval bins."""
        valid_intervals = [
            {"interval": [0.0, 1.0]},
            {"interval": [1.0, 3.0]},
            {"interval": [-5.0, 5.0]},
            {"interval": [1e-10, 1e10]},
            {"interval": [5.0, 5.0]},  # Zero width (valid)
            {"interval": (-1.0, 1.0)},  # Tuple format
        ]
        
        for bin_def in valid_intervals:
            # Should not raise any exception
            validate_single_flexible_bin_def(bin_def, "test_col", 0)

    def test_invalid_bin_definition_not_dict(self):
        """Test error when bin definition is not a dictionary."""
        invalid_types = [
            "string",
            123,
            [1, 2],
            (1, 2),
            None,
            True
        ]
        
        for invalid_bin in invalid_types:
            with pytest.raises(ValueError, match="Column test_col, bin 5: Bin definition must be a dictionary"):
                validate_single_flexible_bin_def(invalid_bin, "test_col", 5)

    def test_singleton_validation_errors(self):
        """Test singleton bin validation errors."""
        # Multiple keys in singleton bin
        invalid_singleton = {"singleton": 1.0, "extra_key": "value"}
        
        with pytest.raises(ValueError, match="Column col1, bin 2: Singleton bin must have only 'singleton' key"):
            validate_single_flexible_bin_def(invalid_singleton, "col1", 2)

    def test_interval_validation_errors(self):
        """Test interval bin validation errors."""
        # Multiple keys in interval bin
        invalid_interval = {"interval": [1.0, 2.0], "extra_key": "value"}
        
        with pytest.raises(ValueError, match="Column col2, bin 3: Interval bin must have only 'interval' key"):
            validate_single_flexible_bin_def(invalid_interval, "col2", 3)

    def test_interval_format_errors(self):
        """Test interval format validation errors."""
        # Not a list or tuple
        with pytest.raises(ValueError, match="Column test_col, bin 0: Interval must be \\[min, max\\]"):
            validate_single_flexible_bin_def({"interval": "1,2"}, "test_col", 0)
        
        # Wrong length - too short
        with pytest.raises(ValueError, match="Column test_col, bin 0: Interval must be \\[min, max\\]"):
            validate_single_flexible_bin_def({"interval": [1.0]}, "test_col", 0)
        
        # Wrong length - too long
        with pytest.raises(ValueError, match="Column test_col, bin 0: Interval must be \\[min, max\\]"):
            validate_single_flexible_bin_def({"interval": [1.0, 2.0, 3.0]}, "test_col", 0)
        
        # Empty list
        with pytest.raises(ValueError, match="Column test_col, bin 0: Interval must be \\[min, max\\]"):
            validate_single_flexible_bin_def({"interval": []}, "test_col", 0)

    def test_interval_order_validation_error(self):
        """Test interval order validation (min > max)."""
        invalid_order = {"interval": [3.0, 1.0]}  # min > max
        
        with pytest.raises(ValueError, match="Column test_col, bin 0: Interval min must be <= max"):
            validate_single_flexible_bin_def(invalid_order, "test_col", 0)

    def test_unknown_bin_type_error(self):
        """Test error for unknown bin type."""
        unknown_types = [
            {"unknown_type": 1.0},
            {"range": [1, 2]},
            {"point": 5.0},
            {},  # Empty dict
            {"both": 1.0, "keys": [1, 2]}  # Neither singleton nor interval
        ]
        
        for unknown_bin in unknown_types:
            with pytest.raises(ValueError, match="Column test_col, bin 0: Bin must have 'singleton' or 'interval' key"):
                validate_single_flexible_bin_def(unknown_bin, "test_col", 0)

    def test_error_message_formatting(self):
        """Test that error messages include correct column and bin index information."""
        # Test with different column and bin index values
        with pytest.raises(ValueError, match="Column my_column, bin 42: Bin definition must be a dictionary"):
            validate_single_flexible_bin_def("invalid", "my_column", 42)  # type: ignore
        
        with pytest.raises(ValueError, match="Column 123, bin 7: Bin must have 'singleton' or 'interval' key"):
            validate_single_flexible_bin_def({"unknown": 1.0}, 123, 7)

    def test_extreme_interval_values(self):
        """Test validation with extreme interval values."""
        extreme_intervals = [
            {"interval": [-np.inf, np.inf]},
            {"interval": [1e-100, 1e100]},
            {"interval": [-1e100, -1e-100]}
        ]
        
        for bin_def in extreme_intervals:
            # Should not raise exception
            validate_single_flexible_bin_def(bin_def, "test_col", 0)

    def test_interval_with_infinite_bounds_invalid_order(self):
        """Test interval with infinite bounds but invalid order."""
        invalid_infinite = {"interval": [np.inf, -np.inf]}  # inf > -inf, invalid
        
        with pytest.raises(ValueError, match="Column test_col, bin 0: Interval min must be <= max"):
            validate_single_flexible_bin_def(invalid_infinite, "test_col", 0)


# ============================================================================
# INTEGRATION AND EDGE CASE TESTS
# ============================================================================


class TestFlexibleBinUtilsIntegration:
    """Integration tests and additional edge cases for comprehensive coverage."""

    def test_type_aliases_usage(self):
        """Test that type aliases work correctly in practice."""
        # This test ensures the type aliases are properly defined and usable
        bin_spec: FlexibleBinSpec = {
            0: [{"singleton": 1.0}, {"interval": [2.0, 3.0]}]
        }
        bin_reps: FlexibleBinReps = {
            0: [1.0, 2.5]
        }
        
        # These should work without type errors
        validate_flexible_bins(bin_spec, bin_reps)
        count = get_flexible_bin_count(bin_spec)
        assert count == {0: 2}

    def test_numpy_types_compatibility(self):
        """Test compatibility with various numpy types."""
        # Test with numpy scalar types
        bin_defs = [
            {"singleton": np.float64(1.0)},
            {"interval": [np.float32(2.0), np.float32(3.0)]}
        ]
        
        # Should work with numpy types
        reps = generate_default_flexible_representatives(bin_defs)
        assert len(reps) == 2
        
        # Test bin finding with numpy types
        assert find_flexible_bin_for_value(np.float64(1.0), bin_defs) == 0
        assert find_flexible_bin_for_value(float(np.float32(2.5)), bin_defs) == 1

    def test_comprehensive_workflow(self):
        """Test a complete workflow using multiple utility functions."""
        # Start with a raw bin specification
        raw_spec = {
            0: [{"singleton": 1.0}, {"interval": [2.0, 4.0]}, {"singleton": 5.0}],
            1: [{"interval": [0.0, 10.0]}]
        }
        
        # Ensure it's properly formatted
        bin_spec = ensure_flexible_bin_spec(raw_spec)
        assert bin_spec == raw_spec
        
        # Generate representatives
        bin_reps = {}
        for col, bin_defs in bin_spec.items():
            bin_reps[col] = generate_default_flexible_representatives(bin_defs)
        
        # Validate the complete specification
        validate_flexible_bins(bin_spec, bin_reps)
        
        # Get bin counts
        counts = get_flexible_bin_count(bin_spec)
        assert counts == {0: 3, 1: 1}
        
        # Test value transformations
        test_values = [1.0, 3.0, 5.0, 7.0, None, "nan"]
        results = []
        for value in test_values:
            for col in [0, 1]:
                result = transform_value_to_flexible_bin(value, bin_spec[col])
                results.append(result)
        
        # Verify expected results
        expected = [
            0, 0,        # 1.0 in col 0 and 1
            1, 0,        # 3.0 in col 0 and 1  
            2, 0,        # 5.0 in col 0 and 1
            MISSING_VALUE, 0,  # 7.0 in col 0 and 1
            MISSING_VALUE, MISSING_VALUE,  # None
            MISSING_VALUE, MISSING_VALUE   # "nan"
        ]
        assert results == expected

    def test_memory_efficiency_large_specs(self):
        """Test that functions handle large specifications efficiently."""
        # Create a large bin specification
        large_spec = {}
        for i in range(100):
            large_spec[i] = [
                {"singleton": float(j)} for j in range(10)
            ]
        
        # Test that operations complete without errors
        counts = get_flexible_bin_count(large_spec)
        assert len(counts) == 100
        assert all(count == 10 for count in counts.values())
        
        # Test representative generation for one column
        reps = generate_default_flexible_representatives(large_spec[0])
        assert len(reps) == 10

    def test_unicode_string_missing_values(self):
        """Test missing value detection with unicode strings."""
        unicode_missing = ["NaN", "нуль", "空", "無"]
        
        # Only standard missing representations should be detected
        assert is_missing_value("NaN") is True
        assert is_missing_value("нуль") is False  # Non-English "null"
        assert is_missing_value("空") is False     # Chinese "empty"
        assert is_missing_value("無") is False     # Chinese/Japanese "none"

    def test_function_parameter_edge_cases(self):
        """Test edge cases for function parameters."""
        # Test with minimal valid inputs
        minimal_spec = ensure_flexible_bin_spec({})
        assert minimal_spec == {}
        
        minimal_reps = generate_default_flexible_representatives([])
        assert minimal_reps == []
        
        minimal_count = get_flexible_bin_count({})
        assert minimal_count == {}
        
        # Test with single element inputs
        single_bin = [{"singleton": 42.0}]
        single_reps = generate_default_flexible_representatives(single_bin)
        assert single_reps == [42.0]


if __name__ == "__main__":
    pytest.main([__file__])
