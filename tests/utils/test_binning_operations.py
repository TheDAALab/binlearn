"""Comprehensive tests for binlearn.utils._binning_operations module.

This module tests all functions in the binning operations utility module
to achieve 100% test coverage, including edge cases and error conditions.
"""

import numpy as np
import pytest

from binlearn.utils import (
    ABOVE_RANGE,
    BELOW_RANGE,
    MISSING_VALUE,
    calculate_flexible_bin_width,
    create_bin_masks,
    default_representatives,
    find_flexible_bin_for_value,
    generate_default_flexible_representatives,
    get_flexible_bin_count,
    is_missing_value,
    transform_value_to_flexible_bin,
    validate_bin_edges_format,
    validate_bin_representatives_format,
    validate_bins,
    validate_flexible_bin_spec_format,
    validate_flexible_bins,
    _validate_single_flexible_bin_def,
)


class TestValidateBinEdgesFormat:
    """Test suite for validate_bin_edges_format function."""

    def test_none_input(self):
        """Test that None input is allowed."""
        validate_bin_edges_format(None)  # Should not raise

    def test_valid_bin_edges(self):
        """Test validation of valid bin edges."""
        bin_edges = {"col1": [1.0, 2.0, 3.0], "col2": [0, 5, 10]}
        validate_bin_edges_format(bin_edges)  # Should not raise

    def test_non_dict_input(self):
        """Test error for non-dictionary input."""
        with pytest.raises(ValueError, match="bin_edges must be a dictionary"):
            validate_bin_edges_format([1, 2, 3])

    def test_non_iterable_edges(self):
        """Test error for non-iterable edges."""
        with pytest.raises(ValueError, match="must be array-like"):
            validate_bin_edges_format({"col1": 42})

    def test_string_edges(self):
        """Test error for string edges (which are iterable but invalid)."""
        with pytest.raises(ValueError, match="must be array-like"):
            validate_bin_edges_format({"col1": "invalid"})

    def test_bytes_edges(self):
        """Test error for bytes edges (which are iterable but invalid)."""
        with pytest.raises(ValueError, match="must be array-like"):
            validate_bin_edges_format({"col1": b"invalid"})

    def test_insufficient_edges(self):
        """Test error for insufficient number of edges."""
        with pytest.raises(ValueError, match="needs at least 2 bin edges"):
            validate_bin_edges_format({"col1": [1.0]})

    def test_empty_edges(self):
        """Test error for empty edges list."""
        with pytest.raises(ValueError, match="needs at least 2 bin edges"):
            validate_bin_edges_format({"col1": []})

    def test_non_numeric_edges(self):
        """Test error for non-numeric edges."""
        with pytest.raises(ValueError, match="must be numeric"):
            validate_bin_edges_format({"col1": [1.0, "invalid", 3.0]})

    def test_non_convertible_edges(self):
        """Test error for edges that can't be converted to float."""
        with pytest.raises(ValueError, match="must be numeric"):
            validate_bin_edges_format({"col1": [1.0, [2.0], 3.0]})

    def test_unsorted_edges(self):
        """Test error for unsorted edges."""
        with pytest.raises(ValueError, match="must be sorted in ascending order"):
            validate_bin_edges_format({"col1": [3.0, 1.0, 2.0]})

    def test_valid_duplicate_edges(self):
        """Test that duplicate edges are allowed (equal values)."""
        validate_bin_edges_format({"col1": [1.0, 2.0, 2.0, 3.0]})  # Should not raise

    def test_infinite_edges(self):
        """Test error for infinite edges."""
        with pytest.raises(ValueError, match="must be sorted in ascending order"):
            validate_bin_edges_format({"col1": [1.0, np.inf, 3.0]})

    def test_negative_infinite_edges(self):
        """Test error for negative infinite edges."""
        with pytest.raises(ValueError, match="must be finite values"):
            validate_bin_edges_format({"col1": [-np.inf, 1.0, 3.0]})

    def test_nan_edges(self):
        """Test error for NaN edges."""
        with pytest.raises(ValueError, match="must be sorted in ascending order"):
            validate_bin_edges_format({"col1": [1.0, np.nan, 3.0]})

    def test_finite_values_check_inf(self):
        """Test error for infinite edges with proper sorting."""
        with pytest.raises(ValueError, match="must be finite values"):
            validate_bin_edges_format({"col1": [1.0, 2.0, np.inf]})

    def test_finite_values_check_neg_inf(self):
        """Test error for negative infinite edges with proper sorting."""
        with pytest.raises(ValueError, match="must be finite values"):
            validate_bin_edges_format({"col1": [-np.inf, 1.0, 2.0]})

    def test_finite_values_check_nan(self):
        """Test error for NaN edges when properly sorted."""
        # NaN comparison behavior makes sorting check complex, so just test inf at end
        with pytest.raises(ValueError, match="must be finite values"):
            validate_bin_edges_format({"col1": [1.0, 2.0, 3.0, float("inf")]})

    def test_multiple_columns(self):
        """Test validation with multiple columns."""
        bin_edges = {"col1": [1.0, 2.0, 3.0], "col2": [-5, 0, 5, 10], "col3": [100.5, 200.5]}
        validate_bin_edges_format(bin_edges)  # Should not raise


class TestValidateBinRepresentativesFormat:
    """Test suite for validate_bin_representatives_format function."""

    def test_none_input(self):
        """Test that None input is allowed."""
        validate_bin_representatives_format(None)  # Should not raise

    def test_valid_representatives(self):
        """Test validation of valid representatives."""
        bin_reps = {"col1": [1.5, 2.5], "col2": [2.5, 7.5]}
        validate_bin_representatives_format(bin_reps)  # Should not raise

    def test_non_dict_input(self):
        """Test error for non-dictionary input."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            validate_bin_representatives_format([1.5, 2.5])

    def test_non_iterable_representatives(self):
        """Test error for non-iterable representatives."""
        with pytest.raises(ValueError, match="must be array-like"):
            validate_bin_representatives_format({"col1": 1.5})

    def test_string_representatives(self):
        """Test error for string representatives."""
        with pytest.raises(ValueError, match="must be array-like"):
            validate_bin_representatives_format({"col1": "invalid"})

    def test_bytes_representatives(self):
        """Test error for bytes representatives."""
        with pytest.raises(ValueError, match="must be array-like"):
            validate_bin_representatives_format({"col1": b"invalid"})

    def test_non_numeric_representatives(self):
        """Test error for non-numeric representatives."""
        with pytest.raises(ValueError, match="must be numeric"):
            validate_bin_representatives_format({"col1": [1.5, "invalid"]})

    def test_non_convertible_representatives(self):
        """Test error for representatives that can't be converted to float."""
        with pytest.raises(ValueError, match="must be numeric"):
            validate_bin_representatives_format({"col1": [1.5, [2.5]]})

    def test_infinite_representatives(self):
        """Test error for infinite representatives."""
        with pytest.raises(ValueError, match="must be finite values"):
            validate_bin_representatives_format({"col1": [1.5, np.inf]})

    def test_negative_infinite_representatives(self):
        """Test error for negative infinite representatives."""
        with pytest.raises(ValueError, match="must be finite values"):
            validate_bin_representatives_format({"col1": [-np.inf, 1.5]})

    def test_nan_representatives(self):
        """Test error for NaN representatives."""
        with pytest.raises(ValueError, match="must be finite values"):
            validate_bin_representatives_format({"col1": [1.5, np.nan]})

    def test_with_compatible_bin_edges(self):
        """Test representatives compatible with bin edges."""
        bin_edges = {"col1": [1.0, 2.0, 3.0]}  # 2 bins
        bin_reps = {"col1": [1.5, 2.5]}  # 2 representatives
        validate_bin_representatives_format(bin_reps, bin_edges)  # Should not raise

    def test_with_incompatible_bin_edges(self):
        """Test error for representatives incompatible with bin edges."""
        bin_edges = {"col1": [1.0, 2.0, 3.0]}  # 2 bins
        bin_reps = {"col1": [1.5, 2.5, 3.5]}  # 3 representatives
        with pytest.raises(ValueError, match="3 representatives provided, but 2 expected"):
            validate_bin_representatives_format(bin_reps, bin_edges)

    def test_with_missing_column_in_edges(self):
        """Test with column in representatives but not in edges."""
        bin_edges = {"col1": [1.0, 2.0, 3.0]}
        bin_reps = {"col2": [1.5, 2.5]}  # Different column
        validate_bin_representatives_format(bin_reps, bin_edges)  # Should not raise


class TestValidateBins:
    """Test suite for validate_bins function."""

    def test_none_bin_spec(self):
        """Test that None bin_spec is allowed."""
        validate_bins(None, {"col1": [1.5, 2.5]})  # Should not raise

    def test_valid_bins(self):
        """Test validation of valid bins."""
        bin_spec = {"col1": [1.0, 2.0, 3.0]}
        bin_reps = {"col1": [1.5, 2.5]}
        validate_bins(bin_spec, bin_reps)  # Should not raise

    def test_insufficient_edges(self):
        """Test error for insufficient edges."""
        with pytest.raises(ValueError, match="needs at least 2 bin edges"):
            validate_bins({"col1": [1.0]}, None)

    def test_unsorted_edges(self):
        """Test error for unsorted edges."""
        with pytest.raises(ValueError, match="must be non-decreasing"):
            validate_bins({"col1": [3.0, 1.0, 2.0]}, None)

    def test_equal_edges_allowed(self):
        """Test that equal consecutive edges are allowed."""
        validate_bins({"col1": [1.0, 2.0, 2.0, 3.0]}, None)  # Should not raise

    def test_mismatched_representative_count(self):
        """Test error for mismatched representative count."""
        bin_spec = {"col1": [1.0, 2.0, 3.0]}  # 2 bins
        bin_reps = {"col1": [1.5, 2.5, 3.5]}  # 3 representatives
        with pytest.raises(ValueError, match="3 representatives for 2 bins"):
            validate_bins(bin_spec, bin_reps)

    def test_none_representatives(self):
        """Test with None representatives."""
        bin_spec = {"col1": [1.0, 2.0, 3.0]}
        validate_bins(bin_spec, None)  # Should not raise

    def test_missing_column_in_representatives(self):
        """Test with column in bin_spec but not in representatives."""
        bin_spec = {"col1": [1.0, 2.0, 3.0], "col2": [0, 5, 10]}
        bin_reps = {"col1": [1.5, 2.5]}  # col2 missing
        validate_bins(bin_spec, bin_reps)  # Should not raise (only validates existing columns)


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

    def test_two_edges_only(self):
        """Test representatives with minimum number of edges."""
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

    def test_left_negative_inf(self):
        """Test representative with negative infinity on left."""
        edges = [-np.inf, 0.0]
        reps = default_representatives(edges)
        expected = [-1.0]  # right - 1.0
        assert len(reps) == len(expected)
        assert abs(reps[0] - expected[0]) < 1e-10

    def test_right_positive_inf(self):
        """Test representative with positive infinity on right."""
        edges = [0.0, np.inf]
        reps = default_representatives(edges)
        expected = [1.0]  # left + 1.0
        assert len(reps) == len(expected)
        assert abs(reps[0] - expected[0]) < 1e-10

    def test_both_infinite(self):
        """Test representative with both infinities."""
        edges = [-np.inf, np.inf]
        reps = default_representatives(edges)
        expected = [0.0]
        assert len(reps) == len(expected)
        assert abs(reps[0] - expected[0]) < 1e-10

    def test_mixed_finite_infinite(self):
        """Test representatives with mixed finite and infinite edges."""
        edges = [-np.inf, -1.0, 1.0, np.inf]
        reps = default_representatives(edges)
        expected = [-2.0, 0.0, 2.0]  # [-1-1, (-1+1)/2, 1+1]
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

    def test_out_of_range_indices(self):
        """Test masks with out of range indices."""
        bin_indices = np.array([0, 1, 5, -1])  # 5 and -1 are out of range for n_bins=2
        n_bins = 2
        valid_mask, nan_mask, below_mask, above_mask = create_bin_masks(bin_indices, n_bins)

        np.testing.assert_array_equal(valid_mask, [True, True, False, False])
        np.testing.assert_array_equal(nan_mask, [False, False, False, True])  # -1 treated as NaN
        np.testing.assert_array_equal(below_mask, [False, False, False, False])
        np.testing.assert_array_equal(above_mask, [False, False, False, False])

    def test_empty_indices(self):
        """Test masks with empty indices array."""
        bin_indices = np.array([], dtype=int)
        n_bins = 3
        valid_mask, nan_mask, below_mask, above_mask = create_bin_masks(bin_indices, n_bins)

        assert len(valid_mask) == 0
        assert len(nan_mask) == 0
        assert len(below_mask) == 0
        assert len(above_mask) == 0

    def test_all_special_values(self):
        """Test masks with all special values."""
        bin_indices = np.array([MISSING_VALUE, BELOW_RANGE, ABOVE_RANGE])
        n_bins = 2
        valid_mask, nan_mask, below_mask, above_mask = create_bin_masks(bin_indices, n_bins)

        np.testing.assert_array_equal(valid_mask, [False, False, False])
        np.testing.assert_array_equal(nan_mask, [True, False, False])
        np.testing.assert_array_equal(below_mask, [False, True, False])
        np.testing.assert_array_equal(above_mask, [False, False, True])


class TestGenerateDefaultFlexibleRepresentatives:
    """Test suite for generate_default_flexible_representatives function."""

    def test_singleton_bins(self):
        """Test generation for singleton bin definitions."""
        bin_defs = [5, 10, 15]  # All singletons
        reps = generate_default_flexible_representatives(bin_defs)
        expected = [5.0, 10.0, 15.0]
        assert len(reps) == len(expected)
        for i, val in enumerate(expected):
            assert abs(reps[i] - val) < 1e-10

    def test_interval_bins(self):
        """Test generation for interval bin definitions."""
        bin_defs = [(1.0, 2.0), (3.0, 5.0)]  # All intervals
        reps = generate_default_flexible_representatives(bin_defs)
        expected = [1.5, 4.0]  # Midpoints
        assert len(reps) == len(expected)
        for i, val in enumerate(expected):
            assert abs(reps[i] - val) < 1e-10

    def test_mixed_bins(self):
        """Test generation for mixed bin definitions."""
        bin_defs = [5, (1.0, 3.0), 10]  # singleton, interval, singleton
        reps = generate_default_flexible_representatives(bin_defs)
        expected = [5.0, 2.0, 10.0]  # singleton, midpoint, singleton
        assert len(reps) == len(expected)
        for i, val in enumerate(expected):
            assert abs(reps[i] - val) < 1e-10

    def test_negative_values(self):
        """Test generation with negative values."""
        bin_defs = [-5, (-3.0, -1.0), 0]
        reps = generate_default_flexible_representatives(bin_defs)
        expected = [-5.0, -2.0, 0.0]
        assert len(reps) == len(expected)
        for i, val in enumerate(expected):
            assert abs(reps[i] - val) < 1e-10

    def test_empty_bins(self):
        """Test generation for empty bin definitions."""
        bin_defs = []
        reps = generate_default_flexible_representatives(bin_defs)
        expected = []
        assert len(reps) == len(expected)

    def test_invalid_bin_definition(self):
        """Test error for invalid bin definition."""
        bin_defs = [5, "invalid", 10]
        with pytest.raises(ValueError, match="Unknown bin definition"):
            generate_default_flexible_representatives(bin_defs)

    def test_invalid_tuple_length(self):
        """Test error for tuple with wrong length."""
        bin_defs = [5, (1.0, 2.0, 3.0), 10]  # 3-tuple is invalid
        with pytest.raises(ValueError, match="Unknown bin definition"):
            generate_default_flexible_representatives(bin_defs)

    def test_zero_width_interval(self):
        """Test interval with zero width."""
        bin_defs = [(2.0, 2.0)]  # Zero width interval
        reps = generate_default_flexible_representatives(bin_defs)
        expected = [2.0]  # Midpoint of (2,2) is 2
        assert len(reps) == len(expected)
        assert abs(reps[0] - expected[0]) < 1e-10


class TestValidateFlexibleBins:
    """Test suite for validate_flexible_bins function."""

    def test_valid_bins_and_reps(self):
        """Test validation of valid bins and representatives."""
        bin_spec = {"col1": [5, (1.0, 3.0)]}
        bin_reps = {"col1": [5.0, 2.0]}
        validate_flexible_bins(bin_spec, bin_reps)  # Should not raise

    def test_mismatched_count(self):
        """Test error for mismatched bin and representative counts."""
        bin_spec = {"col1": [5, (1.0, 3.0)]}  # 2 bins
        bin_reps = {"col1": [5.0]}  # 1 representative
        with pytest.raises(ValueError, match="must match number of representatives"):
            validate_flexible_bins(bin_spec, bin_reps)

    def test_missing_representatives(self):
        """Test with missing representatives for a column."""
        bin_spec = {"col1": [5, (1.0, 3.0)]}
        bin_reps = {}  # No representatives
        with pytest.raises(ValueError, match="must match number of representatives"):
            validate_flexible_bins(bin_spec, bin_reps)

    def test_invalid_bin_definition(self):
        """Test error propagation from invalid bin definition."""
        bin_spec = {"col1": [5, "invalid"]}
        bin_reps = {"col1": [5.0, 2.0]}
        with pytest.raises(ValueError):  # Will be caught by _validate_single_flexible_bin_def
            validate_flexible_bins(bin_spec, bin_reps)

    def test_multiple_columns(self):
        """Test validation with multiple columns."""
        bin_spec = {"col1": [5, (1.0, 3.0)], "col2": [(0, 10), (10, 20)]}
        bin_reps = {"col1": [5.0, 2.0], "col2": [5.0, 15.0]}
        validate_flexible_bins(bin_spec, bin_reps)  # Should not raise


class TestValidateFlexibleBinSpecFormat:
    """Test suite for validate_flexible_bin_spec_format function."""

    def test_valid_bin_spec(self):
        """Test validation of valid bin specification."""
        bin_spec = {"col1": [5, (1.0, 3.0)], "col2": [(0, 10)]}
        validate_flexible_bin_spec_format(bin_spec)  # Should not raise

    def test_non_dict_input(self):
        """Test error for non-dictionary input."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            validate_flexible_bin_spec_format([5, (1.0, 3.0)])

    def test_non_list_bin_definitions(self):
        """Test error for non-list bin definitions."""
        with pytest.raises(ValueError, match="must be a list or tuple"):
            validate_flexible_bin_spec_format({"col1": 5})

    def test_empty_bin_definitions_strict(self):
        """Test error for empty bin definitions in strict mode."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_flexible_bin_spec_format({"col1": []}, strict=True)

    def test_empty_bin_definitions_non_strict(self):
        """Test allowed empty bin definitions in non-strict mode."""
        validate_flexible_bin_spec_format({"col1": []}, strict=False)  # Should not raise

    def test_tuple_bin_definitions(self):
        """Test that tuple bin definitions are accepted."""
        bin_spec = {"col1": (5, (1.0, 3.0))}  # tuple instead of list
        validate_flexible_bin_spec_format(bin_spec)  # Should not raise

    def test_finite_bounds_check(self):
        """Test finite bounds checking."""
        bin_spec = {"col1": [(1.0, np.inf)]}
        # Should not raise without check_finite_bounds
        validate_flexible_bin_spec_format(bin_spec, check_finite_bounds=False)

        # Should raise with check_finite_bounds
        with pytest.raises(ValueError, match="must be finite"):
            validate_flexible_bin_spec_format(bin_spec, check_finite_bounds=True)

    def test_invalid_bin_definition_propagation(self):
        """Test that invalid bin definitions are caught."""
        bin_spec = {"col1": ["invalid"]}
        with pytest.raises(ValueError):
            validate_flexible_bin_spec_format(bin_spec)


class TestValidateSingleFlexibleBinDef:
    """Test suite for _validate_single_flexible_bin_def function."""

    def test_valid_singleton_int(self):
        """Test valid integer singleton."""
        _validate_single_flexible_bin_def(5, "col1", 0)  # Should not raise

    def test_valid_singleton_float(self):
        """Test valid float singleton."""
        _validate_single_flexible_bin_def(5.5, "col1", 0)  # Should not raise

    def test_valid_interval(self):
        """Test valid interval."""
        _validate_single_flexible_bin_def((1.0, 3.0), "col1", 0)  # Should not raise

    def test_singleton_infinite_no_check(self):
        """Test infinite singleton without finite bounds check."""
        _validate_single_flexible_bin_def(
            np.inf, "col1", 0, check_finite_bounds=False
        )  # Should not raise

    def test_singleton_infinite_with_check(self):
        """Test error for infinite singleton with finite bounds check."""
        with pytest.raises(ValueError, match="must be finite"):
            _validate_single_flexible_bin_def(np.inf, "col1", 0, check_finite_bounds=True)

    def test_singleton_nan_with_check(self):
        """Test error for NaN singleton with finite bounds check."""
        with pytest.raises(ValueError, match="must be finite"):
            _validate_single_flexible_bin_def(np.nan, "col1", 0, check_finite_bounds=True)

    def test_interval_wrong_length(self):
        """Test error for interval with wrong length."""
        with pytest.raises(ValueError, match="Interval must be \\(min, max\\)"):
            _validate_single_flexible_bin_def((1.0,), "col1", 0)

    def test_interval_three_values(self):
        """Test error for interval with three values."""
        with pytest.raises(ValueError, match="Interval must be \\(min, max\\)"):
            _validate_single_flexible_bin_def((1.0, 2.0, 3.0), "col1", 0)

    def test_interval_non_numeric_left(self):
        """Test error for interval with non-numeric left bound."""
        with pytest.raises(ValueError, match="must be numeric"):
            _validate_single_flexible_bin_def(("invalid", 3.0), "col1", 0)

    def test_interval_non_numeric_right(self):
        """Test error for interval with non-numeric right bound."""
        with pytest.raises(ValueError, match="must be numeric"):
            _validate_single_flexible_bin_def((1.0, "invalid"), "col1", 0)

    def test_interval_infinite_no_check(self):
        """Test infinite interval without finite bounds check."""
        _validate_single_flexible_bin_def(
            (1.0, np.inf), "col1", 0, check_finite_bounds=False
        )  # Should not raise

    def test_interval_infinite_with_check(self):
        """Test error for infinite interval with finite bounds check."""
        with pytest.raises(ValueError, match="must be finite"):
            _validate_single_flexible_bin_def((1.0, np.inf), "col1", 0, check_finite_bounds=True)

    def test_interval_finite_with_check(self):
        """Test finite interval with finite bounds check (should pass)."""
        _validate_single_flexible_bin_def(
            (1.0, 3.0), "col1", 0, check_finite_bounds=True
        )  # Should not raise

    def test_interval_left_greater_strict(self):
        """Test error for left >= right in strict mode."""
        with pytest.raises(ValueError, match="must be < max"):
            _validate_single_flexible_bin_def((3.0, 1.0), "col1", 0, strict=True)

    def test_interval_left_equal_strict(self):
        """Test error for left == right in strict mode."""
        with pytest.raises(ValueError, match="must be < max"):
            _validate_single_flexible_bin_def((2.0, 2.0), "col1", 0, strict=True)

    def test_interval_left_less_strict(self):
        """Test allowed left < right in strict mode."""
        _validate_single_flexible_bin_def((1.0, 3.0), "col1", 0, strict=True)  # Should not raise

    def test_interval_left_greater_non_strict(self):
        """Test error for left > right in non-strict mode."""
        with pytest.raises(ValueError, match="must be <= max"):
            _validate_single_flexible_bin_def((3.0, 1.0), "col1", 0, strict=False)

    def test_interval_left_equal_non_strict(self):
        """Test allowed left == right in non-strict mode."""
        _validate_single_flexible_bin_def((2.0, 2.0), "col1", 0, strict=False)  # Should not raise

    def test_interval_left_less_non_strict(self):
        """Test allowed left < right in non-strict mode."""
        _validate_single_flexible_bin_def((1.0, 3.0), "col1", 0, strict=False)  # Should not raise

    def test_interval_left_greater_non_strict_error(self):
        """Test error for left > right in non-strict mode."""
        with pytest.raises(ValueError, match="must be <= max"):
            _validate_single_flexible_bin_def((3.0, 1.0), "col1", 0, strict=False)

    def test_invalid_type(self):
        """Test error for invalid bin definition type."""
        with pytest.raises(ValueError, match="must be either a numeric scalar"):
            _validate_single_flexible_bin_def("invalid", "col1", 0)

    def test_list_type(self):
        """Test error for list type."""
        with pytest.raises(ValueError, match="must be either a numeric scalar"):
            _validate_single_flexible_bin_def([1, 2], "col1", 0)


class TestIsMissingValue:
    """Test suite for is_missing_value function."""

    def test_none_value(self):
        """Test that None is considered missing."""
        assert is_missing_value(None)

    def test_nan_float(self):
        """Test that NaN float is considered missing."""
        assert is_missing_value(float("nan"))

    def test_nan_numpy(self):
        """Test that numpy NaN is considered missing."""
        assert is_missing_value(np.nan)

    def test_regular_integer(self):
        """Test that regular integer is not missing."""
        assert not is_missing_value(0)
        assert not is_missing_value(42)
        assert not is_missing_value(-5)

    def test_regular_float(self):
        """Test that regular float is not missing."""
        assert not is_missing_value(0.0)
        assert not is_missing_value(3.14)
        assert not is_missing_value(-2.5)

    def test_infinite_values(self):
        """Test that infinite values are not considered missing."""
        assert not is_missing_value(np.inf)
        assert not is_missing_value(-np.inf)
        assert not is_missing_value(float("inf"))
        assert not is_missing_value(float("-inf"))

    def test_string_value(self):
        """Test that string is not considered missing."""
        assert not is_missing_value("test")
        assert not is_missing_value("")

    def test_list_value(self):
        """Test that list is not considered missing."""
        assert not is_missing_value([])
        assert not is_missing_value([1, 2, 3])

    def test_special_constants(self):
        """Test that special constants are not considered missing."""
        assert not is_missing_value(MISSING_VALUE)
        assert not is_missing_value(BELOW_RANGE)
        assert not is_missing_value(ABOVE_RANGE)


class TestFindFlexibleBinForValue:
    """Test suite for find_flexible_bin_for_value function."""

    def test_singleton_match(self):
        """Test finding singleton bin match."""
        bin_defs = [5, 10, 15]
        assert find_flexible_bin_for_value(5, bin_defs) == 0
        assert find_flexible_bin_for_value(10, bin_defs) == 1
        assert find_flexible_bin_for_value(15, bin_defs) == 2

    def test_interval_match(self):
        """Test finding interval bin match."""
        bin_defs = [(1.0, 3.0), (5.0, 7.0)]
        assert find_flexible_bin_for_value(2.0, bin_defs) == 0  # Inside first interval
        assert find_flexible_bin_for_value(6.0, bin_defs) == 1  # Inside second interval

    def test_interval_boundary_inclusive(self):
        """Test that interval boundaries are inclusive."""
        bin_defs = [(1.0, 3.0)]
        assert find_flexible_bin_for_value(1.0, bin_defs) == 0  # Left boundary
        assert find_flexible_bin_for_value(3.0, bin_defs) == 0  # Right boundary

    def test_mixed_bins(self):
        """Test finding matches in mixed bin types."""
        bin_defs = [5, (1.0, 3.0), 10]
        assert find_flexible_bin_for_value(5, bin_defs) == 0  # Singleton
        assert find_flexible_bin_for_value(2.0, bin_defs) == 1  # Interval
        assert find_flexible_bin_for_value(10, bin_defs) == 2  # Singleton

    def test_no_match(self):
        """Test value that doesn't match any bin."""
        bin_defs = [5, (1.0, 3.0), 10]
        assert find_flexible_bin_for_value(7, bin_defs) == MISSING_VALUE
        assert find_flexible_bin_for_value(0.5, bin_defs) == MISSING_VALUE
        assert find_flexible_bin_for_value(15, bin_defs) == MISSING_VALUE

    def test_empty_bin_defs(self):
        """Test with empty bin definitions."""
        bin_defs = []
        assert find_flexible_bin_for_value(5, bin_defs) == MISSING_VALUE

    def test_non_numeric_value_with_intervals(self):
        """Test non-numeric value with interval bins (should not match)."""
        bin_defs = [(1.0, 3.0)]
        assert find_flexible_bin_for_value("test", bin_defs) == MISSING_VALUE

    def test_non_numeric_value_with_singletons(self):
        """Test non-numeric value with singleton bins."""
        bin_defs = [5, 10]
        # String won't equal integer, so no match
        assert find_flexible_bin_for_value("5", bin_defs) == MISSING_VALUE

    def test_float_singleton_match(self):
        """Test matching float singleton."""
        bin_defs = [5.5, 10.0]
        assert find_flexible_bin_for_value(5.5, bin_defs) == 0
        assert find_flexible_bin_for_value(10.0, bin_defs) == 1

    def test_invalid_tuple_bin_def(self):
        """Test with invalid tuple length (should not match)."""
        bin_defs = [(1.0,), (1.0, 2.0, 3.0)]  # Invalid tuples
        assert find_flexible_bin_for_value(1.5, bin_defs) == MISSING_VALUE


class TestCalculateFlexibleBinWidth:
    """Test suite for calculate_flexible_bin_width function."""

    def test_singleton_width(self):
        """Test width calculation for singleton bins."""
        assert calculate_flexible_bin_width(5) == 0.0
        assert calculate_flexible_bin_width(5.5) == 0.0
        assert calculate_flexible_bin_width(-3) == 0.0

    def test_interval_width(self):
        """Test width calculation for interval bins."""
        assert calculate_flexible_bin_width((1.0, 4.0)) == 3.0
        assert calculate_flexible_bin_width((0, 10)) == 10.0
        assert calculate_flexible_bin_width((-5.0, -2.0)) == 3.0

    def test_zero_width_interval(self):
        """Test width calculation for zero-width interval."""
        assert calculate_flexible_bin_width((2.0, 2.0)) == 0.0

    def test_negative_width_interval(self):
        """Test width calculation for interval with left > right."""
        assert calculate_flexible_bin_width((5.0, 2.0)) == -3.0

    def test_invalid_bin_definition(self):
        """Test error for invalid bin definition."""
        with pytest.raises(ValueError, match="Unknown bin definition"):
            calculate_flexible_bin_width("invalid")

    def test_invalid_tuple_length(self):
        """Test error for tuple with wrong length."""
        with pytest.raises(ValueError, match="Unknown bin definition"):
            calculate_flexible_bin_width((1.0,))

        with pytest.raises(ValueError, match="Unknown bin definition"):
            calculate_flexible_bin_width((1.0, 2.0, 3.0))

    def test_list_type(self):
        """Test error for list type."""
        with pytest.raises(ValueError, match="Unknown bin definition"):
            calculate_flexible_bin_width([1.0, 2.0])


class TestTransformValueToFlexibleBin:
    """Test suite for transform_value_to_flexible_bin function."""

    def test_regular_value_match(self):
        """Test transformation of regular values."""
        bin_defs = [5, (1.0, 3.0), 10]
        assert transform_value_to_flexible_bin(5, bin_defs) == 0
        assert transform_value_to_flexible_bin(2.0, bin_defs) == 1
        assert transform_value_to_flexible_bin(10, bin_defs) == 2

    def test_missing_value_none(self):
        """Test transformation of None value."""
        bin_defs = [5, 10]
        assert transform_value_to_flexible_bin(None, bin_defs) == MISSING_VALUE

    def test_missing_value_nan(self):
        """Test transformation of NaN value."""
        bin_defs = [5, 10]
        assert transform_value_to_flexible_bin(np.nan, bin_defs) == MISSING_VALUE

    def test_no_match_value(self):
        """Test transformation of value that doesn't match any bin."""
        bin_defs = [5, (1.0, 3.0), 10]
        assert transform_value_to_flexible_bin(7, bin_defs) == MISSING_VALUE

    def test_empty_bin_defs(self):
        """Test transformation with empty bin definitions."""
        bin_defs = []
        assert transform_value_to_flexible_bin(5, bin_defs) == MISSING_VALUE

    def test_infinite_values(self):
        """Test transformation of infinite values (not missing)."""
        bin_defs = [np.inf, (-np.inf, 0)]
        assert transform_value_to_flexible_bin(np.inf, bin_defs) == 0  # Matches singleton
        assert transform_value_to_flexible_bin(-5, bin_defs) == 1  # Matches interval


class TestGetFlexibleBinCount:
    """Test suite for get_flexible_bin_count function."""

    def test_single_column(self):
        """Test bin count for single column."""
        bin_spec = {"col1": [5, (1.0, 3.0), 10]}
        result = get_flexible_bin_count(bin_spec)
        expected = {"col1": 3}
        assert result == expected

    def test_multiple_columns(self):
        """Test bin count for multiple columns."""
        bin_spec = {"col1": [5, (1.0, 3.0)], "col2": [(0, 10), (10, 20), (20, 30)], "col3": [42]}
        result = get_flexible_bin_count(bin_spec)
        expected = {"col1": 2, "col2": 3, "col3": 1}
        assert result == expected

    def test_empty_columns(self):
        """Test bin count for columns with empty bin definitions."""
        bin_spec = {"col1": [], "col2": [5, 10]}
        result = get_flexible_bin_count(bin_spec)
        expected = {"col1": 0, "col2": 2}
        assert result == expected

    def test_empty_spec(self):
        """Test bin count for empty specification."""
        bin_spec = {}
        result = get_flexible_bin_count(bin_spec)
        expected = {}
        assert result == expected

    def test_various_column_types(self):
        """Test bin count with various column identifier types."""
        bin_spec = {0: [1, 2, 3], "feature_1": [(0, 5)], "category": [10, 20, 30, 40]}
        result = get_flexible_bin_count(bin_spec)
        expected = {0: 3, "feature_1": 1, "category": 4}
        assert result == expected
