from unittest.mock import patch

import numpy as np
import pytest

from binlearn.base._flexible_binning_base import FlexibleBinningBase


class DummyFlexibleBinning(FlexibleBinningBase):
    def __init__(self, bin_spec=None, bin_representatives=None, **kwargs):
        super().__init__(bin_spec=bin_spec, bin_representatives=bin_representatives, **kwargs)

    def _calculate_flexible_bins(self, x_col, col_id, guidance_data=None):
        # Return dummy flexible bin definitions and representatives
        return [0], [0.0]  # New simplified format


class MinimalFlexibleBinning(FlexibleBinningBase):
    """Minimal implementation that doesn't override joint methods."""

    def __init__(self, bin_spec=None, bin_representatives=None, **kwargs):
        super().__init__(bin_spec=bin_spec, bin_representatives=bin_representatives, **kwargs)

    def _calculate_flexible_bins(self, x_col, col_id, guidance_data=None):
        # Return dummy flexible bin definitions and representatives
        return [0], [0.0]  # New simplified format


def test_init_default():
    """Test initialization with default parameters."""
    obj = DummyFlexibleBinning()
    assert obj.bin_spec is None
    assert obj.bin_representatives is None
    assert obj.bin_spec_ == {}
    assert obj.bin_representatives_ == {}


def test_init_with_bin_spec():
    """Test initialization with bin_spec provided."""
    bin_spec = {0: [1]}  # New simplified format

    # Test that initialization works and processes the bin_spec
    obj = DummyFlexibleBinning(bin_spec=bin_spec)
    assert obj.bin_spec == bin_spec
    # The processing now happens in _validate_params during __init__
    assert obj.bin_spec_ == bin_spec  # Should be processed and stored
    assert obj._fitted is True  # Should be marked as fitted with complete spec


def test_init_with_bin_representatives_another():
    """Test initialization with bin_representatives provided."""
    bin_reps = {0: [1.0]}
    obj = DummyFlexibleBinning(bin_representatives=bin_reps)
    assert obj.bin_representatives == bin_reps


def test_init_with_both_spec_and_reps():
    """Test initialization with both bin_spec and bin_representatives."""
    bin_spec = {0: [1]}
    bin_reps = {0: [1.0]}

    obj = DummyFlexibleBinning(bin_spec=bin_spec, bin_representatives=bin_reps)
    assert obj.bin_spec == bin_spec
    assert obj.bin_representatives == bin_reps
    # Should be processed and marked as fitted
    assert obj.bin_spec_ == bin_spec
    assert obj.bin_representatives_ == bin_reps
    assert obj._fitted is True


def test_init_validation_error():
    """Test that validation errors are caught during initialization."""
    # Test with invalid bin_spec that should cause validation to fail
    with pytest.raises(ValueError, match="Failed to process provided flexible bin specifications"):
        DummyFlexibleBinning(bin_spec={0: [(2, 3, 4)]})  # Invalid tuple format


def test_init_with_bin_representatives():
    """Test initialization with bin_representatives provided."""
    bin_reps = {0: [1.0]}
    obj = DummyFlexibleBinning(bin_representatives=bin_reps)
    assert obj.bin_representatives == bin_reps


@patch.object(DummyFlexibleBinning, "_finalize_fitting")
def test_fit_per_column_success(mock_finalize):
    """Test _fit_per_column successful execution."""
    obj = DummyFlexibleBinning()
    X = np.array([[1, 2], [3, 4]])
    columns = [0, 1]

    result = obj._fit_per_column(X, columns)

    assert result is obj
    mock_finalize.assert_called_once()


def test_fit_per_column_with_existing_specs():
    """Test _fit_per_column with existing specs.

    The fit method should always calculate bins for all columns,
    even when existing specs are present.
    """
    obj = DummyFlexibleBinning()
    obj.bin_spec_ = {0: [1]}  # New simplified format
    obj.bin_representatives_ = {0: [1.0]}

    X = np.array([[1, 2], [3, 4]])
    columns = [0, 1]

    with patch.object(obj, "_calculate_flexible_bins") as mock_calc:
        mock_calc.return_value = ([2], [2.0])  # New simplified format

        with patch.object(obj, "_finalize_fitting"):
            obj._fit_per_column(X, columns)

        # Should call _calculate_flexible_bins for both columns now
        assert mock_calc.call_count == 2
        # Both columns should have new specs calculated
        assert 0 in obj.bin_spec_
        assert 1 in obj.bin_spec_
        assert 0 in obj.bin_representatives_
        assert 1 in obj.bin_representatives_


def test_fit_per_column_error_handling():
    """Test _fit_per_column error handling."""
    obj = DummyFlexibleBinning()

    with patch.object(obj, "_calculate_flexible_bins") as mock_calc:
        mock_calc.side_effect = Exception("Test error")

        X = np.array([[1, 2], [3, 4]])
        columns = [0, 1]

        with pytest.raises(ValueError, match="Failed to fit per-column bins"):
            obj._fit_per_column(X, columns)


def test_fit_per_column_reraise_known_errors():
    """Test _fit_per_column re-raises known errors."""
    obj = DummyFlexibleBinning()

    with patch.object(obj, "_calculate_flexible_bins") as mock_calc:
        mock_calc.side_effect = ValueError("Known error")

        X = np.array([[1, 2], [3, 4]])
        columns = [0, 1]

        with pytest.raises(ValueError, match="Known error"):
            obj._fit_per_column(X, columns)


@patch.object(DummyFlexibleBinning, "_calculate_flexible_bins_jointly")
@patch.object(DummyFlexibleBinning, "_finalize_fitting")
def test_fit_jointly_success(mock_finalize, mock_calc_jointly):
    """Test _fit_jointly successful execution."""
    # Configure the mock to return the expected tuple
    mock_calc_jointly.return_value = ([0], [0.0])  # New simplified format

    obj = DummyFlexibleBinning()
    X = np.array([[1, 2], [3, 4]])
    columns = [0, 1]

    obj._fit_jointly(X, columns)

    mock_calc_jointly.assert_called()  # Should be called for each column
    mock_finalize.assert_called_once()


def test_fit_jointly_error_handling():
    """Test _fit_jointly error handling."""
    obj = DummyFlexibleBinning()

    with patch.object(obj, "_calculate_flexible_bins_jointly") as mock_calc:
        mock_calc.side_effect = Exception("Test error")

        X = np.array([[1, 2], [3, 4]])
        columns = [0, 1]

        with pytest.raises(ValueError, match="Failed to fit joint bins"):
            obj._fit_jointly(X, columns)


def test_calculate_flexible_bins_abstract():
    """Test _calculate_flexible_bins is properly abstract."""
    # We can't instantiate the abstract base class directly
    # This test shows that the method must be implemented by subclasses
    obj = DummyFlexibleBinning()
    # The dummy class implements it, so we can call it
    result = obj._calculate_flexible_bins(np.array([1, 2, 3]), 0)
    assert result == ([0], [0.0])  # New simplified format


@patch("binlearn.base._flexible_binning_base.find_flexible_bin_for_value")
def test_get_column_key_direct_match(mock_find):
    """Test _get_column_key with direct match."""
    obj = DummyFlexibleBinning()

    target_col = "A"
    available_keys = ["A", "B", "C"]
    col_index = 0

    result = obj._get_column_key(target_col, available_keys, col_index)
    assert result == "A"


def test_get_column_key_index_fallback():
    """Test _get_column_key with index fallback."""
    obj = DummyFlexibleBinning()

    target_col = "X"
    available_keys = ["A", "B", "C"]
    col_index = 1

    result = obj._get_column_key(target_col, available_keys, col_index)
    assert result == "B"


def test_get_column_key_no_match():
    """Test _get_column_key with no match."""
    obj = DummyFlexibleBinning()

    target_col = "X"
    available_keys = ["A", "B"]
    col_index = 5

    with pytest.raises(ValueError, match="No bin specification found for column X"):
        obj._get_column_key(target_col, available_keys, col_index)


@patch("binlearn.base._flexible_binning_base.transform_value_to_flexible_bin")
@patch("binlearn.base._flexible_binning_base.MISSING_VALUE", 999)
def test_transform_columns(mock_transform):
    """Test _transform_columns method."""
    mock_transform.return_value = 0  # Return bin index 0

    obj = DummyFlexibleBinning()
    obj.bin_spec_ = {0: [1], 1: [2]}  # New simplified format
    obj.bin_representatives_ = {0: [1.0], 1: [2.0]}

    X = np.array([[1, 2], [3, 4]])
    columns = [0, 1]

    result = obj._transform_columns(X, columns)

    assert result.shape == (2, 2)
    # Should call transform_value_to_flexible_bin for each value
    assert mock_transform.call_count == 4


def test_transform_columns_missing_key():
    """Test _transform_columns with missing column key."""
    obj = DummyFlexibleBinning()
    obj.bin_spec_ = {}  # Empty spec

    X = np.array([[1, 2]])
    columns = [0]

    with pytest.raises(ValueError, match="No bin specification found"):
        obj._transform_columns(X, columns)


@patch("binlearn.base._flexible_binning_base.validate_flexible_bins")
def test_finalize_fitting(mock_validate):
    """Test _finalize_fitting method."""
    obj = DummyFlexibleBinning()
    obj.bin_spec_ = {0: [1]}  # New simplified format
    obj.bin_representatives_ = {0: [1.0]}

    obj._finalize_fitting()

    mock_validate.assert_called_once_with(obj.bin_spec_, obj.bin_representatives_)


def test_finalize_fitting_error():
    """Test _finalize_fitting error handling."""
    obj = DummyFlexibleBinning()
    obj.bin_spec_ = {0: [1]}  # New simplified format
    obj.bin_representatives_ = {0: [1.0]}

    with patch("binlearn.base._flexible_binning_base.validate_flexible_bins") as mock_validate:
        mock_validate.side_effect = Exception("Validation error")

        # The method doesn't wrap exceptions, so expect raw Exception
        with pytest.raises(Exception, match="Validation error"):
            obj._finalize_fitting()


def test_calculate_flexible_bins_jointly_abstract():
    """Test _calculate_flexible_bins_jointly default implementation."""
    obj = DummyFlexibleBinning()

    # This should call the base implementation which falls back to _calculate_flexible_bins
    result = obj._calculate_flexible_bins_jointly(np.array([1, 2]), [0])
    assert result == ([0], [0.0])  # New simplified format


@patch("binlearn.base._flexible_binning_base.return_like_input")
def test_inverse_transform_columns(mock_return):
    """Test _inverse_transform_columns method."""
    mock_return.return_value = np.array([[1.0, 2.0]])

    obj = DummyFlexibleBinning()
    obj.bin_representatives_ = {0: [1.0, 2.0], 1: [3.0, 4.0]}

    X = np.array([[0, 1], [1, 0]])
    columns = [0, 1]

    result = obj._inverse_transform_columns(X, columns)

    # Should use bin representatives to inverse transform
    expected = np.array([[1.0, 4.0], [2.0, 3.0]])
    np.testing.assert_array_equal(result, expected)


def test_inverse_transform_columns_missing_key():
    """Test _inverse_transform_columns with missing column key."""
    obj = DummyFlexibleBinning()
    obj.bin_representatives_ = {}  # Empty reps

    X = np.array([[0]])
    columns = [0]

    with pytest.raises(ValueError, match="No bin specification found"):
        obj._inverse_transform_columns(X, columns)


def test_inverse_transform_columns_all_missing():
    """Test _inverse_transform_columns when all values are missing."""
    from binlearn.utils.constants import MISSING_VALUE

    obj = DummyFlexibleBinning()
    obj.bin_representatives_ = {0: [1.0, 2.0]}

    # All values are missing - this should cover the regular_indices.any() == False branch
    X = np.array([[MISSING_VALUE], [MISSING_VALUE]])
    columns = [0]

    result = obj._inverse_transform_columns(X, columns)

    # All results should be NaN
    expected = np.array([[np.nan], [np.nan]])
    np.testing.assert_array_equal(result, expected)


def test_get_fitted_params():
    """Test _get_fitted_params method."""
    obj = DummyFlexibleBinning()
    obj._fitted = True
    obj.bin_spec_ = {0: [1]}  # New simplified format
    obj.bin_representatives_ = {0: [1.0]}

    # Now access fitted attributes directly (sklearn style)
    assert obj.bin_spec_ == {0: [1]}  # New simplified format
    assert obj.bin_representatives_ == {0: [1.0]}


def test_handle_bin_params():
    """Test _handle_bin_params method."""
    obj = DummyFlexibleBinning()

    # Test with bin_spec change
    reset = obj._handle_bin_params({"bin_spec": {0: [1]}})  # New simplified format
    assert reset is True

    # Test with bin_representatives change
    reset = obj._handle_bin_params({"bin_representatives": {0: [1.0]}})
    assert reset is True

    # Test with no relevant changes
    reset = obj._handle_bin_params({"other_param": "value"})
    assert reset is False


def test_finalize_fitting_with_missing_reps():
    """Test _finalize_fitting generates missing representatives."""
    obj = DummyFlexibleBinning()
    obj.bin_spec_ = {0: [1], 1: [2]}  # New simplified format
    obj.bin_representatives_ = {0: [1.0]}  # Missing reps for column 1

    with patch(
        "binlearn.base._flexible_binning_base.generate_default_flexible_representatives"
    ) as mock_gen:
        mock_gen.return_value = [2.0]
        with patch("binlearn.base._flexible_binning_base.validate_flexible_bins"):
            obj._finalize_fitting()

        # Should generate reps for column 1 only
        mock_gen.assert_called_once_with([2])  # New simplified format
        assert obj.bin_representatives_[1] == [2.0]


def test_calculate_flexible_bins_jointly_fallback():
    """Test _calculate_flexible_bins_jointly falls back to _calculate_flexible_bins."""
    obj = DummyFlexibleBinning()
    all_data = np.array([1, 2, 3])

    # Should fall back to _calculate_flexible_bins
    result = obj._calculate_flexible_bins_jointly(all_data, [0])
    assert result == ([0], [0.0])  # New simplified format


def test_generate_default_flexible_representatives():
    """Test deprecated _generate_default_flexible_representatives method."""
    obj = DummyFlexibleBinning()

    with patch(
        "binlearn.base._flexible_binning_base.generate_default_flexible_representatives"
    ) as mock_gen:
        mock_gen.return_value = [1.0, 2.0]

        result = obj._generate_default_flexible_representatives([1, 2])  # New simplified format
        assert result == [1.0, 2.0]
        mock_gen.assert_called_once()


def test_lookup_bin_widths():
    """Test lookup_bin_widths method."""
    obj = DummyFlexibleBinning()
    obj._fitted = True
    obj.bin_spec_ = {0: [(1, 3), 5]}  # New simplified format: interval and singleton

    with patch.object(obj, "_prepare_input") as mock_prepare:
        mock_prepare.return_value = (np.array([[0, 1]]), [0])

        with patch(
            "binlearn.base._flexible_binning_base.calculate_flexible_bin_width"
        ) as mock_calc:
            mock_calc.return_value = 2.0

            with patch("binlearn.base._flexible_binning_base.return_like_input") as mock_return:
                mock_return.return_value = np.array([[2.0]])

                bin_indices = np.array([[0]])
                _ = obj.lookup_bin_widths(bin_indices)

                mock_calc.assert_called_once()
                mock_return.assert_called_once()


def test_lookup_bin_widths_missing_value():
    """Test lookup_bin_widths with missing values."""
    obj = DummyFlexibleBinning()
    obj._fitted = True
    obj.bin_spec_ = {0: [1]}  # New simplified format

    with patch.object(obj, "_prepare_input") as mock_prepare:
        with patch("binlearn.base._flexible_binning_base.MISSING_VALUE", 999):
            mock_prepare.return_value = (np.array([[999]]), [0])  # Missing value

            with patch("binlearn.base._flexible_binning_base.return_like_input") as mock_return:
                mock_return.return_value = np.array([[np.nan]])

                bin_indices = np.array([[999]])
                _ = obj.lookup_bin_widths(bin_indices)

                mock_return.assert_called_once()


def test_lookup_bin_widths_out_of_bounds():
    """Test lookup_bin_widths with out-of-bounds bin indices."""
    obj = DummyFlexibleBinning()
    obj._fitted = True
    obj.bin_spec_ = {0: [1, 2]}  # Two bins: indices 0 and 1

    with patch.object(obj, "_prepare_input") as mock_prepare:
        # Bin index 5 is out of bounds (only have indices 0 and 1)
        mock_prepare.return_value = (np.array([[5]]), [0])

        with patch("binlearn.base._flexible_binning_base.return_like_input") as mock_return:
            mock_return.return_value = np.array([[np.nan]])

            bin_indices = np.array([[5]])
            _ = obj.lookup_bin_widths(bin_indices)

            # Should return NaN for out-of-bounds indices
            mock_return.assert_called_once()


def test_lookup_bin_ranges():
    """Test lookup_bin_ranges method."""
    obj = DummyFlexibleBinning()
    obj._fitted = True
    obj.bin_spec_ = {0: [1, 2], 1: [3]}  # New simplified format

    result = obj.lookup_bin_ranges()

    assert result == {0: 2, 1: 1}  # Column 0 has 2 bins, column 1 has 1 bin


def test_deprecated_validate_flexible_bins():
    """Test deprecated _validate_flexible_bins method."""
    obj = DummyFlexibleBinning()

    with patch("binlearn.base._flexible_binning_base.validate_flexible_bins") as mock_validate:
        bin_spec = {0: [1]}  # New simplified format
        bin_reps = {0: [1.0]}

        obj._validate_flexible_bins(bin_spec, bin_reps)
        mock_validate.assert_called_once_with(bin_spec, bin_reps)


def test_deprecated_is_missing_value():
    """Test deprecated _is_missing_value method."""
    obj = DummyFlexibleBinning()

    with patch("binlearn.base._flexible_binning_base.is_missing_value") as mock_missing:
        mock_missing.return_value = True

        result = obj._is_missing_value(np.nan)
        assert result is True
        mock_missing.assert_called_once_with(np.nan)


def test_deprecated_find_bin_for_value():
    """Test deprecated _find_bin_for_value method."""
    obj = DummyFlexibleBinning()

    with patch("binlearn.base._flexible_binning_base.find_flexible_bin_for_value") as mock_find:
        mock_find.return_value = 1

        bin_defs = [1, 2]  # New simplified format
        result = obj._find_bin_for_value(2.0, bin_defs)
        assert result == 1
        mock_find.assert_called_once_with(2.0, bin_defs)


def test_inverse_transform():
    """Test inverse_transform method."""
    obj = DummyFlexibleBinning()
    obj._fitted = True
    obj.bin_representatives_ = {0: [1.0, 2.0]}

    with patch.object(obj, "_prepare_input") as mock_prepare:
        mock_prepare.return_value = (np.array([[0]]), [0])

        with patch.object(obj, "_inverse_transform_columns") as mock_inverse:
            mock_inverse.return_value = np.array([[1.0]])

            with patch("binlearn.base._flexible_binning_base.return_like_input") as mock_return:
                mock_return.return_value = np.array([[1.0]])

                X = np.array([[0]])
                _ = obj.inverse_transform(X)

                mock_prepare.assert_called_once_with(X)
                mock_inverse.assert_called_once()
                mock_return.assert_called_once()


def test_handle_bin_params_with_fit_jointly():
    """Test _handle_bin_params with fit_jointly parameter."""
    obj = DummyFlexibleBinning()

    reset = obj._handle_bin_params({"fit_jointly": True})
    assert reset is True
    assert obj.fit_jointly is True


def test_handle_bin_params_with_guidance_columns():
    """Test _handle_bin_params with guidance_columns parameter."""
    obj = DummyFlexibleBinning()

    reset = obj._handle_bin_params({"guidance_columns": ["A", "B"]})
    assert reset is True
    assert obj.guidance_columns == ["A", "B"]


def test_handle_bin_params_multiple_changes():
    """Test _handle_bin_params with multiple parameter changes."""
    obj = DummyFlexibleBinning()

    reset = obj._handle_bin_params(
        {
            "bin_spec": {0: [1]},  # New simplified format
            "fit_jointly": False,
            "guidance_columns": None,
            "other_param": "value",
        }
    )
    assert reset is True


def test_fit_per_column_with_guidance_data():
    """Test _fit_per_column with guidance_data parameter."""
    obj = DummyFlexibleBinning()
    X = np.array([[1, 2], [3, 4]])
    columns = [0, 1]
    guidance_data = np.array([[0.5, 1.5], [2.5, 3.5]])

    with patch.object(obj, "_calculate_flexible_bins") as mock_calc:
        mock_calc.return_value = ([1], [1.0])  # New simplified format
        with patch.object(obj, "_finalize_fitting"):
            obj._fit_per_column(X, columns, guidance_data=guidance_data)

        # Should be called with guidance_data
        calls = mock_calc.call_args_list
        assert len(calls) == 2  # Called for both columns
        # Check that guidance_data was passed to at least one call
        assert any(call[0][2] is guidance_data for call in calls)


def test_fit_jointly_with_no_missing_columns():
    """Test _fit_jointly when all columns already have specs.

    The fit method should always calculate bins for all columns,
    even when existing specs are present.
    """
    obj = DummyFlexibleBinning()
    obj.bin_spec_ = {0: [1], 1: [2]}  # New simplified format
    obj.bin_representatives_ = {0: [1.0], 1: [2.0]}

    X = np.array([[1, 2], [3, 4]])
    columns = [0, 1]

    with patch.object(obj, "_calculate_flexible_bins_jointly") as mock_calc:
        mock_calc.return_value = ([99], [99.0])  # New simplified format

        with patch.object(obj, "_finalize_fitting"):
            obj._fit_jointly(X, columns)

        # Should calculate bins for all columns now
        mock_calc.assert_called_once()
        # All columns should have the same jointly calculated specs
        assert obj.bin_spec_[0] == [99]
        assert obj.bin_spec_[1] == [99]
        assert obj.bin_representatives_[0] == [99.0]
        assert obj.bin_representatives_[1] == [99.0]


def test_calculate_flexible_bins_jointly_direct_call():
    """Test _calculate_flexible_bins_jointly method is called directly."""
    obj = MinimalFlexibleBinning()  # Use minimal class that doesn't override the method
    all_data = np.array([1, 2, 3])
    columns = [0]

    # Call the method directly to ensure fallback behavior works
    result = obj._calculate_flexible_bins_jointly(all_data, columns)
    assert result == ([0], [0.0])  # New simplified format - Should call _calculate_flexible_bins


def test_fit_jointly_covers_missing_lines():
    """Test that _fit_jointly actually covers lines 168 and 177."""
    obj = MinimalFlexibleBinning()  # Use minimal class to test base implementations
    X = np.array([[1, 2], [3, 4]])
    columns = [0, 1]

    # Ensure no existing specs so it will call joint methods
    obj.bin_spec_ = {}
    obj.bin_representatives_ = {}
    obj.bin_spec = None  # No user specifications

    # Don't patch anything - let the real methods run
    obj._fit_jointly(X, columns)

    # Verify that the bins were properly created
    assert 0 in obj.bin_spec_
    assert 1 in obj.bin_spec_
    assert 0 in obj.bin_representatives_
    assert 1 in obj.bin_representatives_

    # Verify the bins have expected structure
    assert obj.bin_spec_[0] == [0]  # New simplified format
    assert obj.bin_spec_[1] == [0]  # New simplified format


def test_fit_jointly_enabled_through_handle_bin_params():
    """Test that fit_jointly is enabled and covers the missing lines through the full flow."""
    obj = MinimalFlexibleBinning()  # Use minimal class to test base implementations
    X = np.array([[1, 2], [3, 4]])

    # Enable fit_jointly through handle_bin_params
    changed = obj._handle_bin_params({"fit_jointly": True})
    assert changed is True  # Should indicate parameters changed

    # Ensure no existing specs
    obj.bin_spec_ = {}
    obj.bin_representatives_ = {}
    obj.bin_spec = None

    # Fit should use _fit_jointly which calls our target methods
    obj.fit(X)

    # Verify bins were created through the joint fitting process
    assert 0 in obj.bin_spec_
    assert 1 in obj.bin_spec_


def test_property_setters_reset_fitted_state():
    """Test that property setters reset fitted state when object is fitted."""
    obj = DummyFlexibleBinning()

    # First, mark the object as fitted
    obj._fitted = True

    # Test bin_spec setter resets fitted state
    obj.bin_spec = {0: [1]}  # New simplified format
    assert obj._fitted is False

    # Mark as fitted again
    obj._fitted = True

    # Test bin_representatives setter resets fitted state
    obj.bin_representatives = {0: [1.0]}
    assert obj._fitted is False


def test_property_setters_no_fitted_attribute():
    """Test that property setters work when _fitted attribute doesn't exist yet."""
    obj = DummyFlexibleBinning()

    # Remove _fitted attribute if it exists
    if hasattr(obj, "_fitted"):
        delattr(obj, "_fitted")

    # These should not raise errors
    obj.bin_spec = {0: [1]}  # New simplified format
    obj.bin_representatives = {0: [1.0]}


def test_fit_jointly_minimal_class_edge_cases():
    """Covers edge cases for MinimalFlexibleBinning._fit_jointly and _calculate_flexible_bins_jointly."""
    obj = MinimalFlexibleBinning()
    # Edge case: empty input
    result = obj._calculate_flexible_bins_jointly(np.array([]), [])
    assert result == ([0], [0.0])
    # Edge case: fit with empty columns
    obj._fit_jointly(np.array([[]]), [])
    assert obj.bin_spec_ == {}
    assert obj.bin_representatives_ == {}


def test_dummy_flexible_binning_repr_and_properties():
    """Covers DummyFlexibleBinning property setters and __repr__."""
    obj = DummyFlexibleBinning()
    # Set bin_spec and bin_representatives
    obj.bin_spec = {0: [1]}
    obj.bin_representatives = {0: [1.0]}
    # Check repr
    assert isinstance(repr(obj), str)
    # Remove _fitted and set properties
    if hasattr(obj, "_fitted"):
        delattr(obj, "_fitted")
    obj.bin_spec = {1: [2]}
    obj.bin_representatives = {1: [2.0]}
    assert obj.bin_spec == {1: [2]}
    assert obj.bin_representatives == {1: [2.0]}


def test_set_sklearn_attributes_from_specs_with_nonebin_spec_():
    """Test _set_sklearn_attributes_from_specs when bin_spec is None - covers line 210->exit."""
    obj = DummyFlexibleBinning()  # bin_spec is None by default
    # This should not set any sklearn attributes
    obj._set_sklearn_attributes_from_specs()
    assert obj._feature_names_in is None
    assert obj._n_features_in is None


def test_set_sklearn_attributes_from_specs_with_guidance_columns():
    """Test _set_sklearn_attributes_from_specs with guidance columns - covers lines 217-225."""
    # Test with single guidance column (string)
    bin_spec = {0: [1], 1: [2]}
    obj = DummyFlexibleBinning(bin_spec=bin_spec, guidance_columns="target")

    # Should include both binning columns (0, 1) and guidance column ('target')
    expected_features = [0, 1, "target"]
    assert obj._feature_names_in == expected_features
    assert obj._n_features_in == 3

    # Test with list of guidance columns
    obj2 = DummyFlexibleBinning(bin_spec=bin_spec, guidance_columns=["target1", "target2"])
    expected_features2 = [0, 1, "target1", "target2"]
    assert obj2._feature_names_in == expected_features2
    assert obj2._n_features_in == 4

    # Test when guidance column is already in binning columns (shouldn't duplicate)
    bin_spec_overlap = {0: [1], "target": [2]}
    obj3 = DummyFlexibleBinning(bin_spec=bin_spec_overlap, guidance_columns="target")
    expected_features3 = [0, "target"]  # 'target' shouldn't be duplicated
    assert obj3._feature_names_in == expected_features3
    assert obj3._n_features_in == 2
