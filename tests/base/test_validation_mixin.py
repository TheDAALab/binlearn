# pylint: skip-file
"""Tests for validation mixin module."""

import warnings
from unittest.mock import Mock, patch

import numpy as np
import pytest

from binlearn.base._validation_mixin import ValidationMixin
from binlearn.utils import DataQualityWarning, InvalidDataError


class TestValidationMixinValidateArrayLike:
    """Test validate_array_like method."""

    def test_numpy_array_passthrough(self):
        """Test that numpy arrays pass through unchanged."""
        arr = np.array([1, 2, 3])
        result = ValidationMixin.validate_array_like(arr, "test")
        assert result is not None  # Type guard
        assert np.array_equal(result, arr)
        assert result is arr  # Should be the same object

    def test_list_conversion(self):
        """Test conversion from list to numpy array."""
        data = [1, 2, 3]
        result = ValidationMixin.validate_array_like(data, "test")
        expected = np.array([1, 2, 3])
        assert result is not None  # Type guard
        assert np.array_equal(result, expected)
        assert isinstance(result, np.ndarray)

    def test_nested_list_conversion(self):
        """Test conversion from nested list to 2D array."""
        data = [[1, 2], [3, 4]]
        result = ValidationMixin.validate_array_like(data, "test")
        expected = np.array([[1, 2], [3, 4]])
        assert result is not None  # Type guard
        assert np.array_equal(result, expected)
        assert result.shape == (2, 2)

    def test_tuple_conversion(self):
        """Test conversion from tuple to numpy array."""
        data = (1, 2, 3)
        result = ValidationMixin.validate_array_like(data, "test")
        expected = np.array([1, 2, 3])
        assert result is not None  # Type guard
        assert np.array_equal(result, expected)

    def test_scalar_conversion(self):
        """Test conversion from scalar to numpy array."""
        data = 42
        result = ValidationMixin.validate_array_like(data, "test")
        expected = np.array(42)  # Scalar becomes 0-d array, not 1-d
        assert result is not None  # Type guard
        assert np.array_equal(result, expected)

    def test_none_input(self):
        """Test that None input raises InvalidDataError."""
        with pytest.raises(InvalidDataError) as exc_info:
            ValidationMixin.validate_array_like(None, "test_param")
        assert "test_param cannot be None" in str(exc_info.value)

    def test_none_input_allowed(self):
        """Test that None input is allowed when allow_none=True."""
        result = ValidationMixin.validate_array_like(None, "test_param", allow_none=True)
        assert result is None

    def test_invalid_input_type(self):
        """Test that invalid input type can still be converted."""
        # Even lambda functions get converted by numpy
        result = ValidationMixin.validate_array_like(lambda x: x, "test_param")
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_conversion_error_handling(self):
        """Test handling of conversion errors."""
        # Create a mock object that fails on np.asarray
        mock_obj = Mock()
        mock_obj.__array__ = Mock(side_effect=TypeError("Cannot convert"))

        with pytest.raises(InvalidDataError) as exc_info:
            ValidationMixin.validate_array_like(mock_obj, "test_param")
        # Check that the error message contains information about conversion failure
        assert "Could not convert test_param to array" in str(exc_info.value)

    def test_float_array(self):
        """Test with float array."""
        data = [1.5, 2.7, 3.1]
        result = ValidationMixin.validate_array_like(data, "test")
        assert result is not None  # Type guard
        expected = np.array([1.5, 2.7, 3.1])
        assert np.array_equal(result, expected)
        assert result.dtype == np.float64

    def test_mixed_type_array(self):
        """Test with mixed type array (should convert to string dtype by default)."""
        data = [1, "string", 3.5]
        result = ValidationMixin.validate_array_like(data, "test")
        assert result is not None  # Type guard
        assert len(result) == 3
        # numpy will convert to string array by default
        assert str(result[0]) == "1"
        assert str(result[1]) == "string"
        assert str(result[2]) == "3.5"


class TestValidationMixinValidateColumnSpecification:
    """Test validate_column_specification method."""

    def test_valid_single_column(self):
        """Test valid single column specification."""
        result = ValidationMixin.validate_column_specification([0], (5, 5))
        assert result == [0]

    def test_valid_multiple_columns(self):
        """Test valid multiple column specification."""
        result = ValidationMixin.validate_column_specification([0, 2, 4], (5, 5))
        assert result == [0, 2, 4]

    def test_valid_all_columns(self):
        """Test all columns specification."""
        result = ValidationMixin.validate_column_specification(list(range(5)), (5, 5))
        assert result == [0, 1, 2, 3, 4]

    def test_none_input_returns_all_columns(self):
        """Test that None returns all columns."""
        result = ValidationMixin.validate_column_specification(None, (5, 5))
        assert result == [0, 1, 2, 3, 4]

    def test_single_integer(self):
        """Test single integer gets wrapped in list."""
        result = ValidationMixin.validate_column_specification(0, (5, 5))
        assert result == [0]

    def test_negative_column_index(self):
        """Test negative column index raises InvalidDataError."""
        with pytest.raises(InvalidDataError) as exc_info:
            ValidationMixin.validate_column_specification([-1], (5, 5))
        assert "out of range" in str(exc_info.value)

    def test_column_index_too_large(self):
        """Test column index >= n_columns raises InvalidDataError."""
        with pytest.raises(InvalidDataError) as exc_info:
            ValidationMixin.validate_column_specification([5], (5, 5))
        assert "out of range" in str(exc_info.value)

    def test_mixed_valid_invalid_columns(self):
        """Test mix of valid and invalid column indices."""
        with pytest.raises(InvalidDataError) as exc_info:
            ValidationMixin.validate_column_specification([0, 2, 5], (5, 5))
        assert "out of range" in str(exc_info.value)

    def test_duplicate_columns(self):
        """Test duplicate column indices are preserved."""
        result = ValidationMixin.validate_column_specification([0, 1, 0, 2], (5, 5))
        assert result == [0, 1, 0, 2]

    def test_empty_list(self):
        """Test empty list is valid."""
        result = ValidationMixin.validate_column_specification([], (5, 5))
        assert result == []

    def test_zero_columns_data(self):
        """Test with data having zero columns."""
        result = ValidationMixin.validate_column_specification(None, (5, 0))
        assert result == []

    def test_string_column_specification(self):
        """Test that string column specifications are accepted."""
        # String columns are valid (for named columns)
        result = ValidationMixin.validate_column_specification("column_name", (5, 5))
        assert result == ["column_name"]

    def test_invalid_type_conversion(self):
        """Test that most types can be converted to column specs."""
        # Test with object that's not int, string, or list - should fail
        with pytest.raises(InvalidDataError):
            ValidationMixin.validate_column_specification({"invalid": "dict"}, (5, 5))


class TestValidationMixinValidateGuidanceColumns:
    """Test validate_guidance_columns method."""

    def test_no_overlap_valid(self):
        """Test valid case with no overlap."""
        result = ValidationMixin.validate_guidance_columns([2, 3], [0, 1], (5, 5))
        assert result == [2, 3]

    def test_none_guidance_returns_empty(self):
        """Test that None guidance returns empty list."""
        result = ValidationMixin.validate_guidance_columns(None, [0, 1], (5, 5))
        assert result == []

    def test_single_guidance_column(self):
        """Test single guidance column."""
        result = ValidationMixin.validate_guidance_columns(2, [0, 1], (5, 5))
        assert result == [2]

    def test_overlap_raises_error(self):
        """Test that overlapping columns raise InvalidDataError."""
        with pytest.raises(InvalidDataError) as exc_info:
            ValidationMixin.validate_guidance_columns([1, 2, 3], [0, 1, 2], (5, 5))
        assert "overlap" in str(exc_info.value)

    def test_single_overlap(self):
        """Test single column overlap."""
        with pytest.raises(InvalidDataError) as exc_info:
            ValidationMixin.validate_guidance_columns([1, 2], [0, 1], (5, 5))
        assert "overlap" in str(exc_info.value)

    def test_identical_columns(self):
        """Test completely identical column sets."""
        with pytest.raises(InvalidDataError) as exc_info:
            ValidationMixin.validate_guidance_columns([0, 1, 2], [0, 1, 2], (5, 5))
        assert "overlap" in str(exc_info.value)

    def test_empty_lists_valid(self):
        """Test empty lists are valid."""
        result = ValidationMixin.validate_guidance_columns([], [0, 1], (5, 5))
        assert result == []

    def test_guidance_column_validation(self):
        """Test that guidance columns are validated for range."""
        with pytest.raises(InvalidDataError):
            ValidationMixin.validate_guidance_columns([5], [0], (5, 5))  # Index 5 out of range


class TestValidationMixinCheckDataQuality:
    """Test check_data_quality method."""

    @patch("binlearn.base._validation_mixin.get_config")
    def test_config_disabled_no_warnings(self, mock_config):
        """Test that no warnings are issued when config disabled."""
        mock_config.return_value.show_warnings = False
        data = np.array([1, np.nan, np.inf])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidationMixin.check_data_quality(data, "test_data")
            assert len(w) == 0

    @patch("binlearn.base._validation_mixin.get_config")
    def test_clean_data_no_warnings(self, mock_config):
        """Test that clean data produces no warnings."""
        mock_config.return_value.show_warnings = True
        data = np.array([1, 2, 3, 4, 5])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidationMixin.check_data_quality(data, "test_data")
            assert len(w) == 0

    @patch("binlearn.base._validation_mixin.get_config")
    def test_low_nan_percentage_no_warning(self, mock_config):
        """Test that low percentage NaN doesn't trigger warning."""
        mock_config.return_value.show_warnings = True
        data = np.array([1, 2, np.nan, 4, 5])  # 20% missing, below threshold

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidationMixin.check_data_quality(data, "test_data")
            assert len(w) == 0

    @patch("binlearn.base._validation_mixin.get_config")
    def test_high_nan_percentage_warning(self, mock_config):
        """Test warning for high percentage NaN values."""
        mock_config.return_value.show_warnings = True
        data = np.array([1, np.nan, np.nan, np.nan])  # 75% missing, above threshold

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidationMixin.check_data_quality(data, "test_data")
            assert len(w) == 1
            assert issubclass(w[0].category, DataQualityWarning)
            assert "75.0%" in str(w[0].message)

    @patch("binlearn.base._validation_mixin.get_config")
    def test_infinite_values_warning(self, mock_config):
        """Test warning for infinite values."""
        mock_config.return_value.show_warnings = True
        data = np.array([1, 2, np.inf, 4, 5])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidationMixin.check_data_quality(data, "test_data")
            assert len(w) == 1
            assert issubclass(w[0].category, DataQualityWarning)
            assert "infinite values" in str(w[0].message)

    @patch("binlearn.base._validation_mixin.get_config")
    def test_negative_infinite_values_warning(self, mock_config):
        """Test warning for negative infinite values."""
        mock_config.return_value.show_warnings = True
        data = np.array([1, 2, -np.inf, 4, 5])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidationMixin.check_data_quality(data, "test_data")
            assert len(w) == 1
            assert issubclass(w[0].category, DataQualityWarning)
            assert "infinite values" in str(w[0].message)

    @patch("binlearn.base._validation_mixin.get_config")
    def test_both_high_nan_and_inf_warnings(self, mock_config):
        """Test warnings for both high NaN percentage and infinite values."""
        mock_config.return_value.show_warnings = True
        data = np.array([np.nan, np.nan, np.inf, np.nan])  # 75% missing + inf

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidationMixin.check_data_quality(data, "test_data")
            assert len(w) == 2
            warning_messages = [str(warning.message) for warning in w]
            assert any("75.0%" in msg for msg in warning_messages)  # Missing values warning
            assert any("infinite values" in msg for msg in warning_messages)  # Infinite warning

    @patch("binlearn.base._validation_mixin.get_config")
    def test_constant_column_warning_2d(self, mock_config):
        """Test warning for constant columns in 2D data."""
        mock_config.return_value.show_warnings = True
        data = np.array([[1, 2], [1, 3], [1, 4]])  # First column is constant

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidationMixin.check_data_quality(data, "test_data")
            assert len(w) == 1
            assert issubclass(w[0].category, DataQualityWarning)
            assert "Column 0 in test_data appears to be constant" in str(w[0].message)

    @patch("binlearn.base._validation_mixin.get_config")
    def test_multiple_constant_columns_warning(self, mock_config):
        """Test warnings for multiple constant columns."""
        mock_config.return_value.show_warnings = True
        data = np.array([[1, 2, 3], [1, 2, 4], [1, 2, 5]])  # First two columns constant

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidationMixin.check_data_quality(data, "test_data")
            assert len(w) == 2
            warning_messages = [str(warning.message) for warning in w]
            assert any("Column 0" in msg for msg in warning_messages)
            assert any("Column 1" in msg for msg in warning_messages)

    @patch("binlearn.base._validation_mixin.get_config")
    def test_constant_column_with_nan_still_constant(self, mock_config):
        """Test that constant column check handles NaN values properly."""
        mock_config.return_value.show_warnings = True
        data = np.array([[1, 2], [1, 3], [np.nan, 4]])  # First column constant except NaN

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidationMixin.check_data_quality(data, "test_data")

            # Should get constant column warning because finite values [1, 1] have zero variance
            warning_messages = [str(warning.message) for warning in w]
            assert any("Column 0" in msg and "constant" in msg for msg in warning_messages)

    @patch("binlearn.base._validation_mixin.get_config")
    def test_single_finite_value_no_constant_warning(self, mock_config):
        """Test that column with single finite value doesn't trigger constant warning."""
        mock_config.return_value.show_warnings = True
        data = np.array([[np.nan, 2], [np.nan, 3], [1, 4]])  # Only one finite value in first column

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidationMixin.check_data_quality(data, "test_data")

            warning_messages = [str(warning.message) for warning in w]
            # Should not have constant column warning since len(finite_data) <= 1
            assert not any("constant" in msg for msg in warning_messages)

    @patch("binlearn.base._validation_mixin.get_config")
    def test_non_numeric_data_no_inf_check(self, mock_config):
        """Test that non-numeric data doesn't check for infinite values."""
        mock_config.return_value.show_warnings = True
        data = np.array(["a", "b", "c"])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidationMixin.check_data_quality(data, "test_data")
            assert len(w) == 0

    @patch("binlearn.base._validation_mixin.get_config")
    def test_non_numeric_missing_values_check(self, mock_config):
        """Test missing values check for non-numeric data."""
        mock_config.return_value.show_warnings = True
        # More than 50% missing to trigger warning
        data = np.array([None, None, None, "valid"], dtype=object)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidationMixin.check_data_quality(data, "test_data")
            assert len(w) == 1
            assert "75.0%" in str(w[0].message)

    @patch("binlearn.base._validation_mixin.get_config")
    def test_1d_data_no_constant_check(self, mock_config):
        """Test that 1D data doesn't check for constant columns."""
        mock_config.return_value.show_warnings = True
        data = np.array([1, 1, 1, 1])  # Constant but 1D

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidationMixin.check_data_quality(data, "test_data")
            assert len(w) == 0

    @patch("binlearn.base._validation_mixin.get_config")
    def test_empty_data(self, mock_config):
        """Test empty data handling."""
        mock_config.return_value.show_warnings = True
        data = np.array([])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidationMixin.check_data_quality(data, "test_data")
            assert len(w) == 0

    @patch("binlearn.base._validation_mixin.get_config")
    def test_2d_empty_data(self, mock_config):
        """Test 2D empty data handling."""
        mock_config.return_value.show_warnings = True
        data = np.array([]).reshape(0, 2)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidationMixin.check_data_quality(data, "test_data")
            assert len(w) == 0

    @patch("binlearn.base._validation_mixin.get_config")
    def test_all_nan_column_no_constant_warning(self, mock_config):
        """Test that all-NaN column doesn't trigger constant warning."""
        mock_config.return_value.show_warnings = True
        data = np.array([[np.nan, 1], [np.nan, 2], [np.nan, 3]])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidationMixin.check_data_quality(data, "test_data")

            warning_messages = [str(warning.message) for warning in w]
            # Should not have constant column warning since no finite values in first column
            assert not any("constant" in msg for msg in warning_messages)

    @patch("binlearn.base._validation_mixin.get_config")
    def test_constant_column_all_inf_no_warning(self, mock_config):
        """Test constant column check with all infinite values."""
        mock_config.return_value.show_warnings = True
        data = np.array([[np.inf, 1], [np.inf, 2], [np.inf, 3]])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidationMixin.check_data_quality(data, "test_data")

            warning_messages = [str(warning.message) for warning in w]
            assert any("infinite values" in msg for msg in warning_messages)
            # Should not have constant column warning since no finite values
            assert not any("constant" in msg for msg in warning_messages)

    @patch("binlearn.base._validation_mixin.get_config")
    def test_type_error_in_inf_check_handled(self, mock_config):
        """Test handling of TypeError in infinite value check."""
        mock_config.return_value.show_warnings = True
        # Create a data type that might cause issues with inf check
        data = np.array([complex(1, 2), complex(3, 4)])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            ValidationMixin.check_data_quality(data, "test_data")
            # Should not raise error, just skip infinite check
            # Complex numbers might or might not trigger warnings depending on implementation

    @patch("binlearn.base._validation_mixin.get_config")
    def test_value_error_in_inf_check_handled(self, mock_config):
        """Test handling of ValueError in infinite value check."""
        mock_config.return_value.show_warnings = True

        # This test ensures the except clause catches ValueError as well
        with patch("numpy.issubdtype") as mock_issubdtype:
            # First call (for missing values check) returns True
            # Second call (for infinite check) raises ValueError
            mock_issubdtype.side_effect = [True, ValueError("Test error")]

            data = np.array([1, 2, 3])

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ValidationMixin.check_data_quality(data, "test_data")
                # Should not raise error, just skip infinite check
                assert len(w) == 0  # No warnings since we're below missing threshold

    @patch("binlearn.base._validation_mixin.get_config")
    def test_string_missing_values(self, mock_config):
        """Test missing value detection in string data."""
        mock_config.return_value.show_warnings = True
        # More than 50% missing to trigger warning
        data = np.array(["valid", "nan", "na", "null", ""], dtype=object)  # 80% missing

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidationMixin.check_data_quality(data, "test_data")
            assert len(w) == 1
            assert "80.0%" in str(w[0].message)


class TestValidationMixinIntegration:
    """Integration tests for ValidationMixin."""

    def test_full_validation_pipeline(self):
        """Test complete validation pipeline."""
        # Start with list data
        data = [[1, 2, np.nan], [4, 5, 6], [7, 8, np.inf]]

        # Convert to array
        array_data = ValidationMixin.validate_array_like(data, "input_data")
        assert isinstance(array_data, np.ndarray)
        assert array_data.shape == (3, 3)

        # Validate columns
        target_cols = ValidationMixin.validate_column_specification([0, 1], array_data.shape)
        guidance_cols = ValidationMixin.validate_column_specification([2], array_data.shape)

        # Check for column overlap
        ValidationMixin.validate_guidance_columns(guidance_cols, target_cols, array_data.shape)

        # Check data quality
        with patch("binlearn.base._validation_mixin.get_config") as mock_config:
            mock_config.return_value.show_warnings = True
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ValidationMixin.check_data_quality(array_data, "processed_data")
                assert len(w) >= 1  # Should have inf warnings

    def test_validation_with_errors(self):
        """Test validation pipeline with errors."""
        # This should fail at array conversion
        with pytest.raises(InvalidDataError):
            ValidationMixin.validate_array_like(None, "bad_input")

        # This should fail at column validation
        with pytest.raises(InvalidDataError):
            ValidationMixin.validate_column_specification([5], (5, 3))

        # This should fail at guidance validation
        with pytest.raises(InvalidDataError):
            ValidationMixin.validate_guidance_columns([1, 2], [0, 1], (5, 5))
