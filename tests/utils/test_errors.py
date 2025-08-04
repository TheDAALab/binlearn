"""Tests for errors module."""

import warnings
from unittest.mock import Mock, patch

import numpy as np
import pytest

from binlearn.utils.errors import (
    BinningError,
    BinningWarning,
    ConfigurationError,
    DataQualityWarning,
    FittingError,
    InvalidDataError,
    PerformanceWarning,
    TransformationError,
    ValidationError,
    ValidationMixin,
    suggest_alternatives,
    validate_tree_params,
)


class TestBinningError:
    """Test BinningError base class."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = BinningError("Test error")
        assert str(error) == "Test error"
        assert error.suggestions == []

    def test_error_with_suggestions(self):
        """Test error with suggestions."""
        suggestions = ["Try this", "Or try that"]
        error = BinningError("Test error", suggestions=suggestions)

        expected = "Test error\n\nSuggestions:\n  - Try this\n  - Or try that"
        assert str(error) == expected
        assert error.suggestions == suggestions

    def test_error_without_suggestions(self):
        """Test error without suggestions."""
        error = BinningError("Test error", suggestions=None)
        assert str(error) == "Test error"
        assert error.suggestions == []

    def test_empty_suggestions(self):
        """Test error with empty suggestions."""
        error = BinningError("Test error", suggestions=[])
        assert str(error) == "Test error"
        assert error.suggestions == []


class TestErrorSubclasses:
    """Test error subclasses inherit properly."""

    def test_invalid_data_error(self):
        """Test InvalidDataError."""
        error = InvalidDataError("Invalid data", ["Fix the data"])
        assert isinstance(error, BinningError)
        assert "Invalid data" in str(error)
        assert "Fix the data" in str(error)

    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Bad config")
        assert isinstance(error, BinningError)
        assert str(error) == "Bad config"

    def test_fitting_error(self):
        """Test FittingError."""
        error = FittingError("Fitting failed")
        assert isinstance(error, BinningError)
        assert str(error) == "Fitting failed"

    def test_transformation_error(self):
        """Test TransformationError."""
        error = TransformationError("Transform failed")
        assert isinstance(error, BinningError)
        assert str(error) == "Transform failed"

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("Validation failed")
        assert isinstance(error, BinningError)
        assert str(error) == "Validation failed"


class TestWarningClasses:
    """Test warning classes."""

    def test_binning_warning(self):
        """Test BinningWarning."""
        assert issubclass(BinningWarning, UserWarning)

    def test_data_quality_warning(self):
        """Test DataQualityWarning."""
        assert issubclass(DataQualityWarning, BinningWarning)
        assert issubclass(DataQualityWarning, UserWarning)

    def test_performance_warning(self):
        """Test PerformanceWarning."""
        assert issubclass(PerformanceWarning, BinningWarning)
        assert issubclass(PerformanceWarning, UserWarning)


# pylint: disable=too-many-public-methods
class TestValidationMixin:
    """Test ValidationMixin methods."""

    def test_validate_array_like_none_allowed(self):
        """Test validate_array_like with None allowed."""
        result = ValidationMixin.validate_array_like(None, "test", allow_none=True)
        assert result is None

    def test_validate_array_like_none_not_allowed(self):
        """Test validate_array_like with None not allowed."""
        with pytest.raises(InvalidDataError, match="test cannot be None"):
            ValidationMixin.validate_array_like(None, "test", allow_none=False)

    def test_validate_array_like_valid_data(self):
        """Test validate_array_like with valid data."""
        data = [1, 2, 3, 4]
        result = ValidationMixin.validate_array_like(data, "test")
        np.testing.assert_array_equal(result, np.array([1, 2, 3, 4]))

    def test_validate_array_like_numpy_array(self):
        """Test validate_array_like with numpy array."""
        data = np.array([1, 2, 3, 4])
        result = ValidationMixin.validate_array_like(data, "test")
        np.testing.assert_array_equal(result, data)

    def test_validate_array_like_conversion_error(self):
        """Test validate_array_like with data that can't be converted."""

        # pylint: disable=too-few-public-methods
        class UnconvertibleType:
            """Test class that cannot be converted to array."""

            def __array__(self):
                raise ValueError("Cannot convert to array")

        data = UnconvertibleType()
        with pytest.raises(InvalidDataError, match="Could not convert test to array"):
            ValidationMixin.validate_array_like(data, "test")

    def test_validate_column_specification_none(self):
        """Test validate_column_specification with None."""
        data_shape = (10, 3)
        result = ValidationMixin.validate_column_specification(None, data_shape)
        assert result == [0, 1, 2]

    def test_validate_column_specification_single_string(self):
        """Test validate_column_specification with single string."""
        data_shape = (10, 3)
        result = ValidationMixin.validate_column_specification("col1", data_shape)
        assert result == ["col1"]

    def test_validate_column_specification_single_int(self):
        """Test validate_column_specification with single valid int."""
        data_shape = (10, 3)
        result = ValidationMixin.validate_column_specification(1, data_shape)
        assert result == [1]

    def test_validate_column_specification_list(self):
        """Test validate_column_specification with list."""
        data_shape = (10, 3)
        result = ValidationMixin.validate_column_specification([0, "col1", 2], data_shape)
        assert result == [0, "col1", 2]

    def test_validate_column_specification_out_of_range(self):
        """Test validate_column_specification with out of range index."""
        data_shape = (10, 3)
        with pytest.raises(InvalidDataError, match="Column index 5 is out of range"):
            ValidationMixin.validate_column_specification(5, data_shape)

    def test_validate_column_specification_negative_index(self):
        """Test validate_column_specification with negative index."""
        data_shape = (10, 3)
        with pytest.raises(InvalidDataError, match="Column index -1 is out of range"):
            ValidationMixin.validate_column_specification(-1, data_shape)

    def test_validate_column_specification_invalid_type(self):
        """Test validate_column_specification with invalid type."""
        data_shape = (10, 3)
        with pytest.raises(InvalidDataError, match="Invalid column specification"):
            ValidationMixin.validate_column_specification([1.5], data_shape)

    def test_validate_guidance_columns_none(self):
        """Test validate_guidance_columns with None."""
        result = ValidationMixin.validate_guidance_columns(None, [0, 1], (10, 3))
        assert not result

    def test_validate_guidance_columns_single(self):
        """Test validate_guidance_columns with single column."""
        result = ValidationMixin.validate_guidance_columns(2, [0, 1], (10, 3))
        assert result == [2]

    def test_validate_guidance_columns_list(self):
        """Test validate_guidance_columns with list."""
        result = ValidationMixin.validate_guidance_columns([2, "col3"], [0, 1], (10, 4))
        assert result == [2, "col3"]

    def test_validate_guidance_columns_overlap(self):
        """Test validate_guidance_columns with overlap."""
        with pytest.raises(InvalidDataError, match="Guidance columns cannot overlap"):
            ValidationMixin.validate_guidance_columns([0, 2], [0, 1], (10, 3))

    @patch("binlearn.utils.errors.get_config")
    def test_check_data_quality_warnings_disabled(self, mock_get_config):
        """Test check_data_quality with warnings disabled."""
        mock_config = Mock()
        mock_config.show_warnings = False
        mock_get_config.return_value = mock_config

        data = np.array([1, 2, np.nan, 4])

        # Should not raise any warnings and should return early
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ValidationMixin.check_data_quality(data, "test")
            assert len(w) == 0
            assert result is None  # Method returns None when warnings are disabled

        # Verify that get_config was called
        mock_get_config.assert_called_once()

    @patch("binlearn.config.get_config")
    def test_check_data_quality_missing_values(self, mock_get_config):
        """Test check_data_quality with missing values."""
        mock_config = Mock()
        mock_config.show_warnings = True
        mock_get_config.return_value = mock_config

        # More than 50% missing values
        data = np.array([1, np.nan, np.nan, np.nan])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidationMixin.check_data_quality(data, "test")
            assert len(w) == 1
            assert issubclass(w[0].category, DataQualityWarning)
            assert "75.0% missing values" in str(w[0].message)

    @patch("binlearn.config.get_config")
    def test_check_data_quality_infinite_values(self, mock_get_config):
        """Test check_data_quality with infinite values."""
        mock_config = Mock()
        mock_config.show_warnings = True
        mock_get_config.return_value = mock_config

        data = np.array([1, 2, np.inf, 4])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidationMixin.check_data_quality(data, "test")
            assert len(w) == 1
            assert issubclass(w[0].category, DataQualityWarning)
            assert "infinite values" in str(w[0].message)

    @patch("binlearn.config.get_config")
    def test_check_data_quality_constant_column(self, mock_get_config):
        """Test check_data_quality with constant column."""
        mock_config = Mock()
        mock_config.show_warnings = True
        mock_get_config.return_value = mock_config

        # 2D array with constant column
        data = np.array([[1, 5], [2, 5], [3, 5]])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidationMixin.check_data_quality(data, "test")
            assert len(w) == 1
            assert issubclass(w[0].category, DataQualityWarning)
            assert "Column 1" in str(w[0].message)
            assert "constant" in str(w[0].message)

    @patch("binlearn.config.get_config")
    def test_check_data_quality_string_missing_values(self, mock_get_config):
        """Test check_data_quality with string data containing missing values."""
        mock_config = Mock()
        mock_config.show_warnings = True
        mock_get_config.return_value = mock_config

        # String data with missing values (>50% to trigger warning)
        data = np.array(["a", "nan", None, "", "b"], dtype=object)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidationMixin.check_data_quality(data, "test")
            assert len(w) == 1
            assert issubclass(w[0].category, DataQualityWarning)
            assert "60.0% missing values" in str(w[0].message)

    @patch("binlearn.config.get_config")
    def test_check_data_quality_complex_dtype_exception(self, mock_get_config):
        """Test check_data_quality with complex dtype that raises exceptions."""
        mock_config = Mock()
        mock_config.show_warnings = True
        mock_get_config.return_value = mock_config

        # Create data that will trigger the exception handling
        complex_data = np.array([1 + 2j, 3 + 4j])  # Complex dtype

        # Should not raise exception, just skip checks
        ValidationMixin.check_data_quality(complex_data, "test")

    @patch("binlearn.config.get_config")
    def test_check_data_quality_type_error_exception(self, mock_get_config):
        """Test check_data_quality with data that triggers TypeError in missing value check."""
        mock_config = Mock()
        mock_config.show_warnings = True
        mock_get_config.return_value = mock_config

        # Test with data that will trigger the except block
        # Use a dtype that will cause problems in the missing value check
        special_data = np.array([{"a": 1}, {"b": 2}], dtype=object)

        # Should not raise exception, just skip checks
        ValidationMixin.check_data_quality(special_data, "test")

    @patch("binlearn.config.get_config")
    def test_check_data_quality_inf_exception(self, mock_get_config):
        """Test check_data_quality with infinite value check exception."""
        mock_config = Mock()
        mock_config.show_warnings = True
        mock_get_config.return_value = mock_config

        # Mock np.isinf to raise a TypeError (which should be caught)
        with patch("numpy.isinf") as mock_isinf:
            mock_isinf.side_effect = TypeError("Data type not supported")

            # Should not raise exception, just skip checks
            ValidationMixin.check_data_quality(np.array([1, 2, 3]), "test")

    @patch("binlearn.config.get_config")
    def test_check_data_quality_complex_dtype_skip(self, mock_get_config):
        """Test check_data_quality skips complex dtypes."""
        mock_config = Mock()
        mock_config.show_warnings = True
        mock_get_config.return_value = mock_config

        # Complex data type that should be skipped
        data = np.array([object(), object()])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidationMixin.check_data_quality(data, "test")
            # Should not raise warnings due to exception handling
            assert len(w) == 0


class TestValidateTreeParams:
    """Test validate_tree_params function."""

    def test_empty_params(self):
        """Test with empty parameters."""
        result = validate_tree_params("classification", {})
        assert not result

    def test_none_params(self):
        """Test with None parameters."""
        # The function signature expects Dict but handles None internally
        # This test checks the actual behavior when None is passed
        # even though it's not in the type signature
        result = validate_tree_params("classification", None)  # type: ignore
        assert not result

    def test_valid_params(self):
        """Test with valid parameters."""
        params = {
            "max_depth": 5,
            "min_samples_split": 10,
            "min_samples_leaf": 3,
            "random_state": 42,
        }
        result = validate_tree_params("classification", params)
        assert result == params

    def test_invalid_param_names(self):
        """Test with invalid parameter names."""
        params = {"invalid_param": 5, "another_invalid": 10}
        with pytest.raises(ConfigurationError, match="Invalid tree parameters"):
            validate_tree_params("classification", params)

    def test_invalid_max_depth_type(self):
        """Test with invalid max_depth type."""
        params = {"max_depth": "invalid"}
        with pytest.raises(ConfigurationError, match="max_depth must be a positive integer"):
            validate_tree_params("classification", params)

    def test_invalid_max_depth_value(self):
        """Test with invalid max_depth value."""
        params = {"max_depth": 0}
        with pytest.raises(ConfigurationError, match="max_depth must be a positive integer"):
            validate_tree_params("classification", params)

    def test_valid_max_depth_none(self):
        """Test with max_depth as None."""
        params = {"max_depth": None}
        result = validate_tree_params("classification", params)
        assert result == params

    def test_invalid_min_samples_split(self):
        """Test with invalid min_samples_split."""
        params = {"min_samples_split": 1}  # Must be >= 2
        with pytest.raises(ConfigurationError, match="min_samples_split must be an integer >= 2"):
            validate_tree_params("classification", params)

    def test_invalid_min_samples_leaf(self):
        """Test with invalid min_samples_leaf."""
        params = {"min_samples_leaf": 0}  # Must be >= 1
        with pytest.raises(ConfigurationError, match="min_samples_leaf must be a positive integer"):
            validate_tree_params("classification", params)

    def test_all_valid_params(self):
        """Test with all valid parameters."""
        params = {
            "max_depth": 5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": 42,
            "max_leaf_nodes": 100,
            "min_impurity_decrease": 0.0,
            "class_weight": "balanced",
            "ccp_alpha": 0.0,
            "criterion": "gini",
        }
        result = validate_tree_params("classification", params)
        assert result == params


class TestSuggestAlternatives:
    """Test suggest_alternatives function."""

    def test_supervised_alternatives(self):
        """Test alternatives for supervised method."""
        result = suggest_alternatives("tree")
        assert "supervised" in result
        assert "decision_tree" in result

    def test_equal_width_alternatives(self):
        """Test alternatives for equal_width method."""
        result = suggest_alternatives("uniform")
        assert "equal_width" in result
        assert "equidistant" in result

    def test_singleton_alternatives(self):
        """Test alternatives for singleton method."""
        result = suggest_alternatives("categorical")
        assert "singleton" in result
        assert "nominal" in result

    def test_quantile_alternatives(self):
        """Test alternatives for quantile method."""
        result = suggest_alternatives("percentile")
        assert "quantile" in result

    def test_unknown_method(self):
        """Test with unknown method."""
        result = suggest_alternatives("unknown_method")
        assert not result

    def test_exact_match(self):
        """Test with exact method name."""
        result = suggest_alternatives("supervised")
        assert "supervised" in result
        assert "tree" in result
        assert "decision_tree" in result

    def test_case_insensitive(self):
        """Test case insensitive matching."""
        result = suggest_alternatives("SUPERVISED")
        assert "supervised" in result

        result = suggest_alternatives("Tree")
        assert "supervised" in result
