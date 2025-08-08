"""Comprehensive tests for binlearn.base._data_handling_base module.

This module tests all functionality in the DataHandlingBase class
to achieve 100% test coverage, including edge cases and error conditions.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

# Import binlearn level config variables
from binlearn import PANDAS_AVAILABLE, pd
from binlearn.base._data_handling_base import DataHandlingBase


class MockDataHandler(DataHandlingBase):
    """Mock data handler for testing DataHandlingBase functionality."""

    def __init__(self, preserve_dataframe=False):
        """Initialize mock data handler with test parameters."""
        super().__init__()
        self.preserve_dataframe = preserve_dataframe
        self._fitted_attributes = ["test_fitted_"]

    def fit(self, X):
        """Mock fit method."""
        self.test_fitted_ = "fitted_value"
        return self

    def get_input_columns(self):
        """Mock implementation of get_input_columns."""
        return ["col1", "col2", "col3"] if hasattr(self, "test_fitted_") else None

    def _get_binning_columns(self):
        """Mock binning columns method."""
        return ["col1", "col2", "col3"]


class TestDataHandlingBase:
    """Test suite for DataHandlingBase class."""

    def test_initialization(self):
        """Test basic initialization of DataHandlingBase."""
        handler = DataHandlingBase()
        assert hasattr(handler, "_original_columns")
        assert handler._original_columns is None
        assert hasattr(handler, "_feature_names_in")
        assert handler._feature_names_in is None

    def test_mock_initialization(self):
        """Test initialization of mock data handler."""
        handler = MockDataHandler(preserve_dataframe=True)
        assert handler.preserve_dataframe is True
        assert hasattr(handler, "_fitted_attributes")
        assert handler._fitted_attributes == ["test_fitted_"]

    def test_get_input_columns_default(self):
        """Test default get_input_columns implementation."""
        handler = DataHandlingBase()
        result = handler.get_input_columns()
        assert result is None

    def test_get_input_columns_mock_implementation(self):
        """Test mock implementation of get_input_columns."""
        handler = MockDataHandler()

        # Not fitted - should return None
        assert handler.get_input_columns() is None

        # Fitted - should return columns
        handler.fit([[1, 2, 3], [4, 5, 6]])
        assert handler.get_input_columns() == ["col1", "col2", "col3"]

    def test_prepare_input_not_fitted(self):
        """Test _prepare_input when not fitted."""
        handler = MockDataHandler()
        data = np.array([[1, 2, 3], [4, 5, 6]])

        array, columns = handler._prepare_input(data)

        np.testing.assert_array_equal(array, data)
        assert columns == [0, 1, 2]  # Numeric indices for unfitted

    def test_prepare_input_fitted_no_binning_columns(self):
        """Test _prepare_input when fitted but no _get_binning_columns method."""
        handler = MockDataHandler()
        handler.test_fitted_ = "fitted"

        # Create a handler without the _get_binning_columns method
        class HandlerNoBinning(DataHandlingBase):
            def __init__(self):
                super().__init__()
                self._fitted_attributes = ["test_fitted_"]
                self.test_fitted_ = "fitted"

        handler_no_binning = HandlerNoBinning()

        data = np.array([[1, 2, 3], [4, 5, 6]])
        array, columns = handler_no_binning._prepare_input(data)

        np.testing.assert_array_equal(array, data)
        assert columns == [0, 1, 2]  # Should fallback to numeric indices

    def test_prepare_input_fitted_non_callable_binning_columns(self):
        """Test _prepare_input when _get_binning_columns exists but is not callable."""
        handler = MockDataHandler()
        handler.test_fitted_ = "fitted"

        # Use setattr to set _get_binning_columns to non-callable
        handler._get_binning_columns = "not_callable"  # type: ignore

        data = np.array([[1, 2, 3], [4, 5, 6]])
        array, columns = handler._prepare_input(data)

        np.testing.assert_array_equal(array, data)
        assert columns == [0, 1, 2]  # Should fallback to numeric indices

    def test_prepare_input_fitted_with_binning_columns(self):
        """Test _prepare_input when fitted with valid _get_binning_columns."""
        handler = MockDataHandler()
        handler.fit([[1, 2, 3], [4, 5, 6]])

        data = np.array([[1, 2, 3], [4, 5, 6]])
        array, columns = handler._prepare_input(data)

        np.testing.assert_array_equal(array, data)
        assert columns == ["col1", "col2", "col3"]

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_prepare_input_pandas_dataframe(self):
        """Test _prepare_input with pandas DataFrame."""
        handler = MockDataHandler()
        data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])

        array, columns = handler._prepare_input(data)

        np.testing.assert_array_equal(array, data.values)
        assert columns == ["A", "B", "C"]

    def test_validate_and_prepare_input_valid_data(self):
        """Test _validate_and_prepare_input with valid data."""
        handler = MockDataHandler()
        data = np.array([[1, 2, 3], [4, 5, 6]])

        result = handler._validate_and_prepare_input(data, "test_data")

        np.testing.assert_array_equal(result, data)

    def test_validate_and_prepare_input_custom_name(self):
        """Test _validate_and_prepare_input with custom parameter name."""
        handler = MockDataHandler()
        data = [[1, 2, 3], [4, 5, 6]]

        result = handler._validate_and_prepare_input(data, "custom_X")

        expected = np.array(data)
        np.testing.assert_array_equal(result, expected)

    def test_validate_and_prepare_input_none_result(self):
        """Test _validate_and_prepare_input when validation returns None."""
        handler = MockDataHandler()

        # Mock validate_array_like to return None
        with patch.object(handler, "validate_array_like", return_value=None):
            result = handler._validate_and_prepare_input([[1, 2, 3]], "test")
            assert result is None

    def test_format_output_like_input_default_preserve(self):
        """Test _format_output_like_input with default preserve_dataframe."""
        handler = MockDataHandler(preserve_dataframe=False)
        original = np.array([[1, 2], [3, 4]])
        result = np.array([[10, 20], [30, 40]])
        columns = [0, 1]

        output = handler._format_output_like_input(result, original, columns)

        np.testing.assert_array_equal(output, result)
        assert isinstance(output, np.ndarray)

    def test_format_output_like_input_explicit_preserve_none(self):
        """Test _format_output_like_input with explicit preserve_dataframe=None."""
        handler = MockDataHandler(preserve_dataframe=True)
        original = np.array([[1, 2], [3, 4]])
        result = np.array([[10, 20], [30, 40]])
        columns = [0, 1]

        output = handler._format_output_like_input(
            result, original, columns, preserve_dataframe=None
        )

        np.testing.assert_array_equal(output, result)

    def test_format_output_like_input_explicit_preserve_true(self):
        """Test _format_output_like_input with explicit preserve_dataframe=True."""
        handler = MockDataHandler(preserve_dataframe=False)
        original = np.array([[1, 2], [3, 4]])
        result = np.array([[10, 20], [30, 40]])
        columns = [0, 1]

        output = handler._format_output_like_input(
            result, original, columns, preserve_dataframe=True
        )

        np.testing.assert_array_equal(output, result)

    def test_format_output_like_input_explicit_preserve_false(self):
        """Test _format_output_like_input with explicit preserve_dataframe=False."""
        handler = MockDataHandler(preserve_dataframe=True)
        original = np.array([[1, 2], [3, 4]])
        result = np.array([[10, 20], [30, 40]])
        columns = [0, 1]

        output = handler._format_output_like_input(
            result, original, columns, preserve_dataframe=False
        )

        np.testing.assert_array_equal(output, result)
        assert isinstance(output, np.ndarray)

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_format_output_like_input_preserve_dataframe(self):
        """Test _format_output_like_input preserving DataFrame format."""
        handler = MockDataHandler(preserve_dataframe=True)
        original = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        result = np.array([[10, 20], [30, 40]])
        columns = ["A", "B"]

        output = handler._format_output_like_input(result, original, columns)

        assert isinstance(output, pd.DataFrame)
        expected = pd.DataFrame(result, columns=columns, index=original.index)
        pd.testing.assert_frame_equal(output, expected)

    def test_format_output_no_preserve_dataframe_attribute(self):
        """Test _format_output_like_input when handler has no preserve_dataframe attribute."""
        handler = DataHandlingBase()  # No preserve_dataframe attribute
        original = np.array([[1, 2], [3, 4]])
        result = np.array([[10, 20], [30, 40]])
        columns = [0, 1]

        output = handler._format_output_like_input(result, original, columns)

        np.testing.assert_array_equal(output, result)
        assert isinstance(output, np.ndarray)

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_extract_and_validate_feature_info_dataframe_columns(self):
        """Test _extract_and_validate_feature_info with DataFrame columns."""
        handler = MockDataHandler()
        data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])

        feature_names = handler._extract_and_validate_feature_info(data)

        assert feature_names == ["A", "B", "C"]
        assert handler._feature_names_in is None  # Not reset by default

    def test_extract_and_validate_feature_info_with_feature_names_attr(self):
        """Test _extract_and_validate_feature_info with feature_names attribute."""
        handler = MockDataHandler()

        # Create mock object with feature_names attribute
        mock_data = Mock()
        mock_data.feature_names = ["feat1", "feat2", "feat3"]
        # Mock should not have columns attribute
        del mock_data.columns

        feature_names = handler._extract_and_validate_feature_info(mock_data)

        assert feature_names == ["feat1", "feat2", "feat3"]

    def test_extract_and_validate_feature_info_array_with_shape(self):
        """Test _extract_and_validate_feature_info with array having shape."""
        handler = MockDataHandler()
        data = np.array([[1, 2, 3], [4, 5, 6]])

        feature_names = handler._extract_and_validate_feature_info(data)

        assert feature_names == ["feature_0", "feature_1", "feature_2"]

    def test_extract_and_validate_feature_info_list_input(self):
        """Test _extract_and_validate_feature_info with list input."""
        handler = MockDataHandler()
        data = [[1, 2, 3], [4, 5, 6]]

        feature_names = handler._extract_and_validate_feature_info(data)

        assert feature_names == ["feature_0", "feature_1", "feature_2"]

    def test_extract_and_validate_feature_info_with_reset(self):
        """Test _extract_and_validate_feature_info with reset=True."""
        handler = MockDataHandler()
        data = np.array([[1, 2, 3], [4, 5, 6]])

        feature_names = handler._extract_and_validate_feature_info(data, reset=True)

        assert feature_names == ["feature_0", "feature_1", "feature_2"]
        assert handler._feature_names_in == ["feature_0", "feature_1", "feature_2"]
        assert handler._n_features_in == 3

    def test_feature_names_in_property_getter(self):
        """Test feature_names_in_ property getter."""
        handler = MockDataHandler()

        # No _feature_names_in attribute
        assert handler.feature_names_in_ is None

        # Set _feature_names_in attribute
        handler._feature_names_in = ["A", "B", "C"]
        assert handler.feature_names_in_ == ["A", "B", "C"]

    def test_feature_names_in_property_setter(self):
        """Test feature_names_in_ property setter."""
        handler = MockDataHandler()

        handler.feature_names_in_ = ["X", "Y", "Z"]

        assert handler._feature_names_in == ["X", "Y", "Z"]
        assert handler.feature_names_in_ == ["X", "Y", "Z"]

    def test_feature_names_in_property_setter_none(self):
        """Test feature_names_in_ property setter with None."""
        handler = MockDataHandler()
        handler._feature_names_in = ["A", "B", "C"]

        handler.feature_names_in_ = None

        assert handler._feature_names_in is None
        assert handler.feature_names_in_ is None

    def test_n_features_in_property_getter_with_names(self):
        """Test n_features_in_ property getter with feature names."""
        handler = MockDataHandler()
        handler._feature_names_in = ["A", "B", "C", "D"]

        assert handler.n_features_in_ == 4

    def test_n_features_in_property_getter_no_names(self):
        """Test n_features_in_ property getter without feature names."""
        handler = MockDataHandler()

        assert handler.n_features_in_ == 0

    def test_n_features_in_property_getter_none_names(self):
        """Test n_features_in_ property getter with None feature names."""
        handler = MockDataHandler()
        handler._feature_names_in = None

        assert handler.n_features_in_ == 0

    def test_n_features_in_property_setter(self):
        """Test n_features_in_ property setter."""
        handler = MockDataHandler()

        handler.n_features_in_ = 5

        assert handler._n_features_in == 5

    def test_validate_numeric_data_valid_integer(self):
        """Test _validate_numeric_data with valid integer data."""
        handler = MockDataHandler()
        data = np.array([1, 2, 3, 4, 5], dtype=int)

        # Should not raise
        handler._validate_numeric_data(data, "test_col")

    def test_validate_numeric_data_valid_float(self):
        """Test _validate_numeric_data with valid float data."""
        handler = MockDataHandler()
        data = np.array([1.1, 2.2, 3.3, np.nan, np.inf], dtype=float)

        # Should not raise (NaN and inf are allowed)
        handler._validate_numeric_data(data, "test_col")

    def test_validate_numeric_data_valid_unsigned(self):
        """Test _validate_numeric_data with valid unsigned integer data."""
        handler = MockDataHandler()
        data = np.array([1, 2, 3, 4, 5], dtype=np.uint32)

        # Should not raise
        handler._validate_numeric_data(data, "test_col")

    def test_validate_numeric_data_valid_complex(self):
        """Test _validate_numeric_data with valid complex data."""
        handler = MockDataHandler()
        data = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=complex)

        # Should not raise
        handler._validate_numeric_data(data, "test_col")

    def test_validate_numeric_data_invalid_string(self):
        """Test _validate_numeric_data with invalid string data."""
        handler = MockDataHandler()
        data = np.array(["a", "b", "c"], dtype=object)

        with pytest.raises(ValueError, match="Column test_col contains non-numeric data"):
            handler._validate_numeric_data(data, "test_col")

    def test_validate_numeric_data_invalid_object(self):
        """Test _validate_numeric_data with invalid object data."""
        handler = MockDataHandler()
        data = np.array([{"a": 1}, {"b": 2}], dtype=object)

        with pytest.raises(ValueError, match="Column test_col contains non-numeric data"):
            handler._validate_numeric_data(data, "test_col")

    def test_validate_numeric_data_invalid_boolean(self):
        """Test _validate_numeric_data with boolean data (should fail)."""
        handler = MockDataHandler()
        data = np.array([True, False, True], dtype=bool)

        with pytest.raises(ValueError, match="Column test_col contains non-numeric data"):
            handler._validate_numeric_data(data, "test_col")

    def test_validate_all_numeric_data_valid(self):
        """Test _validate_all_numeric_data with valid numeric data."""
        handler = MockDataHandler()
        data = np.array([[1, 2.5, 3], [4, 5.5, 6]], dtype=float)

        # Should not raise
        handler._validate_all_numeric_data(data)

    def test_validate_all_numeric_data_with_column_names(self):
        """Test _validate_all_numeric_data with provided column names."""
        handler = MockDataHandler()
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=int)
        column_names = ["A", "B", "C"]

        # Should not raise
        handler._validate_all_numeric_data(data, column_names)

    def test_validate_all_numeric_data_invalid_column(self):
        """Test _validate_all_numeric_data with one invalid column."""
        handler = MockDataHandler()
        # Create mixed data - first column numeric, second column string
        data = np.array([[1, "a"], [2, "b"]], dtype=object)

        with pytest.raises(ValueError, match="Column column_0 contains non-numeric data"):
            handler._validate_all_numeric_data(data)

    def test_validate_all_numeric_data_invalid_with_names(self):
        """Test _validate_all_numeric_data with invalid data and custom column names."""
        handler = MockDataHandler()
        data = np.array([[1, "a"], [2, "b"]], dtype=object)
        column_names = ["num_col", "str_col"]

        with pytest.raises(ValueError, match="Column num_col contains non-numeric data"):
            handler._validate_all_numeric_data(data, column_names)

    def test_validate_all_numeric_data_fewer_names_than_columns(self):
        """Test _validate_all_numeric_data with fewer column names than columns."""
        handler = MockDataHandler()
        data = np.array([[1, 2, "c"], [4, 5, "d"]], dtype=object)
        column_names = ["A", "B"]  # Only 2 names for 3 columns

        with pytest.raises(ValueError, match="Column A contains non-numeric data"):
            handler._validate_all_numeric_data(data, column_names)

    def test_validate_all_numeric_data_more_names_than_columns(self):
        """Test _validate_all_numeric_data with more column names than columns."""
        handler = MockDataHandler()
        data = np.array([[1, "b"], [4, "e"]], dtype=object)
        column_names = ["A", "B", "C", "D"]  # More names than columns

        with pytest.raises(ValueError, match="Column A contains non-numeric data"):
            handler._validate_all_numeric_data(data, column_names)

    def test_inheritance_from_base_classes(self):
        """Test that DataHandlingBase properly inherits from base classes."""
        handler = DataHandlingBase()

        # Should inherit from SklearnIntegrationBase
        assert hasattr(handler, "get_params")
        assert hasattr(handler, "set_params")
        assert hasattr(handler, "_fitted")

        # Should inherit from ValidationMixin
        assert hasattr(handler, "validate_array_like")
        assert hasattr(handler, "validate_column_specification")

    def test_integration_with_validation_mixin(self):
        """Test integration with ValidationMixin methods."""
        handler = MockDataHandler()

        # Test that validation methods work
        valid_data = np.array([[1, 2, 3], [4, 5, 6]])
        result = handler.validate_array_like(valid_data, "test_data")
        np.testing.assert_array_equal(result, valid_data)

    def test_integration_with_sklearn_base(self):
        """Test integration with SklearnIntegrationBase methods."""
        handler = MockDataHandler()

        # Test get_params
        params = handler.get_params()
        assert "preserve_dataframe" in params
        assert params["preserve_dataframe"] is False

        # Test set_params
        handler.set_params(preserve_dataframe=True)
        assert handler.preserve_dataframe is True

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_complete_workflow_with_dataframe(self):
        """Test complete workflow with pandas DataFrame."""
        handler = MockDataHandler(preserve_dataframe=True)

        # Input data
        input_data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])

        # Prepare input
        array, columns = handler._prepare_input(input_data)

        # Validate input
        handler._validate_and_prepare_input(input_data, "X")

        # Extract feature info
        feature_names = handler._extract_and_validate_feature_info(input_data, reset=True)

        # Process data (simulate transformation)
        result = array * 10

        # Format output
        output = handler._format_output_like_input(result, input_data, columns)

        # Verify results
        assert isinstance(output, pd.DataFrame)
        assert output.columns.tolist() == ["A", "B", "C"]
        assert feature_names == ["A", "B", "C"]
        assert handler.feature_names_in_ == ["A", "B", "C"]
        assert handler.n_features_in_ == 3

        expected_output = pd.DataFrame([[10, 20, 30], [40, 50, 60]], columns=["A", "B", "C"])
        pd.testing.assert_frame_equal(output, expected_output)

    def test_edge_case_empty_array(self):
        """Test handling of empty arrays."""
        handler = MockDataHandler()

        empty_data = np.array([]).reshape(0, 3)
        array, columns = handler._prepare_input(empty_data)

        assert array.shape == (0, 3)
        assert columns == [0, 1, 2]

    def test_edge_case_single_row(self):
        """Test handling of single row data."""
        handler = MockDataHandler()

        single_row = np.array([[1, 2, 3]])
        array, columns = handler._prepare_input(single_row)

        np.testing.assert_array_equal(array, single_row)
        assert columns == [0, 1, 2]

    def test_edge_case_single_column(self):
        """Test handling of single column data."""
        handler = MockDataHandler()

        single_col = np.array([[1], [2], [3]])
        array, columns = handler._prepare_input(single_col)

        np.testing.assert_array_equal(array, single_col)
        assert columns == [0]

    def test_mock_fitted_attribute_check(self):
        """Test that fitted attribute checking works properly."""
        handler = MockDataHandler()

        # Not fitted initially
        assert not handler._fitted

        # Fit the handler
        handler.fit([[1, 2, 3]])
        assert handler._fitted
        assert handler.test_fitted_ == "fitted_value"
