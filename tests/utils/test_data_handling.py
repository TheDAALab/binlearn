"""Comprehensive tests for binlearn.utils._data_handling module.

This module tests all functions in the data handling utility module
to achieve 100% test coverage, including edge cases and error conditions.
"""

from unittest.mock import patch

import numpy as np
import pytest

# Import binlearn level config variables
from binlearn import PANDAS_AVAILABLE, POLARS_AVAILABLE, pd, pl
from binlearn.utils._data_handling import (
    _determine_columns,
    _is_pandas_df,
    _is_polars_df,
    convert_to_python_types,
    prepare_array,
    prepare_input_with_columns,
    return_like_input,
)


class TestConvertToPythonTypes:
    """Test suite for convert_to_python_types function."""

    def test_basic_python_types_passthrough(self):
        """Test that basic Python types pass through unchanged."""
        # Test basic types
        assert convert_to_python_types(42) == 42
        assert convert_to_python_types(3.14) == 3.14
        assert convert_to_python_types("hello") == "hello"
        assert convert_to_python_types(True) is True
        assert convert_to_python_types(None) is None

    def test_numpy_boolean_conversion(self):
        """Test conversion of numpy boolean types."""
        np_true = np.bool_(True)
        np_false = np.bool_(False)

        result_true = convert_to_python_types(np_true)
        result_false = convert_to_python_types(np_false)

        assert result_true is True
        assert result_false is False
        assert isinstance(result_true, bool)
        assert isinstance(result_false, bool)

    def test_numpy_integer_conversion(self):
        """Test conversion of numpy integer types."""
        # Test different numpy integer types
        np_int8 = np.int8(42)
        np_int16 = np.int16(1000)
        np_int32 = np.int32(100000)
        np_int64 = np.int64(10000000000)
        np_uint8 = np.uint8(255)

        assert convert_to_python_types(np_int8) == 42
        assert convert_to_python_types(np_int16) == 1000
        assert convert_to_python_types(np_int32) == 100000
        assert convert_to_python_types(np_int64) == 10000000000
        assert convert_to_python_types(np_uint8) == 255

        # Ensure they're actual Python ints
        assert isinstance(convert_to_python_types(np_int8), int)
        assert isinstance(convert_to_python_types(np_uint8), int)

    def test_numpy_floating_conversion(self):
        """Test conversion of numpy floating point types."""
        # Test different numpy float types
        np_float16 = np.float16(3.14)
        np_float32 = np.float32(2.718)
        np_float64 = np.float64(1.414)

        assert abs(convert_to_python_types(np_float16) - 3.14) < 0.01  # float16 has lower precision
        assert abs(convert_to_python_types(np_float32) - 2.718) < 0.001
        assert abs(convert_to_python_types(np_float64) - 1.414) < 0.001

        # Ensure they're actual Python floats
        assert isinstance(convert_to_python_types(np_float32), float)
        assert isinstance(convert_to_python_types(np_float64), float)

    def test_numpy_complex_conversion(self):
        """Test conversion of numpy complex types."""
        np_complex = np.complex64(1 + 2j)
        result = convert_to_python_types(np_complex)

        # Complex numbers use .item() method
        assert result == (1 + 2j)

    def test_numpy_array_conversion(self):
        """Test conversion of numpy arrays."""
        # 1D array
        arr_1d = np.array([1, 2, 3])
        result_1d = convert_to_python_types(arr_1d)
        assert result_1d == [1, 2, 3]
        assert isinstance(result_1d, list)

        # 2D array
        arr_2d = np.array([[1, 2], [3, 4]])
        result_2d = convert_to_python_types(arr_2d)
        assert result_2d == [[1, 2], [3, 4]]
        assert isinstance(result_2d, list)
        assert isinstance(result_2d[0], list)

        # Array with numpy types
        arr_mixed = np.array([np.int32(1), np.float64(2.5), np.bool_(True)])
        result_mixed = convert_to_python_types(arr_mixed)
        assert result_mixed == [1, 2.5, True]
        assert all(isinstance(x, (int | float | bool)) for x in result_mixed)

    def test_list_conversion(self):
        """Test conversion of lists."""
        # List with numpy types
        list_with_numpy = [np.int32(1), np.float64(2.5), np.bool_(True), "string"]
        result = convert_to_python_types(list_with_numpy)
        assert result == [1, 2.5, True, "string"]
        assert isinstance(result, list)

        # Nested list
        nested_list = [[np.int32(1), np.float64(2.5)], [np.bool_(False), "text"]]
        result_nested = convert_to_python_types(nested_list)
        assert result_nested == [[1, 2.5], [False, "text"]]
        assert isinstance(result_nested, list)
        assert isinstance(result_nested[0], list)

    def test_tuple_conversion(self):
        """Test conversion of tuples."""
        # Tuple with numpy types
        tuple_with_numpy = (np.int32(1), np.float64(2.5), np.bool_(True))
        result = convert_to_python_types(tuple_with_numpy)
        assert result == (1, 2.5, True)
        assert isinstance(result, tuple)

        # Nested tuple
        nested_tuple = ((np.int32(1), np.float64(2.5)), (np.bool_(False), np.int64(100)))
        result_nested = convert_to_python_types(nested_tuple)
        assert result_nested == ((1, 2.5), (False, 100))
        assert isinstance(result_nested, tuple)
        assert isinstance(result_nested[0], tuple)

    def test_dict_conversion(self):
        """Test conversion of dictionaries."""
        # Dict with numpy types
        dict_with_numpy = {
            "int": np.int32(42),
            "float": np.float64(3.14),
            "bool": np.bool_(True),
            "array": np.array([1, 2, 3]),
        }
        result = convert_to_python_types(dict_with_numpy)
        expected = {"int": 42, "float": 3.14, "bool": True, "array": [1, 2, 3]}
        assert result == expected
        assert isinstance(result, dict)
        assert isinstance(result["array"], list)

        # Nested dict
        nested_dict = {
            "outer": {"inner": np.int32(123), "list": [np.float64(1.1), np.bool_(False)]}
        }
        result_nested = convert_to_python_types(nested_dict)
        expected_nested = {"outer": {"inner": 123, "list": [1.1, False]}}
        assert result_nested == expected_nested

    def test_complex_nested_structure(self):
        """Test conversion of complex nested structures."""
        complex_data = {
            "array": np.array([[np.int32(1), np.float64(2.5)], [np.bool_(True), np.int64(100)]]),
            "list": [
                {"nested": np.array([np.float32(1.1), np.float32(2.2)])},
                (np.int16(42), np.bool_(False)),
            ],
            "tuple": (np.array([1, 2, 3]), {"key": np.float64(3.14159)}),
        }

        result = convert_to_python_types(complex_data)
        expected = {
            "array": [[1, 2.5], [True, 100]],
            "list": [
                {"nested": [1.1000000238418579, 2.200000047683716]},  # float32 precision
                (42, False),
            ],
            "tuple": ([1, 2, 3], {"key": 3.14159}),
        }

        assert result == expected
        assert isinstance(result["array"], list)
        assert isinstance(result["list"][1], tuple)
        assert isinstance(result["tuple"], tuple)

    def test_empty_containers(self):
        """Test conversion of empty containers."""
        # Empty containers
        assert convert_to_python_types([]) == []
        assert convert_to_python_types(()) == ()
        assert convert_to_python_types({}) == {}
        assert convert_to_python_types(np.array([])) == []

        # Empty nested containers
        nested_empty = {"empty_list": [], "empty_dict": {}, "empty_tuple": ()}
        result = convert_to_python_types(nested_empty)
        assert result == {"empty_list": [], "empty_dict": {}, "empty_tuple": ()}

    def test_special_numpy_values(self):
        """Test conversion of special numpy values."""
        # NaN and inf values
        nan_val = np.nan
        inf_val = np.inf
        ninf_val = -np.inf

        result_nan = convert_to_python_types(nan_val)
        result_inf = convert_to_python_types(inf_val)
        result_ninf = convert_to_python_types(ninf_val)

        assert np.isnan(result_nan)
        assert np.isinf(result_inf) and result_inf > 0
        assert np.isinf(result_ninf) and result_ninf < 0

        # Arrays with special values
        special_array = np.array([1.0, np.nan, np.inf, -np.inf])
        result_array = convert_to_python_types(special_array)
        assert len(result_array) == 4
        assert result_array[0] == 1.0
        assert np.isnan(result_array[1])
        assert np.isinf(result_array[2]) and result_array[2] > 0
        assert np.isinf(result_array[3]) and result_array[3] < 0

    def test_zero_dimensional_array(self):
        """Test conversion of zero-dimensional numpy arrays."""
        scalar_array = np.array(42)
        result = convert_to_python_types(scalar_array)
        assert result == 42  # Should become the scalar value directly
        assert isinstance(result, int)

    def test_large_numpy_values(self):
        """Test conversion of large numpy values."""
        # Test large integers
        large_int = np.int64(9223372036854775807)  # Max int64
        result_int = convert_to_python_types(large_int)
        assert result_int == 9223372036854775807
        assert isinstance(result_int, int)

        # Test very small float
        small_float = np.float64(1e-308)
        result_float = convert_to_python_types(small_float)
        assert result_float == 1e-308
        assert isinstance(result_float, float)


class TestPrepareArray:
    """Test suite for prepare_array function."""

    def test_numpy_array_input(self):
        """Test preparation of numpy array input."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        array, columns, index = prepare_array(data)

        np.testing.assert_array_equal(array, data)
        assert columns is None
        assert index is None

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_dataframe_input(self):
        """Test preparation of pandas DataFrame input."""
        data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])
        array, columns, index = prepare_array(data)

        np.testing.assert_array_equal(array, data.values)
        assert columns == ["A", "B", "C"]
        pd.testing.assert_index_equal(index, data.index)

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_series_input(self):
        """Test that pandas Series input is converted properly."""
        data = pd.Series([1, 2, 3, 4], name="test_series")
        array, columns, index = prepare_array(data)
        # Series should be converted to 2D array (column vector)
        assert array.shape == (4, 1)
        np.testing.assert_array_equal(array[:, 0], [1, 2, 3, 4])
        assert columns is None  # Series doesn't have column names like DataFrame
        assert index is None  # Series is treated like a regular array, not DataFrame

    def test_list_input(self):
        """Test preparation of list input."""
        data = [[1, 2, 3], [4, 5, 6]]
        array, columns, index = prepare_array(data)

        expected = np.array(data)
        np.testing.assert_array_equal(array, expected)
        assert columns is None
        assert index is None

    def test_1d_array_reshaping(self):
        """Test that 1D array input is reshaped to 2D (column vector)."""
        data = np.array([1, 2, 3, 4])
        array, columns, index = prepare_array(data)
        # 1D array should be reshaped to column vector
        assert array.shape == (4, 1)
        np.testing.assert_array_equal(array[:, 0], [1, 2, 3, 4])
        assert columns is None
        assert index is None

    def test_empty_input(self):
        """Test preparation of empty input."""
        data = np.array([])
        array, columns, index = prepare_array(data)

        assert array.shape[0] == 0
        assert columns is None
        assert index is None

    def test_single_value_input(self):
        """Test preparation of single value input."""
        data = 5
        array, columns, index = prepare_array(data)

        expected = np.array([[5]])
        np.testing.assert_array_equal(array, expected)
        assert columns is None
        assert index is None

    def test_0d_scalar_input(self):
        """Test preparation of 0-dimensional scalar input."""
        data = np.array(42)  # 0-dimensional array
        array, columns, index = prepare_array(data)

        expected = np.array([[42]])
        np.testing.assert_array_equal(array, expected)
        assert columns is None
        assert index is None


class TestReturnLikeInput:
    """Test suite for return_like_input function."""

    def test_return_numpy_array_no_preserve(self):
        """Test returning data as numpy array when preserve_dataframe=False."""
        original = np.array([[1, 2], [3, 4]])
        processed = np.array([[10, 20], [30, 40]])

        result = return_like_input(processed, original, None, preserve_dataframe=False)
        np.testing.assert_array_equal(result, processed)
        assert isinstance(result, np.ndarray)

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_return_pandas_dataframe_preserve(self):
        """Test returning data as pandas DataFrame when preserve_dataframe=True."""
        original = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        processed = np.array([[10, 20], [30, 40]])
        columns = ["A", "B"]

        result = return_like_input(processed, original, columns, preserve_dataframe=True)
        assert isinstance(result, pd.DataFrame)
        expected = pd.DataFrame(processed, columns=columns, index=original.index)
        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_return_pandas_dataframe_no_preserve(self):
        """Test returning numpy array even for DataFrame input when preserve_dataframe=False."""
        original = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        processed = np.array([[10, 20], [30, 40]])

        result = return_like_input(processed, original, ["A", "B"], preserve_dataframe=False)
        np.testing.assert_array_equal(result, processed)
        assert isinstance(result, np.ndarray)

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_return_with_none_columns(self):
        """Test returning DataFrame with None columns (uses original columns)."""
        original = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        processed = np.array([[10, 20], [30, 40]])

        result = return_like_input(processed, original, None, preserve_dataframe=True)
        assert isinstance(result, pd.DataFrame)
        expected = pd.DataFrame(processed, columns=["A", "B"], index=original.index)
        pd.testing.assert_frame_equal(result, expected)

    def test_return_non_dataframe_preserve_true(self):
        """Test returning numpy array for non-DataFrame input even when preserve_dataframe=True."""
        original = [[1, 2], [3, 4]]
        processed = np.array([[10, 20], [30, 40]])

        result = return_like_input(processed, original, None, preserve_dataframe=True)
        np.testing.assert_array_equal(result, processed)
        assert isinstance(result, np.ndarray)

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
    def test_return_polars_dataframe_preserve(self):
        """Test returning data as polars DataFrame when preserve_dataframe=True."""
        assert pl is not None

        original = pl.DataFrame({"A": [1, 2], "B": [3, 4]})
        processed = np.array([[10, 20], [30, 40]])
        columns = ["A", "B"]

        result = return_like_input(processed, original, columns, preserve_dataframe=True)

        # Should return polars DataFrame if module is available
        assert isinstance(result, pl.DataFrame)
        expected = pl.DataFrame(processed, schema=columns)
        assert result.equals(expected)

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
    def test_return_polars_dataframe_module_none(self):
        """Test polars DataFrame handling when polars module is None."""
        # Test the case where we have a polars DataFrame but the polars module
        # is None in _polars_config
        assert pl is not None

        # Create a polars DataFrame
        original = pl.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        processed = np.array([[10, 20], [30, 40]])

        # Mock _polars_config.pl to be None to simulate the case where polars module is None
        with patch("binlearn.utils._data_handling._polars_config") as mock_polars_config:
            mock_polars_config.pl = None  # Simulate polars module being None

            # This should fall back to returning the array since polars module is None
            result = return_like_input(processed, original, None, preserve_dataframe=True)
            np.testing.assert_array_equal(result, processed)
            assert isinstance(result, np.ndarray)

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_return_pandas_dataframe_module_none(self):
        """Test pandas DataFrame handling when pandas module is None."""
        # Test the case where we have a pandas DataFrame but the pandas module
        # is None in _pandas_config
        original = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        processed = np.array([[10, 20], [30, 40]])

        # Mock _pandas_config.pd to be None to simulate the case where pandas module is None
        with patch("binlearn.utils._data_handling._pandas_config") as mock_pandas_config:
            mock_pandas_config.pd = None  # Simulate pandas module being None

            # This should fall back to returning the array since pandas module is None
            result = return_like_input(processed, original, None, preserve_dataframe=True)
            np.testing.assert_array_equal(result, processed)
            assert isinstance(result, np.ndarray)


class TestPrepareInputWithColumns:
    """Test suite for prepare_input_with_columns function."""

    def test_numpy_array_no_fitted(self):
        """Test numpy array input when not fitted."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        array, columns = prepare_input_with_columns(data, fitted=False)

        np.testing.assert_array_equal(array, data)
        assert columns == [0, 1, 2]  # Numeric column indices

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_dataframe_no_fitted(self):
        """Test pandas DataFrame input when not fitted."""
        data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])
        array, columns = prepare_input_with_columns(data, fitted=False)

        np.testing.assert_array_equal(array, data.values)
        assert columns == ["A", "B", "C"]

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_series_input(self):
        """Test that pandas Series input is handled correctly."""
        data = pd.Series([1, 2, 3, 4], name="test_series")
        array, columns = prepare_input_with_columns(data, fitted=False)
        # Series should be converted to 2D array (column vector)
        assert array.shape == (4, 1)
        np.testing.assert_array_equal(array[:, 0], [1, 2, 3, 4])
        assert columns == [0]  # Single column gets index 0

    def test_list_input_no_fitted(self):
        """Test list input when not fitted."""
        data = [[1, 2, 3], [4, 5, 6]]
        array, columns = prepare_input_with_columns(data, fitted=False)

        expected = np.array(data)
        np.testing.assert_array_equal(array, expected)
        assert columns == [0, 1, 2]  # Numeric column indices

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_fitted_with_original_columns_dataframe(self):
        """Test fitted=True with original columns for DataFrame."""
        data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])
        original_columns = ["A", "B", "C"]

        array, columns = prepare_input_with_columns(
            data, fitted=True, original_columns=original_columns
        )

        np.testing.assert_array_equal(array, data.values)
        assert columns == ["A", "B", "C"]

    def test_fitted_with_original_columns_numpy(self):
        """Test fitted=True with original columns for numpy array."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        original_columns = ["col1", "col2", "col3"]

        array, columns = prepare_input_with_columns(
            data, fitted=True, original_columns=original_columns
        )

        np.testing.assert_array_equal(array, data)
        assert columns == ["col1", "col2", "col3"]

    def test_fitted_no_original_columns(self):
        """Test fitted=True without original columns."""
        data = np.array([[1, 2, 3], [4, 5, 6]])

        array, columns = prepare_input_with_columns(data, fitted=True, original_columns=None)

        np.testing.assert_array_equal(array, data)
        assert columns == [0, 1, 2]  # Falls back to numeric indices

    def test_1d_array_reshaping(self):
        """Test that 1D array input is reshaped to 2D."""
        data = np.array([1, 2, 3, 4])
        array, columns = prepare_input_with_columns(data, fitted=False)
        # 1D array should be reshaped to column vector
        assert array.shape == (4, 1)
        np.testing.assert_array_equal(array[:, 0], [1, 2, 3, 4])
        assert columns == [0]  # Single column gets index 0

    def test_single_value_input(self):
        """Test single value input."""
        data = 42

        array, columns = prepare_input_with_columns(data, fitted=False)

        expected = np.array([[42]])
        np.testing.assert_array_equal(array, expected)
        assert columns == [0]


class TestHelperFunctions:
    """Test suite for helper functions and edge cases."""

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_df_detection(self):
        """Test pandas DataFrame detection."""

        # Test with DataFrame
        df = pd.DataFrame([[1, 2], [3, 4]])
        assert _is_pandas_df(df) is True

        # Test with non-DataFrame
        arr = np.array([[1, 2], [3, 4]])
        assert _is_pandas_df(arr) is False

        # Test with None
        assert _is_pandas_df(None) is False

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
    def test_polars_df_detection(self):
        """Test polars DataFrame detection (if available)."""

        # Test with numpy array (should be False)
        arr = np.array([[1, 2], [3, 4]])
        assert _is_polars_df(arr) is False

        # Test with pandas DataFrame (should be False)
        df = pd.DataFrame([[1, 2], [3, 4]])
        assert _is_polars_df(df) is False

        # Test with None
        assert _is_polars_df(None) is False

        assert pl is not None

        # Create a polars DataFrame and test
        polars_df = pl.DataFrame({"A": [1, 2], "B": [3, 4]})
        assert _is_polars_df(polars_df) is True

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
    def test_polars_dataframe_input(self):
        """Test polars DataFrame input (if available)."""

        assert pl is not None

        # Create polars DataFrame
        data = pl.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        array, columns, index = prepare_array(data)

        # Should convert to numpy array
        expected = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        np.testing.assert_array_equal(array, expected)
        assert columns == ["A", "B", "C"]
        assert index is None  # Polars returns None for index

    def test_determine_columns_helper(self):
        """Test the _determine_columns helper function."""

        # Test with col_names provided
        result = _determine_columns(
            X=None, col_names=["A", "B"], fitted=False, original_columns=None, arr_shape=(10, 2)
        )
        assert result == ["A", "B"]

        # Test with fitted and original_columns
        result = _determine_columns(
            X=None, col_names=None, fitted=True, original_columns=["X", "Y"], arr_shape=(10, 2)
        )
        assert result == ["X", "Y"]

        # Test fallback to numeric indices
        result = _determine_columns(
            X=None, col_names=None, fitted=False, original_columns=None, arr_shape=(10, 3)
        )
        assert result == [0, 1, 2]


class TestIntegrationScenarios:
    """Test suite for integration scenarios combining multiple functions."""

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_complete_workflow_pandas_dataframe(self):
        """Test complete workflow with pandas DataFrame."""
        # Original data
        original = pd.DataFrame(
            [[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"], index=["row1", "row2"]
        )

        # Prepare array
        array, columns, _ = prepare_array(original)

        # Simulate processing (e.g., binning)
        processed = array * 10

        # Return in original format
        result = return_like_input(processed, original, columns, preserve_dataframe=True)

        assert isinstance(result, pd.DataFrame)
        expected = pd.DataFrame(
            [[10, 20, 30], [40, 50, 60]], columns=["A", "B", "C"], index=["row1", "row2"]
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_complete_workflow_numpy_array(self):
        """Test complete workflow with numpy array."""
        # Original data
        original = np.array([[1, 2], [3, 4]])

        # Prepare array
        array, columns, _ = prepare_array(original)

        # Simulate processing
        processed = array + 10

        # Return in original format (as numpy array)
        result = return_like_input(processed, original, columns, preserve_dataframe=False)

        assert isinstance(result, np.ndarray)
        expected = np.array([[11, 12], [13, 14]])
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_prepare_input_with_columns_workflow(self):
        """Test workflow with prepare_input_with_columns."""
        # Original data
        original = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=["A", "B", "C", "D"])

        # Prepare input with columns (simulating unfitted estimator)
        _, columns = prepare_input_with_columns(original, fitted=False)

        # Simulate fitting and storing columns
        fitted_columns = columns.copy()

        # New data for prediction (simulating fitted estimator)
        new_data = pd.DataFrame([[10, 20, 30, 40]], columns=["A", "B", "C", "D"])
        new_array, new_columns = prepare_input_with_columns(
            new_data, fitted=True, original_columns=fitted_columns
        )

        # Should preserve column order from fitting
        assert new_columns == fitted_columns
        np.testing.assert_array_equal(new_array, new_data.values)

    def test_mixed_input_types_handling(self):
        """Test handling of various input types."""
        inputs = [
            np.array([[1, 2], [3, 4]]),  # 2D array (should work)
            [[1, 2], [3, 4]],  # List (should work)
            42,  # Scalar (should work)
            np.array([1, 2, 3]),  # 1D array (gets reshaped)
        ]

        for input_data in inputs:
            array, _, _ = prepare_array(input_data)
            # Should always return 2D array
            assert len(array.shape) == 2
            assert array.shape[0] >= 1  # At least one row

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_mixed_input_types_with_pandas_series(self):
        """Test handling of pandas Series input."""
        data = pd.Series([5, 6, 7])  # Series (gets reshaped)
        array, _, _ = prepare_array(data)
        # Should always return 2D array
        assert len(array.shape) == 2
        assert array.shape[0] >= 1  # At least one row

    def test_error_handling_edge_cases(self):
        """Test error handling for edge cases."""
        # Test with None input - np.asarray(None) creates array(None) which has shape ()
        # This might not raise an error but creates a 0-dimensional array
        result_array, columns, index = prepare_array(None)
        # The result should be reshaped to (1, 1) since 0-dimensional arrays get reshaped
        assert result_array.shape == (1, 1)
        assert columns is None
        assert index is None

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_preserve_data_types(self):
        """Test that data types are preserved where possible."""
        # Integer data
        int_data = pd.DataFrame([[1, 2], [3, 4]], dtype=int)
        array, columns, _ = prepare_array(int_data)
        result = return_like_input(array, int_data, columns, preserve_dataframe=True)

        assert isinstance(result, pd.DataFrame)
        # Note: dtype preservation might not always be exact after processing

        # Float data
        float_data = pd.DataFrame([[1.1, 2.2], [3.3, 4.4]], dtype=float)
        array, columns, _ = prepare_array(float_data)
        result = return_like_input(array, float_data, columns, preserve_dataframe=True)

        assert isinstance(result, pd.DataFrame)

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_large_data_handling(self):
        """Test handling of larger datasets."""
        # Create moderately large dataset
        large_data = pd.DataFrame(
            np.random.randn(1000, 10), columns=[f"col_{i}" for i in range(10)]
        )

        # Should handle without issues
        array, columns, _ = prepare_array(large_data)
        assert array.shape == (1000, 10)
        assert columns is not None and len(columns) == 10

        # Test prepare_input_with_columns on large data
        prepared_array, prepared_columns = prepare_input_with_columns(large_data, fitted=False)
        assert prepared_array.shape == (1000, 10)
        assert prepared_columns == [f"col_{i}" for i in range(10)]

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame(columns=["A", "B", "C"])

        array, columns, _ = prepare_array(empty_df)
        assert array.shape == (0, 3)
        assert columns == ["A", "B", "C"]

        # Test prepare_input_with_columns
        prep_array, prep_columns = prepare_input_with_columns(empty_df, fitted=False)
        assert prep_array.shape == (0, 3)
        assert prep_columns == ["A", "B", "C"]

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_single_column_dataframe(self):
        """Test handling of single-column DataFrames."""
        single_col_df = pd.DataFrame([1, 2, 3], columns=["single"])

        array, columns, _ = prepare_array(single_col_df)
        assert array.shape == (3, 1)
        assert columns == ["single"]

        # Test return_like_input
        processed = array * 2
        result = return_like_input(processed, single_col_df, columns, preserve_dataframe=True)

        assert isinstance(result, pd.DataFrame)
        expected = pd.DataFrame([2, 4, 6], columns=["single"])
        pd.testing.assert_frame_equal(result, expected)
