"""Comprehensive tests for binlearn.utils._data_handling module.

This module tests all functions in the data handling utility module
to achieve 100% test coverage, including edge cases and error conditions.
"""

import numpy as np
import pandas as pd
import pytest

from binlearn.utils._data_handling import (
    prepare_array,
    return_like_input,
    prepare_input_with_columns,
)


class TestPrepareArray:
    """Test suite for prepare_array function."""

    def test_numpy_array_input(self):
        """Test handling of numpy array input."""
        # 2D numpy array
        X = np.array([[1, 2], [3, 4], [5, 6]])
        result = prepare_array(X)
        np.testing.assert_array_equal(result, X)
        assert isinstance(result, np.ndarray)

        # 1D numpy array should be reshaped to 2D
        X_1d = np.array([1, 2, 3])
        result_1d = prepare_array(X_1d)
        expected_1d = X_1d.reshape(-1, 1)
        np.testing.assert_array_equal(result_1d, expected_1d)
        assert result_1d.shape == (3, 1)

    def test_pandas_dataframe_input(self):
        """Test handling of pandas DataFrame input."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        result = prepare_array(df)
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(result, expected)
        assert isinstance(result, np.ndarray)

    def test_pandas_series_input(self):
        """Test handling of pandas Series input."""
        series = pd.Series([1, 2, 3], name="test_series")
        result = prepare_array(series)
        expected = np.array([[1], [2], [3]])
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (3, 1)

    def test_list_input(self):
        """Test handling of list input."""
        # 2D list
        X_list = [[1, 2], [3, 4], [5, 6]]
        result = prepare_array(X_list)
        expected = np.array([[1, 2], [3, 4], [5, 6]])
        np.testing.assert_array_equal(result, expected)
        assert isinstance(result, np.ndarray)

        # 1D list
        X_list_1d = [1, 2, 3]
        result_1d = prepare_array(X_list_1d)
        expected_1d = np.array([[1], [2], [3]])
        np.testing.assert_array_equal(result_1d, expected_1d)
        assert result_1d.shape == (3, 1)

    def test_nested_list_input(self):
        """Test handling of nested list input with different structures."""
        # Irregular nested list (should still work with numpy)
        X_irregular = [[1, 2, 3], [4, 5], [6]]
        try:
            result = prepare_array(X_irregular)
            # This might create an object array or raise an error depending on numpy version
        except (ValueError, TypeError):
            # This is acceptable for irregular nested structures
            pass

    def test_single_value_input(self):
        """Test handling of single value input."""
        # Single number should be converted to 2D array
        result = prepare_array(5)
        expected = np.array([[5]])
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (1, 1)

        # Single float
        result_float = prepare_array(5.5)
        expected_float = np.array([[5.5]])
        np.testing.assert_array_equal(result_float, expected_float)

    def test_empty_input(self):
        """Test handling of empty input."""
        # Empty numpy array
        empty_array = np.array([])
        result = prepare_array(empty_array)
        assert result.shape[0] == 0

        # Empty list
        empty_list = []
        result_list = prepare_array(empty_list)
        assert result_list.shape[0] == 0

        # Empty DataFrame
        empty_df = pd.DataFrame()
        result_df = prepare_array(empty_df)
        assert result_df.shape[0] == 0

    def test_mixed_data_types(self):
        """Test handling of mixed data types."""
        # DataFrame with mixed types
        df_mixed = pd.DataFrame(
            {"int_col": [1, 2, 3], "float_col": [1.1, 2.2, 3.3], "str_col": ["a", "b", "c"]}
        )
        result = prepare_array(df_mixed)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)

        # The string column should be converted appropriately
        assert result.dtype == object or result.dtype.char == "U"

    def test_boolean_data(self):
        """Test handling of boolean data."""
        bool_data = [[True, False], [False, True]]
        result = prepare_array(bool_data)
        expected = np.array([[True, False], [False, True]])
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == bool

    def test_large_array_handling(self):
        """Test handling of large arrays."""
        large_array = np.random.rand(1000, 50)
        result = prepare_array(large_array)
        np.testing.assert_array_equal(result, large_array)
        assert result.shape == (1000, 50)

    def test_preserve_data_type(self):
        """Test that data types are preserved when appropriate."""
        # Integer array
        int_array = np.array([[1, 2], [3, 4]], dtype=np.int32)
        result_int = prepare_array(int_array)
        assert result_int.dtype == np.int32

        # Float array
        float_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        result_float = prepare_array(float_array)
        assert result_float.dtype == np.float64

    def test_special_values(self):
        """Test handling of special values like NaN and inf."""
        special_array = np.array([[1.0, np.nan], [np.inf, -np.inf]])
        result = prepare_array(special_array)
        np.testing.assert_array_equal(result, special_array)
        assert np.isnan(result[0, 1])
        assert np.isinf(result[1, 0])
        assert np.isinf(result[1, 1])


class TestReturnLikeInput:
    """Test suite for return_like_input function."""

    def test_return_numpy_array(self):
        """Test returning data in numpy array format."""
        original = np.array([[1, 2], [3, 4]])
        processed = np.array([[10, 20], [30, 40]])
        result = return_like_input(processed, original)

        np.testing.assert_array_equal(result, processed)
        assert isinstance(result, np.ndarray)

    def test_return_pandas_dataframe(self):
        """Test returning data in pandas DataFrame format."""
        original = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        processed = np.array([[10, 20], [30, 40]])
        result = return_like_input(processed, original)

        assert isinstance(result, pd.DataFrame)
        expected_df = pd.DataFrame(processed, columns=["A", "B"], index=original.index)
        pd.testing.assert_frame_equal(result, expected_df)

    def test_return_pandas_series(self):
        """Test returning data in pandas Series format."""
        original = pd.Series([1, 2, 3], name="test_series")
        processed = np.array([[10], [20], [30]])
        result = return_like_input(processed, original)

        assert isinstance(result, pd.Series)
        expected_series = pd.Series([10, 20, 30], name="test_series", index=original.index)
        pd.testing.assert_series_equal(result, expected_series)

    def test_return_pandas_series_multiple_columns(self):
        """Test returning data as DataFrame when original is Series but processed has multiple columns."""
        original = pd.Series([1, 2, 3], name="test_series")
        processed = np.array([[10, 100], [20, 200], [30, 300]])
        result = return_like_input(processed, original)

        # Should return DataFrame since processed has multiple columns
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 2)

    def test_return_list(self):
        """Test returning data in list format."""
        original = [[1, 2], [3, 4]]
        processed = np.array([[10, 20], [30, 40]])
        result = return_like_input(processed, original)

        assert isinstance(result, list)
        assert result == [[10, 20], [30, 40]]

    def test_return_single_value_original(self):
        """Test returning data when original was a single value."""
        original = 5
        processed = np.array([[10]])
        result = return_like_input(processed, original)

        # Should return the scalar value
        assert result == 10
        assert not isinstance(result, np.ndarray)

    def test_return_single_value_from_1d_processed(self):
        """Test returning single value from 1D processed array."""
        original = 5
        processed = np.array([10])
        result = return_like_input(processed, original)

        # Should return the scalar value
        assert result == 10
        assert not isinstance(result, np.ndarray)

    def test_preserve_pandas_index_and_columns(self):
        """Test that pandas index and column names are preserved."""
        original = pd.DataFrame(
            {"feature_1": [1, 2, 3], "feature_2": [4, 5, 6]}, index=["a", "b", "c"]
        )
        processed = np.array([[10, 40], [20, 50], [30, 60]])
        result = return_like_input(processed, original)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["feature_1", "feature_2"]
        assert list(result.index) == ["a", "b", "c"]
        pd.testing.assert_frame_equal(
            result, pd.DataFrame(processed, columns=original.columns, index=original.index)
        )

    def test_preserve_series_name_and_index(self):
        """Test that Series name and index are preserved."""
        original = pd.Series([1, 2, 3], name="my_series", index=["x", "y", "z"])
        processed = np.array([[10], [20], [30]])
        result = return_like_input(processed, original)

        assert isinstance(result, pd.Series)
        assert result.name == "my_series"
        assert list(result.index) == ["x", "y", "z"]
        pd.testing.assert_series_equal(
            result, pd.Series([10, 20, 30], name="my_series", index=original.index)
        )

    def test_different_shapes_handling(self):
        """Test handling when processed array has different shape."""
        original = np.array([[1, 2, 3], [4, 5, 6]])
        processed = np.array([[10], [20]])  # Different number of columns
        result = return_like_input(processed, original)

        # Should still return numpy array
        np.testing.assert_array_equal(result, processed)
        assert isinstance(result, np.ndarray)

    def test_empty_processed_array(self):
        """Test handling of empty processed arrays."""
        original = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        processed = np.array([]).reshape(0, 2)
        result = return_like_input(processed, original)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 0
        assert list(result.columns) == ["A", "B"]

    def test_type_preservation_for_arrays(self):
        """Test that array data types are preserved."""
        original = np.array([[1, 2], [3, 4]], dtype=np.int32)
        processed = np.array([[10, 20], [30, 40]], dtype=np.int32)
        result = return_like_input(processed, original)

        np.testing.assert_array_equal(result, processed)
        assert result.dtype == np.int32

    def test_fallback_to_numpy_array(self):
        """Test fallback to numpy array for unknown types."""

        # Use a custom class that's not specifically handled
        class CustomArray:
            def __init__(self, data):
                self.data = data

        original = CustomArray([[1, 2], [3, 4]])
        processed = np.array([[10, 20], [30, 40]])
        result = return_like_input(processed, original)

        # Should fallback to returning the processed array as-is
        np.testing.assert_array_equal(result, processed)
        assert isinstance(result, np.ndarray)


class TestPrepareInputWithColumns:
    """Test suite for prepare_input_with_columns function."""

    def test_numpy_array_no_columns(self):
        """Test numpy array input without column specification."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        array, columns, original_input = prepare_input_with_columns(X)

        np.testing.assert_array_equal(array, X)
        assert columns is None
        np.testing.assert_array_equal(original_input, X)

    def test_numpy_array_with_columns(self):
        """Test numpy array input with column specification."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        columns = ["A", "B"]
        array, result_columns, original_input = prepare_input_with_columns(X, columns=columns)

        expected_array = X[:, :2]  # Should select first 2 columns
        np.testing.assert_array_equal(array, expected_array)
        assert result_columns == ["A", "B"]
        np.testing.assert_array_equal(original_input, X)

    def test_pandas_dataframe_no_columns(self):
        """Test pandas DataFrame input without column specification."""
        df = pd.DataFrame({"A": [1, 4], "B": [2, 5], "C": [3, 6]})
        array, columns, original_input = prepare_input_with_columns(df)

        expected_array = df.values
        np.testing.assert_array_equal(array, expected_array)
        assert columns == ["A", "B", "C"]
        pd.testing.assert_frame_equal(original_input, df)

    def test_pandas_dataframe_with_columns(self):
        """Test pandas DataFrame input with column specification."""
        df = pd.DataFrame({"A": [1, 4], "B": [2, 5], "C": [3, 6]})
        columns = ["B", "C"]
        array, result_columns, original_input = prepare_input_with_columns(df, columns=columns)

        expected_array = df[columns].values
        np.testing.assert_array_equal(array, expected_array)
        assert result_columns == ["B", "C"]
        pd.testing.assert_frame_equal(original_input, df)

    def test_pandas_series_no_columns(self):
        """Test pandas Series input without column specification."""
        series = pd.Series([1, 2, 3], name="test_series")
        array, columns, original_input = prepare_input_with_columns(series)

        expected_array = series.values.reshape(-1, 1)
        np.testing.assert_array_equal(array, expected_array)
        assert columns == ["test_series"]
        pd.testing.assert_series_equal(original_input, series)

    def test_pandas_series_unnamed(self):
        """Test pandas Series input without name."""
        series = pd.Series([1, 2, 3])  # No name
        array, columns, original_input = prepare_input_with_columns(series)

        expected_array = series.values.reshape(-1, 1)
        np.testing.assert_array_equal(array, expected_array)
        assert columns == [0]  # Should use index 0 as column name
        pd.testing.assert_series_equal(original_input, series)

    def test_list_input_no_columns(self):
        """Test list input without column specification."""
        X = [[1, 2, 3], [4, 5, 6]]
        array, columns, original_input = prepare_input_with_columns(X)

        expected_array = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(array, expected_array)
        assert columns is None
        assert original_input == X

    def test_list_input_with_columns(self):
        """Test list input with column specification."""
        X = [[1, 2, 3], [4, 5, 6]]
        columns = ["A", "B"]
        array, result_columns, original_input = prepare_input_with_columns(X, columns=columns)

        expected_array = np.array([[1, 2], [4, 5]])  # First 2 columns
        np.testing.assert_array_equal(array, expected_array)
        assert result_columns == ["A", "B"]
        assert original_input == X

    def test_single_column_selection(self):
        """Test selection of single column."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        columns = ["B"]
        array, result_columns, original_input = prepare_input_with_columns(df, columns=columns)

        expected_array = df[["B"]].values
        np.testing.assert_array_equal(array, expected_array)
        assert result_columns == ["B"]
        assert array.shape == (2, 1)

    def test_column_order_preservation(self):
        """Test that column order is preserved when specified."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        columns = ["C", "A"]  # Different order
        array, result_columns, original_input = prepare_input_with_columns(df, columns=columns)

        expected_array = df[["C", "A"]].values
        np.testing.assert_array_equal(array, expected_array)
        assert result_columns == ["C", "A"]
        # Check that first column is C values, second is A values
        np.testing.assert_array_equal(array[:, 0], df["C"].values)
        np.testing.assert_array_equal(array[:, 1], df["A"].values)

    def test_column_indices_for_numpy(self):
        """Test column selection using indices for numpy arrays."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        columns = [0, 2]  # Select first and third columns
        array, result_columns, original_input = prepare_input_with_columns(X, columns=columns)

        expected_array = X[:, [0, 2]]
        np.testing.assert_array_equal(array, expected_array)
        assert result_columns == [0, 2]
        np.testing.assert_array_equal(original_input, X)

    def test_invalid_column_names_dataframe(self):
        """Test error handling for invalid column names in DataFrame."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        columns = ["A", "INVALID"]

        with pytest.raises(KeyError):
            prepare_input_with_columns(df, columns=columns)

    def test_invalid_column_indices_numpy(self):
        """Test error handling for invalid column indices in numpy array."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        columns = [0, 5]  # Index 5 doesn't exist

        with pytest.raises(IndexError):
            prepare_input_with_columns(X, columns=columns)

    def test_empty_columns_list(self):
        """Test handling of empty columns list."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        columns = []
        array, result_columns, original_input = prepare_input_with_columns(df, columns=columns)

        # Should return empty array but preserve other information
        assert array.shape[0] == 2  # Same number of rows
        assert array.shape[1] == 0  # No columns
        assert result_columns == []

    def test_duplicate_columns(self):
        """Test handling of duplicate column specifications."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        columns = ["A", "B", "A"]  # Duplicate 'A'
        array, result_columns, original_input = prepare_input_with_columns(df, columns=columns)

        # Should select columns as specified, including duplicates
        expected_array = df[["A", "B", "A"]].values
        np.testing.assert_array_equal(array, expected_array)
        assert result_columns == ["A", "B", "A"]
        assert array.shape == (2, 3)

    def test_mixed_column_types(self):
        """Test DataFrame with mixed column data types."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2],
                "float_col": [1.1, 2.2],
                "str_col": ["a", "b"],
                "bool_col": [True, False],
            }
        )
        array, columns, original_input = prepare_input_with_columns(df)

        assert array.shape == (2, 4)
        assert columns == ["int_col", "float_col", "str_col", "bool_col"]
        # Array should handle mixed types (likely as object dtype)
        assert array.dtype == object or array.dtype.char == "U"

    def test_large_dataframe_column_selection(self):
        """Test column selection from large DataFrame."""
        # Create a large DataFrame
        np.random.seed(42)
        data = {f"col_{i}": np.random.rand(100) for i in range(50)}
        df = pd.DataFrame(data)

        # Select a subset of columns
        selected_cols = [f"col_{i}" for i in [0, 5, 10, 15, 20]]
        array, columns, original_input = prepare_input_with_columns(df, columns=selected_cols)

        assert array.shape == (100, 5)
        assert columns == selected_cols
        np.testing.assert_array_equal(array, df[selected_cols].values)


class TestIntegrationScenarios:
    """Integration tests combining multiple data handling functions."""

    def test_complete_data_handling_workflow_numpy(self):
        """Test complete workflow with numpy arrays."""
        # Original data
        original = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Prepare input
        prepared = prepare_array(original)
        np.testing.assert_array_equal(prepared, original)

        # Simulate some processing (e.g., binning results)
        processed = prepared * 10

        # Return in original format
        result = return_like_input(processed, original)
        expected = original * 10
        np.testing.assert_array_equal(result, expected)
        assert isinstance(result, np.ndarray)

    def test_complete_data_handling_workflow_pandas(self):
        """Test complete workflow with pandas DataFrame."""
        # Original data
        original = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "feature3": [7, 8, 9]},
            index=["a", "b", "c"],
        )

        # Prepare input with specific columns
        columns = ["feature1", "feature3"]
        array, result_columns, original_input = prepare_input_with_columns(
            original, columns=columns
        )

        # Verify preparation
        expected_array = original[columns].values
        np.testing.assert_array_equal(array, expected_array)
        assert result_columns == columns

        # Simulate processing
        processed = array * 100

        # Return in original format
        result = return_like_input(processed, original[columns])

        # Verify result
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == columns
        assert list(result.index) == ["a", "b", "c"]
        expected_df = pd.DataFrame(
            {"feature1": [100, 200, 300], "feature3": [700, 800, 900]}, index=["a", "b", "c"]
        )
        pd.testing.assert_frame_equal(result, expected_df)

    def test_complete_data_handling_workflow_series(self):
        """Test complete workflow with pandas Series."""
        # Original data
        original = pd.Series([10, 20, 30], name="my_feature", index=["x", "y", "z"])

        # Prepare input
        array, columns, original_input = prepare_input_with_columns(original)

        # Verify preparation
        expected_array = original.values.reshape(-1, 1)
        np.testing.assert_array_equal(array, expected_array)
        assert columns == ["my_feature"]

        # Simulate processing
        processed = array + 5

        # Return in original format
        result = return_like_input(processed, original)

        # Verify result
        assert isinstance(result, pd.Series)
        assert result.name == "my_feature"
        assert list(result.index) == ["x", "y", "z"]
        expected_series = pd.Series([15, 25, 35], name="my_feature", index=["x", "y", "z"])
        pd.testing.assert_series_equal(result, expected_series)

    def test_mixed_input_types_workflow(self):
        """Test workflow with different input types."""
        # Test various input types through complete workflow
        inputs = [
            np.array([[1, 2], [3, 4]]),
            pd.DataFrame({"A": [1, 3], "B": [2, 4]}),
            pd.Series([1, 3], name="test"),
            [[1, 2], [3, 4]],
            [1, 2, 3],
        ]

        for original in inputs:
            # Prepare array
            prepared = prepare_array(original)
            assert isinstance(prepared, np.ndarray)
            assert prepared.ndim == 2

            # Simulate processing
            processed = prepared + 10

            # Return in original format
            result = return_like_input(processed, original)

            # Verify type consistency
            if isinstance(original, np.ndarray):
                assert isinstance(result, np.ndarray)
            elif isinstance(original, pd.DataFrame):
                assert isinstance(result, pd.DataFrame)
            elif isinstance(original, pd.Series):
                if processed.shape[1] == 1:
                    assert isinstance(result, pd.Series)
                else:
                    assert isinstance(result, pd.DataFrame)
            elif isinstance(original, list):
                assert isinstance(result, list)

    def test_error_handling_workflow(self):
        """Test error handling throughout the workflow."""
        # Test with mismatched column specifications
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

        with pytest.raises(KeyError):
            prepare_input_with_columns(df, columns=["A", "NONEXISTENT"])

        # Test with invalid numpy column indices
        arr = np.array([[1, 2], [3, 4]])

        with pytest.raises(IndexError):
            prepare_input_with_columns(arr, columns=[0, 5])

    def test_edge_cases_workflow(self):
        """Test workflow with edge cases."""
        # Empty inputs
        empty_df = pd.DataFrame()
        prepared_empty = prepare_array(empty_df)
        assert prepared_empty.shape[0] == 0

        # Single value inputs
        single_val = 42
        prepared_single = prepare_array(single_val)
        assert prepared_single.shape == (1, 1)
        result_single = return_like_input(prepared_single * 2, single_val)
        assert result_single == 84

        # Very wide DataFrames
        wide_df = pd.DataFrame({f"col_{i}": [1, 2] for i in range(100)})
        array, columns, _ = prepare_input_with_columns(wide_df)
        assert array.shape == (2, 100)
        assert len(columns) == 100


if __name__ == "__main__":
    pytest.main([__file__])
