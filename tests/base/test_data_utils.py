"""
Comprehensive tests for simplified _data_utils.py.
"""

import numpy as np
import pytest
from typing import Any, List, Optional

from binning.base._data_utils import (
    is_pandas_df,
    is_polars_df,
    prepare_array,
    return_like_input,
    prepare_input_with_columns,
)
from binning import _pandas_config, _polars_config

PANDAS_AVAILABLE = _pandas_config.PANDAS_AVAILABLE
POLARS_AVAILABLE = _polars_config.POLARS_AVAILABLE


class TestDataFrameDetection:
    """Test is_pandas_df and is_polars_df functions."""

    def test_is_pandas_df_without_pandas(self, monkeypatch):
        """Test is_pandas_df when pandas is not available."""
        monkeypatch.setattr("binning._pandas_config.pd", None)
        assert not is_pandas_df(np.array([1, 2, 3]))
        assert not is_pandas_df([1, 2, 3])
        assert not is_pandas_df({"a": [1, 2]})

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_is_pandas_df_with_pandas(self):
        """Test is_pandas_df when pandas is available."""
        import pandas as pd

        # Test with actual DataFrame
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert is_pandas_df(df)

        # Test with non-DataFrame objects
        assert not is_pandas_df(np.array([1, 2, 3]))
        assert not is_pandas_df([1, 2, 3])
        assert not is_pandas_df({"a": [1, 2]})
        assert not is_pandas_df(pd.Series([1, 2, 3]))

    def test_is_polars_df_without_polars(self, monkeypatch):
        """Test is_polars_df when polars is not available."""
        monkeypatch.setattr("binning._polars_config.pl", None)
        assert not is_polars_df(np.array([1, 2, 3]))
        assert not is_polars_df([1, 2, 3])
        assert not is_polars_df({"a": [1, 2]})

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
    def test_is_polars_df_with_polars(self):
        """Test is_polars_df when polars is available."""
        import polars as pl

        # Test with actual DataFrame
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert is_polars_df(df)

        # Test with non-DataFrame objects
        assert not is_polars_df(np.array([1, 2, 3]))
        assert not is_polars_df([1, 2, 3])
        assert not is_polars_df({"a": [1, 2]})


class TestPrepareArray:
    """Test prepare_array function."""

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_prepare_array_pandas_dataframe(self):
        """Test prepare_array with pandas DataFrame."""
        import pandas as pd

        # Standard DataFrame
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=[10, 20])
        arr, columns, index = prepare_array(df)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 2)
        assert np.array_equal(arr, [[1, 3], [2, 4]])
        assert columns == ["a", "b"]
        assert list(index) == [10, 20]

        # Empty DataFrame
        empty_df = pd.DataFrame(columns=["x", "y"])
        arr, columns, index = prepare_array(empty_df)
        assert arr.shape == (0, 2)
        assert columns == ["x", "y"]

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
    def test_prepare_array_polars_dataframe(self):
        """Test prepare_array with polars DataFrame."""
        import polars as pl

        # Standard DataFrame
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        arr, columns, index = prepare_array(df)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 2)
        assert np.array_equal(arr, [[1, 3], [2, 4]])
        assert columns == ["a", "b"]
        assert index is None  # polars doesn't have index

        # Empty DataFrame
        empty_df = pl.DataFrame(schema={"x": pl.Int64, "y": pl.Int64})
        arr, columns, index = prepare_array(empty_df)
        assert arr.shape == (0, 2)
        assert columns == ["x", "y"]

    def test_prepare_array_numpy_2d(self):
        """Test prepare_array with 2D numpy array."""
        arr_input = np.array([[1, 2], [3, 4]])
        arr, columns, index = prepare_array(arr_input)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 2)
        assert np.array_equal(arr, arr_input)
        assert columns is None
        assert index is None

    def test_prepare_array_scalar(self):
        """Test prepare_array with scalar input."""
        scalar = 42
        arr, columns, index = prepare_array(scalar)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (1, 1)
        assert arr[0, 0] == 42
        assert columns is None
        assert index is None

    def test_prepare_array_1d_array(self):
        """Test prepare_array with 1D array."""
        arr_1d = [1, 2, 3, 4]
        arr, columns, index = prepare_array(arr_1d)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (4, 1)
        assert np.array_equal(arr.flatten(), arr_1d)
        assert columns is None
        assert index is None

    def test_prepare_array_list_2d(self):
        """Test prepare_array with 2D list."""
        list_2d = [[1, 2], [3, 4]]
        arr, columns, index = prepare_array(list_2d)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 2)
        assert np.array_equal(arr, list_2d)
        assert columns is None
        assert index is None

    def test_prepare_array_edge_cases(self):
        """Test prepare_array with edge cases."""
        # Tuple input
        tuple_input = ((1, 2), (3, 4))
        arr, columns, index = prepare_array(tuple_input)
        assert arr.shape == (2, 2)

        # Single element list
        single_list = [42]
        arr, columns, index = prepare_array(single_list)
        assert arr.shape == (1, 1)
        assert arr[0, 0] == 42


class TestReturnLikeInput:
    """Test return_like_input function."""

    def test_return_like_input_preserve_false(self):
        """Test return_like_input with preserve_dataframe=False."""
        arr = np.array([[1, 2], [3, 4]])
        result = return_like_input(arr, "dummy_input", preserve_dataframe=False)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, arr)

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_return_like_input_pandas_preserve_true(self):
        """Test return_like_input with pandas DataFrame and preserve_dataframe=True."""
        import pandas as pd

        arr = np.array([[1, 2], [3, 4]])
        original_df = pd.DataFrame({"a": [10, 20], "b": [30, 40]}, index=[100, 200])

        # With custom columns
        result = return_like_input(arr, original_df, columns=["x", "y"], preserve_dataframe=True)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["x", "y"]
        assert list(result.index) == [100, 200]
        assert np.array_equal(result.values, arr)

        # Without custom columns (use original)
        result = return_like_input(arr, original_df, preserve_dataframe=True)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]
        assert list(result.index) == [100, 200]
        assert np.array_equal(result.values, arr)

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_return_like_input_pandas_no_module(self, monkeypatch):
        """Test return_like_input when pandas is not available but input was pandas."""
        import pandas as pd

        arr = np.array([[1, 2], [3, 4]])
        original_df = pd.DataFrame({"a": [10, 20], "b": [30, 40]})

        # Mock pandas as unavailable
        monkeypatch.setattr("binning._pandas_config.pd", None)

        result = return_like_input(arr, original_df, preserve_dataframe=True)
        # Should return numpy array since pandas is not available
        assert isinstance(result, np.ndarray)

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
    def test_return_like_input_polars_preserve_true(self):
        """Test return_like_input with polars DataFrame and preserve_dataframe=True."""
        import polars as pl

        arr = np.array([[1, 2], [3, 4]])
        original_df = pl.DataFrame({"a": [10, 20], "b": [30, 40]})

        # With custom columns
        result = return_like_input(arr, original_df, columns=["x", "y"], preserve_dataframe=True)
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["x", "y"]
        assert np.array_equal(result.to_numpy(), arr)

        # Without custom columns (use original)
        result = return_like_input(arr, original_df, preserve_dataframe=True)
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["a", "b"]

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
    def test_return_like_input_polars_no_module(self, monkeypatch):
        """Test return_like_input when polars is not available but input was polars."""
        import polars as pl

        arr = np.array([[1, 2], [3, 4]])
        original_df = pl.DataFrame({"a": [10, 20], "b": [30, 40]})

        # Mock polars as unavailable
        monkeypatch.setattr("binning._polars_config.pl", None)

        result = return_like_input(arr, original_df, preserve_dataframe=True)
        # Should return numpy array since polars is not available
        assert isinstance(result, np.ndarray)

    def test_return_like_input_numpy_preserve_true(self):
        """Test return_like_input with numpy array and preserve_dataframe=True."""
        arr = np.array([[1, 2], [3, 4]])
        original_arr = np.array([[10, 20], [30, 40]])

        result = return_like_input(arr, original_arr, preserve_dataframe=True)
        # Should return numpy array since original was numpy
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, arr)

    def test_return_like_input_list_preserve_true(self):
        """Test return_like_input with list input and preserve_dataframe=True."""
        arr = np.array([[1, 2], [3, 4]])
        original_list = [[10, 20], [30, 40]]

        result = return_like_input(arr, original_list, preserve_dataframe=True)
        # Should return numpy array since original was not a DataFrame
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, arr)


class TestPrepareInputWithColumnsIntegration:
    """Test prepare_input_with_columns function integration."""

    def test_prepare_input_with_columns_basic(self):
        """Test basic functionality."""
        X = np.array([[1, 2], [3, 4]])
        arr, columns = prepare_input_with_columns(X)

        assert np.array_equal(arr, X)
        assert columns == [0, 1]

    def test_prepare_input_with_columns_fitted_state(self):
        """Test with fitted state and original columns."""
        X = np.array([[1, 2], [3, 4]])
        original_cols = ["a", "b", "c"]

        arr, columns = prepare_input_with_columns(X, fitted=True, original_columns=original_cols)

        # 2 <= 3, so should use range(2)
        assert columns == [0, 1]

    def test_prepare_input_with_columns_dimension_condition_coverage(self):
        """Test the specific dimension matching condition."""
        # Create a scenario that covers the target branch:
        # hasattr(X, "shape") and len(X.shape) == 2 and X.shape[1] <= len(original_columns)

        X = np.array([[1, 2, 3], [4, 5, 6]])  # Shape (2, 3)
        original_cols = ["a", "b", "c", "d"]  # 4 columns, so 3 <= 4

        # Mock prepare_array to return None for col_names
        from unittest.mock import patch

        with patch("binning.base._data_utils.prepare_array") as mock_prepare:
            mock_prepare.return_value = (X, None, None)  # No column names

            arr, columns = prepare_input_with_columns(
                X, fitted=True, original_columns=original_cols
            )

            # Should hit the target condition and return range(X.shape[1])
            assert columns == [0, 1, 2]  # range(3)
            assert len(columns) == X.shape[1]

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_prepare_input_with_columns_pandas(self):
        """Test with pandas DataFrame."""
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        arr, columns = prepare_input_with_columns(df)

        assert columns == ["a", "b"]
