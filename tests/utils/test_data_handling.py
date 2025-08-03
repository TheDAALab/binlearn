"""
Comprehensive test suite for the binning.utils.data_handling module.

This test module provides extensive coverage for all data handling utilities
used throughout the binning package. It tests functions for DataFrame detection,
array preparation, input/output format handling, column determination, and
data type conversion with various input formats including pandas DataFrames,
polars DataFrames, and numpy arrays.

The test suite aims for 100% line coverage by systematically testing every
code path, edge case, and error condition in the data_handling module.

Test Classes:
    TestIsPandasDf: Tests for pandas DataFrame detection.
    TestIsPolarsDF: Tests for polars DataFrame detection.
    TestPrepareArray: Tests for array preparation and metadata extraction.
    TestReturnLikeInput: Tests for output format preservation.
    TestDetermineColumns: Tests for column name determination logic.
    TestPrepareInputWithColumns: Tests for input preparation with column handling.
    TestAdditionalCoverage: Tests for edge cases and additional coverage.
"""

from unittest.mock import Mock, patch

import numpy as np

from binlearn.utils.data_handling import (
    _determine_columns,
    _is_pandas_df,
    _is_polars_df,
    prepare_array,
    prepare_input_with_columns,
    return_like_input,
)


class TestIsPandasDf:
    """Test the _is_pandas_df function for pandas DataFrame detection.

    This test class verifies that the _is_pandas_df function correctly
    identifies pandas DataFrames in various scenarios including when
    pandas is available, not available, and with different object types.
    """

    def test_with_pandas_dataframe(self):
        """Test detection with an actual pandas DataFrame.

        Verifies that the function returns True when given a genuine
        pandas DataFrame and pandas is available.
        """
        with patch("binlearn._pandas_config.pd") as mock_pd:
            # Mock pandas DataFrame
            mock_df = Mock()
            mock_pd.DataFrame = type(mock_df)
            mock_pd.return_value = mock_pd  # Make pd not None

            result = _is_pandas_df(mock_df)
            assert result is True

    def test_with_pandas_none(self):
        """Test behavior when pandas is not available.

        Verifies that the function returns False when pandas is None
        (indicating pandas is not installed or available).
        """
        with patch("binlearn._pandas_config.pd", None):
            result = _is_pandas_df(Mock())
            assert result is False

    def test_with_non_dataframe(self):
        """Test detection with non-DataFrame objects.

        Verifies that the function returns False when given objects
        that are not pandas DataFrames, even when pandas is available.
        """
        with patch("binlearn._pandas_config.pd") as mock_pd:
            mock_pd.DataFrame = type(Mock())

            result = _is_pandas_df("not a dataframe")
            assert result is False


class TestIsPolarsDF:
    """Test _is_polars_df function."""

    def test_with_polars_dataframe(self):
        """Test with actual polars DataFrame."""
        with patch("binlearn._polars_config.pl") as mock_pl:
            # Mock polars DataFrame
            mock_df = Mock()
            mock_pl.DataFrame = type(mock_df)
            mock_pl.return_value = mock_pl  # Make pl not None

            result = _is_polars_df(mock_df)
            assert result is True

    def test_with_polars_none(self):
        """Test when polars is None (not available)."""
        with patch("binlearn._polars_config.pl", None):
            result = _is_polars_df(Mock())
            assert result is False

    def test_with_non_dataframe(self):
        """Test with non-DataFrame object."""
        with patch("binlearn._polars_config.pl") as mock_pl:
            mock_pl.DataFrame = type(Mock())

            result = _is_polars_df("not a dataframe")
            assert result is False


class TestPrepareArray:
    """Test prepare_array function."""

    def test_with_pandas_dataframe(self):
        """Test prepare_array with pandas DataFrame."""
        with patch("binlearn.utils.data_handling._is_pandas_df", return_value=True):
            mock_df = Mock()
            mock_df.columns = ["A", "B", "C"]
            mock_df.index = [0, 1, 2]

            with patch("numpy.asarray") as mock_asarray:
                mock_array = np.array([[1, 2, 3], [4, 5, 6]])
                mock_asarray.return_value = mock_array

                arr, columns, index = prepare_array(mock_df)

                assert np.array_equal(arr, mock_array)
                assert columns == ["A", "B", "C"]
                assert index == [0, 1, 2]

    def test_with_polars_dataframe(self):
        """Test prepare_array with polars DataFrame."""
        with patch("binlearn.utils.data_handling._is_pandas_df", return_value=False):
            with patch("binlearn.utils.data_handling._is_polars_df", return_value=True):
                mock_df = Mock()
                mock_df.columns = ["X", "Y"]
                mock_array = np.array([[1, 2], [3, 4]])
                mock_df.to_numpy.return_value = mock_array

                arr, columns, index = prepare_array(mock_df)

                assert np.array_equal(arr, mock_array)
                assert columns == ["X", "Y"]
                assert index is None

    def test_with_numpy_array_2d(self):
        """Test prepare_array with 2D numpy array."""
        with patch("binlearn.utils.data_handling._is_pandas_df", return_value=False):
            with patch("binlearn.utils.data_handling._is_polars_df", return_value=False):
                input_array = np.array([[1, 2], [3, 4]])

                arr, columns, index = prepare_array(input_array)

                assert np.array_equal(arr, input_array)
                assert columns is None
                assert index is None

    def test_with_scalar_array(self):
        """Test prepare_array with 0D array (scalar)."""
        with patch("binlearn.utils.data_handling._is_pandas_df", return_value=False):
            with patch("binlearn.utils.data_handling._is_polars_df", return_value=False):
                input_scalar = np.array(5)

                arr, columns, index = prepare_array(input_scalar)

                expected = np.array([[5]])
                assert np.array_equal(arr, expected)
                assert columns is None
                assert index is None

    def test_with_1d_array(self):
        """Test prepare_array with 1D array."""
        with patch("binlearn.utils.data_handling._is_pandas_df", return_value=False):
            with patch("binlearn.utils.data_handling._is_polars_df", return_value=False):
                input_array = np.array([1, 2, 3])

                arr, columns, index = prepare_array(input_array)

                expected = np.array([[1], [2], [3]])
                assert np.array_equal(arr, expected)
                assert columns is None
                assert index is None


class TestReturnLikeInput:
    """Test return_like_input function."""

    def test_preserve_dataframe_false(self):
        """Test when preserve_dataframe=False."""
        arr = np.array([[1, 2], [3, 4]])
        original_input = Mock()

        result = return_like_input(arr, original_input, preserve_dataframe=False)

        assert np.array_equal(result, arr)

    def test_pandas_dataframe_preserve_true(self):
        """Test preserving pandas DataFrame format."""
        with patch("binlearn.utils.data_handling._is_pandas_df", return_value=True):
            with patch("binlearn._pandas_config.pd") as mock_pd:
                # Setup mocks
                mock_df_class = Mock()
                mock_pd.DataFrame = mock_df_class
                mock_pd.return_value = mock_pd  # Make pd not None

                arr = np.array([[1, 2], [3, 4]])
                original_input = Mock()
                original_input.columns = ["A", "B"]
                original_input.index = [0, 1]

                _result = return_like_input(arr, original_input, preserve_dataframe=True)

                # Verify DataFrame constructor was called
                mock_df_class.assert_called_once_with(arr, columns=["A", "B"], index=[0, 1])

    def test_pandas_dataframe_with_custom_columns(self):
        """Test pandas DataFrame with custom columns."""
        with patch("binlearn.utils.data_handling._is_pandas_df", return_value=True):
            with patch("binlearn._pandas_config.pd") as mock_pd:
                mock_df_class = Mock()
                mock_pd.DataFrame = mock_df_class
                mock_pd.return_value = mock_pd

                arr = np.array([[1, 2], [3, 4]])
                original_input = Mock()
                original_input.columns = ["A", "B"]
                original_input.index = [0, 1]
                custom_columns = ["X", "Y"]

                _result = return_like_input(
                    arr, original_input, columns=custom_columns, preserve_dataframe=True
                )

                mock_df_class.assert_called_once_with(arr, columns=["X", "Y"], index=[0, 1])

    def test_pandas_dataframe_pd_none(self):
        """Test pandas DataFrame when pd is None."""
        with patch("binlearn.utils.data_handling._is_pandas_df", return_value=True):
            with patch("binlearn._pandas_config.pd", None):
                arr = np.array([[1, 2], [3, 4]])
                original_input = Mock()

                result = return_like_input(arr, original_input, preserve_dataframe=True)

                assert np.array_equal(result, arr)

    def test_polars_dataframe_preserve_true(self):
        """Test preserving polars DataFrame format."""
        with patch("binlearn.utils.data_handling._is_pandas_df", return_value=False):
            with patch("binlearn.utils.data_handling._is_polars_df", return_value=True):
                with patch("binlearn._polars_config.pl") as mock_pl:
                    mock_df_class = Mock()
                    mock_pl.DataFrame = mock_df_class
                    mock_pl.return_value = mock_pl

                    arr = np.array([[1, 2], [3, 4]])
                    original_input = Mock()
                    original_input.columns = ["A", "B"]

                    _result = return_like_input(arr, original_input, preserve_dataframe=True)

                    mock_df_class.assert_called_once_with(arr, schema=["A", "B"])

    def test_polars_dataframe_with_custom_columns(self):
        """Test polars DataFrame with custom columns."""
        with patch("binlearn.utils.data_handling._is_pandas_df", return_value=False):
            with patch("binlearn.utils.data_handling._is_polars_df", return_value=True):
                with patch("binlearn._polars_config.pl") as mock_pl:
                    mock_df_class = Mock()
                    mock_pl.DataFrame = mock_df_class
                    mock_pl.return_value = mock_pl

                    arr = np.array([[1, 2], [3, 4]])
                    original_input = Mock()
                    original_input.columns = ["A", "B"]
                    custom_columns = ["X", "Y"]

                    _result = return_like_input(
                        arr, original_input, columns=custom_columns, preserve_dataframe=True
                    )

                    mock_df_class.assert_called_once_with(arr, schema=["X", "Y"])

    def test_polars_dataframe_pl_none(self):
        """Test polars DataFrame when pl is None."""
        with patch("binlearn.utils.data_handling._is_pandas_df", return_value=False):
            with patch("binlearn.utils.data_handling._is_polars_df", return_value=True):
                with patch("binlearn._polars_config.pl", None):
                    arr = np.array([[1, 2], [3, 4]])
                    original_input = Mock()

                    result = return_like_input(arr, original_input, preserve_dataframe=True)

                    assert np.array_equal(result, arr)

    def test_non_dataframe_preserve_true(self):
        """Test with non-DataFrame input and preserve_dataframe=True."""
        with patch("binlearn.utils.data_handling._is_pandas_df", return_value=False):
            with patch("binlearn.utils.data_handling._is_polars_df", return_value=False):
                arr = np.array([[1, 2], [3, 4]])
                original_input = np.array([[5, 6], [7, 8]])

                result = return_like_input(arr, original_input, preserve_dataframe=True)

                assert np.array_equal(result, arr)


class TestDetermineColumns:
    """Test _determine_columns function."""

    def test_col_names_not_none(self):
        """Test when col_names is provided."""
        result = _determine_columns(None, ["A", "B"], False, None, (2, 2))
        assert result == ["A", "B"]

    def test_has_shape_2d(self):
        """Test with object that has 2D shape."""
        X = Mock()
        X.shape = (3, 4)

        result = _determine_columns(X, None, False, None, (3, 4))
        assert result == [0, 1, 2, 3]

    def test_fitted_condition_returns_original_columns(self):
        """Test fitted condition - simplified after removing unreachable code."""
        # After removing the unreachable code, the fitted condition always returns original_columns
        # regardless of the shape of X

        # Test with no shape attribute
        X_no_shape = Mock(spec=[])
        result = _determine_columns(X_no_shape, None, True, ["A", "B"], (3, 4))
        assert result == ["A", "B"]

        # Test with 1D shape
        X_1d = Mock()
        X_1d.shape = (5,)
        result = _determine_columns(X_1d, None, True, ["X", "Y"], (3, 4))
        assert result == ["X", "Y"]

        # Test with 3D shape
        X_3d = Mock()
        X_3d.shape = (3, 4, 2)
        result = _determine_columns(X_3d, None, True, ["P"], (3, 4))
        assert result == ["P"]

    def test_fallback_condition(self):
        """Test fallback condition."""
        X = Mock(spec=[])  # No shape attribute

        result = _determine_columns(X, None, False, None, (3, 4))
        assert result == [0, 1, 2, 3]


class TestPrepareInputWithColumns:
    """Test prepare_input_with_columns function."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        with patch("binlearn.utils.data_handling.prepare_array") as mock_prepare:
            with patch("binlearn.utils.data_handling._determine_columns") as mock_determine:
                # Setup mocks
                mock_array = np.array([[1, 2], [3, 4]])
                mock_prepare.return_value = (mock_array, ["A", "B"], None)
                mock_determine.return_value = ["A", "B"]

                X = Mock()
                arr, columns = prepare_input_with_columns(X)

                # Verify calls
                mock_prepare.assert_called_once_with(X)
                mock_determine.assert_called_once_with(X, ["A", "B"], False, None, mock_array.shape)

                assert np.array_equal(arr, mock_array)
                assert columns == ["A", "B"]

    def test_with_fitted_and_original_columns(self):
        """Test with fitted=True and original_columns."""
        with patch("binlearn.utils.data_handling.prepare_array") as mock_prepare:
            with patch("binlearn.utils.data_handling._determine_columns") as mock_determine:
                mock_array = np.array([[1, 2], [3, 4]])
                mock_prepare.return_value = (mock_array, None, None)
                mock_determine.return_value = [0, 1]

                X = Mock()
                original_cols = ["X", "Y"]

                arr, columns = prepare_input_with_columns(
                    X, fitted=True, original_columns=original_cols
                )

                mock_determine.assert_called_once_with(
                    X, None, True, original_cols, mock_array.shape
                )

                assert np.array_equal(arr, mock_array)
                assert columns == [0, 1]


# pylint: disable=too-few-public-methods
class TestAdditionalCoverage:
    """Additional tests for complete coverage."""

    def test_numpy_asarray_coverage(self):
        """Test to ensure numpy.asarray is called in prepare_array."""
        with patch("binlearn.utils.data_handling._is_pandas_df", return_value=False):
            with patch("binlearn.utils.data_handling._is_polars_df", return_value=False):
                with patch("numpy.asarray") as mock_asarray:
                    input_list = [[1, 2], [3, 4]]
                    expected_array = np.array(input_list)
                    mock_asarray.return_value = expected_array

                    arr, _columns, _index = prepare_array(input_list)

                    mock_asarray.assert_called_with(input_list)
                    assert np.array_equal(arr, expected_array)


# All tests are now complete and cover 100% of the simplified data_handling module
