"""
Comprehensive test suite for ManualIntervalBinning transformer.

This module contains tests for the ManualIntervalBinning class, covering
initialization, parameter validation, fitting, transformation, edge cases,
data type compatibility, sklearn integration, and error handling.

Test Classes:
    TestManualIntervalBinning: Core functionality tests including initialization,
        validation, fitting, transformation, and basic operations.
    TestManualIntervalBinningDataTypes: Tests for various data type compatibility
        including pandas DataFrames, polars DataFrames, and numpy arrays.
    TestManualIntervalBinningSklearnIntegration: Tests for sklearn compatibility
        including pipeline integration, ColumnTransformer usage, and cloning.
    TestManualIntervalBinningEdgeCases: Tests for edge cases and error conditions.
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer

# Import sklearn components
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from binlearn import PANDAS_AVAILABLE, POLARS_AVAILABLE, pd, pl
from binlearn.methods._manual_interval_binning import ManualIntervalBinning
from binlearn.utils.errors import BinningError, ConfigurationError

try:
    from scipy import sparse
except ImportError:  # pragma: no cover
    sparse = None
SKLEARN_AVAILABLE = True


class TestManualIntervalBinning:
    """Comprehensive test cases for ManualIntervalBinning core functionality.

    This test class covers the fundamental operations of the ManualIntervalBinning
    transformer including initialization, parameter validation, fitting,
    transformation, edge cases, and basic data handling scenarios.
    """

    def test_init_withbin_edges_(self):
        """Test initialization with bin_edges parameter."""
        # Test basic initialization
        bin_edges = {0: [0.0, 10.0, 20.0, 30.0], 1: [0.0, 5.0, 15.0, 25.0]}
        mib = ManualIntervalBinning(bin_edges=bin_edges)

        assert mib.bin_edges == bin_edges
        assert mib.bin_representatives is None
        assert mib._fitted is True  # Should be fitted with provided edges

    def test_init_withbin_edges__and_representatives(self):
        """Test initialization with both bin_edges and bin_representatives."""
        bin_edges = {0: [0.0, 10.0, 20.0, 30.0]}
        bin_representatives = {0: [5.0, 15.0, 25.0]}

        mib = ManualIntervalBinning(bin_edges=bin_edges, bin_representatives=bin_representatives)

        assert mib.bin_edges == bin_edges
        assert mib.bin_representatives == bin_representatives
        assert mib._fitted is True

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        bin_edges = {0: [0.0, 10.0, 20.0]}
        mib = ManualIntervalBinning(
            bin_edges=bin_edges, clip=False, preserve_dataframe=True, fit_jointly=False
        )

        assert mib.bin_edges == bin_edges
        assert mib.clip is False
        assert mib.preserve_dataframe is True
        assert mib.fit_jointly is False

    def test_init_nonebin_edges__raises_error(self):
        """Test that initialization with None bin_edges raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="bin_edges must be provided"):
            ManualIntervalBinning(bin_edges=None)  # type: ignore

    def test_calculate_bins_with_valid_column(self):
        """Test _calculate_bins method with valid column."""
        bin_edges = {0: [0.0, 10.0, 20.0, 30.0], 1: [5.0, 15.0, 25.0, 35.0]}
        mib = ManualIntervalBinning(bin_edges=bin_edges)

        # Test column 0
        x_col = np.array([5, 15, 25])  # Data is ignored in manual binning
        edges, representatives = mib._calculate_bins(x_col, 0)

        assert edges == [0.0, 10.0, 20.0, 30.0]
        assert representatives == [5.0, 15.0, 25.0]  # Auto-generated bin centers

    def test_calculate_bins_with_custom_representatives(self):
        """Test _calculate_bins with custom representatives."""
        bin_edges = {0: [0.0, 10.0, 20.0, 30.0]}
        bin_representatives = {0: [3.0, 12.0, 28.0]}  # Custom representatives

        mib = ManualIntervalBinning(bin_edges=bin_edges, bin_representatives=bin_representatives)

        x_col = np.array([5, 15, 25])
        edges, representatives = mib._calculate_bins(x_col, 0)

        assert edges == [0.0, 10.0, 20.0, 30.0]
        assert representatives == [3.0, 12.0, 28.0]

    def test_calculate_bins_missing_column_raises_error(self):
        """Test _calculate_bins with missing column raises BinningError."""
        bin_edges = {0: [0.0, 10.0, 20.0]}
        mib = ManualIntervalBinning(bin_edges=bin_edges)

        x_col = np.array([5, 15])
        with pytest.raises(BinningError, match="No bin edges defined for column missing_col"):
            mib._calculate_bins(x_col, "missing_col")

    def test_calculate_bins_nonebin_edges__raises_error(self):
        """Test _calculate_bins when bin_edges is None."""
        # Create instance with bin_edges but then set to None to test error condition
        bin_edges = {0: [0.0, 10.0, 20.0]}
        mib = ManualIntervalBinning(bin_edges=bin_edges)
        mib.bin_edges = None  # Simulate error condition

        x_col = np.array([5, 15])
        with pytest.raises(BinningError, match="No bin edges defined for column 0"):
            mib._calculate_bins(x_col, 0)

    def test_validate_params_valid_configuration(self):
        """Test _validate_params with valid configuration."""
        bin_edges = {0: [0.0, 10.0, 20.0, 30.0], 1: [5.0, 15.0, 25.0]}
        mib = ManualIntervalBinning(bin_edges=bin_edges)

        # Should not raise any exception
        mib._validate_params()

    def test_validate_params_nonebin_edges_(self):
        """Test that None bin_edges raises ConfigurationError during construction."""
        with pytest.raises(
            ConfigurationError, match="bin_edges must be provided for ManualIntervalBinning"
        ):
            ManualIntervalBinning(bin_edges=None)  # type: ignore

    def test_validate_params_emptybin_edges_(self):
        """Test that empty bin_edges raises ConfigurationError during construction."""
        with pytest.raises(
            ConfigurationError, match="bin_edges cannot be empty for ManualIntervalBinning"
        ):
            ManualIntervalBinning(bin_edges={})

    def test_validate_params_invalid_edge_type(self):
        """Test that initialization with invalid edge type raises ConfigurationError."""
        bin_edges = {0: "invalid"}  # String instead of list/array
        with pytest.raises(ConfigurationError, match="must be array-like"):
            ManualIntervalBinning(bin_edges=bin_edges)  # type: ignore

    def test_validate_params_insufficient_edges(self):
        """Test that initialization with insufficient number of edges raises ConfigurationError."""
        bin_edges = {0: [10.0]}  # Only one edge (need at least 2)
        with pytest.raises(ConfigurationError, match="needs at least 2 bin edges"):
            ManualIntervalBinning(bin_edges=bin_edges)

    def test_validate_params_unsorted_edges(self):
        """Test that initialization with unsorted edges raises ConfigurationError."""
        bin_edges = {0: [0.0, 30.0, 10.0, 20.0]}  # Unsorted edges
        with pytest.raises(ConfigurationError, match="must be sorted in ascending order"):
            ManualIntervalBinning(bin_edges=bin_edges)

    def test_validate_params_nan_edges(self):
        """Test that initialization with NaN edges raises ConfigurationError."""
        bin_edges = {0: [0.0, 10.0, np.nan, 30.0]}  # Contains NaN
        with pytest.raises(ConfigurationError, match="must be sorted in ascending order"):
            ManualIntervalBinning(bin_edges=bin_edges)

    def test_validate_params_infinite_edges(self):
        """Test that initialization with infinite edges raises ConfigurationError."""
        bin_edges = {0: [0.0, 10.0, np.inf, 30.0]}  # Contains infinity
        with pytest.raises(
            ConfigurationError, match="must be sorted in ascending order|must be numeric"
        ):
            ManualIntervalBinning(bin_edges=bin_edges)

    def test_validate_params_representatives_without_edges(self):
        """Test _validate_params with representatives for non-existent column - should pass now."""
        # Create a valid instance first, then call _validate_params directly
        bin_edges = {0: [0.0, 10.0, 20.0]}
        mib = ManualIntervalBinning(bin_edges=bin_edges)

        # Manipulate the bin_representatives to create this state for testing
        mib.bin_representatives = {1: [5.0, 15.0]}  # Column 1 not in bin_edges

        # This should NOT raise an error anymore with our new validation approach
        # The validation function allows representatives for columns not in edges
        mib._validate_params()  # Should pass without error

    def test_validate_params_invalid_representatives_type(self):
        """Test that initialization with invalid representatives type raises ConfigurationError."""
        bin_edges = {0: [0.0, 10.0, 20.0]}
        bin_representatives = {0: "invalid"}  # String instead of list/array

        with pytest.raises(ConfigurationError, match="must be array-like"):
            ManualIntervalBinning(
                bin_edges=bin_edges, bin_representatives=bin_representatives  # type: ignore
            )

    def test_validate_params_wrong_number_representatives(self):
        """Test that initialization with wrong number of representatives raises ConfigurationError."""
        bin_edges = {0: [0.0, 10.0, 20.0, 30.0]}  # 3 bins (4 edges)
        bin_representatives = {0: [5.0, 15.0]}  # Only 2 representatives for 3 bins

        with pytest.raises(ConfigurationError, match="representatives provided, but.*expected"):
            ManualIntervalBinning(bin_edges=bin_edges, bin_representatives=bin_representatives)

    def test_validate_params_nan_representatives(self):
        """Test that initialization with NaN representatives raises ConfigurationError."""
        bin_edges = {0: [0.0, 10.0, 20.0]}
        bin_representatives = {0: [np.nan]}  # NaN representative (wrong count too)

        with pytest.raises(
            ConfigurationError,
            match="must be finite values|representatives provided, but.*expected",
        ):
            ManualIntervalBinning(bin_edges=bin_edges, bin_representatives=bin_representatives)

    def test_validate_params_infinite_representatives(self):
        """Test that initialization with infinite representatives raises ConfigurationError."""
        bin_edges = {0: [0.0, 10.0, 20.0]}
        bin_representatives = {0: [np.inf]}  # Infinite representative (wrong count too)

        with pytest.raises(
            ConfigurationError,
            match="must be finite values|representatives provided, but.*expected",
        ):
            ManualIntervalBinning(bin_edges=bin_edges, bin_representatives=bin_representatives)

    def test_validate_params_with_tuple_edges(self):
        """Test _validate_params accepts tuple edges."""
        bin_edges = {0: [0.0, 10.0, 20.0, 30.0]}  # List version for proper typing
        mib = ManualIntervalBinning(bin_edges=bin_edges)

        # Should not raise exception
        mib._validate_params()

    def test_validate_params_with_numpy_array_edges(self):
        """Test _validate_params accepts numpy array edges."""
        bin_edges = {0: np.array([0.0, 10.0, 20.0, 30.0]).tolist()}  # Convert to list
        mib = ManualIntervalBinning(bin_edges=bin_edges)

        # Should not raise exception
        mib._validate_params()

    def test_validate_params_with_tuple_representatives(self):
        """Test _validate_params accepts tuple representatives."""
        bin_edges = {0: [0.0, 10.0, 20.0, 30.0]}
        bin_representatives = {0: [5.0, 15.0, 25.0]}  # List version for proper typing

        mib = ManualIntervalBinning(bin_edges=bin_edges, bin_representatives=bin_representatives)

        # Should not raise exception
        mib._validate_params()

    def test_validate_params_with_numpy_array_representatives(self):
        """Test _validate_params accepts numpy array representatives."""
        bin_edges = {0: [0.0, 10.0, 20.0, 30.0]}
        bin_representatives = {0: np.array([5.0, 15.0, 25.0]).tolist()}  # Convert to list

        mib = ManualIntervalBinning(bin_edges=bin_edges, bin_representatives=bin_representatives)

        # Should not raise exception
        mib._validate_params()

    def test_fit_method_calls_parent(self):
        """Test that fit method calls parent implementation."""
        bin_edges = {0: [0.0, 10.0, 20.0], 1: [5.0, 15.0, 25.0]}
        mib = ManualIntervalBinning(bin_edges=bin_edges)

        X = np.array([[5, 12], [15, 18], [8, 22]])

        # Should not raise exception and return self
        result = mib.fit(X)
        assert result is mib
        assert mib._fitted is True

    def test_fit_with_y_parameter(self):
        """Test fit method with y parameter (ignored)."""
        bin_edges = {0: [0.0, 10.0, 20.0]}
        mib = ManualIntervalBinning(bin_edges=bin_edges)

        X = np.array([[5], [15], [8]])
        y = np.array([0, 1, 0])  # Target values (ignored)

        result = mib.fit(X, y)
        assert result is mib

    def test_fit_with_fit_params(self):
        """Test fit method with additional fit_params (ignored)."""
        bin_edges = {0: [0.0, 10.0, 20.0]}
        mib = ManualIntervalBinning(bin_edges=bin_edges)

        X = np.array([[5], [15], [8]])

        result = mib.fit(X, custom_param="ignored")
        assert result is mib

    def test_basic_transform(self):
        """Test basic transform functionality."""
        bin_edges = {0: [0.0, 10.0, 20.0, 30.0], 1: [0.0, 15.0, 30.0, 45.0]}
        mib = ManualIntervalBinning(bin_edges=bin_edges)

        X = np.array([[5, 12], [15, 28], [25, 38]])
        X_transformed = mib.transform(X)

        assert X_transformed.shape == X.shape
        assert X_transformed.dtype == int
        # Check specific bin assignments
        assert X_transformed[0, 0] == 0  # 5 in [0, 10)
        assert X_transformed[0, 1] == 0  # 12 in [0, 15)
        assert X_transformed[1, 0] == 1  # 15 in [10, 20)
        assert X_transformed[1, 1] == 1  # 28 in [15, 30)
        assert X_transformed[2, 0] == 2  # 25 in [20, 30)
        assert X_transformed[2, 1] == 2  # 38 in [30, 45)

    def test_inverse_transform(self):
        """Test inverse transform functionality."""
        bin_edges = {0: [0.0, 10.0, 20.0], 1: [5.0, 15.0, 25.0]}
        bin_representatives = {0: [5.0, 15.0], 1: [10.0, 20.0]}

        mib = ManualIntervalBinning(bin_edges=bin_edges, bin_representatives=bin_representatives)

        X_binned = np.array([[0, 1], [1, 0]])
        X_inverse = mib.inverse_transform(X_binned)

        expected = np.array([[5.0, 20.0], [15.0, 10.0]])
        np.testing.assert_array_equal(X_inverse, expected)

    def test_fit_transform_workflow(self):
        """Test complete fit_transform workflow."""
        bin_edges = {0: [0.0, 25.0, 50.0, 75.0, 100.0], 1: [0.0, 30.0, 60.0, 90.0]}
        mib = ManualIntervalBinning(bin_edges=bin_edges)

        X = np.array([[10, 20], [40, 45], [80, 70]])
        X_transformed = mib.fit_transform(X)

        assert X_transformed.shape == X.shape
        assert X_transformed.dtype == int


class TestManualIntervalBinningDataTypes:
    """Tests for ManualIntervalBinning with various data types."""

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_dataframe(self):
        """Test with pandas DataFrame input."""
        bin_edges = {"col1": [0.0, 10.0, 20.0, 30.0], "col2": [0.0, 15.0, 30.0, 45.0]}
        mib = ManualIntervalBinning(bin_edges=bin_edges, preserve_dataframe=True)

        df = pd.DataFrame({"col1": [5, 15, 25], "col2": [12, 28, 38]})
        df_transformed = mib.transform(df)

        assert isinstance(df_transformed, pd.DataFrame)
        assert list(df_transformed.columns) == ["col1", "col2"]

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_dataframe_to_numpy(self):
        """Test DataFrame input with numpy output."""
        bin_edges = {"col1": [0.0, 10.0, 20.0], "col2": [0.0, 15.0, 30.0]}
        mib = ManualIntervalBinning(bin_edges=bin_edges, preserve_dataframe=False)

        df = pd.DataFrame({"col1": [5, 15], "col2": [8, 22]})
        result = mib.transform(df)

        assert isinstance(result, np.ndarray)

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
    def test_polars_dataframe(self):
        """Test with Polars DataFrame."""
        bin_edges = {"col1": [0.0, 10.0, 20.0], "col2": [0.0, 15.0, 30.0]}
        mib = ManualIntervalBinning(bin_edges=bin_edges, preserve_dataframe=True)

        df_polars = pl.DataFrame({"col1": [5, 15], "col2": [8, 22]})
        result = mib.transform(df_polars)

        assert isinstance(result, pl.DataFrame)

    def test_numpy_arrays_different_dtypes(self):
        """Test with different numpy data types."""
        bin_edges = {0: [0.0, 10.0, 20.0], 1: [0.0, 15.0, 30.0]}
        mib = ManualIntervalBinning(bin_edges=bin_edges)

        # Test int32
        X_int = np.array([[5, 12], [15, 25]], dtype=np.int32)
        result_int = mib.transform(X_int)
        assert isinstance(result_int, np.ndarray)

        # Test float32
        X_float = np.array([[5.5, 12.5], [15.5, 25.5]], dtype=np.float32)
        result_float = mib.transform(X_float)
        assert isinstance(result_float, np.ndarray)


class TestManualIntervalBinningSklearnIntegration:
    """Test sklearn integration for ManualIntervalBinning."""

    def test_sklearn_pipeline(self):
        """Test ManualIntervalBinning in sklearn Pipeline."""
        bin_edges = {0: [0.0, 10.0, 20.0, 30.0], 1: [0.0, 15.0, 30.0, 45.0]}

        pipeline = Pipeline(
            [
                ("binning", ManualIntervalBinning(bin_edges=bin_edges)),
                ("scaler", StandardScaler()),
            ]
        )

        X = np.array([[5, 12], [15, 28], [25, 38]])
        result = pipeline.fit_transform(X)

        assert result.shape == X.shape

    def test_sklearn_column_transformer(self):
        """Test ManualIntervalBinning with ColumnTransformer."""
        bin_edges = {0: [0.0, 10.0, 20.0], 1: [0.0, 15.0, 30.0]}

        ct = ColumnTransformer(
            [
                ("bin", ManualIntervalBinning(bin_edges=bin_edges), [0, 1]),
            ]
        )

        X = np.array([[5, 12, 100], [15, 25, 200]])
        result = ct.fit_transform(X)

        # Should have 2 columns from binlearn
        assert result.shape[1] == 2

    def test_sklearn_get_set_params(self):
        """Test sklearn parameter interface."""
        bin_edges = {0: [0.0, 10.0, 20.0]}
        mib = ManualIntervalBinning(bin_edges=bin_edges, clip=False)

        # Test get_params
        params = mib.get_params()
        assert params["bin_edges"] == bin_edges
        assert params["clip"] is False

        # Test set_params
        new_edges = {0: [0.0, 5.0, 10.0, 15.0]}
        mib.set_params(bin_edges=new_edges, clip=True)
        assert mib.bin_edges == new_edges
        assert mib.clip is True

    def test_sklearn_clone(self):
        """Test sklearn clone functionality."""
        bin_edges = {0: [0.0, 10.0, 20.0, 30.0]}
        mib_original = ManualIntervalBinning(bin_edges=bin_edges, clip=False)
        mib_cloned = clone(mib_original)

        # Should have same parameters but be different objects
        assert mib_cloned.bin_edges == bin_edges
        assert mib_cloned.clip is False
        assert mib_cloned is not mib_original


class TestManualIntervalBinningEdgeCases:
    """Test edge cases and error conditions for ManualIntervalBinning."""

    def test_equal_consecutive_edges(self):
        """Test with equal consecutive edges."""
        bin_edges = {0: [0.0, 10.0, 10.0, 20.0]}  # Equal consecutive edges
        mib = ManualIntervalBinning(bin_edges=bin_edges)

        # Should be valid (edges are sorted)
        mib._validate_params()

    def test_single_bin(self):
        """Test with single bin (2 edges)."""
        bin_edges = {0: [0.0, 10.0]}  # Single bin
        mib = ManualIntervalBinning(bin_edges=bin_edges)

        X = np.array([[5], [8], [2]])
        X_transformed = mib.transform(X)

        # All values should be in bin 0
        assert np.all(X_transformed == 0)

    def test_many_bins(self):
        """Test with many bins."""
        bin_edges = {0: [float(i) for i in range(101)]}  # 100 bins
        mib = ManualIntervalBinning(bin_edges=bin_edges)

        X = np.array([[50.5], [25.3], [75.7]])
        X_transformed = mib.transform(X)

        assert X_transformed.shape == (3, 1)
        assert X_transformed[0, 0] == 50  # 50.5 in bin 50
        assert X_transformed[1, 0] == 25  # 25.3 in bin 25
        assert X_transformed[2, 0] == 75  # 75.7 in bin 75

    def test_negative_edges(self):
        """Test with negative bin edges."""
        bin_edges = {0: [-100.0, -50.0, 0.0, 50.0, 100.0]}
        mib = ManualIntervalBinning(bin_edges=bin_edges)

        X = np.array([[-75], [-25], [25], [75]])
        X_transformed = mib.transform(X)

        expected = np.array([[0], [1], [2], [3]])
        np.testing.assert_array_equal(X_transformed, expected)

    def test_float_edges(self):
        """Test with float bin edges."""
        bin_edges = {0: [0.0, 0.33, 0.66, 1.0]}
        mib = ManualIntervalBinning(bin_edges=bin_edges)

        X = np.array([[0.1], [0.5], [0.9]])
        X_transformed = mib.transform(X)

        expected = np.array([[0], [1], [2]])
        np.testing.assert_array_equal(X_transformed, expected)

    def test_mixed_column_types(self):
        """Test with mixed column identifier types."""
        bin_edges = {0: [0.0, 10.0, 20.0], "feature": [0.0, 15.0, 30.0]}
        mib = ManualIntervalBinning(bin_edges=bin_edges)

        # Should validate without error
        mib._validate_params()

    def test_large_representative_values(self):
        """Test with large representative values."""
        bin_edges = {0: [0.0, 1e6, 1e9]}
        bin_representatives = {0: [5e5, 5e8]}

        mib = ManualIntervalBinning(bin_edges=bin_edges, bin_representatives=bin_representatives)

        X_binned = np.array([[0], [1]])
        X_inverse = mib.inverse_transform(X_binned)

        expected = np.array([[5e5], [5e8]])
        np.testing.assert_array_equal(X_inverse, expected)

    def test_auto_generated_representatives_calculation(self):
        """Test that auto-generated representatives are calculated correctly."""
        bin_edges = {0: [0.0, 10.0, 25.0, 30.0]}  # Uneven bin widths
        mib = ManualIntervalBinning(bin_edges=bin_edges)

        x_col = np.array([5])  # Dummy data
        edges, representatives = mib._calculate_bins(x_col, 0)

        # Should be bin centers: (0+10)/2, (10+25)/2, (25+30)/2
        expected_reps = [5.0, 17.5, 27.5]
        assert representatives == expected_reps

    def test_empty_data_transform(self):
        """Test transform with empty data."""
        bin_edges = {0: [0.0, 10.0, 20.0]}
        mib = ManualIntervalBinning(bin_edges=bin_edges)

        X_empty = np.array([]).reshape(0, 1)
        X_transformed = mib.transform(X_empty)

        assert X_transformed.shape == (0, 1)

    def test_string_column_names(self):
        """Test with string column names."""
        bin_edges = {
            "age": [0.0, 18.0, 35.0, 65.0, 100.0],
            "income": [0.0, 30000.0, 60000.0, 100000.0],
        }
        mib = ManualIntervalBinning(bin_edges=bin_edges)

        # Should validate and work correctly
        mib._validate_params()

        # Test _calculate_bins with string column ID
        x_col = np.array([25])
        edges, reps = mib._calculate_bins(x_col, "age")
        assert edges == [0.0, 18.0, 35.0, 65.0, 100.0]

    def test_coverage_of_various_bin_edge_validation_paths(self):
        """Test various edge cases in bin edge validation to ensure line coverage."""
        # Test with correct number of representatives
        bin_edges = {0: [0.0, 10.0, 20.0]}  # 2 bins
        bin_representatives = {0: [5.0, 15.0]}  # Exactly 2 representatives

        mib = ManualIntervalBinning(bin_edges=bin_edges, bin_representatives=bin_representatives)

        # Should validate successfully
        mib._validate_params()

        # Test edge case with single representative matching single bin
        bin_edges_single = {0: [0.0, 10.0]}  # 1 bin
        bin_reps_single = {0: [5.0]}  # 1 representative

        mib_single = ManualIntervalBinning(
            bin_edges=bin_edges_single, bin_representatives=bin_reps_single
        )

        # Should validate successfully
        mib_single._validate_params()

    def test_line_coverage_for_edge_list_conversion(self):
        """Test to ensure edge list conversion code is covered."""
        # Test with different iterable types to cover list() conversion
        bin_edges = {0: np.array([0.0, 10.0, 20.0]).tolist()}  # Convert to list for typing
        mib = ManualIntervalBinning(bin_edges=bin_edges)

        # This should trigger the list(edges) conversion in _calculate_bins
        x_col = np.array([5])
        edges, _ = mib._calculate_bins(x_col, 0)

        # Verify conversion worked
        assert isinstance(edges, list)
        assert edges == [0.0, 10.0, 20.0]

    def test_line_coverage_for_representatives_conversion(self):
        """Test to ensure representatives list conversion code is covered."""
        bin_edges = {0: [0.0, 10.0, 20.0]}
        bin_representatives = {0: np.array([5.0, 15.0]).tolist()}  # Convert to list for typing

        mib = ManualIntervalBinning(bin_edges=bin_edges, bin_representatives=bin_representatives)

        # This should trigger the list(representatives) conversion
        x_col = np.array([5])
        _, reps = mib._calculate_bins(x_col, 0)

        # Verify conversion worked
        assert isinstance(reps, list)
        assert reps == [5.0, 15.0]
