"""
Comprehensive test suite for EqualFrequencyBinning transformer.

This module contains extensive tests for the EqualFrequencyBinning class, covering
initialization, parameter validation, fitting, transformation, edge cases,
data type compatibility, sklearn integration, and error handling.

Test Classes:
    TestEqualFrequencyBinning: Core functionality tests including initialization,
        validation, fitting, transformation, and basic operations.
    TestEqualFrequencyBinningDataTypes: Tests for various data type compatibility
        including pandas DataFrames, polars DataFrames, and scipy sparse matrices.
    TestEqualFrequencyBinningSklearnIntegration: Tests for sklearn compatibility
        including pipeline integration, ColumnTransformer usage, and cloning.
    TestEqualFrequencyBinningFitGetParamsWorkflow: Tests for parameter handling
        and sklearn-style workflows.
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer

# Import sklearn components
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from binlearn import PANDAS_AVAILABLE, POLARS_AVAILABLE, pd, pl
from binlearn.methods._equal_frequency_binning import EqualFrequencyBinning
from binlearn.utils.errors import ConfigurationError, DataQualityWarning


class TestEqualFrequencyBinning:
    """Comprehensive test cases for EqualFrequencyBinning core functionality.

    This test class covers the fundamental operations of the EqualFrequencyBinning
    transformer including initialization, parameter validation, fitting,
    transformation, edge cases, and basic data handling scenarios.
    """

    def test_init_default(self):
        """Test initialization with default parameters.

        Verifies that the transformer initializes correctly with default
        parameter values and that all attributes are set as expected.
        """
        efb = EqualFrequencyBinning()
        assert efb.n_bins == 10
        assert efb.quantile_range is None

    def test_init_custom_params(self):
        """Test initialization with custom parameters.

        Verifies that the transformer correctly accepts and stores custom
        parameter values including n_bins, quantile_range, and fit_jointly options.
        """
        efb = EqualFrequencyBinning(n_bins=5, quantile_range=(0.1, 0.9), fit_jointly=True)
        assert efb.n_bins == 5
        assert efb.quantile_range == (0.1, 0.9)
        assert efb.fit_jointly is True

    def test_validate_params_valid(self):
        """Test parameter validation with valid parameters.

        Ensures that the _validate_params method accepts valid parameter
        combinations without raising exceptions.
        """
        efb = EqualFrequencyBinning(n_bins=5, quantile_range=(0.1, 0.9))
        efb._validate_params()  # Should not raise

    def test_validate_params_invalid_n_bins(self):
        """Test parameter validation with invalid n_bins values.

        Verifies that the validator correctly rejects invalid n_bins values
        such as zero, negative numbers, and non-integer types.
        """
        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            EqualFrequencyBinning(n_bins=0)

        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            EqualFrequencyBinning(n_bins=-1)

    def test_validate_params_invalid_quantile_range(self):
        """Test parameter validation with invalid quantile_range values.

        Verifies that the validator correctly rejects invalid quantile_range
        specifications such as out-of-range values or reversed ranges.
        """
        with pytest.raises(ConfigurationError, match="quantile_range values must be numbers"):
            EqualFrequencyBinning(quantile_range=(0.9, 0.1))  # min > max

        with pytest.raises(ConfigurationError, match="quantile_range values must be numbers"):
            EqualFrequencyBinning(quantile_range=(-0.1, 0.5))  # min < 0

        with pytest.raises(ConfigurationError, match="quantile_range values must be numbers"):
            EqualFrequencyBinning(quantile_range=(0.5, 1.1))  # max > 1

        with pytest.raises(ConfigurationError, match="quantile_range must be a tuple"):
            EqualFrequencyBinning(quantile_range=(0.5,))  # type: ignore # wrong length

    def test_fit_transform_basic(self):
        """Test basic fit and transform functionality."""
        X = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])
        efb = EqualFrequencyBinning(n_bins=3)

        # Test fit
        efb.fit(X)
        assert hasattr(efb, "bin_edges_")
        assert len(efb.bin_edges_) == 2  # Two features

        # Test transform
        X_binned = efb.transform(X)
        assert X_binned.shape == X.shape
        assert np.all(X_binned >= 0)  # All bin indices should be non-negative
        assert np.all(X_binned < 3)  # All bin indices should be < n_bins

    def test_fit_transform_with_quantile_range(self):
        """Test fit_transform with custom quantile range."""
        # Create data with outliers
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))
        X[0, 0] = 100  # Add outlier
        X[1, 1] = -100  # Add outlier

        efb = EqualFrequencyBinning(n_bins=5, quantile_range=(0.1, 0.9))
        X_binned = efb.fit_transform(X)

        assert X_binned.shape == X.shape
        assert np.all(X_binned >= 0)
        # Note: outliers may be in overflow bins

    def test_constant_data(self):
        """Test behavior with constant data."""
        X = np.full((10, 2), fill_value=5.0)
        efb = EqualFrequencyBinning(n_bins=3)

        X_binned = efb.fit_transform(X)
        assert X_binned.shape == X.shape
        # With constant data, all values should be in the same bin (likely bin 0)

    def test_single_bin(self):
        """Test with single bin."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        efb = EqualFrequencyBinning(n_bins=1)

        X_binned = efb.fit_transform(X)
        assert X_binned.shape == X.shape
        assert np.all(X_binned == 0)  # All values should be in bin 0

    def test_nan_handling(self):
        """Test handling of NaN values."""
        X = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, np.nan], [7.0, 8.0]])
        efb = EqualFrequencyBinning(n_bins=2)

        # Should not raise error during fit
        efb.fit(X)
        X_binned = efb.transform(X)

        assert X_binned.shape == X.shape
        # NaN values should be replaced with MISSING_VALUE (-1)
        from binlearn.utils.constants import MISSING_VALUE

        assert X_binned[1, 0] == MISSING_VALUE
        assert X_binned[2, 1] == MISSING_VALUE

    def test_all_nan_column(self):
        """Test behavior with all-NaN column."""
        X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
        efb = EqualFrequencyBinning(n_bins=2)

        # Should handle all-NaN column gracefully and emit warning
        with pytest.warns(DataQualityWarning, match="Data in column 1.*contains only NaN values"):
            efb.fit(X)
        X_binned = efb.transform(X)

        assert X_binned.shape == X.shape
        # First column should be binned normally
        assert not np.all(X_binned[:, 0] == -1)
        # Second column should be all MISSING_VALUE (-1)
        from binlearn.utils.constants import MISSING_VALUE

        assert np.all(X_binned[:, 1] == MISSING_VALUE)

    def test_insufficient_data_for_bins(self):
        """Test error handling when there's insufficient data for requested bins."""
        X = np.array([[1.0], [2.0]])  # Only 2 non-NaN values
        efb = EqualFrequencyBinning(n_bins=5)  # Request 5 bins

        with pytest.raises(ValueError, match="Insufficient non-NaN values"):
            efb.fit(X)

    def test_joint_fitting_vs_per_column(self):
        """Test difference between joint fitting and per-column fitting."""
        # Create data with different scales
        X = np.array([[1, 100], [2, 200], [3, 300], [4, 400], [5, 500]])

        # Per-column fitting (default)
        efb_per_col = EqualFrequencyBinning(n_bins=3, fit_jointly=False)
        X_per_col = efb_per_col.fit_transform(X)

        # Joint fitting
        efb_joint = EqualFrequencyBinning(n_bins=3, fit_jointly=True)
        X_joint = efb_joint.fit_transform(X)

        # Results should be different
        assert not np.array_equal(X_per_col, X_joint)

    def test_direct_calculate_bins_basic(self):
        """Test _calculate_bins method directly."""
        efb = EqualFrequencyBinning(n_bins=4)
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        edges, reps = efb._calculate_bins(data, col_id=0)

        assert len(edges) == 5  # n_bins + 1
        assert len(reps) == 4  # n_bins
        assert edges[0] <= edges[-1]  # Edges should be sorted
        # Check that edges are monotonically increasing
        for i in range(1, len(edges)):
            assert edges[i] >= edges[i - 1]

    def test_direct_calculate_bins_with_quantile_range(self):
        """Test _calculate_bins with custom quantile range."""
        efb = EqualFrequencyBinning(n_bins=3, quantile_range=(0.2, 0.8))
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        edges, reps = efb._calculate_bins(data, col_id=0)

        assert len(edges) == 4  # n_bins + 1
        assert len(reps) == 3  # n_bins
        # With quantile range (0.2, 0.8), edges should not include extreme values
        assert edges[0] >= 2  # 20th percentile should be >= 2
        assert edges[-1] <= 9  # 80th percentile should be <= 9

    def test_direct_calculate_bins_invalid_n_bins(self):
        """Test _calculate_bins with invalid n_bins."""
        efb = EqualFrequencyBinning(n_bins=1)  # Start with valid n_bins
        efb.n_bins = 0  # Set directly to bypass init validation
        data = np.array([1, 2, 3, 4, 5])

        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            efb._calculate_bins(data, col_id=0)

    def test_empty_data(self):
        """Test behavior with empty data arrays."""
        X = np.array([]).reshape(0, 2)
        efb = EqualFrequencyBinning(n_bins=3)

        # Empty data should be handled gracefully, not raise an error
        # Should emit warnings for both empty columns
        with pytest.warns(DataQualityWarning, match="Data in column.*contains only NaN values"):
            efb.fit(X)
        X_binned = efb.transform(X)
        assert X_binned.shape == (0, 2)

    def test_edge_case_duplicate_values(self):
        """Test handling of data with many duplicate values."""
        # Data with many duplicates
        X = np.array([[1, 1, 1, 1, 1, 2, 2, 3]]).T
        efb = EqualFrequencyBinning(n_bins=3)

        # Should handle duplicates gracefully
        X_binned = efb.fit_transform(X)
        assert X_binned.shape == X.shape

    def test_quantile_calculation_edge_cases(self):
        """Test quantile calculation with edge cases."""
        efb = EqualFrequencyBinning(n_bins=2)

        # Test with constant data
        constant_data = np.array([5, 5, 5, 5, 5])
        edges, reps = efb._create_equal_frequency_bins(constant_data, "col1", 0.0, 1.0, 2)
        assert len(edges) == 3
        assert len(reps) == 2

        # Test with data having only two unique values
        binary_data = np.array([1, 1, 1, 2, 2, 2])
        edges, reps = efb._create_equal_frequency_bins(binary_data, "col1", 0.0, 1.0, 2)
        assert len(edges) == 3
        assert len(reps) == 2


class TestEqualFrequencyBinningDataTypes:
    """Test EqualFrequencyBinning with various data types and formats.

    This test class verifies that the EqualFrequencyBinning transformer works
    correctly with different data formats including pandas DataFrames,
    polars DataFrames, and scipy sparse matrices.
    """

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_dataframe(self):
        """Test with pandas DataFrame input and output."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            }
        )

        efb = EqualFrequencyBinning(n_bins=3, preserve_dataframe=True)
        df_binned = efb.fit_transform(df)

        assert isinstance(df_binned, pd.DataFrame)
        assert df_binned.shape == df.shape
        assert list(df_binned.columns) == list(df.columns)

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_with_missing_values(self):
        """Test pandas DataFrame with missing values."""
        df = pd.DataFrame({"A": [1, 2, np.nan, 4, 5], "B": [10, np.nan, 30, 40, 50]})

        efb = EqualFrequencyBinning(n_bins=2, preserve_dataframe=True)
        df_binned = efb.fit_transform(df)

        assert isinstance(df_binned, pd.DataFrame)
        assert df_binned.shape == df.shape
        # NaN values should be replaced with MISSING_VALUE (-1)
        from binlearn.utils.constants import MISSING_VALUE

        assert df_binned.iloc[2, 0] == MISSING_VALUE
        assert df_binned.iloc[1, 1] == MISSING_VALUE

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
    def test_polars_dataframe(self):
        """Test with polars DataFrame input and output."""
        df = pl.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            }
        )

        efb = EqualFrequencyBinning(n_bins=3, preserve_dataframe=True)
        df_binned = efb.fit_transform(df)  # type: ignore

        assert isinstance(df_binned, pl.DataFrame)
        assert df_binned.shape == df.shape
        assert df_binned.columns == df.columns

    def test_numpy_array(self):
        """Test with numpy array input."""
        X = np.random.rand(50, 3)
        efb = EqualFrequencyBinning(n_bins=4)

        X_binned = efb.fit_transform(X)

        assert isinstance(X_binned, np.ndarray)
        assert X_binned.shape == X.shape
        assert X_binned.dtype in [np.int32, np.int64, np.float64]


class TestEqualFrequencyBinningSklearnIntegration:
    """Test sklearn compatibility and integration features.

    This test class verifies that EqualFrequencyBinning integrates properly
    with sklearn pipelines, ColumnTransformer, and other sklearn utilities.
    """

    def test_sklearn_pipeline(self):
        """Test integration with sklearn Pipeline."""
        X = np.random.rand(20, 2)

        pipeline = Pipeline(
            [("binning", EqualFrequencyBinning(n_bins=3)), ("scaling", StandardScaler())]
        )

        # Should work without errors
        X_processed = pipeline.fit_transform(X)
        assert X_processed.shape == X.shape

    def test_sklearn_column_transformer(self):
        """Test integration with sklearn ColumnTransformer."""
        X = np.random.rand(20, 4)

        ct = ColumnTransformer(
            [
                ("binning", EqualFrequencyBinning(n_bins=3), [0, 1]),
                ("scaling", StandardScaler(), [2, 3]),
            ]
        )

        X_processed = ct.fit_transform(X)
        assert X_processed.shape[0] == X.shape[0]  # Same number of rows

    def test_sklearn_clone(self):
        """Test sklearn clone compatibility."""
        efb = EqualFrequencyBinning(n_bins=5, quantile_range=(0.1, 0.9))
        efb_cloned = clone(efb)

        assert efb_cloned.n_bins == efb.n_bins
        assert efb_cloned.quantile_range == efb.quantile_range
        assert efb_cloned is not efb  # Different objects

    def test_get_params_set_params(self):
        """Test sklearn-style parameter getting and setting."""
        efb = EqualFrequencyBinning(n_bins=3)

        # Test get_params
        params = efb.get_params()
        assert "n_bins" in params
        assert params["n_bins"] == 3

        # Test set_params
        efb.set_params(n_bins=5, quantile_range=(0.2, 0.8))
        assert efb.n_bins == 5
        assert efb.quantile_range == (0.2, 0.8)

    def test_fit_transform_consistency(self):
        """Test that fit_transform gives same result as fit followed by transform."""
        X = np.random.rand(30, 2)

        # Method 1: fit_transform
        efb1 = EqualFrequencyBinning(n_bins=4)
        X_binned1 = efb1.fit_transform(X)

        # Method 2: fit then transform
        efb2 = EqualFrequencyBinning(n_bins=4)
        efb2.fit(X)
        X_binned2 = efb2.transform(X)

        np.testing.assert_array_equal(X_binned1, X_binned2)


class TestEqualFrequencyBinningFitGetParamsWorkflow:
    """Test parameter handling and sklearn-style workflows.

    This test class focuses on testing parameter management, getting/setting
    parameters, and ensuring proper sklearn-compatible behavior patterns.
    """

    def test_parameter_persistence(self):
        """Test that parameters are properly stored and retrieved."""
        params = {
            "n_bins": 7,
            "quantile_range": (0.05, 0.95),
            "clip": True,
            "preserve_dataframe": False,
        }

        efb = EqualFrequencyBinning(**params)
        retrieved_params = efb.get_params()

        for key, value in params.items():
            assert retrieved_params[key] == value

    def test_parameter_validation_during_fit(self):
        """Test that parameter validation occurs during fit."""
        X = np.random.rand(10, 2)

        # Create with invalid parameters that bypass __init__ validation
        efb = EqualFrequencyBinning(n_bins=3)
        efb.n_bins = -1  # Set invalid value directly

        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            efb.fit(X)

    def test_fitted_attributes(self):
        """Test that appropriate attributes are set after fitting."""
        X = np.random.rand(15, 3)
        efb = EqualFrequencyBinning(n_bins=4)

        # Before fitting - bin_edges_ should exist but be empty due to __init__
        assert hasattr(efb, "bin_edges_") and len(efb.bin_edges_) == 0

        # After fitting
        efb.fit(X)
        assert hasattr(efb, "bin_edges_")
        assert len(efb.bin_edges_) == 3  # Number of features

        # Each column should have n_bins+1 edges
        for col_edges in efb.bin_edges_.values():
            assert len(col_edges) == 5  # n_bins + 1

    def test_transform_before_fit_error(self):
        """Test that transform raises error when called before fit."""
        X = np.random.rand(10, 2)
        efb = EqualFrequencyBinning(n_bins=3)

        with pytest.raises(RuntimeError, match="not fitted yet"):
            efb.transform(X)

    def test_refit_behavior(self):
        """Test that refitting updates the fitted parameters."""
        X1 = np.random.rand(20, 2) * 10  # Scale 0-10
        X2 = np.random.rand(20, 2) * 100  # Scale 0-100

        efb = EqualFrequencyBinning(n_bins=3)

        # First fit
        efb.fit(X1)
        edges1 = efb.bin_edges_.copy()

        # Second fit with different data
        efb.fit(X2)
        edges2 = efb.bin_edges_

        # Edges should be different due to different data scales
        assert not np.allclose(edges1[0], edges2[0])
        assert not np.allclose(edges1[1], edges2[1])

    def test_parameter_immutability_during_use(self):
        """Test that parameters don't change unexpectedly during use."""
        original_params = {"n_bins": 4, "quantile_range": (0.1, 0.9)}
        efb = EqualFrequencyBinning(**original_params)

        X = np.random.rand(25, 2)
        efb.fit_transform(X)

        # Parameters should remain unchanged
        current_params = efb.get_params()
        for key, value in original_params.items():
            assert current_params[key] == value

    def test_quantile_calculation_error_handling(self):
        """Test error handling when np.quantile fails.

        This test covers lines 232-233 in _equal_frequency_binning.py by
        triggering an exception during quantile calculation and verifying
        that it's properly wrapped and re-raised with context information.
        """
        efb = EqualFrequencyBinning(n_bins=3)

        # Create a mock that will replace np.quantile temporarily
        original_quantile = np.quantile

        def failing_quantile(data, quantiles):
            """Mock quantile function that raises ValueError."""
            raise ValueError("Simulated quantile calculation failure")

        # Patch np.quantile to raise an exception
        import binlearn.methods._equal_frequency_binning as efb_module

        efb_module.np.quantile = failing_quantile

        try:
            data = np.array([1, 2, 3, 4, 5])
            # This should trigger the exception handler on lines 232-233
            with pytest.raises(
                ValueError,
                match="Column 0: Error calculating quantiles: Simulated quantile calculation failure",
            ):
                efb._create_equal_frequency_bins(
                    data, col_id=0, min_quantile=0.0, max_quantile=1.0, n_bins=3
                )
        finally:
            # Restore original function
            efb_module.np.quantile = original_quantile

    def test_quantile_calculation_index_error_handling(self):
        """Test error handling when np.quantile fails with IndexError.

        This test also covers lines 232-233 in _equal_frequency_binning.py by
        triggering an IndexError during quantile calculation and verifying
        that it's properly wrapped and re-raised with context information.
        """
        efb = EqualFrequencyBinning(n_bins=3)

        # Create a mock that will replace np.quantile temporarily
        original_quantile = np.quantile

        def failing_quantile_index_error(data, quantiles):
            """Mock quantile function that raises IndexError."""
            raise IndexError("Simulated index error in quantile calculation")

        # Patch np.quantile to raise an exception
        import binlearn.methods._equal_frequency_binning as efb_module

        efb_module.np.quantile = failing_quantile_index_error

        try:
            data = np.array([1, 2, 3, 4, 5])
            # This should trigger the exception handler on lines 232-233
            with pytest.raises(
                ValueError,
                match="Column test_col: Error calculating quantiles: Simulated index error in quantile calculation",
            ):
                efb._create_equal_frequency_bins(
                    data, col_id="test_col", min_quantile=0.0, max_quantile=1.0, n_bins=3
                )
        finally:
            # Restore original function
            efb_module.np.quantile = original_quantile
