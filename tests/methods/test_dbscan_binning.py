"""
Comprehensive test suite for DBSCANBinning transformer.

This module contains extensive tests for the DBSCANBinning class, covering
initialization, parameter validation, fitting, transformation, edge cases,
data type compatibility, sklearn integration, and error handling.

Test Classes:
    TestDBSCANBinning: Core functionality tests including initialization,
        validation, fitting, transformation, and basic operations.
    TestDBSCANBinningDataTypes: Tests for various data type compatibility
        including pandas DataFrames and polars DataFrames.
    TestDBSCANBinningSklearnIntegration: Tests for sklearn compatibility
        including pipeline integration, ColumnTransformer usage, and cloning.
    TestDBSCANBinningFitGetParamsWorkflow: Tests for parameter handling
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
from binlearn.methods._dbscan_binning import DBSCANBinning
from binlearn.utils.errors import ConfigurationError, DataQualityWarning


class TestDBSCANBinning:
    """Comprehensive test cases for DBSCANBinning core functionality.

    This test class covers the fundamental operations of the DBSCANBinning
    transformer including initialization, parameter validation, fitting,
    transformation, edge cases, and basic data handling scenarios.
    """

    def test_init_default(self) -> None:
        """Test initialization with default parameters.

        Verifies that the transformer initializes correctly with default
        parameter values and that all attributes are set as expected.
        """

        dbb = DBSCANBinning()
        assert dbb.eps == 0.1
        assert dbb.min_samples == 5
        assert dbb.min_bins == 2

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters.

        Verifies that the transformer correctly stores custom initialization
        parameter values including eps, min_samples, min_bins, and fit_jointly options.
        """

        dbb = DBSCANBinning(eps=0.2, min_samples=10, min_bins=3, fit_jointly=True)
        assert dbb.eps == 0.2
        assert dbb.min_samples == 10
        assert dbb.min_bins == 3

    def test_repr(self) -> None:
        """Test string representation of the transformer."""

        dbb = DBSCANBinning(eps=0.2, min_samples=10, min_bins=3)
        repr_str = repr(dbb)
        assert "DBSCANBinning" in repr_str
        assert "eps=0.2" in repr_str
        assert "min_samples=10" in repr_str
        assert "min_bins=3" in repr_str

    def test_validate_params_invalid_eps(self) -> None:
        """Test parameter validation with invalid eps values.

        Verifies that the validator correctly rejects non-positive eps.
        """

        with pytest.raises(ConfigurationError, match="eps must be a positive number"):
            DBSCANBinning(eps=0)

        with pytest.raises(ConfigurationError, match="eps must be a positive number"):
            DBSCANBinning(eps=-0.1)

    def test_validate_params_invalid_min_samples(self) -> None:
        """Test parameter validation with invalid min_samples values.

        Verifies that the validator correctly rejects invalid min_samples values.
        """

        with pytest.raises(ConfigurationError, match="min_samples must be a positive integer"):
            DBSCANBinning(min_samples=0)

        with pytest.raises(ConfigurationError, match="min_samples must be a positive integer"):
            DBSCANBinning(min_samples=-1)

    def test_validate_params_invalid_min_bins(self) -> None:
        """Test parameter validation with invalid min_bins values.

        Verifies that the validator correctly rejects invalid min_bins values.
        """

        with pytest.raises(ConfigurationError, match="min_bins must be a positive integer"):
            DBSCANBinning(min_bins=0)

        with pytest.raises(ConfigurationError, match="min_bins must be a positive integer"):
            DBSCANBinning(min_bins=-1)

    def test_fit_transform_basic(self) -> None:
        """Test basic fit_transform functionality."""

        # Create data with clear density-based clusters
        np.random.seed(42)
        cluster1 = np.random.normal(5, 0.5, 20).reshape(-1, 1)
        cluster2 = np.random.normal(15, 0.5, 20).reshape(-1, 1)
        # Add some noise points
        noise = np.random.uniform(0, 25, 5).reshape(-1, 1)
        X = np.vstack([cluster1, cluster2, noise])

        dbb = DBSCANBinning(eps=1.0, min_samples=3, min_bins=2)

        X_binned = dbb.fit_transform(X)

        assert X_binned.shape == X.shape
        assert np.all(X_binned >= 0)  # All values should be non-negative bin indices

    def test_fit_transform_fallback_to_equal_width(self) -> None:
        """Test fallback to equal-width binning when DBSCAN finds too few clusters."""

        # Create uniform data where DBSCAN might not find enough clusters
        X = np.random.uniform(0, 10, (30, 1))

        dbb = DBSCANBinning(eps=0.1, min_samples=10, min_bins=5)  # Strict parameters

        X_binned = dbb.fit_transform(X)

        assert X_binned.shape == X.shape
        assert np.all(X_binned >= 0)

    def test_separate_fit_transform(self) -> None:
        """Test separate fit and transform calls."""

        X = np.random.rand(20, 3) * 100
        dbb = DBSCANBinning(eps=5.0, min_samples=3, min_bins=2)

        # Fit and transform separately
        dbb.fit(X)
        X_binned = dbb.transform(X)

        assert X_binned.shape == X.shape
        assert hasattr(dbb, "bin_edges_")

    def test_all_nan_column(self) -> None:
        """Test behavior with all-NaN column."""

        X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
        dbb = DBSCANBinning(eps=0.5, min_samples=2)

        # Should handle all-NaN column gracefully and emit warning
        with pytest.warns(DataQualityWarning, match="Data in column 1.*contains only NaN values"):
            dbb.fit(X)
        X_binned = dbb.transform(X)

        assert X_binned.shape == X.shape
        # First column should be binned normally
        assert not np.all(X_binned[:, 0] == -1)
        # Second column should be all MISSING_VALUE (-1)
        from binlearn.utils.constants import MISSING_VALUE

        assert np.all(X_binned[:, 1] == MISSING_VALUE)

    def test_insufficient_data_for_clustering(self) -> None:
        """Test handling with insufficient data for DBSCAN clustering."""

        # Only 2 data points but min_samples=5
        X = np.array([[1.0], [2.0]])
        dbb = DBSCANBinning(eps=0.5, min_samples=5, min_bins=2)

        # Should fall back to equal-width binning
        X_binned = dbb.fit_transform(X)
        assert X_binned.shape == X.shape

    def test_fit_jointly_vs_per_column(self) -> None:
        """Test difference between joint and per-column fitting."""

        # Create data with different scales in different columns
        X = np.array([[1, 100], [2, 200], [3, 300], [10, 400], [11, 500], [12, 600]]).astype(float)

        # Per-column fitting (default)
        dbb_per_col = DBSCANBinning(eps=2.0, min_samples=2, fit_jointly=False)
        X_per_col = dbb_per_col.fit_transform(X)

        # Joint fitting
        dbb_joint = DBSCANBinning(eps=50.0, min_samples=2, fit_jointly=True)
        X_joint = dbb_joint.fit_transform(X)

        # Results should be different
        assert not np.array_equal(X_per_col, X_joint)

    def test_direct_calculate_bins_basic(self) -> None:
        """Test _calculate_bins method directly."""

        dbb = DBSCANBinning(eps=2.0, min_samples=3, min_bins=2)

        # Create data with clear density-based structure
        np.random.seed(42)
        data = np.concatenate([np.random.normal(5, 0.5, 10), np.random.normal(15, 0.5, 10)])

        edges, reps = dbb._calculate_bins(data, col_id=0)

        assert len(edges) >= 2  # At least min_bins + 1
        assert len(reps) >= 1  # At least min_bins
        assert edges[0] <= edges[-1]  # Edges should be sorted
        # Check that edges are monotonically increasing
        for i in range(1, len(edges)):
            assert edges[i] >= edges[i - 1]

    def test_empty_data(self) -> None:
        """Test behavior with empty data arrays."""

        X = np.array([]).reshape(0, 2)
        dbb = DBSCANBinning(eps=0.5, min_samples=3)

        # Empty data should be handled gracefully, not raise an error
        # Should emit warnings for both empty columns
        with pytest.warns(DataQualityWarning, match="Data in column.*contains only NaN values"):
            dbb.fit(X)
        X_binned = dbb.transform(X)
        assert X_binned.shape == (0, 2)

    def test_edge_case_duplicate_values(self) -> None:
        """Test handling of data with many duplicate values."""

        # Data with many duplicates
        X = np.array([[1, 1, 1, 1, 1, 2, 2, 3]]).T
        dbb = DBSCANBinning(eps=0.5, min_samples=3)

        # Should handle duplicates gracefully
        X_binned = dbb.fit_transform(X)
        assert X_binned.shape == X.shape

    def test_edge_case_constant_data(self) -> None:
        """Test handling of constant data (all values the same)."""

        # All values are the same
        X = np.array([[5, 5, 5, 5, 5]]).T
        dbb = DBSCANBinning(eps=0.5, min_samples=3)

        # Should handle constant data gracefully
        X_binned = dbb.fit_transform(X)
        assert X_binned.shape == X.shape

    def test_edge_case_insufficient_unique_values(self) -> None:
        """Test handling when there are fewer unique values than desired bins."""

        # Only 2 unique values but requesting 5 bins
        X = np.array([[1, 1, 1, 2, 2, 2]]).T
        dbb = DBSCANBinning(eps=0.1, min_samples=2, min_bins=5)

        # Should handle gracefully by falling back to equal-width binning
        X_binned = dbb.fit_transform(X)
        assert X_binned.shape == X.shape

    def test_dbscan_clustering_error_handling(self) -> None:
        """Test error handling when DBSCAN clustering encounters errors.

        This test covers the exception handling in _create_dbscan_bins by
        testing with problematic data and verifying fallback behavior.
        """

        dbb = DBSCANBinning(eps=0.1, min_samples=3, min_bins=2)

        # Create data that might cause DBSCAN issues (extreme values)
        data = np.array([1e-15, 1e-15, 1e-15, 1e15, 1e15])

        # Should handle DBSCAN issues gracefully by falling back
        edges, reps = dbb._create_dbscan_bins(data, col_id=0)
        assert len(edges) >= 2
        assert len(reps) >= 1

    def test_different_eps_values(self) -> None:
        """Test with different eps values to verify behavior."""

        np.random.seed(42)
        # Create data with clear clusters
        cluster1 = np.random.normal(5, 0.5, 15).reshape(-1, 1)
        cluster2 = np.random.normal(15, 0.5, 15).reshape(-1, 1)
        X = np.vstack([cluster1, cluster2])

        eps_values = [0.5, 1.0, 2.0, 5.0]

        for eps in eps_values:
            dbb = DBSCANBinning(eps=eps, min_samples=3, min_bins=2)
            X_binned = dbb.fit_transform(X)
            assert X_binned.shape == X.shape

    def test_different_min_samples_values(self) -> None:
        """Test with different min_samples values to verify behavior."""

        np.random.seed(42)
        X = np.random.normal(10, 5, (50, 1))

        min_samples_values = [2, 5, 10, 15]

        for min_samples in min_samples_values:
            dbb = DBSCANBinning(eps=2.0, min_samples=min_samples, min_bins=2)
            X_binned = dbb.fit_transform(X)
            assert X_binned.shape == X.shape

    def test_noise_handling(self) -> None:
        """Test that DBSCAN properly handles noise points."""

        np.random.seed(42)
        # Create well-separated clusters with noise
        cluster1 = np.random.normal(0, 0.5, 10)
        cluster2 = np.random.normal(10, 0.5, 10)
        noise = np.array([5, -5, 15])  # Isolated noise points

        X = np.concatenate([cluster1, cluster2, noise]).reshape(-1, 1)

        dbb = DBSCANBinning(eps=1.0, min_samples=3, min_bins=2)
        X_binned = dbb.fit_transform(X)

        assert X_binned.shape == X.shape
        # Should successfully bin the data even with noise points


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
class TestDBSCANBinningDataTypes:
    """Test DBSCANBinning with different data types."""

    def test_pandas_dataframe(self) -> None:
        """Test with pandas DataFrame input and output."""

        np.random.seed(42)
        df = pd.DataFrame(
            {
                "feature1": np.concatenate(
                    [np.random.normal(5, 1, 15), np.random.normal(15, 1, 15)]
                ),
                "feature2": np.concatenate(
                    [np.random.normal(10, 1, 15), np.random.normal(20, 1, 15)]
                ),
            }
        )

        dbb = DBSCANBinning(eps=2.0, min_samples=3, preserve_dataframe=True)
        df_binned = dbb.fit_transform(df)

        assert isinstance(df_binned, pd.DataFrame)
        assert df_binned.shape == df.shape
        assert list(df_binned.columns) == list(df.columns)

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
    def test_polars_dataframe(self) -> None:
        """Test with polars DataFrame input and output."""

        np.random.seed(42)
        df = pl.DataFrame(
            {
                "feature1": np.concatenate(
                    [np.random.normal(5, 1, 15), np.random.normal(15, 1, 15)]
                ),
                "feature2": np.concatenate(
                    [np.random.normal(10, 1, 15), np.random.normal(20, 1, 15)]
                ),
            }
        )

        dbb = DBSCANBinning(eps=2.0, min_samples=3, preserve_dataframe=True)
        df_binned = dbb.fit(df).transform(df)

        assert isinstance(df_binned, pl.DataFrame)
        assert df_binned.shape == df.shape


class TestDBSCANBinningSklearnIntegration:
    """Test sklearn compatibility and integration."""

    def test_sklearn_pipeline(self) -> None:
        """Test integration with sklearn Pipeline."""

        X = np.random.rand(20, 3) * 100

        pipeline = Pipeline(
            [
                ("binner", DBSCANBinning(eps=10.0, min_samples=3, min_bins=2)),
                ("scaler", StandardScaler()),
            ]
        )

        X_transformed = pipeline.fit_transform(X)
        assert X_transformed.shape == X.shape

    def test_sklearn_column_transformer(self) -> None:
        """Test integration with sklearn ColumnTransformer."""

        X = np.random.rand(20, 4) * 100

        ct = ColumnTransformer(
            [
                ("binner", DBSCANBinning(eps=10.0, min_samples=3, min_bins=2), [0, 2]),
                ("scaler", StandardScaler(), [1, 3]),
            ]
        )

        X_transformed = ct.fit_transform(X)
        assert X_transformed.shape[0] == X.shape[0]

    def test_sklearn_clone(self) -> None:
        """Test sklearn clone functionality."""

        original = DBSCANBinning(eps=0.2, min_samples=10, min_bins=3)
        cloned = clone(original)

        assert cloned.eps == original.eps
        assert cloned.min_samples == original.min_samples
        assert cloned.min_bins == original.min_bins
        assert cloned is not original

    def test_get_params(self) -> None:
        """Test get_params method for sklearn compatibility."""

        dbb = DBSCANBinning(eps=0.3, min_samples=7, min_bins=4)
        params = dbb.get_params()

        assert params["eps"] == 0.3
        assert params["min_samples"] == 7
        assert params["min_bins"] == 4

    def test_set_params(self) -> None:
        """Test set_params method for sklearn compatibility."""

        dbb = DBSCANBinning()
        dbb.set_params(eps=0.4, min_samples=8, min_bins=5)

        assert dbb.eps == 0.4
        assert dbb.min_samples == 8
        assert dbb.min_bins == 5


class TestDBSCANBinningFitGetParamsWorkflow:
    """Test parameter handling and sklearn-style workflows."""

    def test_fit_params_immutability(self) -> None:
        """Test that parameters don't change during fitting."""

        original_params = {"eps": 0.15, "min_samples": 6, "min_bins": 3}
        dbb = DBSCANBinning(
            eps=original_params["eps"],
            min_samples=original_params["min_samples"],
            min_bins=original_params["min_bins"],
        )

        X = np.random.rand(25, 2) * 100
        dbb.fit(X)

        # Parameters should remain unchanged after fitting
        current_params = dbb.get_params()
        for key, value in original_params.items():
            assert current_params[key] == value

    def test_refit_behavior(self) -> None:
        """Test that refitting updates the binning edges appropriately."""

        dbb = DBSCANBinning(eps=2.0, min_samples=3, min_bins=2)

        # First fit
        np.random.seed(42)
        X1 = np.random.normal(10, 2, (20, 2))
        dbb.fit(X1)
        edges1 = dbb.bin_edges_

        # Second fit with different data
        np.random.seed(123)
        X2 = np.random.normal(100, 20, (20, 2))
        dbb.fit(X2)
        edges2 = dbb.bin_edges_

        # Edges should be different due to different data scales
        assert not np.allclose(edges1[0], edges2[0])
        assert not np.allclose(edges1[1], edges2[1])

    def test_parameter_immutability_during_use(self) -> None:
        """Test that parameters don't change unexpectedly during use."""

        original_params = {"eps": 1.5, "min_samples": 4, "min_bins": 2}
        dbb = DBSCANBinning(
            eps=original_params["eps"],
            min_samples=original_params["min_samples"],
            min_bins=original_params["min_bins"],
        )

        X = np.random.rand(25, 2) * 100
        dbb.fit_transform(X)

        # Parameters should remain unchanged
        current_params = dbb.get_params()
        for key, value in original_params.items():
            assert current_params[key] == value


class TestDBSCANBinningExceptionHandling:
    """Test exception handling and edge cases."""

    def test_insufficient_data_fallback(self) -> None:
        """Test fallback behavior when there's insufficient data for DBSCAN."""

        db = DBSCANBinning(
            eps=0.5, min_samples=5, min_bins=3
        )  # Requires 5 samples but we'll give fewer

        # Create data with insufficient samples for clustering
        data = np.array([1.0, 2.0, 3.0])  # Only 3 samples, but min_samples=5

        edges, reps = db._create_dbscan_bins(data, col_id=0)

        # Should fallback to equal width binning with min_bins
        assert len(edges) == 4  # min_bins + 1
        assert len(reps) == 3  # min_bins

    def test_insufficient_data_fallback_constant_data(self) -> None:
        """Test fallback behavior with constant data - covers lines 219-220."""

        db = DBSCANBinning(
            eps=0.5, min_samples=5, min_bins=3
        )  # Requires 5 samples but we'll give fewer

        # Create constant data with insufficient samples for clustering
        data = np.array([5.0, 5.0])  # Only 2 constant samples, but min_samples=5

        edges, reps = db._create_dbscan_bins(data, col_id=0)

        # Should fallback to equal width binning with constant data handling (epsilon)
        assert len(edges) == 4  # min_bins + 1
        assert len(reps) == 3  # min_bins
        # Check that edges span around the constant value
        assert edges[0] < 5.0 < edges[-1]

    def test_dbscan_clustering_exception_handling(self) -> None:
        """Test exception handling in DBSCAN clustering."""

        db = DBSCANBinning(eps=0.5, min_samples=2, min_bins=3)

        # Mock DBSCAN to raise an exception
        from unittest.mock import patch

        with patch("binlearn.methods._dbscan_binning.DBSCAN") as mock_dbscan:
            mock_dbscan.return_value.fit_predict.side_effect = Exception("DBSCAN clustering failed")

            # Should handle the exception and fallback to equal width binning
            edges, reps = db._create_dbscan_bins(np.array([1, 2, 3, 4, 5]), col_id=0)

            # Should fallback to equal width binning
            assert len(edges) == 4  # min_bins + 1
            assert len(reps) == 3  # min_bins

    def test_dbscan_clustering_exception_on_labels_processing(self) -> None:
        """Test exception handling during label processing."""

        db = DBSCANBinning(eps=0.5, min_samples=2, min_bins=2)

        # Mock numpy.mean to raise an exception during cluster center calculation
        from unittest.mock import patch

        with patch("numpy.mean") as mock_mean:
            mock_mean.side_effect = Exception("Mean calculation failed")

            # Should handle the exception and fallback to equal width binning
            edges, reps = db._create_dbscan_bins(np.array([1, 2, 3, 4, 5]), col_id=0)

            # Should fallback to equal width binning
            assert len(edges) == 3  # min_bins + 1
            assert len(reps) == 2  # min_bins

    def test_single_cluster_edge_extension(self) -> None:
        """Test edge extension logic for single cluster case."""

        # Create data that will form exactly one cluster to trigger single cluster logic
        db = DBSCANBinning(eps=10.0, min_samples=2, min_bins=2)  # Large eps to force single cluster

        # Create data that forms a single cluster
        data = np.array([10.0, 10.1, 10.2, 10.3, 10.4])

        edges, reps = db._create_dbscan_bins(data, col_id=0)

        # Should have bins based on single cluster - either min_bins or single bin
        assert len(edges) >= 2
        assert len(reps) >= 1

    def test_single_cluster_zero_center_edge_extension(self) -> None:
        """Test edge extension for single cluster with zero center - covers lines 289, 306."""

        # Create a single cluster around zero to test the zero-handling edge case
        db = DBSCANBinning(eps=1.0, min_samples=1, min_bins=2)

        # Data clustered exactly at zero to trigger the zero case
        data = np.array([0.0, 0.0, 0.0])

        edges, reps = db._create_dbscan_bins(data, col_id=0)

        # Should handle single cluster around zero properly
        assert len(edges) >= 2
        assert len(reps) >= 1

    def test_single_cluster_nonzero_center_edge_extension(self) -> None:
        """Test edge extension for single cluster with non-zero center - covers lines 288, 305."""

        # Create a single cluster away from zero to test non-zero edge calculation
        db = DBSCANBinning(eps=1.0, min_samples=1, min_bins=2)

        # Data clustered around a non-zero value
        data = np.array([10.0, 10.0, 10.0])

        edges, reps = db._create_dbscan_bins(data, col_id=0)

        # Should handle single cluster with proper edge extension
        assert len(edges) >= 2
        assert len(reps) >= 1

    def test_edge_extension_for_single_cluster_case(self) -> None:
        """Test another edge extension scenario for single cluster."""

        db = DBSCANBinning(eps=1.0, min_samples=1, min_bins=2)

        # Create tight clustered data
        data = np.array([5.0, 5.0, 5.0, 5.0])

        edges, reps = db._create_dbscan_bins(data, col_id=0)

        # Should handle single cluster with proper edge extension
        assert len(edges) >= 2
        assert len(reps) >= 1

    def test_calculate_first_edge_data_min_case(self) -> None:
        """Test _calculate_first_edge when cluster center > data_min."""

        db = DBSCANBinning(eps=0.5, min_samples=2, min_bins=2)

        # Test case where cluster center is greater than data min
        cluster_centers = [5.0, 10.0]
        data_min = 1.0  # Much smaller than first cluster center

        first_edge = db._calculate_first_edge(cluster_centers, data_min)
        assert first_edge == data_min

    def test_calculate_first_edge_extension_multiple_clusters(self) -> None:
        """Test _calculate_first_edge extension with multiple clusters."""

        db = DBSCANBinning(eps=0.5, min_samples=2, min_bins=2)

        # Test case where cluster center <= data_min with multiple clusters
        cluster_centers = [10.0, 20.0]
        data_min = 10.0  # Equal to first cluster center

        first_edge = db._calculate_first_edge(cluster_centers, data_min)
        expected = 10.0 - (20.0 - 10.0) * 0.05  # 10.0 - 0.5 = 9.5
        assert first_edge == expected

    def test_calculate_first_edge_extension_single_cluster_nonzero(self) -> None:
        """Test _calculate_first_edge extension with single non-zero cluster."""

        db = DBSCANBinning(eps=0.5, min_samples=2, min_bins=2)

        # Test case with single cluster at non-zero value
        cluster_centers = [10.0]
        data_min = 10.0

        first_edge = db._calculate_first_edge(cluster_centers, data_min)
        expected = 10.0 - abs(10.0) * 0.05  # 10.0 - 0.5 = 9.5
        assert first_edge == expected

    def test_calculate_first_edge_extension_single_cluster_zero(self) -> None:
        """Test _calculate_first_edge extension with single cluster at zero."""

        db = DBSCANBinning(eps=0.5, min_samples=2, min_bins=2)

        # Test case with single cluster at zero
        cluster_centers = [0.0]
        data_min = 0.0

        first_edge = db._calculate_first_edge(cluster_centers, data_min)
        expected = 0.0 - 0.05  # Default extension for zero
        assert first_edge == expected

    def test_calculate_last_edge_data_max_case(self) -> None:
        """Test _calculate_last_edge when cluster center < data_max."""

        db = DBSCANBinning(eps=0.5, min_samples=2, min_bins=2)

        # Test case where last cluster center is less than data max
        cluster_centers = [5.0, 10.0]
        data_max = 15.0  # Much larger than last cluster center

        last_edge = db._calculate_last_edge(cluster_centers, data_max)
        assert last_edge == data_max

    def test_calculate_last_edge_extension_multiple_clusters(self) -> None:
        """Test _calculate_last_edge extension with multiple clusters."""

        db = DBSCANBinning(eps=0.5, min_samples=2, min_bins=2)

        # Test case where last cluster center >= data_max with multiple clusters
        cluster_centers = [10.0, 20.0]
        data_max = 20.0  # Equal to last cluster center

        last_edge = db._calculate_last_edge(cluster_centers, data_max)
        expected = 20.0 + (20.0 - 10.0) * 0.05  # 20.0 + 0.5 = 20.5
        assert last_edge == expected

    def test_calculate_last_edge_extension_single_cluster_nonzero(self) -> None:
        """Test _calculate_last_edge extension with single non-zero cluster."""

        db = DBSCANBinning(eps=0.5, min_samples=2, min_bins=2)

        # Test case with single cluster at non-zero value
        cluster_centers = [10.0]
        data_max = 10.0

        last_edge = db._calculate_last_edge(cluster_centers, data_max)
        expected = 10.0 + abs(10.0) * 0.05  # 10.0 + 0.5 = 10.5
        assert last_edge == expected

    def test_calculate_last_edge_extension_single_cluster_zero(self) -> None:
        """Test _calculate_last_edge extension with single cluster at zero."""

        db = DBSCANBinning(eps=0.5, min_samples=2, min_bins=2)

        # Test case with single cluster at zero
        cluster_centers = [0.0]
        data_max = 0.0

        last_edge = db._calculate_last_edge(cluster_centers, data_max)
        expected = 0.0 + 0.05  # Default extension for zero
        assert last_edge == expected
