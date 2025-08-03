"""
Comprehensive test suite for KMeansBinning transformer.

This module contains extensive tests for the KMeansBinning class, covering
initialization, parameter validation, fitting, transformation, edge cases,
data type compatibility, sklearn integration, and error handling.

Test Classes:
    TestKMeansBinning: Core functionality tests including initialization,
        validation, fitting, transformation, and basic operations.
    TestKMeansBinningDataTypes: Tests for various data type compatibility
        including pandas DataFrames and polars DataFrames.
    TestKMeansBinningSklearnIntegration: Tests for sklearn compatibility
        including pipeline integration, ColumnTransformer usage, and cloning.
    TestKMeansBinningFitGetParamsWorkflow: Tests for parameter handling
        and sklearn-style workflows.
"""

import kmeans1d
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer

# Import sklearn components
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from binlearn import PANDAS_AVAILABLE, POLARS_AVAILABLE, pd, pl
from binlearn.methods._kmeans_binning import KMeansBinning
from binlearn.utils.errors import ConfigurationError, DataQualityWarning


class TestKMeansBinning:
    """Comprehensive test cases for KMeansBinning core functionality.

    This test class covers the fundamental operations of the KMeansBinning
    transformer including initialization, parameter validation, fitting,
    transformation, edge cases, and basic data handling scenarios.
    """

    def test_init_default(self):
        """Test initialization with default parameters.

        Verifies that the transformer initializes correctly with default
        parameter values and that all attributes are set as expected.
        """

        kmb = KMeansBinning()
        assert kmb.n_bins == 10
        assert kmb.random_state is None

    def test_init_custom_params(self):
        """Test initialization with custom parameters.

        Verifies that the transformer correctly stores custom initialization
        parameter values including n_bins, random_state, and fit_jointly options.
        """

        kmb = KMeansBinning(n_bins=5, random_state=42, fit_jointly=True)
        assert kmb.n_bins == 5
        assert kmb.random_state == 42

    def test_repr(self):
        """Test string representation of the transformer."""

        kmb = KMeansBinning(n_bins=5, random_state=42)
        repr_str = repr(kmb)
        assert "KMeansBinning" in repr_str
        assert "n_bins=5" in repr_str
        assert "random_state=42" in repr_str

    def test_validate_params_invalid_n_bins(self):
        """Test parameter validation with invalid n_bins values.

        Verifies that the validator correctly rejects non-positive n_bins.
        """

        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            KMeansBinning(n_bins=0)

        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            KMeansBinning(n_bins=-1)

    def test_validate_params_invalid_random_state(self):
        """Test parameter validation with invalid random_state values.

        Verifies that the validator correctly rejects invalid random_state
        values (negative integers).
        """

        with pytest.raises(ConfigurationError, match="random_state must be a non-negative integer"):
            KMeansBinning(random_state=-1)

    def test_fit_transform_basic(self):
        """Test basic fit_transform functionality."""

        # Create data with clear clusters
        X = np.array([[1, 10], [2, 11], [3, 12], [20, 30], [21, 31], [22, 32]]).astype(float)
        kmb = KMeansBinning(n_bins=2, random_state=42)

        X_binned = kmb.fit_transform(X)

        assert X_binned.shape == X.shape
        assert np.all(X_binned >= 0)  # All values should be non-negative bin indices

    def test_fit_transform_with_random_state(self):
        """Test fit_transform with custom random_state for reproducibility."""

        # Generate data with some randomness
        np.random.seed(123)
        X = np.random.rand(50, 2) * 100

        kmb1 = KMeansBinning(n_bins=5, random_state=42)
        kmb2 = KMeansBinning(n_bins=5, random_state=42)

        X_binned1 = kmb1.fit_transform(X)
        X_binned2 = kmb2.fit_transform(X)

        # Results should be identical with same random_state
        assert np.array_equal(X_binned1, X_binned2)

    def test_separate_fit_transform(self):
        """Test separate fit and transform calls."""

        X = np.random.rand(20, 3) * 100
        kmb = KMeansBinning(n_bins=4, random_state=42)

        # Fit and transform separately
        kmb.fit(X)
        X_binned = kmb.transform(X)

        assert X_binned.shape == X.shape
        assert hasattr(kmb, "bin_edges_")

    def test_all_nan_column(self):
        """Test behavior with all-NaN column."""

        X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
        kmb = KMeansBinning(n_bins=2)

        # Should handle all-NaN column gracefully and emit warning
        with pytest.warns(DataQualityWarning, match="Data in column 1.*contains only NaN values"):
            kmb.fit(X)
        X_binned = kmb.transform(X)

        assert X_binned.shape == X.shape
        # First column should be binned normally
        assert not np.all(X_binned[:, 0] == -1)
        # Second column should be all MISSING_VALUE (-1)
        from binlearn.utils.constants import MISSING_VALUE

        assert np.all(X_binned[:, 1] == MISSING_VALUE)

    def test_insufficient_data_for_clusters(self):
        """Test error handling with insufficient data for clustering."""

        # Only 2 data points but requesting 5 clusters
        X = np.array([[1.0], [2.0]])
        kmb = KMeansBinning(n_bins=5)

        with pytest.raises(ValueError, match="Insufficient non-NaN values"):
            kmb.fit(X)

    def test_fit_jointly_vs_per_column(self):
        """Test difference between joint and per-column fitting."""

        # Create data with different scales in different columns
        X = np.array([[1, 100], [2, 200], [3, 300], [10, 400], [11, 500], [12, 600]]).astype(float)

        # Per-column fitting (default)
        kmb_per_col = KMeansBinning(n_bins=2, fit_jointly=False, random_state=42)
        X_per_col = kmb_per_col.fit_transform(X)

        # Joint fitting
        kmb_joint = KMeansBinning(n_bins=2, fit_jointly=True, random_state=42)
        X_joint = kmb_joint.fit_transform(X)

        # Results should be different
        assert not np.array_equal(X_per_col, X_joint)

    def test_direct_calculate_bins_basic(self):
        """Test _calculate_bins method directly."""

        kmb = KMeansBinning(n_bins=3, random_state=42)
        data = np.array([1, 2, 3, 10, 11, 12, 20, 21, 22])

        edges, reps = kmb._calculate_bins(data, col_id=0)

        assert len(edges) == 4  # n_bins + 1
        assert len(reps) == 3  # n_bins
        assert edges[0] <= edges[-1]  # Edges should be sorted
        # Check that edges are monotonically increasing
        for i in range(1, len(edges)):
            assert edges[i] >= edges[i - 1]

    def test_direct_calculate_bins_invalid_n_bins(self):
        """Test _calculate_bins with invalid n_bins."""

        kmb = KMeansBinning(n_bins=1)  # Start with valid n_bins
        kmb.n_bins = 0  # Set directly to bypass init validation
        data = np.array([1, 2, 3, 4, 5])

        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            kmb._calculate_bins(data, col_id=0)

    def test_empty_data(self):
        """Test behavior with empty data arrays."""

        X = np.array([]).reshape(0, 2)
        kmb = KMeansBinning(n_bins=3)

        # Empty data should be handled gracefully, not raise an error
        # Should emit warnings for both empty columns
        with pytest.warns(DataQualityWarning, match="Data in column.*contains only NaN values"):
            kmb.fit(X)
        X_binned = kmb.transform(X)
        assert X_binned.shape == (0, 2)

    def test_edge_case_duplicate_values(self):
        """Test handling of data with many duplicate values."""

        # Data with many duplicates
        X = np.array([[1, 1, 1, 1, 1, 2, 2, 3]]).T
        kmb = KMeansBinning(n_bins=3, random_state=42)

        # Should handle duplicates gracefully
        X_binned = kmb.fit_transform(X)
        assert X_binned.shape == X.shape

    def test_edge_case_constant_data(self):
        """Test handling of constant data (all values the same)."""

        # All values are the same
        X = np.array([[5, 5, 5, 5, 5]]).T
        kmb = KMeansBinning(n_bins=3, random_state=42)

        # Should handle constant data gracefully
        X_binned = kmb.fit_transform(X)
        assert X_binned.shape == X.shape

    def test_edge_case_insufficient_unique_values(self):
        """Test handling when there are fewer unique values than desired bins."""

        # Only 2 unique values but requesting 5 bins
        X = np.array([[1, 1, 1, 2, 2, 2]]).T
        kmb = KMeansBinning(n_bins=5, random_state=42)

        # Should handle gracefully by creating bins around unique values
        X_binned = kmb.fit_transform(X)
        assert X_binned.shape == X.shape

    def test_kmeans_clustering_error_handling(self):
        """Test error handling when K-means clustering fails.

        This test covers the exception handling in _create_kmeans_bins by
        triggering an exception during clustering and verifying that it's
        properly wrapped and re-raised with context information.
        """

        kmb = KMeansBinning(n_bins=3)

        # Create a mock that will replace kmeans1d.cluster temporarily

        original_cluster = kmeans1d.cluster

        def failing_cluster(data, k):
            """Mock cluster function that raises an exception."""
            raise RuntimeError("Simulated clustering failure")

        # Patch kmeans1d.cluster to raise an exception
        kmeans1d.cluster = failing_cluster

        try:
            data = np.array([1, 2, 3, 4, 5])
            # This should trigger the exception handler
            with pytest.raises(
                ValueError,
                match="Column 0: Error in K-means clustering: Simulated clustering failure",
            ):
                kmb._create_kmeans_bins(data, col_id=0, n_bins=3)
        finally:
            # Restore original function
            kmeans1d.cluster = original_cluster


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
class TestKMeansBinningDataTypes:
    """Test KMeansBinning with different data types."""

    def test_pandas_dataframe(self):
        """Test with pandas DataFrame input and output."""

        df = pd.DataFrame({"feature1": [1, 2, 3, 20, 21, 22], "feature2": [10, 11, 12, 30, 31, 32]})

        kmb = KMeansBinning(n_bins=2, random_state=42, preserve_dataframe=True)
        df_binned = kmb.fit_transform(df)

        assert isinstance(df_binned, pd.DataFrame)
        assert df_binned.shape == df.shape
        assert list(df_binned.columns) == list(df.columns)

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
    def test_polars_dataframe(self):
        """Test with polars DataFrame input and output."""

        df = pl.DataFrame({"feature1": [1, 2, 3, 20, 21, 22], "feature2": [10, 11, 12, 30, 31, 32]})

        kmb = KMeansBinning(n_bins=2, random_state=42, preserve_dataframe=True)
        df_binned = kmb.fit(df).transform(df)

        assert isinstance(df_binned, pl.DataFrame)
        assert df_binned.shape == df.shape


class TestKMeansBinningSklearnIntegration:
    """Test sklearn compatibility and integration."""

    def test_sklearn_pipeline(self):
        """Test integration with sklearn Pipeline."""

        X = np.random.rand(20, 3) * 100

        pipeline = Pipeline(
            [("binner", KMeansBinning(n_bins=3, random_state=42)), ("scaler", StandardScaler())]
        )

        X_transformed = pipeline.fit_transform(X)
        assert X_transformed.shape == X.shape

    def test_sklearn_column_transformer(self):
        """Test integration with sklearn ColumnTransformer."""

        X = np.random.rand(20, 4) * 100

        ct = ColumnTransformer(
            [
                ("binner", KMeansBinning(n_bins=3, random_state=42), [0, 2]),
                ("scaler", StandardScaler(), [1, 3]),
            ]
        )

        X_transformed = ct.fit_transform(X)
        assert X_transformed.shape[0] == X.shape[0]

    def test_sklearn_clone(self):
        """Test sklearn clone functionality."""

        original = KMeansBinning(n_bins=5, random_state=42)
        cloned = clone(original)

        assert cloned.n_bins == original.n_bins
        assert cloned.random_state == original.random_state
        assert cloned is not original

    def test_get_params(self):
        """Test get_params method for sklearn compatibility."""

        kmb = KMeansBinning(n_bins=7, random_state=123)
        params = kmb.get_params()

        assert params["n_bins"] == 7
        assert params["random_state"] == 123

    def test_set_params(self):
        """Test set_params method for sklearn compatibility."""

        kmb = KMeansBinning()
        kmb.set_params(n_bins=8, random_state=456)

        assert kmb.n_bins == 8
        assert kmb.random_state == 456


class TestKMeansBinningFitGetParamsWorkflow:
    """Test parameter handling and sklearn-style workflows."""

    def test_fit_params_immutability(self):
        """Test that parameters don't change during fitting."""

        original_params = {"n_bins": 6, "random_state": 789}
        kmb = KMeansBinning(
            n_bins=original_params["n_bins"], random_state=original_params["random_state"]
        )

        X = np.random.rand(25, 2) * 100
        kmb.fit(X)

        # Parameters should remain unchanged after fitting
        current_params = kmb.get_params()
        for key, value in original_params.items():
            assert current_params[key] == value

    def test_refit_behavior(self):
        """Test that refitting updates the binning edges appropriately."""

        kmb = KMeansBinning(n_bins=3, random_state=42)

        # First fit
        X1 = np.array([[1, 2], [3, 4], [5, 6], [10, 20], [11, 21], [12, 22]]).astype(float)
        kmb.fit(X1)
        edges1 = kmb.bin_edges_

        # Second fit with different data
        X2 = np.array(
            [[100, 200], [300, 400], [500, 600], [1000, 2000], [1100, 2100], [1200, 2200]]
        ).astype(float)
        kmb.fit(X2)
        edges2 = kmb.bin_edges_

        # Edges should be different due to different data scales
        assert not np.allclose(edges1[0], edges2[0])
        assert not np.allclose(edges1[1], edges2[1])

    def test_parameter_immutability_during_use(self):
        """Test that parameters don't change unexpectedly during use."""

        original_params = {"n_bins": 4, "random_state": 999}
        kmb = KMeansBinning(
            n_bins=original_params["n_bins"], random_state=original_params["random_state"]
        )

        X = np.random.rand(25, 2) * 100
        kmb.fit_transform(X)

        # Parameters should remain unchanged
        current_params = kmb.get_params()
        for key, value in original_params.items():
            assert current_params[key] == value
