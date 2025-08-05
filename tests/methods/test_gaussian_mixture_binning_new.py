"""
Comprehensive test suite for GaussianMixtureBinning transformer.

This module contains extensive tests for the GaussianMixtureBinning class, covering
initialization, parameter validation, fitting, transformation, edge cases,
data type compatibility, sklearn integration, and error handling.

Test Classes:
    TestGaussianMixtureBinning: Core functionality tests including initialization,
        validation, fitting, transformation, and basic operations.
    TestGaussianMixtureBinningDataTypes: Tests for various data type compatibility
        including pandas DataFrames and polars DataFrames.
    TestGaussianMixtureBinningSklearnIntegration: Tests for sklearn compatibility
        including pipeline integration, ColumnTransformer usage, and cloning.
    TestGaussianMixtureBinningFitGetParamsWorkflow: Tests for parameter handling
        and sklearn-style workflows.
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.mixture import GaussianMixture

# Import sklearn components
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from binlearn import PANDAS_AVAILABLE, POLARS_AVAILABLE, pd, pl
from binlearn.methods._gaussian_mixture_binning import GaussianMixtureBinning
from binlearn.utils.errors import ConfigurationError, DataQualityWarning


class TestGaussianMixtureBinning:
    """Comprehensive test cases for GaussianMixtureBinning core functionality.

    This test class covers the fundamental operations of the GaussianMixtureBinning
    transformer including initialization, parameter validation, fitting,
    transformation, edge cases, and basic data handling scenarios.
    """

    def test_init_default(self):
        """Test initialization with default parameters.

        Verifies that the transformer initializes correctly with default
        parameter values and that all attributes are set as expected.
        """

        gmb = GaussianMixtureBinning()
        assert gmb.n_components == 10
        assert gmb.random_state is None

    def test_init_custom_params(self):
        """Test initialization with custom parameters.

        Verifies that the transformer correctly stores custom initialization
        parameter values including n_components, random_state, and fit_jointly options.
        """

        gmb = GaussianMixtureBinning(n_components=5, random_state=42, fit_jointly=True)
        assert gmb.n_components == 5
        assert gmb.random_state == 42

    def test_repr(self):
        """Test string representation of the transformer."""

        gmb = GaussianMixtureBinning(n_components=5, random_state=42)
        repr_str = repr(gmb)
        assert "GaussianMixtureBinning" in repr_str
        assert "n_components=5" in repr_str
        assert "random_state=42" in repr_str

    def test_validate_params_invalid_n_components(self):
        """Test parameter validation with invalid n_components values.

        Verifies that the validator correctly rejects non-positive n_components.
        """

        with pytest.raises(ConfigurationError, match="n_components must be a positive integer"):
            GaussianMixtureBinning(n_components=0)

        with pytest.raises(ConfigurationError, match="n_components must be a positive integer"):
            GaussianMixtureBinning(n_components=-1)

    def test_validate_params_invalid_random_state(self):
        """Test parameter validation with invalid random_state values.

        Verifies that the validator correctly rejects invalid random_state
        values (negative integers).
        """

        with pytest.raises(ConfigurationError, match="random_state must be a non-negative integer"):
            GaussianMixtureBinning(random_state=-1)

    def test_fit_transform_basic(self):
        """Test basic fit_transform functionality."""

        # Create data with clear clusters (mixed Gaussians)
        np.random.seed(42)
        cluster1 = np.random.normal(5, 1, 30).reshape(-1, 1)
        cluster2 = np.random.normal(15, 1, 30).reshape(-1, 1)
        X = np.vstack([cluster1, cluster2])

        gmb = GaussianMixtureBinning(n_components=2, random_state=42)

        X_binned = gmb.fit_transform(X)

        assert X_binned.shape == X.shape
        assert np.all(X_binned >= 0)  # All values should be non-negative bin indices

    def test_fit_transform_with_random_state(self):
        """Test fit_transform with custom random_state for reproducibility."""

        # Generate data with some randomness
        np.random.seed(123)
        X = np.random.rand(50, 2) * 100

        gmb1 = GaussianMixtureBinning(n_components=5, random_state=42)
        gmb2 = GaussianMixtureBinning(n_components=5, random_state=42)

        X_binned1 = gmb1.fit_transform(X)
        X_binned2 = gmb2.fit_transform(X)

        # Results should be identical with same random_state
        assert np.array_equal(X_binned1, X_binned2)

    def test_separate_fit_transform(self):
        """Test separate fit and transform calls."""

        X = np.random.rand(20, 3) * 100
        gmb = GaussianMixtureBinning(n_components=4, random_state=42)

        # Fit and transform separately
        gmb.fit(X)
        X_binned = gmb.transform(X)

        assert X_binned.shape == X.shape
        assert hasattr(gmb, "bin_edges_")

    def test_all_nan_column(self):
        """Test behavior with all-NaN column."""

        X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
        gmb = GaussianMixtureBinning(n_components=2)

        # Should handle all-NaN column gracefully and emit warning
        with pytest.warns(DataQualityWarning, match="Data in column 1.*contains only NaN values"):
            gmb.fit(X)
        X_binned = gmb.transform(X)

        assert X_binned.shape == X.shape
        # First column should be binned normally
        assert not np.all(X_binned[:, 0] == -1)
        # Second column should be all MISSING_VALUE (-1)
        from binlearn.utils.constants import MISSING_VALUE

        assert np.all(X_binned[:, 1] == MISSING_VALUE)

    def test_insufficient_data_for_components(self):
        """Test error handling with insufficient data for components."""

        # Only 2 data points but requesting 5 components
        X = np.array([[1.0], [2.0]])
        gmb = GaussianMixtureBinning(n_components=5)

        # Should raise ValueError for insufficient data
        with pytest.raises(ValueError, match="Insufficient non-NaN values"):
            gmb.fit(X)

    def test_fit_jointly_vs_per_column(self):
        """Test difference between joint and per-column fitting."""

        # Create data with different scales in different columns
        X = np.array([[1, 100], [2, 200], [3, 300], [10, 400], [11, 500], [12, 600]]).astype(float)

        # Per-column fitting (default)
        gmb_per_col = GaussianMixtureBinning(n_components=2, fit_jointly=False, random_state=42)
        X_per_col = gmb_per_col.fit_transform(X)

        # Joint fitting
        gmb_joint = GaussianMixtureBinning(n_components=2, fit_jointly=True, random_state=42)
        X_joint = gmb_joint.fit_transform(X)

        # Results should be different
        assert not np.array_equal(X_per_col, X_joint)

    def test_direct_calculate_bins_basic(self):
        """Test _calculate_bins method directly."""

        gmb = GaussianMixtureBinning(n_components=3, random_state=42)

        # Create data with clear structure
        np.random.seed(42)
        data = np.concatenate(
            [np.random.normal(5, 1, 10), np.random.normal(15, 1, 10), np.random.normal(25, 1, 10)]
        )

        edges, reps = gmb._calculate_bins(data, col_id=0)

        assert len(edges) >= 2  # At least min_components + 1
        assert len(reps) >= 1  # At least min_components
        assert edges[0] <= edges[-1]  # Edges should be sorted
        # Check that edges are monotonically increasing
        for i in range(1, len(edges)):
            assert edges[i] >= edges[i - 1]

    def test_empty_data(self):
        """Test behavior with empty data arrays."""

        X = np.array([]).reshape(0, 2)
        gmb = GaussianMixtureBinning(n_components=3)

        # Empty data should be handled gracefully, not raise an error
        # Should emit warnings for both empty columns
        with pytest.warns(DataQualityWarning, match="Data in column.*contains only NaN values"):
            gmb.fit(X)
        X_binned = gmb.transform(X)
        assert X_binned.shape == (0, 2)

    def test_edge_case_duplicate_values(self):
        """Test handling of data with many duplicate values."""

        # Data with many duplicates
        X = np.array([[1, 1, 1, 1, 1, 2, 2, 3]]).T
        gmb = GaussianMixtureBinning(n_components=3, random_state=42)

        # Should handle duplicates gracefully
        X_binned = gmb.fit_transform(X)
        assert X_binned.shape == X.shape

    def test_edge_case_constant_data(self):
        """Test handling of constant data (all values the same)."""

        # All values are the same
        X = np.array([[5, 5, 5, 5, 5]]).T
        gmb = GaussianMixtureBinning(n_components=3, random_state=42)

        # Should handle constant data gracefully
        X_binned = gmb.fit_transform(X)
        assert X_binned.shape == X.shape

    def test_edge_case_insufficient_unique_values(self):
        """Test handling when there are fewer unique values than desired components."""

        # Only 2 unique values but requesting 5 components
        X = np.array([[1, 1, 1, 2, 2, 2]]).T
        gmb = GaussianMixtureBinning(n_components=5, random_state=42)

        # Should handle gracefully by falling back to unique value bins
        X_binned = gmb.fit_transform(X)
        assert X_binned.shape == X.shape

    def test_gaussian_mixture_error_handling(self):
        """Test error handling when Gaussian mixture fitting fails.

        This test covers the exception handling in _create_gmm_bins by
        triggering an exception during fitting and verifying fallback behavior.
        """

        gmb = GaussianMixtureBinning(n_components=3)

        # Create data that might cause convergence issues
        data = np.array([1e-10, 1e-10, 1e-10, 1e10, 1e10])

        # Should handle fitting issues gracefully by falling back
        edges, reps = gmb._create_gmm_bins(data, col_id=0, n_components=3)
        assert len(edges) >= 2
        assert len(reps) >= 1

    def test_direct_calculate_bins_invalid_n_components(self):
        """Test _calculate_bins with invalid n_components."""

        gmb = GaussianMixtureBinning(n_components=1)  # Start with valid n_components
        gmb.n_components = 0  # Set directly to bypass init validation
        data = np.array([1, 2, 3, 4, 5])

        with pytest.raises(ValueError, match="n_components must be >= 1"):
            gmb._calculate_bins(data, col_id=0)


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
class TestGaussianMixtureBinningDataTypes:
    """Test GaussianMixtureBinning with different data types."""

    def test_pandas_dataframe(self):
        """Test with pandas DataFrame input and output."""

        np.random.seed(42)
        df = pd.DataFrame(
            {"feature1": np.random.normal(5, 2, 20), "feature2": np.random.normal(15, 3, 20)}
        )

        gmb = GaussianMixtureBinning(n_components=2, random_state=42, preserve_dataframe=True)
        df_binned = gmb.fit_transform(df)

        assert isinstance(df_binned, pd.DataFrame)
        assert df_binned.shape == df.shape
        assert list(df_binned.columns) == list(df.columns)

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
    def test_polars_dataframe(self):
        """Test with polars DataFrame input and output."""

        np.random.seed(42)
        df = pl.DataFrame(
            {"feature1": np.random.normal(5, 2, 20), "feature2": np.random.normal(15, 3, 20)}
        )

        gmb = GaussianMixtureBinning(n_components=2, random_state=42, preserve_dataframe=True)
        df_binned = gmb.fit(df).transform(df)

        assert isinstance(df_binned, pl.DataFrame)
        assert df_binned.shape == df.shape


class TestGaussianMixtureBinningSklearnIntegration:
    """Test sklearn compatibility and integration."""

    def test_sklearn_pipeline(self):
        """Test integration with sklearn Pipeline."""

        X = np.random.rand(20, 3) * 100

        pipeline = Pipeline(
            [
                ("binner", GaussianMixtureBinning(n_components=3, random_state=42)),
                ("scaler", StandardScaler()),
            ]
        )

        X_transformed = pipeline.fit_transform(X)
        assert X_transformed.shape == X.shape

    def test_sklearn_column_transformer(self):
        """Test integration with sklearn ColumnTransformer."""

        X = np.random.rand(20, 4) * 100

        ct = ColumnTransformer(
            [
                ("binner", GaussianMixtureBinning(n_components=3, random_state=42), [0, 2]),
                ("scaler", StandardScaler(), [1, 3]),
            ]
        )

        X_transformed = ct.fit_transform(X)
        assert X_transformed.shape[0] == X.shape[0]

    def test_sklearn_clone(self):
        """Test sklearn clone functionality."""

        original = GaussianMixtureBinning(n_components=5, random_state=42)
        cloned = clone(original)

        assert cloned.n_components == original.n_components
        assert cloned.random_state == original.random_state
        assert cloned is not original

    def test_get_params(self):
        """Test get_params method for sklearn compatibility."""

        gmb = GaussianMixtureBinning(n_components=7, random_state=123)
        params = gmb.get_params()

        assert params["n_components"] == 7
        assert params["random_state"] == 123

    def test_set_params(self):
        """Test set_params method for sklearn compatibility."""

        gmb = GaussianMixtureBinning()
        gmb.set_params(n_components=8, random_state=456)

        assert gmb.n_components == 8
        assert gmb.random_state == 456


class TestGaussianMixtureBinningFitGetParamsWorkflow:
    """Test parameter handling and sklearn-style workflows."""

    def test_fit_params_immutability(self):
        """Test that parameters don't change during fitting."""

        original_params = {"n_components": 6, "random_state": 789}
        gmb = GaussianMixtureBinning(
            n_components=original_params["n_components"],
            random_state=original_params["random_state"],
        )

        X = np.random.rand(25, 2) * 100
        gmb.fit(X)

        # Parameters should remain unchanged after fitting
        current_params = gmb.get_params()
        for key, value in original_params.items():
            assert current_params[key] == value

    def test_refit_behavior(self):
        """Test that refitting updates the binning edges appropriately."""

        gmb = GaussianMixtureBinning(n_components=3, random_state=42)

        # First fit
        np.random.seed(42)
        X1 = np.random.normal(10, 2, (20, 2))
        gmb.fit(X1)
        edges1 = gmb.bin_edges_

        # Second fit with different data
        np.random.seed(123)
        X2 = np.random.normal(100, 20, (20, 2))
        gmb.fit(X2)
        edges2 = gmb.bin_edges_

        # Edges should be different due to different data scales
        assert not np.allclose(edges1[0], edges2[0])
        assert not np.allclose(edges1[1], edges2[1])

    def test_parameter_immutability_during_use(self):
        """Test that parameters don't change unexpectedly during use."""

        original_params = {"n_components": 4, "random_state": 999}
        gmb = GaussianMixtureBinning(
            n_components=original_params["n_components"],
            random_state=original_params["random_state"],
        )

        X = np.random.rand(25, 2) * 100
        gmb.fit_transform(X)

        # Parameters should remain unchanged
        current_params = gmb.get_params()
        for key, value in original_params.items():
            assert current_params[key] == value
