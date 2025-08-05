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

    def test_init_default(self) -> None:
        """Test initialization with default parameters.

        Verifies that the transformer initializes correctly with default
        parameter values and that all attributes are set as expected.
        """

        gmb = GaussianMixtureBinning()
        assert gmb.n_components == 10
        assert gmb.random_state is None

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters.

        Verifies that the transformer correctly stores custom initialization
        parameter values including n_components, random_state, and fit_jointly options.
        """

        gmb = GaussianMixtureBinning(n_components=5, random_state=42, fit_jointly=True)
        assert gmb.n_components == 5
        assert gmb.random_state == 42

    def test_repr(self) -> None:
        """Test string representation of the transformer."""

        gmb = GaussianMixtureBinning(n_components=5, random_state=42)
        repr_str = repr(gmb)
        assert "GaussianMixtureBinning" in repr_str
        assert "n_components=5" in repr_str
        assert "random_state=42" in repr_str

    def test_validate_params_invalid_n_components(self) -> None:
        """Test parameter validation with invalid n_components values.

        Verifies that the validator correctly rejects non-positive n_components.
        """

        with pytest.raises(ConfigurationError, match="n_components must be a positive integer"):
            GaussianMixtureBinning(n_components=0)

        with pytest.raises(ConfigurationError, match="n_components must be a positive integer"):
            GaussianMixtureBinning(n_components=-1)

    def test_validate_params_invalid_random_state(self) -> None:
        """Test parameter validation with invalid random_state values.

        Verifies that the validator correctly rejects invalid random_state
        values (negative integers).
        """

        with pytest.raises(ConfigurationError, match="random_state must be a non-negative integer"):
            GaussianMixtureBinning(random_state=-1)

    def test_fit_transform_basic(self) -> None:
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

    def test_fit_transform_with_random_state(self) -> None:
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

    def test_separate_fit_transform(self) -> None:
        """Test separate fit and transform calls."""

        X = np.random.rand(20, 3) * 100
        gmb = GaussianMixtureBinning(n_components=4, random_state=42)

        # Fit and transform separately
        gmb.fit(X)
        X_binned = gmb.transform(X)

        assert X_binned.shape == X.shape
        assert hasattr(gmb, "bin_edges_")

    def test_all_nan_column(self) -> None:
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

    def test_insufficient_data_for_components(self) -> None:
        """Test error handling with insufficient data for components."""

        # Only 2 data points but requesting 5 components
        X = np.array([[1.0], [2.0]])
        gmb = GaussianMixtureBinning(n_components=5)

        # Should raise ValueError for insufficient data
        with pytest.raises(ValueError, match="Insufficient non-NaN values"):
            gmb.fit(X)

    def test_fit_jointly_vs_per_column(self) -> None:
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

    def test_direct_calculate_bins_basic(self) -> None:
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

    def test_empty_data(self) -> None:
        """Test behavior with empty data arrays."""

        X = np.array([]).reshape(0, 2)
        gmb = GaussianMixtureBinning(n_components=3)

        # Empty data should be handled gracefully, not raise an error
        # Should emit warnings for both empty columns
        with pytest.warns(DataQualityWarning, match="Data in column.*contains only NaN values"):
            gmb.fit(X)
        X_binned = gmb.transform(X)
        assert X_binned.shape == (0, 2)

    def test_edge_case_duplicate_values(self) -> None:
        """Test handling of data with many duplicate values."""

        # Data with many duplicates
        X = np.array([[1, 1, 1, 1, 1, 2, 2, 3]]).T
        gmb = GaussianMixtureBinning(n_components=3, random_state=42)

        # Should handle duplicates gracefully
        X_binned = gmb.fit_transform(X)
        assert X_binned.shape == X.shape

    def test_edge_case_constant_data(self) -> None:
        """Test handling of constant data (all values the same)."""

        # All values are the same
        X = np.array([[5, 5, 5, 5, 5]]).T
        gmb = GaussianMixtureBinning(n_components=3, random_state=42)

        # Should handle constant data gracefully
        X_binned = gmb.fit_transform(X)
        assert X_binned.shape == X.shape

    def test_edge_case_insufficient_unique_values(self) -> None:
        """Test handling when there are fewer unique values than desired components."""

        # Only 2 unique values but requesting 5 components
        X = np.array([[1, 1, 1, 2, 2, 2]]).T
        gmb = GaussianMixtureBinning(n_components=5, random_state=42)

        # Should handle gracefully by falling back to unique value bins
        X_binned = gmb.fit_transform(X)
        assert X_binned.shape == X.shape

    def test_gaussian_mixture_error_handling(self) -> None:
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

    def test_direct_calculate_bins_invalid_n_components(self) -> None:
        """Test _calculate_bins with invalid n_components."""

        gmb = GaussianMixtureBinning(n_components=1)  # Start with valid n_components
        gmb.n_components = 0  # Set directly to bypass init validation
        data = np.array([1, 2, 3, 4, 5])

        with pytest.raises(ValueError, match="n_components must be >= 1"):
            gmb._calculate_bins(data, col_id=0)

    def test_gmm_clustering_exception_handling(self) -> None:
        """Test exception handling in GMM clustering with problematic data."""

        gmb = GaussianMixtureBinning(n_components=2)

        # Create data with enough unique values to reach GMM clustering
        # but then have the GMM fail
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 5 unique values > 2 components

        # Mock GaussianMixture to raise an exception
        from unittest.mock import MagicMock, patch

        with patch("binlearn.methods._gaussian_mixture_binning.GaussianMixture") as mock_gmm_class:
            # Create a mock instance that raises exception on fit
            mock_gmm_instance = MagicMock()
            mock_gmm_instance.fit.side_effect = Exception("Convergence failed")
            mock_gmm_class.return_value = mock_gmm_instance

            # Should handle the exception and raise ValueError with context
            with pytest.raises(
                ValueError, match="Column 0: Error in GMM clustering: Convergence failed"
            ):
                gmb._create_gmm_bins(data, col_id=0, n_components=2)

    def test_single_cluster_edge_extension(self) -> None:
        """Test edge extension logic for single cluster case."""

        gmb = GaussianMixtureBinning(n_components=1, random_state=42)

        # Create data with a single tight cluster that will result in a single mean
        # that is at the data_max position to trigger lines 298-299
        data = np.array([10.0, 10.0, 10.0, 10.0, 10.1])  # Very tight cluster with max at end

        edges, reps = gmb._create_gmm_bins(data, col_id=0, n_components=1)

        # Should have proper edge extension
        assert len(edges) == 2
        assert len(reps) == 1
        assert edges[0] < reps[0] < edges[1]

    def test_single_component_with_extension_beyond_max(self) -> None:
        """Test single component where mean is beyond data max - triggers lines 298-299."""

        gmb = GaussianMixtureBinning(n_components=1, random_state=42)

        # Create carefully chosen data to test the edge extension case
        # where the mean might be near the data max
        data = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # Outlier to influence mean position

        edges, reps = gmb._create_gmm_bins(data, col_id=0, n_components=1)

        # Should have proper edge extension - test both cases
        assert len(edges) == 2
        assert len(reps) == 1
        # The algorithm should handle edge extension properly
        assert edges[0] < edges[1]

    def test_gmm_edge_extension_beyond_data_max(self) -> None:
        """Test GMM edge extension when mean is at or beyond data max."""

        gmb = GaussianMixtureBinning(n_components=2, random_state=42)

        # Create data where GMM means will be at the extremes
        # This should trigger the edge extension logic
        data = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])

        edges, reps = gmb._create_gmm_bins(data, col_id=0, n_components=2)

        # Should have proper edge extension
        assert len(edges) == 3  # n_components + 1
        assert len(reps) == 2  # n_components
        # Verify edges are properly ordered
        assert edges[0] < edges[1] < edges[2]

    def test_calculate_gmm_last_edge_data_max_case(self) -> None:
        """Test _calculate_gmm_last_edge when last mean < data_max."""

        gmb = GaussianMixtureBinning(n_components=2)

        # Test case where last mean is less than data max
        means = [2.0, 5.0]
        data_max = 10.0  # Much larger than last mean

        last_edge = gmb._calculate_gmm_last_edge(means, data_max)
        assert last_edge == data_max

    def test_calculate_gmm_last_edge_extension_case(self) -> None:
        """Test _calculate_gmm_last_edge when last mean >= data_max."""

        gmb = GaussianMixtureBinning(n_components=2)

        # Test case where last mean >= data_max
        means = [2.0, 10.0]
        data_max = 10.0  # Equal to last mean

        last_edge = gmb._calculate_gmm_last_edge(means, data_max)
        expected = 10.0 + (10.0 - 2.0) * 0.05  # 10.0 + 0.4 = 10.4
        assert last_edge == expected


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
class TestGaussianMixtureBinningDataTypes:
    """Test GaussianMixtureBinning with different data types."""

    def test_pandas_dataframe(self) -> None:
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
    def test_polars_dataframe(self) -> None:
        """Test with polars DataFrame input and output."""

        np.random.seed(42)
        df = pl.DataFrame(  # type: ignore[union-attr]
            {"feature1": np.random.normal(5, 2, 20), "feature2": np.random.normal(15, 3, 20)}
        )

        gmb = GaussianMixtureBinning(n_components=2, random_state=42, preserve_dataframe=True)
        df_binned = gmb.fit(df).transform(df)

        assert isinstance(df_binned, pl.DataFrame)  # type: ignore[union-attr]
        assert df_binned.shape == df.shape


class TestGaussianMixtureBinningSklearnIntegration:
    """Test sklearn compatibility and integration."""

    def test_sklearn_pipeline(self) -> None:
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

    def test_sklearn_column_transformer(self) -> None:
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

    def test_sklearn_clone(self) -> None:
        """Test sklearn clone functionality."""

        original = GaussianMixtureBinning(n_components=5, random_state=42)
        cloned = clone(original)

        assert cloned.n_components == original.n_components
        assert cloned.random_state == original.random_state
        assert cloned is not original

    def test_get_params(self) -> None:
        """Test get_params method for sklearn compatibility."""

        gmb = GaussianMixtureBinning(n_components=7, random_state=123)
        params = gmb.get_params()

        assert params["n_components"] == 7
        assert params["random_state"] == 123

    def test_set_params(self) -> None:
        """Test set_params method for sklearn compatibility."""

        gmb = GaussianMixtureBinning()
        gmb.set_params(n_components=8, random_state=456)

        assert gmb.n_components == 8
        assert gmb.random_state == 456


class TestGaussianMixtureBinningFitGetParamsWorkflow:
    """Test parameter handling and sklearn-style workflows."""

    def test_fit_params_immutability(self) -> None:
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

    def test_refit_behavior(self) -> None:
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

    def test_parameter_immutability_during_use(self) -> None:
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

    def test_string_n_components_parameter_support(self):
        """Test support for string n_components parameters like 'sqrt', 'log', etc."""
        X = np.random.rand(100, 2) * 100

        # Test sqrt specification
        gmb_sqrt = GaussianMixtureBinning(n_components="sqrt")
        result_sqrt = gmb_sqrt.fit_transform(X)
        assert result_sqrt is not None

        # Test log specification
        gmb_log = GaussianMixtureBinning(n_components="log")
        result_log = gmb_log.fit_transform(X)
        assert result_log is not None

        # Test case insensitive
        gmb_upper = GaussianMixtureBinning(n_components="SQRT")
        result_upper = gmb_upper.fit_transform(X)
        assert result_upper is not None

    def test_edge_case_validation_paths(self):
        """Test specific validation edge cases for coverage."""
        # Test that normal operation works
        X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]).T
        gmb = GaussianMixtureBinning(n_components=2)
        result = gmb.fit_transform(X)
        assert result is not None

    def test_resolved_n_components_validation_edge_case(self):
        """Test the resolved_n_components < 1 validation path (line 189)."""
        from unittest.mock import patch

        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 1D array

        gmb = GaussianMixtureBinning(n_components=2)

        # Mock the functions as imported in the gaussian_mixture_binning module
        with (
            patch(
                "binlearn.methods._gaussian_mixture_binning.validate_bin_number_for_calculation"
            ) as mock_validate,
            patch(
                "binlearn.methods._gaussian_mixture_binning.resolve_n_bins_parameter",
                return_value=0,
            ),
        ):

            # Make validate_bin_number_for_calculation do nothing (just pass)
            mock_validate.return_value = None

            with pytest.raises(ValueError, match="n_components must be >= 1, got 0"):
                gmb._calculate_bins(X, col_id="test_col")
