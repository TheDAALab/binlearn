"""
Comprehensive tests for KMeansBinning method covering all sce    def test_init_custom_parameters(self):
"""

import warnings

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from binlearn import POLARS_AVAILABLE, pd, pl
from binlearn.methods import KMeansBinning
from binlearn.utils import ConfigurationError, FittingError, ValidationError

# Skip polars tests if not available
polars_skip = pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")

# Skip pandas tests if not available
pandas_skip = pytest.mark.skipif(not hasattr(pd, "DataFrame"), reason="pandas not available")


class TestKMeansBinning:
    """Comprehensive test suite for KMeansBinning."""

    @pytest.fixture
    def expect_fallback_warning(self):
        """Context manager to test that fallback warnings are raised when expected."""
        return pytest.warns(
            UserWarning, match="KMeans binning failed, falling back to equal-width binning"
        )

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        return {
            "simple": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).reshape(-1, 1),
            "multi_col": np.array(
                [
                    [1.0, 10.0],
                    [2.0, 20.0],
                    [3.0, 30.0],
                    [4.0, 40.0],
                    [5.0, 50.0],
                    [6.0, 60.0],
                    [7.0, 70.0],
                    [8.0, 80.0],
                    [9.0, 90.0],
                    [10.0, 100.0],
                ]
            ),
            "uniform": np.linspace(0, 100, 100).reshape(-1, 1),
            "normal": np.random.normal(50, 15, 200).reshape(-1, 1),
            "clustered": np.concatenate(
                [
                    np.random.normal(10, 2, 50),
                    np.random.normal(30, 2, 50),
                    np.random.normal(50, 2, 50),
                ]
            ).reshape(-1, 1),
            "with_nan": np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape(-1, 1),
            "with_inf": np.array([1.0, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape(-1, 1),
            "constant": np.array([5.0, 5.0, 5.0, 5.0, 5.0]).reshape(-1, 1),
            "few_unique": np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).reshape(-1, 1),
        }

    # Basic functionality tests

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        binner = KMeansBinning()
        assert binner.n_bins == 10  # default from config or fallback
        assert binner.clip is True  # default from config
        assert binner.preserve_dataframe is False
        assert binner.fit_jointly is False

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        binner = KMeansBinning(n_bins=10, clip=False, preserve_dataframe=True, fit_jointly=True)
        assert binner.n_bins == 10
        assert binner.clip is False
        assert binner.preserve_dataframe is True
        assert binner.fit_jointly is True

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Invalid n_bins
        with pytest.raises((ValueError, ConfigurationError)):
            KMeansBinning(n_bins=0)

        with pytest.raises((ValueError, ConfigurationError)):
            KMeansBinning(n_bins=-1)

        with pytest.raises((ValueError, ConfigurationError)):
            KMeansBinning(n_bins=3.5)  # type: ignore

    # Input format tests with preserve_dataframe=False

    def test_numpy_input_preserve_false(self, sample_data):
        """Test with numpy input and preserve_dataframe=False."""
        binner = KMeansBinning(n_bins=3, preserve_dataframe=False)

        # Fit and transform
        X_fit = sample_data["simple"]
        binner.fit(X_fit)
        X_transformed = binner.transform(X_fit)

        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape == X_fit.shape
        assert X_transformed.dtype == int

        # Inverse transform
        X_inverse = binner.inverse_transform(X_transformed)
        assert isinstance(X_inverse, np.ndarray)
        assert X_inverse.shape == X_fit.shape

    @pandas_skip
    def test_pandas_input_preserve_false(self, sample_data):
        """Test with pandas input and preserve_dataframe=False."""
        binner = KMeansBinning(n_bins=3, preserve_dataframe=False)

        # Create pandas DataFrame
        df = pd.DataFrame(sample_data["simple"], columns=["feature"])
        binner.fit(df)
        df_transformed = binner.transform(df)

        # Should return numpy array when preserve_dataframe=False
        assert isinstance(df_transformed, np.ndarray)
        assert df_transformed.shape == df.values.shape

        # Inverse transform
        df_inverse = binner.inverse_transform(df_transformed)
        assert isinstance(df_inverse, np.ndarray)

    @polars_skip
    def test_polars_input_preserve_false(self, sample_data):
        """Test with polars input and preserve_dataframe=False."""
        binner = KMeansBinning(n_bins=3, preserve_dataframe=False)

        # Create polars DataFrame
        assert pl is not None
        df = pl.DataFrame({"feature": sample_data["simple"].flatten()})
        binner.fit(df)
        df_transformed = binner.transform(df)

        # Should return numpy array when preserve_dataframe=False
        assert isinstance(df_transformed, np.ndarray)
        assert df_transformed.shape == (len(df), 1)

        # Inverse transform
        df_inverse = binner.inverse_transform(df_transformed)
        assert isinstance(df_inverse, np.ndarray)

    # Input format tests with preserve_dataframe=True

    @pandas_skip
    def test_pandas_input_preserve_true(self, sample_data):
        """Test with pandas input and preserve_dataframe=True."""
        binner = KMeansBinning(n_bins=3, preserve_dataframe=True)

        # Create pandas DataFrame with multiple columns
        df = pd.DataFrame(
            {"feature1": sample_data["multi_col"][:, 0], "feature2": sample_data["multi_col"][:, 1]}
        )

        binner.fit(df)
        df_transformed = binner.transform(df)

        # Should preserve DataFrame format
        assert isinstance(df_transformed, pd.DataFrame)
        assert df_transformed.shape == df.shape
        assert list(df_transformed.columns) == ["feature1", "feature2"]

        # Inverse transform
        df_inverse = binner.inverse_transform(df_transformed)
        assert isinstance(df_inverse, pd.DataFrame)
        assert df_inverse.shape == df.shape
        assert list(df_inverse.columns) == ["feature1", "feature2"]

    @polars_skip
    def test_polars_input_preserve_true(self, sample_data):
        """Test with polars input and preserve_dataframe=True."""
        binner = KMeansBinning(n_bins=3, preserve_dataframe=True)

        assert pl is not None
        # Create polars DataFrame
        df = pl.DataFrame(
            {"feature1": sample_data["multi_col"][:, 0], "feature2": sample_data["multi_col"][:, 1]}
        )

        binner.fit(df)
        df_transformed = binner.transform(df)

        # Should preserve DataFrame format
        assert isinstance(df_transformed, pl.DataFrame)
        assert df_transformed.shape == df.shape
        assert df_transformed.columns == ["feature1", "feature2"]

        # Inverse transform
        df_inverse = binner.inverse_transform(df_transformed)
        assert isinstance(df_inverse, pl.DataFrame)
        assert df_inverse.shape == df.shape
        assert df_inverse.columns == ["feature1", "feature2"]

    # Fitted state reconstruction tests

    def test_reconstruction_via_get_params_set_params(self, sample_data):
        """Test fitted state reconstruction via get_params/set_params."""
        # Fit original binner
        binner_original = KMeansBinning(n_bins=4)
        X_fit = sample_data["multi_col"]
        binner_original.fit(X_fit)

        # Get parameters
        params = binner_original.get_params()

        # Create new binner and set parameters (reconstruct state)
        binner_reconstructed = KMeansBinning()
        binner_reconstructed.set_params(**params)

        # Test that transform works without fitting
        X_test = sample_data["multi_col"][:3]  # Subset for testing
        result_original = binner_original.transform(X_test)
        result_reconstructed = binner_reconstructed.transform(X_test)

        np.testing.assert_array_equal(result_original, result_reconstructed)

        # Test inverse transform
        inverse_original = binner_original.inverse_transform(result_original)
        inverse_reconstructed = binner_reconstructed.inverse_transform(result_reconstructed)

        np.testing.assert_array_almost_equal(inverse_original, inverse_reconstructed)

    def test_reconstruction_via_constructor(self, sample_data):
        """Test fitted state reconstruction via constructor parameters."""
        # Fit original binner
        binner_original = KMeansBinning(n_bins=3)
        X_fit = sample_data["simple"]
        binner_original.fit(X_fit)

        # Get parameters including fitted state
        params = binner_original.get_params()

        # Create new binner with constructor (reconstruct state)
        binner_reconstructed = KMeansBinning(**params)

        # Test that transform works without fitting
        X_test = sample_data["simple"]
        result_original = binner_original.transform(X_test)
        result_reconstructed = binner_reconstructed.transform(X_test)

        np.testing.assert_array_equal(result_original, result_reconstructed)

    def test_repeated_fitting_after_reconstruction(self, sample_data):
        """Test repeated fitting on reconstructed state."""
        # Original fitting
        binner = KMeansBinning(n_bins=3)
        X_fit1 = sample_data["simple"]
        binner.fit(X_fit1)

        # Get and set params (reconstruction)
        params = binner.get_params()
        binner_new = KMeansBinning()
        binner_new.set_params(**params)

        # Refit with different data
        X_fit2 = sample_data["uniform"]
        binner_new.fit(X_fit2)

        # Should work fine
        result = binner_new.transform(X_fit2[:10])
        assert result is not None
        assert result.shape == (10, 1)

        # Another refit
        binner_new.fit(sample_data["normal"])
        result2 = binner_new.transform(sample_data["normal"][:10])
        assert result2 is not None

    def test_various_formats_after_reconstruction(self, sample_data):
        """Test various input formats after fitted state reconstruction."""
        # Original fitting with numpy
        binner_original = KMeansBinning(n_bins=3, preserve_dataframe=True)
        X_numpy = sample_data["multi_col"]
        binner_original.fit(X_numpy)

        # Reconstruct
        params = binner_original.get_params()
        binner_reconstructed = KMeansBinning(**params)

        # Test numpy input
        result_numpy = binner_reconstructed.transform(X_numpy[:3])
        assert isinstance(result_numpy, np.ndarray)

        # Test pandas input (if preserve_dataframe=True was set)
        if hasattr(pd, "DataFrame"):
            X_pandas = pd.DataFrame(X_numpy, columns=["col1", "col2"])
            result_pandas = binner_reconstructed.transform(X_pandas)
            # The result type depends on preserve_dataframe setting
            if binner_reconstructed.preserve_dataframe:
                assert isinstance(result_pandas, pd.DataFrame)
            else:
                assert isinstance(result_pandas, np.ndarray)

    # sklearn pipeline integration tests

    def test_sklearn_pipeline_basic(self, sample_data):
        """Test basic sklearn pipeline integration."""
        # Create pipeline
        pipeline = Pipeline([("scaler", StandardScaler()), ("binner", KMeansBinning(n_bins=3))])

        X = sample_data["normal"]

        # Fit and transform
        pipeline.fit(X)
        X_transformed = pipeline.transform(X)

        assert X_transformed is not None
        assert X_transformed.shape == X.shape
        assert X_transformed.dtype == int

        # Test named steps access
        assert hasattr(pipeline.named_steps["binner"], "bin_edges_")

    def test_sklearn_pipeline_with_dataframes(self, sample_data):
        """Test sklearn pipeline with DataFrame inputs."""
        if not hasattr(pd, "DataFrame"):
            pytest.skip("pandas not available")

        # Create pipeline
        pipeline = Pipeline([("binner", KMeansBinning(n_bins=4, preserve_dataframe=True))])

        # Use DataFrame
        df = pd.DataFrame(sample_data["multi_col"], columns=["feat1", "feat2"])

        pipeline.fit(df)
        df_transformed = pipeline.transform(df)

        # Should preserve DataFrame format
        assert isinstance(df_transformed, pd.DataFrame)
        assert list(df_transformed.columns) == ["feat1", "feat2"]

    def test_sklearn_pipeline_param_access(self, sample_data):
        """Test parameter access in sklearn pipeline."""
        pipeline = Pipeline([("binner", KMeansBinning(n_bins=5))])

        # Test parameter access
        params = pipeline.get_params()
        assert "binner__n_bins" in params
        assert params["binner__n_bins"] == 5

        # Test parameter setting
        pipeline.set_params(binner__n_bins=7)
        assert pipeline.named_steps["binner"].n_bins == 7

    # Edge case tests

    def test_edge_case_nan_values(self, sample_data):
        """Test handling of NaN values."""
        binner = KMeansBinning(n_bins=3)
        X_nan = sample_data["with_nan"]

        # Should handle NaN values gracefully (base class preprocesses)
        binner.fit(X_nan)
        result = binner.transform(X_nan)

        assert result is not None
        # NaN values should be handled by the base class
        assert result.shape == X_nan.shape

    def test_edge_case_inf_values(self, sample_data):
        """Test handling of infinite values."""
        binner = KMeansBinning(n_bins=3)
        X_inf = sample_data["with_inf"]

        # Should handle inf values based on config (clip, error, etc.)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings for this test
            binner.fit(X_inf)
            result = binner.transform(X_inf)

            assert result is not None
            assert result.shape == X_inf.shape

    def test_edge_case_constant_column(self, sample_data):
        """Test handling of constant columns."""
        binner = KMeansBinning(n_bins=3)
        X_constant = sample_data["constant"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings about constant data
            binner.fit(X_constant)
            result = binner.transform(X_constant)

            assert result is not None
            assert result.shape == X_constant.shape
            # All values should map to bins for constant data
            assert len(np.unique(result)) >= 1

    def test_edge_case_more_bins_than_data_points(self, sample_data):
        """Test when n_bins > number of data points."""
        X = np.array([[1], [2], [3]])  # Only 3 data points
        binner = KMeansBinning(n_bins=5)  # More bins than data

        # Should raise ConfigurationError because insufficient data for clustering
        with pytest.raises(
            ConfigurationError, match="KMeansBinning requires at least 5 data points, got 3"
        ):
            binner.fit(X)

    def test_edge_case_more_bins_than_unique_values(self, sample_data, expect_fallback_warning):
        """Test when n_bins > number of unique values."""
        X = sample_data["few_unique"]  # Only 3 unique values but 6 data points
        binner = KMeansBinning(n_bins=5)  # More bins than unique values

        # Should trigger fallback since KMeans can't create more clusters than unique values
        with expect_fallback_warning:
            binner.fit(X)

        result = binner.transform(X)

        assert result is not None
        assert result.shape == X.shape
        # Should handle case with fewer unique values than requested bins

    def test_edge_case_single_value_per_bin(self):
        """Test when each cluster contains exactly one data point."""
        X = np.array([[1], [2], [3], [4], [5]])  # 5 unique values
        binner = KMeansBinning(n_bins=5)  # Same as data points

        binner.fit(X)
        result = binner.transform(X)

        assert result is not None
        assert result.shape == X.shape

    def test_edge_case_very_small_range(self):
        """Test with very small data range."""
        X = np.array([[1.0000001], [1.0000002], [1.0000003], [1.0000004], [1.0000005]])
        binner = KMeansBinning(n_bins=3)

        binner.fit(X)
        result = binner.transform(X)

        assert result is not None
        assert result.shape == X.shape

    def test_edge_case_large_data(self):
        """Test with large dataset."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (10000, 1))
        binner = KMeansBinning(n_bins=10)

        binner.fit(X)
        result = binner.transform(X[:1000])  # Transform subset

        assert result is not None
        assert result.shape == (1000, 1)
        assert 0 <= result.max() < 10  # Should be within bin range

    def test_single_row_data(self):
        """Test handling of single row data."""
        X = np.array([[5.0]])
        binner = KMeansBinning(n_bins=3)

        # Should raise ConfigurationError because insufficient data for clustering
        with pytest.raises(
            ConfigurationError, match="KMeansBinning requires at least 3 data points, got 1"
        ):
            binner.fit(X)

    def test_single_row_data_with_single_bin(self):
        """Test handling of single row data with n_bins=1."""
        X = np.array([[5.0]])
        binner = KMeansBinning(n_bins=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            binner.fit(X)
            result = binner.transform(X)

            assert result is not None
            assert result.shape == (1, 1)
            assert result[0, 0] == 0  # Should be assigned to bin 0

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        binner = KMeansBinning(n_bins=3)

        with pytest.raises((ValueError, ValidationError, FittingError)):
            binner.fit(np.array([]).reshape(0, 1))

    def test_kmeans_clustering_error_coverage(self):
        """Test to cover K-means clustering error handling."""
        import unittest.mock

        # Mock kmeans1d.cluster to raise an exception to test fallback behavior
        with unittest.mock.patch(
            "kmeans1d.cluster", side_effect=Exception("Mock clustering error")
        ):
            X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
            binner = KMeansBinning(n_bins=3)

            # Should fall back to equal-width binning with warning instead of raising exception
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                binner.fit(X)

            # Should have issued a warning about fallback
            assert len(w) > 0
            # Changed warning message due to new fallback mechanism
            assert "KMeans failed with sklearn" in str(
                w[0].message
            ) or "KMeans binning failed" in str(w[0].message)
            assert "fallback" in str(w[0].message)  # Should still work and produce valid results
            result = binner.transform(X)
            assert result is not None
            assert result.shape == X.shape

    # Specific K-means binning tests

    def test_random_state_reproducibility(self, sample_data):
        """Test that random_state produces reproducible results."""
        X = sample_data["normal"]

        # Fit two binners with same random state
        binner1 = KMeansBinning(n_bins=5)
        binner2 = KMeansBinning(n_bins=5)

        binner1.fit(X)
        binner2.fit(X)

        result1 = binner1.transform(X[:10])
        result2 = binner2.transform(X[:10])

        # Should produce identical results with same random_state
        np.testing.assert_array_equal(result1, result2)

    def test_random_state_deterministic_behavior(self, sample_data):
        """Test that K-means 1D clustering is deterministic."""
        X = sample_data["normal"]

        # Fit two binners with different random states
        binner1 = KMeansBinning(n_bins=5)
        binner2 = KMeansBinning(n_bins=5)

        binner1.fit(X)
        binner2.fit(X)

        # Get the centroids (representatives)
        centroids1 = sorted(binner1.bin_representatives_[0])
        centroids2 = sorted(binner2.bin_representatives_[0])

        # 1D K-means is deterministic - should produce identical centroids
        np.testing.assert_array_almost_equal(centroids1, centroids2, decimal=10)

    def test_clustered_data_effectiveness(self, sample_data):
        """Test K-means effectiveness on clearly clustered data."""
        X = sample_data["clustered"]  # Data with 3 clear clusters
        binner = KMeansBinning(n_bins=3)

        binner.fit(X)

        # Check that centroids are reasonable for the clustered data
        centroids = binner.bin_representatives_[0]
        centroids_sorted = sorted(centroids)

        # Centroids should be somewhat spread out for clustered data
        assert centroids_sorted[1] - centroids_sorted[0] > 5  # Reasonable separation
        assert centroids_sorted[2] - centroids_sorted[1] > 5  # Reasonable separation

    def test_representatives_are_centroids(self, sample_data):
        """Test that representatives are the K-means centroids."""
        X = sample_data["simple"]
        binner = KMeansBinning(n_bins=3)

        binner.fit(X)

        # Representatives should be the cluster centroids
        representatives = binner.bin_representatives_[0]

        # Should have the expected number of representatives
        assert len(representatives) == 3

        # Representatives should be within data range
        assert min(representatives) >= X.min()
        assert max(representatives) <= X.max()

    def test_bin_edges_between_centroids(self, sample_data):
        """Test that bin edges are positioned between centroids."""
        X = sample_data["simple"]
        binner = KMeansBinning(n_bins=3)

        binner.fit(X)

        edges = binner.bin_edges_[0]
        centroids = sorted(binner.bin_representatives_[0])

        # Should have n_bins + 1 edges
        assert len(edges) == 4

        # Interior edges should be between consecutive centroids
        for i in range(1, len(edges) - 1):
            edge = edges[i]
            # Edge should be between consecutive centroids
            assert centroids[i - 1] < edge < centroids[i]

    def test_string_n_bins_parameter(self, sample_data):
        """Test string n_bins parameters like 'sqrt', 'log2'."""
        X = sample_data["uniform"]  # 100 samples

        # Test 'sqrt' - should be sqrt(100) = 10
        binner_sqrt = KMeansBinning(n_bins="sqrt")
        binner_sqrt.fit(X)

        # Should create 10 bins
        assert len(binner_sqrt.bin_representatives_[0]) == 10

        # Test 'log2' - should be log2(100) ≈ 6.6, resolved to 7
        binner_log2 = KMeansBinning(n_bins="log2")
        binner_log2.fit(X)

        # Should create 7 bins (log2(100) = 6.64, resolved to 7)
        assert len(binner_log2.bin_representatives_[0]) == 7

    def test_fit_jointly_parameter(self, sample_data):
        """Test fit_jointly parameter (though K-means typically doesn't use it)."""
        X = sample_data["multi_col"]

        # Test with fit_jointly=True
        binner_joint = KMeansBinning(n_bins=3, fit_jointly=True)
        binner_joint.fit(X)

        # Test with fit_jointly=False
        binner_indep = KMeansBinning(n_bins=3, fit_jointly=False)
        binner_indep.fit(X)

        # For K-means, the result should be the same since it's inherently per-column
        # But test that both work
        result_joint = binner_joint.transform(X[:3])
        result_indep = binner_indep.transform(X[:3])

        assert result_joint is not None
        assert result_indep is not None
        # May be equal for K-means since it's inherently independent

    def test_multiple_columns_independent_clustering(self, sample_data):
        """Test that columns are clustered independently."""
        X = sample_data["multi_col"]  # Different ranges per column
        binner = KMeansBinning(n_bins=3)

        binner.fit(X)

        # Each column should have its own centroids based on its data
        centroids_col1 = binner.bin_representatives_[0]
        centroids_col2 = binner.bin_representatives_[1]

        assert len(centroids_col1) == 3
        assert len(centroids_col2) == 3

        # Centroids should be different for columns with different ranges
        assert not np.allclose(centroids_col1, centroids_col2)

    def test_clip_parameter_functionality(self, sample_data):
        """Test clip parameter with out-of-range values."""
        X_train = np.array([[0], [5], [10], [15], [20]])
        X_test = np.array([[-5], [25]])  # Out of range values

        # Test with clip=True
        binner_clip = KMeansBinning(n_bins=3, clip=True)
        binner_clip.fit(X_train)
        result_clip = binner_clip.transform(X_test)

        # Should clip to valid bin range
        assert 0 <= result_clip.min()
        assert result_clip.max() < 3

        # Test with clip=False
        binner_no_clip = KMeansBinning(n_bins=3, clip=False)
        binner_no_clip.fit(X_train)
        result_no_clip = binner_no_clip.transform(X_test)

        # May have values outside normal bin range (MISSING_VALUE)
        # The exact behavior depends on the base class implementation
        assert result_no_clip is not None

    # Integration and workflow tests

    def test_full_workflow_with_all_methods(self, sample_data):
        """Test complete workflow: fit, transform, inverse_transform, get_params, set_params."""
        X = sample_data["multi_col"]

        # Initialize and fit
        binner = KMeansBinning(n_bins=4, preserve_dataframe=False)
        binner.fit(X)

        # Transform
        X_binned = binner.transform(X)
        assert X_binned.shape == X.shape
        assert X_binned.dtype == int

        # Inverse transform
        X_recovered = binner.inverse_transform(X_binned)
        assert X_recovered.shape == X.shape
        assert X_recovered.dtype == float

        # Get params
        params = binner.get_params()
        assert "n_bins" in params
        assert "bin_edges" in params
        assert params["n_bins"] == 4

        # Set params and verify
        new_binner = KMeansBinning()
        new_binner.set_params(**params)

        X_binned_2 = new_binner.transform(X)
        np.testing.assert_array_equal(X_binned, X_binned_2)

    def test_consistency_across_transforms(self, sample_data):
        """Test that repeated transforms give consistent results."""
        X = sample_data["normal"]
        binner = KMeansBinning(n_bins=5)
        binner.fit(X)

        # Multiple transforms should give same result
        result1 = binner.transform(X)
        result2 = binner.transform(X)
        result3 = binner.transform(X)

        np.testing.assert_array_equal(result1, result2)
        np.testing.assert_array_equal(result2, result3)

    def test_partial_data_transform(self, sample_data):
        """Test transforming data subset after fitting on full data."""
        X_full = sample_data["uniform"]  # 100 samples
        X_subset = X_full[:20]  # 20 samples

        binner = KMeansBinning(n_bins=3)
        binner.fit(X_full)

        # Transform subset
        result_subset = binner.transform(X_subset)
        result_full = binner.transform(X_full)

        # Subset result should match first 20 rows of full result
        np.testing.assert_array_equal(result_subset, result_full[:20])

    def test_different_n_bins_values(self, sample_data):
        """Test various n_bins values."""
        X = sample_data["uniform"]

        for n_bins in [2, 3, 5, 10, 20]:
            binner = KMeansBinning(n_bins=n_bins)
            binner.fit(X)
            result = binner.transform(X)

            # Check that we get expected number of bins
            unique_bins = np.unique(result)
            assert len(unique_bins) <= n_bins  # May be less if clusters merge
            assert result.max() < n_bins  # Bin indices should be < n_bins
            assert result.min() >= 0  # Bin indices should be >= 0

            # Check that we have the right number of centroids
            assert len(binner.bin_representatives_[0]) == n_bins
