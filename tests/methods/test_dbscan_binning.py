"""
Comprehensive tests for DBSCANBinning method covering all scenarios.
"""

import warnings

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from binlearn import POLARS_AVAILABLE, pd, pl
from binlearn.methods import DBSCANBinning
from binlearn.utils import ConfigurationError, FittingError, ValidationError

# Skip polars tests if not available
polars_skip = pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")

# Skip pandas tests if not available
pandas_skip = pytest.mark.skipif(not hasattr(pd, "DataFrame"), reason="pandas not available")


class TestDBSCANBinning:
    """Comprehensive test suite for DBSCANBinning."""

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
                    np.random.normal(10, 1, 50),  # Tight cluster 1
                    np.random.normal(30, 1, 50),  # Tight cluster 2
                    np.random.normal(60, 1, 50),  # Tight cluster 3
                ]
            ).reshape(-1, 1),
            "with_nan": np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape(-1, 1),
            "with_inf": np.array([1.0, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape(-1, 1),
            "constant": np.array([5.0, 5.0, 5.0, 5.0, 5.0]).reshape(-1, 1),
            "few_unique": np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).reshape(-1, 1),
            "sparse": np.array([1.0, 5.0, 10.0, 50.0, 100.0]).reshape(-1, 1),  # Sparse data
        }

    @pytest.fixture
    def expect_fallback_warning(self):
        """Context manager to test that fallback warnings are raised when expected."""
        return pytest.warns(
            UserWarning, match="DBSCAN binning failed, falling back to equal-width binning"
        )

    # Basic functionality tests

    def test_init_default_parameters(self):
        """Test DBSCANBinning initialization with default parameters."""
        binner = DBSCANBinning()
        assert binner.eps == 0.5
        assert binner.min_samples == 5
        assert binner.min_bins == 2

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        binner = DBSCANBinning(
            eps=0.5,
            min_samples=3,
            min_bins=4,
            clip=False,
            preserve_dataframe=True,
            fit_jointly=True,
        )
        assert binner.eps == 0.5
        assert binner.min_samples == 3
        assert binner.min_bins == 4
        assert binner.clip is False
        assert binner.preserve_dataframe is True
        assert binner.fit_jointly is True

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Invalid eps
        with pytest.raises((ValueError, ConfigurationError)):
            DBSCANBinning(eps=0)

        with pytest.raises((ValueError, ConfigurationError)):
            DBSCANBinning(eps=-0.1)

        with pytest.raises((ValueError, ConfigurationError)):
            DBSCANBinning(eps="invalid")  # type: ignore

        # Invalid min_samples
        with pytest.raises((ValueError, ConfigurationError)):
            DBSCANBinning(min_samples=0)

        with pytest.raises((ValueError, ConfigurationError)):
            DBSCANBinning(min_samples=-1)

        with pytest.raises((ValueError, ConfigurationError)):
            DBSCANBinning(min_samples=3.5)  # type: ignore

        # Invalid min_bins
        with pytest.raises((ValueError, ConfigurationError)):
            DBSCANBinning(min_bins=0)

        with pytest.raises((ValueError, ConfigurationError)):
            DBSCANBinning(min_bins=-1)

        with pytest.raises((ValueError, ConfigurationError)):
            DBSCANBinning(min_bins=3.5)  # type: ignore

    # Input format tests with preserve_dataframe=False

    def test_numpy_input_preserve_false(self, sample_data, expect_fallback_warning):
        """Test with numpy input and preserve_dataframe=False."""
        binner = DBSCANBinning(eps=2.0, min_samples=3, preserve_dataframe=False)

        # Fit and transform
        X_fit = sample_data["simple"]
        with expect_fallback_warning:
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
    def test_pandas_input_preserve_false(self, sample_data, expect_fallback_warning):
        """Test with pandas input and preserve_dataframe=False."""
        binner = DBSCANBinning(eps=2.0, min_samples=3, preserve_dataframe=False)

        # Create pandas DataFrame
        df = pd.DataFrame(sample_data["simple"], columns=["feature"])

        # This should trigger fallback since eps=2.0 may not find proper clusters in simple data
        with expect_fallback_warning:
            binner.fit(df)

        df_transformed = binner.transform(df)

        # Should return numpy array when preserve_dataframe=False
        assert isinstance(df_transformed, np.ndarray)
        assert df_transformed.shape == df.values.shape

        # Inverse transform
        df_inverse = binner.inverse_transform(df_transformed)
        assert isinstance(df_inverse, np.ndarray)

    @polars_skip
    def test_polars_input_preserve_false(self, sample_data, expect_fallback_warning):
        """Test with polars input and preserve_dataframe=False."""
        binner = DBSCANBinning(eps=2.0, min_samples=3, preserve_dataframe=False)

        # Create polars DataFrame
        assert pl is not None
        df = pl.DataFrame({"feature": sample_data["simple"].flatten()})

        # This should trigger fallback since eps=2.0 may not find proper clusters in simple data
        with expect_fallback_warning:
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
    def test_pandas_input_preserve_true(self, sample_data, expect_fallback_warning):
        """Test with pandas input and preserve_dataframe=True."""
        binner = DBSCANBinning(eps=10.0, min_samples=3, preserve_dataframe=True)

        # Create pandas DataFrame with multiple columns
        df = pd.DataFrame(
            {"feature1": sample_data["multi_col"][:, 0], "feature2": sample_data["multi_col"][:, 1]}
        )

        with expect_fallback_warning:
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
    def test_polars_input_preserve_true(self, sample_data, expect_fallback_warning):
        """Test with polars input and preserve_dataframe=True."""
        binner = DBSCANBinning(eps=10.0, min_samples=3, preserve_dataframe=True)

        assert pl is not None
        # Create polars DataFrame
        df = pl.DataFrame(
            {"feature1": sample_data["multi_col"][:, 0], "feature2": sample_data["multi_col"][:, 1]}
        )

        with expect_fallback_warning:
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

    def test_reconstruction_via_get_params_set_params(self, sample_data, expect_fallback_warning):
        """Test fitted state reconstruction via get_params/set_params."""
        # Fit original binner
        binner_original = DBSCANBinning(eps=10.0, min_samples=3)
        X_fit = sample_data["multi_col"]
        with expect_fallback_warning:
            binner_original.fit(X_fit)

        # Get parameters
        params = binner_original.get_params()

        # Create new binner and set parameters (reconstruct state)
        binner_reconstructed = DBSCANBinning()
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

    def test_reconstruction_via_constructor(self, sample_data, expect_fallback_warning):
        """Test fitted state reconstruction via constructor parameters."""
        # Fit original binner
        binner_original = DBSCANBinning(eps=2.0, min_samples=3)
        X_fit = sample_data["simple"]

        # Should trigger fallback since eps=2.0 may not find proper clusters in simple data
        with expect_fallback_warning:
            binner_original.fit(X_fit)

        # Get parameters including fitted state
        params = binner_original.get_params()

        # Create new binner with constructor (reconstruct state)
        binner_reconstructed = DBSCANBinning(**params)

        # Test that transform works without fitting
        X_test = sample_data["simple"]
        result_original = binner_original.transform(X_test)
        result_reconstructed = binner_reconstructed.transform(X_test)

        np.testing.assert_array_equal(result_original, result_reconstructed)

        np.testing.assert_array_equal(result_original, result_reconstructed)

    def test_repeated_fitting_after_reconstruction(self, sample_data, expect_fallback_warning):
        """Test repeated fitting on reconstructed state."""
        # Original fitting
        binner = DBSCANBinning(eps=2.0, min_samples=3)
        X_fit1 = sample_data["simple"]

        # Should trigger fallback since eps=2.0 may not find proper clusters in simple data
        with expect_fallback_warning:
            binner.fit(X_fit1)

        # Get and set params (reconstruction)
        params = binner.get_params()
        binner_new = DBSCANBinning()
        binner_new.set_params(**params)

        # Refit with different data - uniform data might also trigger fallback with these parameters
        X_fit2 = sample_data["uniform"]
        with expect_fallback_warning:
            binner_new.fit(X_fit2)

        # Should work fine
        result = binner_new.transform(X_fit2[:10])
        assert result is not None
        assert result.shape == (10, 1)

        # Another refit with normal data - this should work without warnings
        binner_new.fit(sample_data["normal"])
        result2 = binner_new.transform(sample_data["normal"][:10])
        assert result2 is not None

    def test_various_formats_after_reconstruction(self, sample_data, expect_fallback_warning):
        """Test various input formats after fitted state reconstruction."""
        # Original fitting with numpy
        binner_original = DBSCANBinning(eps=10.0, min_samples=3, preserve_dataframe=True)
        X_numpy = sample_data["multi_col"]

        with expect_fallback_warning:
            binner_original.fit(X_numpy)

        # Reconstruct
        params = binner_original.get_params()
        binner_reconstructed = DBSCANBinning(**params)

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

    def test_sklearn_pipeline_basic(self, sample_data, expect_fallback_warning):
        """Test basic sklearn pipeline integration."""
        # Create pipeline
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("binner", DBSCANBinning(eps=0.5, min_samples=3))]
        )

        X = sample_data["normal"]

        # Should trigger fallback since eps=0.5 on scaled normal data might not find proper clusters
        with expect_fallback_warning:
            pipeline.fit(X)

        X_transformed = pipeline.transform(X)

        assert X_transformed is not None
        assert X_transformed.shape == X.shape
        assert X_transformed.dtype == int

        # Test named steps access
        assert hasattr(pipeline.named_steps["binner"], "bin_edges_")

    def test_sklearn_pipeline_with_dataframes(self, sample_data, expect_fallback_warning):
        """Test sklearn pipeline with DataFrame inputs."""
        if not hasattr(pd, "DataFrame"):
            pytest.skip("pandas not available")

        # Create pipeline
        pipeline = Pipeline(
            [("binner", DBSCANBinning(eps=10.0, min_samples=3, preserve_dataframe=True))]
        )

        # Use DataFrame
        df = pd.DataFrame(sample_data["multi_col"], columns=["feat1", "feat2"])

        with expect_fallback_warning:
            pipeline.fit(df)
        df_transformed = pipeline.transform(df)

        # Should preserve DataFrame format
        assert isinstance(df_transformed, pd.DataFrame)
        assert list(df_transformed.columns) == ["feat1", "feat2"]

    def test_sklearn_pipeline_param_access(self, sample_data):
        """Test parameter access in sklearn pipeline."""
        pipeline = Pipeline([("binner", DBSCANBinning(eps=0.5, min_samples=3))])

        # Test parameter access
        params = pipeline.get_params()
        assert "binner__eps" in params
        assert params["binner__eps"] == 0.5
        assert "binner__min_samples" in params
        assert params["binner__min_samples"] == 3

        # Test parameter setting
        pipeline.set_params(binner__eps=1.0, binner__min_samples=5)
        assert pipeline.named_steps["binner"].eps == 1.0
        assert pipeline.named_steps["binner"].min_samples == 5

    # Edge case tests

    def test_edge_case_nan_values(self, sample_data, expect_fallback_warning):
        """Test handling of NaN values."""
        binner = DBSCANBinning(eps=1.0, min_samples=3)
        X_nan = sample_data["with_nan"]

        # Should handle NaN values gracefully (base class preprocesses)
        with expect_fallback_warning:
            binner.fit(X_nan)
        result = binner.transform(X_nan)

        assert result is not None
        # NaN values should be handled by the base class
        assert result.shape == X_nan.shape

    def test_edge_case_inf_values(self, sample_data):
        """Test handling of infinite values."""
        binner = DBSCANBinning(eps=1.0, min_samples=3)
        X_inf = sample_data["with_inf"]

        # Should handle inf values based on config (clip, error, etc.)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings for this test
            binner.fit(X_inf)
            result = binner.transform(X_inf)

            assert result is not None
            assert result.shape == X_inf.shape

    def test_edge_case_constant_column(self, sample_data, expect_fallback_warning):
        """Test handling of constant columns."""
        binner = DBSCANBinning(eps=0.1, min_samples=3, min_bins=2)
        X_constant = sample_data["constant"]

        # Should fall back to equal-width binning for constant data
        with expect_fallback_warning:
            binner.fit(X_constant)

        result = binner.transform(X_constant)

        assert result is not None
        assert result.shape == X_constant.shape
        # All values should map to bins for constant data
        assert len(np.unique(result)) >= 1

    def test_edge_case_insufficient_data_for_min_samples(self, sample_data):
        """Test when data size < min_samples."""
        X = np.array([[1], [2], [3]])  # Only 3 data points
        binner = DBSCANBinning(eps=0.5, min_samples=5)  # More min_samples than data

        # Should raise ValueError because insufficient data for DBSCAN
        with pytest.raises(
            ValueError, match="Insufficient finite values \\(3\\) for DBSCAN clustering"
        ):
            binner.fit(X)

    def test_edge_case_no_clusters_found(self, sample_data, expect_fallback_warning):
        """Test when DBSCAN finds no clusters (all noise)."""
        # Use very small eps so no points are close enough to form clusters
        binner = DBSCANBinning(eps=0.001, min_samples=3, min_bins=2)
        X = sample_data["sparse"]  # Sparse data with large gaps

        # Should fall back to equal-width binning when no clusters found
        with expect_fallback_warning:
            binner.fit(X)

        result = binner.transform(X)

        assert result is not None
        assert result.shape == X.shape

    def test_edge_case_too_few_clusters(self, sample_data, expect_fallback_warning):
        """Test when DBSCAN finds fewer clusters than min_bins."""
        # Use parameters that will likely find only 1 cluster
        binner = DBSCANBinning(eps=100.0, min_samples=3, min_bins=3)
        X = sample_data["simple"]

        # Should fall back to equal-width binning when too few clusters
        with expect_fallback_warning:
            binner.fit(X)

        result = binner.transform(X)

        assert result is not None
        assert result.shape == X.shape

    def test_edge_case_single_value_per_cluster(self):
        """Test when each cluster contains very few points."""
        X = np.array([[1], [2], [3], [4], [5]])  # 5 unique values
        binner = DBSCANBinning(eps=0.6, min_samples=1)  # Each point could be its own cluster

        binner.fit(X)
        result = binner.transform(X)

        assert result is not None
        assert result.shape == X.shape

    def test_edge_case_very_small_range(self, expect_fallback_warning):
        """Test with very small data range."""
        X = np.array([[1.0000001], [1.0000002], [1.0000003], [1.0000004], [1.0000005]])
        binner = DBSCANBinning(eps=0.000001, min_samples=2)

        with expect_fallback_warning:
            binner.fit(X)
        result = binner.transform(X)

        assert result is not None
        assert result.shape == X.shape

    def test_edge_case_large_data(self, expect_fallback_warning):
        """Test with large dataset."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (10000, 1))
        binner = DBSCANBinning(eps=0.5, min_samples=10)

        with expect_fallback_warning:
            binner.fit(X)
        result = binner.transform(X[:1000])  # Transform subset

        assert result is not None
        assert result.shape == (1000, 1)
        assert result.min() >= 0  # Bin indices should be >= 0

    def test_single_row_data(self):
        """Test handling of single row data."""
        X = np.array([[5.0]])
        binner = DBSCANBinning(eps=0.5, min_samples=3)

        # Should raise ValueError because insufficient data for DBSCAN
        with pytest.raises(
            ValueError, match="Insufficient finite values \\(1\\) for DBSCAN clustering"
        ):
            binner.fit(X)

    def test_single_row_data_with_min_samples_one(self):
        """Test handling of single row data with min_samples=1."""
        X = np.array([[5.0]])
        binner = DBSCANBinning(eps=0.5, min_samples=1, min_bins=1)

        binner.fit(X)
        result = binner.transform(X)

        assert result is not None
        assert result.shape == (1, 1)
        assert result[0, 0] == 0  # Should be assigned to bin 0

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        binner = DBSCANBinning(eps=0.5, min_samples=3)

        with pytest.raises((ValueError, ValidationError, FittingError)):
            binner.fit(np.array([]).reshape(0, 1))

    # Specific DBSCAN binning tests

    def test_clustered_data_effectiveness(self, sample_data):
        """Test DBSCAN effectiveness on clearly clustered data."""
        X = sample_data["clustered"]  # Data with 3 clear clusters
        binner = DBSCANBinning(eps=2.0, min_samples=5)

        binner.fit(X)

        # Check that DBSCAN found meaningful clusters
        representatives = binner.bin_representatives_[0]

        # Should have found at least 2 bins (could be 3 if all clusters detected)
        assert len(representatives) >= 2

        # Representatives should be reasonably separated for clustered data
        representatives_sorted = sorted(representatives)
        if len(representatives_sorted) >= 2:
            assert (
                representatives_sorted[1] - representatives_sorted[0] > 5
            )  # Reasonable separation

    def test_representatives_are_cluster_centers(self, sample_data):
        """Test that representatives are cluster centers."""
        X = sample_data["clustered"]
        binner = DBSCANBinning(eps=2.0, min_samples=5)

        binner.fit(X)

        # Representatives should be cluster centers
        representatives = binner.bin_representatives_[0]

        # Should have reasonable number of representatives
        assert len(representatives) >= 1

        # Representatives should be within data range
        assert min(representatives) >= X.min()
        assert max(representatives) <= X.max()

    def test_bin_edges_between_clusters(self, sample_data):
        """Test that bin edges are positioned between clusters."""
        X = sample_data["clustered"]
        binner = DBSCANBinning(eps=2.0, min_samples=5)

        binner.fit(X)

        edges = binner.bin_edges_[0]
        representatives = sorted(binner.bin_representatives_[0])

        # Should have len(representatives) + 1 edges
        assert len(edges) == len(representatives) + 1

        # First and last edges should encompass all data
        assert edges[0] <= X.min()
        assert edges[-1] >= X.max()

        # Interior edges should be between consecutive representatives
        if len(representatives) >= 2:
            for i in range(1, len(edges) - 1):
                edge = edges[i]
                # Edge should be between consecutive representatives
                assert representatives[i - 1] < edge < representatives[i]

    def test_eps_parameter_effects(self, sample_data):
        """Test different eps parameter values."""
        X = sample_data["clustered"]

        # Small eps - should find more, smaller clusters or fall back
        binner_small = DBSCANBinning(eps=0.5, min_samples=3, min_bins=2)
        binner_small.fit(X)
        result_small = binner_small.transform(X)

        # Large eps - should find fewer, larger clusters
        binner_large = DBSCANBinning(eps=10.0, min_samples=3, min_bins=2)
        binner_large.fit(X)
        result_large = binner_large.transform(X)

        # Both should work
        assert result_small is not None
        assert result_large is not None

        # Large eps should generally produce fewer bins (unless fallback)
        n_bins_small = len(np.unique(result_small))
        n_bins_large = len(np.unique(result_large))

        # At least one should produce meaningful results
        assert n_bins_small >= 1
        assert n_bins_large >= 1

    def test_min_samples_parameter_effects(self, sample_data):
        """Test different min_samples parameter values."""
        X = sample_data["clustered"]

        # Small min_samples - should be more permissive for forming clusters
        binner_small = DBSCANBinning(eps=2.0, min_samples=2)
        binner_small.fit(X)
        result_small = binner_small.transform(X)

        # Large min_samples - should be more restrictive
        binner_large = DBSCANBinning(eps=2.0, min_samples=10)
        binner_large.fit(X)
        result_large = binner_large.transform(X)

        # Both should work
        assert result_small is not None
        assert result_large is not None

    def test_min_bins_parameter_fallback(self, sample_data, expect_fallback_warning):
        """Test min_bins parameter triggers fallback."""
        X = sample_data["simple"]

        # Use parameters that will find few clusters but require more bins
        binner = DBSCANBinning(eps=10.0, min_samples=3, min_bins=5)

        with expect_fallback_warning:
            binner.fit(X)
        result = binner.transform(X)

        # Should fall back to equal-width binning with min_bins
        assert result is not None
        assert len(binner.bin_representatives_[0]) == 5  # Should have min_bins representatives

    def test_fit_jointly_parameter(self, sample_data, expect_fallback_warning):
        """Test fit_jointly parameter (though DBSCAN typically doesn't use it)."""
        X = sample_data["multi_col"]

        # Test with fit_jointly=True
        binner_joint = DBSCANBinning(eps=10.0, min_samples=3, fit_jointly=True)
        with expect_fallback_warning:
            binner_joint.fit(X)

        # Test with fit_jointly=False
        binner_indep = DBSCANBinning(eps=10.0, min_samples=3, fit_jointly=False)
        with expect_fallback_warning:
            binner_indep.fit(X)

        # For DBSCAN, the result should be the same since it's inherently per-column
        # But test that both work
        result_joint = binner_joint.transform(X[:3])
        result_indep = binner_indep.transform(X[:3])

        assert result_joint is not None
        assert result_indep is not None

    def test_multiple_columns_independent_dbscan(self, sample_data, expect_fallback_warning):
        """Test that columns use independent DBSCAN clustering."""
        X = sample_data["multi_col"]  # Different ranges per column
        binner = DBSCANBinning(eps=5.0, min_samples=3)

        with expect_fallback_warning:
            binner.fit(X)

        # Each column should have its own cluster centers based on its data
        reps_col1 = binner.bin_representatives_[0]
        reps_col2 = binner.bin_representatives_[1]

        assert len(reps_col1) >= 1
        assert len(reps_col2) >= 1

        # Representatives should be different for columns with different ranges
        # (unless both fall back to equal-width with same parameters)
        if len(reps_col1) == len(reps_col2):
            # They might be equal length but should have different values
            assert not np.allclose(reps_col1, reps_col2, rtol=1e-2)

    def test_clip_parameter_functionality(self, sample_data, expect_fallback_warning):
        """Test clip parameter with out-of-range values."""
        X_train = np.array([[0], [5], [10], [15], [20]])
        X_test = np.array([[-5], [25]])  # Out of range values

        # Test with clip=True
        binner_clip = DBSCANBinning(eps=3.0, min_samples=2, clip=True)
        with expect_fallback_warning:
            binner_clip.fit(X_train)
        result_clip = binner_clip.transform(X_test)

        # Should clip to valid bin range
        n_bins = len(binner_clip.bin_representatives_[0])
        assert 0 <= result_clip.min()
        assert result_clip.max() < n_bins

        # Test with clip=False
        binner_no_clip = DBSCANBinning(eps=3.0, min_samples=2, clip=False)
        with expect_fallback_warning:
            binner_no_clip.fit(X_train)
        result_no_clip = binner_no_clip.transform(X_test)

        # May have values outside normal bin range (MISSING_VALUE)
        # The exact behavior depends on the base class implementation
        assert result_no_clip is not None

    # Integration and workflow tests

    def test_full_workflow_with_all_methods(self, sample_data, expect_fallback_warning):
        """Test complete workflow: fit, transform, inverse_transform, get_params, set_params."""
        X = sample_data["multi_col"]

        # Initialize and fit
        binner = DBSCANBinning(eps=10.0, min_samples=3, preserve_dataframe=False)
        with expect_fallback_warning:
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
        assert "eps" in params
        assert "min_samples" in params
        assert "bin_edges" in params
        assert params["eps"] == 10.0
        assert params["min_samples"] == 3

        # Set params and verify
        new_binner = DBSCANBinning()
        new_binner.set_params(**params)

        X_binned_2 = new_binner.transform(X)
        np.testing.assert_array_equal(X_binned, X_binned_2)

    def test_consistency_across_transforms(self, sample_data):
        """Test that repeated transforms give consistent results."""
        X = sample_data["normal"]
        binner = DBSCANBinning(eps=2.0, min_samples=5)
        binner.fit(X)

        # Multiple transforms should give same result
        result1 = binner.transform(X)
        result2 = binner.transform(X)
        result3 = binner.transform(X)

        np.testing.assert_array_equal(result1, result2)
        np.testing.assert_array_equal(result2, result3)

    def test_partial_data_transform(self, sample_data, expect_fallback_warning):
        """Test transforming data subset after fitting on full data."""
        X_full = sample_data["uniform"]  # 100 samples
        X_subset = X_full[:20]  # 20 samples

        binner = DBSCANBinning(eps=5.0, min_samples=5)
        with expect_fallback_warning:
            binner.fit(X_full)

        # Transform subset
        result_subset = binner.transform(X_subset)
        result_full = binner.transform(X_full)

        # Subset result should match first 20 rows of full result
        np.testing.assert_array_equal(result_subset, result_full[:20])

    def test_different_parameter_combinations(self, sample_data):
        """Test various parameter combinations."""
        X = sample_data["clustered"]

        test_params = [
            {"eps": 1.0, "min_samples": 3, "min_bins": 2},
            {"eps": 2.0, "min_samples": 5, "min_bins": 3},
            {"eps": 3.0, "min_samples": 10, "min_bins": 2},
        ]

        for params in test_params:
            binner = DBSCANBinning(**params)
            binner.fit(X)
            result = binner.transform(X)

            # Check that we get valid results
            assert result is not None
            assert result.shape == X.shape
            assert result.min() >= 0  # Bin indices should be >= 0

            # Check that we have at least min_bins
            n_bins = len(binner.bin_representatives_[0])
            assert n_bins >= params["min_bins"]

    def test_noise_handling_in_dbscan(self):
        """Test DBSCAN's handling of noise points."""
        # Create data with clear clusters and noise
        np.random.seed(42)
        cluster1 = np.random.normal(10, 0.5, 20)
        cluster2 = np.random.normal(30, 0.5, 20)
        noise = np.array([0, 50, 100])  # Outlier points
        X = np.concatenate([cluster1, cluster2, noise]).reshape(-1, 1)

        binner = DBSCANBinning(eps=2.0, min_samples=5)
        binner.fit(X)
        result = binner.transform(X)

        # Should still produce valid binning despite noise points
        assert result is not None
        assert result.shape == X.shape
        assert result.min() >= 0

        # Should have found the main clusters
        n_representatives = len(binner.bin_representatives_[0])
        assert n_representatives >= 2  # At least found the two main clusters

    def test_dbscan_no_fallback_error(self):
        """Test DBSCAN error when fallback is disabled and insufficient clusters found."""
        # Create data that will result in very few or no clusters
        X = np.array([[1.0], [100.0]])  # Two very distant points

        binner = DBSCANBinning(
            eps=0.1,  # Very small eps to prevent clustering
            min_samples=2,
            min_bins=5,  # More bins than possible clusters
            allow_fallback=False,  # Disable fallback to trigger error
        )

        # This should raise ConfigurationError (covers line 296 in _dbscan_binning.py)
        with pytest.raises(ConfigurationError, match="DBSCAN found only .* clusters"):
            binner.fit(X)

    def test_dbscan_fallback_warnings(self):
        """Test DBSCAN fallback warnings can be suppressed."""
        X = np.array([[1.0], [100.0]])  # Data that will cause fallback

        binner = DBSCANBinning(
            eps=0.1,
            min_samples=2,
            min_bins=5,
            allow_fallback=True,  # Allow fallback to trigger warning
        )

        # Suppress warnings during test to avoid test output noise
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            binner.fit(X)
            result = binner.transform(X)

        # Should still work despite fallback
        assert result is not None
        assert result.shape == X.shape
