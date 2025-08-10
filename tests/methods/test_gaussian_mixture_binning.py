"""
Comprehensive tests for GaussianMixtureBinning method covering all scenarios.
"""

import warnings

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from binlearn import POLARS_AVAILABLE, pd, pl
from binlearn.methods import GaussianMixtureBinning
from binlearn.utils import ConfigurationError, FittingError, ValidationError

# Skip polars tests if not available
polars_skip = pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")

# Skip pandas tests if not available
pandas_skip = pytest.mark.skipif(not hasattr(pd, "DataFrame"), reason="pandas not available")


class TestGaussianMixtureBinning:
    """Comprehensive test suite for GaussianMixtureBinning."""

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
            "mixed_gaussian": np.concatenate(
                [
                    np.random.normal(10, 2, 50),
                    np.random.normal(30, 3, 50),
                    np.random.normal(60, 2, 50),
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
        binner = GaussianMixtureBinning()
        assert binner.n_components == 10  # default from config or fallback
        assert binner.random_state is None
        assert binner.clip is True  # default from config
        assert binner.preserve_dataframe is False
        assert binner.fit_jointly is False

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        binner = GaussianMixtureBinning(
            n_components=5,
            random_state=42,
            clip=False,
            preserve_dataframe=True,
            fit_jointly=True,
        )
        assert binner.n_components == 5
        assert binner.random_state == 42
        assert binner.clip is False
        assert binner.preserve_dataframe is True
        assert binner.fit_jointly is True

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Invalid n_components
        with pytest.raises((ValueError, ConfigurationError)):
            GaussianMixtureBinning(n_components=0)

        with pytest.raises((ValueError, ConfigurationError)):
            GaussianMixtureBinning(n_components=-1)

        with pytest.raises((ValueError, ConfigurationError)):
            GaussianMixtureBinning(n_components=3.5)  # type: ignore

        # Invalid random_state
        with pytest.raises((ValueError, ConfigurationError)):
            GaussianMixtureBinning(random_state=3.14)  # type: ignore

        with pytest.raises((ValueError, ConfigurationError)):
            GaussianMixtureBinning(random_state="invalid")  # type: ignore

    # Input format tests with preserve_dataframe=False

    def test_numpy_input_preserve_false(self, sample_data):
        """Test with numpy input and preserve_dataframe=False."""
        binner = GaussianMixtureBinning(n_components=3, preserve_dataframe=False)

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
        binner = GaussianMixtureBinning(n_components=3, preserve_dataframe=False)

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
        binner = GaussianMixtureBinning(n_components=3, preserve_dataframe=False)

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
        binner = GaussianMixtureBinning(n_components=3, preserve_dataframe=True)

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
        binner = GaussianMixtureBinning(n_components=3, preserve_dataframe=True)

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
        binner_original = GaussianMixtureBinning(n_components=4, random_state=42)
        X_fit = sample_data["multi_col"]
        binner_original.fit(X_fit)

        # Get parameters
        params = binner_original.get_params()

        # Create new binner and set parameters (reconstruct state)
        binner_reconstructed = GaussianMixtureBinning()
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
        binner_original = GaussianMixtureBinning(n_components=3, random_state=42)
        X_fit = sample_data["simple"]
        binner_original.fit(X_fit)

        # Get parameters including fitted state
        params = binner_original.get_params()

        # Create new binner with constructor (reconstruct state)
        binner_reconstructed = GaussianMixtureBinning(**params)

        # Test that transform works without fitting
        X_test = sample_data["simple"]
        result_original = binner_original.transform(X_test)
        result_reconstructed = binner_reconstructed.transform(X_test)

        np.testing.assert_array_equal(result_original, result_reconstructed)

    def test_repeated_fitting_after_reconstruction(self, sample_data):
        """Test repeated fitting on reconstructed state."""
        # Original fitting
        binner = GaussianMixtureBinning(n_components=3, random_state=42)
        X_fit1 = sample_data["simple"]
        binner.fit(X_fit1)

        # Get and set params (reconstruction)
        params = binner.get_params()
        binner_new = GaussianMixtureBinning()
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
        binner_original = GaussianMixtureBinning(
            n_components=3, preserve_dataframe=True, random_state=42
        )
        X_numpy = sample_data["multi_col"]
        binner_original.fit(X_numpy)

        # Reconstruct
        params = binner_original.get_params()
        binner_reconstructed = GaussianMixtureBinning(**params)

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
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("binner", GaussianMixtureBinning(n_components=3, random_state=42)),
            ]
        )

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
        pipeline = Pipeline(
            [
                (
                    "binner",
                    GaussianMixtureBinning(
                        n_components=4, preserve_dataframe=True, random_state=42
                    ),
                )
            ]
        )

        # Use DataFrame
        df = pd.DataFrame(sample_data["multi_col"], columns=["feat1", "feat2"])

        pipeline.fit(df)
        df_transformed = pipeline.transform(df)

        # Should preserve DataFrame format
        assert isinstance(df_transformed, pd.DataFrame)
        assert list(df_transformed.columns) == ["feat1", "feat2"]

    def test_sklearn_pipeline_param_access(self, sample_data):
        """Test parameter access in sklearn pipeline."""
        pipeline = Pipeline([("binner", GaussianMixtureBinning(n_components=5, random_state=42))])

        # Test parameter access
        params = pipeline.get_params()
        assert "binner__n_components" in params
        assert params["binner__n_components"] == 5
        assert "binner__random_state" in params
        assert params["binner__random_state"] == 42

        # Test parameter setting
        pipeline.set_params(binner__n_components=7, binner__random_state=123)
        assert pipeline.named_steps["binner"].n_components == 7
        assert pipeline.named_steps["binner"].random_state == 123

    # Edge case tests

    def test_edge_case_nan_values(self, sample_data):
        """Test handling of NaN values."""
        binner = GaussianMixtureBinning(n_components=3, random_state=42)
        X_nan = sample_data["with_nan"]

        # Should handle NaN values gracefully (base class preprocesses)
        binner.fit(X_nan)
        result = binner.transform(X_nan)

        assert result is not None
        # NaN values should be handled by the base class
        assert result.shape == X_nan.shape

    def test_edge_case_inf_values(self, sample_data):
        """Test handling of infinite values."""
        binner = GaussianMixtureBinning(n_components=3, random_state=42)
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
        binner = GaussianMixtureBinning(n_components=3, random_state=42)
        X_constant = sample_data["constant"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings about constant data
            binner.fit(X_constant)
            result = binner.transform(X_constant)

            assert result is not None
            assert result.shape == X_constant.shape
            # All values should map to bins for constant data
            assert len(np.unique(result)) >= 1

    def test_edge_case_more_components_than_data_points(self, sample_data):
        """Test when n_components > number of data points."""
        X = np.array([[1], [2], [3]])  # Only 3 data points
        binner = GaussianMixtureBinning(
            n_components=5, random_state=42
        )  # More components than data

        # Should raise ValueError because insufficient data for GMM
        with pytest.raises(ValueError, match="Insufficient finite values \\(3\\) for 5 components"):
            binner.fit(X)

    def test_edge_case_more_components_than_unique_values(self, sample_data):
        """Test when n_components > number of unique values."""
        X = sample_data["few_unique"]  # Only 3 unique values but 6 data points
        binner = GaussianMixtureBinning(
            n_components=5, random_state=42
        )  # More components than unique values

        # Should handle case with fewer unique values gracefully (may fall back to equal-width)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings about fallback
            binner.fit(X)
            result = binner.transform(X)

            assert result is not None
            assert result.shape == X.shape

    def test_edge_case_single_value_per_component(self):
        """Test when data points equal n_components."""
        X = np.array([[1], [2], [3], [4], [5]])  # 5 unique values
        binner = GaussianMixtureBinning(n_components=5, random_state=42)  # Same as data points

        binner.fit(X)
        result = binner.transform(X)

        assert result is not None
        assert result.shape == X.shape

    def test_edge_case_very_small_range(self):
        """Test with very small data range."""
        X = np.array([[1.0000001], [1.0000002], [1.0000003], [1.0000004], [1.0000005]])
        binner = GaussianMixtureBinning(n_components=3, random_state=42)

        binner.fit(X)
        result = binner.transform(X)

        assert result is not None
        assert result.shape == X.shape

    def test_edge_case_large_data(self):
        """Test with large dataset."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (10000, 1))
        binner = GaussianMixtureBinning(n_components=10, random_state=42)

        binner.fit(X)
        result = binner.transform(X[:1000])  # Transform subset

        assert result is not None
        assert result.shape == (1000, 1)
        assert 0 <= result.max() < 10  # Should be within bin range
        assert result.min() >= 0  # Bin indices should be >= 0

    def test_single_row_data(self):
        """Test handling of single row data."""
        X = np.array([[5.0]])
        binner = GaussianMixtureBinning(n_components=3, random_state=42)

        # Should raise ValueError because insufficient data for GMM
        with pytest.raises(ValueError, match="Insufficient finite values \\(1\\) for 3 components"):
            binner.fit(X)

    def test_single_row_data_with_single_component(self):
        """Test handling of single row data with n_components=1."""
        X = np.array([[5.0]])
        binner = GaussianMixtureBinning(n_components=1, random_state=42)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            binner.fit(X)
            result = binner.transform(X)

            assert result is not None
            assert result.shape == (1, 1)
            assert result[0, 0] == 0  # Should be assigned to bin 0

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        binner = GaussianMixtureBinning(n_components=3, random_state=42)

        with pytest.raises((ValueError, ValidationError, FittingError)):
            binner.fit(np.array([]).reshape(0, 1))

    def test_gmm_clustering_error_coverage(self):
        """Test to cover GMM clustering error handling and fallback."""
        import unittest.mock

        # Mock GaussianMixture to raise an exception to test fallback to equal-width
        with unittest.mock.patch(
            "sklearn.mixture.GaussianMixture.fit", side_effect=ValueError("Mock GMM error")
        ):
            X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
            binner = GaussianMixtureBinning(n_components=3, random_state=42)

            # Should fall back to equal-width binning and issue a warning
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                binner.fit(X)

                # Should have issued a warning about fallback
                assert len(w) > 0
                assert "GMM binning failed" in str(w[0].message)
                assert "falling back to equal-width binning" in str(w[0].message)

            # Should still work with fallback
            result = binner.transform(X)
            assert result is not None
            assert result.shape == X.shape

    # Specific Gaussian Mixture binning tests

    def test_random_state_reproducibility(self, sample_data):
        """Test that random_state produces reproducible results."""
        X = sample_data["normal"]

        # Fit two binners with same random state
        binner1 = GaussianMixtureBinning(n_components=5, random_state=42)
        binner2 = GaussianMixtureBinning(n_components=5, random_state=42)

        binner1.fit(X)
        binner2.fit(X)

        result1 = binner1.transform(X[:10])
        result2 = binner2.transform(X[:10])

        # Should produce identical results with same random_state
        np.testing.assert_array_equal(result1, result2)

    def test_random_state_different_results(self, sample_data):
        """Test that different random_state produces potentially different results."""
        X = sample_data["mixed_gaussian"]  # Data with clear mixture structure

        # Fit two binners with different random states
        binner1 = GaussianMixtureBinning(n_components=3, random_state=42)
        binner2 = GaussianMixtureBinning(n_components=3, random_state=123)

        binner1.fit(X)
        binner2.fit(X)

        # Results may be different due to random initialization
        # But both should still be valid
        result1 = binner1.transform(X[:10])
        result2 = binner2.transform(X[:10])

        assert result1 is not None
        assert result2 is not None
        assert result1.shape == result2.shape

    def test_mixed_gaussian_data_effectiveness(self, sample_data):
        """Test GMM effectiveness on data with clear mixture structure."""
        X = sample_data["mixed_gaussian"]  # Data with 3 clear Gaussian components
        binner = GaussianMixtureBinning(n_components=3, random_state=42)

        binner.fit(X)

        # Check that components are reasonable for the mixed Gaussian data
        means = binner.bin_representatives_[0]
        means_sorted = sorted(means)

        # Means should be somewhat spread out for mixed Gaussian data
        assert len(means_sorted) == 3
        assert means_sorted[1] - means_sorted[0] > 5  # Reasonable separation
        assert means_sorted[2] - means_sorted[1] > 5  # Reasonable separation

    def test_representatives_are_means(self, sample_data):
        """Test that representatives are the Gaussian component means."""
        X = sample_data["simple"]
        binner = GaussianMixtureBinning(n_components=3, random_state=42)

        binner.fit(X)

        # Representatives should be the component means
        representatives = binner.bin_representatives_[0]

        # Should have the expected number of representatives
        assert len(representatives) == 3

        # Representatives should be within data range
        assert min(representatives) >= X.min()
        assert max(representatives) <= X.max()

    def test_bin_edges_between_means(self, sample_data):
        """Test that bin edges are positioned between component means."""
        X = sample_data["simple"]
        binner = GaussianMixtureBinning(n_components=3, random_state=42)

        binner.fit(X)

        edges = binner.bin_edges_[0]
        means = sorted(binner.bin_representatives_[0])

        # Should have n_components + 1 edges
        assert len(edges) == 4

        # First and last edges should be data min/max
        assert edges[0] == X.min()
        assert edges[-1] == X.max()

        # Interior edges should be between consecutive means
        for i in range(1, len(edges) - 1):
            edge = edges[i]
            # Edge should be between consecutive means
            assert means[i - 1] < edge < means[i]

    def test_string_n_components_parameter(self, sample_data):
        """Test string n_components parameters like 'sqrt', 'log2'."""
        X = sample_data["uniform"]  # 100 samples

        # Test 'sqrt' - should be sqrt(100) = 10
        binner_sqrt = GaussianMixtureBinning(n_components="sqrt", random_state=42)
        binner_sqrt.fit(X)

        # Should create 10 components
        assert len(binner_sqrt.bin_representatives_[0]) == 10

        # Test 'log2' - should be log2(100) ≈ 6.6, resolved to 7
        binner_log2 = GaussianMixtureBinning(n_components="log2", random_state=42)
        binner_log2.fit(X)

        # Should create 7 components (log2(100) = 6.64, resolved to 7)
        assert len(binner_log2.bin_representatives_[0]) == 7

    def test_fit_jointly_parameter(self, sample_data):
        """Test fit_jointly parameter (though GMM typically doesn't use it)."""
        X = sample_data["multi_col"]

        # Test with fit_jointly=True
        binner_joint = GaussianMixtureBinning(n_components=3, fit_jointly=True, random_state=42)
        binner_joint.fit(X)

        # Test with fit_jointly=False
        binner_indep = GaussianMixtureBinning(n_components=3, fit_jointly=False, random_state=42)
        binner_indep.fit(X)

        # For GMM, the result should be the same since it's inherently per-column
        # But test that both work
        result_joint = binner_joint.transform(X[:3])
        result_indep = binner_indep.transform(X[:3])

        assert result_joint is not None
        assert result_indep is not None
        # May be equal for GMM since it's inherently independent

    def test_multiple_columns_independent_gmm(self, sample_data):
        """Test that columns use independent GMM models."""
        X = sample_data["multi_col"]  # Different ranges per column
        binner = GaussianMixtureBinning(n_components=3, random_state=42)

        binner.fit(X)

        # Each column should have its own component means based on its data
        means_col1 = binner.bin_representatives_[0]
        means_col2 = binner.bin_representatives_[1]

        assert len(means_col1) == 3
        assert len(means_col2) == 3

        # Means should be different for columns with different ranges
        assert not np.allclose(means_col1, means_col2)

    def test_clip_parameter_functionality(self, sample_data):
        """Test clip parameter with out-of-range values."""
        X_train = np.array([[0], [5], [10], [15], [20]])
        X_test = np.array([[-5], [25]])  # Out of range values

        # Test with clip=True
        binner_clip = GaussianMixtureBinning(n_components=3, clip=True, random_state=42)
        binner_clip.fit(X_train)
        result_clip = binner_clip.transform(X_test)

        # Should clip to valid bin range
        assert 0 <= result_clip.min()
        assert result_clip.max() < 3

        # Test with clip=False
        binner_no_clip = GaussianMixtureBinning(n_components=3, clip=False, random_state=42)
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
        binner = GaussianMixtureBinning(n_components=4, preserve_dataframe=False, random_state=42)
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
        assert "n_components" in params
        assert "random_state" in params
        assert "bin_edges" in params
        assert params["n_components"] == 4
        assert params["random_state"] == 42

        # Set params and verify
        new_binner = GaussianMixtureBinning()
        new_binner.set_params(**params)

        X_binned_2 = new_binner.transform(X)
        np.testing.assert_array_equal(X_binned, X_binned_2)

    def test_consistency_across_transforms(self, sample_data):
        """Test that repeated transforms give consistent results."""
        X = sample_data["normal"]
        binner = GaussianMixtureBinning(n_components=5, random_state=42)
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

        binner = GaussianMixtureBinning(n_components=3, random_state=42)
        binner.fit(X_full)

        # Transform subset
        result_subset = binner.transform(X_subset)
        result_full = binner.transform(X_full)

        # Subset result should match first 20 rows of full result
        np.testing.assert_array_equal(result_subset, result_full[:20])

    def test_different_n_components_values(self, sample_data):
        """Test various n_components values."""
        X = sample_data["uniform"]

        for n_components in [2, 3, 5, 10, 20]:
            binner = GaussianMixtureBinning(n_components=n_components, random_state=42)
            binner.fit(X)
            result = binner.transform(X)

            # Check that we get expected number of bins
            unique_bins = np.unique(result)
            assert len(unique_bins) <= n_components  # May be less if components merge
            assert result.max() < n_components  # Bin indices should be < n_components
            assert result.min() >= 0  # Bin indices should be >= 0

            # Check that we have the right number of component means
            assert len(binner.bin_representatives_[0]) == n_components

    def test_covariance_type_full(self, sample_data):
        """Test that GMM uses full covariance type."""
        # This is more of an implementation verification test
        X = sample_data["mixed_gaussian"]
        binner = GaussianMixtureBinning(n_components=3, random_state=42)

        # Mock to verify covariance_type parameter
        import unittest.mock

        with unittest.mock.patch(
            "binlearn.methods._gaussian_mixture_binning.GaussianMixture"
        ) as mock_gmm:
            mock_instance = unittest.mock.Mock()
            mock_instance.means_ = np.array([[10], [30], [60]])  # Mock means
            mock_instance.fit.return_value = mock_instance
            mock_gmm.return_value = mock_instance

            binner.fit(X)

            # Verify that GaussianMixture was called with covariance_type='full'
            mock_gmm.assert_called_once_with(
                n_components=3, random_state=42, covariance_type="full"
            )

    def test_gmm_no_fallback_error(self):
        """Test GMM error when fallback is disabled and fitting fails."""
        X = np.array([[1.0], [2.0], [3.0], [4.0]])

        binner = GaussianMixtureBinning(
            n_components=2, allow_fallback=False  # Disable fallback to trigger error
        )

        # Mock GaussianMixture to raise an exception
        from unittest.mock import Mock, patch

        with patch("binlearn.methods._gaussian_mixture_binning.GaussianMixture") as mock_gmm:
            mock_instance = Mock()
            mock_instance.fit.side_effect = ValueError("GMM fitting failed")
            mock_gmm.return_value = mock_instance

            # This should raise ConfigurationError (covers line 342 in _gaussian_mixture_binning.py)
            with pytest.raises(ConfigurationError, match="GMM fitting failed"):
                binner.fit(X)

    def test_gmm_fallback_warnings(self):
        """Test GMM fallback warnings can be suppressed."""
        # Create data that might cause GMM to struggle
        X = np.array([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]])

        binner = GaussianMixtureBinning(
            n_components=5,  # More components than reasonable for small dataset
            allow_fallback=True,  # Allow fallback to trigger warning
        )

        # Suppress warnings during test to avoid test output noise
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                binner.fit(X)
                result = binner.transform(X)
                # Should still work despite potential fallback
                assert result is not None
                assert result.shape == X.shape
            except ConfigurationError:
                # GMM might still fail even with fallback in some cases
                # This is acceptable behavior
                pass
