"""
Comprehensive tests for EqualWidthBinning method covering all scenarios:
- Various input/output formats (numpy, pandas, polars)
- Fitted state reconstruction
- Repeated fitting
- sklearn pipeline integration
- Edge cases
"""

import warnings

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from binlearn import POLARS_AVAILABLE, pd, pl
from binlearn.methods import EqualWidthBinning
from binlearn.utils import FittingError, ValidationError

# Skip polars tests if not available
polars_skip = pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")

# Skip pandas tests if not available
pandas_skip = pytest.mark.skipif(not hasattr(pd, "DataFrame"), reason="pandas not available")


class TestEqualWidthBinning:
    """Comprehensive test suite for EqualWidthBinning."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        return {
            "simple": np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1),
            "multi_col": np.array(
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]]
            ),
            "uniform": np.linspace(0, 100, 100).reshape(-1, 1),
            "normal": np.random.normal(50, 15, 200).reshape(-1, 1),
            "with_nan": np.array([1.0, 2.0, np.nan, 4.0, 5.0]).reshape(-1, 1),
            "with_inf": np.array([1.0, 2.0, np.inf, 4.0, 5.0]).reshape(-1, 1),
            "constant": np.array([5.0, 5.0, 5.0, 5.0, 5.0]).reshape(-1, 1),
        }

    # Basic functionality tests

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        binner = EqualWidthBinning()
        assert binner.n_bins == 5  # default from config or fallback
        assert binner.bin_range is None
        assert binner.clip is True  # default from config
        assert binner.preserve_dataframe is False
        assert binner.fit_jointly is False

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        binner = EqualWidthBinning(
            n_bins=10, bin_range=(0, 100), clip=False, preserve_dataframe=True, fit_jointly=True
        )
        assert binner.n_bins == 10
        assert binner.bin_range == (0, 100)
        assert binner.clip is False
        assert binner.preserve_dataframe is True
        assert binner.fit_jointly is True

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Invalid n_bins
        with pytest.raises(ValueError, match="n_bins must be a positive integer"):
            EqualWidthBinning(n_bins=0)

        with pytest.raises(ValueError, match="n_bins must be a positive integer"):
            EqualWidthBinning(n_bins=-1)

        with pytest.raises(ValueError, match="n_bins must be a positive integer"):
            EqualWidthBinning(n_bins=3.5)  # type: ignore

        # Invalid bin_range
        with pytest.raises(ValueError, match="bin_range must be a tuple"):
            EqualWidthBinning(bin_range=(5, 5))  # min == max

        with pytest.raises(ValueError, match="bin_range must be a tuple"):
            EqualWidthBinning(bin_range=(10, 5))  # min > max

        with pytest.raises((ValueError, TypeError)):
            EqualWidthBinning(bin_range=(1, 2, 3))  # type: ignore  # wrong length

    # Input format tests with preserve_dataframe=False

    def test_numpy_input_preserve_false(self, sample_data):
        """Test with numpy input and preserve_dataframe=False."""
        binner = EqualWidthBinning(n_bins=3, preserve_dataframe=False)

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
        binner = EqualWidthBinning(n_bins=3, preserve_dataframe=False)

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
        binner = EqualWidthBinning(n_bins=3, preserve_dataframe=False)

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
        binner = EqualWidthBinning(n_bins=3, preserve_dataframe=True)

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
        binner = EqualWidthBinning(n_bins=3, preserve_dataframe=True)

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
        binner_original = EqualWidthBinning(n_bins=4)
        X_fit = sample_data["multi_col"]
        binner_original.fit(X_fit)

        # Get parameters
        params = binner_original.get_params()

        # Create new binner and set parameters (reconstruct state)
        binner_reconstructed = EqualWidthBinning()
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
        binner_original = EqualWidthBinning(n_bins=3, bin_range=(0, 10))
        X_fit = sample_data["simple"]
        binner_original.fit(X_fit)

        # Get parameters including fitted state
        params = binner_original.get_params()

        # Create new binner with constructor (reconstruct state)
        binner_reconstructed = EqualWidthBinning(**params)

        # Test that transform works without fitting
        X_test = sample_data["simple"]
        result_original = binner_original.transform(X_test)
        result_reconstructed = binner_reconstructed.transform(X_test)

        np.testing.assert_array_equal(result_original, result_reconstructed)

    def test_repeated_fitting_after_reconstruction(self, sample_data):
        """Test repeated fitting on reconstructed state."""
        # Original fitting
        binner = EqualWidthBinning(n_bins=3)
        X_fit1 = sample_data["simple"]
        binner.fit(X_fit1)

        # Get and set params (reconstruction)
        params = binner.get_params()
        binner_new = EqualWidthBinning()
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
        binner_original = EqualWidthBinning(n_bins=3, preserve_dataframe=True)
        X_numpy = sample_data["multi_col"]
        binner_original.fit(X_numpy)

        # Reconstruct
        params = binner_original.get_params()
        binner_reconstructed = EqualWidthBinning(**params)

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
        pipeline = Pipeline([("scaler", StandardScaler()), ("binner", EqualWidthBinning(n_bins=3))])

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
        pipeline = Pipeline([("binner", EqualWidthBinning(n_bins=4, preserve_dataframe=True))])

        # Use DataFrame
        df = pd.DataFrame(sample_data["multi_col"], columns=["feat1", "feat2"])

        pipeline.fit(df)
        df_transformed = pipeline.transform(df)

        # Should preserve DataFrame format
        assert isinstance(df_transformed, pd.DataFrame)
        assert list(df_transformed.columns) == ["feat1", "feat2"]

    def test_sklearn_pipeline_param_access(self, sample_data):
        """Test parameter access in sklearn pipeline."""
        pipeline = Pipeline([("binner", EqualWidthBinning(n_bins=5))])

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
        binner = EqualWidthBinning(n_bins=3)
        X_nan = sample_data["with_nan"]

        # Should handle NaN values gracefully (base class preprocesses)
        binner.fit(X_nan)
        result = binner.transform(X_nan)

        assert result is not None
        # NaN values should be handled by the base class
        assert result.shape == X_nan.shape

    def test_edge_case_inf_values(self, sample_data):
        """Test handling of infinite values."""
        binner = EqualWidthBinning(n_bins=3)
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
        binner = EqualWidthBinning(n_bins=3)
        X_constant = sample_data["constant"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings about constant data
            binner.fit(X_constant)
            result = binner.transform(X_constant)

            assert result is not None
            assert result.shape == X_constant.shape
            # All values should map to the same bin for constant data
            assert len(np.unique(result)) <= 1

    def test_edge_case_single_value_per_bin(self, sample_data):
        """Test when n_bins equals number of unique values."""
        X = np.array([[1], [2], [3]])  # 3 unique values
        binner = EqualWidthBinning(n_bins=3)  # Same as unique values

        binner.fit(X)
        result = binner.transform(X)

        assert result is not None
        assert result.shape == X.shape

    def test_edge_case_more_bins_than_data_points(self, sample_data):
        """Test when n_bins > number of data points."""
        X = np.array([[1], [2]])  # Only 2 data points
        binner = EqualWidthBinning(n_bins=5)  # More bins than data

        binner.fit(X)
        result = binner.transform(X)

        assert result is not None
        assert result.shape == X.shape

    def test_edge_case_very_small_range(self):
        """Test with very small data range."""
        X = np.array([[1.0000001], [1.0000002], [1.0000003]])
        binner = EqualWidthBinning(n_bins=3)

        binner.fit(X)
        result = binner.transform(X)

        assert result is not None
        assert result.shape == X.shape

    def test_edge_case_large_data(self):
        """Test with large dataset."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (10000, 1))
        binner = EqualWidthBinning(n_bins=10)

        binner.fit(X)
        result = binner.transform(X[:1000])  # Transform subset

        assert result is not None
        assert result.shape == (1000, 1)
        assert 0 <= result.max() < 10  # Should be within bin range

    # Specific equal width binning tests

    def test_equal_width_calculation(self):
        """Test that bins have equal widths."""
        X = np.array([[0], [10], [20], [30], [40], [50]])
        binner = EqualWidthBinning(n_bins=5)

        binner.fit(X)

        # Check bin edges
        edges = binner.bin_edges_[0]  # Column 0

        # Should have 6 edges for 5 bins
        assert len(edges) == 6

        # Check that widths are equal
        widths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
        np.testing.assert_array_almost_equal(widths, [widths[0]] * len(widths))

    def test_bin_range_parameter(self):
        """Test bin_range parameter functionality."""
        X = np.array([[5], [15], [25]])  # Data in range 5-25

        # Use wider range
        binner = EqualWidthBinning(n_bins=4, bin_range=(0, 40))
        binner.fit(X)

        edges = binner.bin_edges_[0]
        assert edges[0] == 0  # Should start from bin_range min
        assert edges[-1] == 40  # Should end at bin_range max

        # Check equal widths
        expected_width = 40 / 4
        for i in range(len(edges) - 1):
            assert abs((edges[i + 1] - edges[i]) - expected_width) < 1e-10

    def test_representatives_are_bin_centers(self):
        """Test that representatives are bin centers."""
        X = np.array([[0], [10]])
        binner = EqualWidthBinning(n_bins=2)

        binner.fit(X)

        edges = binner.bin_edges_[0]
        representatives = binner.bin_representatives_[0]

        # Check that representatives are midpoints
        for i in range(len(representatives)):
            expected_center = (edges[i] + edges[i + 1]) / 2
            assert abs(representatives[i] - expected_center) < 1e-10

    def test_clip_parameter_functionality(self, sample_data):
        """Test clip parameter with out-of-range values."""
        X_train = np.array([[0], [5], [10]])
        X_test = np.array([[-5], [15]])  # Out of range values

        # Test with clip=True
        binner_clip = EqualWidthBinning(n_bins=3, clip=True)
        binner_clip.fit(X_train)
        result_clip = binner_clip.transform(X_test)

        # Should clip to valid bin range
        assert 0 <= result_clip.min()
        assert result_clip.max() < 3

        # Test with clip=False
        binner_no_clip = EqualWidthBinning(n_bins=3, clip=False)
        binner_no_clip.fit(X_train)
        result_no_clip = binner_no_clip.transform(X_test)

        # May have values outside normal bin range (MISSING_VALUE)
        # The exact behavior depends on the base class implementation
        assert result_no_clip is not None

    def test_multiple_columns_independent_binning(self, sample_data):
        """Test that columns are binned independently."""
        X = sample_data["multi_col"]  # Different ranges per column
        binner = EqualWidthBinning(n_bins=3)

        binner.fit(X)

        # Each column should have its own bin edges based on its range
        edges_col1 = binner.bin_edges_[0]  # or 'feature_0'
        edges_col2 = binner.bin_edges_[1]  # or 'feature_1'

        assert len(edges_col1) == 4  # 3 bins = 4 edges
        assert len(edges_col2) == 4

        # Ranges should be different
        range1 = edges_col1[-1] - edges_col1[0]
        range2 = edges_col2[-1] - edges_col2[0]
        assert range1 != range2  # Different columns have different ranges

    def test_fit_jointly_parameter(self, sample_data):
        """Test fit_jointly parameter (though EqualWidth typically doesn't use it)."""
        X = sample_data["multi_col"]

        # Test with fit_jointly=True
        binner_joint = EqualWidthBinning(n_bins=3, fit_jointly=True)
        binner_joint.fit(X)

        # Test with fit_jointly=False
        binner_indep = EqualWidthBinning(n_bins=3, fit_jointly=False)
        binner_indep.fit(X)

        # For EqualWidth, the result should be the same since it's inherently per-column
        # But test that both work
        result_joint = binner_joint.transform(X[:3])
        result_indep = binner_indep.transform(X[:3])

        assert result_joint is not None
        assert result_indep is not None
        # May be equal for EqualWidth since it's inherently independent

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        binner = EqualWidthBinning(n_bins=3)

        with pytest.raises((ValueError, ValidationError, FittingError)):
            binner.fit(np.array([]).reshape(0, 1))

    def test_single_row_data(self):
        """Test handling of single row data."""
        X = np.array([[5.0]])
        binner = EqualWidthBinning(n_bins=3)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            binner.fit(X)
            result = binner.transform(X)

            assert result is not None
            assert result.shape == (1, 1)

    # Integration and workflow tests

    def test_full_workflow_with_all_methods(self, sample_data):
        """Test complete workflow: fit, transform, inverse_transform, get_params, set_params."""
        X = sample_data["multi_col"]

        # Initialize and fit
        binner = EqualWidthBinning(n_bins=4, preserve_dataframe=False)
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
        new_binner = EqualWidthBinning()
        new_binner.set_params(**params)

        X_binned_2 = new_binner.transform(X)
        np.testing.assert_array_equal(X_binned, X_binned_2)

    def test_consistency_across_transforms(self, sample_data):
        """Test that repeated transforms give consistent results."""
        X = sample_data["normal"]
        binner = EqualWidthBinning(n_bins=5)
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

        binner = EqualWidthBinning(n_bins=3)
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
            binner = EqualWidthBinning(n_bins=n_bins)
            binner.fit(X)
            result = binner.transform(X)

            # Check that we get expected number of bins
            unique_bins = np.unique(result)
            assert len(unique_bins) <= n_bins  # May be less if data maps to fewer bins
            assert result.max() < n_bins  # Bin indices should be < n_bins
            assert result.min() >= 0  # Bin indices should be >= 0
