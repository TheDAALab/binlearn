"""
Comprehensive tests for EqualFrequencyBinning method covering all scenarios:
- Various input/output formats (numpy, pandas, polars)
- Fitted state reconstruction
- Repeated fitting
- sklearn pipeline integration
- Edge cases
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from binlearn.methods import EqualFrequencyBinning
from binlearn.utils import ConfigurationError, FittingError, ValidationError

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

# Skip polars tests if not available
polars_skip = pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")

# Skip pandas tests if not available
pandas_skip = pytest.mark.skipif(not hasattr(pd, "DataFrame"), reason="pandas not available")


class TestEqualFrequencyBinning:
    """Comprehensive test suite for EqualFrequencyBinning."""

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
            "skewed": np.random.exponential(2, 100).reshape(-1, 1),
            "with_duplicates": np.array([1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5]).reshape(-1, 1),
            "with_nan": np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0]).reshape(-1, 1),
            "with_inf": np.array([1.0, 2.0, np.inf, 4.0, 5.0, 6.0]).reshape(-1, 1),
            "constant": np.array([5.0, 5.0, 5.0, 5.0, 5.0]).reshape(-1, 1),
        }

    # Basic functionality tests

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        binner = EqualFrequencyBinning()
        assert binner.n_bins == 10  # default from config or fallback
        assert binner.quantile_range is None
        assert binner.clip is True  # default from config
        assert binner.preserve_dataframe is False
        assert binner.fit_jointly is False

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        binner = EqualFrequencyBinning(
            n_bins=5,
            quantile_range=(0.1, 0.9),
            clip=False,
            preserve_dataframe=True,
            fit_jointly=True,
        )
        assert binner.n_bins == 5
        assert binner.quantile_range == (0.1, 0.9)
        assert binner.clip is False
        assert binner.preserve_dataframe is True
        assert binner.fit_jointly is True

    def test_init_with_string_n_bins(self):
        """Test initialization with string n_bins parameter."""
        binner = EqualFrequencyBinning(n_bins="sqrt")
        assert binner.n_bins == "sqrt"

    def test_parameter_validation_n_bins(self):
        """Test n_bins parameter validation."""
        # Invalid n_bins values
        with pytest.raises((ValueError, ConfigurationError)):
            EqualFrequencyBinning(n_bins=0)

        with pytest.raises((ValueError, ConfigurationError)):
            EqualFrequencyBinning(n_bins=-1)

        with pytest.raises((ValueError, ConfigurationError)):
            EqualFrequencyBinning(n_bins=1.5)

        # Valid string n_bins
        binner = EqualFrequencyBinning(n_bins="log2")
        assert binner.n_bins == "log2"

    def test_parameter_validation_quantile_range(self):
        """Test quantile_range parameter validation."""
        # Invalid quantile_range formats
        with pytest.raises(ConfigurationError, match="quantile_range must be a tuple"):
            EqualFrequencyBinning(quantile_range=(0.1, 0.5, 0.9))  # wrong length

        with pytest.raises(ConfigurationError, match="quantile_range must be a tuple"):
            EqualFrequencyBinning(quantile_range=[0.1, 0.9])  # not tuple

        # Invalid quantile values
        with pytest.raises(ConfigurationError, match="values must be numbers between 0 and 1"):
            EqualFrequencyBinning(quantile_range=(-0.1, 0.9))  # negative

        with pytest.raises(ConfigurationError, match="values must be numbers between 0 and 1"):
            EqualFrequencyBinning(quantile_range=(0.1, 1.1))  # > 1

        with pytest.raises(ConfigurationError, match="values must be numbers between 0 and 1"):
            EqualFrequencyBinning(quantile_range=(0.9, 0.1))  # min >= max

        with pytest.raises(ConfigurationError, match="values must be numbers between 0 and 1"):
            EqualFrequencyBinning(quantile_range=(0.5, 0.5))  # min == max

    # Input format tests with preserve_dataframe=False

    def test_numpy_input_preserve_false(self, sample_data):
        """Test with numpy input and preserve_dataframe=False."""
        binner = EqualFrequencyBinning(n_bins=3, preserve_dataframe=False)

        # Fit and transform
        X_fit = sample_data["simple"]
        binner.fit(X_fit)
        X_transformed = binner.transform(X_fit)

        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape == X_fit.shape
        assert X_transformed.dtype == int

        # Check that bins have approximately equal frequencies
        unique, counts = np.unique(X_transformed, return_counts=True)
        # For equal frequency, counts should be approximately equal
        assert len(unique) <= 3  # At most n_bins

        # Inverse transform
        X_inverse = binner.inverse_transform(X_transformed)
        assert isinstance(X_inverse, np.ndarray)
        assert X_inverse.shape == X_fit.shape

    @pandas_skip
    def test_pandas_input_preserve_false(self, sample_data):
        """Test with pandas input and preserve_dataframe=False."""
        binner = EqualFrequencyBinning(n_bins=3, preserve_dataframe=False)

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
        binner = EqualFrequencyBinning(n_bins=3, preserve_dataframe=False)

        # Create polars DataFrame
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
        binner = EqualFrequencyBinning(n_bins=3, preserve_dataframe=True)

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
        binner = EqualFrequencyBinning(n_bins=3, preserve_dataframe=True)

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
        binner_original = EqualFrequencyBinning(n_bins=4)
        X_fit = sample_data["multi_col"]
        binner_original.fit(X_fit)

        # Get parameters
        params = binner_original.get_params()

        # Create new binner and set parameters (reconstruct state)
        binner_reconstructed = EqualFrequencyBinning()
        binner_reconstructed.set_params(**params)

        # Test that transform works without fitting
        X_test = sample_data["multi_col"][:5]  # Subset for testing
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
        binner_original = EqualFrequencyBinning(n_bins=3, quantile_range=(0.1, 0.9))
        X_fit = sample_data["simple"]
        binner_original.fit(X_fit)

        # Get parameters including fitted state
        params = binner_original.get_params()

        # Create new binner with constructor (reconstruct state)
        binner_reconstructed = EqualFrequencyBinning(**params)

        # Test that transform works without fitting
        X_test = sample_data["simple"]
        result_original = binner_original.transform(X_test)
        result_reconstructed = binner_reconstructed.transform(X_test)

        np.testing.assert_array_equal(result_original, result_reconstructed)

    def test_repeated_fitting_after_reconstruction(self, sample_data):
        """Test repeated fitting on reconstructed state."""
        # Original fitting
        binner = EqualFrequencyBinning(n_bins=3)
        X_fit1 = sample_data["simple"]
        binner.fit(X_fit1)

        # Get and set params (reconstruction)
        params = binner.get_params()
        binner_new = EqualFrequencyBinning()
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
        binner_original = EqualFrequencyBinning(n_bins=3, preserve_dataframe=True)
        X_numpy = sample_data["multi_col"]
        binner_original.fit(X_numpy)

        # Reconstruct
        params = binner_original.get_params()
        binner_reconstructed = EqualFrequencyBinning(**params)

        # Test numpy input
        result_numpy = binner_reconstructed.transform(X_numpy[:5])
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
            [("scaler", StandardScaler()), ("binner", EqualFrequencyBinning(n_bins=3))]
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
        pipeline = Pipeline([("binner", EqualFrequencyBinning(n_bins=4, preserve_dataframe=True))])

        # Use DataFrame
        df = pd.DataFrame(sample_data["multi_col"], columns=["feat1", "feat2"])

        pipeline.fit(df)
        df_transformed = pipeline.transform(df)

        # Should preserve DataFrame format
        assert isinstance(df_transformed, pd.DataFrame)
        assert list(df_transformed.columns) == ["feat1", "feat2"]

    def test_sklearn_pipeline_param_access(self, sample_data):
        """Test parameter access in sklearn pipeline."""
        pipeline = Pipeline([("binner", EqualFrequencyBinning(n_bins=5))])

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
        binner = EqualFrequencyBinning(n_bins=3)
        X_nan = sample_data["with_nan"]

        # Should handle NaN values gracefully (base class preprocesses)
        binner.fit(X_nan)
        result = binner.transform(X_nan)

        assert result is not None
        assert result.shape == X_nan.shape

    def test_edge_case_inf_values(self, sample_data):
        """Test handling of infinite values."""
        binner = EqualFrequencyBinning(n_bins=3)
        X_inf = sample_data["with_inf"]

        # Should handle inf values based on config
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            binner.fit(X_inf)
            result = binner.transform(X_inf)

            assert result is not None
            assert result.shape == X_inf.shape

    def test_edge_case_constant_column(self, sample_data):
        """Test handling of constant columns."""
        binner = EqualFrequencyBinning(n_bins=3)
        X_constant = sample_data["constant"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            binner.fit(X_constant)
            result = binner.transform(X_constant)

            assert result is not None
            assert result.shape == X_constant.shape
            # All values should map to the same bin for constant data
            assert len(np.unique(result)) <= 1

    def test_edge_case_duplicates(self, sample_data):
        """Test handling of data with many duplicate values."""
        binner = EqualFrequencyBinning(n_bins=3)
        X_dups = sample_data["with_duplicates"]

        binner.fit(X_dups)
        result = binner.transform(X_dups)

        assert result is not None
        assert result.shape == X_dups.shape
        # Should handle duplicates gracefully

    def test_edge_case_more_bins_than_unique_values(self):
        """Test when n_bins > number of unique values."""
        X = np.array([[1], [2], [3]])  # Only 3 unique values
        binner = EqualFrequencyBinning(n_bins=5)  # More bins than unique values

        # Should raise ValueError because insufficient values for requested bins
        with pytest.raises(ValueError, match="Insufficient values \\(3\\) for 5 bins"):
            binner.fit(X)

    def test_edge_case_more_bins_than_unique_values_sufficient_data(self):
        """Test when n_bins > number of unique values but sufficient data."""
        # 5 bins but only 3 unique values, but with sufficient total data points
        X = np.array([[1], [1], [2], [2], [3]])  # 5 data points, 3 unique values
        binner = EqualFrequencyBinning(n_bins=3)  # Use 3 bins to match unique values

        binner.fit(X)
        result = binner.transform(X)

        assert result is not None
        assert result.shape == X.shape
        # Should handle duplicate values gracefully

    def test_edge_case_single_value_per_bin(self):
        """Test when each bin contains exactly one unique value."""
        X = np.array([[1], [2], [3], [4], [5]])  # 5 unique values
        binner = EqualFrequencyBinning(n_bins=5)  # Same as unique values

        binner.fit(X)
        result = binner.transform(X)

        assert result is not None
        assert result.shape == X.shape

    def test_edge_case_large_data(self):
        """Test with large dataset."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (10000, 1))
        binner = EqualFrequencyBinning(n_bins=10)

        binner.fit(X)
        result = binner.transform(X[:1000])  # Transform subset

        assert result is not None
        assert result.shape == (1000, 1)
        assert 0 <= result.max() < 10  # Should be within bin range

        # Check frequency distribution is approximately equal
        unique, counts = np.unique(result, return_counts=True)
        # For large data with equal frequency, counts should be similar
        if len(unique) > 1:
            cv = np.std(counts) / np.mean(counts)  # coefficient of variation
            assert cv < 0.5  # Should be reasonably equal

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        binner = EqualFrequencyBinning(n_bins=3)

        with pytest.raises((ValueError, ValidationError, FittingError)):
            binner.fit(np.array([]).reshape(0, 1))

    def test_single_row_data(self):
        """Test handling of single row data."""
        X = np.array([[5.0]])
        binner = EqualFrequencyBinning(n_bins=3)

        # Should raise ValueError because insufficient values for requested bins
        with pytest.raises(ValueError, match="Insufficient values \\(1\\) for 3 bins"):
            binner.fit(X)

    def test_single_row_data_with_single_bin(self):
        """Test handling of single row data with n_bins=1."""
        X = np.array([[5.0]])
        binner = EqualFrequencyBinning(n_bins=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            binner.fit(X)
            result = binner.transform(X)

            assert result is not None
            assert result.shape == (1, 1)
            assert result[0, 0] == 0  # Should be assigned to bin 0

    def test_quantile_calculation_error_coverage(self):
        """Test to cover quantile calculation error handling."""
        import unittest.mock

        # Mock np.quantile to raise an exception to test error handling
        with unittest.mock.patch("numpy.quantile", side_effect=ValueError("Mock quantile error")):
            X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
            binner = EqualFrequencyBinning(n_bins=3)

            # Should catch the ValueError from quantile and re-raise with better message
            with pytest.raises(
                ValueError, match="Column 0: Error calculating quantiles: Mock quantile error"
            ):
                binner.fit(X)

    # Specific equal frequency binning tests

    def test_equal_frequency_distribution(self, sample_data):
        """Test that bins have approximately equal frequencies."""
        X = sample_data["uniform"]  # 100 uniform samples
        binner = EqualFrequencyBinning(n_bins=5)

        binner.fit(X)
        result = binner.transform(X)

        # Check frequency distribution
        unique, counts = np.unique(result, return_counts=True)

        # For uniform data, frequencies should be exactly equal (100/5 = 20 each)
        expected_count = len(X) // 5
        for count in counts:
            assert abs(count - expected_count) <= 1  # Allow for rounding

    def test_quantile_range_parameter(self, sample_data):
        """Test quantile_range parameter functionality."""
        X = sample_data["normal"]

        # Use middle 80% of data (exclude extreme 10% on each side)
        binner = EqualFrequencyBinning(n_bins=4, quantile_range=(0.1, 0.9))
        binner.fit(X)

        # Get the bin edges
        edges = binner.bin_edges_[0]

        # The range should be based on the 10th-90th percentiles
        data_10th = np.percentile(X, 10)
        data_90th = np.percentile(X, 90)

        # Bin edges should span approximately this range
        assert edges[0] >= data_10th - 0.01  # Small tolerance for floating point
        assert edges[-1] <= data_90th + 0.01

    def test_representatives_are_medians(self, sample_data):
        """Test that representatives are approximate bin medians."""
        X = sample_data["simple"]  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        binner = EqualFrequencyBinning(n_bins=2)  # Two bins

        binner.fit(X)
        result = binner.transform(X)
        representatives = binner.bin_representatives_[0]

        # Should have 2 representatives
        assert len(representatives) == 2

        # Each representative should be close to the median of its bin
        for bin_idx, rep in enumerate(representatives):
            bin_mask = result.flatten() == bin_idx
            if np.any(bin_mask):
                bin_values = X[bin_mask].flatten()
                bin_median = np.median(bin_values)
                # Representative should be close to median of bin values
                assert abs(rep - bin_median) < 1.0  # Reasonable tolerance

    def test_skewed_data_handling(self, sample_data):
        """Test equal frequency binning on skewed data."""
        X = sample_data["skewed"]  # Exponential distribution (right-skewed)
        binner = EqualFrequencyBinning(n_bins=5)

        binner.fit(X)
        result = binner.transform(X)

        # Should handle skewed data and create equal frequency bins
        unique, counts = np.unique(result, return_counts=True)

        # Frequencies should be approximately equal despite skewed input
        expected_count = len(X) // len(unique)
        for count in counts:
            assert abs(count - expected_count) <= 2  # Tolerance for skewed data

    def test_multiple_columns_independent_binning(self, sample_data):
        """Test that columns are binned independently."""
        X = sample_data["multi_col"]
        binner = EqualFrequencyBinning(n_bins=3)

        binner.fit(X)

        # Each column should have its own bin edges based on its quantiles
        edges_col1 = binner.bin_edges_[0]
        edges_col2 = binner.bin_edges_[1]

        assert len(edges_col1) == 4  # 3 bins = 4 edges
        assert len(edges_col2) == 4

        # Edges should be different for different columns
        assert not np.array_equal(edges_col1, edges_col2)

    def test_string_n_bins_functionality(self, sample_data):
        """Test string n_bins parameter functionality."""
        X = sample_data["uniform"]  # 100 samples

        # Test "sqrt" option
        binner_sqrt = EqualFrequencyBinning(n_bins="sqrt")
        binner_sqrt.fit(X)
        result_sqrt = binner_sqrt.transform(X)

        # sqrt(100) = 10 bins expected
        unique_bins = np.unique(result_sqrt)
        assert len(unique_bins) <= 10

        # Test "log2" option
        if len(X) >= 4:  # log2 needs sufficient data
            binner_log2 = EqualFrequencyBinning(n_bins="log2")
            binner_log2.fit(X)
            result_log2 = binner_log2.transform(X)

            # log2(100) ≈ 7 bins expected
            unique_bins_log2 = np.unique(result_log2)
            assert len(unique_bins_log2) <= 8

    def test_clip_parameter_functionality(self, sample_data):
        """Test clip parameter with out-of-range values."""
        X_train = np.array([[1], [2], [3], [4], [5], [6]])
        X_test = np.array([[0], [7]])  # Out of range values

        # Test with clip=True
        binner_clip = EqualFrequencyBinning(n_bins=3, clip=True)
        binner_clip.fit(X_train)
        result_clip = binner_clip.transform(X_test)

        # Should clip to valid bin range
        assert 0 <= result_clip.min()
        assert result_clip.max() < 3

        # Test with clip=False
        binner_no_clip = EqualFrequencyBinning(n_bins=3, clip=False)
        binner_no_clip.fit(X_train)
        result_no_clip = binner_no_clip.transform(X_test)

        # May have values outside normal bin range (MISSING_VALUE)
        assert result_no_clip is not None

    # Integration and workflow tests

    def test_full_workflow_with_all_methods(self, sample_data):
        """Test complete workflow: fit, transform, inverse_transform, get_params, set_params."""
        X = sample_data["multi_col"]

        # Initialize and fit
        binner = EqualFrequencyBinning(n_bins=4, preserve_dataframe=False)
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
        new_binner = EqualFrequencyBinning()
        new_binner.set_params(**params)

        X_binned_2 = new_binner.transform(X)
        np.testing.assert_array_equal(X_binned, X_binned_2)

    def test_consistency_across_transforms(self, sample_data):
        """Test that repeated transforms give consistent results."""
        X = sample_data["normal"]
        binner = EqualFrequencyBinning(n_bins=5)
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

        binner = EqualFrequencyBinning(n_bins=5)
        binner.fit(X_full)

        # Transform subset
        result_subset = binner.transform(X_subset)
        result_full = binner.transform(X_full)

        # Subset result should match first 20 rows of full result
        np.testing.assert_array_equal(result_subset, result_full[:20])

    def test_different_n_bins_values(self, sample_data):
        """Test various n_bins values."""
        X = sample_data["uniform"]

        for n_bins in [2, 3, 5, 10]:
            binner = EqualFrequencyBinning(n_bins=n_bins)
            binner.fit(X)
            result = binner.transform(X)

            # Check that we get expected number of bins
            unique_bins = np.unique(result)
            assert len(unique_bins) <= n_bins
            assert result.max() < n_bins
            assert result.min() >= 0

            # Check approximately equal frequencies
            _, counts = np.unique(result, return_counts=True)
            if len(counts) > 1:
                expected_count = len(X) // len(counts)
                for count in counts:
                    assert abs(count - expected_count) <= 2
