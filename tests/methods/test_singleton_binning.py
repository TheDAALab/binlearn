"""
Comprehensive tests for SingletonBinning method covering all scenarios:
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
from binlearn.methods import SingletonBinning

# Skip polars tests if not available
polars_skip = pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")

# Skip pandas tests if not available
pandas_skip = pytest.mark.skipif(not hasattr(pd, "DataFrame"), reason="pandas not available")


class TestSingletonBinning:
    """Comprehensive test suite for SingletonBinning."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        return {
            "simple": np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1),
            "multi_col": np.array(
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]]
            ),
            "discrete": np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).reshape(-1, 1),
            "unique_values": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).reshape(
                -1, 1
            ),
            "with_nan": np.array([1.0, 2.0, np.nan, 4.0, 5.0]).reshape(-1, 1),
            "with_inf": np.array([1.0, 2.0, np.inf, 4.0, 5.0]).reshape(-1, 1),
            "constant": np.array([5.0, 5.0, 5.0, 5.0, 5.0]).reshape(-1, 1),
            "mixed_scale": np.array([0.1, 0.2, 0.3, 100.0, 200.0, 300.0]).reshape(-1, 1),
        }

    # Basic functionality tests

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        binner = SingletonBinning()
        assert binner.preserve_dataframe is False
        assert binner.fit_jointly is False
        assert binner.guidance_columns is None

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        binner = SingletonBinning(preserve_dataframe=True, fit_jointly=True)
        assert binner.preserve_dataframe is True
        assert binner.fit_jointly is True
        assert binner.guidance_columns is None  # Singleton doesn't use guidance

    # Input format tests with preserve_dataframe=False

    def test_numpy_input_preserve_false(self, sample_data):
        """Test with numpy input and preserve_dataframe=False."""
        binner = SingletonBinning(preserve_dataframe=False)

        # Fit and transform
        X_fit = sample_data["simple"]
        binner.fit(X_fit)
        X_transformed = binner.transform(X_fit)

        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape == X_fit.shape
        assert X_transformed.dtype == int

        # Each unique value should get its own bin
        unique_values = np.unique(X_fit[np.isfinite(X_fit)])
        unique_bins = np.unique(X_transformed)
        assert len(unique_bins) == len(unique_values)

        # Inverse transform
        X_inverse = binner.inverse_transform(X_transformed)
        assert isinstance(X_inverse, np.ndarray)
        assert X_inverse.shape == X_fit.shape
        # For singleton binning, inverse should match original exactly
        np.testing.assert_array_equal(X_inverse, X_fit)

    @pandas_skip
    def test_pandas_input_preserve_false(self, sample_data):
        """Test with pandas input and preserve_dataframe=False."""
        binner = SingletonBinning(preserve_dataframe=False)

        # Create pandas DataFrame
        df = pd.DataFrame(sample_data["discrete"], columns=["feature"])
        binner.fit(df)
        df_transformed = binner.transform(df)

        # Should return numpy array when preserve_dataframe=False
        assert isinstance(df_transformed, np.ndarray)
        assert df_transformed.shape == df.values.shape

        # Inverse transform
        df_inverse = binner.inverse_transform(df_transformed)
        assert isinstance(df_inverse, np.ndarray)
        # Values should be preserved exactly
        np.testing.assert_array_equal(df_inverse, df.values)

    @polars_skip
    def test_polars_input_preserve_false(self, sample_data):
        """Test with polars input and preserve_dataframe=False."""
        binner = SingletonBinning(preserve_dataframe=False)

        # Create polars DataFrame
        assert pl is not None
        df = pl.DataFrame({"feature": sample_data["discrete"].flatten()})
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
        binner = SingletonBinning(preserve_dataframe=True)

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

        # Values should be preserved exactly
        pd.testing.assert_frame_equal(df_inverse, df)

    @polars_skip
    def test_polars_input_preserve_true(self, sample_data):
        """Test with polars input and preserve_dataframe=True."""
        binner = SingletonBinning(preserve_dataframe=True)

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
        binner_original = SingletonBinning()
        X_fit = sample_data["multi_col"]
        binner_original.fit(X_fit)

        # Get parameters
        params = binner_original.get_params()

        # Create new binner and set parameters (reconstruct state)
        binner_reconstructed = SingletonBinning()
        binner_reconstructed.set_params(**params)

        # Test that transform works without fitting
        X_test = sample_data["multi_col"][:3]  # Subset for testing
        result_original = binner_original.transform(X_test)
        result_reconstructed = binner_reconstructed.transform(X_test)

        np.testing.assert_array_equal(result_original, result_reconstructed)

        # Test inverse transform
        inverse_original = binner_original.inverse_transform(result_original)
        inverse_reconstructed = binner_reconstructed.inverse_transform(result_reconstructed)

        np.testing.assert_array_equal(inverse_original, inverse_reconstructed)

    def test_reconstruction_via_constructor(self, sample_data):
        """Test fitted state reconstruction via constructor parameters."""
        # Fit original binner
        binner_original = SingletonBinning()
        X_fit = sample_data["unique_values"]
        binner_original.fit(X_fit)

        # Get parameters including fitted state
        params = binner_original.get_params()

        # Create new binner with constructor (reconstruct state)
        binner_reconstructed = SingletonBinning(**params)

        # Test that transform works without fitting
        X_test = sample_data["unique_values"]
        result_original = binner_original.transform(X_test)
        result_reconstructed = binner_reconstructed.transform(X_test)

        np.testing.assert_array_equal(result_original, result_reconstructed)

    def test_repeated_fitting_after_reconstruction(self, sample_data):
        """Test repeated fitting on reconstructed state."""
        # Original fitting
        binner = SingletonBinning()
        X_fit1 = sample_data["simple"]
        binner.fit(X_fit1)

        # Get and set params (reconstruction)
        params = binner.get_params()
        binner_new = SingletonBinning()
        binner_new.set_params(**params)

        # Refit with different data
        X_fit2 = sample_data["discrete"]
        binner_new.fit(X_fit2)

        # Should work fine
        result = binner_new.transform(X_fit2[:6])
        assert result is not None
        assert result.shape == (6, 1)

        # Another refit
        binner_new.fit(sample_data["mixed_scale"])
        result2 = binner_new.transform(sample_data["mixed_scale"][:4])
        assert result2 is not None

    def test_various_formats_after_reconstruction(self, sample_data):
        """Test various input formats after fitted state reconstruction."""
        # Original fitting with numpy
        binner_original = SingletonBinning(preserve_dataframe=True)
        X_numpy = sample_data["multi_col"]
        binner_original.fit(X_numpy)

        # Reconstruct
        params = binner_original.get_params()
        binner_reconstructed = SingletonBinning(**params)

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
        pipeline = Pipeline([("scaler", StandardScaler()), ("binner", SingletonBinning())])

        X = sample_data["discrete"]

        # Fit and transform
        pipeline.fit(X)
        X_transformed = pipeline.transform(X)

        assert X_transformed is not None
        assert X_transformed.shape == X.shape
        assert X_transformed.dtype == int

        # Test named steps access
        assert hasattr(pipeline.named_steps["binner"], "bin_spec_")

    def test_sklearn_pipeline_with_dataframes(self, sample_data):
        """Test sklearn pipeline with DataFrame inputs."""
        if not hasattr(pd, "DataFrame"):
            pytest.skip("pandas not available")

        # Create pipeline
        pipeline = Pipeline([("binner", SingletonBinning(preserve_dataframe=True))])

        # Use DataFrame
        df = pd.DataFrame(sample_data["multi_col"], columns=["feat1", "feat2"])

        pipeline.fit(df)
        df_transformed = pipeline.transform(df)

        # Should preserve DataFrame format
        assert isinstance(df_transformed, pd.DataFrame)
        assert list(df_transformed.columns) == ["feat1", "feat2"]

    def test_sklearn_pipeline_param_access(self, sample_data):
        """Test parameter access in sklearn pipeline."""
        pipeline = Pipeline([("binner", SingletonBinning())])

        # Test parameter access
        params = pipeline.get_params()
        assert "binner__preserve_dataframe" in params
        assert "binner__fit_jointly" in params

        # Test parameter setting
        pipeline.set_params(binner__preserve_dataframe=True)
        assert pipeline.named_steps["binner"].preserve_dataframe is True

    # Edge case tests

    def test_edge_case_nan_values(self, sample_data):
        """Test handling of NaN values."""
        binner = SingletonBinning()
        X_nan = sample_data["with_nan"]

        # Should handle NaN values gracefully (base class preprocesses)
        binner.fit(X_nan)
        result = binner.transform(X_nan)

        assert result is not None
        # NaN values should be handled by the base class
        assert result.shape == X_nan.shape

    def test_edge_case_inf_values(self, sample_data):
        """Test handling of infinite values."""
        binner = SingletonBinning()
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
        binner = SingletonBinning()
        X_constant = sample_data["constant"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings about constant data
            binner.fit(X_constant)
            result = binner.transform(X_constant)

            assert result is not None
            assert result.shape == X_constant.shape
            # All values should map to the same bin for constant data
            assert len(np.unique(result)) == 1

    def test_edge_case_single_unique_value(self):
        """Test when data has only one unique value."""
        X = np.array([[5], [5], [5], [5]])  # All same value
        binner = SingletonBinning()

        binner.fit(X)
        result = binner.transform(X)

        assert result is not None
        assert result.shape == X.shape
        # Should create exactly one bin
        assert len(np.unique(result)) == 1
        assert result.max() == result.min()  # All same bin

    def test_edge_case_many_unique_values(self):
        """Test with many unique values."""
        X = np.arange(100).reshape(-1, 1).astype(float)  # 100 unique values
        binner = SingletonBinning()

        binner.fit(X)
        result = binner.transform(X)

        assert result is not None
        assert result.shape == X.shape
        # Should create 100 different bins
        unique_bins = np.unique(result)
        assert len(unique_bins) == 100

    def test_edge_case_very_small_values(self):
        """Test with very small numeric values."""
        X = np.array([[1e-10], [2e-10], [3e-10]])
        binner = SingletonBinning()

        binner.fit(X)
        result = binner.transform(X)

        assert result is not None
        assert result.shape == X.shape
        # Should handle small values correctly
        assert len(np.unique(result)) == 3

    def test_edge_case_very_large_values(self):
        """Test with very large numeric values."""
        X = np.array([[1e10], [2e10], [3e10]])
        binner = SingletonBinning()

        binner.fit(X)
        result = binner.transform(X)

        assert result is not None
        assert result.shape == X.shape
        # Should handle large values correctly
        assert len(np.unique(result)) == 3

    def test_edge_case_negative_values(self):
        """Test with negative values."""
        X = np.array([[-1.0], [-2.0], [-3.0], [1.0], [2.0]])
        binner = SingletonBinning()

        binner.fit(X)
        result = binner.transform(X)

        assert result is not None
        assert result.shape == X.shape
        # Should handle negative values correctly
        assert len(np.unique(result)) == 5

        # Inverse should preserve exact values
        X_inverse = binner.inverse_transform(result)
        np.testing.assert_array_equal(X_inverse, X)

    def test_edge_case_zero_values(self):
        """Test with zero values."""
        X = np.array([[0.0], [0.0], [1.0], [2.0]])
        binner = SingletonBinning()

        binner.fit(X)
        result = binner.transform(X)

        assert result is not None
        assert result.shape == X.shape
        # Should have 3 unique bins (0, 1, 2)
        assert len(np.unique(result)) == 3

    def test_edge_case_large_data(self):
        """Test with large dataset."""
        np.random.seed(42)
        X = np.random.randint(0, 50, (10000, 1)).astype(float)  # Random integers as floats
        binner = SingletonBinning()

        binner.fit(X)
        result = binner.transform(X[:1000])  # Transform subset

        assert result is not None
        assert result.shape == (1000, 1)

    def test_nan_matching_logic_directly(self):
        """Test NaN matching logic in _match_values_to_bin method."""
        # Create data where NaN is included in the training data (finite values)
        # This is tricky because SingletonBinning filters out NaN during fitting
        # But we need to test the NaN matching logic in _match_values_to_bin

        # Let's create a test that forces NaN to be a bin value through the base class
        X_train = np.array([[1.0], [2.0], [3.0]])  # No NaN in training
        binner = SingletonBinning()
        binner.fit(X_train)

        # Test the NaN matching logic by creating a scenario where NaN is a bin_value
        # Since we can't easily inject NaN as a bin value, let's test with a different approach

        # Create data with mixed NaN and regular values for transform
        X_test = np.array([[1.0], [np.nan], [2.0], [np.nan]])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = binner.transform(X_test)

            assert result is not None
            assert result.shape == X_test.shape

    def test_match_values_to_bin_direct_coverage(self):
        """Test to ensure _match_values_to_bin method is covered."""
        # Create a simple case that will definitely call _match_values_to_bin
        X = np.array([[1.0], [2.0], [3.0]])
        binner = SingletonBinning()
        binner.fit(X)

        # Transform the same data - this should call _match_values_to_bin
        result = binner.transform(X)
        assert result is not None
        assert len(np.unique(result)) == 3

        # Transform with exact same values to ensure matching
        X_exact = np.array([[1.0], [2.0], [3.0]])
        result_exact = binner.transform(X_exact)
        np.testing.assert_array_equal(result, result_exact)

        # Transform subset to ensure all bin values are matched
        X_subset = np.array([[2.0]])
        result_subset = binner.transform(X_subset)
        assert result_subset[0, 0] == result[1, 0]  # Should match the bin for 2.0

    def test_match_values_to_bin_edge_cases(self):
        """Test edge cases in _match_values_to_bin method including type checking."""
        binner = SingletonBinning()

        # Test with integer data to exercise int type path
        X_int = np.array([[1], [2], [3]], dtype=np.int32)
        binner.fit(X_int)

        # Transform integer data
        result = binner.transform(X_int)
        assert result is not None
        assert len(np.unique(result)) == 3

        # Test with float data to exercise float type path
        X_float = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        binner_float = SingletonBinning()
        binner_float.fit(X_float)

        result_float = binner_float.transform(X_float)
        assert result_float is not None
        assert len(np.unique(result_float)) == 3

        # Test with mixed data including potential edge case values
        X_mixed = np.array([[0.0], [1.0], [-1.0], [1e-10], [1e10]])
        binner_mixed = SingletonBinning()
        binner_mixed.fit(X_mixed)

        result_mixed = binner_mixed.transform(X_mixed)
        assert result_mixed is not None
        assert len(np.unique(result_mixed)) == 5

    def test_edge_case_all_nan_data(self):
        """Test with data containing only NaN values."""
        X = np.array([[np.nan], [np.nan], [np.nan]])
        binner = SingletonBinning()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings about NaN data
            binner.fit(X)

            # Should create a default bin for all NaN case
            assert hasattr(binner, "bin_spec_")
            assert hasattr(binner, "bin_representatives_")

            # Transform should handle NaN data
            result = binner.transform(X)
            assert result is not None
            assert result.shape == X.shape

    def test_edge_case_all_inf_data(self):
        """Test with data containing only infinite values."""
        X = np.array([[np.inf], [np.inf], [-np.inf]])
        binner = SingletonBinning()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings about inf data
            binner.fit(X)

            # Should create a default bin for all inf case
            assert hasattr(binner, "bin_spec_")
            assert hasattr(binner, "bin_representatives_")

            # Transform should handle inf data
            result = binner.transform(X)
            assert result is not None
            assert result.shape == X.shape

    def test_edge_case_nan_in_bin_values(self):
        """Test handling of NaN values in transform when NaN is a bin value."""
        # Create data with NaN as one of the values
        X = np.array([[1.0], [2.0], [np.nan], [3.0]])
        binner = SingletonBinning()

        binner.fit(X)

        # Transform should handle NaN correctly
        X_test = np.array([[1.0], [np.nan], [2.0]])
        result = binner.transform(X_test)

        assert result is not None
        assert result.shape == X_test.shape

        # NaN values should be handled by the matching logic
        # The exact behavior depends on the base class but should not error

    # Specific singleton binning tests

    def test_singleton_property_unique_bins(self, sample_data):
        """Test that each unique value gets its own bin."""
        X = sample_data["discrete"]  # Has duplicate values
        binner = SingletonBinning()

        binner.fit(X)
        result = binner.transform(X)

        # Get unique values from original data
        unique_original = np.unique(X[np.isfinite(X.flatten())])
        unique_bins = np.unique(result)

        # Number of bins should equal number of unique values
        assert len(unique_bins) == len(unique_original)

    def test_singleton_exact_value_preservation(self, sample_data):
        """Test that inverse transform preserves exact original values."""
        X = sample_data["mixed_scale"]  # Has values of different scales
        binner = SingletonBinning()

        binner.fit(X)
        result = binner.transform(X)
        X_recovered = binner.inverse_transform(result)

        # Should recover exactly the same values
        np.testing.assert_array_equal(X_recovered, X)

    def test_singleton_bin_representatives_are_values(self, sample_data):
        """Test that bin representatives are the original unique values."""
        X = sample_data["unique_values"]
        binner = SingletonBinning()

        binner.fit(X)

        # Get bin representatives for the column
        representatives = binner.bin_representatives_[0]  # Column 0
        unique_original = np.unique(X.flatten())

        # Representatives should be the same as unique values
        np.testing.assert_array_equal(sorted(representatives), sorted(unique_original))

    def test_singleton_with_duplicate_values(self):
        """Test singleton binning with repeated values."""
        X = np.array([[1.0], [2.0], [1.0], [3.0], [2.0], [1.0]])
        binner = SingletonBinning()

        binner.fit(X)
        result = binner.transform(X)

        # Should have 3 unique bins for values 1, 2, 3
        unique_bins = np.unique(result)
        assert len(unique_bins) == 3

        # Same input values should map to same bins
        assert result[0] == result[2] == result[5]  # All 1.0 values
        assert result[1] == result[4]  # All 2.0 values

    def test_multiple_columns_independent_binning(self, sample_data):
        """Test that columns are binned independently."""
        X = sample_data["multi_col"]  # Different unique values per column
        binner = SingletonBinning()

        binner.fit(X)

        # Each column should have its own unique values
        representatives_col1 = binner.bin_representatives_[0]
        representatives_col2 = binner.bin_representatives_[1]

        # Representatives should reflect the unique values in each column
        unique_col1 = np.unique(X[:, 0])
        unique_col2 = np.unique(X[:, 1])

        np.testing.assert_array_equal(sorted(representatives_col1), sorted(unique_col1))
        np.testing.assert_array_equal(sorted(representatives_col2), sorted(unique_col2))

    def test_fit_jointly_parameter(self, sample_data):
        """Test fit_jointly parameter behavior."""
        X = sample_data["multi_col"]

        # Test with fit_jointly=True
        binner_joint = SingletonBinning(fit_jointly=True)
        binner_joint.fit(X)

        # Test with fit_jointly=False
        binner_indep = SingletonBinning(fit_jointly=False)
        binner_indep.fit(X)

        # For SingletonBinning, results should be the same since each unique value
        # gets its own bin regardless of fit_jointly setting
        result_joint = binner_joint.transform(X[:3])
        result_indep = binner_indep.transform(X[:3])

        assert result_joint is not None
        assert result_indep is not None

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        binner = SingletonBinning()

        # Should handle empty data gracefully by creating a default bin
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings about empty data
            binner.fit(np.array([]).reshape(0, 1))

            # Should create a default bin
            assert hasattr(binner, "bin_spec_")
            assert hasattr(binner, "bin_representatives_")

            # Transform should work (though may produce warnings for empty data)
            result = binner.transform(np.array([[1.0]]))  # Test with some data
            assert result is not None

    def test_single_row_data(self):
        """Test handling of single row data."""
        X = np.array([[5.0]])
        binner = SingletonBinning()

        binner.fit(X)
        result = binner.transform(X)

        assert result is not None
        assert result.shape == (1, 1)
        assert result[0, 0] == 0  # Should be bin 0

        # Inverse should recover original value
        X_recovered = binner.inverse_transform(result)
        np.testing.assert_array_equal(X_recovered, X)

    # Integration and workflow tests

    def test_full_workflow_with_all_methods(self, sample_data):
        """Test complete workflow: fit, transform, inverse_transform, get_params, set_params."""
        X = sample_data["multi_col"]

        # Initialize and fit
        binner = SingletonBinning(preserve_dataframe=False)
        binner.fit(X)

        # Transform
        X_binned = binner.transform(X)
        assert X_binned.shape == X.shape
        assert X_binned.dtype == int

        # Inverse transform should preserve exact values
        X_recovered = binner.inverse_transform(X_binned)
        assert X_recovered.shape == X.shape
        assert X_recovered.dtype == float
        np.testing.assert_array_equal(X_recovered, X)

        # Get params
        params = binner.get_params()
        assert "preserve_dataframe" in params
        assert "bin_spec" in params
        assert "bin_representatives" in params

        # Set params and verify
        new_binner = SingletonBinning()
        new_binner.set_params(**params)

        X_binned_2 = new_binner.transform(X)
        np.testing.assert_array_equal(X_binned, X_binned_2)

    def test_consistency_across_transforms(self, sample_data):
        """Test that repeated transforms give consistent results."""
        X = sample_data["discrete"]
        binner = SingletonBinning()
        binner.fit(X)

        # Multiple transforms should give same result
        result1 = binner.transform(X)
        result2 = binner.transform(X)
        result3 = binner.transform(X)

        np.testing.assert_array_equal(result1, result2)
        np.testing.assert_array_equal(result2, result3)

    def test_partial_data_transform(self, sample_data):
        """Test transforming data subset after fitting on full data."""
        X_full = sample_data["unique_values"]  # 10 samples
        X_subset = X_full[:5]  # 5 samples

        binner = SingletonBinning()
        binner.fit(X_full)

        # Transform subset
        result_subset = binner.transform(X_subset)
        result_full = binner.transform(X_full)

        # Subset result should match first 5 rows of full result
        np.testing.assert_array_equal(result_subset, result_full[:5])

    def test_different_data_types_consistency(self):
        """Test consistency with different numeric data types."""
        # Test with different numpy dtypes
        X_int = np.array([[1], [2], [3]], dtype=np.int32)
        X_float32 = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        X_float64 = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)

        binner_int = SingletonBinning()
        binner_float32 = SingletonBinning()
        binner_float64 = SingletonBinning()

        binner_int.fit(X_int)
        binner_float32.fit(X_float32)
        binner_float64.fit(X_float64)

        result_int = binner_int.transform(X_int)
        result_float32 = binner_float32.transform(X_float32)
        result_float64 = binner_float64.transform(X_float64)

        # All should produce the same binning pattern
        np.testing.assert_array_equal(result_int, result_float32)
        np.testing.assert_array_equal(result_float32, result_float64)

    def test_out_of_sample_data_handling(self):
        """Test behavior with out-of-sample data during transform."""
        # Fit on limited data
        X_fit = np.array([[1.0], [2.0], [3.0]])
        binner = SingletonBinning()
        binner.fit(X_fit)

        # Transform with new unseen values
        X_new = np.array([[1.0], [2.0], [4.0], [5.0]])  # 4.0 and 5.0 are new

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # May generate warnings for unseen values
            result = binner.transform(X_new)

            assert result is not None
            assert result.shape == X_new.shape

            # Known values should map to their original bins
            # New values behavior depends on base class implementation
            # but the transform should complete without error

    def test_reconstruction_preserves_exact_behavior(self, sample_data):
        """Test that reconstructed binner behaves exactly like original."""
        X = sample_data["mixed_scale"]

        # Create and fit original
        binner_original = SingletonBinning()
        binner_original.fit(X)

        # Transform and get results
        result_original = binner_original.transform(X)
        inverse_original = binner_original.inverse_transform(result_original)

        # Reconstruct via get_params/set_params
        params = binner_original.get_params()
        binner_reconstructed = SingletonBinning()
        binner_reconstructed.set_params(**params)

        # Test identical behavior
        result_reconstructed = binner_reconstructed.transform(X)
        inverse_reconstructed = binner_reconstructed.inverse_transform(result_reconstructed)

        np.testing.assert_array_equal(result_original, result_reconstructed)
        np.testing.assert_array_equal(inverse_original, inverse_reconstructed)

        # Test with new data
        X_new = X[:3]
        result_orig_new = binner_original.transform(X_new)
        result_recon_new = binner_reconstructed.transform(X_new)

        np.testing.assert_array_equal(result_orig_new, result_recon_new)
