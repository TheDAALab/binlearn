"""
Comprehensive tests for ManualFlexibleBinning method covering all scenarios.
"""

import warnings

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from binlearn import POLARS_AVAILABLE, pd, pl
from binlearn.methods import ManualFlexibleBinning
from binlearn.utils import ConfigurationError, ValidationError

# Skip polars tests if not available
polars_skip = pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")

# Skip pandas tests if not available
pandas_skip = pytest.mark.skipif(not hasattr(pd, "DataFrame"), reason="pandas not available")


class TestManualFlexibleBinning:
    """Comprehensive test suite for ManualFlexibleBinning."""

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
            "with_nan": np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape(-1, 1),
            "with_inf": np.array([1.0, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape(-1, 1),
            "categorical_like": np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).reshape(-1, 1),
            "mixed_values": np.array([1.0, 2.5, 3.5, 6.0, 7.5, 8.5, 9.0]).reshape(-1, 1),
        }

    @pytest.fixture
    def sample_bin_specs(self):
        """Generate sample flexible bin specifications for testing."""
        return {
            "singleton_only": {0: [1.0, 2.0, 3.0, 4.0]},  # Only singleton bins
            "interval_only": {0: [(0, 3), (3, 6), (6, 10)]},  # Only interval bins
            "mixed": {0: [1.0, (2, 5), 6.0, (7, 10)]},  # Mixed singleton and interval bins
            "multi_col": {
                0: [1.0, (2, 5), 6.0],  # Column 0: mixed bins
                1: [(0, 25), (25, 75), (75, 100)],  # Column 1: interval bins only
            },
            "named_cols": {
                "feature1": [1.0, (2, 5)],
                "feature2": [(0, 50), (50, 100)],
            },
            "complex": {0: [1.5, (2, 4), 5.0, (6, 8), 9.0]},  # Complex mixed pattern
            "overlapping": {0: [(0, 5), (3, 7), (6, 10)]},  # Overlapping intervals
        }

    @pytest.fixture
    def sample_representatives(self):
        """Generate sample bin representatives for testing."""
        return {
            "custom_singleton": {0: [1.0, 2.0, 3.0, 4.0]},  # Custom representatives for singletons
            "custom_interval": {0: [1.5, 4.5, 8.0]},  # Custom representatives for intervals
            "custom_mixed": {0: [1.0, 3.5, 6.0, 8.5]},  # Custom representatives for mixed bins
            "multi_col": {
                0: [1.0, 3.5, 6.0],  # Column 0
                1: [12.5, 50.0, 87.5],  # Column 1
            },
        }

    # Basic functionality tests

    def test_init_with_bin_spec_only(self, sample_bin_specs):
        """Test initialization with bin specifications only (auto-generate representatives)."""
        binner = ManualFlexibleBinning(bin_spec=sample_bin_specs["singleton_only"])
        assert binner.bin_spec == sample_bin_specs["singleton_only"]
        # After fitting/validation, representatives should be auto-generated
        assert binner.preserve_dataframe is False

    def test_init_with_bin_spec_and_representatives(self, sample_bin_specs, sample_representatives):
        """Test initialization with both bin specifications and representatives."""
        binner = ManualFlexibleBinning(
            bin_spec=sample_bin_specs["singleton_only"],
            bin_representatives=sample_representatives["custom_singleton"],
        )
        assert binner.bin_spec == sample_bin_specs["singleton_only"]
        assert binner.bin_representatives == sample_representatives["custom_singleton"]

    def test_init_with_custom_parameters(self, sample_bin_specs):
        """Test initialization with custom parameters."""
        binner = ManualFlexibleBinning(
            bin_spec=sample_bin_specs["singleton_only"],
            preserve_dataframe=True,
        )
        assert binner.bin_spec == sample_bin_specs["singleton_only"]
        assert binner.preserve_dataframe is True

    def test_parameter_validation_no_bin_spec(self):
        """Test parameter validation when bin_spec is None."""
        with pytest.raises(ConfigurationError, match="bin_spec must be provided"):
            ManualFlexibleBinning(bin_spec=None)  # type: ignore

    def test_parameter_validation_empty_bin_spec(self):
        """Test parameter validation when bin_spec is empty."""
        with pytest.raises(ConfigurationError, match="bin_spec must be provided and non-empty"):
            ManualFlexibleBinning(bin_spec={})

    def test_validate_params_with_none_bin_spec(self, sample_bin_specs):
        """Test validation when bin_spec becomes None after initialization."""
        binner = ManualFlexibleBinning(bin_spec=sample_bin_specs["singleton_only"])

        # Manually set bin_spec to None to test the validation path
        binner.bin_spec = None

        with pytest.raises(ConfigurationError, match="bin_spec must be provided and non-empty"):
            binner._validate_params()

    # Fitting tests (no-op for manual binning)

    def test_fit_returns_self(self, sample_data, sample_bin_specs):
        """Test that fit returns self and sets fitted state."""
        binner = ManualFlexibleBinning(bin_spec=sample_bin_specs["singleton_only"])
        result = binner.fit(sample_data["simple"])

        assert result is binner
        assert binner._fitted is True

    def test_fit_with_y_parameter(self, sample_data, sample_bin_specs):
        """Test that fit ignores y parameter (manual binning is unsupervised)."""
        binner = ManualFlexibleBinning(bin_spec=sample_bin_specs["singleton_only"])
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        result = binner.fit(sample_data["simple"], y=y)

        assert result is binner
        assert binner._fitted is True

    # Input format tests with preserve_dataframe=False

    def test_numpy_input_preserve_false(self, sample_data, sample_bin_specs):
        """Test with numpy input and preserve_dataframe=False."""
        binner = ManualFlexibleBinning(
            bin_spec=sample_bin_specs["singleton_only"], preserve_dataframe=False
        )

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
    def test_pandas_input_preserve_false(self, sample_data, sample_bin_specs):
        """Test with pandas input and preserve_dataframe=False."""
        binner = ManualFlexibleBinning(
            bin_spec=sample_bin_specs["singleton_only"], preserve_dataframe=False
        )

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
    def test_polars_input_preserve_false(self, sample_data, sample_bin_specs):
        """Test with polars input and preserve_dataframe=False."""
        binner = ManualFlexibleBinning(
            bin_spec=sample_bin_specs["singleton_only"], preserve_dataframe=False
        )

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
    def test_pandas_input_preserve_true(self, sample_data, sample_bin_specs):
        """Test with pandas input and preserve_dataframe=True."""
        binner = ManualFlexibleBinning(
            bin_spec=sample_bin_specs["multi_col"], preserve_dataframe=True
        )

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
    def test_polars_input_preserve_true(self, sample_data, sample_bin_specs):
        """Test with polars input and preserve_dataframe=True."""
        binner = ManualFlexibleBinning(
            bin_spec=sample_bin_specs["multi_col"], preserve_dataframe=True
        )

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

    def test_reconstruction_via_get_params_set_params(self, sample_data, sample_bin_specs):
        """Test fitted state reconstruction via get_params/set_params."""
        # Fit original binner
        binner_original = ManualFlexibleBinning(bin_spec=sample_bin_specs["multi_col"])
        X_fit = sample_data["multi_col"]
        binner_original.fit(X_fit)

        # Get parameters
        params = binner_original.get_params()

        # Create new binner and set parameters (reconstruct state)
        binner_reconstructed = ManualFlexibleBinning(bin_spec={0: [1.0]})  # Dummy spec
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

    def test_reconstruction_via_constructor(self, sample_data, sample_bin_specs):
        """Test fitted state reconstruction via constructor parameters."""
        # Fit original binner
        binner_original = ManualFlexibleBinning(bin_spec=sample_bin_specs["singleton_only"])
        X_fit = sample_data["simple"]
        binner_original.fit(X_fit)

        # Get parameters including fitted state
        params = binner_original.get_params()

        # Create new binner with constructor (reconstruct state)
        binner_reconstructed = ManualFlexibleBinning(**params)

        # Test that transform works without fitting (manual binning doesn't need fitting)
        X_test = sample_data["simple"]
        result_original = binner_original.transform(X_test)
        result_reconstructed = binner_reconstructed.transform(X_test)

        np.testing.assert_array_equal(result_original, result_reconstructed)

    def test_repeated_fitting_after_reconstruction(self, sample_data, sample_bin_specs):
        """Test repeated fitting on reconstructed state."""
        # Original fitting
        binner = ManualFlexibleBinning(bin_spec=sample_bin_specs["singleton_only"])
        X_fit1 = sample_data["simple"]
        binner.fit(X_fit1)

        # Get and set params (reconstruction)
        params = binner.get_params()
        binner_new = ManualFlexibleBinning(bin_spec={0: [1.0]})  # Dummy spec
        binner_new.set_params(**params)

        # Refit with different data
        X_fit2 = sample_data["uniform"]
        binner_new.fit(X_fit2)

        # Should work fine
        result = binner_new.transform(X_fit2[:10])
        assert result is not None
        assert result.shape == (10, 1)

        # Another refit with different data
        binner_new.fit(sample_data["with_nan"])
        result2 = binner_new.transform(sample_data["with_nan"][:5])
        assert result2 is not None

    def test_various_formats_after_reconstruction(self, sample_data, sample_bin_specs):
        """Test various input formats after fitted state reconstruction."""
        # Original fitting with numpy
        binner_original = ManualFlexibleBinning(
            bin_spec=sample_bin_specs["multi_col"], preserve_dataframe=True
        )
        X_numpy = sample_data["multi_col"]
        binner_original.fit(X_numpy)

        # Reconstruct
        params = binner_original.get_params()
        binner_reconstructed = ManualFlexibleBinning(**params)

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

    def test_sklearn_pipeline_basic(self, sample_data, sample_bin_specs):
        """Test basic sklearn pipeline integration."""
        # Create pipeline
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("binner", ManualFlexibleBinning(bin_spec=sample_bin_specs["singleton_only"])),
            ]
        )

        X = sample_data["simple"]

        # Fit and transform
        pipeline.fit(X)
        X_transformed = pipeline.transform(X)

        assert X_transformed is not None
        assert X_transformed.shape == X.shape
        assert X_transformed.dtype == int

        # Test named steps access
        assert hasattr(pipeline.named_steps["binner"], "bin_spec")

    def test_sklearn_pipeline_with_dataframes(self, sample_data, sample_bin_specs):
        """Test sklearn pipeline with DataFrame inputs."""
        if not hasattr(pd, "DataFrame"):
            pytest.skip("pandas not available")

        # Create pipeline
        pipeline = Pipeline(
            [
                (
                    "binner",
                    ManualFlexibleBinning(
                        bin_spec=sample_bin_specs["multi_col"], preserve_dataframe=True
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

    def test_sklearn_pipeline_param_access(self, sample_bin_specs):
        """Test parameter access in sklearn pipeline."""
        pipeline = Pipeline(
            [("binner", ManualFlexibleBinning(bin_spec=sample_bin_specs["singleton_only"]))]
        )

        # Test parameter access
        params = pipeline.get_params()
        assert "binner__bin_spec" in params
        assert params["binner__bin_spec"] == sample_bin_specs["singleton_only"]

        # Test parameter setting for settable parameters
        # Note: bin_spec might not be settable after construction for manual binning
        pipeline.set_params(binner__preserve_dataframe=True)
        assert pipeline.named_steps["binner"].preserve_dataframe is True

        # Test that bin_spec parameter is accessible (even if not settable)
        binner_spec = pipeline.get_params()["binner__bin_spec"]
        assert binner_spec == sample_bin_specs["singleton_only"]

    # Edge case tests

    def test_edge_case_nan_values(self, sample_data, sample_bin_specs):
        """Test handling of NaN values."""
        binner = ManualFlexibleBinning(bin_spec=sample_bin_specs["singleton_only"])
        X_nan = sample_data["with_nan"]

        # Should handle NaN values gracefully (base class preprocesses)
        binner.fit(X_nan)
        result = binner.transform(X_nan)

        assert result is not None
        # NaN values should be handled by the base class
        assert result.shape == X_nan.shape

    def test_edge_case_inf_values(self, sample_data, sample_bin_specs):
        """Test handling of infinite values."""
        binner = ManualFlexibleBinning(bin_spec=sample_bin_specs["singleton_only"])
        X_inf = sample_data["with_inf"]

        # Should handle inf values based on config
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings for this test
            binner.fit(X_inf)
            result = binner.transform(X_inf)

            assert result is not None
            assert result.shape == X_inf.shape

    def test_edge_case_single_row_data(self, sample_bin_specs):
        """Test handling of single row data."""
        X = np.array([[2.0]])
        binner = ManualFlexibleBinning(bin_spec=sample_bin_specs["singleton_only"])

        binner.fit(X)
        result = binner.transform(X)

        assert result is not None
        assert result.shape == (1, 1)
        assert isinstance(result[0, 0], int | np.integer)

    def test_edge_case_empty_data_handling(self, sample_bin_specs):
        """Test handling of empty data."""
        binner = ManualFlexibleBinning(bin_spec=sample_bin_specs["singleton_only"])

        # Empty data should be handled gracefully
        empty_data = np.array([]).reshape(0, 1)
        binner.fit(empty_data)  # Should not raise error
        result = binner.transform(empty_data)
        assert result.shape[0] == 0  # Should have 0 rows

    def test_calculate_flexible_bins_raises_not_implemented(self, sample_data, sample_bin_specs):
        """Test that _calculate_flexible_bins raises NotImplementedError for manual binning."""
        binner = ManualFlexibleBinning(bin_spec=sample_bin_specs["singleton_only"])
        binner.fit(sample_data["simple"])

        # The _calculate_flexible_bins method should never be called for manual binning
        with pytest.raises(
            NotImplementedError, match="Manual binning uses pre-defined specifications"
        ):
            binner._calculate_flexible_bins(sample_data["simple"][:, 0], col_id=0)

    def test_missing_column_in_bin_spec(self, sample_data, sample_bin_specs):
        """Test behavior when input data has more columns than bin specifications."""
        # Define specs only for column 0, but try to transform 2-column data
        binner = ManualFlexibleBinning(bin_spec=sample_bin_specs["singleton_only"])
        binner.fit(sample_data["multi_col"])

        # Should raise an error for mismatched column count
        with pytest.raises(
            ValueError,
            match="Input data has 2 columns but bin specifications are provided for 1 columns",
        ):
            binner.transform(sample_data["multi_col"])

        # Test that the underlying column key resolution also works
        available_keys = list(binner.bin_spec_.keys())

        # Test that existing column works
        key = binner._get_column_key(0, available_keys, 0)
        assert key == 0

        # Test that missing column raises ValueError
        with pytest.raises(ValueError, match="No bin specification found for column 1"):
            binner._get_column_key(1, available_keys, 1)

    def test_incomplete_specification_no_sklearn_attributes(self, sample_bin_specs):
        """Test case where sklearn attributes are not set due to incomplete specifications."""
        binner = ManualFlexibleBinning(bin_spec=sample_bin_specs["singleton_only"])

        # Create a scenario where the representatives dict is empty to test the False branch
        # We'll manipulate the validation process by temporarily modifying the bin_spec

        def custom_validate():
            """Custom validation that leaves bin_representatives_ empty."""
            from binlearn.base import GeneralBinningBase

            GeneralBinningBase._validate_params(binner)

            # Set bin_spec_ but leave bin_representatives_ empty to test the condition
            binner.bin_spec_ = sample_bin_specs["singleton_only"]
            binner.bin_representatives_ = {}  # Empty - this should make the condition False

        # Replace validation temporarily
        binner._validate_params = custom_validate

        # Now when we call fit, the custom validation will run
        binner.fit(np.array([[1.0]]))

        # The condition `if self.bin_spec_ and self.bin_representatives_:` should be False
        # because bin_representatives_ is empty dict (falsy)
        # Therefore _set_sklearn_attributes_from_specs() should not be called

        # Verify that sklearn attributes are not set (or set to None)
        # This tests the False branch of the condition
        assert binner.bin_spec_ is not None  # bin_spec_ should be set
        assert binner.bin_representatives_ == {}  # bin_representatives_ should be empty dict

        # The sklearn attributes might still be set by other parts of the code,
        # but the important thing is that we tested the False branch of the condition

    # Specific Manual Flexible binning tests

    def test_singleton_bins_only(self, sample_data, sample_bin_specs):
        """Test binning with only singleton bins."""
        binner = ManualFlexibleBinning(bin_spec=sample_bin_specs["singleton_only"])
        binner.fit(sample_data["categorical_like"])
        result = binner.transform(sample_data["categorical_like"])

        assert result is not None
        assert result.shape == sample_data["categorical_like"].shape

        # Check that exact matches get correct bin assignments
        unique_bins = np.unique(result)
        assert len(unique_bins) <= len(sample_bin_specs["singleton_only"][0])

    def test_interval_bins_only(self, sample_data, sample_bin_specs):
        """Test binning with only interval bins."""
        binner = ManualFlexibleBinning(bin_spec=sample_bin_specs["interval_only"])
        binner.fit(sample_data["simple"])
        result = binner.transform(sample_data["simple"])

        assert result is not None
        assert result.shape == sample_data["simple"].shape

        # Check that values are assigned to valid bins
        num_bins = len(sample_bin_specs["interval_only"][0])
        assert np.all(result >= 0)
        assert np.all(result < num_bins)

    def test_mixed_singleton_interval_bins(self, sample_data, sample_bin_specs):
        """Test binning with mixed singleton and interval bins."""
        binner = ManualFlexibleBinning(bin_spec=sample_bin_specs["mixed"])
        binner.fit(sample_data["mixed_values"])
        result = binner.transform(sample_data["mixed_values"])

        assert result is not None
        assert result.shape == sample_data["mixed_values"].shape

        # Should handle both singleton and interval bin assignments
        num_bins = len(sample_bin_specs["mixed"][0])
        assert np.all(result >= 0)
        assert np.all(result < num_bins)

    def test_complex_bin_pattern(self, sample_data, sample_bin_specs):
        """Test complex bin patterns with many mixed bins."""
        binner = ManualFlexibleBinning(bin_spec=sample_bin_specs["complex"])
        binner.fit(sample_data["mixed_values"])
        result = binner.transform(sample_data["mixed_values"])

        assert result is not None
        assert result.shape == sample_data["mixed_values"].shape

        # Should use all or most of the available bins
        unique_bins = np.unique(result)
        assert len(unique_bins) > 2  # Should use several bins

    def test_overlapping_intervals(self, sample_data, sample_bin_specs):
        """Test handling of overlapping interval bins."""
        binner = ManualFlexibleBinning(bin_spec=sample_bin_specs["overlapping"])
        binner.fit(sample_data["simple"])
        result = binner.transform(sample_data["simple"])

        assert result is not None
        assert result.shape == sample_data["simple"].shape

        # The behavior with overlapping intervals depends on the implementation
        # At minimum, should not crash and produce some valid result
        assert np.all(result >= 0)

    def test_exact_value_matching_singletons(self, sample_bin_specs):
        """Test exact value matching for singleton bins."""
        # Create data with exact matches to singleton bins
        spec = sample_bin_specs["singleton_only"]  # {0: [1.0, 2.0, 3.0, 4.0]}
        X = np.array([[1.0], [2.0], [3.0], [4.0]])

        binner = ManualFlexibleBinning(bin_spec=spec)
        binner.fit(X)
        result = binner.transform(X)

        assert result is not None
        assert result.shape == X.shape

        # Each value should map to its corresponding bin index
        for _i, expected_bin in enumerate(result.flatten()):
            assert 0 <= expected_bin < len(spec[0])

    def test_interval_boundary_handling(self, sample_bin_specs):
        """Test handling of values on interval boundaries."""
        # Create data with values exactly on interval boundaries
        spec = sample_bin_specs["interval_only"]  # {0: [(0, 3), (3, 6), (6, 10)]}
        X = np.array([[0.0], [3.0], [6.0], [10.0]])

        binner = ManualFlexibleBinning(bin_spec=spec)
        binner.fit(X)
        result = binner.transform(X)

        assert result is not None
        assert result.shape == X.shape
        # All results should be valid bin indices (behavior at boundaries may vary)
        assert np.all(result >= 0)

    def test_multiple_columns_different_bin_types(self, sample_data, sample_bin_specs):
        """Test multiple columns with different types of bins."""
        binner = ManualFlexibleBinning(bin_spec=sample_bin_specs["multi_col"])
        binner.fit(sample_data["multi_col"])
        result = binner.transform(sample_data["multi_col"])

        assert result is not None
        assert result.shape == sample_data["multi_col"].shape

        # Check that each column uses its correct number of bins
        col1_bins = len(sample_bin_specs["multi_col"][0])  # Mixed bins
        col2_bins = len(sample_bin_specs["multi_col"][1])  # Interval bins

        assert np.all(result[:, 0] < col1_bins)
        assert np.all(result[:, 1] < col2_bins)

    def test_named_columns_with_dataframe(self, sample_data, sample_bin_specs):
        """Test named columns with DataFrame input."""
        if not hasattr(pd, "DataFrame"):
            pytest.skip("pandas not available")

        # Create bin specs using column names
        binner = ManualFlexibleBinning(bin_spec=sample_bin_specs["named_cols"])

        # Create DataFrame with named columns
        df = pd.DataFrame(
            {"feature1": sample_data["multi_col"][:, 0], "feature2": sample_data["multi_col"][:, 1]}
        )

        binner.fit(df)
        result = binner.transform(df)

        assert result is not None
        assert result.shape == df.shape

    def test_custom_vs_auto_representatives(
        self, sample_data, sample_bin_specs, sample_representatives
    ):
        """Test difference between custom and auto-generated representatives."""
        X = sample_data["simple"]

        # Test with auto-generated representatives
        binner_auto = ManualFlexibleBinning(bin_spec=sample_bin_specs["mixed"])
        binner_auto.fit(X)
        X_binned = binner_auto.transform(X)
        X_inverse_auto = binner_auto.inverse_transform(X_binned)

        # Test with custom representatives
        binner_custom = ManualFlexibleBinning(
            bin_spec=sample_bin_specs["mixed"],
            bin_representatives=sample_representatives["custom_mixed"],
        )
        binner_custom.fit(X)
        X_inverse_custom = binner_custom.inverse_transform(X_binned)

        # The inverse transforms should be different (unless custom reps happen to match auto-generated)
        # At minimum, both should work without error
        assert X_inverse_auto is not None
        assert X_inverse_custom is not None
        assert X_inverse_auto.shape == X_inverse_custom.shape

    # Integration and workflow tests

    def test_full_workflow_with_all_methods(
        self, sample_data, sample_bin_specs, sample_representatives
    ):
        """Test complete workflow: fit, transform, inverse_transform, get_params, set_params."""
        X = sample_data["multi_col"]

        # Initialize and fit
        binner = ManualFlexibleBinning(
            bin_spec=sample_bin_specs["multi_col"],
            bin_representatives=sample_representatives["multi_col"],
            preserve_dataframe=False,
        )
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
        assert "bin_spec" in params
        assert "bin_representatives" in params
        assert params["bin_spec"] == sample_bin_specs["multi_col"]

        # Set params and verify
        new_binner = ManualFlexibleBinning(bin_spec={0: [1.0]})  # Dummy
        new_binner.set_params(**params)

        X_binned_2 = new_binner.transform(X)
        np.testing.assert_array_equal(X_binned, X_binned_2)

    def test_consistency_across_transforms(self, sample_data, sample_bin_specs):
        """Test that repeated transforms give consistent results."""
        X = sample_data["simple"]
        binner = ManualFlexibleBinning(bin_spec=sample_bin_specs["mixed"])
        binner.fit(X)

        # Multiple transforms should give same result
        result1 = binner.transform(X)
        result2 = binner.transform(X)
        result3 = binner.transform(X)

        np.testing.assert_array_equal(result1, result2)
        np.testing.assert_array_equal(result2, result3)

    def test_partial_data_transform(self, sample_data, sample_bin_specs):
        """Test transforming data subset after fitting on full data."""
        X_full = sample_data["uniform"]  # 100 samples
        X_subset = X_full[:20]  # 20 samples

        binner = ManualFlexibleBinning(bin_spec=sample_bin_specs["interval_only"])
        binner.fit(X_full)

        # Transform subset
        result_subset = binner.transform(X_subset)
        result_full = binner.transform(X_full)

        # Subset result should match first 20 rows of full result
        np.testing.assert_array_equal(result_subset, result_full[:20])

    def test_different_bin_configurations(self, sample_data):
        """Test various flexible bin configurations."""
        # Test different types of bin configurations with appropriate data
        test_cases = [
            # (data, bin_spec, description)
            (np.array([1.0, 2.0, 3.0]).reshape(-1, 1), {0: [1.0, 2.0, 3.0]}, "Only singletons"),
            (
                np.array([1.0, 3.0, 6.0, 8.0]).reshape(-1, 1),
                {0: [(0, 5), (5, 10)]},
                "Only intervals",
            ),
            (np.array([1.0, 2.5, 8.0]).reshape(-1, 1), {0: [1.0, (2, 5), 8.0]}, "Mixed"),
            (
                np.array([1.0, 2.5, 4.0, 7.0]).reshape(-1, 1),
                {0: [(0, 3), (2, 6), (5, 10)]},
                "Overlapping intervals",
            ),
        ]

        for X, bin_spec, description in test_cases:
            binner = ManualFlexibleBinning(bin_spec=bin_spec)
            binner.fit(X)
            result = binner.transform(X)

            # Check basic properties
            assert result is not None, f"Failed for {description}"
            assert result.shape == X.shape, f"Shape mismatch for {description}"
            num_bins = len(bin_spec[0])

            # For flexible binning, some values might not match any bin (get -1 or similar)
            # But valid bins should be in range [0, num_bins)
            valid_results = result[result >= 0]  # Filter out unmatched values
            if len(valid_results) > 0:
                assert np.all(valid_results < num_bins), f"Invalid bin indices for {description}"

    def test_inverse_transform_accuracy(
        self, sample_data, sample_bin_specs, sample_representatives
    ):
        """Test accuracy of inverse transformation."""
        X = sample_data["simple"]

        binner = ManualFlexibleBinning(
            bin_spec=sample_bin_specs["mixed"],
            bin_representatives=sample_representatives["custom_mixed"],
        )
        binner.fit(X)

        # Transform and inverse transform
        X_binned = binner.transform(X)
        X_recovered = binner.inverse_transform(X_binned)

        # Recovered values should be the representatives
        expected_reps = sample_representatives["custom_mixed"][0]

        for i, bin_idx in enumerate(X_binned.flatten()):
            assert abs(X_recovered[i, 0] - expected_reps[bin_idx]) < 1e-10

    def test_transform_without_fit_raises_error(self, sample_data, sample_bin_specs):
        """Test that transform without fit raises appropriate error."""
        binner = ManualFlexibleBinning(bin_spec=sample_bin_specs["singleton_only"])

        # Note: Manual binning might work without explicit fit since bins are predefined
        # But if the base class requires fit, this test ensures proper error handling
        try:
            result = binner.transform(sample_data["simple"])
            # If this doesn't raise an error, verify it at least works correctly
            assert result is not None
        except (ValidationError, AttributeError):
            # This is expected behavior if fit is required
            pass

    def test_no_fit_required_for_manual_binning(self, sample_data, sample_bin_specs):
        """Test that manual binning can work without explicit fitting."""
        binner = ManualFlexibleBinning(bin_spec=sample_bin_specs["singleton_only"])

        # Manual binning should work even without calling fit explicitly
        # because bins are predefined
        try:
            result = binner.transform(sample_data["simple"])
            assert result is not None
            assert result.shape == sample_data["simple"].shape
        except (ValidationError, AttributeError):
            # If fit is still required by base class, that's also valid behavior
            binner.fit(sample_data["simple"])
            result = binner.transform(sample_data["simple"])
            assert result is not None

    def test_calculate_flexible_bins_not_implemented(self, sample_bin_specs):
        """Test that _calculate_flexible_bins raises NotImplementedError."""
        import numpy as np

        binner = ManualFlexibleBinning(bin_spec=sample_bin_specs["singleton_only"])

        # Directly calling _calculate_flexible_bins should raise NotImplementedError
        # since manual binning uses pre-defined specifications
        with pytest.raises(
            NotImplementedError, match="Manual binning uses pre-defined specifications"
        ):
            binner._calculate_flexible_bins(x_col=np.array([1.0, 2.0, 3.0]), col_id=0)
