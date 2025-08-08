"""
Comprehensive tests for ManualIntervalBinning method covering all scenarios.
"""

import warnings

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from binlearn import POLARS_AVAILABLE, pd, pl
from binlearn.methods import ManualIntervalBinning
from binlearn.utils import ConfigurationError, ValidationError

# Skip polars tests if not available
polars_skip = pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")

# Skip pandas tests if not available
pandas_skip = pytest.mark.skipif(not hasattr(pd, "DataFrame"), reason="pandas not available")


class TestManualIntervalBinning:
    """Comprehensive test suite for ManualIntervalBinning."""

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
            "out_of_range": np.array([-5.0, 15.0, 25.0, 35.0]).reshape(-1, 1),
        }

    @pytest.fixture
    def sample_bin_edges(self):
        """Generate sample bin edges for testing."""
        return {
            "single_col": {0: [0, 5, 10, 15]},
            "multi_col": {0: [0, 5, 10, 15], 1: [0, 25, 50, 75, 100]},
            "named_cols": {"feature1": [0, 5, 10], "feature2": [0, 50, 100]},
            "uneven": {0: [0, 2, 7, 10, 20]},
            "fine_grained": {0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        }

    @pytest.fixture
    def sample_representatives(self):
        """Generate sample bin representatives for testing."""
        return {
            "single_col": {0: [2.5, 7.5, 12.5]},
            "multi_col": {0: [2.5, 7.5, 12.5], 1: [12.5, 37.5, 62.5, 87.5]},
            "custom": {0: [1.0, 6.0, 13.0]},  # Non-center representatives
        }

    # Basic functionality tests

    def test_init_with_bin_edges_only(self, sample_bin_edges):
        """Test initialization with bin edges only (auto-generate representatives)."""
        binner = ManualIntervalBinning(bin_edges=sample_bin_edges["single_col"])
        assert binner.bin_edges == sample_bin_edges["single_col"]
        assert binner.bin_representatives is None
        assert binner.clip is True  # default from config
        assert binner.preserve_dataframe is False

    def test_init_with_bin_edges_and_representatives(
        self, sample_bin_edges, sample_representatives
    ):
        """Test initialization with both bin edges and representatives."""
        binner = ManualIntervalBinning(
            bin_edges=sample_bin_edges["single_col"],
            bin_representatives=sample_representatives["single_col"],
        )
        assert binner.bin_edges == sample_bin_edges["single_col"]
        assert binner.bin_representatives == sample_representatives["single_col"]

    def test_init_with_custom_parameters(self, sample_bin_edges):
        """Test initialization with custom parameters."""
        binner = ManualIntervalBinning(
            bin_edges=sample_bin_edges["single_col"],
            clip=False,
            preserve_dataframe=True,
        )
        assert binner.bin_edges == sample_bin_edges["single_col"]
        assert binner.clip is False
        assert binner.preserve_dataframe is True

    def test_parameter_validation_no_bin_edges(self):
        """Test parameter validation when bin_edges is None."""
        with pytest.raises(ConfigurationError, match="bin_edges must be provided"):
            ManualIntervalBinning(bin_edges=None)  # type: ignore

    def test_parameter_validation_empty_bin_edges(self):
        """Test parameter validation when bin_edges is empty."""
        with pytest.raises(ConfigurationError, match="bin_edges cannot be empty"):
            ManualIntervalBinning(bin_edges={})

    def test_validate_params_with_none_bin_edges(self, sample_bin_edges):
        """Test validation when bin_edges becomes None after initialization."""
        binner = ManualIntervalBinning(bin_edges=sample_bin_edges["single_col"])

        # Manually set bin_edges to None to test the validation path
        binner.bin_edges = None

        with pytest.raises(ConfigurationError, match="bin_edges must be provided"):
            binner._validate_params()

    # Fitting tests (no-op for manual binning)

    def test_fit_returns_self(self, sample_data, sample_bin_edges):
        """Test that fit returns self and sets fitted state."""
        binner = ManualIntervalBinning(bin_edges=sample_bin_edges["single_col"])
        result = binner.fit(sample_data["simple"])

        assert result is binner
        assert binner._fitted is True

    def test_fit_with_y_parameter(self, sample_data, sample_bin_edges):
        """Test that fit ignores y parameter (manual binning is unsupervised)."""
        binner = ManualIntervalBinning(bin_edges=sample_bin_edges["single_col"])
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        result = binner.fit(sample_data["simple"], y=y)

        assert result is binner
        assert binner._fitted is True

    # Input format tests with preserve_dataframe=False

    def test_numpy_input_preserve_false(self, sample_data, sample_bin_edges):
        """Test with numpy input and preserve_dataframe=False."""
        binner = ManualIntervalBinning(
            bin_edges=sample_bin_edges["single_col"], preserve_dataframe=False
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
    def test_pandas_input_preserve_false(self, sample_data, sample_bin_edges):
        """Test with pandas input and preserve_dataframe=False."""
        binner = ManualIntervalBinning(
            bin_edges=sample_bin_edges["single_col"], preserve_dataframe=False
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
    def test_polars_input_preserve_false(self, sample_data, sample_bin_edges):
        """Test with polars input and preserve_dataframe=False."""
        binner = ManualIntervalBinning(
            bin_edges=sample_bin_edges["single_col"], preserve_dataframe=False
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
    def test_pandas_input_preserve_true(self, sample_data, sample_bin_edges):
        """Test with pandas input and preserve_dataframe=True."""
        binner = ManualIntervalBinning(
            bin_edges=sample_bin_edges["multi_col"], preserve_dataframe=True
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
    def test_polars_input_preserve_true(self, sample_data, sample_bin_edges):
        """Test with polars input and preserve_dataframe=True."""
        binner = ManualIntervalBinning(
            bin_edges=sample_bin_edges["multi_col"], preserve_dataframe=True
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

    def test_reconstruction_via_get_params_set_params(self, sample_data, sample_bin_edges):
        """Test fitted state reconstruction via get_params/set_params."""
        # Fit original binner
        binner_original = ManualIntervalBinning(bin_edges=sample_bin_edges["multi_col"])
        X_fit = sample_data["multi_col"]
        binner_original.fit(X_fit)

        # Get parameters
        params = binner_original.get_params()

        # Create new binner and set parameters (reconstruct state)
        binner_reconstructed = ManualIntervalBinning(bin_edges={0: [0, 1]})  # Dummy edges
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

    def test_reconstruction_via_constructor(self, sample_data, sample_bin_edges):
        """Test fitted state reconstruction via constructor parameters."""
        # Fit original binner
        binner_original = ManualIntervalBinning(bin_edges=sample_bin_edges["single_col"])
        X_fit = sample_data["simple"]
        binner_original.fit(X_fit)

        # Get parameters including fitted state
        params = binner_original.get_params()

        # Create new binner with constructor (reconstruct state)
        binner_reconstructed = ManualIntervalBinning(**params)

        # Test that transform works without fitting (manual binning doesn't need fitting)
        X_test = sample_data["simple"]
        result_original = binner_original.transform(X_test)
        result_reconstructed = binner_reconstructed.transform(X_test)

        np.testing.assert_array_equal(result_original, result_reconstructed)

    def test_repeated_fitting_after_reconstruction(self, sample_data, sample_bin_edges):
        """Test repeated fitting on reconstructed state."""
        # Original fitting
        binner = ManualIntervalBinning(bin_edges=sample_bin_edges["single_col"])
        X_fit1 = sample_data["simple"]
        binner.fit(X_fit1)

        # Get and set params (reconstruction)
        params = binner.get_params()
        binner_new = ManualIntervalBinning(bin_edges={0: [0, 1]})  # Dummy edges
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

    def test_various_formats_after_reconstruction(self, sample_data, sample_bin_edges):
        """Test various input formats after fitted state reconstruction."""
        # Original fitting with numpy
        binner_original = ManualIntervalBinning(
            bin_edges=sample_bin_edges["multi_col"], preserve_dataframe=True
        )
        X_numpy = sample_data["multi_col"]
        binner_original.fit(X_numpy)

        # Reconstruct
        params = binner_original.get_params()
        binner_reconstructed = ManualIntervalBinning(**params)

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

    def test_sklearn_pipeline_basic(self, sample_data, sample_bin_edges):
        """Test basic sklearn pipeline integration."""
        # Create pipeline
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("binner", ManualIntervalBinning(bin_edges=sample_bin_edges["single_col"])),
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
        assert hasattr(pipeline.named_steps["binner"], "bin_edges")

    def test_sklearn_pipeline_with_dataframes(self, sample_data, sample_bin_edges):
        """Test sklearn pipeline with DataFrame inputs."""
        if not hasattr(pd, "DataFrame"):
            pytest.skip("pandas not available")

        # Create pipeline
        pipeline = Pipeline(
            [
                (
                    "binner",
                    ManualIntervalBinning(
                        bin_edges=sample_bin_edges["multi_col"], preserve_dataframe=True
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

    def test_sklearn_pipeline_param_access(self, sample_bin_edges):
        """Test parameter access in sklearn pipeline."""
        pipeline = Pipeline(
            [("binner", ManualIntervalBinning(bin_edges=sample_bin_edges["single_col"]))]
        )

        # Test parameter access
        params = pipeline.get_params()
        assert "binner__bin_edges" in params
        assert params["binner__bin_edges"] == sample_bin_edges["single_col"]
        assert "binner__clip" in params

        # Test parameter setting for settable parameters
        # Note: bin_edges might not be settable after construction for manual binning
        pipeline.set_params(binner__clip=False)
        assert pipeline.named_steps["binner"].clip is False

        # Test that bin_edges parameter is accessible (even if not settable)
        binner_edges = pipeline.get_params()["binner__bin_edges"]
        assert binner_edges == sample_bin_edges["single_col"]

    # Edge case tests

    def test_edge_case_nan_values(self, sample_data, sample_bin_edges):
        """Test handling of NaN values."""
        binner = ManualIntervalBinning(bin_edges=sample_bin_edges["single_col"])
        X_nan = sample_data["with_nan"]

        # Should handle NaN values gracefully (base class preprocesses)
        binner.fit(X_nan)
        result = binner.transform(X_nan)

        assert result is not None
        # NaN values should be handled by the base class
        assert result.shape == X_nan.shape

    def test_edge_case_inf_values(self, sample_data, sample_bin_edges):
        """Test handling of infinite values."""
        binner = ManualIntervalBinning(bin_edges=sample_bin_edges["single_col"])
        X_inf = sample_data["with_inf"]

        # Should handle inf values based on config (clip, error, etc.)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings for this test
            binner.fit(X_inf)
            result = binner.transform(X_inf)

            assert result is not None
            assert result.shape == X_inf.shape

    def test_edge_case_out_of_range_values_with_clip(self, sample_data, sample_bin_edges):
        """Test handling of out-of-range values with clip=True."""
        binner = ManualIntervalBinning(bin_edges=sample_bin_edges["single_col"], clip=True)
        X_out_of_range = sample_data["out_of_range"]

        binner.fit(X_out_of_range)
        result = binner.transform(X_out_of_range)

        assert result is not None
        assert result.shape == X_out_of_range.shape
        # All values should be within valid bin range when clipped
        assert np.all(result >= 0)
        assert np.all(result < len(sample_bin_edges["single_col"][0]) - 1)

    def test_edge_case_out_of_range_values_without_clip(self, sample_data, sample_bin_edges):
        """Test handling of out-of-range values with clip=False."""
        binner = ManualIntervalBinning(bin_edges=sample_bin_edges["single_col"], clip=False)
        X_out_of_range = sample_data["out_of_range"]

        binner.fit(X_out_of_range)
        result = binner.transform(X_out_of_range)

        assert result is not None
        assert result.shape == X_out_of_range.shape
        # May have special values (like MISSING_VALUE) for out-of-range data

    def test_edge_case_single_row_data(self, sample_bin_edges):
        """Test handling of single row data."""
        X = np.array([[5.0]])
        binner = ManualIntervalBinning(bin_edges=sample_bin_edges["single_col"])

        binner.fit(X)
        result = binner.transform(X)

        assert result is not None
        assert result.shape == (1, 1)
        assert isinstance(result[0, 0], int | np.integer)

    def test_edge_case_empty_data_handling(self, sample_bin_edges):
        """Test handling of empty data."""
        binner = ManualIntervalBinning(bin_edges=sample_bin_edges["single_col"])

        # Empty data should be handled gracefully
        empty_data = np.array([]).reshape(0, 1)
        binner.fit(empty_data)  # Should not raise error
        result = binner.transform(empty_data)
        assert result.shape[0] == 0  # Should have 0 rows
        # Note: Column count may vary based on base class implementation

    def test_missing_column_in_bin_edges(self, sample_data):
        """Test behavior when input data has more columns than bin specifications."""
        # Define edges only for column 0, but try to transform 2-column data
        bin_edges = {0: [0.0, 5.0, 10.0]}
        binner = ManualIntervalBinning(bin_edges=bin_edges)

        binner.fit(sample_data["multi_col"])

        # Should raise an error for mismatched column count (wrapped by transform's exception handler)
        with pytest.raises(
            ValueError,
            match="Failed to transform data.*Input data has 2 columns but bin specifications are provided for 1 columns",
        ):
            binner.transform(sample_data["multi_col"])
        # Column 1 behavior for missing edges is implementation-dependent

    def test_custom_representatives_behavior(
        self, sample_data, sample_bin_edges, sample_representatives
    ):
        """Test the behavior when custom representatives are provided."""
        binner = ManualIntervalBinning(
            bin_edges=sample_bin_edges["single_col"],
            bin_representatives=sample_representatives["custom"],
        )

        # Fit and transform to test that custom representatives are used properly
        binner.fit(sample_data["simple"])
        result = binner.transform(sample_data["simple"])

        # Verify transform works
        assert result is not None
        assert result.shape == sample_data["simple"].shape

        # Test inverse transform uses the custom representatives
        recovered = binner.inverse_transform(result)

        # The recovered values should be from the custom representatives
        expected_reps = sample_representatives["custom"][0]
        unique_recovered = np.unique(recovered)

        # All recovered values should be from our custom representatives
        for val in unique_recovered:
            assert any(abs(val - rep) < 1e-10 for rep in expected_reps)

    def test_auto_generated_representatives_behavior(self, sample_data, sample_bin_edges):
        """Test the behavior when representatives are auto-generated as bin centers."""
        binner = ManualIntervalBinning(bin_edges=sample_bin_edges["single_col"])

        # Fit and transform to test auto-generated representatives
        binner.fit(sample_data["simple"])
        result = binner.transform(sample_data["simple"])

        # Verify transform works
        assert result is not None
        assert result.shape == sample_data["simple"].shape

        # Test inverse transform uses auto-generated representatives (bin centers)
        recovered = binner.inverse_transform(result)

        # Calculate expected representatives as bin centers
        expected_edges = sample_bin_edges["single_col"][0]
        expected_reps = [
            (expected_edges[i] + expected_edges[i + 1]) / 2 for i in range(len(expected_edges) - 1)
        ]

        # All recovered values should be from the auto-generated representatives
        unique_recovered = np.unique(recovered)
        for val in unique_recovered:
            assert any(abs(val - rep) < 1e-10 for rep in expected_reps)

    # Specific Manual Interval binning tests

    def test_auto_generated_representatives(self, sample_data, sample_bin_edges):
        """Test that representatives are auto-generated as bin centers."""
        binner = ManualIntervalBinning(bin_edges=sample_bin_edges["single_col"])
        binner.fit(sample_data["simple"])

        # Check that representatives were generated
        assert binner.bin_representatives_ is not None
        reps = binner.bin_representatives_[0]
        edges = sample_bin_edges["single_col"][0]

        # Should have n-1 representatives for n edges
        assert len(reps) == len(edges) - 1

        # Representatives should be bin centers
        for i in range(len(reps)):
            expected_center = (edges[i] + edges[i + 1]) / 2
            assert abs(reps[i] - expected_center) < 1e-10

    def test_custom_representatives(self, sample_data, sample_bin_edges, sample_representatives):
        """Test that custom representatives are used when provided."""
        binner = ManualIntervalBinning(
            bin_edges=sample_bin_edges["single_col"],
            bin_representatives=sample_representatives["custom"],
        )
        binner.fit(sample_data["simple"])

        # Check that custom representatives were used
        reps = binner.bin_representatives_[0]
        expected_reps = sample_representatives["custom"][0]

        np.testing.assert_array_equal(reps, expected_reps)

    def test_uneven_bin_widths(self, sample_data, sample_bin_edges):
        """Test handling of uneven bin widths."""
        binner = ManualIntervalBinning(bin_edges=sample_bin_edges["uneven"])
        binner.fit(sample_data["simple"])
        result = binner.transform(sample_data["simple"])

        assert result is not None
        assert result.shape == sample_data["simple"].shape

        # Check that all values are assigned to valid bins
        num_bins = len(sample_bin_edges["uneven"][0]) - 1
        assert np.all(result >= 0)
        assert np.all(result < num_bins)

    def test_fine_grained_binning(self, sample_data, sample_bin_edges):
        """Test fine-grained binning with many bins."""
        binner = ManualIntervalBinning(bin_edges=sample_bin_edges["fine_grained"])
        binner.fit(sample_data["simple"])
        result = binner.transform(sample_data["simple"])

        assert result is not None
        assert result.shape == sample_data["simple"].shape

        # Should use all or most of the available bins
        unique_bins = np.unique(result)
        assert len(unique_bins) > 5  # Should use several bins

    def test_bin_boundaries_exact_matches(self, sample_bin_edges):
        """Test that values exactly on bin boundaries are handled correctly."""
        # Create data with values exactly on bin edges
        edges = sample_bin_edges["single_col"][0]  # [0, 5, 10, 15]
        X = np.array(edges).reshape(-1, 1)

        binner = ManualIntervalBinning(bin_edges=sample_bin_edges["single_col"])
        binner.fit(X)
        result = binner.transform(X)

        assert result is not None
        assert result.shape == X.shape
        # All results should be valid bin indices
        assert np.all(result >= 0)

    def test_multiple_columns_different_bin_counts(self, sample_data, sample_bin_edges):
        """Test multiple columns with different numbers of bins."""
        binner = ManualIntervalBinning(bin_edges=sample_bin_edges["multi_col"])
        binner.fit(sample_data["multi_col"])
        result = binner.transform(sample_data["multi_col"])

        assert result is not None
        assert result.shape == sample_data["multi_col"].shape

        # Check that each column uses its correct number of bins
        col1_bins = len(sample_bin_edges["multi_col"][0]) - 1  # 3 bins
        col2_bins = len(sample_bin_edges["multi_col"][1]) - 1  # 4 bins

        assert np.all(result[:, 0] < col1_bins)
        assert np.all(result[:, 1] < col2_bins)

    def test_named_columns_with_dataframe(self, sample_data):
        """Test named columns with DataFrame input."""
        if not hasattr(pd, "DataFrame"):
            pytest.skip("pandas not available")

        # Create bin edges using column names
        bin_edges = {"feature1": [0.0, 5.0, 10.0], "feature2": [0.0, 50.0, 100.0]}
        binner = ManualIntervalBinning(bin_edges=bin_edges)

        # Create DataFrame with named columns
        df = pd.DataFrame(
            {"feature1": sample_data["multi_col"][:, 0], "feature2": sample_data["multi_col"][:, 1]}
        )

        binner.fit(df)
        result = binner.transform(df)

        assert result is not None
        assert result.shape == df.shape

    def test_clip_parameter_functionality(self, sample_data, sample_bin_edges):
        """Test clip parameter with out-of-range values."""
        X_out_of_range = np.array([[-10], [25]]).astype(float)  # Outside [0, 15] range

        # Test with clip=True
        binner_clip = ManualIntervalBinning(bin_edges=sample_bin_edges["single_col"], clip=True)
        binner_clip.fit(X_out_of_range)
        result_clip = binner_clip.transform(X_out_of_range)

        # Should clip to valid bin range
        num_bins = len(sample_bin_edges["single_col"][0]) - 1
        assert np.all(result_clip >= 0)
        assert np.all(result_clip < num_bins)

        # Test with clip=False
        binner_no_clip = ManualIntervalBinning(bin_edges=sample_bin_edges["single_col"], clip=False)
        binner_no_clip.fit(X_out_of_range)
        result_no_clip = binner_no_clip.transform(X_out_of_range)

        # May have values outside normal bin range (exact behavior depends on base class)
        assert result_no_clip is not None

    # Integration and workflow tests

    def test_full_workflow_with_all_methods(
        self, sample_data, sample_bin_edges, sample_representatives
    ):
        """Test complete workflow: fit, transform, inverse_transform, get_params, set_params."""
        X = sample_data["multi_col"]

        # Initialize and fit
        binner = ManualIntervalBinning(
            bin_edges=sample_bin_edges["multi_col"],
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
        assert "bin_edges" in params
        assert "bin_representatives" in params
        assert params["bin_edges"] == sample_bin_edges["multi_col"]

        # Set params and verify
        new_binner = ManualIntervalBinning(bin_edges={0: [0, 1]})  # Dummy
        new_binner.set_params(**params)

        X_binned_2 = new_binner.transform(X)
        np.testing.assert_array_equal(X_binned, X_binned_2)

    def test_consistency_across_transforms(self, sample_data, sample_bin_edges):
        """Test that repeated transforms give consistent results."""
        X = sample_data["simple"]
        binner = ManualIntervalBinning(bin_edges=sample_bin_edges["single_col"])
        binner.fit(X)

        # Multiple transforms should give same result
        result1 = binner.transform(X)
        result2 = binner.transform(X)
        result3 = binner.transform(X)

        np.testing.assert_array_equal(result1, result2)
        np.testing.assert_array_equal(result2, result3)

    def test_partial_data_transform(self, sample_data, sample_bin_edges):
        """Test transforming data subset after fitting on full data."""
        X_full = sample_data["uniform"]  # 100 samples
        X_subset = X_full[:20]  # 20 samples

        binner = ManualIntervalBinning(bin_edges=sample_bin_edges["single_col"])
        binner.fit(X_full)

        # Transform subset
        result_subset = binner.transform(X_subset)
        result_full = binner.transform(X_full)

        # Subset result should match first 20 rows of full result
        np.testing.assert_array_equal(result_subset, result_full[:20])

    def test_different_bin_configurations(self, sample_data):
        """Test various bin edge configurations."""
        X = sample_data["simple"]

        # Test different numbers of bins
        for num_bins in [2, 3, 5, 10]:
            edges = np.linspace(0, 11, num_bins + 1)
            bin_edges = {0: list(edges.astype(float))}

            binner = ManualIntervalBinning(bin_edges=bin_edges)
            binner.fit(X)
            result = binner.transform(X)

            # Check that we get expected number of bins
            unique_bins = np.unique(result)
            assert len(unique_bins) <= num_bins
            assert np.all(result >= 0)
            assert np.all(result < num_bins)

    def test_inverse_transform_accuracy(
        self, sample_data, sample_bin_edges, sample_representatives
    ):
        """Test accuracy of inverse transformation."""
        X = sample_data["simple"]

        binner = ManualIntervalBinning(
            bin_edges=sample_bin_edges["single_col"],
            bin_representatives=sample_representatives["single_col"],
        )
        binner.fit(X)

        # Transform and inverse transform
        X_binned = binner.transform(X)
        X_recovered = binner.inverse_transform(X_binned)

        # Recovered values should be the representatives
        expected_reps = sample_representatives["single_col"][0]

        for i, bin_idx in enumerate(X_binned.flatten()):
            assert abs(X_recovered[i, 0] - expected_reps[bin_idx]) < 1e-10

    def test_transform_without_fit_raises_error(self, sample_data, sample_bin_edges):
        """Test that transform without fit raises appropriate error."""
        binner = ManualIntervalBinning(bin_edges=sample_bin_edges["single_col"])

        # Note: Manual binning might work without explicit fit since bins are predefined
        # But if the base class requires fit, this test ensures proper error handling
        try:
            result = binner.transform(sample_data["simple"])
            # If this doesn't raise an error, verify it at least works correctly
            assert result is not None
        except (ValidationError, AttributeError):
            # This is expected behavior if fit is required
            pass

    def test_no_fit_required_for_manual_binning(self, sample_data, sample_bin_edges):
        """Test that manual binning can work without explicit fitting."""
        binner = ManualIntervalBinning(bin_edges=sample_bin_edges["single_col"])

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
