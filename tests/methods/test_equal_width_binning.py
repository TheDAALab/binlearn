"""
Comprehensive test suite for EqualWidthBinning transformer.

This module contains extensive tests for the EqualWidthBinning class, covering
initialization, parameter validation, fitting, transformation, edge cases,
data type compatibility, sklearn integration, and error handling.

Test Classes:
    TestEqualWidthBinning: Core functionality tests including initialization,
        validation, fitting, transformation, and basic operations.
    TestEqualWidthBinningDataTypes: Tests for various data type compatibility
        including pandas DataFrames, polars DataFrames, and scipy sparse matrices.
    TestEqualWidthBinningSklearnIntegration: Tests for sklearn compatibility
        including pipeline integration, ColumnTransformer usage, and cloning.
    TestEqualWidthBinningFitGetParamsWorkflow: Tests for parameter handling
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
from binlearn.methods._equal_width_binning import EqualWidthBinning
from binlearn.utils.errors import ConfigurationError, DataQualityWarning

try:
    from scipy import sparse
except ImportError:  # pragma: no cover
    sparse = None
SKLEARN_AVAILABLE = True


class TestEqualWidthBinning:
    """Comprehensive test cases for EqualWidthBinning core functionality.

    This test class covers the fundamental operations of the EqualWidthBinning
    transformer including initialization, parameter validation, fitting,
    transformation, edge cases, and basic data handling scenarios.
    """

    def test_init_default(self):
        """Test initialization with default parameters.

        Verifies that the transformer initializes correctly with default
        parameter values and that all attributes are set as expected.
        """
        ewb = EqualWidthBinning()
        assert ewb.n_bins == 10
        assert ewb.bin_range is None

    def test_init_custom_params(self):
        """Test initialization with custom parameters.

        Verifies that the transformer correctly accepts and stores custom
        parameter values including n_bins, bin_range, and fit_jointly options.
        """
        ewb = EqualWidthBinning(n_bins=5, bin_range=(0, 100), fit_jointly=True)
        assert ewb.n_bins == 5
        assert ewb.bin_range == (0, 100)
        assert ewb.fit_jointly is True

    def test_validate_params_valid(self):
        """Test parameter validation with valid parameters.

        Ensures that the _validate_params method accepts valid parameter
        combinations without raising exceptions.
        """
        ewb = EqualWidthBinning(n_bins=5, bin_range=(0, 100))
        ewb._validate_params()  # Should not raise

    def test_validate_params_invalid_n_bins(self):
        """Test parameter validation with invalid n_bins values.

        Verifies that the validator correctly rejects invalid n_bins values
        such as zero, negative numbers, and non-integer types.
        """
        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            EqualWidthBinning(n_bins=0)

        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            EqualWidthBinning(n_bins=-1)

    def test_validate_params_invalid_bin_range(self):
        """Test parameter validation with invalid bin_range values.

        Verifies that the validator correctly rejects invalid bin_range
        specifications such as reversed ranges (min > max).
        """
        with pytest.raises(ConfigurationError, match="bin_range must be a tuple"):
            EqualWidthBinning(bin_range=(10, 5))  # min > max

        with pytest.raises(ConfigurationError, match="bin_range must be a tuple"):
            EqualWidthBinning(bin_range=(5, 5))  # min == max

    def test_fit_transform_basic(self):
        """Test basic fit and transform functionality."""
        X = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])
        ewb = EqualWidthBinning(n_bins=3)

        # Test fit
        ewb.fit(X)
        assert hasattr(ewb, "bin_edges_")  # Check that fitting has occurred
        assert len(ewb.bin_edges_) == 2  # Two columns

        # Test transform
        X_transformed = ewb.transform(X)
        assert X_transformed.shape == X.shape
        assert X_transformed.dtype == int
        assert np.all(X_transformed >= 0)
        assert np.all(X_transformed < 3)  # Should be in range [0, n_bins)

    def test_fit_transform_with_bin_range(self):
        """Test fit and transform with specified bin_range."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        ewb = EqualWidthBinning(n_bins=2, bin_range=(0, 10))

        ewb.fit(X)
        X_transformed = ewb.transform(X)

        # Data values 1-6 in range 0-10 with 2 bins: [0,5) and [5,10]
        # So values 1,2,3,4 should be in bin 0, and values 5,6 should be in bin 1
        expected = np.array([[0, 0], [0, 0], [1, 1]])
        np.testing.assert_array_equal(X_transformed, expected)

    def test_joint_binning_vs_individual(self):
        """Test joint binning vs individual binning."""
        X = np.array([[1, 100], [2, 200], [3, 300], [4, 400], [5, 500]])

        # Individual binning
        ewb_individual = EqualWidthBinning(n_bins=3, fit_jointly=False)
        ewb_individual.fit(X)

        # Joint binning
        ewb_joint = EqualWidthBinning(n_bins=3, fit_jointly=True)
        ewb_joint.fit(X)

        # Bin edges should be different
        edges_individual_col0 = ewb_individual.bin_edges_[0]
        edges_individual_col1 = ewb_individual.bin_edges_[1]
        edges_joint_col0 = ewb_joint.bin_edges_[0]
        edges_joint_col1 = ewb_joint.bin_edges_[1]

        # Individual: each column has its own range
        assert not np.allclose(edges_individual_col0, edges_individual_col1)

        # Joint: both columns should have the same range
        assert np.allclose(edges_joint_col0, edges_joint_col1)

    def test_constant_data(self):
        """Test handling of constant data."""
        X = np.array([[5, 5], [5, 5], [5, 5]])
        ewb = EqualWidthBinning(n_bins=3)

        ewb.fit(X)
        X_transformed = ewb.transform(X)

        # All values should be in the same bin (typically bin 0)
        assert np.all(X_transformed == X_transformed[0, 0])

    def test_nan_data(self):
        """Test handling of NaN data."""
        X = np.array([[1, np.nan], [2, np.nan], [3, np.nan]])
        ewb = EqualWidthBinning(n_bins=3)

        # Second column has all NaN values, which will trigger a warning
        with pytest.warns(DataQualityWarning, match="contains only NaN values"):
            ewb.fit(X)
        X_transformed = ewb.transform(X)

        # First column should be binned normally, second column should handle NaN
        assert not np.any(np.isnan(X_transformed[:, 0]))
        # Second column behavior depends on implementation details

    def test_single_column(self):
        """Test with single column data."""
        X = np.array([[1], [2], [3], [4], [5]])
        ewb = EqualWidthBinning(n_bins=3)

        ewb.fit(X)
        X_transformed = ewb.transform(X)

        assert X_transformed.shape == (5, 1)
        assert len(ewb.bin_edges_) == 1

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_dataframe(self):
        """Test with pandas DataFrame input."""
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [10, 20, 30, 40, 50]})
        ewb = EqualWidthBinning(n_bins=3, preserve_dataframe=True)

        ewb.fit(df)
        df_transformed = ewb.transform(df)

        assert isinstance(df_transformed, pd.DataFrame)
        assert df_transformed.shape == df.shape
        assert list(df_transformed.columns) == list(df.columns)

    def test_handle_bin_params(self):
        """Test _handle_bin_params method."""
        ewb = EqualWidthBinning()

        params = {"n_bins": 7, "bin_range": (0, 50)}

        reset_fitted = ewb._handle_bin_params(params)

        assert reset_fitted is True
        assert ewb.n_bins == 7
        assert ewb.bin_range == (0, 50)

    def test_create_equal_width_bins(self):
        """Test _create_equal_width_bins method."""
        ewb = EqualWidthBinning()

        edges, reps = ewb._create_equal_width_bins(0.0, 10.0, 5)

        assert len(edges) == 6  # n_bins + 1
        assert len(reps) == 5  # n_bins
        assert edges[0] == 0.0
        assert edges[-1] == 10.0
        assert np.allclose(np.diff(edges), 2.0)  # Equal width bins

    def test_get_data_range(self):
        """Test _get_data_range method."""
        ewb = EqualWidthBinning()

        # Normal data
        x_col = np.array([1, 2, 3, 4, 5])
        min_val, max_val = ewb._get_data_range(x_col, "test")
        assert min_val == 1.0
        assert max_val == 5.0

        # All NaN data
        x_col_nan = np.array([np.nan, np.nan, np.nan])
        min_val, max_val = ewb._get_data_range(x_col_nan, "test")
        assert min_val == 0.0
        assert max_val == 1.0

    def test_repr(self):
        """Test string representation."""
        ewb = EqualWidthBinning(n_bins=5, bin_range=(0, 100))
        repr_str = repr(ewb)
        assert "EqualWidthBinning" in repr_str
        assert "n_bins=5" in repr_str

    def test_invalid_data_range(self):
        """Test error handling for invalid data ranges."""
        ewb = EqualWidthBinning()

        # Data with infinite values
        x_col = np.array([1, 2, np.inf])
        with pytest.raises(ValueError, match="min and max must be finite"):
            ewb._get_data_range(x_col, "test")

    def test_data_range_with_inf_values_in_finite_check(self):
        """Test _get_data_range when nanmin/nanmax succeed but values are infinite."""
        ewb = EqualWidthBinning()

        # Create data where nanmin/nanmax don't raise ValueError but result in inf
        # This happens when we have finite values mixed with inf
        x_col = np.array([1.0, 2.0, 3.0, np.inf])
        with pytest.raises(ValueError, match="min and max must be finite"):
            ewb._get_data_range(x_col, "test_col")

    def test_data_range_with_negative_inf_values(self):
        """Test _get_data_range with negative infinity values."""
        ewb = EqualWidthBinning()

        # Test with -inf values
        x_col = np.array([-np.inf, 1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="min and max must be finite"):
            ewb._get_data_range(x_col, "test_col")

    def test_joint_binning_with_range(self):
        """Test joint binning with specified range."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        ewb = EqualWidthBinning(n_bins=2, bin_range=(0, 10), fit_jointly=True)

        ewb.fit(X)

        # Both columns should use the same range (0, 10)
        for col in [0, 1]:
            edges = ewb.bin_edges_[col]
            assert edges[0] == 0.0
            assert edges[-1] == 10.0

    def test_joint_binning_invalid_n_bins(self):
        """Test joint binning with invalid n_bins."""
        # Now fails during initialization (better behavior!)
        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            EqualWidthBinning(n_bins=0, fit_jointly=True)

    def test_per_column_invalid_n_bins(self):
        """Test per-column binning with invalid n_bins."""
        # Now fails during initialization (better behavior!)
        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            EqualWidthBinning(n_bins=-1)

    def test_create_bins_constant_data_handling(self):
        """Test _create_equal_width_bins with constant data."""
        ewb = EqualWidthBinning()

        # Test constant data handling (min_val == max_val)
        edges, reps = ewb._create_equal_width_bins(5.0, 5.0, 3)

        # Should add epsilon to create a valid range
        assert len(edges) == 4  # n_bins + 1
        assert len(reps) == 3  # n_bins
        assert edges[0] < edges[-1]  # Should have created a valid range

    def test_validate_params_non_integer_n_bins(self):
        """Test parameter validation with non-integer n_bins."""
        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            ewb = EqualWidthBinning()
            ewb.n_bins = "invalid"  # Bypass type checking
            ewb._validate_params()

    def test_validate_params_invalid_bin_range_length(self):
        """Test parameter validation with invalid bin_range length."""
        with pytest.raises(ConfigurationError, match="bin_range must be a tuple"):
            ewb = EqualWidthBinning()
            ewb.bin_range = 1, 2, 3  # Bypass type checking, wrong length
            ewb._validate_params()

    def test_validate_params_non_tuple_bin_range(self):
        """Test parameter validation with non-tuple bin_range."""
        with pytest.raises(ConfigurationError, match="bin_range must be a tuple"):
            ewb = EqualWidthBinning()
            ewb.bin_range = [0, 10]  # Bypass type checking, list instead of tuple
            ewb._validate_params()

    def test_empty_params_in_handle_bin_params(self):
        """Test _handle_bin_params with no relevant parameters."""
        ewb = EqualWidthBinning()

        # Test with empty params dict
        reset_fitted = ewb._handle_bin_params({})
        assert reset_fitted is False

        # Test with irrelevant params
        reset_fitted = ewb._handle_bin_params({"irrelevant_param": "value"})
        assert reset_fitted is False

    def test_data_range_with_all_nan_values(self):
        """Test _get_data_range with all NaN values."""
        ewb = EqualWidthBinning()

        x_col = np.array([np.nan, np.nan, np.nan])
        min_val, max_val = ewb._get_data_range(x_col, "test_col")

        # Should return default range (0.0, 1.0)
        assert min_val == 0.0
        assert max_val == 1.0

    def test_direct_calculate_bins_invalid_n_bins_joint_fitting(self):
        """Test _calculate_bins with joint fitting data and invalid n_bins."""
        ewb = EqualWidthBinning()
        ewb.n_bins = 0  # Set invalid value directly

        X = np.array([1, 2, 3, 4])  # Flattened data as would be passed for joint fitting
        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            ewb._calculate_bins(X, 0)

    def test_direct_calculate_bins_invalid_n_bins(self):
        """Test _calculate_bins directly with invalid n_bins."""
        ewb = EqualWidthBinning()
        ewb.n_bins = -1  # Set invalid value directly

        x_col = np.array([1, 2, 3, 4])
        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            ewb._calculate_bins(x_col, 0)


class TestEqualWidthBinningDataTypes:
    """Comprehensive tests for EqualWidthBinning with various data types.

    This test class verifies that the EqualWidthBinning transformer works
    correctly with different input data types including numpy arrays of
    various dtypes, pandas DataFrames, polars DataFrames, and scipy sparse
    matrices. It ensures proper type preservation and format handling.
    """

    def test_numpy_arrays_2d(self):
        """Test functionality with 2D numpy arrays.

        Verifies that the transformer correctly handles standard 2D numpy
        arrays and produces output of the expected shape and dtype.
        """
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        ewb = EqualWidthBinning(n_bins=2)

        ewb.fit(X)
        result = ewb.transform(X)

        assert isinstance(result, np.ndarray)
        assert result.shape == X.shape
        assert result.dtype == int

    def test_numpy_arrays_1d(self):
        """Test functionality with 1D numpy arrays.

        Ensures that 1D arrays are properly handled when reshaped to
        2D format and that the output maintains correct dimensionality.
        """
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1)
        ewb = EqualWidthBinning(n_bins=3)

        ewb.fit(X)
        result = ewb.transform(X)

        assert isinstance(result, np.ndarray)
        assert result.shape == X.shape
        assert result.dtype == int

    def test_numpy_different_dtypes(self):
        """Test compatibility with different numpy data types.

        Verifies that the transformer works correctly with various numpy
        dtypes including int32, float32, and others, ensuring proper
        type handling and conversion.
        """
        # Test int32
        X_int = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.int32)
        ewb = EqualWidthBinning(n_bins=2)
        ewb.fit(X_int)
        result_int = ewb.transform(X_int)
        assert isinstance(result_int, np.ndarray)

        # Test float32
        X_float32 = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)
        ewb = EqualWidthBinning(n_bins=2)
        ewb.fit(X_float32)
        result_float32 = ewb.transform(X_float32)
        assert isinstance(result_float32, np.ndarray)

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_dataframe_input_output(self):
        """Test pandas DataFrame input and output handling.

        Verifies that pandas DataFrames are processed correctly and that
        the preserve_dataframe option works as expected for maintaining
        DataFrame format in the output.
        """
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
                "feature3": [100.0, 200.0, 300.0, 400.0, 500.0],
            }
        )

        # Test with preserve_dataframe=True (default for DataFrame input)
        ewb = EqualWidthBinning(n_bins=3, preserve_dataframe=True)
        ewb.fit(df)
        result = ewb.transform(df)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape
        assert list(result.columns) == list(df.columns)
        assert result.dtypes.apply(lambda x: x.name).eq("int64").all()

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_dataframe_to_numpy(self):
        """Test DataFrame input with numpy output."""
        df = pd.DataFrame({"col1": [1.0, 2.0, 3.0, 4.0], "col2": [10.0, 20.0, 30.0, 40.0]})

        ewb = EqualWidthBinning(n_bins=2, preserve_dataframe=False)
        ewb.fit(df)
        result = ewb.transform(df)

        assert isinstance(result, np.ndarray)
        assert result.shape == df.shape

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
    def test_polars_dataframe(self):
        """Test with Polars DataFrame."""
        df_polars = pl.DataFrame(
            {  # type: ignore[name-defined]
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )

        ewb = EqualWidthBinning(n_bins=3, preserve_dataframe=True)
        ewb.fit(df_polars)
        result = ewb.transform(df_polars)

        # Should preserve Polars format
        assert isinstance(result, pl.DataFrame)  # type: ignore[name-defined]
        assert result.shape == df_polars.shape

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_mixed_data_types_in_dataframe(self):
        """Test DataFrame with mixed numeric types."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3, 4, 5],
                "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
                "int64_col": np.array([10, 20, 30, 40, 50], dtype=np.int64),
            }
        )

        ewb = EqualWidthBinning(n_bins=2)
        ewb.fit(df)
        result = ewb.transform(df)

        assert result.shape == df.shape


class TestEqualWidthBinningSklearnIntegration:
    """Test sklearn pipeline integration."""

    def test_sklearn_pipeline_basic(self):
        """Test EqualWidthBinning in sklearn Pipeline."""
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        y = np.array([0, 1, 0, 1])

        pipeline = Pipeline(
            [  # type: ignore[name-defined]
                ("binning", EqualWidthBinning(n_bins=2)),
                ("scaler", StandardScaler()),  # type: ignore[name-defined]
            ]
        )

        pipeline.fit(X, y)
        result = pipeline.transform(X)

        assert result.shape == X.shape
        # After binning and scaling, should be standardized
        assert np.allclose(result.mean(axis=0), 0, atol=1e-10)

    def test_sklearn_column_transformer(self):
        """Test EqualWidthBinning with ColumnTransformer."""
        X = np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0]])

        ct = ColumnTransformer(
            [  # type: ignore[name-defined]
                ("bin_first_two", EqualWidthBinning(n_bins=2), [0, 1]),
                ("scale_third", StandardScaler(), [2]),  # type: ignore[name-defined]
            ]
        )

        result = ct.fit_transform(X)

        # Convert to dense array if sparse
        if sparse is not None and sparse.issparse(result):
            result = result.toarray()  # type: ignore[attr-defined] # pragma: no cover

        # All columns should be numeric (ColumnTransformer typically converts to float64)
        assert np.issubdtype(result[:, 0].dtype, np.number)  # type: ignore[index]
        assert np.issubdtype(result[:, 1].dtype, np.number)  # type: ignore[index]
        assert np.issubdtype(result[:, 2].dtype, np.floating)  # type: ignore[index]

    def test_sklearn_get_set_params(self):
        """Test sklearn parameter interface."""
        ewb = EqualWidthBinning(n_bins=5, bin_range=(0, 100))

        # Test get_params
        params = ewb.get_params()
        assert params["n_bins"] == 5
        assert params["bin_range"] == (0, 100)

        # Test set_params
        ewb.set_params(n_bins=10, bin_range=(0, 200))
        assert ewb.n_bins == 10
        assert ewb.bin_range == (0, 200)

    def test_sklearn_clone_estimator(self):
        """Test sklearn clone functionality."""
        ewb_original = EqualWidthBinning(n_bins=7, bin_range=(0, 50))
        ewb_cloned = clone(ewb_original)

        # Cloned estimator should have same parameters
        assert ewb_cloned.n_bins == 7
        assert ewb_cloned.bin_range == (0, 50)
        # But should be different objects
        assert ewb_cloned is not ewb_original

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_sklearn_pipeline_with_dataframe(self):
        """Test pipeline with DataFrame input/output."""
        df = pd.DataFrame(
            {"feature1": [1.0, 2.0, 3.0, 4.0, 5.0], "feature2": [10.0, 20.0, 30.0, 40.0, 50.0]}
        )

        pipeline = Pipeline(
            [  # type: ignore[name-defined]
                ("binning", EqualWidthBinning(n_bins=3, preserve_dataframe=True))
            ]
        )

        result = pipeline.fit_transform(df)

        # Should preserve DataFrame format
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == list(df.columns)


class TestEqualWidthBinningFitGetParamsWorkflow:
    """Test fit → get_params → instantiation → binning workflow."""

    def test_fit_get_params_reinstantiate_workflow(self):
        """Test the complete workflow: fit → get_params → new instance → binning without fit."""
        # Original data and binning
        X_train = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        X_test = np.array([[1.5, 15.0], [2.5, 25.0], [3.5, 35.0]])

        # Step 1: Fit original transformer
        ewb_original = EqualWidthBinning(n_bins=3)
        ewb_original.fit(X_train)

        # Step 2: Get all parameters from fitted transformer
        params = ewb_original.get_params()

        # Step 3: Create new instance with all parameters
        ewb_new = EqualWidthBinning(**params)

        # Step 4: Use new instance for binning WITHOUT fitting
        result_original = ewb_original.transform(X_test)
        result_new = ewb_new.transform(X_test)

        # Results should be identical
        np.testing.assert_array_equal(result_original, result_new)

    def test_get_params_preserves_all_parameters(self):
        """Test that get_params preserves all necessary parameters."""
        ewb = EqualWidthBinning(
            n_bins=7, bin_range=(0, 100), clip=False, preserve_dataframe=True, fit_jointly=True
        )

        params = ewb.get_params()

        # Check that all EqualWidthBinning-specific params are preserved
        assert params["n_bins"] == 7
        assert params["bin_range"] == (0, 100)

        # Should also include parent class parameters
        assert "clip" in params
        assert "preserve_dataframe" in params
        assert "fit_jointly" in params

    def test_parameter_changes_reset_fitted_state(self):
        """Test that parameter changes properly reset fitted state."""
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

        ewb = EqualWidthBinning(n_bins=3)
        ewb.fit(X)

        # Should be fitted
        assert hasattr(ewb, "bin_edges_")
        assert hasattr(ewb, "bin_representatives_")

        # Change parameters via _handle_bin_params
        params = {"n_bins": 5, "bin_range": (0, 10)}
        reset_fitted = ewb._handle_bin_params(params)

        # Should indicate that fitted state needs reset
        assert reset_fitted is True
        assert ewb.n_bins == 5
        assert ewb.bin_range == (0, 10)

    def test_joint_vs_per_column_parameter_transfer(self):
        """Test parameter transfer works for both joint and per-column fitting."""
        X = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0]])

        # Test per-column fitting
        ewb_per_col = EqualWidthBinning(n_bins=3, fit_jointly=False)
        ewb_per_col.fit(X)

        params_per_col = ewb_per_col.get_params()
        ewb_new_per_col = EqualWidthBinning(**params_per_col)

        # Test joint fitting
        ewb_joint = EqualWidthBinning(n_bins=3, fit_jointly=True)
        ewb_joint.fit(X)

        params_joint = ewb_joint.get_params()
        ewb_new_joint = EqualWidthBinning(**params_joint)

        # Both should work for transformation
        X_test = np.array([[1.5, 150.0], [3.5, 350.0]])

        result_per_col_original = ewb_per_col.transform(X_test)
        result_per_col_new = ewb_new_per_col.transform(X_test)
        np.testing.assert_array_equal(result_per_col_original, result_per_col_new)

        result_joint_original = ewb_joint.transform(X_test)
        result_joint_new = ewb_new_joint.transform(X_test)
        np.testing.assert_array_equal(result_joint_original, result_joint_new)

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_dataframe_preservation_in_workflow(self):
        """Test DataFrame preservation through the fit-params-reinstantiate workflow."""
        df_train = pd.DataFrame({"col1": [1.0, 2.0, 3.0, 4.0], "col2": [10.0, 20.0, 30.0, 40.0]})
        df_test = pd.DataFrame({"col1": [1.5, 3.5], "col2": [15.0, 35.0]})

        # Fit with DataFrame preservation
        ewb_original = EqualWidthBinning(n_bins=2, preserve_dataframe=True)
        ewb_original.fit(df_train)

        # Transfer parameters
        params = ewb_original.get_params()
        ewb_new = EqualWidthBinning(**params)

        # Transform with new instance
        result_original = ewb_original.transform(df_test)
        result_new = ewb_new.transform(df_test)

        # Both should return DataFrames with same content
        assert isinstance(result_original, pd.DataFrame)
        assert isinstance(result_new, pd.DataFrame)
        pd.testing.assert_frame_equal(result_original, result_new)

    def test_user_providedbin_edges_(self):
        """Test EqualWidthBinning with user-provided bin_edges."""
        # Test data
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

        # Test with user-provided bin_edges
        ewb = EqualWidthBinning(bin_edges={0: [0, 2, 4, 6, 8], 1: [1, 3, 5, 7, 9]})

        # Should be able to transform without fitting
        result = ewb.transform(X)
        assert result.shape == (4, 2)

        # Check that the binning used the provided edges
        assert ewb._fitted is True  # Should be marked as fitted
        assert ewb.bin_edges_[0] == [0, 2, 4, 6, 8]
        assert ewb.bin_edges_[1] == [1, 3, 5, 7, 9]

        # Test with both bin_edges and bin_representatives
        ewb2 = EqualWidthBinning(
            bin_edges={0: [0, 2, 4], 1: [0, 3, 6]}, bin_representatives={0: [1, 3], 1: [1.5, 4.5]}
        )

        result2 = ewb2.transform(X)
        assert result2.shape == (4, 2)
        assert ewb2.bin_representatives_[0] == [1, 3]
        assert ewb2.bin_representatives_[1] == [1.5, 4.5]

    def test_fit_with_user_providedbin_edges_(self):
        """Test calling fit() when bin_edges are already provided.

        The fit() method should always calculate bin edges from the data,
        even when user-provided bin edges exist. User-provided bin edges
        serve as parameters for the binning algorithm but don't skip fitting.
        """
        # Test data
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

        # Create binner with pre-provided bin_edges (these serve as algorithm parameters)
        ewb = EqualWidthBinning(n_bins=5, bin_edges={0: [0, 2, 4, 6, 8], 1: [1, 3, 5, 7, 9]})

        # Reset fitted state to test the fit path
        ewb._fitted = False

        # Call fit - this should calculate new bin edges from the data
        ewb.fit(X)

        # Should be fitted and have calculated new edges from the data
        assert ewb._fitted is True

        # The bin edges should be calculated from the actual data range (1-7)
        # For equal width binning with 5 bins: [1.0, 2.2, 3.4, 4.6, 5.8, 7.0]
        expected_edges_col0 = [1.0, 2.2, 3.4, 4.6, 5.8, 7.0]
        _ = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]  # expected_edges_col1

        # Check that edges were calculated from data, not from user-provided values
        assert len(ewb.bin_edges_[0]) == 6  # 5 bins = 6 edges
        assert len(ewb.bin_edges_[1]) == 6  # 5 bins = 6 edges
        assert (
            abs(ewb.bin_edges_[0][0] - expected_edges_col0[0]) < 0.01
        )  # First edge should be close to min value
        assert (
            abs(ewb.bin_edges_[0][-1] - expected_edges_col0[-1]) < 0.01
        )  # Last edge should be close to max value

        # Should be able to transform
        result = ewb.transform(X)
        assert result.shape == (4, 2)

        # Test fit_transform as well
        ewb3 = EqualWidthBinning(n_bins=3, bin_edges={0: [0, 3, 6], 1: [1, 4, 7]})
        ewb3._fitted = False  # Reset to test fit path

        result3 = ewb3.fit_transform(X)
        assert result3.shape == (4, 2)
        assert ewb3._fitted is True

        # Should have 3 bins, so 4 edges (calculated from data, not user-provided)
        assert len(ewb3.bin_edges_[0]) == 4
        assert len(ewb3.bin_edges_[1]) == 4


def test_import_availability():
    """Test import availability flags."""
    assert isinstance(SKLEARN_AVAILABLE, bool)
    assert isinstance(POLARS_AVAILABLE, bool)

    # Test sparse availability when sklearn is available
    if SKLEARN_AVAILABLE:
        # sparse can be None or the actual module
        assert sparse is None or hasattr(sparse, "csr_matrix")
