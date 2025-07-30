"""
Comprehensive test suite for OneHotBinning transformer.

This module provides extensive testing for the OneHotBinning class, which creates
singleton bins for each unique value in the data. The tests cover initialization,
parameter validation, basic functionality, pandas/polars integration, sklearn
compatibility, error handling, and various edge cases.

Test Classes:
    TestOneHotBinningInitialization: Tests for parameter initialization and validation.
    TestOneHotBinningBasicFunctionality: Core fitting and transformation tests.
    TestOneHotBinningPandasIntegration: Tests for pandas DataFrame compatibility.
    TestOneHotBinningPolarsIntegration: Tests for polars DataFrame compatibility.
    TestOneHotBinningSklearnIntegration: Tests for sklearn pipeline compatibility.
    TestOneHotBinningWorkflows: Tests for complete ML workflows.
    TestOneHotBinningFitGetParamsWorkflow: Tests for parameter handling workflows.
    TestOneHotBinningRepr: Tests for string representation functionality.
    TestOneHotBinningErrorHandling: Tests for error conditions and edge cases.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from binning import PANDAS_AVAILABLE, pd, POLARS_AVAILABLE, pl
from binning.methods._onehot_binning import OneHotBinning
from binning.utils.errors import BinningError
from binning.utils.constants import MISSING_VALUE

try:
    from scipy import sparse

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    sparse = None
    SCIPY_AVAILABLE = False

# Import sklearn components
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

SKLEARN_AVAILABLE = True


class TestOneHotBinningInitialization:
    """Test OneHotBinning initialization and parameter validation.

    This test class verifies that the OneHotBinning transformer initializes
    correctly with both default and custom parameters, handles parameter
    validation properly, and maintains the correct state during initialization.
    """

    def test_default_initialization(self):
        """Test initialization with default parameters.

        Verifies that all default parameter values are set correctly and
        that the transformer is in the expected initial state.
        """
        binning = OneHotBinning()
        assert binning.max_unique_values == 100
        assert binning.preserve_dataframe is False  # Default from config
        assert binning.fit_jointly is False  # Always False for OneHotBinning
        assert binning.bin_spec is None
        assert binning.bin_representatives is None

    def test_custom_initialization(self):
        """Test initialization with custom parameters.

        Verifies that custom parameters are correctly accepted and stored,
        including max_unique_values, preserve_dataframe options, and
        pre-defined bin specifications.
        """
        binning = OneHotBinning(
            max_unique_values=50,
            preserve_dataframe=True,
            bin_spec={"col1": [{"singleton": 1.0}]},
            bin_representatives={"col1": [1.0]},
        )
        assert binning.max_unique_values == 50
        assert binning.preserve_dataframe is True
        assert binning.fit_jointly is False
        assert binning.bin_spec == {"col1": [{"singleton": 1.0}]}
        assert binning.bin_representatives == {"col1": [1.0]}

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test basic creation works - OneHotBinning doesn't validate max_unique_values in constructor
        binning = OneHotBinning(max_unique_values=-1)
        assert binning.max_unique_values == -1  # Should be set even if invalid

        # The validation happens during fit when it's actually used
        X = np.array([[1], [2]])
        with pytest.raises(ValueError):
            binning.fit(X)

    def test_get_binning_params(self):
        """Test _get_binning_params method."""
        binning = OneHotBinning(max_unique_values=50)
        params = binning._get_binning_params()
        assert "max_unique_values" in params
        assert params["max_unique_values"] == 50

    def test_handle_bin_params(self):
        """Test _handle_bin_params method."""
        binning = OneHotBinning()

        # Test updating max_unique_values
        params = {"max_unique_values": 200}
        reset_fitted = binning._handle_bin_params(params)
        assert reset_fitted is True
        assert binning.max_unique_values == 200
        assert params == {}  # Should be popped

        # Test with no relevant params
        params = {"some_other_param": 123}
        reset_fitted = binning._handle_bin_params(params)
        assert "some_other_param" in params  # Should not be popped


class TestOneHotBinningBasicFunctionality:
    """Test basic OneHotBinning functionality."""

    def test_simple_numeric_data(self):
        """Test with simple numeric data."""
        X = np.array([[1, 2], [2, 3], [1, 2], [3, 1]])
        binning = OneHotBinning()

        # Fit the binning
        binning.fit(X)

        # Check bin specifications
        assert 0 in binning.bin_spec_
        assert 1 in binning.bin_spec_

        # Column 0 should have bins for values 1, 2, 3
        col0_bins = binning.bin_spec_[0]
        expected_values = {1.0, 2.0, 3.0}
        actual_values = {bin_def["singleton"] for bin_def in col0_bins}
        assert actual_values == expected_values

        # Transform the data
        X_binned = binning.transform(X)
        assert X_binned.shape == X.shape

    def test_with_missing_values(self):
        """Test handling of missing values."""
        X = np.array([[1, 2], [np.nan, 3], [1, np.inf], [3, 1]])
        binning = OneHotBinning()

        binning.fit(X)
        X_binned = binning.transform(X)

        # Missing values should be handled appropriately
        assert not np.isnan(X_binned).all()
        assert X_binned.shape == X.shape

    def test_all_nan_column(self):
        """Test with column containing only NaN values."""
        X = np.array([[np.nan, 1], [np.nan, 2], [np.nan, 1]])
        binning = OneHotBinning()

        binning.fit(X)
        X_binned = binning.transform(X)

        assert X_binned.shape == X.shape

    def test_max_unique_values_limit(self):
        """Test max_unique_values parameter."""
        # Create data with many unique values
        X = np.arange(0, 200).reshape(-1, 1)
        binning = OneHotBinning(max_unique_values=10)

        with pytest.raises(ValueError, match="exceeds max_unique_values"):
            binning.fit(X)

    def test_non_numeric_data_error(self):
        """Test that non-numeric data raises appropriate error."""
        X = np.array([["a", "b"], ["c", "d"]], dtype=object)
        binning = OneHotBinning()

        with pytest.raises(ValueError, match="only supports numeric data"):
            binning.fit(X)

    def test_fit_transform(self):
        """Test fit_transform method."""
        X = np.array([[1, 2], [2, 3], [1, 2]])
        binning = OneHotBinning()

        X_binned = binning.fit_transform(X)
        assert X_binned.shape == X.shape
        assert hasattr(binning, "bin_spec_")

    def test_multiple_fits(self):
        """Test that multiple fits reset the state properly."""
        X1 = np.array([[1, 2], [2, 3]])
        X2 = np.array([[4, 5], [5, 6]])

        binning = OneHotBinning()

        # First fit
        binning.fit(X1)
        first_spec = binning.bin_spec_.copy()

        # Second fit should reset
        binning.fit(X2)
        second_spec = binning.bin_spec_

        # Specs should be different
        assert first_spec != second_spec


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
class TestOneHotBinningPandasIntegration:
    """Test OneHotBinning with pandas DataFrames."""

    def test_pandas_dataframe_basic(self):
        """Test basic functionality with pandas DataFrame."""
        df = pd.DataFrame({"A": [1, 2, 1, 3], "B": [2, 3, 2, 1]})

        binning = OneHotBinning(preserve_dataframe=True)

        # Fit and transform
        binning.fit(df)
        df_binned = binning.transform(df)

        # Should return DataFrame
        assert isinstance(df_binned, pd.DataFrame)
        assert list(df_binned.columns) == ["A", "B"]
        assert df_binned.shape == df.shape

    def test_pandas_dataframe_without_preserve(self):
        """Test pandas DataFrame without preserve_dataframe."""
        df = pd.DataFrame({"A": [1, 2, 1, 3], "B": [2, 3, 2, 1]})

        binning = OneHotBinning(preserve_dataframe=False)

        binning.fit(df)
        result = binning.transform(df)

        # Should return numpy array
        assert isinstance(result, np.ndarray)
        assert result.shape == df.shape

    def test_pandas_with_column_names(self):
        """Test that column names are preserved in bin specifications."""
        df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})

        binning = OneHotBinning()
        binning.fit(df)

        # Column names should be in bin_spec_
        assert "feature1" in binning.bin_spec_
        assert "feature2" in binning.bin_spec_

    def test_pandas_with_missing_values(self):
        """Test pandas DataFrame with missing values."""
        df = pd.DataFrame({"A": [1, np.nan, 2], "B": [3, 4, np.nan]})

        binning = OneHotBinning(preserve_dataframe=True)

        binning.fit(df)
        df_binned = binning.transform(df)

        assert isinstance(df_binned, pd.DataFrame)
        assert df_binned.shape == df.shape


@pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
class TestOneHotBinningPolarsIntegration:
    """Test OneHotBinning with Polars DataFrames."""

    def test_polars_dataframe_basic(self):
        """Test basic functionality with Polars DataFrame."""
        df = pl.DataFrame({"A": [1, 2, 1, 3], "B": [2, 3, 2, 1]})  # type: ignore[name-defined]

        binning = OneHotBinning(preserve_dataframe=True)

        # Fit and transform
        binning.fit(df)
        df_binned = binning.transform(df)

        # Should return Polars DataFrame
        assert isinstance(df_binned, pl.DataFrame)  # type: ignore[name-defined]
        assert df_binned.columns == ["A", "B"]
        assert df_binned.shape == df.shape

    def test_polars_dataframe_without_preserve(self):
        """Test Polars DataFrame without preserve_dataframe."""
        df = pl.DataFrame({"A": [1, 2, 1, 3], "B": [2, 3, 2, 1]})  # type: ignore[name-defined]

        binning = OneHotBinning(preserve_dataframe=False)

        binning.fit(df)
        result = binning.transform(df)

        # Should return numpy array
        assert isinstance(result, np.ndarray)
        assert result.shape == df.shape

    def test_polars_with_column_names(self):
        """Test that column names are preserved in bin specifications."""
        df = pl.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6]}  # type: ignore[name-defined]
        )

        binning = OneHotBinning()
        binning.fit(df)

        # Column names should be in bin_spec_
        assert "feature1" in binning.bin_spec_
        assert "feature2" in binning.bin_spec_


class TestOneHotBinningSklearnIntegration:
    """Test OneHotBinning with scikit-learn components."""

    def test_sklearn_pipeline_compatibility(self):
        """Test that OneHotBinning works in sklearn pipelines."""
        X = np.array([[1, 2], [2, 3], [1, 2], [3, 1]])

        # Create pipeline
        pipeline = Pipeline([("binning", OneHotBinning()), ("scaler", StandardScaler())])

        # Should work without errors
        X_transformed = pipeline.fit_transform(X)
        assert X_transformed.shape == X.shape

    def test_sklearn_column_transformer(self):
        """Test OneHotBinning with ColumnTransformer."""
        X = np.array([[1, 2, 3], [2, 3, 4], [1, 2, 3]])

        # Create ColumnTransformer
        ct = ColumnTransformer(
            [("binning", OneHotBinning(), [0, 1]), ("scaler", StandardScaler(), [2])]
        )

        X_transformed = ct.fit_transform(X)
        # Note: ColumnTransformer typically returns float64
        # Just check that transformation was successful and shape is correct
        assert X_transformed is not None
        assert X_transformed.shape[1] == 3  # Same number of features

    def test_sklearn_feature_names_out(self):
        """Test get_feature_names_out method for sklearn compatibility."""
        binning = OneHotBinning()

        # Check if method exists and works
        if hasattr(binning, "get_feature_names_out"):
            X = np.array([[1, 2], [2, 3]])
            binning.fit(X)
            feature_names = binning.get_feature_names_out()
            assert len(feature_names) == 2


class TestOneHotBinningWorkflows:
    """Test complete workflows and edge cases."""

    def test_single_column_workflow(self):
        """Test complete workflow with single column."""
        X = np.array([[1], [2], [1], [3], [2]])
        binning = OneHotBinning()

        # Fit
        binning.fit(X)
        assert hasattr(binning, "bin_spec_")
        assert 0 in binning.bin_spec_

        # Transform
        X_binned = binning.transform(X)
        assert X_binned.shape == X.shape

        # Fit-transform
        X_binned2 = binning.fit_transform(X)
        np.testing.assert_array_equal(X_binned, X_binned2)

    def test_multi_column_workflow(self):
        """Test complete workflow with multiple columns."""
        X = np.array([[1, 10], [2, 20], [1, 10], [3, 30]])
        binning = OneHotBinning()

        # Fit
        binning.fit(X)
        assert 0 in binning.bin_spec_
        assert 1 in binning.bin_spec_

        # Each column should have appropriate bins
        col0_values = {bin_def["singleton"] for bin_def in binning.bin_spec_[0]}
        col1_values = {bin_def["singleton"] for bin_def in binning.bin_spec_[1]}

        assert col0_values == {1.0, 2.0, 3.0}
        assert col1_values == {10.0, 20.0, 30.0}

        # Transform
        X_binned = binning.transform(X)
        assert X_binned.shape == X.shape

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        X = np.array([]).reshape(0, 2)
        binning = OneHotBinning()

        # Empty data should be handled gracefully
        binning.fit(X)
        X_transformed = binning.transform(X)
        assert X_transformed.shape == (0, 2)

    def test_single_value_column(self):
        """Test column with single unique value."""
        X = np.array([[5], [5], [5]])
        binning = OneHotBinning()

        binning.fit(X)
        assert len(binning.bin_spec_[0]) == 1
        assert binning.bin_spec_[0][0]["singleton"] == 5.0

        X_binned = binning.transform(X)
        assert X_binned.shape == X.shape

    def test_large_dataset_workflow(self):
        """Test with larger dataset."""
        np.random.seed(42)
        X = np.random.randint(1, 20, size=(1000, 3))

        binning = OneHotBinning()

        # Should handle larger datasets efficiently
        binning.fit(X)
        X_binned = binning.transform(X)

        assert X_binned.shape == X.shape
        assert len(binning.bin_spec_) == 3

    def test_parameter_updates(self):
        """Test parameter updates after initialization."""
        binning = OneHotBinning(max_unique_values=50)

        # Update parameters
        binning.set_params(max_unique_values=200)
        assert binning.max_unique_values == 200

        # Should work with new parameters
        X = np.arange(0, 100).reshape(-1, 1)
        binning.fit(X)  # Should not raise max_unique_values error

    def test_bin_representatives_consistency(self):
        """Test that bin representatives are consistent with bin specs."""
        X = np.array([[1, 10], [2, 20], [3, 30]])
        binning = OneHotBinning()

        binning.fit(X)

        for col_id in binning.bin_spec_:
            bin_spec = binning.bin_spec_[col_id]
            bin_repr = binning.bin_representatives_[col_id]

            assert len(bin_spec) == len(bin_repr)

            # Each representative should match the singleton value
            for i, bin_def in enumerate(bin_spec):
                assert bin_repr[i] == bin_def["singleton"]


class TestOneHotBinningFitGetParamsWorkflow:
    """Test fit → get_params → instantiation → binning workflow."""

    def test_fit_get_params_reinstantiate_workflow(self):
        """Test the complete workflow: fit → get_params → new instance → binning without fit."""
        # Original data and binning
        X_train = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        X_test = np.array([[1.5, 15.0], [2.5, 25.0], [3.5, 35.0]])

        # Step 1: Fit original transformer
        binning_original = OneHotBinning(max_unique_values=50)
        binning_original.fit(X_train)

        # Step 2: Get all parameters from fitted transformer
        params = binning_original.get_params()

        # Step 3: Create new instance with all parameters
        binning_new = OneHotBinning(**params)

        # Step 4: Use new instance for binning WITHOUT fitting
        result_original = binning_original.transform(X_test)
        result_new = binning_new.transform(X_test)

        # Results should be identical
        np.testing.assert_array_equal(result_original, result_new)

    def test_get_params_preserves_all_parameters(self):
        """Test that get_params preserves all necessary parameters."""
        binning = OneHotBinning(
            max_unique_values=25,
            preserve_dataframe=True,
        )

        # Fit to get bin specifications
        X = np.array([[1, 2], [2, 3], [1, 2]])
        binning.fit(X)

        params = binning.get_params()

        # Check that all OneHotBinning-specific params are preserved
        assert params["max_unique_values"] == 25
        assert params["preserve_dataframe"] is True

        # Should also include fitted bin specifications
        assert "bin_spec" in params
        assert "bin_representatives" in params

        # Fitted bin specs should match current state
        assert params["bin_spec"] == binning.bin_spec_
        assert params["bin_representatives"] == binning.bin_representatives_

    def test_reinstantiate_without_fit_workflow(self):
        """Test that reinstantiated transformer works without calling fit."""
        X_fit = np.array([[1, 10], [2, 20], [3, 30]])
        X_transform = np.array([[2, 20], [1, 10], [3, 30]])

        # Original workflow
        original_binning = OneHotBinning()
        original_binning.fit(X_fit)

        # Get params and reinstantiate
        params = original_binning.get_params()
        new_binning = OneHotBinning(**params)

        # Should be able to transform without fitting
        result = new_binning.transform(X_transform)

        assert result.shape == X_transform.shape
        assert new_binning.is_fitted_

        # Results should match original
        expected = original_binning.transform(X_transform)
        np.testing.assert_array_equal(result, expected)


class TestOneHotBinningRepr:
    """Test string representation and debugging features."""

    def test_str_representation(self):
        """Test __str__ method."""
        binning = OneHotBinning(max_unique_values=50)
        str_repr = str(binning)
        assert "OneHotBinning" in str_repr
        assert "max_unique_values=50" in str_repr

    def test_repr_representation(self):
        """Test __repr__ method."""
        binning = OneHotBinning()
        repr_str = repr(binning)
        assert "OneHotBinning" in repr_str

    def test_fitted_representation(self):
        """Test representation after fitting."""
        X = np.array([[1, 2], [2, 3]])
        binning = OneHotBinning()
        binning.fit(X)

        str_repr = str(binning)
        # After fitting, it should show fitted parameters or at least not crash
        assert "OneHotBinning" in str_repr
        # Check that we can access fitted state
        assert hasattr(binning, "bin_spec_")
        assert len(binning.bin_spec_) > 0


class TestOneHotBinningErrorHandling:
    """Test error handling and edge cases."""

    def test_transform_before_fit_error(self):
        """Test that transform before fit raises appropriate error."""
        X = np.array([[1, 2], [2, 3]])
        binning = OneHotBinning()

        with pytest.raises(RuntimeError, match="not fitted yet"):
            binning.transform(X)

    def test_inconsistent_column_count_error(self):
        """Test error when transform data has different column count."""
        X_fit = np.array([[1, 2], [2, 3]])
        X_transform = np.array([[1, 2, 3], [2, 3, 4]])

        binning = OneHotBinning()
        binning.fit(X_fit)

        with pytest.raises((ValueError, BinningError)):
            binning.transform(X_transform)

    def test_invalid_data_types(self):
        """Test with various invalid data types."""
        binning = OneHotBinning()

        # Test with string data
        with pytest.raises(ValueError, match="only supports numeric data"):
            binning.fit([["a", "b"], ["c", "d"]])

        # Test with truly non-numeric strings (not convertible to float)
        X_mixed = np.array([[1, "abc"], [2, "def"]], dtype=object)
        with pytest.raises(ValueError, match="only supports numeric data"):
            binning.fit(X_mixed)

    def test_extreme_values_handling(self):
        """Test handling of extreme values."""
        X = np.array([[np.finfo(float).max, 1], [np.finfo(float).min, 2], [0, 3]])

        binning = OneHotBinning()

        # Should handle extreme values
        binning.fit(X)
        X_binned = binning.transform(X)
        assert X_binned.shape == X.shape


def test_import_availability():
    """Test import availability flags."""
    assert isinstance(POLARS_AVAILABLE, bool)
    assert isinstance(SKLEARN_AVAILABLE, bool)
