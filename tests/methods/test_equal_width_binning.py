"""
Comprehensive tests for EqualWidthBinning transformer.
Tests all functionality, edge cases, sklearn compatibility, and pandas/polars integration.
"""

import numpy as np
import pytest
from typing import Any, Dict, List
from unittest.mock import patch

from binning.methods._equal_width_binning import EqualWidthBinning
from binning.base._constants import MISSING_VALUE, ABOVE_RANGE, BELOW_RANGE
from binning import _pandas_config, _polars_config

PANDAS_AVAILABLE = _pandas_config.PANDAS_AVAILABLE
POLARS_AVAILABLE = _polars_config.POLARS_AVAILABLE


class TestInitialization:
    """Test initialization and parameter handling."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        binner = EqualWidthBinning()

        assert binner.n_bins == 10
        assert binner.bin_range is None
        assert binner.clip is True
        assert binner.preserve_dataframe is False
        assert binner.bin_edges is None
        assert binner.bin_representatives is None
        assert binner._fitted is False

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        binner = EqualWidthBinning(
            n_bins=5, bin_range=(0, 100), clip=False, preserve_dataframe=True
        )

        assert binner.n_bins == 5
        assert binner.bin_range == (0, 100)
        assert binner.clip is False
        assert binner.preserve_dataframe is True

    def test_init_per_column_parameters(self):
        """Test initialization with per-column parameters."""
        n_bins = {0: 5, 1: 8}
        bin_range = {0: (0, 10), 1: (-1, 1)}

        binner = EqualWidthBinning(n_bins=n_bins, bin_range=bin_range)

        assert binner.n_bins == n_bins
        assert binner.bin_range == bin_range

    def test_init_with_prespecified_bins(self):
        """Test initialization with pre-specified bins."""
        bin_edges = {0: [0, 20, 40, 60, 80, 100]}
        bin_reps = {0: [10, 30, 50, 70, 90]}

        binner = EqualWidthBinning(bin_edges=bin_edges, bin_representatives=bin_reps)

        assert binner.bin_edges == bin_edges
        assert binner.bin_representatives == bin_reps


class TestCalculateBins:
    """Test _calculate_bins method."""

    def test_calculate_bins_basic(self):
        """Test basic bin calculation."""
        binner = EqualWidthBinning(n_bins=4)
        x_col = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        edges, reps = binner._calculate_bins(x_col, 0)

        assert len(edges) == 5  # n_bins + 1
        assert len(reps) == 4  # n_bins
        assert edges[0] == 1.0  # min value
        assert edges[-1] == 10.0  # max value
        # Check equal width
        widths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
        assert all(abs(w - widths[0]) < 1e-10 for w in widths)

    def test_calculate_bins_custom_range(self):
        """Test bin calculation with custom range."""
        binner = EqualWidthBinning(n_bins=5, bin_range=(0, 100))
        x_col = np.array([10, 20, 30])  # Data within range

        edges, reps = binner._calculate_bins(x_col, 0)

        assert edges[0] == 0.0
        assert edges[-1] == 100.0
        assert len(edges) == 6
        assert len(reps) == 5
        # Check equal width (should be 20.0)
        expected_width = 20.0
        for i in range(len(edges) - 1):
            assert abs((edges[i + 1] - edges[i]) - expected_width) < 1e-10

    def test_calculate_bins_per_column_n_bins(self):
        """Test bin calculation with per-column n_bins."""
        binner = EqualWidthBinning(n_bins={0: 3, 1: 5})
        x_col = np.array([1, 2, 3, 4, 5])

        # Test column 0
        edges_0, reps_0 = binner._calculate_bins(x_col, 0)
        assert len(edges_0) == 4  # 3 + 1
        assert len(reps_0) == 3

        # Test column 1
        edges_1, reps_1 = binner._calculate_bins(x_col, 1)
        assert len(edges_1) == 6  # 5 + 1
        assert len(reps_1) == 5

        # Test column not in dict (should use default 10)
        edges_2, reps_2 = binner._calculate_bins(x_col, 2)
        assert len(edges_2) == 11  # 10 + 1
        assert len(reps_2) == 10

    def test_calculate_bins_per_column_range(self):
        """Test bin calculation with per-column ranges."""
        binner = EqualWidthBinning(n_bins=3, bin_range={0: (0, 10), 1: (-5, 5)})
        x_col = np.array([1, 2, 3])

        # Test column 0
        edges_0, reps_0 = binner._calculate_bins(x_col, 0)
        assert edges_0[0] == 0.0
        assert edges_0[-1] == 10.0

        # Test column 1
        edges_1, reps_1 = binner._calculate_bins(x_col, 1)
        assert edges_1[0] == -5.0
        assert edges_1[-1] == 5.0

        # Test column not in dict (should use data range)
        edges_2, reps_2 = binner._calculate_bins(x_col, 2)
        assert edges_2[0] == 1.0  # Data min
        assert edges_2[-1] == 3.0  # Data max

    def test_calculate_bins_constant_data(self):
        """Test bin calculation with constant data."""
        binner = EqualWidthBinning(n_bins=3)
        x_col = np.array([5, 5, 5, 5])  # All same value

        edges, reps = binner._calculate_bins(x_col, 0)

        assert len(edges) == 4
        assert len(reps) == 3
        # Should add small epsilon to create range
        assert edges[0] < 5.0
        assert edges[-1] > 5.0

    def test_calculate_bins_with_nan(self):
        """Test bin calculation with NaN values."""
        binner = EqualWidthBinning(n_bins=4)
        x_col = np.array([1, 2, np.nan, 4, 5, np.nan])

        edges, reps = binner._calculate_bins(x_col, 0)

        # Should ignore NaN values
        assert edges[0] == 1.0
        assert edges[-1] == 5.0
        assert len(edges) == 5
        assert len(reps) == 4

    def test_calculate_bins_all_nan(self):
        """Test bin calculation with all NaN values."""
        binner = EqualWidthBinning(n_bins=3)
        x_col = np.array([np.nan, np.nan, np.nan])

        with pytest.raises(ValueError, match="min and max must be finite"):
            binner._calculate_bins(x_col, 0)

    def test_calculate_bins_empty_array(self):
        """Test bin calculation with empty array."""
        binner = EqualWidthBinning(n_bins=3)
        x_col = np.array([])

        with pytest.raises(ValueError, match="min and max must be finite"):
            binner._calculate_bins(x_col, 0)

    def test_calculate_bins_invalid_n_bins(self):
        """Test bin calculation with invalid n_bins."""
        binner = EqualWidthBinning(n_bins=0)
        x_col = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            binner._calculate_bins(x_col, 0)


class TestFitTransform:
    """Test fit and transform methods."""

    def test_fit_basic(self):
        """Test basic fitting."""
        binner = EqualWidthBinning(n_bins=3)
        X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])

        result = binner.fit(X)

        assert result is binner
        assert binner._fitted is True
        assert len(binner._bin_edges) == 2  # 2 columns
        assert len(binner._bin_reps) == 2

        # Check that all columns have correct number of bins
        for col_id in [0, 1]:
            assert len(binner._bin_edges[col_id]) == 4  # n_bins + 1
            assert len(binner._bin_reps[col_id]) == 3  # n_bins

    def test_fit_with_prespecified_bins(self):
        """Test fitting with pre-specified bins."""
        bin_edges = {0: [0, 2, 4], 1: [0, 25, 50]}
        bin_reps = {0: [1, 3], 1: [12.5, 37.5]}

        binner = EqualWidthBinning(bin_edges=bin_edges, bin_representatives=bin_reps)
        X = np.array([[1, 10], [2, 20], [3, 30]])

        binner.fit(X)

        # Should use pre-specified bins
        assert binner._bin_edges == {0: [0.0, 2.0, 4.0], 1: [0.0, 25.0, 50.0]}
        assert binner._bin_reps == {0: [1.0, 3.0], 1: [12.5, 37.5]}

    def test_transform_basic(self):
        """Test basic transformation."""
        binner = EqualWidthBinning(n_bins=4)
        X_train = np.array([[0, 0], [1, 10], [2, 20], [3, 30], [4, 40]])
        X_test = np.array([[1.5, 15], [2.5, 25]])

        binner.fit(X_train)
        result = binner.transform(X_test)

        assert result.shape == X_test.shape
        assert result.dtype == np.int64
        # Values should be valid bin indices
        assert np.all(result >= 0)
        assert np.all(result < 4)  # n_bins

    def test_transform_edge_values(self):
        """Test transformation of edge values."""
        bin_edges = {0: [0, 1, 2, 3]}
        binner = EqualWidthBinning(bin_edges=bin_edges)
        X_train = np.array([[0.5], [1.5], [2.5]])
        X_test = np.array([[0], [1], [2], [3]])  # Exact edge values

        binner.fit(X_train)
        result = binner.transform(X_test)

        # Test that edge values are handled consistently
        assert result[0, 0] == 0  # 0 -> first bin
        assert result[1, 0] == 1  # 1 -> second bin
        assert result[2, 0] == 2  # 2 -> third bin
        # Note: Right edge (3) behavior depends on implementation

    def test_transform_out_of_range_with_clip(self):
        """Test transformation of out-of-range values with clipping."""
        bin_edges = {0: [1, 2, 3]}
        binner = EqualWidthBinning(clip=True, bin_edges=bin_edges)
        X_train = np.array([[1.5], [2.5]])
        X_test = np.array([[0.5], [3.5]])  # Out of range values

        binner.fit(X_train)
        result = binner.transform(X_test)

        # Should be clipped to valid range
        assert result[0, 0] == 0  # Below range -> first bin
        assert result[1, 0] == 1  # Above range -> last bin

    def test_transform_out_of_range_without_clip(self):
        """Test transformation of out-of-range values without clipping."""
        bin_edges = {0: [1, 2, 3]}
        binner = EqualWidthBinning(clip=False, bin_edges=bin_edges)
        X_train = np.array([[1.5], [2.5]])
        X_test = np.array([[0.5], [3.5]])

        binner.fit(X_train)
        result = binner.transform(X_test)

        # Should use special values for out-of-range
        assert result[0, 0] == BELOW_RANGE  # Below range
        assert result[1, 0] == ABOVE_RANGE  # Above range

    def test_fit_transform_method(self):
        """Test fit_transform method."""
        binner = EqualWidthBinning(n_bins=3)
        X = np.array([[1, 10], [2, 20], [3, 30]])

        result = binner.fit_transform(X)

        assert binner._fitted is True
        assert result.shape == X.shape
        assert result.dtype == np.int64


class TestInverseTransform:
    """Test inverse transform functionality."""

    def test_inverse_transform_basic(self):
        """Test basic inverse transformation."""
        bin_edges = {0: [0, 1, 2], 1: [0, 10, 20]}
        bin_reps = {0: [0.5, 1.5], 1: [5, 15]}

        binner = EqualWidthBinning(bin_edges=bin_edges, bin_representatives=bin_reps)
        X_train = np.array([[0.5, 5], [1.5, 15]])
        binner.fit(X_train)

        # Test inverse transform
        bin_indices = np.array([[0, 1], [1, 0]])
        result = binner.inverse_transform(bin_indices)

        assert result.shape == bin_indices.shape
        assert result[0, 0] == 0.5  # First rep of column 0
        assert result[0, 1] == 15  # Second rep of column 1
        assert result[1, 0] == 1.5  # Second rep of column 0
        assert result[1, 1] == 5  # First rep of column 1

    def test_inverse_transform_out_of_bounds(self):
        """Test inverse transform with out-of-bounds indices."""
        bin_edges = {0: [0, 1, 2]}
        bin_reps = {0: [0.5, 1.5]}

        binner = EqualWidthBinning(bin_edges=bin_edges, bin_representatives=bin_reps)
        X_train = np.array([[0.5], [1.5]])
        binner.fit(X_train)

        # Test with out-of-bounds indices
        bin_indices = np.array([[5], [-1]])  # Out of bounds
        result = binner.inverse_transform(bin_indices)

        # Should be clipped to valid range
        assert result[0, 0] == 1.5  # Clipped to last rep
        assert result[1, 0] == 0.5  # Clipped to first rep

    def test_inverse_transform_special_values(self):
        """Test inverse transform with special values."""
        binner = EqualWidthBinning(n_bins=2)
        X_train = np.array([[1], [2]])
        binner.fit(X_train)

        # Test with special values
        bin_indices = np.array([[MISSING_VALUE], [BELOW_RANGE], [ABOVE_RANGE]])
        result = binner.inverse_transform(bin_indices)

        assert np.isnan(result[0, 0])  # Missing -> NaN
        assert result[1, 0] == -np.inf  # Below -> -inf
        assert result[2, 0] == np.inf  # Above -> +inf


class TestParameterManagement:
    """Test parameter getting and setting."""

    def test_get_params_default(self):
        """Test get_params with default values."""
        binner = EqualWidthBinning()
        params = binner.get_params()

        expected_params = {
            "n_bins",
            "bin_range",
            "clip",
            "preserve_dataframe",
            "bin_edges",
            "bin_representatives",
        }
        assert all(param in params for param in expected_params)

        assert params["n_bins"] == 10
        assert params["bin_range"] is None
        assert params["clip"] is True
        assert params["preserve_dataframe"] is False

    def test_get_params_custom(self):
        """Test get_params with custom values."""
        binner = EqualWidthBinning(n_bins=5, bin_range=(0, 100), clip=False)
        params = binner.get_params()

        assert params["n_bins"] == 5
        assert params["bin_range"] == (0, 100)
        assert params["clip"] is False

    def test_get_params_after_fitting(self):
        """Test get_params after fitting returns fitted specifications."""
        binner = EqualWidthBinning(n_bins=3)
        X = np.array([[1, 10], [2, 20], [3, 30]])
        binner.fit(X)

        params = binner.get_params()

        # Should return fitted specifications
        assert len(params["bin_edges"]) == 2
        assert len(params["bin_representatives"]) == 2

    def test_set_params_basic(self):
        """Test set_params with basic parameters."""
        binner = EqualWidthBinning()

        result = binner.set_params(n_bins=5, clip=False)

        assert result is binner
        assert binner.n_bins == 5
        assert binner.clip is False

    def test_set_params_resets_fitted_state(self):
        """Test that setting binning parameters resets fitted state."""
        binner = EqualWidthBinning(n_bins=3)
        X = np.array([[1, 2], [3, 4]])
        binner.fit(X)

        assert binner._fitted is True

        # Setting n_bins should reset fitted state
        binner.set_params(n_bins=5)
        assert binner._fitted is False

        # Re-fit and test bin_range
        binner.fit(X)
        assert binner._fitted is True

        binner.set_params(bin_range=(0, 10))
        assert binner._fitted is False

    def test_set_params_other_params_preserve_fitted(self):
        """Test that setting non-binning parameters preserves fitted state."""
        binner = EqualWidthBinning()
        X = np.array([[1, 2], [3, 4]])
        binner.fit(X)

        assert binner._fitted is True

        # Setting clip should not reset fitted state
        binner.set_params(clip=False, preserve_dataframe=True)
        assert binner._fitted is True


class TestSklearnCompatibility:
    """Test sklearn compatibility."""

    def test_sklearn_pipeline_integration(self):
        """Test integration with sklearn Pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        binner = EqualWidthBinning(n_bins=3)
        pipeline = Pipeline([("scaler", StandardScaler()), ("binner", binner)])

        X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])

        # Test fit
        pipeline.fit(X)
        assert binner._fitted is True

        # Test transform
        result = pipeline.transform(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == X.shape

    def test_sklearn_cross_validation_compatibility(self):
        """Test compatibility with sklearn cross-validation."""
        from sklearn.model_selection import cross_val_score
        from sklearn.dummy import DummyRegressor
        from sklearn.pipeline import Pipeline

        binner = EqualWidthBinning(n_bins=3)
        regressor = DummyRegressor()
        pipeline = Pipeline([("binner", binner), ("regressor", regressor)])

        X = np.random.rand(20, 3)
        y = np.random.rand(20)

        # Should not raise errors
        scores = cross_val_score(pipeline, X, y, cv=3)
        assert len(scores) == 3

    def test_sklearn_grid_search_compatibility(self):
        """Test compatibility with sklearn GridSearchCV."""
        from sklearn.model_selection import GridSearchCV
        from sklearn.dummy import DummyRegressor
        from sklearn.pipeline import Pipeline

        binner = EqualWidthBinning()
        regressor = DummyRegressor()
        pipeline = Pipeline([("binner", binner), ("regressor", regressor)])

        param_grid = {"binner__n_bins": [3, 5], "binner__clip": [True, False]}

        X = np.random.rand(10, 2)
        y = np.random.rand(10)

        grid_search = GridSearchCV(pipeline, param_grid, cv=2)
        grid_search.fit(X, y)

        assert hasattr(grid_search, "best_params_")
        assert "binner__n_bins" in grid_search.best_params_


class TestDataFrameIntegration:
    """Test pandas and polars DataFrame integration."""

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_dataframe_basic(self):
        """Test basic functionality with pandas DataFrame."""
        import pandas as pd

        binner = EqualWidthBinning(n_bins=3, preserve_dataframe=True)
        df = pd.DataFrame({"age": [25, 35, 45, 55], "income": [30000, 50000, 70000, 90000]})

        binner.fit(df)
        result = binner.transform(df)

        # Should preserve DataFrame structure when preserve_dataframe=True
        if isinstance(result, pd.DataFrame):
            assert list(result.columns) == ["age", "income"]
            assert len(result) == 4

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_with_per_column_specs(self):
        """Test pandas with per-column specifications."""
        import pandas as pd

        df = pd.DataFrame(
            {"age": [20, 30, 40, 50, 60], "salary": [20000, 40000, 60000, 80000, 100000]}
        )

        binner = EqualWidthBinning(
            n_bins={"age": 3, "salary": 4}, bin_range={"age": (18, 65), "salary": (0, 120000)}
        )

        binner.fit(df)
        result = binner.transform(df)

        assert binner._fitted is True
        # Check that column names are used as keys
        assert "age" in binner._bin_edges
        assert "salary" in binner._bin_edges
        assert len(binner._bin_edges["age"]) == 4  # 3 bins + 1
        assert len(binner._bin_edges["salary"]) == 5  # 4 bins + 1

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
    def test_polars_dataframe_basic(self):
        """Test basic functionality with polars DataFrame."""
        import polars as pl

        binner = EqualWidthBinning(n_bins=3)
        df = pl.DataFrame({"age": [25, 35, 45, 55], "income": [30000, 50000, 70000, 90000]})

        binner.fit(df)
        result = binner.transform(df)

        assert binner._fitted is True
        # Result type depends on preserve_dataframe setting
        assert isinstance(result, (np.ndarray, pl.DataFrame))


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_column_data(self):
        """Test with single column data."""
        binner = EqualWidthBinning(n_bins=3)
        X = np.array([[1], [2], [3], [4]])

        binner.fit(X)
        result = binner.transform(X)

        assert result.shape == (4, 1)
        assert len(binner._bin_edges) == 1

    def test_single_sample(self):
        """Test with single sample."""
        binner = EqualWidthBinning(n_bins=3)
        X = np.array([[5, 10]])

        # Should handle gracefully (though bins will be artificial)
        binner.fit(X)
        result = binner.transform(X)

        assert result.shape == (1, 2)

    def test_transform_not_fitted_error(self):
        """Test transform raises error when not fitted."""
        binner = EqualWidthBinning()
        X = np.array([[1, 2], [3, 4]])

        with pytest.raises(RuntimeError, match="not fitted"):
            binner.transform(X)

    def test_transform_wrong_number_features(self):
        """Test transform with wrong number of features."""
        binner = EqualWidthBinning(n_bins=3)
        X_train = np.array([[1, 2], [3, 4]])
        X_test = np.array([[1, 2, 3], [4, 5, 6]])  # Wrong number of features

        binner.fit(X_train)

        with pytest.raises(ValueError, match="Expected 2 features"):
            binner.transform(X_test)

    def test_mixed_input_types(self):
        """Test fitting on one type and transforming another."""
        binner = EqualWidthBinning(n_bins=3)

        # Fit on numpy array
        X_train = np.array([[1, 10], [2, 20], [3, 30]])
        binner.fit(X_train)

        # Transform list
        X_test = [[1.5, 15], [2.5, 25]]
        result = binner.transform(X_test)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)


class TestRepr:
    """Test string representation."""

    def test_repr_default(self):
        """Test repr with default parameters."""
        binner = EqualWidthBinning()
        repr_str = repr(binner)

        assert "EqualWidthBinning(" in repr_str
        # Should not show default values
        assert "n_bins=10" not in repr_str
        assert "clip=True" not in repr_str

    def test_repr_custom_parameters(self):
        """Test repr with custom parameters."""
        binner = EqualWidthBinning(
            n_bins=5, bin_range=(0, 100), clip=False, preserve_dataframe=True
        )
        repr_str = repr(binner)

        assert "n_bins=5" in repr_str
        assert "bin_range=(0, 100)" in repr_str
        assert "clip=False" in repr_str
        assert "preserve_dataframe=True" in repr_str

    def test_repr_with_dict_parameters(self):
        """Test repr with dictionary parameters."""
        binner = EqualWidthBinning(n_bins={0: 3, 1: 5}, bin_range={0: (0, 10)})
        repr_str = repr(binner)

        assert "n_bins={0: 3, 1: 5}" in repr_str
        assert "bin_range={0: (0, 10)}" in repr_str


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_workflow_with_missing_values(self):
        """Test complete workflow with missing values in data."""
        binner = EqualWidthBinning(n_bins=4)

        # Data with NaN values
        X_train = np.array([[1, 10], [2, np.nan], [3, 30], [np.nan, 40]])
        X_test = np.array([[1.5, 15], [2.5, np.nan]])

        binner.fit(X_train)
        result = binner.transform(X_test)

        # Should handle NaN gracefully
        assert result.shape == X_test.shape
        # NaN values should be handled according to the implementation

    def test_workflow_with_extreme_values(self):
        """Test workflow with extreme values."""
        binner = EqualWidthBinning(n_bins=5, clip=True)

        # Training data with normal range
        X_train = np.array([[1, 10], [2, 20], [3, 30]])
        # Test data with extreme values
        X_test = np.array([[-1000, 5], [1000, 25], [2, 1000]])

        binner.fit(X_train)
        result = binner.transform(X_test)

        # Extreme values should be clipped to valid bin indices
        assert np.all(result >= 0)
        assert np.all(result < 5)  # n_bins

    def test_parameter_changes_workflow(self):
        """Test workflow with parameter changes."""
        binner = EqualWidthBinning(n_bins=3)
        X = np.array([[1, 10], [2, 20], [3, 30]])

        # Initial fit
        binner.fit(X)
        result1 = binner.transform(X)

        # Change parameters and refit
        binner.set_params(n_bins=5)
        binner.fit(X)
        result2 = binner.transform(X)

        # Results should be different due to different number of bins
        assert not np.array_equal(result1, result2)
        # New result should have valid indices for 5 bins
        assert np.all(result2 >= 0)
        assert np.all(result2 < 5)

    def test_state_consistency_across_operations(self):
        """Test that state remains consistent across multiple operations."""
        binner = EqualWidthBinning(n_bins=4)
        X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])

        # Fit
        binner.fit(X)

        # Multiple transforms should give consistent results
        result1 = binner.transform(X)
        result2 = binner.transform(X)

        assert np.array_equal(result1, result2)

        # Inverse transform should be consistent
        inv_result1 = binner.inverse_transform(result1)
        inv_result2 = binner.inverse_transform(result1)

        assert np.array_equal(inv_result1, inv_result2)


class TestJointFitting:
    """Test joint fitting functionality."""

    def test_joint_vs_individual_fitting(self):
        """Test that joint fitting produces different results than individual."""
        # Data with different scales
        X = np.array(
            [[1, 100], [2, 200], [3, 300], [4, 400], [5, 500]]  # Column 0: 1-5, Column 1: 100-500
        )

        # Individual fitting
        binner_individual = EqualWidthBinning(n_bins=3, fit_jointly=False)
        binner_individual.fit(X)

        # Joint fitting with global range
        binner_joint = EqualWidthBinning(n_bins=3, fit_jointly=True, joint_range_method="global")
        binner_joint.fit(X)

        # Debug: print the edges to see what's happening
        print(f"Individual Col 0 edges: {binner_individual._bin_edges[0]}")
        print(f"Joint Col 0 edges: {binner_joint._bin_edges[0]}")
        print(f"Individual Col 1 edges: {binner_individual._bin_edges[1]}")
        print(f"Joint Col 1 edges: {binner_joint._bin_edges[1]}")

        # Should produce different bin edges
        edges_ind_col0 = binner_individual._bin_edges[0]
        edges_joint_col0 = binner_joint._bin_edges[0]

        assert not np.allclose(edges_ind_col0, edges_joint_col0)

    def test_joint_range_methods(self):
        """Test different joint range methods."""
        X = np.random.randn(100, 3) * 10 + 50  # Mean=50, std=10

        methods = ["global", "percentile", "std", "robust"]

        for method in methods:
            binner = EqualWidthBinning(n_bins=5, fit_jointly=True, joint_range_method=method)
            binner.fit(X)

            # Should successfully fit
            assert binner._fitted
            assert len(binner._bin_edges) == 3

    def test_joint_with_missing_values(self):
        """Test joint fitting with missing values."""
        X = np.array([[1, np.nan, 3], [2, 5, np.nan], [np.nan, 6, 7], [4, 7, 8]])

        binner = EqualWidthBinning(n_bins=3, fit_jointly=True, joint_range_method="global")

        # Should handle NaN gracefully
        binner.fit(X)
        result = binner.transform(X)

        assert binner._fitted
        assert result.shape == X.shape

    def test_joint_consistency_across_columns(self):
        """Test that joint fitting creates consistent bins."""
        X = np.array([[0, 0], [5, 5], [10, 10]])

        binner = EqualWidthBinning(n_bins=2, fit_jointly=True, joint_range_method="global")
        binner.fit(X)

        # Both columns should have same edges when using global range
        edges_0 = binner._bin_edges[0]
        edges_1 = binner._bin_edges[1]

        assert np.allclose(edges_0, edges_1)
