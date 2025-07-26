"""
Comprehensive test suite for GeneralBinningBase covering every line of code.
"""

import pytest
import numpy as np
import pickle
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict, List, Optional, Union

# Optional imports for testing different data formats
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.datasets import make_classification
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.base import clone

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from binning.base._general_binning_base import GeneralBinningBase
from binning.base._constants import MISSING_VALUE


# ============================================================================
# FACTORED OUT TEST CLASSES - Moved to module level for reusability
# ============================================================================


class ConcreteBinning(GeneralBinningBase):
    """Concrete implementation for comprehensive testing."""

    def __init__(
        self,
        n_bins: int = 3,
        preserve_dataframe: bool = False,
        fit_jointly: bool = False,
        guidance_columns: Optional[Union[List[Any], Any]] = None,
        **kwargs,
    ):
        super().__init__(
            preserve_dataframe=preserve_dataframe,
            fit_jointly=fit_jointly,
            guidance_columns=guidance_columns,
            **kwargs,
        )
        self.n_bins = n_bins
        self._bin_edges = {}
        self._bin_reps = {}
        self._categorical_mappings = {}  # Store categorical value mappings

    def _fit_per_column(
        self,
        X: np.ndarray,
        columns: List[Any],
        guidance_data: Optional[np.ndarray] = None,
        **fit_params,
    ) -> None:
        """Simple equal-width binning for testing."""
        for i, col in enumerate(columns):
            x_col = X[:, i]

            # Handle guidance data if provided
            guidance_info = ""
            if guidance_data is not None and guidance_data.shape[1] > 0:
                guidance_info = f"_guided_by_{guidance_data.shape[1]}_cols"

            try:
                x_numeric = x_col.astype(float)
                finite_mask = np.isfinite(x_numeric)
            except (ValueError, TypeError):
                unique_vals = np.unique(x_col)
                edges = list(range(len(unique_vals) + 1))
                reps = list(range(len(unique_vals)))
                self._bin_edges[f"{col}{guidance_info}"] = edges
                self._bin_reps[f"{col}{guidance_info}"] = reps
                # Store categorical mapping for transform
                self._categorical_mappings[f"{col}{guidance_info}"] = {
                    val: idx for idx, val in enumerate(unique_vals)
                }
                continue

            if not finite_mask.any():
                edges = [0.0, 1.0]
                reps = [0.5]
            else:
                x_finite = x_numeric[finite_mask]
                min_val, max_val = float(x_finite.min()), float(x_finite.max())

                if min_val == max_val:
                    edges = [min_val, max_val]
                    reps = [min_val]
                else:
                    edges = np.linspace(min_val, max_val, self.n_bins + 1).tolist()
                    reps = [(edges[j] + edges[j + 1]) / 2 for j in range(len(edges) - 1)]

            self._bin_edges[f"{col}{guidance_info}"] = edges
            self._bin_reps[f"{col}{guidance_info}"] = reps

    def _fit_jointly(self, X: np.ndarray, columns: List[Any], **fit_params) -> None:
        """Simple joint binning for testing."""
        finite_mask = np.isfinite(X)
        if not finite_mask.any():
            global_min, global_max = 0.0, 1.0
        else:
            global_min = float(X[finite_mask].min())
            global_max = float(X[finite_mask].max())

        if global_min == global_max:
            edges = [global_min, global_max]
            reps = [global_min]
        else:
            edges = np.linspace(global_min, global_max, self.n_bins + 1).tolist()
            reps = [(edges[j] + edges[j + 1]) / 2 for j in range(len(edges) - 1)]

        for col in columns:
            self._bin_edges[col] = edges
            self._bin_reps[col] = reps

    def _transform_columns(self, X: np.ndarray, columns: List[Any]) -> np.ndarray:
        """Transform using digitize."""
        result = np.full(X.shape, MISSING_VALUE, dtype=int)

        for i, col in enumerate(columns):
            x_col = X[:, i]

            # Check for guidance-modified column name
            guidance_key = None
            for key in self._bin_edges.keys():
                if str(key).startswith(str(col)):
                    guidance_key = key
                    break

            edges = self._bin_edges[guidance_key or col]

            for row_idx, value in enumerate(x_col):
                try:
                    numeric_value = float(value)
                    if np.isnan(numeric_value):
                        result[row_idx, i] = MISSING_VALUE
                    else:
                        bin_idx = np.digitize(numeric_value, edges) - 1
                        bin_idx = max(0, min(bin_idx, len(edges) - 2))
                        result[row_idx, i] = bin_idx
                except (ValueError, TypeError):
                    # Use stored categorical mapping from training
                    categorical_mapping = self._categorical_mappings.get(guidance_key or col, {})
                    if value in categorical_mapping:
                        result[row_idx, i] = categorical_mapping[value]
                    else:
                        result[row_idx, i] = MISSING_VALUE

        return result

    def _inverse_transform_columns(self, X: np.ndarray, columns: List[Any]) -> np.ndarray:
        """Inverse transform using representatives."""
        result = np.full(X.shape, np.nan, dtype=float)

        for i, col in enumerate(columns):
            x_col = X[:, i]

            # Check for guidance-modified column name
            guidance_key = None
            for key in self._bin_reps.keys():
                if str(key).startswith(str(col)):
                    guidance_key = key
                    break

            reps = self._bin_reps[guidance_key or col]

            for row_idx, bin_idx in enumerate(x_col):
                if bin_idx == MISSING_VALUE or bin_idx < 0 or bin_idx >= len(reps):
                    result[row_idx, i] = np.nan
                else:
                    result[row_idx, i] = reps[bin_idx]

        return result


class MinimalBinning(GeneralBinningBase):
    """Minimal implementation for testing abstract method enforcement."""

    def _fit_per_column(self, X, columns, guidance_data=None, **fit_params):
        pass

    def _transform_columns(self, X, columns):
        return np.zeros_like(X, dtype=int)

    def _inverse_transform_columns(self, X, columns):
        return X.astype(float)


class IncompleteBinding1(GeneralBinningBase):
    """Test class missing _fit_per_column implementation."""

    def _transform_columns(self, X, columns):
        return np.zeros_like(X, dtype=int)

    def _inverse_transform_columns(self, X, columns):
        return X.astype(float)


class IncompleteBinding2(GeneralBinningBase):
    """Test class missing _transform_columns implementation."""

    def _fit_per_column(self, X, columns, guidance_data=None, **fit_params):
        pass

    def _inverse_transform_columns(self, X, columns):
        return X.astype(float)


class IncompleteBinding3(GeneralBinningBase):
    """Test class missing _inverse_transform_columns implementation."""

    def _fit_per_column(self, X, columns, guidance_data=None, **fit_params):
        pass

    def _transform_columns(self, X, columns):
        return np.zeros_like(X, dtype=int)


# ============================================================================
# TESTS FOR DEFINED CLASSES
# ============================================================================


class TestDefinedClasses:
    """Test all classes defined in this file for complete line coverage."""

    def test_concrete_binning_basic_functionality(self):
        """Test ConcreteBinning basic functionality to cover all lines."""
        # Test initialization with all parameters
        binner = ConcreteBinning(
            n_bins=5, preserve_dataframe=True, fit_jointly=False, guidance_columns=[1]
        )
        assert binner.n_bins == 5
        assert binner._bin_edges == {}
        assert binner._bin_reps == {}

        # Test fitting with numeric data
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        binner.fit(X)

        # Check that bin edges and reps were created
        assert len(binner._bin_edges) > 0
        assert len(binner._bin_reps) > 0

        # Test transform and inverse transform
        result = binner.transform(X)
        assert result.shape == (3, 1)  # Only binning column

        reconstructed = binner.inverse_transform(result)
        assert reconstructed.shape == (3, 1)

    def test_concrete_binning_edge_cases(self):
        """Test ConcreteBinning edge cases to cover all code paths."""
        binner = ConcreteBinning(n_bins=3)

        # Test with non-numeric data (string values)
        X_str = np.array([["a", 1], ["b", 2], ["c", 3]], dtype=object)
        binner.fit(X_str)
        result = binner.transform(X_str)
        assert result.shape == (3, 2)

        # Test with all NaN values
        X_nan = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        binner = ConcreteBinning()
        binner.fit(X_nan)
        result = binner.transform(X_nan)
        assert result.shape == (2, 2)

        # Test with identical values
        X_identical = np.array([[5.0, 5.0], [5.0, 5.0]])
        binner = ConcreteBinning()
        binner.fit(X_identical)
        result = binner.transform(X_identical)
        assert result.shape == (2, 2)

        # Test with guidance data (triggers guidance_info path)
        binner = ConcreteBinning(guidance_columns=[1])
        X_with_guidance = np.array([[1.0, 0.1], [2.0, 0.2]])
        binner.fit(X_with_guidance)

        # Verify guidance info was added to keys
        guidance_keys = [k for k in binner._bin_edges.keys() if "_guided_by_" in str(k)]
        assert len(guidance_keys) > 0

    def test_concrete_binning_joint_fitting(self):
        """Test ConcreteBinning joint fitting functionality."""
        binner = ConcreteBinning(fit_jointly=True, n_bins=4)
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

        binner.fit(X)

        # In joint mode, all columns should have same edges
        edges_list = list(binner._bin_edges.values())
        assert len(set(str(edges) for edges in edges_list)) == 1  # All same

        # Test with all NaN in joint mode
        X_nan = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        binner = ConcreteBinning(fit_jointly=True)
        binner.fit(X_nan)
        result = binner.transform(X_nan)
        assert result.shape == (2, 2)

        # Test with identical values in joint mode
        X_identical = np.array([[5.0, 5.0], [5.0, 5.0]])
        binner = ConcreteBinning(fit_jointly=True)
        binner.fit(X_identical)
        result = binner.transform(X_identical)
        assert result.shape == (2, 2)

    def test_concrete_binning_transform_edge_cases(self):
        """Test ConcreteBinning transform edge cases."""
        binner = ConcreteBinning()
        X = np.array([[1.0, "x"], [2.0, "y"], [3.0, "z"]], dtype=object)
        binner.fit(X)

        # Test transform with mixed data types
        result = binner.transform(X)
        assert result.shape == (3, 2)

        # Test transform with values not in training data (triggers IndexError path)
        X_new = np.array([[4.0, "w"], [5.0, "q"]], dtype=object)
        result = binner.transform(X_new)
        assert result.shape == (2, 2)

        # Check that missing values are handled
        assert np.any(result == MISSING_VALUE)

    def test_concrete_binning_inverse_transform_edge_cases(self):
        """Test ConcreteBinning inverse transform edge cases."""
        binner = ConcreteBinning()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        binner.fit(X)

        # Test with invalid bin indices
        invalid_bins = np.array([[MISSING_VALUE, -1], [999, 0]])
        result = binner.inverse_transform(invalid_bins)
        assert result.shape == (2, 2)
        assert np.isnan(result[0, 0])  # MISSING_VALUE -> NaN
        assert np.isnan(result[0, 1])  # -1 -> NaN
        assert np.isnan(result[1, 0])  # 999 -> NaN

    def test_minimal_binning_coverage(self):
        """Test MinimalBinning for complete line coverage."""
        binner = MinimalBinning()

        # Test all methods exist and execute
        X = np.array([[1, 2], [3, 4]])

        # _fit_per_column should do nothing (just pass)
        binner._fit_per_column(X, [0, 1])

        # _transform_columns should return zeros
        result = binner._transform_columns(X, [0, 1])
        expected = np.zeros_like(X, dtype=int)
        np.testing.assert_array_equal(result, expected)

        # _inverse_transform_columns should convert to float
        result = binner._inverse_transform_columns(X, [0, 1])
        expected = X.astype(float)
        np.testing.assert_array_equal(result, expected)

    def test_incomplete_binding_classes(self):
        """Test all IncompleteBinding classes for line coverage."""
        # Test IncompleteBinding1 (missing _fit_per_column)
        binner1 = IncompleteBinding1()

        # Should work fine
        result = binner1._transform_columns(np.array([[1, 2]]), [0, 1])
        assert result.shape == (1, 2)

        result = binner1._inverse_transform_columns(np.array([[1, 2]]), [0, 1])
        assert result.dtype == float

        # Test IncompleteBinding2 (missing _transform_columns)
        binner2 = IncompleteBinding2()

        # _fit_per_column should work (does nothing)
        binner2._fit_per_column(np.array([[1, 2]]), [0, 1])

        # _inverse_transform_columns should work
        result = binner2._inverse_transform_columns(np.array([[1, 2]]), [0, 1])
        assert result.dtype == float

        # Test IncompleteBinding3 (missing _inverse_transform_columns)
        binner3 = IncompleteBinding3()

        # _fit_per_column should work (does nothing)
        binner3._fit_per_column(np.array([[1, 2]]), [0, 1])

        # _transform_columns should work
        result = binner3._transform_columns(np.array([[1, 2]]), [0, 1])
        assert result.shape == (1, 2)
        assert result.dtype == int

    def test_concrete_binning_guidance_key_lookup(self):
        """Test ConcreteBinning guidance key lookup in transform/inverse_transform."""
        binner = ConcreteBinning(guidance_columns=[1])
        X = np.array([[1.0, 0.1], [2.0, 0.2]])
        binner.fit(X)

        # This should trigger the guidance key lookup logic
        result = binner.transform(X)
        assert result.shape == (2, 1)

        # Test inverse transform with guidance key lookup
        reconstructed = binner.inverse_transform(result)
        assert reconstructed.shape == (2, 1)

        # Test case where guidance_key is None (fallback to col)
        # Manually modify to test the fallback path
        original_edges = binner._bin_edges.copy()
        original_reps = binner._bin_reps.copy()

        # Clear guidance keys and add simple keys
        binner._bin_edges = {0: original_edges[list(original_edges.keys())[0]]}
        binner._bin_reps = {0: original_reps[list(original_reps.keys())[0]]}

        # This should use the fallback path (guidance_key or col)
        result = binner.transform(X)
        assert result.shape == (2, 1)


# ============================================================================
# MAIN TEST CLASS (UPDATED TO REMOVE INLINE CLASS DEFINITIONS)
# ============================================================================


class TestGeneralBinningBaseComprehensive:
    """Comprehensive test class covering all GeneralBinningBase functionality."""

    def test_initialization_and_validation(self):
        """Test __init__ with all parameter combinations and validation."""
        # Default initialization
        binner = ConcreteBinning()
        assert binner.preserve_dataframe == False
        assert binner.fit_jointly == False
        assert binner.guidance_columns is None
        assert binner._fitted == False
        assert binner._binning_columns is None
        assert binner._guidance_columns is None
        assert binner._original_columns is None
        assert binner._n_features_in is None

        # Custom parameters
        binner = ConcreteBinning(
            preserve_dataframe=True, fit_jointly=False, guidance_columns=[1, 2]
        )
        assert binner.preserve_dataframe == True
        assert binner.guidance_columns == [1, 2]

        # Single guidance column (not list)
        binner = ConcreteBinning(guidance_columns=1)
        assert binner.guidance_columns == 1

        # Error: incompatible guidance + joint fitting
        with pytest.raises(ValueError, match="guidance_columns and fit_jointly.*incompatible"):
            ConcreteBinning(guidance_columns=[1], fit_jointly=True)

    def test_prepare_input_wrapper(self):
        """Test _prepare_input as wrapper around prepare_input_with_columns."""
        binner = ConcreteBinning()
        X = np.array([[1, 2], [3, 4]])

        # Mock the imported function
        with patch("binning.base._general_binning_base.prepare_input_with_columns") as mock_func:
            mock_func.return_value = (X, [0, 1])

            result = binner._prepare_input(X)

            # Verify the wrapper calls the function with correct parameters
            mock_func.assert_called_once_with(X, fitted=False, original_columns=None)
            assert result == (X, [0, 1])

        # Test with fitted state
        binner._fitted = True
        binner._original_columns = ["a", "b"]

        with patch("binning.base._general_binning_base.prepare_input_with_columns") as mock_func:
            mock_func.return_value = (X, ["a", "b"])

            binner._prepare_input(X)
            mock_func.assert_called_once_with(X, fitted=True, original_columns=["a", "b"])

    def test_check_fitted(self):
        """Test _check_fitted method."""
        binner = ConcreteBinning()

        # Not fitted - should raise error
        with pytest.raises(RuntimeError, match="not fitted yet"):
            binner._check_fitted()

        # Fitted - should not raise error
        binner._fitted = True
        binner._check_fitted()  # Should pass without error

    def test_separate_columns_comprehensive(self):
        """Test _separate_columns with all scenarios."""
        # No guidance columns
        binner = ConcreteBinning()
        X = np.array([[1, 2, 3], [4, 5, 6]])

        X_binning, X_guidance, binning_cols, guidance_cols = binner._separate_columns(X)

        assert np.array_equal(X_binning, X)
        assert X_guidance is None
        assert binning_cols == [0, 1, 2]
        assert guidance_cols == []

        # Single guidance column (not list)
        binner = ConcreteBinning(guidance_columns=1)
        X_binning, X_guidance, binning_cols, guidance_cols = binner._separate_columns(X)

        assert X_binning.shape == (2, 2)  # Columns 0, 2
        assert X_guidance.shape == (2, 1)  # Column 1
        assert binning_cols == [0, 2]
        assert guidance_cols == [1]

        # Multiple guidance columns (list)
        binner = ConcreteBinning(guidance_columns=[0, 2])
        X_binning, X_guidance, binning_cols, guidance_cols = binner._separate_columns(X)

        assert X_binning.shape == (2, 1)  # Column 1
        assert X_guidance.shape == (2, 2)  # Columns 0, 2
        assert binning_cols == [1]
        assert guidance_cols == [0, 2]

        # All columns are guidance
        binner = ConcreteBinning(guidance_columns=[0, 1, 2])
        X_binning, X_guidance, binning_cols, guidance_cols = binner._separate_columns(X)

        assert X_binning.shape == (2, 0)  # No binning columns
        assert X_guidance.shape == (2, 3)  # All columns
        assert binning_cols == []
        assert guidance_cols == [0, 1, 2]

        # Non-existent guidance columns
        binner = ConcreteBinning(guidance_columns=[10, 20])
        X_binning, X_guidance, binning_cols, guidance_cols = binner._separate_columns(X)

        assert np.array_equal(X_binning, X)  # All columns are binning
        assert X_guidance is None
        assert binning_cols == [0, 1, 2]
        assert guidance_cols == []

    def test_fit_comprehensive(self):
        """Test fit method with all scenarios."""
        # Basic fit without guidance
        binner = ConcreteBinning()
        X = np.array([[1, 10], [2, 20], [3, 30]])

        result = binner.fit(X)

        assert result is binner  # Returns self
        assert binner._fitted == True
        assert binner._n_features_in == 2
        assert binner._original_columns == [0, 1]
        assert binner._binning_columns == [0, 1]
        assert binner._guidance_columns == []

        # Fit with guidance columns
        binner = ConcreteBinning(guidance_columns=[1])
        binner.fit(X)

        assert binner._binning_columns == [0]
        assert binner._guidance_columns == [1]

        # Fit with joint mode
        binner = ConcreteBinning(fit_jointly=True)
        binner.fit(X)

        # Should call _fit_jointly instead of _fit_per_column
        assert binner._fitted == True

        # Fit with y parameter (ignored)
        binner = ConcreteBinning()
        y = np.array([0, 1, 0])
        binner.fit(X, y)
        assert binner._fitted == True

        # Fit with additional kwargs
        binner = ConcreteBinning()
        binner.fit(X, sample_weight=np.array([1, 2, 3]))
        assert binner._fitted == True

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_fit_pandas_integration(self):
        """Test fit with pandas DataFrame."""
        binner = ConcreteBinning()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})

        binner.fit(df)

        assert binner._fitted == True
        assert binner._original_columns == ["a", "b"]
        assert binner._n_features_in == 2

    def test_transform_comprehensive(self):
        """Test transform method with all scenarios."""
        # Transform without guidance
        binner = ConcreteBinning()
        X = np.array([[1, 10], [2, 20], [3, 30]])

        binner.fit(X)
        result = binner.transform(X)

        assert result.shape == X.shape
        assert result.dtype == int

        # Transform not fitted - should raise error
        binner_unfitted = ConcreteBinning()
        with pytest.raises(RuntimeError, match="not fitted yet"):
            binner_unfitted.transform(X)

        # Transform with guidance columns
        binner = ConcreteBinning(guidance_columns=[1])
        X = np.array([[1, 0.1], [2, 0.2], [3, 0.3]])

        binner.fit(X)
        result = binner.transform(X)

        assert result.shape == (3, 1)  # Only binning columns

        # Transform with all guidance columns
        binner = ConcreteBinning(guidance_columns=[0, 1])
        binner.fit(X)
        result = binner.transform(X)

        assert result.shape == (3, 0)  # No binning columns (3 rows, 0 columns)

        # Transform with empty binning columns but non-empty input
        X_empty_binning = np.array([[1, 2], [3, 4]])
        binner = ConcreteBinning(guidance_columns=[0, 1])
        binner.fit(X_empty_binning)
        result = binner.transform(X_empty_binning)

        assert result.shape == (2, 0)
        assert result.dtype == int

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_transform_preserve_dataframe(self):
        """Test transform with preserve_dataframe option."""
        binner = ConcreteBinning(preserve_dataframe=True)
        df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})

        binner.fit(df)
        result = binner.transform(df)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]

        # With guidance columns
        binner = ConcreteBinning(guidance_columns=["b"], preserve_dataframe=True)
        binner.fit(df)
        result = binner.transform(df)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a"]

    def test_transform_with_guidance_comprehensive(self):
        """Test transform_with_guidance method."""
        # Not fitted - should raise error
        binner = ConcreteBinning(guidance_columns=[1])
        X = np.array([[1, 0.1], [2, 0.2]])

        with pytest.raises(RuntimeError, match="not fitted yet"):
            binner.transform_with_guidance(X)

        # Basic usage with guidance
        binner.fit(X)
        binned, guidance = binner.transform_with_guidance(X)

        assert binned.shape == (2, 1)
        assert guidance.shape == (2, 1)

        # No guidance columns
        binner = ConcreteBinning()
        binner.fit(X)
        binned, guidance = binner.transform_with_guidance(X)

        assert binned.shape == (2, 2)
        assert guidance is None

        # All guidance columns
        binner = ConcreteBinning(guidance_columns=[0, 1])
        binner.fit(X)
        binned, guidance = binner.transform_with_guidance(X)

        assert binned.shape == (2, 0)
        assert guidance.shape == (2, 2)

        # Empty binning columns case
        X_test = np.array([[1, 2, 3], [4, 5, 6]])
        binner = ConcreteBinning(guidance_columns=[0, 1, 2])
        binner.fit(X_test)
        binned, guidance = binner.transform_with_guidance(X_test)

        assert binned.shape == (2, 0)
        assert binned.dtype == int
        assert guidance.shape == (2, 3)

    def test_inverse_transform_comprehensive(self):
        """Test inverse_transform method with all scenarios."""
        # Not fitted - should raise error
        binner = ConcreteBinning()
        X = np.array([[0, 1], [1, 0]])

        with pytest.raises(RuntimeError, match="not fitted yet"):
            binner.inverse_transform(X)

        # Basic inverse transform without guidance
        X_orig = np.array([[1, 10], [2, 20], [3, 30]])
        binner = ConcreteBinning()
        binner.fit(X_orig)

        transformed = binner.transform(X_orig)
        reconstructed = binner.inverse_transform(transformed)

        assert reconstructed.shape == X_orig.shape
        assert reconstructed.dtype == float

        # Inverse transform with guidance columns
        X_with_guidance = np.array([[1, 0.1], [2, 0.2], [3, 0.3]])
        binner = ConcreteBinning(guidance_columns=[1])
        binner.fit(X_with_guidance)

        transformed = binner.transform(X_with_guidance)  # Shape (3, 1)
        reconstructed = binner.inverse_transform(transformed)

        assert reconstructed.shape == (3, 1)

        # Wrong number of columns for inverse transform with guidance
        wrong_shape = np.array([[0, 1, 2], [1, 0, 2]])  # 3 columns instead of 1
        with pytest.raises(ValueError, match="should have 1.*columns"):
            binner.inverse_transform(wrong_shape)

        # Test case where _binning_columns is None (edge case)
        binner._binning_columns = None
        with pytest.raises(ValueError, match="should have 0.*columns"):
            binner.inverse_transform(transformed)

    def test_abstract_methods_enforcement(self):
        """Test that abstract methods raise NotImplementedError."""
        # Test _fit_per_column
        binner = IncompleteBinding1()
        with pytest.raises(NotImplementedError, match="Subclasses must implement _fit_per_column"):
            binner._fit_per_column(np.array([[1, 2]]), [0, 1])

        # Test _fit_jointly default implementation
        minimal = MinimalBinning()
        with pytest.raises(NotImplementedError, match="Joint fitting not implemented"):
            minimal._fit_jointly(np.array([[1, 2]]), [0, 1])

        # Test _transform_columns
        binner = IncompleteBinding2()
        with pytest.raises(
            NotImplementedError, match="Subclasses must implement _transform_columns"
        ):
            binner._transform_columns(np.array([[1, 2]]), [0, 1])

        # Test _inverse_transform_columns
        binner = IncompleteBinding3()
        with pytest.raises(
            NotImplementedError, match="Subclasses must implement _inverse_transform_columns"
        ):
            binner._inverse_transform_columns(np.array([[1, 2]]), [0, 1])

    def test_parameter_interface_comprehensive(self):
        """Test get_params and set_params methods."""
        binner = ConcreteBinning(n_bins=5, preserve_dataframe=True, guidance_columns=[1])

        # Test get_params
        params = binner.get_params()
        assert params["n_bins"] == 5
        assert params["preserve_dataframe"] == True
        assert params["guidance_columns"] == [1]
        assert "fit_jointly" in params

        # Test get_params with deep=False
        params_shallow = binner.get_params(deep=False)
        assert isinstance(params_shallow, dict)

        # Test set_params valid combination
        result = binner.set_params(n_bins=10, preserve_dataframe=False)
        assert result is binner
        assert binner.n_bins == 10
        assert binner.preserve_dataframe == False

        # Test set_params validation - incompatible guidance + joint fitting
        with pytest.raises(ValueError, match="guidance_columns and fit_jointly.*incompatible"):
            binner.set_params(guidance_columns=[1], fit_jointly=True)

        # Test set_params with guidance_columns=None
        binner.set_params(guidance_columns=None)
        assert binner.guidance_columns is None

        # Test set_params validation using current values
        binner = ConcreteBinning(guidance_columns=[1])
        # This should fail because current guidance_columns=[1] conflicts with fit_jointly=True
        with pytest.raises(ValueError, match="guidance_columns and fit_jointly.*incompatible"):
            binner.set_params(fit_jointly=True)

        # Test that we can set compatible parameters
        binner.set_params(guidance_columns=None, fit_jointly=True)
        assert binner.guidance_columns is None
        assert binner.fit_jointly == True

    def test_properties_comprehensive(self):
        """Test all property methods."""
        binner = ConcreteBinning()

        # Test properties before fitting
        assert binner.is_fitted_ == False
        assert binner.n_features_in_ is None
        assert binner.feature_names_in_ is None
        assert binner.binning_columns_ is None
        assert binner.guidance_columns_ is None

        # Test properties after fitting
        X = np.array([[1, 2, 3], [4, 5, 6]])
        binner = ConcreteBinning(guidance_columns=[2])
        binner.fit(X)

        assert binner.is_fitted_ == True
        assert binner.n_features_in_ == 3
        assert binner.feature_names_in_ == [0, 1, 2]
        assert binner.binning_columns_ == [0, 1]
        assert binner.guidance_columns_ == [2]

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_properties_pandas(self):
        """Test properties with pandas input."""
        binner = ConcreteBinning()
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        binner.fit(df)

        assert binner.feature_names_in_ == ["a", "b", "c"]

    def test_repr_comprehensive(self):
        """Test __repr__ method with all scenarios."""
        # Default parameters
        binner = ConcreteBinning()
        repr_str = repr(binner)
        assert repr_str == "ConcreteBinning()"

        # preserve_dataframe=True
        binner = ConcreteBinning(preserve_dataframe=True)
        repr_str = repr(binner)
        assert "preserve_dataframe=True" in repr_str

        # fit_jointly=True
        binner = ConcreteBinning(fit_jointly=True)
        repr_str = repr(binner)
        assert "fit_jointly=True" in repr_str

        # guidance_columns as list
        binner = ConcreteBinning(guidance_columns=[1, 2])
        repr_str = repr(binner)
        assert "guidance_columns=[1, 2]" in repr_str

        # guidance_columns as single value
        binner = ConcreteBinning(guidance_columns=1)
        repr_str = repr(binner)
        assert "guidance_columns=[1]" in repr_str

        # All parameters
        binner = ConcreteBinning(preserve_dataframe=True, guidance_columns=["weight"])
        repr_str = repr(binner)
        assert "preserve_dataframe=True" in repr_str
        assert "guidance_columns=['weight']" in repr_str

        # Test truncation with very long parameters
        very_long_guidance = [
            f"extremely_long_column_name_with_many_characters_to_force_truncation_{i}"
            for i in range(50)
        ]
        binner = ConcreteBinning(guidance_columns=very_long_guidance)
        repr_str = repr(binner)

        assert len(repr_str) <= 700
        assert repr_str.endswith("...")

        # Test no truncation needed
        short_guidance = [1, 2, 3]
        binner = ConcreteBinning(guidance_columns=short_guidance)
        repr_str = repr(binner)
        assert not repr_str.endswith("...")

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
    def test_polars_integration(self):
        """Test polars DataFrame integration."""
        binner = ConcreteBinning()
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})

        binner.fit(df)
        result = binner.transform(df)

        # Should convert to numpy by default
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)

        # Test with guidance columns
        binner = ConcreteBinning(guidance_columns=[1])
        binner.fit(df)
        result = binner.transform(df)

        # This might return (3, 2) or (3, 1) depending on polars column handling
        assert result.shape[0] == 3

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
    def test_sklearn_integration_comprehensive(self):
        """Test comprehensive sklearn integration."""
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)

        # Pipeline integration
        pipeline = Pipeline(
            [
                ("binning", ConcreteBinning(n_bins=3)),
                ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )

        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        assert len(predictions) == len(y)

        # Cross-validation
        scores = cross_val_score(pipeline, X, y, cv=3)
        assert len(scores) == 3

        # GridSearchCV
        param_grid = {"binning__n_bins": [2, 3, 4]}
        grid_search = GridSearchCV(pipeline, param_grid, cv=2)
        grid_search.fit(X, y)
        assert hasattr(grid_search, "best_params_")

        # Clone functionality
        binner = ConcreteBinning(n_bins=5, guidance_columns=[1])
        cloned = clone(binner)

        assert cloned.n_bins == 5
        assert cloned.guidance_columns == [1]
        assert not cloned.is_fitted_
        assert cloned is not binner

    def test_serialization_comprehensive(self):
        """Test serialization and deserialization."""
        # Pickle before fitting
        binner = ConcreteBinning(n_bins=5, guidance_columns=[1])
        pickled = pickle.dumps(binner)
        unpickled = pickle.loads(pickled)

        assert unpickled.n_bins == 5
        assert unpickled.guidance_columns == [1]
        assert not unpickled.is_fitted_

        # Pickle after fitting
        X = np.array([[1, 0.1], [2, 0.2], [3, 0.3]])
        binner.fit(X)

        pickled = pickle.dumps(binner)
        unpickled = pickle.loads(pickled)

        assert unpickled.is_fitted_
        assert unpickled._binning_columns == [0]
        assert unpickled._guidance_columns == [1]

        # Test functionality after unpickling
        result = unpickled.transform(X)
        assert result.shape == (3, 1)

        # Constructor parameter reconstruction
        original = ConcreteBinning(n_bins=7, preserve_dataframe=True, guidance_columns=[1, 2])

        params = original.get_params()
        reconstructed = ConcreteBinning(**params)

        assert reconstructed.n_bins == 7
        assert reconstructed.preserve_dataframe == True
        assert reconstructed.guidance_columns == [1, 2]

        # JSON serializable parameters
        params = original.get_params()
        json_str = json.dumps(params)
        loaded_params = json.loads(json_str)

        assert loaded_params["n_bins"] == 7
        assert loaded_params["preserve_dataframe"] == True
        assert loaded_params["guidance_columns"] == [1, 2]

    def test_edge_cases_and_error_handling(self):
        """Test edge cases and comprehensive error handling."""
        # Empty data
        binner = ConcreteBinning()
        X_empty = np.array([]).reshape(0, 2)
        binner.fit(X_empty)
        assert binner.is_fitted_
        assert binner._n_features_in == 2

        # Single row data
        X_single = np.array([[1, 2]])
        binner = ConcreteBinning()
        binner.fit(X_single)
        result = binner.transform(X_single)
        assert result.shape == (1, 2)

        # NaN values
        X_nan = np.array([[1, np.nan], [np.nan, 2], [3, 4]])
        binner = ConcreteBinning()
        binner.fit(X_nan)
        result = binner.transform(X_nan)
        assert result.shape == (3, 2)

        # Identical values
        X_identical = np.array([[5, 1], [5, 2], [5, 3]])
        binner = ConcreteBinning()
        binner.fit(X_identical)
        result = binner.transform(X_identical)
        assert len(np.unique(result[:, 0])) == 1

        # Mixed data types
        X_mixed = np.array([[1, "a"], [2, "b"], [3, "c"]], dtype=object)
        binner = ConcreteBinning()
        binner.fit(X_mixed)
        result = binner.transform(X_mixed)
        assert result.shape == (3, 2)

    def test_data_consistency_and_roundtrip(self):
        """Test data consistency across multiple operations."""
        X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])

        # Multiple fits should give consistent results
        binner1 = ConcreteBinning(n_bins=3)
        binner2 = ConcreteBinning(n_bins=3)

        binner1.fit(X)
        binner2.fit(X)

        result1 = binner1.transform(X)
        result2 = binner2.transform(X)
        np.testing.assert_array_equal(result1, result2)

        # Multiple transforms should be consistent
        result3 = binner1.transform(X)
        np.testing.assert_array_equal(result1, result3)

        # Inverse transform consistency
        transformed = binner1.transform(X)
        reconstructed1 = binner1.inverse_transform(transformed)
        reconstructed2 = binner1.inverse_transform(transformed)
        np.testing.assert_array_equal(reconstructed1, reconstructed2)

        # Parameter changes affect results
        binner_2bins = ConcreteBinning(n_bins=2)
        binner_4bins = ConcreteBinning(n_bins=4)

        binner_2bins.fit(X)
        binner_4bins.fit(X)

        result_2bins = binner_2bins.transform(X)
        result_4bins = binner_4bins.transform(X)

        # Different n_bins should give different results
        assert not np.array_equal(result_2bins, result_4bins)

    def test_guidance_comprehensive_scenarios(self):
        """Test comprehensive guidance column scenarios."""
        X = np.array([[1, 0.1, 10, "A"], [2, 0.2, 20, "B"], [3, 0.3, 30, "C"]])

        # Test guidance affects fitting (per-column mode)
        binner_no_guidance = ConcreteBinning()
        binner_with_guidance = ConcreteBinning(guidance_columns=[1])

        binner_no_guidance.fit(X)
        binner_with_guidance.fit(X)

        # Should have different internal state due to guidance
        # (The concrete implementation stores guidance info in bin edges keys)
        assert binner_no_guidance._bin_edges.keys() != binner_with_guidance._bin_edges.keys()

        # Empty guidance list
        binner = ConcreteBinning(guidance_columns=[])
        binner.fit(X)
        assert binner._guidance_columns == []

        # Non-existent guidance columns
        binner = ConcreteBinning(guidance_columns=[10, 20])
        binner.fit(X)
        assert binner._guidance_columns == []

        # All columns as guidance
        binner = ConcreteBinning(guidance_columns=[0, 1, 2, 3])
        binner.fit(X)
        result = binner.transform(X)
        assert result.shape == (3, 0)

    def test_memory_and_performance_edge_cases(self):
        """Test memory efficiency and performance edge cases."""
        # Large number of columns
        X_large = np.random.rand(50, 30)
        binner = ConcreteBinning()
        binner.fit(X_large)
        result = binner.transform(X_large)
        assert result.shape == (50, 30)

        # Many guidance columns
        guidance_cols = list(range(15, 30))  # Half the columns
        binner = ConcreteBinning(guidance_columns=guidance_cols)
        binner.fit(X_large)
        result = binner.transform(X_large)
        assert result.shape == (50, 15)  # Only binning columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
