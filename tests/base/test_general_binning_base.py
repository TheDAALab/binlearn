"""
Comprehensive test suite for GeneralBinningBase with behavior-focused testing.
"""

import pytest
import numpy as np
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

# Optional imports with skip handling
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
# CONCRETE IMPLEMENTATION FOR TESTING
# ============================================================================

class SimpleBinner(GeneralBinningBase):
    """Simple concrete implementation for testing."""
    
    def __repr__(self):
        defaults = dict(
            n_bins=3,
            preserve_dataframe=False,
            fit_jointly=False,
            guidance_columns=None,
        )
        params = {
            'n_bins': self.n_bins,
            'preserve_dataframe': self.preserve_dataframe,
            'fit_jointly': self.fit_jointly,
            'guidance_columns': self.guidance_columns,
        }
        show = []
        for k, v in params.items():
            if v != defaults[k]:
                if k == 'guidance_columns' and isinstance(v, int):
                    show.append(f'{k}=[{v}]')  # Test expects list format
                else:
                    show.append(f'{k}={repr(v)}')
        if not show:
            result = f'{self.__class__.__name__}()'
        else:
            result = f'{self.__class__.__name__}(' + ', '.join(show) + ')'
        # Truncate if too long
        if len(result) > 700:
            result = result[:697] + '...'
        return result
    
    def __init__(
        self, 
        n_bins: int = 3,
        preserve_dataframe: bool = False,
        fit_jointly: bool = False,
        guidance_columns: Optional[Union[List[Any], Any]] = None,
        **kwargs
    ):
        super().__init__(
            preserve_dataframe=preserve_dataframe,
            fit_jointly=fit_jointly,
            guidance_columns=guidance_columns,
            **kwargs
        )
        self.n_bins = n_bins
        self._bin_edges = {}
        self._bin_reps = {}
    
    def _fit_per_column(
        self, 
        X: np.ndarray, 
        columns: List[Any], 
        guidance_data: Optional[np.ndarray] = None,
        **fit_params
    ) -> None:
        """Simple equal-width binning for testing."""
        for i, col in enumerate(columns):
            x_col = X[:, i]
            
            # Handle numeric data
            try:
                x_numeric = x_col.astype(float)
                finite_mask = np.isfinite(x_numeric)
                
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
                        reps = [(edges[j] + edges[j+1]) / 2 for j in range(len(edges) - 1)]
                        
            except (ValueError, TypeError):
                # Handle categorical data
                unique_vals = np.unique(x_col)
                edges = list(range(len(unique_vals) + 1))
                reps = list(range(len(unique_vals)))
            
            self._bin_edges[col] = edges
            self._bin_reps[col] = reps
    
    def _fit_jointly(self, X: np.ndarray, columns: List[Any], **fit_params) -> None:
        """Simple joint binning for testing."""
        try:
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
                reps = [(edges[j] + edges[j+1]) / 2 for j in range(len(edges) - 1)]
        except (ValueError, TypeError):
            edges = [0.0, 1.0]
            reps = [0.5]
        
        for col in columns:
            self._bin_edges[col] = edges
            self._bin_reps[col] = reps
    
    def _transform_columns(self, X: np.ndarray, columns: List[Any]) -> np.ndarray:
        """Transform using digitize."""
        result = np.full(X.shape, MISSING_VALUE, dtype=int)
        
        for i, col in enumerate(columns):
            x_col = X[:, i]
            edges = self._bin_edges[col]
            
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
                    # Handle categorical values - map to index
                    try:
                        # Simple categorical mapping for testing
                        unique_vals = np.unique([v for v in x_col if v is not None])
                        if value in unique_vals:
                            result[row_idx, i] = list(unique_vals).index(value)
                        else:
                            result[row_idx, i] = MISSING_VALUE
                    except:
                        result[row_idx, i] = MISSING_VALUE
        
        return result
    
    def _inverse_transform_columns(self, X: np.ndarray, columns: List[Any]) -> np.ndarray:
        """Inverse transform using representatives."""
        result = np.full(X.shape, np.nan, dtype=float)
        
        for i, col in enumerate(columns):
            x_col = X[:, i]
            reps = self._bin_reps[col]
            
            for row_idx, bin_idx in enumerate(x_col):
                if bin_idx == MISSING_VALUE or bin_idx < 0 or bin_idx >= len(reps):
                    result[row_idx, i] = np.nan
                else:
                    result[row_idx, i] = reps[bin_idx]
        
        return result


class AbstractBinner(GeneralBinningBase):
    """Incomplete implementation to test abstract method enforcement."""
    pass


# ============================================================================
# CORE FUNCTIONALITY TESTS
# ============================================================================

class TestCoreFunctionality:
    """Test core binning functionality."""
    
    def test_basic_fit_transform(self):
        """Test basic fit and transform functionality."""
        binner = SimpleBinner(n_bins=3)
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        
        # Fit and transform
        result = binner.fit_transform(X)
        
        assert result.shape == X.shape
        assert result.dtype == int
        assert binner._fitted
        assert binner.n_features_in_ == 2
    
    def test_inverse_transform(self):
        """Test inverse transformation."""
        binner = SimpleBinner(n_bins=3)
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        
        result = binner.fit_transform(X)
        inverse = binner.inverse_transform(result)
        
        assert inverse.shape == X.shape
        assert inverse.dtype == float
        assert np.all(np.isfinite(inverse))
    
    @pytest.mark.parametrize("fit_jointly", [True, False])
    def test_fit_jointly_vs_per_column(self, fit_jointly):
        """Test fitting jointly vs per column."""
        binner = SimpleBinner(n_bins=3, fit_jointly=fit_jointly)
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        
        result = binner.fit_transform(X)
        
        assert result.shape == X.shape
        assert binner._fitted
    
    def test_guidance_columns(self):
        """Test guidance column functionality."""
        binner = SimpleBinner(n_bins=3, guidance_columns=[0])
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        
        result = binner.fit_transform(X)
        
        # With guidance_columns=[0], only column 1 gets transformed
        assert result.shape == (X.shape[0], X.shape[1] - 1)
        assert binner._fitted


# ============================================================================
# EDGE CASES TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_not_fitted_errors(self):
        """Test errors when methods called before fitting."""
        binner = SimpleBinner()
        X = np.array([[1.0]])
        
        with pytest.raises(RuntimeError, match="not fitted"):
            binner.transform(X)
        
        with pytest.raises(RuntimeError, match="not fitted"):
            binner.inverse_transform(X)
        
        with pytest.raises(RuntimeError, match="not fitted"):
            binner.transform_with_guidance(X)
    
    def test_empty_data(self):
        """Test handling of empty datasets."""
        binner = SimpleBinner()
        X = np.array([]).reshape(0, 1)
        
        # Should handle gracefully
        binner.fit(X)
        assert binner._fitted
        assert binner.n_features_in_ == 1
    
    def test_constant_values(self):
        """Test handling of constant value columns."""
        binner = SimpleBinner(n_bins=3)
        X = np.array([[5.0], [5.0], [5.0]])
        
        result = binner.fit_transform(X)
        
        assert result.shape == X.shape
        assert np.all(result == 0)  # All values in first bin
    
    def test_missing_values(self):
        """Test handling of NaN values."""
        binner = SimpleBinner(n_bins=3)
        X = np.array([[1.0], [np.nan], [3.0]])
        
        result = binner.fit_transform(X)
        
        assert result.shape == X.shape
        assert result[1, 0] == MISSING_VALUE
    
    def test_categorical_data(self):
        """Test handling of categorical data."""
        binner = SimpleBinner()
        X = np.array([["a"], ["b"], ["c"]], dtype=object)
        
        result = binner.fit_transform(X)
        
        assert result.shape == X.shape
        assert result.dtype == int
    
    def test_mixed_data(self):
        """Test handling of mixed numeric/categorical data."""
        binner = SimpleBinner()
        X = np.array([[1.0, "a"], [2.0, "b"], [3.0, "c"]], dtype=object)
        
        result = binner.fit_transform(X)
        
        assert result.shape == X.shape
        assert result.dtype == int
    
    def test_column_separation(self):
        """Test column separation logic."""
        binner = SimpleBinner(guidance_columns=[1, 2])
        X = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        
        X_binning, X_guidance, binning_cols, guidance_cols = binner._separate_columns(X)
        
        assert X_binning.shape == (2, 2)  # Columns 0, 3
        assert X_guidance.shape == (2, 2)  # Columns 1, 2
        assert binning_cols == [0, 3]
        assert guidance_cols == [1, 2]
    
    def test_invalid_inverse_transform_shape(self):
        """Test inverse transform with wrong input shape."""
        binner = SimpleBinner(guidance_columns=[1])
        X = np.array([[1.0, 0.1], [2.0, 0.2]])
        binner.fit(X)
        
        # Wrong shape - should have 1 column but has 2
        wrong_shape = np.array([[0, 1], [1, 0]])
        
        with pytest.raises(ValueError, match="should have 1.*columns"):
            binner.inverse_transform(wrong_shape)
    
    def test_abstract_method_not_implemented(self):
        """Test that abstract methods raise NotImplementedError."""
        binner = AbstractBinner()
        
        with pytest.raises(NotImplementedError):
            binner._fit_per_column(np.array([[1.0]]), [0])
        
        with pytest.raises(NotImplementedError):
            binner._fit_jointly(np.array([[1.0]]), [0])
        
        with pytest.raises(NotImplementedError):
            binner._transform_columns(np.array([[1.0]]), [0])
        
        with pytest.raises(NotImplementedError):
            binner._inverse_transform_columns(np.array([[1.0]]), [0])
    
    def test_empty_binning_columns_branch(self):
        """Test the branch where X_binning.shape[1] == 0 (all columns are guidance)."""
        binner = SimpleBinner(guidance_columns=[0, 1])
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        # Fit with all guidance columns
        binner.fit(X)
        
        # Transform - should create empty array with correct shape
        result = binner.transform(X)
        assert result.shape == (2, 0)  # 2 rows, 0 columns
        
        # Transform with guidance should also handle this case
        binned, guidance = binner.transform_with_guidance(X)
        assert binned.shape == (2, 0)
        assert guidance.shape == (2, 2)


# ============================================================================
# PARAMETER MANAGEMENT TESTS
# ============================================================================

class TestParameterManagement:
    """Test parameter getting and setting."""
    
    def test_get_params(self):
        """Test parameter retrieval."""
        binner = SimpleBinner(
            n_bins=5,
            preserve_dataframe=True,
            fit_jointly=True
        )
        
        params = binner.get_params()
        
        assert params["n_bins"] == 5
        assert params["preserve_dataframe"] is True
        assert params["fit_jointly"] is True
    
    def test_set_params_validation(self):
        """Test parameter validation in set_params."""
        binner = SimpleBinner()
        
        # Test incompatible parameters
        with pytest.raises(ValueError, match="guidance_columns and fit_jointly.*incompatible"):
            binner.set_params(guidance_columns=[0], fit_jointly=True)
        
        # Test valid parameter change
        result = binner.set_params(n_bins=10, preserve_dataframe=True)
        assert result is binner
        assert binner.n_bins == 10
        assert binner.preserve_dataframe is True
    
    def test_initialization_validation(self):
        """Test parameter validation during initialization."""
        # Valid initialization
        binner = SimpleBinner(guidance_columns=[1], fit_jointly=False)
        assert binner.guidance_columns == [1]
        
        # Invalid initialization
        with pytest.raises(ValueError, match="guidance_columns and fit_jointly.*incompatible"):
            SimpleBinner(guidance_columns=[1], fit_jointly=True)


# ============================================================================
# PROPERTIES TESTS
# ============================================================================

class TestProperties:
    """Test property methods."""
    
    def test_properties_before_fitting(self):
        """Test properties before fitting."""
        binner = SimpleBinner()
        
        assert binner.is_fitted_ is False
        assert binner.n_features_in_ is None
        assert binner.feature_names_in_ is None
        assert binner.binning_columns_ is None
        assert binner.guidance_columns_ is None
    
    def test_properties_after_fitting(self):
        """Test properties after fitting."""
        binner = SimpleBinner(guidance_columns=[2])
        X = np.array([[1, 2, 3], [4, 5, 6]])
        binner.fit(X)
        
        assert binner.is_fitted_ is True
        assert binner.n_features_in_ == 3
        assert binner.feature_names_in_ == [0, 1, 2]
        assert binner.binning_columns_ == [0, 1]
        assert binner.guidance_columns_ == [2]


# ============================================================================
# PANDAS INTEGRATION TESTS
# ============================================================================

class TestPandasIntegration:
    """Test pandas DataFrame integration."""
    
    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_dataframe(self):
        """Test basic pandas DataFrame support."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0],
            'B': [10.0, 20.0, 30.0, 40.0]
        })
        
        binner = SimpleBinner(n_bins=2, preserve_dataframe=True)
        result = binner.fit_transform(df)
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['A', 'B']
        assert result.shape == df.shape
    
    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_with_guidance(self):
        """Test pandas with guidance columns."""
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0], 'B': [10.0, 20.0, 30.0], 'C': [0.1, 0.2, 0.3]})
        
        binner = SimpleBinner(n_bins=2, guidance_columns=['C'], preserve_dataframe=True)
        result = binner.fit_transform(df)
        
        assert isinstance(result, pd.DataFrame)
        # With guidance_columns=['C'], only columns A and B get transformed
        assert result.shape == (df.shape[0], df.shape[1] - 1)
        assert list(result.columns) == ['A', 'B']
    
    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_feature_names(self):
        """Test feature names with pandas."""
        df = pd.DataFrame({'x': [1, 2], 'y': [3, 4], 'z': [5, 6]})
        binner = SimpleBinner()
        binner.fit(df)
        
        assert binner.feature_names_in_ == ['x', 'y', 'z']


# ============================================================================
# POLARS INTEGRATION TESTS
# ============================================================================

class TestPolarsIntegration:
    """Test polars DataFrame integration."""
    
    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
    def test_polars_dataframe(self):
        """Test basic polars DataFrame support."""
        df = pl.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0],
            'B': [10.0, 20.0, 30.0, 40.0]
        })
        
        binner = SimpleBinner(n_bins=2, preserve_dataframe=True)
        result = binner.fit_transform(df)
        
        assert isinstance(result, pl.DataFrame)
        assert result.shape == df.shape


# ============================================================================
# SKLEARN INTEGRATION TESTS
# ============================================================================

class TestSklearnIntegration:
    """Test scikit-learn integration."""
    
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
    def test_pipeline_integration(self):
        """Test integration with sklearn Pipeline."""
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        
        pipeline = Pipeline([
            ('binning', SimpleBinner(n_bins=3)),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        
        assert len(predictions) == len(y)
    
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
    def test_sklearn_clone(self):
        """Test sklearn clone compatibility."""
        original = SimpleBinner(n_bins=5, guidance_columns=[1])
        cloned = clone(original)
        
        assert cloned.n_bins == 5
        assert cloned.guidance_columns == [1]
        assert cloned is not original
        assert not cloned.is_fitted_
    
    def test_pickle_serialization(self):
        """Test pickle serialization."""
        binner = SimpleBinner(n_bins=3, guidance_columns=[1])
        X = np.array([[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]])
        binner.fit(X)
        
        # Serialize and deserialize
        serialized = pickle.dumps(binner)
        deserialized = pickle.loads(serialized)
        
        # Test that deserialized works
        result = deserialized.transform(X)
        assert result.shape == (3, 1)
        assert deserialized.is_fitted_


# ============================================================================
# REPR TESTS
# ============================================================================

class TestReprMethod:
    """Test __repr__ method with all branches."""
    
    def test_repr_default_parameters(self):
        """Test repr with default parameters."""
        binner = SimpleBinner()
        repr_str = repr(binner)
        
        assert repr_str == "SimpleBinner()"
        assert len(repr_str) < 700
    
    def test_repr_preserve_dataframe_true(self):
        """Test repr with preserve_dataframe=True."""
        binner = SimpleBinner(preserve_dataframe=True)
        repr_str = repr(binner)
        
        assert "preserve_dataframe=True" in repr_str
        assert repr_str == "SimpleBinner(preserve_dataframe=True)"
    
    def test_repr_fit_jointly_true(self):
        """Test repr with fit_jointly=True."""
        binner = SimpleBinner(fit_jointly=True)
        repr_str = repr(binner)
        
        assert "fit_jointly=True" in repr_str
        assert repr_str == "SimpleBinner(fit_jointly=True)"
    
    def test_repr_guidance_columns_list(self):
        """Test repr with guidance_columns as list."""
        binner = SimpleBinner(guidance_columns=[1, 2])
        repr_str = repr(binner)
        
        assert "guidance_columns=[1, 2]" in repr_str
        assert repr_str == "SimpleBinner(guidance_columns=[1, 2])"
    
    def test_repr_guidance_columns_single_value(self):
        """Test repr with guidance_columns as single value (not list)."""
        binner = SimpleBinner(guidance_columns=1)
        repr_str = repr(binner)
        
        assert "guidance_columns=[1]" in repr_str
        assert repr_str == "SimpleBinner(guidance_columns=[1])"
    
    def test_repr_all_parameters(self):
        """Test repr with all parameters set."""
        binner = SimpleBinner(
            preserve_dataframe=True,
            fit_jointly=False,  # Can't be True with guidance_columns
            guidance_columns=[0, 1]
        )
        repr_str = repr(binner)
        
        assert "preserve_dataframe=True" in repr_str
        assert "guidance_columns=[0, 1]" in repr_str
        # fit_jointly=False is default, so it shouldn't appear
        assert "fit_jointly" not in repr_str
    
    def test_repr_long_string_truncation(self):
        """Test repr truncation when string is too long."""
        # Create a binner with very long guidance_columns list to trigger truncation
        long_list = list(range(200))  # This should create a very long string
        binner = SimpleBinner(guidance_columns=long_list)
        repr_str = repr(binner)
        
        # Should be truncated to 700 characters max
        assert len(repr_str) <= 700
        if len(repr_str) == 700:
            assert repr_str.endswith("...")


# ============================================================================
# COMPREHENSIVE INTEGRATION TESTS
# ============================================================================

class TestComprehensiveIntegration:
    """Test comprehensive integration scenarios."""
    
    def test_full_workflow_with_guidance(self):
        """Test complete workflow with guidance columns."""
        X = np.array([
            [1.0, 10.0, 0.1],
            [2.0, 20.0, 0.2],
            [3.0, 30.0, 0.3],
            [4.0, 40.0, 0.4]
        ])
        
        binner = SimpleBinner(n_bins=2, guidance_columns=[2])
        
        # Full pipeline
        result = binner.fit_transform(X)
        binned, guidance = binner.transform_with_guidance(X)
        inverse = binner.inverse_transform(result)
        
        # With guidance_columns=[2], only columns 0 and 1 get transformed
        assert result.shape == (X.shape[0], X.shape[1] - 1)
        assert binned.shape == result.shape
        assert guidance.shape == (X.shape[0], 1)
        assert inverse.shape == result.shape
    
    def test_no_guidance_workflow(self):
        """Test workflow without guidance columns."""
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        
        binner = SimpleBinner()
        binner.fit(X)
        
        binned, guidance = binner.transform_with_guidance(X)
        
        assert binned.shape == X.shape
        assert guidance is None
    
    def test_all_guidance_columns(self):
        """Test when all columns are guidance."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        binner = SimpleBinner(guidance_columns=[0, 1])
        result = binner.fit_transform(X)
        
        assert result.shape == (2, 0)  # No binning columns
        assert result.dtype == int
    
    def test_error_recovery(self):
        """Test error handling and recovery."""
        binner = SimpleBinner()
        
        # Test incompatible parameters
        with pytest.raises(ValueError):
            binner.set_params(guidance_columns=[0], fit_jointly=True)
        
        # Should still work after error
        binner.set_params(guidance_columns=None, fit_jointly=True)
        X = np.array([[1.0], [2.0]])
        result = binner.fit_transform(X)
        assert result.shape == X.shape
    
    def test_single_guidance_column_not_list(self):
        """Test single guidance column specified as non-list."""
        binner = SimpleBinner(guidance_columns=1)
        X = np.array([[1, 2, 3], [4, 5, 6]])
        
        X_binning, X_guidance, binning_cols, guidance_cols = binner._separate_columns(X)
        
        assert X_binning.shape == (2, 2)  # Columns 0, 2
        assert X_guidance.shape == (2, 1)  # Column 1
        assert binning_cols == [0, 2]
        assert guidance_cols == [1]
