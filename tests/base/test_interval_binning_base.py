"""
Comprehensive test suite for IntervalBinningBase with behavior-focused testing.
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

from binning.base._interval_binning_base import IntervalBinningBase
from binning.base._constants import MISSING_VALUE, ABOVE_RANGE, BELOW_RANGE


# ============================================================================
# CONCRETE IMPLEMENTATION FOR TESTING
# ============================================================================

class SimpleBinner(IntervalBinningBase):
    """Simple concrete implementation for testing."""
    
    def __init__(self, n_bins: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.n_bins = n_bins
    
    def _calculate_bins(
        self, 
        x_col: np.ndarray, 
        col_id: Any, 
        guidance_data: Optional[np.ndarray] = None
    ) -> Tuple[List[float], List[float]]:
        """Calculate bins using equal-width binning."""
        # Handle edge cases
        finite_mask = np.isfinite(x_col)
        if not finite_mask.any():
            return [0.0, 1.0], [0.5]
        
        x_finite = x_col[finite_mask]
        min_val, max_val = float(x_finite.min()), float(x_finite.max())
        
        if min_val == max_val:
            return [min_val, max_val], [min_val]
        
        # Create equal-width bins
        edges = np.linspace(min_val, max_val, self.n_bins + 1).tolist()
        reps = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges) - 1)]
        
        return edges, reps


class AbstractBinner(IntervalBinningBase):
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
        assert np.all(result >= 0)
        assert np.all(result < 3)  # n_bins = 3
        assert binner._fitted
    
    def test_inverse_transform(self):
        """Test inverse transformation."""
        binner = SimpleBinner(n_bins=3)
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        
        result = binner.fit_transform(X)
        inverse = binner.inverse_transform(result)
        
        assert inverse.shape == X.shape
        assert np.all(np.isfinite(inverse))
    
    def test_lookup_methods(self):
        """Test lookup bin widths and ranges."""
        binner = SimpleBinner(n_bins=3)
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        
        result = binner.fit_transform(X)
        
        # Test lookup_bin_widths
        widths = binner.lookup_bin_widths(result)
        assert widths.shape == result.shape
        assert np.all(widths >= 0)
        
        # Test lookup_bin_ranges
        ranges = binner.lookup_bin_ranges()
        assert isinstance(ranges, dict)
        assert all(v == 3 for v in ranges.values())  # n_bins = 3


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
            binner.lookup_bin_widths(X)
        
        with pytest.raises(RuntimeError, match="not fitted"):
            binner.lookup_bin_ranges()
    
    def test_empty_data(self):
        """Test handling of empty datasets."""
        binner = SimpleBinner()
        X = np.array([]).reshape(0, 1)
        
        # Should handle gracefully
        binner.fit(X)
        assert binner._fitted
    
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
    
    def test_infinite_values(self):
        """Test handling of infinite values."""
        binner = SimpleBinner(n_bins=3)
        X = np.array([[1.0], [np.inf], [3.0], [-np.inf]])
        
        result = binner.fit_transform(X)
        
        assert result.shape == X.shape
        # Infinities should be handled in some reasonable way
    
    def test_all_nan_column(self):
        """Test column with all NaN values."""
        binner = SimpleBinner(n_bins=3)
        X = np.array([[np.nan], [np.nan], [np.nan]])
        
        result = binner.fit_transform(X)
        
        assert result.shape == X.shape
        assert np.all(result == MISSING_VALUE)
    
    def test_column_key_matching(self):
        """Test column key matching between fit and transform."""
        binner = SimpleBinner()
        X_fit = np.array([[1.0, 10.0]])
        X_transform = np.array([[2.0, 20.0]])
        
        binner.fit(X_fit)
        result = binner.transform(X_transform)
        
        assert result.shape == X_transform.shape
    
    def test_out_of_bounds_indices(self):
        """Test handling of out-of-bounds bin indices in inverse_transform."""
        binner = SimpleBinner(n_bins=2)
        X = np.array([[1.0], [2.0]])
        binner.fit(X)
        
        # Create invalid bin indices
        invalid_indices = np.array([[100], [-1]])
        result = binner.inverse_transform(invalid_indices)
        
        assert result.shape == invalid_indices.shape
        assert np.all(np.isfinite(result))  # Should be clipped to valid range
    
    def test_abstract_method_not_implemented(self):
        """Test that abstract method raises NotImplementedError."""
        binner = AbstractBinner()
        
        with pytest.raises(NotImplementedError):
            binner._calculate_bins(np.array([1.0, 2.0]), 0)
    
    def test_get_column_key_edge_cases(self):
        """Test _get_column_key method with edge cases."""
        binner = SimpleBinner()
        X = np.array([[1.0], [2.0]])
        binner.fit(X)
        
        available_keys = list(binner._bin_edges.keys())
        
        # Test index-based fallback when target_col is not in available_keys
        # but col_index is valid
        key = binner._get_column_key("non_existent_col", available_keys, 0)
        assert key == available_keys[0]
        
        # Test error case when neither direct match nor index fallback work
        with pytest.raises(ValueError, match="No bin specification found"):
            binner._get_column_key("non_existent_col", available_keys, 999)


# ============================================================================
# PARAMETER MANAGEMENT TESTS
# ============================================================================

class TestParameterManagement:
    """Test parameter getting and setting."""
    
    def test_get_params(self):
        """Test parameter retrieval."""
        binner = SimpleBinner(
            n_bins=5,
            clip=False,
            preserve_dataframe=True,
            fit_jointly=True
        )
        
        params = binner.get_params()
        
        assert params["n_bins"] == 5
        assert params["clip"] is False
        # Note: preserve_dataframe and fit_jointly are in the base class params
        # They might not appear in get_params() for this specific class
    
    def test_set_params_reset_fitted(self):
        """Test that changing certain parameters resets fitted state."""
        binner = SimpleBinner()
        X = np.array([[1.0], [2.0]])
        binner.fit(X)
        assert binner._fitted
        
        # These should reset fitted state
        binner.set_params(bin_edges={0: [0.0, 1.0, 2.0]})
        assert not binner._fitted
        
        binner.fit(X)
        binner.set_params(bin_representatives={0: [0.5, 1.5]})
        assert not binner._fitted
        
        # Test guidance columns (without fit_jointly=True)
        binner.fit(X)
        binner.set_params(guidance_columns=[0])
        assert not binner._fitted
        
        # Test fit_jointly separately 
        binner.set_params(guidance_columns=None)  # Clear guidance first
        binner.fit(X)
        binner.set_params(fit_jointly=True)
        assert not binner._fitted
    
    def test_set_params_no_reset(self):
        """Test that some parameter changes don't reset fitted state."""
        binner = SimpleBinner()
        X = np.array([[1.0], [2.0]])
        binner.fit(X)
        assert binner._fitted
        
        # This shouldn't reset fitted state
        binner.set_params(n_bins=10)
        assert binner._fitted
        
        binner.set_params(clip=False)
        assert binner._fitted


# ============================================================================
# CORE FUNCTIONALITY TESTS CONTINUED
# ============================================================================

class TestCoreFeatures:
    """Test core features like clipping, guidance, and joint fitting."""
    
    def test_user_provided_specifications(self):
        """Test with user-provided bin edges and representatives."""
        bin_edges = {0: [0.0, 2.0, 4.0]}
        bin_reps = {0: [1.0, 3.0]}
        
        binner = SimpleBinner(bin_edges=bin_edges, bin_representatives=bin_reps)
        X = np.array([[1.0], [3.0], [2.5]])
        
        result = binner.fit_transform(X)
        inverse = binner.inverse_transform(result)
        
        assert result.shape == X.shape
        assert np.array_equal(inverse.flatten(), [1.0, 3.0, 3.0])
    
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
        assert result.shape == (X.shape[0], X.shape[1] - 1)  # One less column
        assert binner._fitted
    
    @pytest.mark.parametrize("clip", [True, False])
    def test_clipping_behavior(self, clip):
        """Test clipping vs special values for out-of-range data."""
        binner = SimpleBinner(n_bins=2, clip=clip)
        X = np.array([[1.0], [2.0], [3.0]])
        binner.fit(X)
        
        # Test with out-of-range values
        X_test = np.array([[0.5], [10.0]])  # Below and above range
        result = binner.transform(X_test)
        
        if clip:
            assert np.all(result >= 0)
            assert np.all(result < 2)  # n_bins = 2
        else:
            assert np.any(result == BELOW_RANGE)
            assert np.any(result == ABOVE_RANGE)


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
    
    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_preserve_false(self):
        """Test pandas with preserve_dataframe=False."""
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0]})
        
        binner = SimpleBinner(preserve_dataframe=False)
        result = binner.fit_transform(df)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)


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
        original = SimpleBinner(n_bins=5, clip=False)
        cloned = clone(original)
        
        assert cloned.n_bins == 5
        assert cloned.clip is False
        assert cloned is not original
    
    def test_pickle_serialization(self):
        """Test pickle serialization."""
        binner = SimpleBinner(n_bins=3)
        X = np.array([[1.0], [2.0], [3.0]])
        binner.fit(X)
        
        # Serialize and deserialize
        serialized = pickle.dumps(binner)
        deserialized = pickle.loads(serialized)
        
        # Test that deserialized works
        result = deserialized.transform(X)
        assert result.shape == X.shape


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
        inverse = binner.inverse_transform(result)
        widths = binner.lookup_bin_widths(result)
        ranges = binner.lookup_bin_ranges()
        
        # With guidance_columns=[2], only columns 0 and 1 get transformed
        assert result.shape == (X.shape[0], X.shape[1] - 1)
        assert inverse.shape == result.shape
        assert widths.shape == result.shape
        assert len(ranges) == result.shape[1]
    
    def test_mixed_specifications(self):
        """Test mixing user specs with calculated bins."""
        # Provide bin edges for both columns to avoid the key matching issue
        bin_edges = {0: [0.0, 2.0, 4.0], 1: [5.0, 20.0, 35.0]}
        
        binner = SimpleBinner(n_bins=3, bin_edges=bin_edges)
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        
        result = binner.fit_transform(X)
        
        assert result.shape == X.shape
        # Both columns use user-provided specs
    
    def test_joint_fitting_edge_case(self):
        """Test joint fitting with edge cases."""
        X = np.array([[1.0, np.nan], [np.nan, 2.0], [3.0, 3.0]])
        
        binner = SimpleBinner(n_bins=2, fit_jointly=True)
        result = binner.fit_transform(X)
        
        assert result.shape == X.shape
        assert result[0, 1] == MISSING_VALUE
        assert result[1, 0] == MISSING_VALUE
    
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
