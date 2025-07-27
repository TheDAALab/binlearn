"""
Comprehensive test suite for EqualWidthBinning with behavior-focused testing.
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

from binning.methods._equal_width_binning import EqualWidthBinning
from binning.base._constants import MISSING_VALUE, ABOVE_RANGE, BELOW_RANGE


# ============================================================================
# CORE FUNCTIONALITY TESTS
# ============================================================================

class TestCoreFunctionality:
    """Test core equal-width binning functionality."""
    
    def test_basic_fit_transform(self):
        """Test basic fit and transform functionality."""
        binner = EqualWidthBinning(n_bins=3)
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        
        # Fit and transform
        result = binner.fit_transform(X)
        
        assert result.shape == X.shape
        assert result.dtype == int
        assert binner._fitted
        assert binner.n_features_in_ == 2
        
        # Check that bins are created
        assert 0 in binner._bin_edges
        assert 1 in binner._bin_edges
        assert len(binner._bin_edges[0]) == 4  # 3 bins = 4 edges
        assert len(binner._bin_edges[1]) == 4
    
    def test_inverse_transform(self):
        """Test inverse transformation."""
        binner = EqualWidthBinning(n_bins=3)
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        
        result = binner.fit_transform(X)
        inverse = binner.inverse_transform(result)
        
        assert inverse.shape == X.shape
        assert inverse.dtype == float
        assert np.all(np.isfinite(inverse))
    
    def test_per_column_n_bins(self):
        """Test per-column n_bins specification."""
        binner = EqualWidthBinning(n_bins={0: 2, 1: 4})
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        
        binner.fit(X)
        
        # Check different number of bins per column
        assert len(binner._bin_edges[0]) == 3  # 2 bins = 3 edges
        assert len(binner._bin_edges[1]) == 5  # 4 bins = 5 edges
    
    def test_specified_bin_range(self):
        """Test with specified bin ranges."""
        binner = EqualWidthBinning(n_bins=2, bin_range=(0.0, 10.0))
        X = np.array([[1.0], [5.0], [8.0]])
        
        binner.fit(X)
        
        # Should use specified range, not data range
        edges = binner._bin_edges[0]
        assert edges[0] == 0.0
        assert edges[-1] == 10.0
    
    def test_per_column_bin_range(self):
        """Test per-column bin ranges."""
        binner = EqualWidthBinning(
            n_bins=2, 
            bin_range={0: (0.0, 10.0), 1: (-5.0, 5.0)}
        )
        X = np.array([[1.0, 2.0], [5.0, 3.0]])
        
        binner.fit(X)
        
        # Check ranges are applied correctly
        assert binner._bin_edges[0][0] == 0.0
        assert binner._bin_edges[0][-1] == 10.0
        assert binner._bin_edges[1][0] == -5.0
        assert binner._bin_edges[1][-1] == 5.0
    
    @pytest.mark.parametrize("fit_jointly", [True, False])
    def test_fit_jointly_vs_per_column(self, fit_jointly):
        """Test both fitting modes."""
        binner = EqualWidthBinning(n_bins=3, fit_jointly=fit_jointly)
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        
        result = binner.fit_transform(X)
        
        assert result.shape == X.shape
        assert binner._fitted
    
    @pytest.mark.parametrize("method", ["global", "percentile", "std", "robust"])
    def test_joint_range_methods(self, method):
        """Test different joint range calculation methods."""
        binner = EqualWidthBinning(
            n_bins=2, 
            fit_jointly=True, 
            joint_range_method=method
        )
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        
        result = binner.fit_transform(X)
        assert result.shape == X.shape
    
    def test_user_provided_bin_edges(self):
        """Test with user-provided bin edges."""
        bin_edges = {0: [0.0, 5.0, 10.0], 1: [0.0, 25.0, 50.0]}
        binner = EqualWidthBinning(bin_edges=bin_edges)
        X = np.array([[1.0, 10.0], [7.0, 30.0]])
        
        binner.fit(X)
        result = binner.transform(X)
        
        # Should use provided edges
        assert result.shape == (2, 2)
        assert binner._bin_edges == bin_edges


# ============================================================================
# EDGE CASES TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_not_fitted_errors(self):
        """Test errors when methods called before fitting."""
        binner = EqualWidthBinning()
        X = np.array([[1.0]])
        
        with pytest.raises(RuntimeError, match="not fitted"):
            binner.transform(X)
        
        with pytest.raises(RuntimeError, match="not fitted"):
            binner.inverse_transform(X)
    
    def test_empty_data(self):
        """Test handling of empty datasets."""
        binner = EqualWidthBinning()
        X = np.array([]).reshape(0, 1)
        
        # Should raise error for empty data
        with pytest.raises(ValueError, match="min and max must be finite"):
            binner.fit(X)
    
    def test_constant_values(self):
        """Test handling of constant value columns."""
        binner = EqualWidthBinning(n_bins=3)
        X = np.array([[5.0], [5.0], [5.0]])
        
        result = binner.fit_transform(X)
        
        assert result.shape == X.shape
        # All values should be in the same bin since they're constant
        assert np.all(result[:, 0] == result[0, 0])
    
    def test_missing_values(self):
        """Test handling of NaN values."""
        binner = EqualWidthBinning(n_bins=3)
        X = np.array([[1.0], [np.nan], [3.0]])
        
        result = binner.fit_transform(X)
        
        assert result.shape == X.shape
        assert result[1, 0] == MISSING_VALUE
    
    def test_infinite_values(self):
        """Test handling of infinite values."""
        binner = EqualWidthBinning(n_bins=3)
        X = np.array([[1.0], [np.inf], [3.0]])
        
        # Should raise error for infinite values
        with pytest.raises(ValueError, match="min and max must be finite"):
            binner.fit(X)
    
    def test_all_nan_column(self):
        """Test with column of all NaN values."""
        binner = EqualWidthBinning()
        X = np.array([[np.nan, 1.0], [np.nan, 2.0]])
        
        with pytest.raises(ValueError, match="min and max must be finite"):
            binner.fit(X)
    
    def test_clipping_behavior(self):
        """Test clipping vs non-clipping behavior."""
        # Test with clipping enabled (default)
        binner_clip = EqualWidthBinning(n_bins=2, bin_range=(0.0, 10.0), clip=True)
        X_train = np.array([[5.0]])
        X_test = np.array([[-5.0], [15.0]])  # Out of range values
        
        binner_clip.fit(X_train)
        result_clip = binner_clip.transform(X_test)
        
        # Values should be clipped to valid range
        assert np.all(result_clip >= 0)
        assert np.all(result_clip < 2)  # Valid bin indices are 0, 1
        
        # Test with clipping disabled
        binner_no_clip = EqualWidthBinning(n_bins=2, bin_range=(0.0, 10.0), clip=False)
        binner_no_clip.fit(X_train)
        result_no_clip = binner_no_clip.transform(X_test)
        
        # Should have special codes for out-of-range values
        assert result_no_clip[0, 0] == BELOW_RANGE
        assert result_no_clip[1, 0] == ABOVE_RANGE
    
    def test_invalid_parameters(self):
        """Test invalid parameter combinations."""
        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            binner = EqualWidthBinning(n_bins=0)
            X = np.array([[1.0]])
            binner.fit(X)


# ============================================================================
# PARAMETER MANAGEMENT TESTS
# ============================================================================

class TestParameterManagement:
    """Test parameter getting and setting."""
    
    def test_get_params(self):
        """Test parameter retrieval."""
        binner = EqualWidthBinning(
            n_bins=5,
            bin_range=(0.0, 10.0),
            clip=False,
            preserve_dataframe=True,
            fit_jointly=True,
            joint_range_method="percentile"
        )
        
        params = binner.get_params()
        
        assert params["n_bins"] == 5
        assert params["bin_range"] == (0.0, 10.0)
        assert params["clip"] is False
        assert params["preserve_dataframe"] is True
        assert params["fit_jointly"] is True
        assert params["joint_range_method"] == "percentile"
    
    def test_set_params_reset_fitted(self):
        """Test that parameter changes reset fitted state."""
        binner = EqualWidthBinning()
        X = np.array([[1.0], [2.0]])
        binner.fit(X)
        assert binner._fitted
        
        # These should reset fitted state
        binner.set_params(n_bins=5)
        assert not binner._fitted
        
        binner.fit(X)
        binner.set_params(bin_range=(0.0, 10.0))
        assert not binner._fitted
        
        binner.fit(X)
        binner.set_params(joint_range_method="percentile")
        assert not binner._fitted
        
        binner.fit(X)
        binner.set_params(fit_jointly=True)
        assert not binner._fitted
    
    def test_set_params_returns_self(self):
        """Test that set_params returns self."""
        binner = EqualWidthBinning()
        result = binner.set_params(n_bins=5)
        assert result is binner


# ============================================================================
# PROPERTIES TESTS
# ============================================================================

class TestProperties:
    """Test property methods."""
    
    def test_properties_before_fitting(self):
        """Test properties before fitting."""
        binner = EqualWidthBinning()
        
        assert binner.is_fitted_ is False
        assert binner.n_features_in_ is None
        assert binner.feature_names_in_ is None
    
    def test_properties_after_fitting(self):
        """Test properties after fitting."""
        binner = EqualWidthBinning()
        X = np.array([[1, 2, 3], [4, 5, 6]])
        binner.fit(X)
        
        assert binner.is_fitted_ is True
        assert binner.n_features_in_ == 3
        assert binner.feature_names_in_ == [0, 1, 2]


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
        
        binner = EqualWidthBinning(n_bins=2, preserve_dataframe=True)
        result = binner.fit_transform(df)
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['A', 'B']
        assert result.shape == df.shape
    
    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_preserve_false(self):
        """Test pandas with preserve_dataframe=False."""
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0]})
        
        binner = EqualWidthBinning(preserve_dataframe=False)
        result = binner.fit_transform(df)
        
        assert isinstance(result, np.ndarray)
    
    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_feature_names(self):
        """Test feature names with pandas."""
        df = pd.DataFrame({'x': [1, 2], 'y': [3, 4], 'z': [5, 6]})
        binner = EqualWidthBinning()
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
        
        binner = EqualWidthBinning(n_bins=2, preserve_dataframe=True)
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
            ('binning', EqualWidthBinning(n_bins=3)),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        
        assert len(predictions) == len(y)
    
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
    def test_sklearn_clone(self):
        """Test sklearn clone compatibility."""
        original = EqualWidthBinning(n_bins=5, bin_range=(0.0, 10.0))
        cloned = clone(original)
        
        assert cloned.n_bins == 5
        assert cloned.bin_range == (0.0, 10.0)
        assert cloned is not original
        assert not cloned.is_fitted_
    
    def test_pickle_serialization(self):
        """Test pickle serialization."""
        binner = EqualWidthBinning(n_bins=3)
        X = np.array([[1.0], [2.0], [3.0]])
        binner.fit(X)
        
        # Serialize and deserialize
        serialized = pickle.dumps(binner)
        deserialized = pickle.loads(serialized)
        
        # Test that deserialized works
        result = deserialized.transform(X)
        assert result.shape == X.shape
        assert deserialized.is_fitted_


# ============================================================================
# REPR TESTS
# ============================================================================

class TestReprMethod:
    """Test __repr__ method."""
    
    def test_repr_default_parameters(self):
        """Test repr with default parameters."""
        binner = EqualWidthBinning()
        repr_str = repr(binner)
        
        assert "EqualWidthBinning()" == repr_str
    
    def test_repr_custom_parameters(self):
        """Test repr with custom parameters."""
        binner = EqualWidthBinning(
            n_bins=5,
            bin_range=(0.0, 10.0),
            clip=False,
            preserve_dataframe=True,
            fit_jointly=True
        )
        repr_str = repr(binner)
        
        assert "n_bins=5" in repr_str
        assert "bin_range=(0.0, 10.0)" in repr_str
        assert "clip=False" in repr_str
        assert "preserve_dataframe=True" in repr_str
        assert "fit_jointly=True" in repr_str


# ============================================================================
# COMPREHENSIVE INTEGRATION TESTS
# ============================================================================

class TestComprehensiveIntegration:
    """Test complex scenarios combining multiple features."""
    
    def test_full_workflow_mixed_specifications(self):
        """Test workflow with mixed parameter specifications."""
        X = np.array([
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0]
        ])
        
        # Mix of global and per-column parameters
        binner = EqualWidthBinning(
            n_bins={0: 2, 1: 3},  # Per-column for first two
            bin_range={1: (0.0, 50.0)},  # Range only for second column
            clip=False,
            preserve_dataframe=False
        )
        
        # Full workflow
        result = binner.fit_transform(X)
        inverse = binner.inverse_transform(result)
        
        assert result.shape == X.shape
        assert inverse.shape == X.shape
        
        # Check different bin counts
        assert len(binner._bin_edges[0]) == 3  # 2 bins
        assert len(binner._bin_edges[1]) == 4  # 3 bins
        assert len(binner._bin_edges[2]) == 11  # default 10 bins
    
    def test_joint_fitting_with_different_ranges(self):
        """Test joint fitting with per-column ranges."""
        binner = EqualWidthBinning(
            n_bins=2,
            bin_range={0: (0.0, 10.0), 1: (-5.0, 5.0)},
            fit_jointly=True,
            joint_range_method="global"
        )
        X = np.array([[5.0, 2.0], [8.0, -1.0]])
        
        result = binner.fit_transform(X)
        assert result.shape == X.shape
    
    def test_edge_case_recovery(self):
        """Test recovery from various edge cases."""
        # Test with extreme values and missing data
        X = np.array([[1e-10, np.nan], [1e10, 0.0]])
        
        binner = EqualWidthBinning(n_bins=2, clip=True)
        
        # Should handle gracefully
        binner.fit(X)
        result = binner.transform(X)
        
        assert result.shape == X.shape
        # NaN should become MISSING_VALUE
        assert result[0, 1] == MISSING_VALUE
    
    def test_user_provided_specifications_complete(self):
        """Test with complete user-provided bin specifications."""
        # Provide edges for all columns
        bin_edges = {0: [0.0, 5.0, 10.0], 1: [0.0, 25.0, 50.0]}
        
        binner = EqualWidthBinning(bin_edges=bin_edges)
        X = np.array([[2.0, 20.0], [7.0, 30.0]])
        
        binner.fit(X)
        result = binner.transform(X)
        
        # Should use provided edges for both columns
        assert binner._bin_edges[0] == [0.0, 5.0, 10.0]
        assert binner._bin_edges[1] == [0.0, 25.0, 50.0]
        assert result.shape == X.shape
    
    def test_mixed_specification_modes(self):
        """Test mixing automatic calculation with range specification."""
        # Use automatic binning with specified ranges
        binner = EqualWidthBinning(
            n_bins={0: 2, 1: 3},
            bin_range={0: (0.0, 10.0)}  # Range only for first column
        )
        X = np.array([[2.0, 20.0], [7.0, 30.0]])
        
        binner.fit(X)
        
        # Column 0 should use specified range with 2 bins
        assert len(binner._bin_edges[0]) == 3  # 2 bins = 3 edges
        assert binner._bin_edges[0][0] == 0.0
        assert binner._bin_edges[0][-1] == 10.0
        
        # Column 1 should use data range with 3 bins
        assert len(binner._bin_edges[1]) == 4  # 3 bins = 4 edges
        assert binner._bin_edges[1][0] <= 20.0
        assert binner._bin_edges[1][-1] >= 30.0


# ============================================================================
# ADDITIONAL COVERAGE TESTS
# ============================================================================

class TestAdditionalCoverage:
    """Tests to achieve 100% coverage of uncovered branches."""
    
    def test_joint_fitting_with_per_column_n_bins(self):
        """Test joint fitting when per-column n_bins are specified."""
        # Line 96: isinstance(self.n_bins, dict) and col_id in self.n_bins
        binner = EqualWidthBinning(
            n_bins={0: 2, 1: 4},  # Per-column n_bins
            fit_jointly=True,
            joint_range_method="global"
        )
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        
        binner.fit(X)
        
        # Should use per-column n_bins even in joint fitting
        assert len(binner._bin_edges[0]) == 3  # 2 bins = 3 edges
        assert len(binner._bin_edges[1]) == 5  # 4 bins = 5 edges
    
    def test_joint_fitting_invalid_n_bins(self):
        """Test joint fitting with invalid n_bins."""
        # Line 101: if n_bins < 1 in joint fitting
        binner = EqualWidthBinning(
            n_bins={0: 0},  # Invalid n_bins
            fit_jointly=True
        )
        X = np.array([[1.0], [2.0]])
        
        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            binner.fit(X)
    
    def test_joint_fitting_fallback_to_data_range(self):
        """Test joint fitting falling back to data range."""
        # Line 111: min_val, max_val = self._get_data_range(x_col, col_id)
        # This happens when joint_params doesn't have "global_range"
        binner = EqualWidthBinning(fit_jointly=True)
        # Override _calculate_joint_parameters to return empty dict
        original_method = binner._calculate_joint_parameters
        binner._calculate_joint_parameters = lambda X, columns: {}
        
        X = np.array([[1.0], [2.0], [3.0]])
        
        try:
            binner.fit(X)
            # Should still work by falling back to data range
            assert len(binner._bin_edges[0]) == 11  # default 10 bins = 11 edges
        finally:
            # Restore original method
            binner._calculate_joint_parameters = original_method
    
    def test_dict_bin_range_has_column_check(self):
        """Test _has_range_for_column with dict bin_range."""
        # Line 161: return col_id in self.bin_range
        binner = EqualWidthBinning(bin_range={0: (0.0, 10.0), 2: (5.0, 15.0)})
        
        # Test the method directly
        assert binner._has_range_for_column(0) is True  # Has range
        assert binner._has_range_for_column(1) is False  # No range
        assert binner._has_range_for_column(2) is True  # Has range
    
    def test_single_bin_range_return(self):
        """Test _get_specified_range with single bin_range."""
        # Line 165: return self.bin_range
        binner = EqualWidthBinning(bin_range=(0.0, 10.0))  # Single range for all columns
        
        # Test the method directly
        min_val, max_val = binner._get_specified_range(0)
        assert min_val == 0.0
        assert max_val == 10.0
        
        # Should work for any column
        min_val, max_val = binner._get_specified_range(5)
        assert min_val == 0.0
        assert max_val == 10.0
    
    def test_get_specified_range_error_cases(self):
        """Test error cases in _get_specified_range."""
        # Test dict case with missing column
        binner_dict = EqualWidthBinning(bin_range={0: (0.0, 10.0)})
        with pytest.raises(ValueError, match="No range specified for column 1"):
            binner_dict._get_specified_range(1)
        
        # Test None case
        binner_none = EqualWidthBinning(bin_range=None)
        with pytest.raises(ValueError, match="No range specified for column 0"):
            binner_none._get_specified_range(0)
