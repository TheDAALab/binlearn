"""
Complete test suite for FlexibleBinningBase with 100% line coverage.
Focused on behavior testing rather than implementation details.
"""

import pytest
import numpy as np
import pickle
from typing import Any, Dict, List, Optional, Tuple

# Optional imports
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

try:
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    SKLEARN_AVAILABLE = True
except ImportError:
    clone = None
    Pipeline = None
    LogisticRegression = None
    make_classification = None
    SKLEARN_AVAILABLE = False

from binning.base._flexible_binning_base import FlexibleBinningBase
from binning.base._constants import MISSING_VALUE


# ============================================================================
# CONCRETE IMPLEMENTATION FOR TESTING
# ============================================================================


class SimpleBinner(FlexibleBinningBase):
    """Simple concrete implementation for testing."""
    
    def __repr__(self):
        defaults = dict(
            n_bins=3,
            preserve_dataframe=False,
            bin_spec=None,
            bin_representatives=None,
            fit_jointly=False,
            guidance_columns=None,
        )
        params = {
            'n_bins': self.n_bins,
            'preserve_dataframe': self.preserve_dataframe,
            'bin_spec': self.bin_spec,
            'bin_representatives': self.bin_representatives,
            'fit_jointly': self.fit_jointly,
            'guidance_columns': self.guidance_columns,
        }
        show = []
        for k, v in params.items():
            if v != defaults[k]:
                if k in {'bin_spec', 'bin_representatives'} and v is not None:
                    show.append(f'{k}=...')
                else:
                    show.append(f'{k}={repr(v)}')
        if not show:
            return f'{self.__class__.__name__}()'
        return f'{self.__class__.__name__}(' + ', '.join(show) + ')'
    
    def __init__(self, n_bins: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.n_bins = n_bins
    
    def _calculate_flexible_bins(
        self, 
        x_col: np.ndarray, 
        col_id: Any, 
        guidance_data: Optional[np.ndarray] = None
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Create mix of singleton and interval bins for testing."""
        finite_mask = np.isfinite(x_col)
        if not finite_mask.any():
            return [{"singleton": 0.0}], [0.0]
        
        x_finite = x_col[finite_mask]
        min_val, max_val = float(x_finite.min()), float(x_finite.max())
        
        if min_val == max_val:
            return [{"singleton": min_val}], [min_val]
        
        # Create simple bins for testing
        if self.n_bins <= 2:
            return [
                {"singleton": min_val}, 
                {"interval": [min_val + 0.1, max_val]}
            ], [min_val, (min_val + 0.1 + max_val) / 2]
        
        # For more bins, create a mix
        bin_defs = []
        reps = []
        edges = np.linspace(min_val, max_val, self.n_bins + 1)
        
        for i in range(self.n_bins):
            if i == 0:
                bin_defs.append({"singleton": edges[i]})
                reps.append(edges[i])
            else:
                bin_defs.append({"interval": [edges[i], edges[i + 1]]})
                reps.append((edges[i] + edges[i + 1]) / 2)
        
        return bin_defs, reps


# ============================================================================
# CORE FUNCTIONALITY TESTS
# ============================================================================


class TestCoreFunctionality:
    """Test core binning functionality."""

    def test_basic_fit_transform(self):
        """Test basic fit and transform workflow."""
        binner = SimpleBinner(n_bins=3)
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        
        # Test fit
        binner.fit(X)
        assert binner._fitted
        assert 0 in binner._bin_spec
        assert 1 in binner._bin_spec
        
        # Test transform
        result = binner.transform(X)
        assert result.shape == (3, 2)
        assert result.dtype == int
        
        # Test fit_transform
        result2 = SimpleBinner(n_bins=3).fit_transform(X)
        np.testing.assert_array_equal(result, result2)

    def test_inverse_transform(self):
        """Test inverse transformation."""
        binner = SimpleBinner(n_bins=3)
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        
        binner.fit(X)
        transformed = binner.transform(X)
        reconstructed = binner.inverse_transform(transformed)
        
        assert reconstructed.shape == X.shape
        assert reconstructed.dtype == float
        
        # Test with missing values
        transformed_with_missing = transformed.copy()
        transformed_with_missing[0, 0] = MISSING_VALUE
        reconstructed = binner.inverse_transform(transformed_with_missing)
        assert np.isnan(reconstructed[0, 0])

    def test_lookup_methods(self):
        """Test bin width and range lookup methods."""
        binner = SimpleBinner(n_bins=2)
        X = np.array([[1.0, 10.0], [2.0, 20.0]])
        
        binner.fit(X)
        transformed = binner.transform(X)
        
        # Test bin widths
        widths = binner.lookup_bin_widths(transformed)
        assert widths.shape == transformed.shape
        
        # Test bin ranges
        ranges = binner.lookup_bin_ranges()
        assert isinstance(ranges, dict)
        assert 0 in ranges
        assert 1 in ranges

    def test_user_provided_specifications(self):
        """Test with user-provided bin specifications and representatives."""
        bin_spec = {
            0: [{"singleton": 1.0}, {"interval": [2.0, 3.0]}],
            1: [{"singleton": 10.0}, {"interval": [20.0, 30.0]}]
        }
        bin_reps = {
            0: [1.0, 2.5],
            1: [10.0, 25.0]
        }
        
        binner = SimpleBinner(bin_spec=bin_spec, bin_representatives=bin_reps)
        X = np.array([[1.0, 10.0], [2.5, 25.0]])
        
        binner.fit(X)
        result = binner.transform(X)
        
        # Should use provided specifications
        assert result[0, 0] == 0  # Matches singleton 1.0
        assert result[0, 1] == 0  # Matches singleton 10.0
        assert result[1, 0] == 1  # In interval [2.0, 3.0]
        assert result[1, 1] == 1  # In interval [20.0, 30.0]

    @pytest.mark.parametrize("fit_jointly", [True, False])
    def test_fit_jointly_vs_per_column(self, fit_jointly):
        """Test both fitting modes."""
        binner = SimpleBinner(n_bins=2, fit_jointly=fit_jointly)
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        
        binner.fit(X)
        result = binner.transform(X)
        
        assert result.shape == (3, 2)
        assert binner._fitted

    def test_guidance_columns(self):
        """Test guidance columns functionality."""
        binner = SimpleBinner(n_bins=3, guidance_columns=[1])
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        
        binner.fit(X)
        result = binner.transform(X)
        
        # Should only return binning columns (not guidance)
        assert result.shape[1] == 1  # Only column 0, not guidance column 1
        
        # Test with multiple guidance columns
        binner2 = SimpleBinner(guidance_columns=[0, 1])
        result2 = binner2.fit_transform(X)
        assert result2.shape[1] == 0  # No binning columns left


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_not_fitted_errors(self):
        """Test errors when not fitted."""
        binner = SimpleBinner()
        X = np.array([[1.0, 2.0]])
        
        with pytest.raises(RuntimeError, match="not fitted"):
            binner.transform(X)
        
        with pytest.raises(RuntimeError, match="not fitted"):
            binner.inverse_transform(X)
        
        with pytest.raises(RuntimeError, match="not fitted"):
            binner.lookup_bin_widths(X)
        
        with pytest.raises(RuntimeError, match="not fitted"):
            binner.lookup_bin_ranges()

    def test_empty_data(self):
        """Test with empty data."""
        binner = SimpleBinner()
        X = np.array([]).reshape(0, 2)
        
        binner.fit(X)
        result = binner.transform(X)
        assert result.shape == (0, 2)

    def test_constant_values(self):
        """Test with constant column values."""
        binner = SimpleBinner()
        X = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        
        binner.fit(X)
        result = binner.transform(X)
        assert result.shape == (3, 2)

    def test_missing_values(self):
        """Test with NaN and infinite values."""
        binner = SimpleBinner()
        X = np.array([[1.0, np.nan], [np.inf, 2.0], [3.0, -np.inf]])
        
        binner.fit(X)
        result = binner.transform(X)
        assert result.shape == (3, 2)

    def test_all_nan_column(self):
        """Test with column of all NaN values."""
        binner = SimpleBinner()
        X = np.array([[np.nan, 1.0], [np.nan, 2.0]])
        
        binner.fit(X)
        result = binner.transform(X)
        assert result.shape == (2, 2)

    def test_column_key_matching(self):
        """Test _get_column_key method with various scenarios."""
        binner = SimpleBinner()
        binner._bin_spec = {"a": [], "b": [], "c": []}
        
        # Direct match
        assert binner._get_column_key("a", ["a", "b", "c"], 0) == "a"
        
        # Index fallback
        assert binner._get_column_key("missing", ["a", "b", "c"], 1) == "b"
        
        # Error case
        with pytest.raises(ValueError, match="No bin specification found"):
            binner._get_column_key("missing", ["a"], 5)

    def test_out_of_bounds_indices(self):
        """Test handling of out-of-bounds bin indices."""
        binner = SimpleBinner(n_bins=2)
        X = np.array([[1.0], [2.0]])
        
        binner.fit(X)
        
        # Test inverse transform with out-of-bounds indices
        invalid_indices = np.array([[10], [-5], [MISSING_VALUE]])
        result = binner.inverse_transform(invalid_indices)
        
        # Should clip to valid range or handle missing values
        assert not np.isnan(result[0, 0])  # Clipped to valid range
        assert not np.isnan(result[1, 0])  # Clipped to valid range
        assert np.isnan(result[2, 0])      # Missing value

    def test_abstract_method_not_implemented(self):
        """Test that the abstract _calculate_flexible_bins method raises NotImplementedError."""
        class AbstractBinner(FlexibleBinningBase):
            """Binner that doesn't implement _calculate_flexible_bins."""
            pass
        
        binner = AbstractBinner()
        X = np.array([[1.0], [2.0]])
        
        with pytest.raises(NotImplementedError, match="Must be implemented by subclasses"):
            binner.fit(X)

    def test_lookup_bin_widths_with_missing_values(self):
        """Test lookup_bin_widths when input contains MISSING_VALUE indices."""
        binner = SimpleBinner(n_bins=2)
        X = np.array([[1.0], [2.0]])
        
        binner.fit(X)
        
        # Create bin indices with MISSING_VALUE
        bin_indices = np.array([[0], [MISSING_VALUE], [1]])
        result = binner.lookup_bin_widths(bin_indices)
        
        # Missing value row should remain NaN, others should have width values
        assert not np.isnan(result[0, 0])  # Valid bin index
        assert np.isnan(result[1, 0])      # Missing value - should remain NaN
        assert not np.isnan(result[2, 0])  # Valid bin index


# ============================================================================
# PARAMETER MANAGEMENT TESTS
# ============================================================================


class TestParameterManagement:
    """Test get_params and set_params functionality."""

    def test_get_params(self):
        """Test parameter retrieval."""
        bin_spec = {0: [{"singleton": 1.0}]}
        binner = SimpleBinner(
            n_bins=5,
            bin_spec=bin_spec,
            preserve_dataframe=True,
            fit_jointly=False,  # Changed: guidance_columns and fit_jointly are incompatible
            guidance_columns=[1]
        )
        
        params = binner.get_params()
        assert params["n_bins"] == 5
        assert params["bin_spec"] == bin_spec
        assert params["preserve_dataframe"] == True
        assert params["fit_jointly"] == False
        assert params["guidance_columns"] == [1]

    def test_set_params_reset_fitted(self):
        """Test that certain parameter changes reset fitted state."""
        binner = SimpleBinner()
        X = np.array([[1.0], [2.0]])
        binner.fit(X)
        assert binner._fitted
        
        # These should reset fitted state
        binner.set_params(bin_spec={0: [{"singleton": 5.0}]})
        assert not binner._fitted
        
        # Need to clear bin_reps when changing bin_spec to avoid validation errors
        binner._bin_reps = {}
        binner.fit(X)
        binner.set_params(bin_representatives={0: [5.0]})
        assert not binner._fitted
        
        binner.fit(X)
        binner.set_params(fit_jointly=True)
        assert not binner._fitted
        
        # Reset fit_jointly to False before setting guidance_columns to avoid incompatibility
        binner.set_params(fit_jointly=False)
        binner.fit(X)
        binner.set_params(guidance_columns=[0])
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

    def test_repr(self):
        """Test string representation covers all branches."""
        # Test 1: Default parameters (empty repr)
        binner = SimpleBinner()
        repr_str = repr(binner)
        assert repr_str == "SimpleBinner()"
        
        # Test 2: Only bin_spec (first branch)
        binner = SimpleBinner(bin_spec={0: [{"singleton": 1.0}]})
        repr_str = repr(binner)
        assert "bin_spec=..." in repr_str
        assert "bin_representatives" not in repr_str
        assert "preserve_dataframe" not in repr_str
        assert "fit_jointly" not in repr_str
        assert "guidance_columns" not in repr_str
        
        # Test 3: Only bin_representatives (second branch)
        binner = SimpleBinner(bin_representatives={0: [1.0]})
        repr_str = repr(binner)
        assert "bin_representatives=..." in repr_str
        assert "bin_spec" not in repr_str
        assert "preserve_dataframe" not in repr_str
        assert "fit_jointly" not in repr_str
        assert "guidance_columns" not in repr_str
        
        # Test 4: Only preserve_dataframe=True (third branch)
        binner = SimpleBinner(preserve_dataframe=True)
        repr_str = repr(binner)
        assert "preserve_dataframe=True" in repr_str
        assert "bin_spec" not in repr_str
        assert "bin_representatives" not in repr_str
        assert "fit_jointly" not in repr_str
        assert "guidance_columns" not in repr_str
        
        # Test 5: Only fit_jointly=True (fourth branch)
        binner = SimpleBinner(fit_jointly=True)
        repr_str = repr(binner)
        assert "fit_jointly=True" in repr_str
        assert "bin_spec" not in repr_str
        assert "bin_representatives" not in repr_str
        assert "preserve_dataframe" not in repr_str
        assert "guidance_columns" not in repr_str
        
        # Test 6: Only guidance_columns (fifth branch)
        binner = SimpleBinner(guidance_columns=[1, 2])
        repr_str = repr(binner)
        assert "guidance_columns=[1, 2]" in repr_str
        assert "bin_spec" not in repr_str
        assert "bin_representatives" not in repr_str
        assert "preserve_dataframe" not in repr_str
        assert "fit_jointly" not in repr_str
        
        # Test 7: Multiple compatible parameters (avoid fit_jointly + guidance_columns)
        binner = SimpleBinner(
            bin_spec={0: [{"singleton": 1.0}]},
            bin_representatives={0: [1.0]},
            preserve_dataframe=True,
            guidance_columns=[1, 2]
        )
        repr_str = repr(binner)
        assert "bin_spec=..." in repr_str
        assert "bin_representatives=..." in repr_str
        assert "preserve_dataframe=True" in repr_str
        assert "guidance_columns=[1, 2]" in repr_str
        assert "fit_jointly" not in repr_str
        
        # Test 7b: fit_jointly with other compatible parameters
        binner = SimpleBinner(
            bin_spec={0: [{"singleton": 1.0}]},
            preserve_dataframe=True,
            fit_jointly=True
        )
        repr_str = repr(binner)
        assert "bin_spec=..." in repr_str
        assert "preserve_dataframe=True" in repr_str
        assert "fit_jointly=True" in repr_str
        assert "guidance_columns" not in repr_str
        
        # Test 8: Simple repr format
        result = repr(binner)
        assert "SimpleBinner(" in result
        
        # Test 9: False values don't appear (negative cases)
        binner = SimpleBinner(
            bin_spec=None,
            bin_representatives=None,
            preserve_dataframe=False,
            fit_jointly=False,
            guidance_columns=None
        )
        repr_str = repr(binner)
        assert repr_str == "SimpleBinner()"


# ============================================================================
# DEPRECATED METHODS TESTS
# ============================================================================


class TestDeprecatedMethods:
    """Test deprecated utility methods for backwards compatibility."""

    def test_deprecated_ensure_flexible_bin_dict(self):
        """Test deprecated _ensure_flexible_bin_dict method."""
        binner = SimpleBinner()
        bin_spec = {0: [{"singleton": 1.0}]}
        result = binner._ensure_flexible_bin_dict(bin_spec)
        assert result == bin_spec

    def test_deprecated_generate_default_representatives(self):
        """Test deprecated _generate_default_flexible_representatives method."""
        binner = SimpleBinner()
        bin_defs = [{"singleton": 1.0}, {"interval": [2.0, 3.0]}]
        reps = binner._generate_default_flexible_representatives(bin_defs)
        assert len(reps) == 2
        assert reps[0] == 1.0
        assert reps[1] == 2.5

    def test_deprecated_validate_bins(self):
        """Test deprecated _validate_flexible_bins method."""
        binner = SimpleBinner()
        bin_spec = {0: [{"singleton": 1.0}]}
        bin_reps = {0: [1.0]}
        # Should not raise error
        binner._validate_flexible_bins(bin_spec, bin_reps)

    def test_deprecated_is_missing_value(self):
        """Test deprecated _is_missing_value method."""
        binner = SimpleBinner()
        assert binner._is_missing_value(np.nan)
        # Note: MISSING_VALUE constant may not be considered "missing" by this function
        assert not binner._is_missing_value(1.0)

    def test_deprecated_find_bin_for_value(self):
        """Test deprecated _find_bin_for_value method."""
        binner = SimpleBinner()
        bin_defs = [{"singleton": 1.0}, {"interval": [2.0, 3.0]}]
        assert binner._find_bin_for_value(1.0, bin_defs) == 0
        assert binner._find_bin_for_value(2.5, bin_defs) == 1


# ============================================================================
# DATA FORMAT TESTS
# ============================================================================


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
class TestPandasIntegration:
    """Test pandas DataFrame support."""

    def test_pandas_dataframe(self):
        """Test with pandas DataFrame."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [10.0, 20.0, 30.0]
        })
        
        binner = SimpleBinner(preserve_dataframe=True)
        result = binner.fit_transform(df)
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['A', 'B']
        assert result.shape == (3, 2)

    def test_pandas_with_guidance(self):
        """Test pandas with guidance columns."""
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0], 'B': [10.0, 20.0, 30.0], 'C': [0.1, 0.2, 0.3]})
        
        binner = SimpleBinner(guidance_columns=['C'], preserve_dataframe=True)
        result = binner.fit_transform(df)
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['A', 'B']  # Only binning columns

    def test_pandas_preserve_false(self):
        """Test pandas with preserve_dataframe=False."""
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0]})
        
        binner = SimpleBinner(preserve_dataframe=False)
        result = binner.fit_transform(df)
        
        assert isinstance(result, np.ndarray)


@pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
class TestPolarsIntegration:
    """Test polars DataFrame support."""

    def test_polars_dataframe(self):
        """Test with polars DataFrame."""
        df = pl.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [10.0, 20.0, 30.0]
        })
        
        binner = SimpleBinner(preserve_dataframe=True)
        result = binner.fit_transform(df)
        
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ['A', 'B']


# ============================================================================
# SKLEARN INTEGRATION TESTS
# ============================================================================


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
class TestSklearnIntegration:
    """Test sklearn compatibility."""

    def test_pipeline_integration(self):
        """Test in sklearn Pipeline."""
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        
        pipeline = Pipeline([
            ('binning', SimpleBinner(n_bins=3)),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        assert len(predictions) == 100

    def test_sklearn_clone(self):
        """Test sklearn clone functionality."""
        original = SimpleBinner(n_bins=5, preserve_dataframe=True)
        cloned = clone(original)
        
        assert cloned is not original
        assert cloned.n_bins == original.n_bins
        assert cloned.preserve_dataframe == original.preserve_dataframe

    def test_pickle_serialization(self):
        """Test pickle serialization."""
        X = np.array([[1.0, 10.0], [2.0, 20.0]])
        
        binner = SimpleBinner(n_bins=3)
        binner.fit(X)
        
        # Serialize and deserialize
        serialized = pickle.dumps(binner)
        deserialized = pickle.loads(serialized)
        
        # Should work the same
        original_result = binner.transform(X)
        deserialized_result = deserialized.transform(X)
        
        np.testing.assert_array_equal(original_result, deserialized_result)


# ============================================================================
# COMPREHENSIVE INTEGRATION TESTS
# ============================================================================


class TestComprehensiveIntegration:
    """Test complex scenarios combining multiple features."""

    def test_full_workflow_with_guidance(self):
        """Test complete workflow with guidance columns."""
        X = np.array([
            [1.0, 10.0, 100.0],  # data, guidance, data
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0]
        ])
        
        binner = SimpleBinner(
            n_bins=2,
            guidance_columns=[1],
            preserve_dataframe=False
        )
        
        # Fit and transform
        result = binner.fit_transform(X)
        assert result.shape == (3, 2)  # Only columns 0 and 2
        
        # Inverse transform
        reconstructed = binner.inverse_transform(result)
        assert reconstructed.shape == (3, 2)
        
        # Lookup methods
        widths = binner.lookup_bin_widths(result)
        assert widths.shape == (3, 2)
        
        ranges = binner.lookup_bin_ranges()
        assert len(ranges) == 2

    def test_mixed_specifications(self):
        """Test partial user specifications."""
        # Provide spec for only one column
        bin_spec = {0: [{"singleton": 1.0}, {"interval": [2.0, 4.0]}]}
        
        binner = SimpleBinner(bin_spec=bin_spec, n_bins=3)
        X = np.array([[1.0, 10.0], [2.5, 20.0], [3.0, 30.0]])
        
        binner.fit(X)
        result = binner.transform(X)
        
        # Column 0 should use provided spec, column 1 should be calculated
        assert 0 in binner._bin_spec
        assert 1 in binner._bin_spec
        assert len(binner._bin_spec[0]) == 2  # User provided
        assert len(binner._bin_spec[1]) == 3  # Calculated with n_bins=3

    def test_joint_fitting_edge_case(self):
        """Test joint fitting when some columns already have specs."""
        bin_spec = {1: [{"singleton": 10.0}]}  # Only second column
        
        binner = SimpleBinner(
            bin_spec=bin_spec,
            fit_jointly=True,
            n_bins=2
        )
        X = np.array([[1.0, 10.0], [2.0, 10.0]])
        
        binner.fit(X)
        
        # Should calculate for column 0, use provided for column 1
        assert 0 in binner._bin_spec
        assert 1 in binner._bin_spec
        assert len(binner._bin_spec[1]) == 1  # User provided

    def test_error_recovery(self):
        """Test that the system recovers gracefully from edge cases."""
        # Test with extreme values
        X = np.array([[1e-10, 1e10], [1e-10, 1e10]])
        
        binner = SimpleBinner(n_bins=2)
        binner.fit(X)
        result = binner.transform(X)
        
        assert result.shape == (2, 2)
        assert not np.any(np.isnan(result))
