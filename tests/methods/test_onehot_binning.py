"""
Streamlined test suite for OneHotBinning with behavior-focused testing.
OneHotBinning supports numeric data only.
"""

import pytest
import numpy as np
import pickle

# Optional imports with skip handling
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.base import clone
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from binning.methods._onehot_binning import OneHotBinning
from binning.base._constants import MISSING_VALUE


class TestOneHotBinningBehavior:
    """Test OneHotBinning core behavior and edge cases."""

    def test_singleton_bin_creation(self):
        """Test basic singleton bin creation and transformation."""
        binner = OneHotBinning()
        X = np.array([[1.0, 10.0], [2.0, 20.0], [1.0, 10.0], [3.0, 30.0]])
        
        result = binner.fit_transform(X)
        
        # Core behavior validation
        assert result.shape == X.shape
        assert result.dtype == int
        assert binner.is_fitted_
        assert result[0, 0] == result[2, 0]  # Same values map to same bins
        assert result[0, 1] == result[2, 1]
        
        # Bin structure validation
        assert len(binner._bin_spec[0]) == 3  # Three unique values: 1.0, 2.0, 3.0
        assert len(binner._bin_spec[1]) == 3  # Three unique values: 10.0, 20.0, 30.0
        
        # All bins should be singletons
        for bins in binner._bin_spec.values():
            for bin_def in bins:
                assert "singleton" in bin_def
                assert isinstance(bin_def["singleton"], float)

    def test_joint_vs_per_column_fitting(self):
        """Test fitting modes and their impact on bin structure."""
        X = np.array([[1.0, 10.0], [2.0, 1.0]])  # Overlapping values across columns
        
        # Per-column fitting
        binner_per_col = OneHotBinning(fit_jointly=False)
        binner_per_col.fit(X)
        
        # Joint fitting
        binner_joint = OneHotBinning(fit_jointly=True)
        binner_joint.fit(X)
        
        # Per-column: different bin counts per column
        assert len(binner_per_col._bin_spec[0]) == 2  # {1.0, 2.0}
        assert len(binner_per_col._bin_spec[1]) == 2  # {1.0, 10.0}
        
        # Joint: same bin count (all unique values across all columns)
        assert len(binner_joint._bin_spec[0]) == 3  # {1.0, 2.0, 10.0}
        assert len(binner_joint._bin_spec[1]) == 3  # {1.0, 2.0, 10.0}

    def test_inverse_transform_and_workflow(self):
        """Test complete fit-transform-inverse workflow."""
        X = np.array([[1.0], [2.0], [3.0], [1.0]])
        binner = OneHotBinning()
        
        result = binner.fit_transform(X)
        inverse = binner.inverse_transform(result)
        
        assert inverse.shape == X.shape
        # Values should be recovered exactly (representatives are same as singletons)
        np.testing.assert_array_equal(inverse, X)

    def test_missing_and_infinite_values(self):
        """Test handling of NaN and infinite values."""
        X = np.array([[1.0], [np.nan], [np.inf], [2.0], [-np.inf]])
        binner = OneHotBinning()
        
        result = binner.fit_transform(X)
        
        # Missing/infinite values should map to MISSING_VALUE
        assert result[1, 0] == MISSING_VALUE  # NaN
        assert result[2, 0] == MISSING_VALUE  # inf
        assert result[4, 0] == MISSING_VALUE  # -inf
        
        # Only finite values should create bins
        assert len(binner._bin_spec[0]) == 2  # 1.0, 2.0 only

    def test_max_unique_values_constraint(self):
        """Test max_unique_values parameter enforcement."""
        X = np.arange(150).reshape(150, 1).astype(float)
        
        # Should fail with default limit
        binner = OneHotBinning(max_unique_values=100)
        with pytest.raises(ValueError, match="exceeds max_unique_values"):
            binner.fit(X)
        
        # Should work with higher limit
        binner_high = OneHotBinning(max_unique_values=200)
        binner_high.fit(X)
        assert len(binner_high._bin_spec[0]) == 150
        
        # Test joint fitting max_unique_values constraint (line 141)
        # Create data where joint unique values exceed limit
        X_joint = np.array([[i, i+50] for i in range(60)])  # 110 unique values total
        binner_joint = OneHotBinning(fit_jointly=True, max_unique_values=100)
        with pytest.raises(ValueError, match="Joint fitting found.*exceeds max_unique_values"):
            binner_joint.fit(X_joint)

    def test_edge_cases_and_error_conditions(self):
        """Test various edge cases and error conditions."""
        binner = OneHotBinning()
        
        # Not fitted errors
        with pytest.raises(RuntimeError, match="not fitted"):
            binner.transform(np.array([[1.0]]))
        
        # Empty data
        X_empty = np.array([]).reshape(0, 1)
        binner.fit(X_empty)
        assert binner.is_fitted_
        
        # All NaN data - should create default bin
        X_all_nan = np.array([[np.nan], [np.nan]])
        binner_nan = OneHotBinning()
        result = binner_nan.fit_transform(X_all_nan)
        assert len(binner_nan._bin_spec[0]) == 1
        assert binner_nan._bin_spec[0][0]["singleton"] == 0.0
        assert np.all(result[:, 0] == MISSING_VALUE)
        
        # Joint fitting with all NaN/inf data - should trigger line 135
        X_all_invalid = np.array([[np.nan, np.inf], [np.inf, np.nan]])
        binner_joint_invalid = OneHotBinning(fit_jointly=True)
        binner_joint_invalid.fit(X_all_invalid)
        # Should create default global unique values
        assert len(binner_joint_invalid._bin_spec[0]) == 1
        assert binner_joint_invalid._bin_spec[0][0]["singleton"] == 0.0

    def test_user_provided_specifications(self):
        """Test with user-provided bin specs and representatives."""
        bin_spec = {0: [{"singleton": 1.0}, {"singleton": 2.0}]}
        bin_reps = {0: [1.5, 2.5]}
        
        binner = OneHotBinning(bin_spec=bin_spec, bin_representatives=bin_reps)
        X = np.array([[1.0], [2.0]])
        
        binner.fit(X)
        result = binner.transform(X)
        inverse = binner.inverse_transform(result)
        
        # Should use provided specifications
        assert binner._bin_spec == bin_spec
        assert binner._bin_reps == bin_reps
        assert inverse[0, 0] == 1.5  # Uses provided representatives
        assert inverse[1, 0] == 2.5

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_integration(self):
        """Test pandas DataFrame support."""
        df = pd.DataFrame({'A': [1.0, 2.0, 1.0], 'B': [10.0, 20.0, 10.0]})
        
        # With dataframe preservation
        binner = OneHotBinning(preserve_dataframe=True)
        result = binner.fit_transform(df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['A', 'B']
        
        # Without dataframe preservation
        binner_array = OneHotBinning(preserve_dataframe=False)
        result_array = binner_array.fit_transform(df)
        assert isinstance(result_array, np.ndarray)

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
    def test_sklearn_integration(self):
        """Test scikit-learn pipeline compatibility."""
        X = np.random.randint(0, 5, size=(100, 3)).astype(float)
        y = np.random.randint(0, 2, size=100)
        
        # Pipeline integration
        pipeline = Pipeline([
            ('binning', OneHotBinning()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        assert len(predictions) == len(y)
        
        # Clone compatibility
        original = OneHotBinning(max_unique_values=50)
        cloned = clone(original)
        assert cloned.max_unique_values == 50
        assert not cloned.is_fitted_

    def test_serialization_and_parameters(self):
        """Test pickle serialization and parameter management."""
        binner = OneHotBinning(max_unique_values=50, fit_jointly=True)
        X = np.array([[1.0], [2.0], [3.0]])
        binner.fit(X)
        
        # Pickle serialization
        serialized = pickle.dumps(binner)
        deserialized = pickle.loads(serialized)
        result = deserialized.transform(X)
        assert result.shape == X.shape
        assert deserialized.is_fitted_
        
        # Parameter management
        params = binner.get_params()
        assert params["max_unique_values"] == 50
        assert params["fit_jointly"] is True
        
        # set_params returns self
        result = binner.set_params(max_unique_values=75)
        assert result is binner
        assert binner.max_unique_values == 75

    def test_repr_and_properties(self):
        """Test string representation and property access."""
        # Default parameters
        binner_default = OneHotBinning()
        assert repr(binner_default) == "OneHotBinning()"
        
        # Custom parameters to cover all repr branches (lines 180, 182, 184)
        bin_spec = {0: [{"singleton": 1.0}]}
        bin_reps = {0: [1.0]}
        binner_custom = OneHotBinning(
            max_unique_values=50, 
            preserve_dataframe=True,  # Line 180
            fit_jointly=True,         # Line 182
            bin_spec=bin_spec,        # Line 184
            bin_representatives=bin_reps
        )
        repr_str = repr(binner_custom)
        assert "max_unique_values=50" in repr_str
        assert "preserve_dataframe=True" in repr_str  # Line 180
        assert "fit_jointly=True" in repr_str         # Line 182
        assert "bin_spec=..." in repr_str            # Line 184
        assert "bin_representatives=..." in repr_str
        
        # Properties before and after fitting
        assert not binner_default.is_fitted_
        assert binner_default.n_features_in_ is None
        
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        binner_default.fit(X)
        assert binner_default.is_fitted_
        assert binner_default.n_features_in_ == 2
        assert binner_default.feature_names_in_ == [0, 1]

    def test_specific_line_coverage(self):
        """Test to ensure specific lines are covered."""
        # Line 135: No finite values in joint fitting
        X_all_invalid = np.array([[np.nan, np.inf], [np.inf, np.nan]])
        binner_invalid = OneHotBinning(fit_jointly=True)
        binner_invalid.fit(X_all_invalid)
        # Should use default value 0.0 when no finite values exist
        assert binner_invalid._bin_spec[0][0]["singleton"] == 0.0
        
        # Line 141: Joint fitting max_unique_values exceeded
        X_too_many = np.array([[i, i+50] for i in range(60)])  # 110 unique values
        binner_too_many = OneHotBinning(fit_jointly=True, max_unique_values=100)
        with pytest.raises(ValueError, match="Joint fitting found.*exceeds max_unique_values"):
            binner_too_many.fit(X_too_many)
        
        # Lines 180, 182, 184: All repr parameter branches
        binner_all_params = OneHotBinning(
            preserve_dataframe=True,  # Triggers line 180
            fit_jointly=True,         # Triggers line 182  
            bin_spec={0: [{"singleton": 1.0}]}  # Triggers line 184
        )
        repr_result = repr(binner_all_params)
        assert "preserve_dataframe=True" in repr_result
        assert "fit_jointly=True" in repr_result
        assert "bin_spec=..." in repr_result
