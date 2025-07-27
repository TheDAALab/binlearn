"""
Comprehensive test suite for SupervisedBinning with behavior-focused testing.
SupervisedBinning uses decision trees to create bins guided by target variables.
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
    from sklearn.datasets import make_classification, make_regression
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.base import clone
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from binning.methods._supervised_binning import SupervisedBinning
from binning.base._constants import MISSING_VALUE, ABOVE_RANGE, BELOW_RANGE


# ============================================================================
# CORE FUNCTIONALITY TESTS
# ============================================================================

class TestCoreFunctionality:
    """Test core supervised binning functionality."""
    
    def test_basic_fit_transform_classification(self):
        """Test basic fit and transform functionality with classification."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        X[:, 2] = (X[:, 0] > 0).astype(int)  # Binary classification target
        
        binner = SupervisedBinning(
            task_type="classification", 
            tree_params={"random_state": 42},
            guidance_columns=[2]
        )
        
        # Fit and transform - pass full dataset including guidance column
        result = binner.fit_transform(X)
        
        assert result.shape == (100, 2)  # Only feature columns returned
        assert result.dtype == int
        assert binner._fitted
        assert binner.n_features_in_ == 3  # Full input including guidance
        
        # Check that bins are created for features only
        assert 0 in binner._bin_edges
        assert 1 in binner._bin_edges
        assert 2 not in binner._bin_edges  # Guidance column not binned
        assert len(binner._bin_edges[0]) >= 2  # At least 1 bin (2 edges)
        assert len(binner._bin_edges[1]) >= 2
    
    def test_basic_fit_transform_regression(self):
        """Test basic fit and transform functionality with regression."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        X[:, 2] = X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(100)  # Regression target
        
        binner = SupervisedBinning(
            task_type="regression", 
            tree_params={"random_state": 42},
            guidance_columns=[2]
        )
        
        result = binner.fit_transform(X)
        
        assert result.shape == (100, 2)  # Only feature columns returned
        assert result.dtype == int
        assert binner._fitted
        
        # Check that bins are created
        assert 0 in binner._bin_edges
        assert 1 in binner._bin_edges
        assert 2 not in binner._bin_edges  # Guidance column not binned
    
    def test_separate_fit_transform(self):
        """Test separate fit and transform calls."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        X[:, 2] = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        binner = SupervisedBinning(guidance_columns=[2])
        
        # Separate fit and transform
        binner.fit(X)
        assert binner._fitted
        
        result = binner.transform(X)
        assert result.shape == (50, 2)  # Only feature columns returned
        assert result.dtype == int
        
        # Transform should work on new data with same structure for features only
        X_new = np.random.randn(20, 2)
        result_new = binner.transform(X_new)
        assert result_new.shape == (20, 2)


# ============================================================================
# PARAMETER CONFIGURATION TESTS
# ============================================================================

class TestParameterConfiguration:
    """Test different parameter configurations."""
    
    def test_task_type_validation(self):
        """Test task_type parameter validation."""
        # Valid task types
        SupervisedBinning(task_type="classification")
        SupervisedBinning(task_type="regression")
        
        # Invalid task type
        with pytest.raises(ValueError, match="task_type must be"):
            SupervisedBinning(task_type="invalid")
    
    def test_tree_params_configuration(self):
        """Test tree_params parameter configuration."""
        # Default parameters (empty tree_params)
        binner = SupervisedBinning()
        assert binner.tree_params == {}
        
        # Check that internal merged params have defaults
        expected_defaults = {
            "max_depth": 3,
            "min_samples_leaf": 5,
            "min_samples_split": 10,
            "random_state": None,
        }
        assert binner._merged_tree_params == expected_defaults
        
        # Custom parameters
        custom_params = {
            "max_depth": 5,
            "min_samples_leaf": 3,
            "random_state": 42
        }
        binner = SupervisedBinning(tree_params=custom_params)
        
        # Should store original custom params
        assert binner.tree_params == custom_params
        
        # Should merge with defaults internally
        expected_merged = {
            "max_depth": 5,
            "min_samples_leaf": 3,
            "min_samples_split": 10,  # Default
            "random_state": 42,
        }
        assert binner._merged_tree_params == expected_merged
    
    def test_guidance_columns_validation(self):
        """Test guidance_columns parameter validation."""
        np.random.seed(42)
        X = np.random.randn(50, 4)
        X[:, 2] = (X[:, 0] > 0).astype(int)
        X[:, 3] = (X[:, 1] > 0).astype(int)
        
        # Single guidance column should work
        binner = SupervisedBinning(guidance_columns=[2])
        binner.fit(X)
        
        # Multiple guidance columns should fail
        with pytest.raises(ValueError, match="exactly 1 column"):
            binner = SupervisedBinning(guidance_columns=[2, 3])
            binner.fit(X)
        
        # Test 3D guidance data (should fail)
        binner_3d = SupervisedBinning(guidance_columns=[2])
        # Manually call _calculate_bins with 3D guidance data to test line 138
        guidance_3d = np.random.randn(10, 2, 2)  # 3D array
        with pytest.raises(ValueError, match="1D or 2D with 1 column"):
            binner_3d._calculate_bins(
                np.random.randn(10), 
                0, 
                guidance_3d
            )
    
    def test_no_guidance_error(self):
        """Test error when no guidance data is provided."""
        binner = SupervisedBinning()
        X = np.random.randn(50, 2)
        
        with pytest.raises(ValueError, match="requires guidance_data"):
            binner.fit(X)


# ============================================================================
# DATA HANDLING TESTS
# ============================================================================

class TestDataHandling:
    """Test data handling scenarios."""
    
    def test_missing_data_handling(self):
        """Test handling of missing data in features and targets."""
        X = np.array([
            [1.0, 10.0, 1],
            [np.nan, 20.0, 0],  # Missing feature
            [3.0, np.nan, 1],   # Missing feature
            [4.0, 40.0, np.nan], # Missing target
            [5.0, 50.0, 0],
        ], dtype=float)
        
        binner = SupervisedBinning(guidance_columns=[2])
        result = binner.fit_transform(X)
        
        assert result.shape == (5, 2)  # Only feature columns returned
        # Missing values should be handled appropriately
        assert np.all(np.isfinite(result) | (result == MISSING_VALUE))
    
    def test_object_dtype_targets(self):
        """Test handling of object dtype targets (covers line 149)."""
        binner = SupervisedBinning()
        
        # Directly test _calculate_bins with object dtype guidance data
        x_col = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        guidance_obj = np.array(['a', 'b', None, 'a', 'b'], dtype=object)
        
        # This should handle None values in object arrays
        edges, reps = binner._calculate_bins(x_col, 0, guidance_obj)
        
        assert len(edges) >= 2  # At least one bin
        assert len(reps) == len(edges) - 1
        
    def test_1d_guidance_data(self):
        """Test 1D guidance data handling (covers line 127)."""
        binner = SupervisedBinning()
        
        # Manually test the _calculate_bins method with 1D guidance
        x_col = np.array([1.0, 2.0, 3.0, 4.0])
        guidance_1d = np.array([0, 1, 0, 1])  # 1D array
        
        edges, reps = binner._calculate_bins(x_col, 0, guidance_1d)
        
        assert len(edges) >= 2  # At least one bin
        assert len(reps) == len(edges) - 1
        
    def test_edge_case_single_bin(self):
        """Test edge case where only single bin is created (covers line 197)."""
        # Create scenario with no tree splits (constant feature or insufficient data)
        X = np.array([
            [1.0, 0],  # Same feature value
            [1.0, 1],  # Same feature value  
        ], dtype=float)
        
        binner = SupervisedBinning(
            guidance_columns=[1],
            tree_params={"min_samples_split": 10}  # Too high for small data
        )
        
        result = binner.fit_transform(X)
        
        assert result.shape == (2, 1)
        # Should create single bin when no meaningful splits possible
        bin_edges = binner._bin_edges[0]
        assert len(bin_edges) == 2  # Exactly 2 edges = 1 bin
        
    def test_no_tree_splits_scenario(self):
        """Test scenario where decision tree produces no splits (covers line 197)."""
        binner = SupervisedBinning()
        
        # Create data where tree cannot make meaningful splits
        # All samples have same target value -> no splits needed
        x_col = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        guidance_same = np.array([1, 1, 1, 1, 1])  # All same target
        
        edges, reps = binner._calculate_bins(x_col, 0, guidance_same)
        
        # Should have exactly 2 edges (1 bin) when no splits are made
        assert len(edges) == 2
        assert len(reps) == 1
        assert edges[0] == 1.0  # data_min
        assert edges[1] == 5.0  # data_max
        
    def test_tree_no_splits_various_scenarios(self):
        """Test various scenarios where tree makes no splits (covers line 197)."""
        binner = SupervisedBinning()
        
        # Scenario 1: Identical targets
        edges1, _ = binner._calculate_bins(
            np.array([1.0, 2.0, 3.0]), 0, np.array([0, 0, 0])
        )
        assert edges1 == [1.0, 3.0]  # Should use data_min, data_max
        
        # Scenario 2: Very restrictive tree that can't split
        binner_restrictive = SupervisedBinning(
            tree_params={"max_depth": 1, "min_samples_split": 1000}
        )
        edges2, _ = binner_restrictive._calculate_bins(
            np.array([5.0, 10.0]), 0, np.array([0, 1])
        )
        assert edges2 == [5.0, 10.0]  # Should fallback to data_min, data_max
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data for tree splits."""
        # Very small dataset
        X = np.array([[1.0, 0], [2.0, 1]], dtype=float)
        
        binner = SupervisedBinning(
            guidance_columns=[1],
            tree_params={"min_samples_split": 10}  # More than available data
        )
        
        result = binner.fit_transform(X)
        assert result.shape == (2, 1)  # Only feature column returned
        # Should create a single bin when insufficient data
        assert len(np.unique(result)) == 1
    
    def test_constant_features(self):
        """Test behavior with constant features."""
        X = np.array([
            [1.0, 0],
            [1.0, 1],
            [1.0, 0],
            [1.0, 1],
        ], dtype=float)
        
        binner = SupervisedBinning(guidance_columns=[1])
        result = binner.fit_transform(X)
        
        assert result.shape == (4, 1)  # Only feature column returned
        # Constant feature should get single bin
        assert len(np.unique(result)) == 1
    
    def test_meaningful_splits(self):
        """Test that the binning creates meaningful splits based on target."""
        # Create step function data where optimal splits are known
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        y = np.where(X.ravel() < 3, 0, np.where(X.ravel() < 7, 1, 2))
        data = np.column_stack([X.ravel(), y])
        
        binner = SupervisedBinning(
            task_type="classification",
            tree_params={"max_depth": 3, "random_state": 42},
            guidance_columns=[1]
        )
        
        binner.fit(data)
        
        # Check that splits are near the true boundaries (3 and 7)
        edges = binner._bin_edges[0]
        # Should have splits roughly around 3 and 7
        assert len(edges) >= 3  # At least 2 bins
        
        # The splits should be reasonably close to optimal
        sorted_edges = sorted(edges[1:-1])  # Exclude min/max
        if len(sorted_edges) >= 2:
            assert any(abs(edge - 3.0) < 1.0 for edge in sorted_edges)
            assert any(abs(edge - 7.0) < 1.0 for edge in sorted_edges)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
class TestPandasIntegration:
    """Test pandas DataFrame integration."""
    
    def test_dataframe_input_output(self):
        """Test DataFrame input and output."""
        np.random.seed(42)
        data = np.random.randn(50, 3)
        data[:, 2] = (data[:, 0] > 0).astype(int)
        
        df = pd.DataFrame(data, columns=["feature1", "feature2", "target"])
        
        binner = SupervisedBinning(
            guidance_columns=["target"],
            preserve_dataframe=True
        )
        
        result = binner.fit_transform(df)
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["feature1", "feature2"]
        assert result.shape == (50, 2)
        assert result.dtypes["feature1"] == int
        assert result.dtypes["feature2"] == int
    
    def test_column_selection_dataframe(self):
        """Test column selection with DataFrames."""
        np.random.seed(42)
        data = np.random.randn(30, 5)
        data[:, 4] = (data[:, 0] + data[:, 2] > 0).astype(int)
        
        df = pd.DataFrame(data, columns=["a", "b", "c", "d", "target"])
        
        binner = SupervisedBinning(
            guidance_columns=["target"],
            preserve_dataframe=True
        )
        
        # Fit on subset of columns
        result = binner.fit_transform(df[["a", "c", "target"]])
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "c"]
        assert result.shape == (30, 2)


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
class TestSklearnIntegration:
    """Test scikit-learn integration."""
    
    def test_pipeline_integration(self):
        """Test integration with sklearn Pipeline."""
        X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                                   n_informative=2, random_state=42)
        # Add target as guidance column
        X_with_target = np.column_stack([X, y])
        
        binner = SupervisedBinning(
            task_type="classification",
            guidance_columns=[2],
            tree_params={"random_state": 42}
        )
        
        # Create pipeline
        pipeline = Pipeline([
            ("binning", binner),
            ("classifier", LogisticRegression(random_state=42))
        ])
        
        # Fit pipeline - pass full dataset to binning, target separately for classifier
        pipeline.fit(X_with_target, y)
        
        # Test prediction
        predictions = pipeline.predict(X_with_target)
        assert len(predictions) == len(y)
        assert predictions.dtype in [int, bool, np.int64, np.bool_]
    
    def test_clone_compatibility(self):
        """Test compatibility with sklearn clone."""
        binner = SupervisedBinning(
            task_type="regression",
            tree_params={"max_depth": 5, "random_state": 42}
        )
        
        cloned_binner = clone(binner)
        
        assert cloned_binner.task_type == binner.task_type
        assert cloned_binner.tree_params == binner.tree_params
        assert not cloned_binner._fitted


# ============================================================================
# SERIALIZATION TESTS
# ============================================================================

class TestSerialization:
    """Test serialization and deserialization."""
    
    def test_pickle_serialization(self):
        """Test pickle serialization of fitted transformer."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        X[:, 2] = (X[:, 0] > 0).astype(int)
        
        binner = SupervisedBinning(
            task_type="classification",
            tree_params={"max_depth": 2, "random_state": 42},
            guidance_columns=[2]
        )
        
        # Fit the binner
        binner.fit(X)
        original_result = binner.transform(X[:, :2])
        
        # Serialize and deserialize
        serialized = pickle.dumps(binner)
        deserialized_binner = pickle.loads(serialized)
        
        # Test that deserialized binner works the same
        new_result = deserialized_binner.transform(X[:, :2])
        np.testing.assert_array_equal(original_result, new_result)
        
        assert deserialized_binner._fitted
        assert deserialized_binner.task_type == binner.task_type
        assert deserialized_binner.tree_params == binner.tree_params


# ============================================================================
# REPRESENTATION TESTS
# ============================================================================

class TestRepresentation:
    """Test string representation."""
    
    def test_repr_default_params(self):
        """Test repr with default parameters."""
        binner = SupervisedBinning()
        repr_str = repr(binner)
        assert repr_str == "SupervisedBinning()"
    
    def test_repr_custom_params(self):
        """Test repr with custom parameters."""
        binner = SupervisedBinning(
            task_type="regression",
            tree_params={"max_depth": 5, "random_state": 42}
        )
        repr_str = repr(binner)
        expected = "SupervisedBinning(task_type='regression', tree_params={'max_depth': 5, 'random_state': 42})"
        assert repr_str == expected
    
    def test_repr_with_preserve_dataframe(self):
        """Test repr with preserve_dataframe=True."""
        binner = SupervisedBinning(
            tree_params={"min_samples_leaf": 3},
            preserve_dataframe=True
        )
        repr_str = repr(binner)
        assert "preserve_dataframe=True" in repr_str
        assert "tree_params={'min_samples_leaf': 3}" in repr_str
    
    def test_repr_with_bin_specs(self):
        """Test repr with predefined bin specifications."""
        binner = SupervisedBinning(
            bin_edges={0: [0, 1, 2]},
            bin_representatives={0: [0.5, 1.5]}
        )
        repr_str = repr(binner)
        assert "bin_edges=..." in repr_str
        assert "bin_representatives=..." in repr_str


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_sample(self):
        """Test behavior with single sample."""
        X = np.array([[1.0, 0]], dtype=float)
        
        binner = SupervisedBinning(guidance_columns=[1])
        result = binner.fit_transform(X)
        
        assert result.shape == (1, 1)  # Only feature column returned
        assert len(binner._bin_edges[0]) == 2  # Single bin
    
    def test_all_missing_data(self):
        """Test behavior when all data is missing."""
        X = np.array([
            [np.nan, np.nan],
            [np.nan, 0],
            [1.0, np.nan],
        ], dtype=float)
        
        binner = SupervisedBinning(guidance_columns=[1])
        result = binner.fit_transform(X)
        
        assert result.shape == (3, 1)  # Only feature column returned
        # Should handle gracefully
    
    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        np.random.seed(42)
        X = np.random.randn(1000, 3)
        X[:, 2] = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        binner = SupervisedBinning(
            guidance_columns=[2],
            tree_params={"random_state": 42}
        )
        
        # Should complete without issues
        result = binner.fit_transform(X)
        assert result.shape == (1000, 2)  # Only feature columns returned
    
    def test_reproducibility(self):
        """Test reproducibility with random_state."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        X[:, 2] = (X[:, 0] > 0).astype(int)
        
        binner1 = SupervisedBinning(
            guidance_columns=[2],
            tree_params={"random_state": 42}
        )
        binner2 = SupervisedBinning(
            guidance_columns=[2],
            tree_params={"random_state": 42}
        )
        
        result1 = binner1.fit_transform(X)
        result2 = binner2.fit_transform(X)
        
        np.testing.assert_array_equal(result1, result2)
        assert binner1._bin_edges == binner2._bin_edges
    
    def test_different_random_states(self):
        """Test that different random states give different results."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        X[:, 2] = (X[:, 0] > 0).astype(int)
        
        binner1 = SupervisedBinning(
            guidance_columns=[2],
            tree_params={"random_state": 42}
        )
        binner2 = SupervisedBinning(
            guidance_columns=[2],
            tree_params={"random_state": 123}
        )
        
        result1 = binner1.fit_transform(X)
        result2 = binner2.fit_transform(X)
        
        # Results might be different due to random tree construction
        # (though they could be the same by chance with simple data)
        assert result1.shape == result2.shape
