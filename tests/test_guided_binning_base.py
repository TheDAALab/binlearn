import numpy as np
import pytest

from binning._guided_binning_base import GuidedBinningBase

# --- Pandas availability flag for clean skipping ---
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

class DummyGuidedBinning(GuidedBinningBase):
    def _calculate_bin_edges(self, x_col, col_idx, guidance=None, sample_weight=None):
        return np.array([0, 1])
    def _calculate_bin_representatives(self, x, bin_edges, col_idx, guidance=None, sample_weight=None):
        return np.array([0.5])

def test_transform_with_ndarray():
    X = np.array([[1, 10, 100], [2, 20, 200]])
    binner = DummyGuidedBinning(guidance_columns=[1])
    binner.fit(X)
    X_trans = binner.transform(X)
    assert X_trans.shape == (2, 2)
    assert isinstance(X_trans, np.ndarray)

@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
def test_transform_with_dataframe():
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
    binner = DummyGuidedBinning(guidance_columns=['b'])
    binner.fit(df)
    X_trans_df = binner.transform(df)
    assert X_trans_df.shape == (2, 2)
    assert list(X_trans_df.columns) == ['a', 'c']

def test_inverse_transform_with_ndarray():
    X = np.array([[1, 10, 100], [2, 20, 200]])
    binner = DummyGuidedBinning(guidance_columns=[1])
    binner.fit(X)
    X_trans = binner.transform(X)
    X_inv = binner.inverse_transform(X_trans)
    assert X_inv.shape == X_trans.shape
    assert isinstance(X_inv, np.ndarray)

@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
def test_inverse_transform_with_dataframe():
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
    binner = DummyGuidedBinning(guidance_columns=['b'])
    binner.fit(df)
    X_trans_df = binner.transform(df)
    X_inv_df = binner.inverse_transform(X_trans_df)
    assert X_inv_df.shape == X_trans_df.shape
    assert list(X_inv_df.columns) == ['a', 'c']

def test_inverse_transform_with_ndarray_full_input():
    # This test ensures the branch X_binning = X[:, [int(col) for col in self._bin_cols]] is covered.
    # Pass the original (not reduced) array to inverse_transform.
    X = np.array([[1, 10, 100], [2, 20, 200]])
    binner = DummyGuidedBinning(guidance_columns=[1])
    binner.fit(X)
    # Instead of passing X_trans, pass the original X
    X_inv = binner.inverse_transform(X)
    # Should return an array with shape (2, 2) (the binned columns)
    assert X_inv.shape == (2, 2)
    assert isinstance(X_inv, np.ndarray)

def test_calculate_bin_edges_not_implemented():
    binner = GuidedBinningBase()
    with pytest.raises(NotImplementedError, match="Subclasses must implement _calculate_bin_edges with guidance support."):
        binner._calculate_bin_edges(np.array([1, 2, 3]), 0)

def test_calculate_bin_representatives_not_implemented():
    binner = GuidedBinningBase()
    with pytest.raises(NotImplementedError, match="Subclasses must implement _calculate_bin_representatives with guidance support."):
        binner._calculate_bin_representatives(np.array([1, 2, 3]), np.array([0, 1]), 0)

import importlib
import sys
import types

def test_pandas_importerror(monkeypatch):
    """Test that PANDAS_AVAILABLE is False and pd is None if pandas is not installed."""
    # Simulate ImportError for pandas
    module_name = "binning._guided_binning_base"
    # Remove from sys.modules to force re-import
    sys.modules.pop(module_name, None)

    # Patch import to raise ImportError for pandas
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("No module named 'pandas'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Reload the module
    import importlib
    guided_binning_base = importlib.import_module(module_name)
    assert guided_binning_base.PANDAS_AVAILABLE is False
    assert guided_binning_base.pd is None

def test_pipeline_integration_with_guided_binning_base():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.array([[1, 10, 100], [2, 20, 200], [0, 0, 0]])
    binner = DummyGuidedBinning(guidance_columns=[1])
    pipe = Pipeline([
        ("binning", binner),
        ("scaler", StandardScaler())
    ])
    Xt = pipe.fit_transform(X)
    # Should have shape (3, 2) and be scaled
    assert Xt.shape == (3, 2)
    # The first step outputs bin indices, the second step scales them
    # Check that the mean is approximately 0 (standardized)
    assert np.allclose(Xt.mean(axis=0), 0, atol=1e-7)

def test_guided_binning_base_handles_nan_values():
    class DummyGuidedBinning(GuidedBinningBase):
        def _calculate_bin_edges(self, x_col, col_idx, guidance=None, sample_weight=None):
            # Two bins: [0, 1]
            return np.array([0, 1])
        def _calculate_bin_representatives(self, x, bin_edges, col_idx, guidance=None, sample_weight=None):
            return np.array([0.5])

    # X: first column is guidance, second and third columns are to be binned
    X = np.array([
        [1.0, 0.5, np.nan],
        [0.5, np.nan, 0.5],
        [0.5, 0.5, 0.5]
    ])
    # guidance is column 0, bin columns 1 and 2
    binner = DummyGuidedBinning(guidance_columns=[0], clip=False)
    binner.fit(X)
    result = binner.transform(X)
    # NaN values in bin columns should be assigned -1
    assert result[0, 1] == -1
    assert result[1, 0] == -1
    # Valid values should be binned correctly
    assert result[0, 0] == 0
    assert result[1, 1] == 0
    assert result[2, 0] == 0
    assert result[2, 1] == 0