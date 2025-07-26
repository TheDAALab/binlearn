import numpy as np
import pytest

from binning.methods._onehot_binning import OneHotBinning


def test_onehot_binning_fit_and_transform():
    """Test fitting and transforming with OneHotBinning on numeric data."""
    X = np.array([[1.0], [2.0], [3.0], [2.0], [np.nan]])
    binner = OneHotBinning()
    binner.fit(X)
    # Should create singleton bins for 1.0, 2.0, 3.0 (order may vary)
    bins = binner.bins_[0]
    bin_values = sorted([d["singleton"] for d in bins])
    assert bin_values == [1.0, 2.0, 3.0]
    # Transform: 1.0 -> bin, 2.0 -> bin, 3.0 -> bin, np.nan -> -2
    result = binner.transform(X)
    assert result.shape == (5, 1)
    assert set(result[:-1, 0]) == set([0, 1, 2])
    assert result[-1, 0] == -2


def test_onehot_binning_inverse_transform():
    """Test inverse_transform returns the correct representatives."""
    X = np.array([[1.0], [2.0], [3.0]])
    binner = OneHotBinning()
    binner.fit(X)
    result = binner.transform(X)
    inv = binner.inverse_transform(result)
    # Should recover the original values
    assert np.allclose(sorted(inv.ravel()), [1.0, 2.0, 3.0])


def test_onehot_binning_unseen_value():
    """Test that unseen values are mapped to -1 and inverse_transform returns nan."""
    X_train = np.array([[1.0], [2.0]])
    X_test = np.array([[3.0]])
    binner = OneHotBinning()
    binner.fit(X_train)
    result = binner.transform(X_test)
    assert result[0, 0] == -1
    inv = binner.inverse_transform(result)
    assert np.isnan(inv[0, 0])


def test_onehot_binning_multicolumn():
    """Test OneHotBinning with multiple columns."""
    X = np.array([[1.0, 10.0], [2.0, 20.0], [1.0, 10.0]])
    binner = OneHotBinning()
    binner.fit(X)
    # Each column should have correct unique bins
    bins0 = sorted([d["singleton"] for d in binner.bins_[0]])
    bins1 = sorted([d["singleton"] for d in binner.bins_[1]])
    assert bins0 == [1.0, 2.0]
    assert bins1 == [10.0, 20.0]
    result = binner.transform(X)
    assert result.shape == (3, 2)
    # All values should be assigned to a bin
    assert np.all(result >= 0)


def test_onehot_binning_with_bin_spec():
    """Test OneHotBinning with a provided bin_spec."""
    bin_spec = {0: [{"singleton": 5.0}, {"singleton": 7.0}]}
    binner = OneHotBinning(bin_spec=bin_spec)
    X = np.array([[5.0], [7.0], [8.0]])
    binner.fit(X)
    # Should use the provided bins, not infer from data
    assert binner.bins_[0][0]["singleton"] == 5.0
    assert binner.bins_[0][1]["singleton"] == 7.0
    result = binner.transform(X)
    # 5.0 -> 0, 7.0 -> 1, 8.0 -> -1
    assert (result[:, 0] == np.array([0, 1, -1])).all()
