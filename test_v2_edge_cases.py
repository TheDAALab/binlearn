"""
Test V2 architecture edge case handling.

Tests that the V2 implementations properly handle edge cases like:
- NaN values
- Constant data
- Infinite values
- Empty data
"""

import numpy as np
import pytest
from binlearn.methods import EqualWidthBinningV2, Chi2BinningV2, SingletonBinningV2


def test_equal_width_v2_nan_handling():
    """Test that EqualWidthBinningV2 handles NaN values properly."""
    print("Testing EqualWidthBinningV2 NaN handling...")

    # Test data with NaN values
    X = np.array([[1.0, 2.0, np.nan, 4.0, 5.0]]).T
    y = np.array([0, 1, 0, 1, 0])

    binner = EqualWidthBinningV2(n_bins=3)
    X_binned = binner.fit_transform(X, y)

    assert X_binned.shape == X.shape, "Output shape should match input"
    assert binner.bin_edges_ is not None, "Bin edges should be computed"
    assert len(binner.bin_edges_[0]) == 4, "Should have 4 edges for 3 bins"
    print("  ✓ NaN handling: PASSED")


def test_equal_width_v2_constant_data():
    """Test that EqualWidthBinningV2 handles constant data."""
    print("Testing EqualWidthBinningV2 constant data handling...")

    # Test constant data
    X = np.array([[5.0, 5.0, 5.0, 5.0, 5.0]]).T
    y = np.array([0, 1, 0, 1, 0])

    binner = EqualWidthBinningV2(n_bins=3)
    X_binned = binner.fit_transform(X, y)

    assert X_binned.shape == X.shape, "Output shape should match input"
    assert binner.bin_edges_ is not None, "Bin edges should be computed"
    # Should handle constant data by adding epsilon
    edges = binner.bin_edges_[0]
    assert edges[0] < edges[-1], "Should create valid range for constant data"
    print("  ✓ Constant data handling: PASSED")


def test_equal_width_v2_all_nan():
    """Test that EqualWidthBinningV2 handles all-NaN columns."""
    print("Testing EqualWidthBinningV2 all-NaN handling...")

    # Test all-NaN data
    X = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]).T
    y = np.array([0, 1, 0, 1, 0])

    binner = EqualWidthBinningV2(n_bins=3)
    X_binned = binner.fit_transform(X, y)

    assert X_binned.shape == X.shape, "Output shape should match input"
    assert binner.bin_edges_ is not None, "Bin edges should be computed"
    # Should use default range [0, 1] for all-NaN data
    edges = binner.bin_edges_[0]
    assert edges[0] < edges[-1], "Should create valid default range for all-NaN data"
    print("  ✓ All-NaN handling: PASSED")


def test_equal_width_v2_custom_range():
    """Test that EqualWidthBinningV2 respects custom bin_range."""
    print("Testing EqualWidthBinningV2 custom range...")

    X = np.array([[1.0, 2.0, 8.0, 9.0, 10.0]]).T
    y = np.array([0, 1, 0, 1, 0])

    # Use custom range that's different from data range
    binner = EqualWidthBinningV2(n_bins=4, bin_range=(0.0, 20.0))
    X_binned = binner.fit_transform(X, y)

    assert X_binned.shape == X.shape, "Output shape should match input"
    edges = binner.bin_edges_[0]
    assert edges[0] == 0.0, "Should use custom min"
    assert edges[-1] == 20.0, "Should use custom max"
    assert len(edges) == 5, "Should have 5 edges for 4 bins"
    print("  ✓ Custom range: PASSED")


def test_v2_parameter_reconstruction():
    """Test that V2 classes support parameter reconstruction."""
    print("Testing V2 parameter reconstruction...")

    # Test EqualWidthBinningV2
    ew1 = EqualWidthBinningV2(n_bins=7, clip=True, preserve_dataframe=False)
    params = ew1.get_params()
    ew2 = EqualWidthBinningV2(**params)

    assert ew2.n_bins == 7, "n_bins should be preserved"
    assert ew2.clip == True, "clip should be preserved"
    assert ew2.preserve_dataframe == False, "preserve_dataframe should be preserved"
    print("  ✓ EqualWidthBinningV2 reconstruction: PASSED")

    # Test SingletonBinningV2
    sb1 = SingletonBinningV2(preserve_dataframe=True)
    params = sb1.get_params()
    sb2 = SingletonBinningV2(**params)

    assert sb2.preserve_dataframe == True, "preserve_dataframe should be preserved"
    print("  ✓ SingletonBinningV2 reconstruction: PASSED")


def test_v2_config_integration():
    """Test that V2 classes properly integrate with config."""
    print("Testing V2 config integration...")

    # Test that classes can be created without explicit parameters (using config defaults)
    ew = EqualWidthBinningV2()
    assert hasattr(ew, "n_bins"), "Should have n_bins from config"

    chi2 = Chi2BinningV2()
    assert hasattr(chi2, "max_bins"), "Should have max_bins from config"
    assert hasattr(chi2, "alpha"), "Should have alpha from config"

    sb = SingletonBinningV2()
    assert hasattr(sb, "preserve_dataframe"), "Should have preserve_dataframe from config"

    print("  ✓ Config integration: PASSED")


def test_v2_never_configurable_params():
    """Test that never-configurable params are never set from config."""
    print("Testing never-configurable parameters...")

    # These parameters should never come from config
    custom_edges = {0: [0, 1, 2, 3]}
    custom_reps = {0: [0.5, 1.5, 2.5]}
    guidance_cols = ["target_col"]
    bin_spec = {0: [1, 2, 3]}

    ew = EqualWidthBinningV2(
        bin_edges=custom_edges, bin_representatives=custom_reps, guidance_columns=guidance_cols
    )

    assert ew.bin_edges == custom_edges, "bin_edges should be preserved"
    assert ew.bin_representatives == custom_reps, "bin_representatives should be preserved"
    assert ew.guidance_columns == guidance_cols, "guidance_columns should be preserved"

    sb = SingletonBinningV2(bin_spec=bin_spec, bin_representatives=custom_reps)
    assert sb.bin_spec == bin_spec, "bin_spec should be preserved"
    assert sb.bin_representatives == custom_reps, "bin_representatives should be preserved"

    print("  ✓ Never-configurable parameters: PASSED")


if __name__ == "__main__":
    print("🔧 V2 ARCHITECTURE EDGE CASE TESTING")
    print("=" * 50)

    test_equal_width_v2_nan_handling()
    test_equal_width_v2_constant_data()
    test_equal_width_v2_all_nan()
    test_equal_width_v2_custom_range()
    test_v2_parameter_reconstruction()
    test_v2_config_integration()
    test_v2_never_configurable_params()

    print()
    print("🎉 ALL V2 EDGE CASE TESTS PASSED! 🎉")
    print("✅ Robust NaN handling implemented")
    print("✅ Constant data handling implemented")
    print("✅ Parameter reconstruction working")
    print("✅ Config integration working")
    print("✅ Never-configurable params protected")
