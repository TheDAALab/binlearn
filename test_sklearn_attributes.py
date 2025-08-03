#!/usr/bin/env python3
"""Test script to verify sklearn attributes are set from bin specifications."""

from binlearn.methods import EqualWidthBinning, ManualFlexibleBinning


def test_interval_binning_sklearn_attributes():
    """Test that IntervalBinningBase sets sklearn attributes from bin_edges."""
    print("Testing IntervalBinningBase sklearn attributes...")

    # Create bin edges for 2 columns
    bin_edges = {
        0: [0.0, 1.0, 2.0, 3.0],  # 3 bins for column 0
        1: [10.0, 20.0, 30.0],  # 2 bins for column 1
    }

    # Create instance with bin_edges
    binner = EqualWidthBinning(bin_edges=bin_edges)

    # Check that we can access private attributes directly
    print(f"  _feature_names_in: {binner._feature_names_in}")
    print(f"  _n_features_in: {binner._n_features_in}")
    print(f"  _fitted: {binner._fitted}")

    # Should be [0, 1] and 2
    assert binner._feature_names_in == [0, 1], f"Expected [0, 1], got {binner._feature_names_in}"
    assert binner._n_features_in == 2, f"Expected 2, got {binner._n_features_in}"
    assert binner._fitted is True, f"Expected True, got {binner._fitted}"

    print("  ✓ IntervalBinningBase test passed!")


def test_interval_binning_with_guidance():
    """Test that IntervalBinningBase includes guidance columns in sklearn attributes."""
    print("Testing IntervalBinningBase with guidance columns...")

    # Create bin edges for 2 columns + 1 guidance column
    bin_edges = {
        0: [0.0, 1.0, 2.0, 3.0],  # 3 bins for column 0
        1: [10.0, 20.0, 30.0],  # 2 bins for column 1
    }

    # Create instance with bin_edges and guidance
    binner = EqualWidthBinning(bin_edges=bin_edges, guidance_columns=[2])

    # Check private attributes
    print(f"  _feature_names_in: {binner._feature_names_in}")
    print(f"  _n_features_in: {binner._n_features_in}")

    # Should be [0, 1, 2] and 3
    assert binner._feature_names_in == [
        0,
        1,
        2,
    ], f"Expected [0, 1, 2], got {binner._feature_names_in}"
    assert binner._n_features_in == 3, f"Expected 3, got {binner._n_features_in}"

    print("  ✓ IntervalBinningBase with guidance test passed!")


def test_flexible_binning_sklearn_attributes():
    """Test that FlexibleBinningBase sets sklearn attributes from bin_spec."""
    print("Testing FlexibleBinningBase sklearn attributes...")

    # Create bin spec for 2 columns
    bin_spec = {
        0: [1, 2, (3, 5)],  # Mixed bins for column 0
        1: [(10, 20), (20, 30)],  # Interval bins for column 1
    }

    # Create instance with bin_spec
    binner = ManualFlexibleBinning(bin_spec=bin_spec)

    # Check private attributes
    print(f"  _feature_names_in: {binner._feature_names_in}")
    print(f"  _n_features_in: {binner._n_features_in}")
    print(f"  _fitted: {binner._fitted}")

    # Should be [0, 1] and 2
    assert binner._feature_names_in == [0, 1], f"Expected [0, 1], got {binner._feature_names_in}"
    assert binner._n_features_in == 2, f"Expected 2, got {binner._n_features_in}"
    assert binner._fitted is True, f"Expected True, got {binner._fitted}"

    print("  ✓ FlexibleBinningBase test passed!")


def test_flexible_binning_with_guidance():
    """Test that FlexibleBinningBase includes guidance columns in sklearn attributes."""
    print("Testing FlexibleBinningBase with guidance columns...")

    # Create bin spec for 2 columns + 1 guidance column
    bin_spec = {
        "feature_a": [1, 2, (3, 5)],  # Mixed bins for feature_a
        "feature_b": [(10, 20), (20, 30)],  # Interval bins for feature_b
    }

    # Create instance with bin_spec and guidance
    binner = ManualFlexibleBinning(bin_spec=bin_spec, guidance_columns=["target"])

    # Check private attributes
    print(f"  _feature_names_in: {binner._feature_names_in}")
    print(f"  _n_features_in: {binner._n_features_in}")

    # Should be ['feature_a', 'feature_b', 'target'] and 3
    expected = ["feature_a", "feature_b", "target"]
    assert (
        binner._feature_names_in == expected
    ), f"Expected {expected}, got {binner._feature_names_in}"
    assert binner._n_features_in == 3, f"Expected 3, got {binner._n_features_in}"

    print("  ✓ FlexibleBinningBase with guidance test passed!")


if __name__ == "__main__":
    test_interval_binning_sklearn_attributes()
    test_interval_binning_with_guidance()
    test_flexible_binning_sklearn_attributes()
    test_flexible_binning_with_guidance()
    print("\n🎉 All tests passed!")
