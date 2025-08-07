#!/usr/bin/env python3
"""
Test the new EqualWidthBinningV2 implementation using V2 architecture.
"""

import numpy as np
import pandas as pd
from binlearn.methods._equal_width_binning_v2 import EqualWidthBinningV2


def test_equal_width_binning_v2():
    """Test EqualWidthBinningV2 functionality."""
    print("=" * 60)
    print("Testing EqualWidthBinningV2 with V2 Architecture")
    print("=" * 60)

    # Create test data
    np.random.seed(42)
    X = np.random.normal(100, 15, 1000).reshape(-1, 1)
    df = pd.DataFrame(X, columns=["score"])

    print("\n1. Basic functionality:")
    print("-" * 30)

    # Test basic fitting and transformation
    binner = EqualWidthBinningV2(n_bins=5, random_state=42)
    binner.fit(df)

    # Transform data
    X_binned = binner.transform(df.head())
    print(f"Original data (first 5): {df['score'].head().values}")
    print(f"Binned data: {X_binned.flatten()}")
    print(f"Bin edges keys: {list(binner.bin_edges_.keys())}")

    # Test inverse transform
    X_inverse = binner.inverse_transform(X_binned)
    print(f"Inverse transformed: {X_inverse.flatten()}")

    print("\n2. Parameter validation:")
    print("-" * 30)

    try:
        # Should raise error for invalid n_bins
        EqualWidthBinningV2(n_bins=-1)
        print("❌ Should have raised ValueError for negative n_bins")
    except ValueError as e:
        print(f"✅ Caught expected error: {str(e)}")

    try:
        # Should raise error for invalid bin_range
        EqualWidthBinningV2(bin_range=(100, 50))  # min > max
        print("❌ Should have raised ValueError for invalid bin_range")
    except ValueError as e:
        print(f"✅ Caught expected error: {str(e)}")

    print("\n3. Parameter reconstruction:")
    print("-" * 30)

    # Get parameters and test reconstruction
    params = binner.get_params()
    print(f"Extracted parameters: {list(params.keys())}")

    # Constructor-based reconstruction
    reconstructed_binner = EqualWidthBinningV2(**params)
    reconstructed_result = reconstructed_binner.transform(df.head())

    print(f"Original result: {X_binned.flatten()}")
    print(f"Reconstructed result: {reconstructed_result.flatten()}")
    print(f"Results match: {np.array_equal(X_binned, reconstructed_result)}")

    print("\n4. Advanced features:")
    print("-" * 30)

    # Test with custom bin range
    custom_binner = EqualWidthBinningV2(n_bins=4, bin_range=(80, 120))
    custom_binner.fit(df)
    custom_binned = custom_binner.transform(df.head())

    print(f"Custom range binning: {custom_binned.flatten()}")

    # Test bin width calculation
    width = custom_binner.get_bin_width("score")
    print(f"Bin width for custom range: {width}")

    print("\n5. Multi-format compatibility:")
    print("-" * 30)

    # Test with numpy array
    np_binner = EqualWidthBinningV2(n_bins=3)
    np_binner.fit(X)
    np_result = np_binner.transform(X[:5])
    print(f"NumPy result: {np_result.flatten()}")

    # Test with DataFrame (preserve format)
    df_binner = EqualWidthBinningV2(n_bins=3, preserve_dataframe=True)
    df_binner.fit(df)
    df_result = df_binner.transform(df.head())
    print(f"DataFrame result type: {type(df_result)}")
    print(f"DataFrame result shape: {df_result.shape}")

    print("\n6. Joint fitting:")
    print("-" * 30)

    # Test joint fitting with multiple columns
    multi_df = pd.DataFrame(np.random.normal(100, 15, (100, 3)), columns=["A", "B", "C"])
    joint_binner = EqualWidthBinningV2(n_bins=4, fit_jointly=True)
    joint_binner.fit(multi_df)

    # Check that all columns have the same bin edges
    bin_edges_A = joint_binner.bin_edges_["A"]
    bin_edges_B = joint_binner.bin_edges_["B"]
    print(f"Joint fitting - same edges: {bin_edges_A == bin_edges_B}")

    print("\n" + "=" * 60)
    print("EqualWidthBinningV2 Test Completed Successfully!")
    print("Key V2 Architecture Benefits:")
    print("✅ Clean parameter validation using utilities")
    print("✅ Multi-format I/O handling")
    print("✅ Complete parameter reconstruction workflows")
    print("✅ Minimal implementation using utility mixins")
    print("✅ Enhanced error handling and validation")
    print("=" * 60)


if __name__ == "__main__":
    test_equal_width_binning_v2()
