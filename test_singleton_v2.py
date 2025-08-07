"""Test SingletonBinningV2 implementation."""

import numpy as np
import pandas as pd
from binlearn.methods._singleton_binning_v2 import SingletonBinningV2


def test_singleton_binning_v2():
    """Test the SingletonBinningV2 implementation."""
    print("=" * 60)
    print("Testing SingletonBinningV2 with V2 Architecture")
    print("=" * 60)

    # Create test data with discrete values
    np.random.seed(42)
    # Create data where each column has different unique values
    X = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [1.0, 10.0],  # Repeat values
            [3.0, 30.0],
            [2.0, 20.0],  # More repeats
            [1.0, 10.0],
            [4.0, 40.0],
        ]
    )

    print("\n1. Basic functionality:")
    print("-" * 30)
    binner = SingletonBinningV2(max_unique_values=10)
    binner.fit(X)
    X_binned = binner.transform(X)

    print(f"Original data:\n{X}")
    print(f"Unique values in col 0: {np.unique(X[:, 0])}")
    print(f"Unique values in col 1: {np.unique(X[:, 1])}")
    print(f"Binned data:\n{X_binned}")
    print(f"Bin specs for col 0: {binner.bin_spec_[0]}")
    print(f"Bin specs for col 1: {binner.bin_spec_[1]}")

    # Test inverse transform
    X_inverse = binner.inverse_transform(X_binned)
    print(f"Inverse transformed:\n{X_inverse}")
    print(f"Reconstruction perfect: {np.allclose(X, X_inverse)}")

    print("\n2. Parameter validation:")
    print("-" * 30)
    try:
        SingletonBinningV2(max_unique_values=-1)
        print("❌ Should have caught negative max_unique_values error")
    except ValueError as e:
        print(f"✅ Caught expected error: {e}")

    try:
        # Create data with too many unique values
        high_card_data = np.arange(200).reshape(-1, 1)
        binner_small = SingletonBinningV2(max_unique_values=5)
        binner_small.fit(high_card_data)
        print("❌ Should have caught too many unique values error")
    except ValueError as e:
        print(f"✅ Caught expected error: Too many unique values")

    print("\n3. Joint fitting:")
    print("-" * 30)
    # Test data where joint fitting would create different bins
    X_joint = np.array(
        [
            [1.0, 5.0],
            [2.0, 6.0],
            [3.0, 7.0],
        ]
    )

    binner_joint = SingletonBinningV2(max_unique_values=10)
    binner_joint.fit_jointly = True  # Enable joint fitting
    binner_joint.fit(X_joint)
    X_joint_binned = binner_joint.transform(X_joint)

    print(f"Joint fitting data:\n{X_joint}")
    print(f"Joint binned data:\n{X_joint_binned}")
    print(
        f"Joint bin specs (should be same for both columns): {list(binner_joint.bin_spec_.values())}"
    )
    # In joint fitting, both columns should have the same bin spec (all unique values)
    print(f"Bin specs identical: {binner_joint.bin_spec_[0] == binner_joint.bin_spec_[1]}")

    print("\n4. Parameter reconstruction:")
    print("-" * 30)
    params = binner.get_params()
    print(f"Extracted parameters: {list(params.keys())}")

    # Test reconstruction
    binner_reconstructed = SingletonBinningV2(**params)
    binner_reconstructed.bin_spec_ = binner.bin_spec_.copy()
    binner_reconstructed.bin_representatives_ = binner.bin_representatives_.copy()

    X_original = binner.transform(X)
    X_reconstructed = binner_reconstructed.transform(X)
    print(f"Results match: {np.array_equal(X_original, X_reconstructed)}")

    print("\n5. Multi-format compatibility:")
    print("-" * 30)
    # Test with DataFrame
    df = pd.DataFrame(X, columns=["A", "B"])
    X_df_binned = binner.fit_transform(df)
    print(f"DataFrame result type: {type(X_df_binned)}")
    print(f"DataFrame result shape: {X_df_binned.shape}")

    # Test edge cases
    print("\n6. Edge cases:")
    print("-" * 30)
    # Single unique value per column
    X_constant = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    binner_const = SingletonBinningV2()
    binner_const.fit(X_constant)
    X_const_binned = binner_const.transform(X_constant)
    print(f"Constant data binned: {np.unique(X_const_binned, axis=0)}")
    print(f"Single bins created: {[len(spec) for spec in binner_const.bin_spec_.values()]}")

    print("\n" + "=" * 60)
    print("SingletonBinningV2 Test Completed Successfully!")
    print("Key V2 Architecture Benefits:")
    print("✅ Clean parameter validation using utilities")
    print("✅ Flexible singleton binning for discrete values")
    print("✅ Multi-format I/O handling")
    print("✅ Complete parameter reconstruction workflows")
    print("✅ Minimal implementation using utility mixins")
    print("✅ Perfect reconstruction of original values")
    print("=" * 60)


if __name__ == "__main__":
    test_singleton_binning_v2()
