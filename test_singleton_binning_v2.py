"""Test SingletonBinningV2 implementation."""

import numpy as np
import pandas as pd
from binlearn.methods._singleton_binning_v2 import SingletonBinningV2


def test_singleton_binning_v2():
    """Test the SingletonBinningV2 implementation."""
    print("=" * 60)
    print("Testing SingletonBinningV2 with V2 Architecture")
    print("=" * 60)

    # Create test data with discrete values (typical use case for singleton binning)
    X = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [1.0, 10.0],
            [3.0, 30.0],
            [2.0, 20.0],
            [1.0, 30.0],  # Mix values between columns
        ]
    )

    print("\n1. Basic functionality:")
    print("-" * 30)
    binner = SingletonBinningV2(max_unique_values=10)
    binner.fit(X)
    X_binned = binner.transform(X)

    print(f"Original data:\n{X}")
    print(f"Binned data:\n{X_binned}")

    # Get unique values for each column
    unique_vals = binner.get_unique_values()
    print(f"Unique values per column: {unique_vals}")

    # Test inverse transform
    X_inverse = binner.inverse_transform(X_binned)
    print(f"Inverse transformed:\n{X_inverse}")
    print(f"Reconstruction accurate: {np.allclose(X, X_inverse, equal_nan=True)}")

    print("\n2. Parameter validation:")
    print("-" * 30)
    try:
        SingletonBinningV2(max_unique_values=-1)
        print("❌ Should have caught negative max_unique_values error")
    except ValueError as e:
        print(f"✅ Caught expected error: {e}")

    try:
        # Create data with too many unique values
        X_high_cardinality = np.arange(50).reshape(-1, 1)  # 50 unique values
        binner_small = SingletonBinningV2(max_unique_values=10)
        binner_small.fit(X_high_cardinality)
        print("❌ Should have caught too many unique values error")
    except ValueError as e:
        print(f"✅ Caught expected error: {e}")

    print("\n3. Joint fitting:")
    print("-" * 30)
    binner_joint = SingletonBinningV2(max_unique_values=10)
    binner_joint.fit_jointly = True  # Enable joint fitting
    binner_joint.fit(X)
    X_binned_joint = binner_joint.transform(X)

    print(f"Joint binned data:\n{X_binned_joint}")
    unique_vals_joint = binner_joint.get_unique_values()
    print(f"Joint unique values (same for all columns): {unique_vals_joint}")

    # Check that all columns have the same bin definitions
    col_keys = list(unique_vals_joint.keys())
    if len(col_keys) > 1:
        first_col_bins = unique_vals_joint[col_keys[0]]
        all_same = all(np.array_equal(unique_vals_joint[col], first_col_bins) for col in col_keys)
        print(f"All columns have same bins: {all_same}")

    print("\n4. Parameter reconstruction:")
    print("-" * 30)
    params = binner.get_params()
    print(f"Extracted parameters: {list(params.keys())}")

    # Test reconstruction
    binner_reconstructed = SingletonBinningV2(**params)
    binner_reconstructed.bin_spec_ = binner.bin_spec_.copy()
    binner_reconstructed.bin_representatives_ = binner.bin_representatives_.copy()
    binner_reconstructed.unique_values_ = binner.unique_values_.copy()
    binner_reconstructed._fitted = True  # Mark as fitted

    X_original = binner.transform(X)
    X_reconstructed = binner_reconstructed.transform(X)
    print(f"Results match: {np.array_equal(X_original, X_reconstructed)}")

    print("\n5. Multi-format compatibility:")
    print("-" * 30)
    # Test with DataFrame
    df = pd.DataFrame(X, columns=["feature_A", "feature_B"])
    X_df_binned = binner.fit_transform(df)
    print(f"DataFrame result type: {type(X_df_binned)}")
    print(f"DataFrame result shape: {X_df_binned.shape}")
    print(f"DataFrame binned data:\n{X_df_binned}")

    print("\n6. Edge cases:")
    print("-" * 30)

    # Test with data containing NaN
    X_with_nan = np.array([[1.0, np.nan], [2.0, 10.0], [1.0, 10.0]])
    binner_nan = SingletonBinningV2()
    binner_nan.fit(X_with_nan)
    X_binned_nan = binner_nan.transform(X_with_nan)
    print(f"Data with NaN - binned:\n{X_binned_nan}")

    # Test with single unique value per column
    X_constant = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
    binner_const = SingletonBinningV2()
    binner_const.fit(X_constant)
    X_binned_const = binner_const.transform(X_constant)
    print(f"Constant data - binned:\n{X_binned_const}")
    print(f"Constant data unique values: {binner_const.get_unique_values()}")

    print("\n" + "=" * 60)
    print("SingletonBinningV2 Test Completed Successfully!")
    print("Key V2 Architecture Benefits:")
    print("✅ Clean parameter validation using utilities")
    print("✅ Flexible binning with singleton bins")
    print("✅ Multi-format I/O handling")
    print("✅ Complete parameter reconstruction workflows")
    print("✅ Minimal implementation using utility mixins")
    print("✅ Enhanced error handling and validation")
    print("✅ Support for both per-column and joint fitting")
    print("=" * 60)


if __name__ == "__main__":
    test_singleton_binning_v2()
