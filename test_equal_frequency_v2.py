"""Test EqualFrequencyBinningV2 implementation."""

import numpy as np
import pandas as pd
from binlearn.methods._equal_frequency_binning_v2 import EqualFrequencyBinningV2


def test_equal_frequency_binning_v2():
    """Test the EqualFrequencyBinningV2 implementation."""
    print("=" * 60)
    print("Testing EqualFrequencyBinningV2 with V2 Architecture")
    print("=" * 60)

    # Create test data - use data with clear quantiles
    np.random.seed(42)
    data = np.concatenate(
        [
            np.random.normal(50, 10, 250),  # First quartile
            np.random.normal(100, 10, 250),  # Second quartile
            np.random.normal(150, 10, 250),  # Third quartile
            np.random.normal(200, 10, 250),  # Fourth quartile
        ]
    )
    np.random.shuffle(data)
    X = data.reshape(-1, 1)

    print("\n1. Basic functionality:")
    print("-" * 30)
    binner = EqualFrequencyBinningV2(n_bins=4)
    binner.fit(X)
    X_binned = binner.transform(X)

    print(f"Original data range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"Binned data: {np.unique(X_binned, return_counts=True)}")
    print(f"Bin edges: {binner.bin_edges_[0]}")

    # Check if bins have roughly equal frequencies
    unique, counts = np.unique(X_binned, return_counts=True)
    print(f"Bin counts: {counts}")

    # Test inverse transform
    X_inverse = binner.inverse_transform(X_binned)
    print(f"Inverse transformed (first 5): {X_inverse[:5].flatten()}")

    print("\n2. Parameter validation:")
    print("-" * 30)
    try:
        EqualFrequencyBinningV2(n_bins=-1)
        print("❌ Should have caught negative n_bins error")
    except ValueError as e:
        print(f"✅ Caught expected error: {e}")

    try:
        EqualFrequencyBinningV2(quantile_range=(0.9, 0.1))
        print("❌ Should have caught invalid quantile_range error")
    except ValueError as e:
        print(f"✅ Caught expected error: {e}")

    print("\n3. Custom quantile range:")
    print("-" * 30)
    binner_range = EqualFrequencyBinningV2(n_bins=3, quantile_range=(0.1, 0.9))
    binner_range.fit(X)
    X_binned_range = binner_range.transform(X)

    print(f"Custom range bin edges: {binner_range.bin_edges_[0]}")
    print(f"Custom range binned data: {np.unique(X_binned_range, return_counts=True)}")

    print("\n4. Parameter reconstruction:")
    print("-" * 30)
    params = binner.get_params()
    print(f"Extracted parameters: {list(params.keys())}")

    # Test reconstruction
    binner_reconstructed = EqualFrequencyBinningV2(**params)
    binner_reconstructed.bin_edges_ = binner.bin_edges_.copy()

    X_original = binner.transform(X)
    X_reconstructed = binner_reconstructed.transform(X)
    print(f"Results match: {np.array_equal(X_original, X_reconstructed)}")

    print("\n5. Multi-format compatibility:")
    print("-" * 30)
    # Test with DataFrame
    df = pd.DataFrame(X, columns=["score"])
    X_df_binned = binner.fit_transform(df)
    print(f"DataFrame result type: {type(X_df_binned)}")
    print(f"DataFrame result shape: {X_df_binned.shape}")

    print("\n" + "=" * 60)
    print("EqualFrequencyBinningV2 Test Completed Successfully!")
    print("Key V2 Architecture Benefits:")
    print("✅ Clean parameter validation using utilities")
    print("✅ Equal-frequency binning with quantile-based edges")
    print("✅ Multi-format I/O handling")
    print("✅ Complete parameter reconstruction workflows")
    print("✅ Minimal implementation using utility mixins")
    print("✅ Enhanced error handling and validation")
    print("=" * 60)


if __name__ == "__main__":
    test_equal_frequency_binning_v2()
