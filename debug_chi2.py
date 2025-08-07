#!/usr/bin/env python3

import numpy as np
import warnings
from binlearn.methods import Chi2BinningV2

# Enable all warnings
warnings.filterwarnings("always")


def debug_chi2_binning():
    print("🔍 Debugging Chi2BinningV2 with NaN/inf data")
    print("=" * 50)

    # Test data with NaN and inf
    X = np.array([[1.0, 2.0, np.nan, 4.0, np.inf, 6.0]]).T
    y = np.array([0, 1, 0, 1, 0, 1])

    print(f"Input data X shape: {X.shape}")
    print(f"Input data X: {X.flatten()}")
    print(f"Input data y: {y}")
    print()

    # Create binner
    chi2 = Chi2BinningV2(max_bins=4, min_bins=2)
    print("Chi2BinningV2 created successfully")

    try:
        print("Attempting to fit...")
        chi2.fit(X, y)
        print("✅ Fit successful!")
        print(f"bin_edges_: {chi2.bin_edges_}")
        print(f"bin_representatives_: {chi2.bin_representatives_}")

        print("\nAttempting to transform...")
        X_transformed = chi2.transform(X)
        print("✅ Transform successful!")
        print(f"Transformed shape: {X_transformed.shape}")
        print(f"Transformed data: {X_transformed.flatten()}")

        # Test inverse transform
        print("\nAttempting to inverse transform...")
        X_inverse = chi2.inverse_transform(X_transformed)
        print("✅ Inverse transform successful!")
        print(f"Inverse transformed shape: {X_inverse.shape}")
        print(f"Inverse transformed data: {X_inverse.flatten()}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()

        # Try to inspect the internal state
        print(f"\nInternal state:")
        print(f"bin_edges_: {getattr(chi2, 'bin_edges_', 'NOT SET')}")
        print(f"bin_representatives_: {getattr(chi2, 'bin_representatives_', 'NOT SET')}")


if __name__ == "__main__":
    debug_chi2_binning()
