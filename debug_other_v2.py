#!/usr/bin/env python3
"""Debug other V2 binning classes to ensure column resolution works."""

import numpy as np
from binlearn.methods import EqualWidthBinningV2, SingletonBinningV2


def test_equal_width():
    """Test EqualWidthBinningV2 with numpy array input."""
    print("🔍 Testing EqualWidthBinningV2")
    print("=" * 40)

    # Create test data
    X = np.array([[1.0, 10.0], [2.0, 20.0], [np.nan, 30.0], [4.0, np.inf], [5.0, 50.0]])
    print(f"Input X shape: {X.shape}")
    print(f"Input X:\n{X}")

    try:
        ew = EqualWidthBinningV2(n_bins=3)
        print("EqualWidthBinningV2 created successfully")

        print("Attempting to fit...")
        ew.fit(X)
        print("✅ Fit successful!")

        print("Attempting to transform...")
        X_transformed = ew.transform(X)
        print("✅ Transform successful!")
        print(f"Transformed:\n{X_transformed}")

        print("Attempting to inverse transform...")
        X_inverse = ew.inverse_transform(X_transformed)
        print("✅ Inverse transform successful!")
        print(f"Inverse transformed:\n{X_inverse}")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_singleton():
    """Test SingletonBinningV2 with numpy array input."""
    print("\n🔍 Testing SingletonBinningV2")
    print("=" * 40)

    # Create test data with discrete values
    X = np.array([[1.0], [2.0], [1.0], [3.0], [2.0], [np.nan]])
    print(f"Input X shape: {X.shape}")
    print(f"Input X: {X.flatten()}")

    try:
        sb = SingletonBinningV2()
        print("SingletonBinningV2 created successfully")

        print("Attempting to fit...")
        sb.fit(X)
        print("✅ Fit successful!")

        print("Attempting to transform...")
        X_transformed = sb.transform(X)
        print("✅ Transform successful!")
        print(f"Transformed: {X_transformed.flatten()}")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success1 = test_equal_width()
    success2 = test_singleton()

    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"EqualWidthBinningV2: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"SingletonBinningV2: {'✅ PASSED' if success2 else '❌ FAILED'}")
