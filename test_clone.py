#!/usr/bin/env python3

print("Testing sklearn clone...")

try:
    import sys

    sys.path.insert(0, ".")

    from sklearn.base import clone
    from binning.methods._manual_interval_binning import ManualIntervalBinning

    print("Creating original ManualIntervalBinning...")
    mib = ManualIntervalBinning(bin_edges={0: [0, 10, 20, 30]})
    print("✓ Original created successfully")

    print(f"Original params: {mib.get_params()}")

    print("Testing sklearn clone...")
    mib_cloned = clone(mib)
    print("✓ sklearn clone succeeded!")

    cloned_params = mib_cloned.get_params()
    print(f"Cloned params: {cloned_params}")

    # Verify they're different objects but same params
    print(f"Same object? {mib is mib_cloned}")
    print(f"Params equal? {mib.get_params() == mib_cloned.get_params()}")

    # Test that both can transform (if fitted)
    print("Testing that cloned object works independently...")
    import numpy as np

    X = np.array([[5, 15, 25]]).T

    # Both should be able to transform since ManualIntervalBinning is pre-fitted
    result_orig = mib.transform(X)
    result_cloned = mib_cloned.transform(X)

    print(f"Original transform result: {result_orig.flatten()}")
    print(f"Cloned transform result: {result_cloned.flatten()}")
    print(f"Results equal? {np.array_equal(result_orig, result_cloned)}")

    print("✓ sklearn clone test PASSED!")

except Exception as e:
    print(f"✗ sklearn clone test FAILED: {e}")
    import traceback

    traceback.print_exc()
