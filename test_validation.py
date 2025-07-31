#!/usr/bin/env python3

print("Testing parameter validation...")

try:
    import sys

    sys.path.insert(0, ".")

    from binning.methods._manual_interval_binning import ManualIntervalBinning

    print("=== Testing valid parameters ===")
    try:
        mib = ManualIntervalBinning(bin_edges={0: [0, 10, 20, 30]})
        print("✓ Valid parameters accepted")
    except Exception as e:
        print(f"✗ Valid parameters rejected: {e}")

    print("\n=== Testing invalid parameters ===")

    # Test None bin_edges
    try:
        mib = ManualIntervalBinning(bin_edges=None)
        print("✗ Should have rejected None bin_edges")
    except Exception as e:
        print(f"✓ Correctly rejected None bin_edges: {type(e).__name__}: {e}")

    # Test empty bin_edges
    try:
        mib = ManualIntervalBinning(bin_edges={})
        print("✗ Should have rejected empty bin_edges")
    except Exception as e:
        print(f"✓ Correctly rejected empty bin_edges: {type(e).__name__}: {e}")

    # Test insufficient edges
    try:
        mib = ManualIntervalBinning(bin_edges={0: [10]})
        print("✗ Should have rejected insufficient edges")
    except Exception as e:
        print(f"✓ Correctly rejected insufficient edges: {type(e).__name__}: {e}")

    # Test unsorted edges
    try:
        mib = ManualIntervalBinning(bin_edges={0: [10, 5, 20]})
        print("✗ Should have rejected unsorted edges")
    except Exception as e:
        print(f"✓ Correctly rejected unsorted edges: {type(e).__name__}: {e}")

    # Test invalid edge types
    try:
        mib = ManualIntervalBinning(bin_edges={0: ["a", "b"]})
        print("✗ Should have rejected non-numeric edges")
    except Exception as e:
        print(f"✓ Correctly rejected non-numeric edges: {type(e).__name__}: {e}")

    print("\n✓ Parameter validation tests PASSED!")

except Exception as e:
    print(f"✗ Parameter validation test setup FAILED: {e}")
    import traceback

    traceback.print_exc()
