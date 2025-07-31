#!/usr/bin/env python3

print("Starting import test...")

try:
    import sys

    sys.path.insert(0, ".")

    print("Testing binning.utils.bin_operations...")
    from binning.utils.bin_operations import validate_bin_edges_format

    print("✓ binning.utils.bin_operations import successful")

    print("Testing binning.base._interval_binning_base...")
    from binning.base._interval_binning_base import IntervalBinningBase

    print("✓ IntervalBinningBase import successful")

    print("Testing binning.methods._manual_interval_binning...")
    from binning.methods._manual_interval_binning import ManualIntervalBinning

    print("✓ ManualIntervalBinning import successful")

    print("Creating ManualIntervalBinning instance...")
    mib = ManualIntervalBinning(bin_edges={0: [0, 10, 20, 30]})
    print("✓ ManualIntervalBinning created successfully")

    print("Testing get_params...")
    params = mib.get_params()
    print(f"✓ get_params() works: {params}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()

print("Test completed.")
