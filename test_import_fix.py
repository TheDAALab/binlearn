#!/usr/bin/env python3

# Test that the import error is fixed
try:
    from binning.utils.bin_operations import (
        validate_bin_edges_format,
        validate_bin_representatives_format,
        validate_bins,
        default_representatives,
        create_bin_masks,
    )

    print("✓ All imports from bin_operations work")
except ImportError as e:
    print(f"✗ Import error: {e}")

# Test sklearn cloning
try:
    from sklearn.base import clone
    from binning.methods import ManualIntervalBinning

    binner = ManualIntervalBinning(bin_edges={0: [0, 10, 20, 30]})
    cloned = clone(binner)

    orig_params = binner.get_params()
    cloned_params = cloned.get_params()

    if orig_params == cloned_params:
        print("✓ sklearn cloning works and parameters match")
    else:
        print("✗ sklearn cloning parameter mismatch")

except Exception as e:
    print(f"✗ sklearn cloning failed: {e}")

print("Import error fix verification complete!")
