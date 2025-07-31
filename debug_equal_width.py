#!/usr/bin/env python3

print("Testing EqualWidthBinning attribute issue...")

import sys

sys.path.insert(0, ".")

try:
    from binning.methods import EqualWidthBinning

    # Try to create instance
    print("Creating EqualWidthBinning instance...")
    binner = EqualWidthBinning(n_bins=5)

    print(f"n_bins attribute: {hasattr(binner, 'n_bins')}")
    if hasattr(binner, "n_bins"):
        print(f"n_bins value: {binner.n_bins}")
    else:
        print("n_bins attribute is missing!")
        print(f"Available attributes: {[attr for attr in dir(binner) if not attr.startswith('_')]}")

    # Check if initialization completed
    print(f"Initialization successful: {isinstance(binner, EqualWidthBinning)}")

except Exception as e:
    print(f"Error creating EqualWidthBinning: {e}")
    import traceback

    traceback.print_exc()
