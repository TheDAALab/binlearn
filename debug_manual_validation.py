#!/usr/bin/env python3

import sys

sys.path.insert(0, ".")

from binning.methods import ManualIntervalBinning

print("Testing ManualIntervalBinning validation errors...")

# Test invalid edge type
try:
    ManualIntervalBinning(bin_edges={0: "invalid"})
except Exception as e:
    print(f"Invalid edge type error: {e}")

# Test insufficient edges
try:
    ManualIntervalBinning(bin_edges={0: [10.0]})
except Exception as e:
    print(f"Insufficient edges error: {e}")

# Test unsorted edges
try:
    ManualIntervalBinning(bin_edges={0: [10.0, 5.0, 15.0]})
except Exception as e:
    print(f"Unsorted edges error: {e}")
