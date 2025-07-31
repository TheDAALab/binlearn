#!/usr/bin/env python3
import numpy as np
from binning.utils.flexible_bin_operations import (
    generate_default_flexible_representatives,
    find_flexible_bin_for_value,
)

# Test half-open intervals with infinity
bin_defs = [(-float("inf"), 0), (0, 5), (5, float("inf"))]
print("Bin definitions:", bin_defs)

try:
    reps = generate_default_flexible_representatives(bin_defs)
    print("Representatives:", reps)
except Exception as e:
    print("Error generating representatives:", e)

# Test value finding
test_values = [-1000, -1, 0, 2.5, 5, 10, 1000]
for val in test_values:
    bin_idx = find_flexible_bin_for_value(val, bin_defs)
    print(f"Value {val} -> Bin {bin_idx}")
