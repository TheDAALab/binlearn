#!/usr/bin/env python3
"""
Test script to validate V2 architecture with NaN and infinite values.
"""

import numpy as np
import pandas as pd
from binlearn.methods._singleton_binning_v2 import SingletonBinningV2

# Test data with NaN and inf
X_with_special = pd.DataFrame(
    {
        "numeric1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        "numeric2": [10.0, np.inf, 30.0, 40.0, -np.inf, 60.0],
    }
)

print("Testing V2 Architecture with NaN and infinite values...")
print(f"Data shape: {X_with_special.shape}")
print(f"Data types: {X_with_special.dtypes}")
print(f"Data preview:\n{X_with_special}")
print()

# Test SingletonBinningV2 with special values
print("Testing SingletonBinningV2 with NaN/inf values...")
try:
    binning = SingletonBinningV2()
    binning.fit(X_with_special)

    result = binning.transform(X_with_special)
    print(f"  Transform successful! Result shape: {result.shape}")
    print(f"  Result preview:\n{result}")

    # Test parameter reconstruction with special values
    params = binning.get_params()
    binning_reconstructed = SingletonBinningV2(**params)
    print(f"  Parameter reconstruction: PASSED")
    print("  SingletonBinningV2 with NaN/inf: PASSED")

except Exception as e:
    print(f"  SingletonBinningV2 with NaN/inf: FAILED - {e}")

print("\nNaN/inf handling validation complete!")
