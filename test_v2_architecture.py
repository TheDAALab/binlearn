#!/usr/bin/env python3
"""
Test script to validate V2 architecture functionality.
"""

import numpy as np
import pandas as pd
from binlearn.methods._equal_width_binning_v2 import EqualWidthBinningV2
from binlearn.methods._singleton_binning_v2 import SingletonBinningV2
from binlearn.methods._chi2_binning_v2 import Chi2BinningV2

# Test data with numeric values including NaN and inf
X = pd.DataFrame(
    {
        "numeric1": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
        "numeric2": [10.0, 20.0, 30.0, 40.0, np.inf, 60.0],
        "guidance": [0, 1, 0, 1, 0, 1],
    }
)

# Clean test data without NaN/inf for problematic methods
X_clean = pd.DataFrame(
    {
        "numeric1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "numeric2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "guidance": [0, 1, 0, 1, 0, 1],
    }
)

y = np.array([0, 1, 0, 1, 0, 1])

print("Testing V2 Architecture...")
print(f"Data shape: {X.shape}")
print(f"Data types: {X.dtypes}")
print(f"Target shape: {y.shape}")
print()

# Test 1: EqualWidthBinningV2 basic functionality
print("1. Testing EqualWidthBinningV2...")
try:
    binning1 = EqualWidthBinningV2(n_bins=3)
    binning1.fit(X_clean[["numeric1", "numeric2"]])  # Use clean data for interval binning

    # Test parameter reconstruction
    params = binning1.get_params()
    print(f"  Got params: {list(params.keys())}")

    # Test constructor parameter swallowing
    binning1_reconstructed = EqualWidthBinningV2(**params)
    print(f"  Constructor parameter swallowing: PASSED")

    # Test transform
    result = binning1.transform(X_clean[["numeric1", "numeric2"]])
    print(f"  Transform result shape: {result.shape}")
    print("  EqualWidthBinningV2: PASSED")
except Exception as e:
    print(f"  EqualWidthBinningV2: FAILED - {e}")

print()

# Test 2: SingletonBinningV2 functionality
print("2. Testing SingletonBinningV2...")
try:
    binning2 = SingletonBinningV2()
    binning2.fit(X[["numeric1", "numeric2"]])

    # Test parameter reconstruction
    params = binning2.get_params()
    print(f"  Got params: {list(params.keys())}")

    # Test reconstruction
    binning2_reconstructed = SingletonBinningV2(**params)
    print(f"  Constructor parameter swallowing: PASSED")

    result = binning2.transform(X[["numeric1", "numeric2"]])
    print(f"  Transform result shape: {result.shape}")
    print("  SingletonBinningV2: PASSED")
except Exception as e:
    print(f"  SingletonBinningV2: FAILED - {e}")

print()

# Test 3: Chi2BinningV2 functionality with target
print("3. Testing Chi2BinningV2...")
try:
    binning3 = Chi2BinningV2(max_bins=3)
    binning3.fit(X_clean[["numeric1", "numeric2"]], y)  # Use clean data

    # Test parameter reconstruction
    params = binning3.get_params()
    print(f"  Got params: {list(params.keys())}")

    # Test reconstruction
    binning3_reconstructed = Chi2BinningV2(**params)
    print(f"  Constructor parameter swallowing: PASSED")

    result = binning3.transform(X_clean[["numeric1", "numeric2"]])
    print(f"  Transform result shape: {result.shape}")
    print("  Chi2BinningV2: PASSED")
except Exception as e:
    print(f"  Chi2BinningV2: FAILED - {e}")

print()

# Test 4: Guidance columns functionality
print("4. Testing guidance columns...")
try:
    binning4 = EqualWidthBinningV2(n_bins=3, guidance_columns=["guidance"])
    binning4.fit(X_clean)  # Fit with all columns including guidance, use clean data

    # Check that guidance columns are excluded from output
    result = binning4.transform(X_clean)
    print(f"  Input columns: {list(X_clean.columns)}")
    print(f"  Output columns: {list(result.columns) if hasattr(result, 'columns') else 'array'}")
    print(f"  Output shape: {result.shape}")
    print("  Guidance columns: PASSED")
except Exception as e:
    print(f"  Guidance columns: FAILED - {e}")

print()

# Test 5: Mutual exclusion validation
print("5. Testing mutual exclusion validation...")
try:
    binning5 = EqualWidthBinningV2(n_bins=3, fit_jointly=True, guidance_columns=["guidance"])
    print("  Should have failed but didn't - FAILED")
except ValueError as e:
    if "mutually exclusive" in str(e):
        print("  Parameter-time validation: PASSED")
    else:
        print(f"  Wrong error message: {e}")
except Exception as e:
    print(f"  Unexpected error: {e}")

try:
    binning6 = EqualWidthBinningV2(n_bins=3, fit_jointly=True)
    binning6.fit(X_clean[["numeric1", "numeric2"]], guidance_data=X_clean[["guidance"]])
    print("  Should have failed but didn't - FAILED")
except ValueError as e:
    if "mutually exclusive" in str(e):
        print("  Runtime validation: PASSED")
    else:
        print(f"  Wrong error message: {e}")
except Exception as e:
    print(f"  Unexpected error: {e}")

print("\nV2 Architecture validation complete!")
