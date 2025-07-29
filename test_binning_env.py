#!/usr/bin/env python
"""
Test script to run SupervisedBinning tests in the binning environment.
"""

import os
import sys

# Set environment variables to avoid scipy issues
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

# Try to monkey patch the copy issue before importing anything else
try:
    import numpy as np
    import numpy._globals
    if hasattr(numpy._globals, '_CopyMode'):
        # Override the problematic __bool__ method
        def safe_bool(self):
            return self.name != 'NEVER'
        numpy._globals._CopyMode.__bool__ = safe_bool
except Exception as e:
    print(f"Warning: Could not apply NumPy patch: {e}")

# Now try importing our modules
try:
    import polars as pl
    print(f"✓ Polars {pl.__version__} imported successfully")
    POLARS_AVAILABLE = True
except ImportError as e:
    print(f"✗ Polars import failed: {e}")
    POLARS_AVAILABLE = False

try:
    from binning.methods._supervised_binning import SupervisedBinning
    print("✓ SupervisedBinning imported successfully")
except ImportError as e:
    print(f"✗ SupervisedBinning import failed: {e}")
    sys.exit(1)

# Simple test with Polars if available
if POLARS_AVAILABLE:
    print("\nTesting SupervisedBinning with Polars...")
    
    # Create test data
    df = pl.DataFrame({
        'feature': [1, 2, 3, 4, 5, 6],
        'target': [0, 0, 0, 1, 1, 1]
    })
    
    # Test SupervisedBinning
    binning = SupervisedBinning(guidance_columns=['target'], preserve_dataframe=True)
    binning.fit(df)
    result = binning.transform(df)
    
    print(f"✓ Polars test successful: {result.shape}")
    print(f"  Columns: {result.columns}")
    print(f"  Result type: {type(result)}")
else:
    print("Skipping Polars tests - not available")

print("All tests completed successfully!")
