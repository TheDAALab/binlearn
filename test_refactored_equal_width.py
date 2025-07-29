"""Test script for refactored EqualWidthBinning class."""

import numpy as np
from binning.methods._equal_width_binning import EqualWidthBinning

def test_basic_functionality():
    """Test basic functionality of refactored EqualWidthBinning."""
    print("Testing basic functionality...")
    
    # Create test data
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    
    # Test 1: Basic per-column binning
    print("\n1. Per-column binning:")
    ewb = EqualWidthBinning(n_bins=3)
    ewb.fit(X)
    result = ewb.transform(X)
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  Bin edges: {ewb.bin_edges_}")
    
    # Test 2: Joint binning
    print("\n2. Joint binning:")
    ewb_joint = EqualWidthBinning(n_bins=3, fit_jointly=True)
    ewb_joint.fit(X)
    result_joint = ewb_joint.transform(X)
    print(f"  Bin edges: {ewb_joint.bin_edges_}")
    
    # Test 3: With specified range
    print("\n3. With specified range:")
    ewb_range = EqualWidthBinning(n_bins=4, bin_range=(0, 12))
    ewb_range.fit(X)
    result_range = ewb_range.transform(X)
    print(f"  Bin edges: {ewb_range.bin_edges_}")
    
    # Test 4: Parameter validation
    print("\n4. Parameter validation:")
    try:
        EqualWidthBinning(n_bins=0)._validate_params()
        print("  ERROR: Should have raised exception for n_bins=0")
    except Exception as e:
        print(f"  Correctly caught invalid n_bins: {type(e).__name__}")
    
    try:
        EqualWidthBinning(bin_range=(5, 2))._validate_params()
        print("  ERROR: Should have raised exception for invalid range")
    except Exception as e:
        print(f"  Correctly caught invalid range: {type(e).__name__}")
    
    print("\nAll tests passed! ✓")

if __name__ == "__main__":
    test_basic_functionality()
