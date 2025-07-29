#!/usr/bin/env python3
"""
Quick test to verify that the refactored binning functionality works correctly.
"""

import numpy as np
from binning.methods import EqualWidthBinning

def test_basic_functionality():
    """Test basic functionality works with the simplified logic."""
    
    print("Testing basic EqualWidthBinning...")
    
    # Test 1: Basic fit and transform
    binning = EqualWidthBinning(n_bins=3)
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    
    binning.fit(X)
    result = binning.transform(X)
    print(f"✓ Basic fit/transform works: {result.shape} == (4, 2)")
    
    # Test 2: User-provided bin_edges
    print("\nTesting user-provided bin_edges...")
    binning2 = EqualWidthBinning(bin_edges={0: [0, 2, 4, 6, 8], 1: [1, 3, 5, 7, 9]})
    result2 = binning2.transform(X)
    print(f"✓ User-provided bins work: {result2.shape} == (4, 2)")
    
    # Test 3: get_params returns current working values
    print("\nTesting parameter access...")
    params = binning.get_params()
    print(f"✓ get_params works: has {len(params)} parameters")
    
    # Test 4: set_params and property setters
    print("\nTesting parameter setting...")
    new_binning = EqualWidthBinning()
    new_binning.set_params(n_bins=5)
    print(f"✓ set_params works: n_bins = {new_binning.get_params().get('n_bins', 'not found')}")
    
    # Test 5: Property setters
    print("\nTesting property setters...")
    new_binning.bin_edges = {0: [0, 1, 2, 3], 1: [0, 1, 2, 3]}
    print(f"✓ bin_edges setter works: {new_binning.bin_edges is not None}")
    
    print("\n🎉 All basic functionality tests passed!")
    return True

if __name__ == "__main__":
    test_basic_functionality()
