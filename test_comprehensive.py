#!/usr/bin/env python3
"""Comprehensive test of the simplified inheritance hierarchy."""

import numpy as np
from binning.methods import EqualWidthBinning, OneHotBinning, SupervisedBinning

def test_all_binning_methods():
    """Test all binning methods with the simplified hierarchy."""
    print("Testing all binning methods with simplified inheritance hierarchy...\n")
    
    # Test data
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]])
    
    success_count = 0
    total_tests = 3
    
    # Test 1: EqualWidthBinning (IntervalBinningBase)
    try:
        print("1. Testing EqualWidthBinning...")
        binner = EqualWidthBinning(n_bins=3)
        binner.fit(X)
        X_binned = binner.transform(X)
        print(f"   ✓ EqualWidthBinning works: shape {X_binned.shape}")
        success_count += 1
    except Exception as e:
        print(f"   ✗ EqualWidthBinning failed: {e}")
    
    # Test 2: OneHotBinning (FlexibleBinningBase) 
    try:
        print("2. Testing OneHotBinning...")
        binner = OneHotBinning()
        binner.fit(X)
        X_binned = binner.transform(X)
        print(f"   ✓ OneHotBinning works: shape {X_binned.shape}")
        success_count += 1
    except Exception as e:
        print(f"   ✗ OneHotBinning failed: {e}")
    
    # Test 3: SupervisedBinning (SupervisedBinningBase)
    try:
        print("3. Testing SupervisedBinning...")
        # Create data with target column
        X_with_target = np.array([[1.0, 0], [2.0, 1], [3.0, 0], [4.0, 1], [5.0, 0], [6.0, 1]])
        binner = SupervisedBinning(guidance_columns=[1])
        binner.fit(X_with_target)
        X_binned = binner.transform(X_with_target)
        print(f"   ✓ SupervisedBinning works: shape {X_binned.shape}")
        success_count += 1
    except Exception as e:
        print(f"   ✗ SupervisedBinning failed: {e}")
    
    print(f"\nSummary: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All binning methods work with the simplified hierarchy!")
        return True
    else:
        print("❌ Some binning methods still have issues.")
        return False

def test_inheritance_structure():
    """Test that the inheritance structure is as expected."""
    print("\nTesting inheritance structure...")
    
    # Check inheritance hierarchy
    print("Checking inheritance hierarchy:")
    print(f"EqualWidthBinning MRO: {[cls.__name__ for cls in EqualWidthBinning.__mro__]}")
    print(f"OneHotBinning MRO: {[cls.__name__ for cls in OneHotBinning.__mro__]}")
    print(f"SupervisedBinning MRO: {[cls.__name__ for cls in SupervisedBinning.__mro__]}")
    
    # Verify no multiple inheritance issues
    from binning.base import SupervisedBinningBase
    
    # Check that SupervisedBinning only inherits from SupervisedBinningBase
    direct_bases = SupervisedBinning.__bases__
    print(f"\nSupervisedBinning direct bases: {[cls.__name__ for cls in direct_bases]}")
    
    if len(direct_bases) == 1 and direct_bases[0] == SupervisedBinningBase:
        print("✓ SupervisedBinning has single inheritance (from SupervisedBinningBase)")
        return True
    else:
        print("✗ SupervisedBinning still has multiple inheritance")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("COMPREHENSIVE TEST: Simplified Inheritance Hierarchy")
    print("=" * 60)
    
    functionality_ok = test_all_binning_methods()
    structure_ok = test_inheritance_structure()
    
    print(f"\n{'=' * 60}")
    if functionality_ok and structure_ok:
        print("🎉 SUCCESS: Inheritance hierarchy simplification is complete!")
        print("✓ All binning methods work correctly")
        print("✓ Single inheritance structure achieved")
    else:
        print("❌ ISSUES REMAIN: More work needed on hierarchy")
    print("=" * 60)
