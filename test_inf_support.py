#!/usr/bin/env python3
"""Test script to verify infinite value support across all binning methods."""

import numpy as np
from binning.methods import EqualWidthBinning, OneHotBinning, SupervisedBinning

def test_equal_width_inf():
    """Test EqualWidthBinning with infinite values."""
    print("Testing EqualWidthBinning with infinite values...")
    
    # Test with some finite values and inf
    X = np.array([[1.0], [2.0], [np.inf], [3.0], [-np.inf], [4.0]])
    binner = EqualWidthBinning(n_bins=3)
    
    try:
        binner.fit(X)
        X_binned = binner.transform(X)
        print(f"✓ EqualWidthBinning handled inf values: {X_binned.flatten()}")
        
        # Test with only infinite values
        X_inf_only = np.array([[np.inf], [-np.inf], [np.inf]])
        binner_inf = EqualWidthBinning()
        binner_inf.fit(X_inf_only)
        X_inf_binned = binner_inf.transform(X_inf_only)
        print(f"✓ EqualWidthBinning handled inf-only data gracefully: {X_inf_binned.flatten()}")
            
    except Exception as e:
        print(f"✗ EqualWidthBinning failed with inf values: {e}")

def test_onehot_inf():
    """Test OneHotBinning with infinite values."""
    print("\nTesting OneHotBinning with infinite values...")
    
    # Test with some finite values and inf
    X = np.array([[1.0], [2.0], [np.inf], [3.0], [-np.inf], [1.0]])
    binner = OneHotBinning()
    
    try:
        binner.fit(X)
        X_binned = binner.transform(X)
        print(f"✓ OneHotBinning handled inf values: shape {X_binned.shape}")
        
        # Test with only infinite values
        X_inf_only = np.array([[np.inf], [-np.inf], [np.inf]])
        binner_inf = OneHotBinning()
        binner_inf.fit(X_inf_only)
        X_inf_binned = binner_inf.transform(X_inf_only)
        print(f"✓ OneHotBinning handled inf-only data: shape {X_inf_binned.shape}")
        
    except Exception as e:
        print(f"✗ OneHotBinning failed with inf values: {e}")

def test_supervised_inf():
    """Test SupervisedBinning with infinite values."""
    print("\nTesting SupervisedBinning with infinite values...")
    
    # Test with some finite values and inf
    X = np.array([[1.0, 0], [2.0, 1], [np.inf, 0], [3.0, 1], [-np.inf, 0], [4.0, 1]])
    binner = SupervisedBinning(guidance_columns=[1])
    
    try:
        binner.fit(X)
        X_binned = binner.transform(X)
        print(f"✓ SupervisedBinning handled inf values: {X_binned.flatten()}")
        
        # Test with only infinite values
        X_inf_only = np.array([[np.inf, 0], [-np.inf, 1], [np.inf, 0]])
        binner_inf = SupervisedBinning(guidance_columns=[1])
        binner_inf.fit(X_inf_only)
        X_inf_binned = binner_inf.transform(X_inf_only)
        print(f"✓ SupervisedBinning handled inf-only data: {X_inf_binned.flatten()}")
        
    except Exception as e:
        print(f"✗ SupervisedBinning failed with inf values: {e}")

if __name__ == "__main__":
    test_equal_width_inf()
    test_onehot_inf()
    test_supervised_inf()
