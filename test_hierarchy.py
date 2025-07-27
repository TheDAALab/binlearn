#!/usr/bin/env python3
"""Test script to verify the simplified inheritance hierarchy works."""

import numpy as np
from binning.methods import SupervisedBinning

def test_supervised_binning_basic():
    """Test basic SupervisedBinning functionality with guidance columns."""
    print("Testing SupervisedBinning with simplified hierarchy...")
    
    # Create test data with features and target in separate columns
    X = np.array([[1.0, 0], [2.0, 1], [3.0, 0], [4.0, 1], [5.0, 0], [6.0, 1]])
    
    # Test 1: SupervisedBinning with guidance_columns specified
    try:
        binner = SupervisedBinning(guidance_columns=[1])  # Last column is target
        binner.fit(X)
        X_binned = binner.transform(X)
        print(f"✓ SupervisedBinning with guidance_columns works: shape {X_binned.shape}")
        return True
        
    except Exception as e:
        print(f"✗ SupervisedBinning failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_supervised_binning_basic()
    if success:
        print("\n✓ Simplified inheritance hierarchy is working!")
    else:
        print("\n✗ There are still issues with the hierarchy.")
