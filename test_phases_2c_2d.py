#!/usr/bin/env python3
"""
Test script for Run 2 phases 2C-2D: Abstract methods and sklearn compatibility.
"""

import numpy as np
from abc import ABC
from sklearn.base import clone

# Test imports
print("Testing imports...")
from binning.base._general_binning_base import GeneralBinningBase
from binning.base._interval_binning_base import IntervalBinningBase  
from binning.base._flexible_binning_base import FlexibleBinningBase
from binning.methods._equal_width_binning import EqualWidthBinning
from binning.methods._supervised_binning import SupervisedBinning
print("✓ All imports successful")

def test_abstract_base_classes():
    """Test that base classes are properly abstract."""
    print("\nTesting abstract base classes...")
    
    # Check ABC inheritance
    assert issubclass(GeneralBinningBase, ABC), "GeneralBinningBase should inherit from ABC"
    print("✓ GeneralBinningBase inherits from ABC")
    
    # Try to instantiate abstract classes (should fail)
    try:
        GeneralBinningBase()
        print("✗ GeneralBinningBase instantiation should fail (abstract)")
        return False
    except TypeError as e:
        if "abstract" in str(e):
            print("✓ GeneralBinningBase properly abstract")
        else:
            print(f"✗ Unexpected error: {e}")
            return False
    
    try:
        IntervalBinningBase()
        print("✗ IntervalBinningBase instantiation should fail (abstract)")
        return False
    except TypeError as e:
        if "abstract" in str(e):
            print("✓ IntervalBinningBase properly abstract")
        else:
            print(f"✗ Unexpected error: {e}")
            return False
    
    try:
        FlexibleBinningBase()
        print("✗ FlexibleBinningBase instantiation should fail (abstract)")
        return False
    except TypeError as e:
        if "abstract" in str(e):
            print("✓ FlexibleBinningBase properly abstract")
        else:
            print(f"✗ Unexpected error: {e}")
            return False
    
    return True

def test_concrete_classes():
    """Test that concrete classes can be instantiated."""
    print("\nTesting concrete classes...")
    
    # These should work
    try:
        binner1 = EqualWidthBinning(n_bins=5)
        print("✓ EqualWidthBinning instantiation works")
    except Exception as e:
        print(f"✗ EqualWidthBinning failed: {e}")
        return False
    
    try:
        binner2 = SupervisedBinning(task_type="classification")
        print("✓ SupervisedBinning instantiation works")
    except Exception as e:
        print(f"✗ SupervisedBinning failed: {e}")
        return False
    
    return True

def test_parameter_validation():
    """Test parameter validation for sklearn compatibility."""
    print("\nTesting parameter validation...")
    
    # For now, just test that _validate_params method exists and can be called
    try:
        binner = EqualWidthBinning()
        binner._validate_params()  # Should not raise an error for valid params
        print("✓ _validate_params method works for valid parameters")
    except Exception as e:
        print(f"✗ _validate_params failed unexpectedly: {e}")
        return False
    
    # Test validation exists in base class
    try:
        from binning.base._general_binning_base import GeneralBinningBase
        # Check the method exists
        assert hasattr(GeneralBinningBase, '_validate_params')
        print("✓ _validate_params method exists in base class")
    except Exception as e:
        print(f"✗ _validate_params method check failed: {e}")
        return False
        
    # Test that validation is called during fit
    try:
        binner = EqualWidthBinning()
        X = [[1, 2], [3, 4]]
        
        # Monkey patch _validate_params to track if it's called
        called = []
        original_validate = binner._validate_params
        def tracking_validate():
            called.append(True)
            return original_validate()
        binner._validate_params = tracking_validate
        
        binner.fit(X)
        
        if called:
            print("✓ _validate_params is called during fit")
        else:
            print("✗ _validate_params is not called during fit")
            return False
    except Exception as e:
        print(f"✗ Validation tracking test failed: {e}")
        return False
    
    return True

def test_sklearn_clone_compatibility():
    """Test sklearn clone compatibility."""
    print("\nTesting sklearn clone compatibility...")
    
    # Test EqualWidthBinning clone
    try:
        original = EqualWidthBinning(n_bins=5, preserve_dataframe=True)
        cloned = clone(original)
        
        assert cloned.n_bins == 5
        assert cloned.preserve_dataframe == True
        assert cloned is not original
        print("✓ EqualWidthBinning clone works")
    except Exception as e:
        print(f"✗ EqualWidthBinning clone failed: {e}")
        return False
    
    # Test SupervisedBinning clone
    try:
        original = SupervisedBinning(task_type="regression", guidance_columns=[2])
        cloned = clone(original)
        
        assert cloned.task_type == "regression"
        assert cloned.guidance_columns == [2]
        assert cloned is not original
        print("✓ SupervisedBinning clone works")
    except Exception as e:
        print(f"✗ SupervisedBinning clone failed: {e}")
        return False
    
    return True

def test_functional_compatibility():
    """Test that the enhanced classes still work functionally."""
    print("\nTesting functional compatibility...")
    
    # Test basic binning functionality
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    
    try:
        binner = EqualWidthBinning(n_bins=2)
        binner.fit(X)
        result = binner.transform(X)
        assert result.shape == (4, 2)
        print("✓ EqualWidthBinning functionality works")
    except Exception as e:
        print(f"✗ EqualWidthBinning functionality failed: {e}")
        return False
    
    try:
        X_with_target = np.column_stack([X, [0, 1, 0, 1]])
        binner = SupervisedBinning(guidance_columns=[2])
        binner.fit(X_with_target)
        result = binner.transform(X_with_target)
        assert result.shape[1] == 2  # Only binning columns
        print("✓ SupervisedBinning functionality works")
    except Exception as e:
        print(f"✗ SupervisedBinning functionality failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=================================================================")
    print("TESTING RUN 2 PHASES 2C-2D: Abstract Methods & Sklearn Compatibility")
    print("=================================================================")
    
    all_passed = True
    
    all_passed &= test_abstract_base_classes()
    all_passed &= test_concrete_classes()
    all_passed &= test_parameter_validation()
    all_passed &= test_sklearn_clone_compatibility()
    all_passed &= test_functional_compatibility()
    
    print("\n=================================================================")
    if all_passed:
        print("🎉 ALL TESTS PASSED: Phases 2C-2D completed successfully!")
        print("✓ Abstract method enforcement working")
        print("✓ Parameter validation enhanced")
        print("✓ Sklearn compatibility improved")
        print("✓ Functional compatibility maintained")
    else:
        print("❌ SOME TESTS FAILED")
    print("=================================================================")
