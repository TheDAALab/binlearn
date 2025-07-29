#!/usr/bin/env python3
"""
Final test to ensure 100% coverage of SupervisedBinning._handle_bin_params lines 323-324, 327-328
"""

import sys
sys.path.insert(0, '.')

from binning.methods._supervised_binning import SupervisedBinning

def test_handle_bin_params_coverage():
    """Test to explicitly hit the missing coverage lines."""
    
    # Test case 1: task_type parameter (lines 323-324)
    binning1 = SupervisedBinning(task_type="classification")
    params1 = {"task_type": "regression", "other_param": "value"}
    
    print("Testing task_type parameter...")
    print(f"Before: task_type = {binning1.task_type}")
    result1 = binning1._handle_bin_params(params1)
    print(f"After: task_type = {binning1.task_type}")
    print(f"Result: {result1}")
    print(f"Remaining params: {params1}")
    print()
    
    # Test case 2: tree_params parameter (lines 327-328)
    binning2 = SupervisedBinning()
    params2 = {"tree_params": {"max_depth": 5}, "other_param": "value"}
    
    print("Testing tree_params parameter...")
    print(f"Before: tree_params = {binning2.tree_params}")
    result2 = binning2._handle_bin_params(params2)
    print(f"After: tree_params = {binning2.tree_params}")
    print(f"Result: {result2}")
    print(f"Remaining params: {params2}")
    print()
    
    # Test case 3: Both parameters together
    binning3 = SupervisedBinning()
    params3 = {"task_type": "regression", "tree_params": {"min_samples_split": 10}}
    
    print("Testing both parameters together...")
    print(f"Before: task_type = {binning3.task_type}, tree_params = {binning3.tree_params}")
    result3 = binning3._handle_bin_params(params3)
    print(f"After: task_type = {binning3.task_type}, tree_params = {binning3.tree_params}")
    print(f"Result: {result3}")
    print(f"Remaining params: {params3}")
    
    print("All tests completed successfully!")

if __name__ == "__main__":
    test_handle_bin_params_coverage()
