#!/usr/bin/env python3

import warnings
import numpy as np
import sys
sys.path.insert(0, '/home/gykovacs/workspaces/binning')

from binning.base._supervised_binning_base import SupervisedBinningBase

class DummySupervisedBinning(SupervisedBinningBase):
    def __init__(self, task_type="classification", tree_params=None, **kwargs):
        super().__init__(task_type=task_type, tree_params=tree_params, **kwargs)
    
    def _fit_per_column(self, X, columns, guidance_data=None, **fit_params):
        return {col: [0.0, 1.0] for col in columns}
    
    def _calculate_bins(self, x_col, col_id, guidance_data=None):
        return [0.0, 1.0], [0.5]

# Test different types of col_id to ensure we hit line 291
obj = DummySupervisedBinning(task_type="classification")

test_cases = [
    "string_id",
    3.14,  # float
    (1, 2),  # tuple
    ["list"],  # list
    None,  # None should be handled differently
]

for i, col_id in enumerate(test_cases):
    if col_id is None:
        continue  # Skip None case as it's handled differently
    
    print(f"\nTest case {i}: col_id = {col_id} (type: {type(col_id)})")
    print(f"isinstance(col_id, (int, np.integer)): {isinstance(col_id, (int, np.integer))}")
    
    x_col = np.array([1.0, 2.0, 3.0])
    valid_mask = np.array([False, False, False])  # n_valid = 0
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = obj.handle_insufficient_data(x_col, valid_mask, min_samples=1, col_id=col_id)
        
        if w:
            print(f"Warning message: {w[0].message}")
            if "column '" in str(w[0].message):
                print("✓ Hit line 291 (else branch)")
            elif "column " in str(w[0].message):
                print("✓ Hit line 290 (if branch)")
        else:
            print("No warning generated")
