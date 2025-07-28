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

# Test the exact conditions for line 291
obj = DummySupervisedBinning(task_type="classification")

# Create test data
x_col = np.array([1.0, 2.0, 3.0])
valid_mask = np.array([False, False, False])  # n_valid = 0
min_samples = 1
col_id = "test_string"

print("Before calling handle_insufficient_data...")
print(f"x_col: {x_col}")
print(f"valid_mask: {valid_mask}")
print(f"n_valid: {valid_mask.sum()}")
print(f"col_id: {col_id}")
print(f"isinstance(col_id, (int, np.integer)): {isinstance(col_id, (int, np.integer))}")

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = obj.handle_insufficient_data(x_col, valid_mask, min_samples, col_id)
    
    print(f"Result: {result}")
    print(f"Warnings captured: {len(w)}")
    if w:
        print(f"Warning message: {w[0].message}")
