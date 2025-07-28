#!/usr/bin/env python3
import numpy as np
import warnings
import sys
sys.path.insert(0, '/home/gykovacs/workspaces/binning')

from binning.base._supervised_binning_base import SupervisedBinningBase
from binning.utils.errors import DataQualityWarning

class DummySupervisedBinning(SupervisedBinningBase):
    def __init__(self, task_type='classification', tree_params=None, **kwargs):
        super().__init__(task_type=task_type, tree_params=tree_params, **kwargs)
    
    def _fit_per_column(self, X, columns, guidance_data=None, **fit_params):
        pass

def test_line_291():
    print("Testing line 291 coverage...")
    obj = DummySupervisedBinning(task_type='classification')
    
    # Test with no valid data and non-integer col_id
    x_col = np.array([np.nan, np.nan])
    valid_mask = np.isfinite(x_col)  # All False for NaN data
    
    print(f"x_col: {x_col}")
    print(f"valid_mask: {valid_mask}")
    print(f"n_valid: {valid_mask.sum()}")
    print(f"all nan: {np.isnan(x_col).all()}")
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Use a float to guarantee we hit the else branch (line 291)
        result = obj.handle_insufficient_data(x_col, valid_mask, min_samples=1, col_id=3.14)
        
        print(f"Result: {result}")
        print(f"Warnings: {len(w)}")
        if w:
            print(f"Warning message: {w[0].message}")
            print(f"Col ID type: {type(3.14)}")
            print(f"Is int: {isinstance(3.14, (int, np.integer))}")

if __name__ == "__main__":
    test_line_291()
