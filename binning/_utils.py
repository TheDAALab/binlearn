"""
This module implements some utility functions.
"""

import numpy as np
import pandas as pd

__all__ = ['homogenize_input',]

def homogenize_input(func):
    def homogenize_input_core(input):
        if isinstance(input, pd.Series):
            return input.values
        elif isinstance(input, list):
            return np.array(input)
        elif isinstance(input, np.ndarray):
            if (len(input.shape) > 2) or ((len(input.shape) == 2) and (input.shape[1] != 1)):
                raise ValueError('Invalid np.ndarray input, must be of shape (N,) or (N,1)')
            return input
        else:
            raise ValueError('Invalid input type - must be pd.Series, list, or 1D-like np.ndarray')

    def inner(*args, **kwargs):
        return func(*[homogenize_input_core(arg) for arg in args],
                    **{k: homogenize_input_core(v) for k,v in kwargs.items()})
    return inner
