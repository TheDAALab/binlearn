"""
This module implements some utility functions.
"""

import pandas as pd

__all__ = ['pandizator',]

def pandizator(func):
    def inner(*args, **kwargs):
        return func(*[arg.values if isinstance(arg, pd.Series) else arg for arg in args],
                    **{k: v.values if isinstance(v, pd.Series) else v for k,v in kwargs.items()})
    return inner
