"""
This module implements some utility functions.
"""

import pandas as pd

__all__ = ['pandizator_decorator_in',
           'pandizator_decorator_inout']
def pandizator_decorator_in(func):
    def inner(*args, **kwargs):
        if isinstance(args[1], pd.Series):
            tmp = args[1].values
            args = (args[0], tmp, *args[2:])
        return func(*args, **kwargs)
    return inner

def pandizator_decorator_inout(func):
    def inner(*args, **kwargs):
        if isinstance(args[1], pd.Series):
            series_name = args[1].name
            tmp = args[1].values
            args = (args[0], tmp, *args[2:])
        return pd.Series(func(*args, **kwargs), name=f'{series_name}_binned')
    return inner