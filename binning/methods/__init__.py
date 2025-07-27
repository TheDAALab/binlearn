"""
The module implements the binning methods.
"""

from ._equal_width_binning import EqualWidthBinning
from ._onehot_binning import OneHotBinning  
from ._supervised_binning import SupervisedBinning

__all__ = [
    'EqualWidthBinning',
    'OneHotBinning', 
    'SupervisedBinning'
]
