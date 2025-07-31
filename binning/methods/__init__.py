"""
The module implements the binning methods.
"""

from ._equal_width_binning import EqualWidthBinning
from ._manual_interval_binning import ManualIntervalBinning
from ._manual_flexible_binning import ManualFlexibleBinning
from ._onehot_binning import OneHotBinning
from ._supervised_binning import SupervisedBinning

__all__ = [
    "EqualWidthBinning",
    "ManualIntervalBinning",
    "ManualFlexibleBinning",
    "OneHotBinning",
    "SupervisedBinning",
]
