"""
The module implements the binning methods.
"""

from ._equal_frequency_binning import EqualFrequencyBinning
from ._equal_width_binning import EqualWidthBinning
from ._equal_width_minimum_weight_binning import EqualWidthMinimumWeightBinning
from ._kmeans_binning import KMeansBinning
from ._manual_flexible_binning import ManualFlexibleBinning
from ._manual_interval_binning import ManualIntervalBinning
from ._singleton_binning import SingletonBinning
from ._supervised_binning import SupervisedBinning

__all__ = [
    "EqualWidthBinning",
    "EqualFrequencyBinning",
    "KMeansBinning",
    "EqualWidthMinimumWeightBinning",
    "ManualIntervalBinning",
    "ManualFlexibleBinning",
    "SingletonBinning",
    "SupervisedBinning",
]
