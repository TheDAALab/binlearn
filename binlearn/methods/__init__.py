"""
The module implements the binning methods.
"""

from ._chi2_binning import Chi2Binning
from ._dbscan_binning import DBSCANBinning
from ._equal_frequency_binning import EqualFrequencyBinning
from ._equal_width_binning import EqualWidthBinning
from ._equal_width_minimum_weight_binning import EqualWidthMinimumWeightBinning
from ._gaussian_mixture_binning import GaussianMixtureBinning
from ._isotonic_binning import IsotonicBinning
from ._kmeans_binning import KMeansBinning
from ._manual_flexible_binning import ManualFlexibleBinning
from ._manual_interval_binning import ManualIntervalBinning
from ._singleton_binning import SingletonBinning
from ._supervised_binning import SupervisedBinning

__all__ = [
    "Chi2Binning",
    "DBSCANBinning",
    "EqualWidthBinning",
    "EqualFrequencyBinning",
    "GaussianMixtureBinning",
    "IsotonicBinning",
    "KMeansBinning",
    "EqualWidthMinimumWeightBinning",
    "ManualIntervalBinning",
    "ManualFlexibleBinning",
    "SingletonBinning",
    "SupervisedBinning",
]
