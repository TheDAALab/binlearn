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

# V2 implementations
from ._equal_width_binning_v2 import EqualWidthBinningV2
from ._singleton_binning_v2 import SingletonBinningV2
from ._chi2_binning_v2 import Chi2BinningV2
from ._equal_frequency_binning_v2 import EqualFrequencyBinningV2
from ._dbscan_binning_v2 import DBSCANBinningV2
from ._gaussian_mixture_binning_v2 import GaussianMixtureBinningV2
from ._isotonic_binning_v2 import IsotonicBinningV2
from ._kmeans_binning_v2 import KMeansBinningV2
from ._manual_interval_binning_v2 import ManualIntervalBinningV2
from ._equal_width_minimum_weight_binning_v2 import EqualWidthMinimumWeightBinningV2
from ._tree_binning_v2 import TreeBinningV2
from ._manual_flexible_binning_v2 import ManualFlexibleBinningV2

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
    # V2 implementations
    "EqualWidthBinningV2",
    "SingletonBinningV2",
    "Chi2BinningV2",
    "EqualFrequencyBinningV2",
    "DBSCANBinningV2",
    "GaussianMixtureBinningV2",
    "IsotonicBinningV2",
    "KMeansBinningV2",
    "ManualIntervalBinningV2",
    "EqualWidthMinimumWeightBinningV2",
    "TreeBinningV2",
    "ManualFlexibleBinningV2",
]
