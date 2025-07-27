"""
The module brings together the foundations.
"""

# Core types and constants
from ._constants import MISSING_VALUE, ABOVE_RANGE, BELOW_RANGE
from ._types import BinEdges, IntervalBinSpec, FlexibleBinSpec, BinRepsType

# Utility functions
from ._data_utils import (
    prepare_array, 
    return_like_input, 
    prepare_input_with_columns,
    is_pandas_df,
    is_polars_df
)
from ._bin_utils import (
    ensure_bin_dict,
    validate_bins, 
    default_representatives,
    create_bin_masks
)

# Base classes
from ._general_binning_base import GeneralBinningBase
from ._interval_binning_base import IntervalBinningBase
from ._flexible_binning_base import FlexibleBinningBase
from ._supervised_binning_base import SupervisedBinningBase

# Mixins (legacy - kept for backward compatibility)
from ._guided_binning_mixin import GuidedBinningMixin

__all__ = [
    # Constants
    'MISSING_VALUE', 'ABOVE_RANGE', 'BELOW_RANGE',
    
    # Types
    'BinEdges', 'IntervalBinSpec', 'FlexibleBinSpec', 'BinRepsType',
    
    # Utility functions
    'prepare_array', 'return_like_input', 'prepare_input_with_columns', 
    'is_pandas_df', 'is_polars_df',
    'ensure_bin_dict', 'validate_bins', 'default_representatives', 'create_bin_masks',
    
    # Base classes
    'GeneralBinningBase', 'IntervalBinningBase', 'FlexibleBinningBase', 'SupervisedBinningBase',
    
    # Legacy mixins
    'GuidedBinningMixin'
]
