"""
The module brings together the foundations.
"""

# Core types and constants
from ._constants import MISSING_VALUE, ABOVE_RANGE, BELOW_RANGE
from ._types import BinEdges, IntervalBinSpec, BinRepsType

# Utility functions
from ._data_utils import (
    prepare_array,
    return_like_input,
    prepare_input_with_columns,
    is_pandas_df,
    is_polars_df,
)
from ._bin_utils import (
    # Interval binning utilities
    ensure_bin_dict,
    validate_bins,
    default_representatives,
    create_bin_masks,
    # Flexible binning utilities
    FlexibleBinSpec,
    FlexibleBinReps,
    ensure_flexible_bin_spec,
    generate_default_flexible_representatives,
    validate_flexible_bins,
    is_missing_value,
    find_flexible_bin_for_value,
    calculate_flexible_bin_width,
    transform_value_to_flexible_bin,
    get_flexible_bin_count,
    validate_single_flexible_bin_def,
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
    "MISSING_VALUE",
    "ABOVE_RANGE",
    "BELOW_RANGE",
    # Types
    "BinEdges",
    "IntervalBinSpec",
    "FlexibleBinSpec",
    "BinRepsType",
    # Utility functions
    "prepare_array",
    "return_like_input",
    "prepare_input_with_columns",
    "is_pandas_df",
    "is_polars_df",
    # Interval binning utilities
    "ensure_bin_dict",
    "validate_bins",
    "default_representatives",
    "create_bin_masks",
    # Flexible binning utilities
    "FlexibleBinSpec",
    "FlexibleBinReps",
    "ensure_flexible_bin_spec",
    "generate_default_flexible_representatives",
    "validate_flexible_bins",
    "is_missing_value",
    "find_flexible_bin_for_value",
    "calculate_flexible_bin_width",
    "transform_value_to_flexible_bin",
    "get_flexible_bin_count",
    "validate_single_flexible_bin_def",
    # Base classes
    "GeneralBinningBase",
    "IntervalBinningBase",
    "FlexibleBinningBase",
    "SupervisedBinningBase",
    # Legacy mixins
    "GuidedBinningMixin",
]
