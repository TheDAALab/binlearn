"""
The module brings together the foundations.
"""

# Core types and constants
from ..utils.bin_operations import (
    create_bin_masks,
    default_representatives,
    validate_bin_edges_format,
    validate_bin_representatives_format,
    validate_bins,
)
from ..utils.constants import ABOVE_RANGE, BELOW_RANGE, MISSING_VALUE

# Utility functions
from ..utils.data_handling import (
    prepare_array,
    prepare_input_with_columns,
    return_like_input,
)
from ..utils.flexible_bin_operations import (
    calculate_flexible_bin_width,
    find_flexible_bin_for_value,
    generate_default_flexible_representatives,
    get_flexible_bin_count,
    is_missing_value,
    transform_value_to_flexible_bin,
    validate_flexible_bin_spec_format,
    validate_flexible_bins,
)

# Type aliases
from ..utils.types import (
    # Array types
    Array1D,
    Array2D,
    ArrayLike,
    # Count and validation types
    BinCountDict,
    # Interval binning types
    BinEdges,
    BinEdgesDict,
    BinReps,
    BinRepsDict,
    BooleanMask,
    # Column and data types
    ColumnId,
    ColumnList,
    # Parameter types
    FitParams,
    FlexibleBinCalculationResult,
    # Flexible binning types
    FlexibleBinDef,
    FlexibleBinDefs,
    FlexibleBinSpec,
    GuidanceColumns,
    # Return types
    IntervalBinCalculationResult,
    JointParams,
    OptionalBinEdgesDict,
    OptionalBinRepsDict,
    OptionalColumnList,
    OptionalFlexibleBinSpec,
)
from ._flexible_binning_base import FlexibleBinningBase

# Base classes
from ._general_binning_base import GeneralBinningBase
from ._interval_binning_base import IntervalBinningBase
from ._supervised_binning_base import SupervisedBinningBase

__all__ = [
    # Constants
    "MISSING_VALUE",
    "ABOVE_RANGE",
    "BELOW_RANGE",
    # Type aliases
    "ColumnId",
    "ColumnList",
    "OptionalColumnList",
    "GuidanceColumns",
    "ArrayLike",
    "BinEdges",
    "BinEdgesDict",
    "BinReps",
    "BinRepsDict",
    "OptionalBinEdgesDict",
    "OptionalBinRepsDict",
    "FlexibleBinDef",
    "FlexibleBinDefs",
    "FlexibleBinSpec",
    "OptionalFlexibleBinSpec",
    "IntervalBinCalculationResult",
    "FlexibleBinCalculationResult",
    "BinCountDict",
    "Array1D",
    "Array2D",
    "BooleanMask",
    "FitParams",
    "JointParams",
    # Utility functions
    "prepare_array",
    "return_like_input",
    "prepare_input_with_columns",
    # Interval binning utilities
    "validate_bin_edges_format",
    "validate_bin_representatives_format",
    "validate_bins",
    "default_representatives",
    "create_bin_masks",
    # Flexible binning utilities
    "generate_default_flexible_representatives",
    "validate_flexible_bins",
    "validate_flexible_bin_spec_format",
    "is_missing_value",
    "find_flexible_bin_for_value",
    "calculate_flexible_bin_width",
    "transform_value_to_flexible_bin",
    "get_flexible_bin_count",
    # Base classes
    "GeneralBinningBase",
    "IntervalBinningBase",
    "FlexibleBinningBase",
    "SupervisedBinningBase",
]
