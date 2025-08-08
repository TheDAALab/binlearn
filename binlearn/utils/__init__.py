"""
Utility functions for the binning framework.

This module consolidates all utility functions used throughout the binlearn library,
organized into logical modules for better maintainability and discoverability.
"""

# Import from binning operations (interval + flexible binning)
from ._binning_operations import (
    calculate_flexible_bin_width,
    create_bin_masks,
    default_representatives,
    find_flexible_bin_for_value,
    generate_default_flexible_representatives,
    get_flexible_bin_count,
    is_missing_value,
    transform_value_to_flexible_bin,
    validate_bin_edges_format,
    validate_bin_representatives_format,
    validate_bins,
    validate_flexible_bin_spec_format,
    validate_flexible_bins,
    _validate_single_flexible_bin_def,
)

# Import data handling utilities
from ._data_handling import (
    prepare_array,
    prepare_input_with_columns,
    return_like_input,
)

# Import error classes
from ._errors import (
    BinningError,
    ConfigurationError,
    DataQualityWarning,
    FittingError,
    InvalidDataError,
    TransformationError,
    ValidationError,
    PerformanceWarning,
    BinningWarning,
    suggest_alternatives,
)

# Import type aliases and constants
from ._types import (
    # Constants
    ABOVE_RANGE,
    BELOW_RANGE,
    MISSING_VALUE,
    # Numpy array types
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
    # Calculation return types
    IntervalBinCalculationResult,
    JointParams,
    OptionalBinEdgesDict,
    OptionalBinRepsDict,
    OptionalColumnList,
    OptionalFlexibleBinSpec,
)

# Import validation utilities
from ._validation import (
    resolve_n_bins_parameter,
    resolve_string_parameter,
    validate_bin_number_for_calculation,
    validate_bin_number_parameter,
    validate_numeric_parameter,
    validate_tree_params,
)

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
    # Error classes
    "BinningError",
    "InvalidDataError",
    "ConfigurationError",
    "FittingError",
    "TransformationError",
    "ValidationError",
    "DataQualityWarning",
    "PerformanceWarning",
    "BinningWarning",
    "suggest_alternatives",
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
    "_validate_single_flexible_bin_def",
    # Data handling utilities
    "prepare_array",
    "return_like_input",
    "prepare_input_with_columns",
    # Parameter validation and conversion utilities
    "resolve_n_bins_parameter",
    "resolve_string_parameter",
    "validate_numeric_parameter",
    "validate_bin_number_parameter",
    "validate_bin_number_for_calculation",
    "validate_tree_params",
]
