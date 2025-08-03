"""
Utility functions for the binning framework.

This module consolidates all utility functions used throughout the binlearn library,
organized into logical submodules for better maintainability and discoverability.
"""

# Import constants
# Import utility functions
from .bin_operations import (
    create_bin_masks,
    default_representatives,
    validate_bin_edges_format,
    validate_bin_representatives_format,
    validate_bins,
)
from .constants import ABOVE_RANGE, BELOW_RANGE, MISSING_VALUE
from .data_handling import (
    prepare_array,
    prepare_input_with_columns,
    return_like_input,
)

# Import error classes
from .errors import (
    BinningError,
    ConfigurationError,
    DataQualityWarning,
    FittingError,
    InvalidDataError,
    TransformationError,
    ValidationError,
    ValidationMixin,
)
from .flexible_bin_operations import (
    calculate_flexible_bin_width,
    find_flexible_bin_for_value,
    generate_default_flexible_representatives,
    get_flexible_bin_count,
    is_missing_value,
    transform_value_to_flexible_bin,
    validate_flexible_bin_spec_format,
    validate_flexible_bins,
)
from .inspection import (
    get_class_parameters,
    get_constructor_info,
    safe_get_class_parameters,
    safe_get_constructor_info,
)

# Import sklearn integration utilities
from .sklearn_integration import SklearnCompatibilityMixin

# Import type aliases for re-export
from .types import (
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
    "ValidationMixin",
    "DataQualityWarning",
    # Sklearn integration
    "SklearnCompatibilityMixin",
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
    # Data handling utilities
    "prepare_array",
    "return_like_input",
    "prepare_input_with_columns",
    # Inspection utilities
    "get_class_parameters",
    "get_constructor_info",
    "safe_get_class_parameters",
    "safe_get_constructor_info",
]
