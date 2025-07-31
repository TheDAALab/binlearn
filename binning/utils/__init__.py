"""
Utility functions for the binning framework.

This module consolidates all utility functions used throughout the binning package,
organized into logical submodules for better maintainability and discoverability.
"""

# Import constants
from .constants import MISSING_VALUE, ABOVE_RANGE, BELOW_RANGE

# Import type aliases for re-export
from .types import (
    # Column and data types
    ColumnId,
    ColumnList,
    OptionalColumnList,
    GuidanceColumns,
    ArrayLike,
    # Interval binning types
    BinEdges,
    BinEdgesDict,
    BinReps,
    BinRepsDict,
    OptionalBinEdgesDict,
    OptionalBinRepsDict,
    # Flexible binning types
    FlexibleBinDef,
    FlexibleBinDefs,
    FlexibleBinSpec,
    OptionalFlexibleBinSpec,
    # Calculation return types
    IntervalBinCalculationResult,
    FlexibleBinCalculationResult,
    # Count and validation types
    BinCountDict,
    # Numpy array types
    Array1D,
    Array2D,
    BooleanMask,
    # Parameter types
    FitParams,
    JointParams,
)

# Import error classes
from .errors import (
    BinningError,
    InvalidDataError,
    ConfigurationError,
    FittingError,
    TransformationError,
    ValidationError,
    ValidationMixin,
    DataQualityWarning,
)

# Import sklearn integration utilities
from .sklearn_integration import SklearnCompatibilityMixin

# Import utility functions
from .bin_operations import (
    validate_bin_edges_format,
    validate_bin_representatives_format,
    validate_bins,
    default_representatives,
    create_bin_masks,
)

from .flexible_bin_operations import (
    generate_default_flexible_representatives,
    validate_flexible_bins,
    validate_flexible_bin_spec_format,
    is_missing_value,
    find_flexible_bin_for_value,
    calculate_flexible_bin_width,
    transform_value_to_flexible_bin,
    get_flexible_bin_count,
)

from .data_handling import (
    prepare_array,
    return_like_input,
    prepare_input_with_columns,
)

from .inspection import (
    get_class_parameters,
    get_constructor_info,
    safe_get_class_parameters,
    safe_get_constructor_info,
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
