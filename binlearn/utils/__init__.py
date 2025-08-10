"""
Utility functions for the binning framework.

This module consolidates all utility functions used throughout the binlearn library,
organized into logical modules for better maintainability and discoverability.
"""

# Import from binning operations (interval + flexible binning)
from ._binning_operations import (
    _validate_single_flexible_bin_def,
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
)

# Import data handling utilities
from ._data_handling import (
    convert_to_python_types,
    prepare_array,
    prepare_input_with_columns,
    return_like_input,
)

# Import new utility modules
from ._parameter_validation import (
    validate_common_parameters,
    validate_range_parameter,
    validate_positive_number,
    validate_positive_integer,
    validate_random_state,
    create_configuration_error,
    COMMON_PARAM_SPECS,
)

from ._configuration_utils import (
    standardize_init_pattern,
    create_param_dict_for_config,
    get_effective_n_bins,
    prepare_sklearn_estimator_params,
    handle_common_warnings,
)

from ._equal_width_utils import (
    create_equal_width_bins,
    apply_equal_width_fallback,
    validate_binning_input,
    ensure_monotonic_edges,
)

from ._error_handling import (
    handle_sklearn_import_error,
    handle_insufficient_data_error,
    handle_convergence_warning,
    handle_parameter_bounds_error,
    safe_sklearn_call,
    validate_fitted_state,
)

# Import error classes
from ._errors import (
    BinningError,
    BinningWarning,
    ConfigurationError,
    DataQualityWarning,
    FittingError,
    InvalidDataError,
    PerformanceWarning,
    TransformationError,
    ValidationError,
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
    "convert_to_python_types",
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
    # New refactored utilities
    "validate_common_parameters",
    "validate_range_parameter",
    "validate_positive_number",
    "validate_positive_integer",
    "create_configuration_error",
    "COMMON_PARAM_SPECS",
    "standardize_init_pattern",
    "create_param_dict_for_config",
    "get_effective_n_bins",
    "prepare_sklearn_estimator_params",
    "handle_common_warnings",
    "create_equal_width_bins",
    "apply_equal_width_fallback",
    "validate_binning_input",
    "ensure_monotonic_edges",
    "handle_sklearn_import_error",
    "handle_insufficient_data_error",
    "handle_convergence_warning",
    "handle_parameter_bounds_error",
    "safe_sklearn_call",
    "validate_fitted_state",
]
