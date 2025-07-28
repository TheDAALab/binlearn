"""
This module brings together all tools.
"""

# Configuration management
from .config import get_config, set_config, load_config, reset_config

# Error handling
from .utils.errors import (
    BinningError,
    InvalidDataError,
    ConfigurationError,
    FittingError,
    TransformationError,
    ValidationError,
)

# Sklearn utilities
from .utils.sklearn_integration import BinningFeatureSelector, BinningPipeline

# Base classes and utilities
from .base import (
    # Constants
    MISSING_VALUE,
    ABOVE_RANGE,
    BELOW_RANGE,
    # Type aliases
    ColumnId,
    ColumnList,
    OptionalColumnList,
    GuidanceColumns,
    ArrayLike,
    BinEdges,
    BinEdgesDict,
    BinReps,
    BinRepsDict,
    OptionalBinEdgesDict,
    OptionalBinRepsDict,
    FlexibleBinDef,
    FlexibleBinDefs,
    FlexibleBinSpec,
    OptionalFlexibleBinSpec,
    IntervalBinCalculationResult,
    FlexibleBinCalculationResult,
    BinCountDict,
    Array1D,
    Array2D,
    BooleanMask,
    FitParams,
    JointParams,
    # Base classes
    GeneralBinningBase,
    IntervalBinningBase,
    FlexibleBinningBase,
    SupervisedBinningBase,
    # Utility functions
    prepare_array,
    return_like_input,
    prepare_input_with_columns,
    ensure_bin_dict,
    validate_bins,
    default_representatives,
    create_bin_masks,
)

# Concrete binning methods
from .methods import EqualWidthBinning, OneHotBinning, SupervisedBinning

# Optional pandas/polars configurations (if available)
try:
    from ._pandas_config import PANDAS_AVAILABLE, pd
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    from ._polars_config import POLARS_AVAILABLE, pl
except ImportError:
    POLARS_AVAILABLE = False
    pl = None

__all__ = [
    # Configuration
    "get_config",
    "set_config",
    "load_config",
    "reset_config",
    # Errors
    "BinningError",
    "InvalidDataError",
    "ConfigurationError",
    "FittingError",
    "TransformationError",
    "ValidationError",
    # Sklearn utilities
    "BinningFeatureSelector",
    "BinningPipeline",
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
    # Base classes
    "GeneralBinningBase",
    "IntervalBinningBase",
    "FlexibleBinningBase",
    "SupervisedBinningBase",
    # Utility functions
    "prepare_array",
    "return_like_input",
    "prepare_input_with_columns",
    "ensure_bin_dict",
    "validate_bins",
    "default_representatives",
    "create_bin_masks",
    # Concrete methods
    "EqualWidthBinning",
    "OneHotBinning",
    "SupervisedBinning",
]
