"""Binning: A comprehensive toolkit for data discretization and binning.

This package provides a complete suite of tools for binning (discretizing) continuous
data into discrete intervals or categories. It supports multiple binning strategies,
integrates with popular data science libraries, and provides sklearn-compatible
transformers for machine learning pipelines.

Key Features:
    - Multiple binning methods: equal-width, supervised, one-hot, and flexible binning
    - Support for pandas and polars DataFrames
    - Scikit-learn compatible transformers
    - Advanced features like guidance columns and custom bin specifications
    - Comprehensive error handling and validation
    - Integration utilities for ML workflows

Main Components:
    Methods: EqualWidthBinning, SupervisedBinning, OneHotBinning
    Base Classes: GeneralBinningBase, IntervalBinningBase, FlexibleBinningBase
    Utilities: Data handling, bin operations, error management
    Integration: Feature selection, pipeline utilities, scoring functions

Example:
    >>> from binning import EqualWidthBinning
    >>> import numpy as np
    >>> X = np.random.rand(100, 3)
    >>> binner = EqualWidthBinning(n_bins=5)
    >>> X_binned = binner.fit_transform(X)
"""

# Version information
from ._version import __version__

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
from .utils.sklearn_integration import SklearnCompatibilityMixin

# Tools and integrations
from .tools import (
    BinningFeatureSelector,
    BinningPipeline,
    make_binning_scorer,
)

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
except ImportError:  # pragma: no cover
    PANDAS_AVAILABLE = False
    pd = None

try:
    from ._polars_config import POLARS_AVAILABLE, pl
except ImportError:  # pragma: no cover
    POLARS_AVAILABLE = False
    pl = None

__all__ = [
    # Version
    "__version__",
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
    # Optional dependencies
    "PANDAS_AVAILABLE",
    "pd",
    "POLARS_AVAILABLE",
    "pl",
]
