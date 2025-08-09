"""Binning: A comprehensive toolkit for data discretization and binning.

This package provides a complete suite of tools for binning (discretizing) continuous
data into discrete intervals or categories. It supports multiple binning strategies,
integrates with popular data science libraries, and provides sklearn-compatible
transformers for machine learning pipelines.

Key Features:
    - Multiple binning methods: equal-width, supervised, singleton, and flexible binning
    - Support for pandas and polars DataFrames
    - Scikit-learn compatible transformers
    - Advanced features like guidance columns and custom bin specifications
    - Comprehensive error handling and validation
    - Integration utilities for ML workflows

Main Components:
    Methods: EqualWidthBinning, SupervisedBinning, SingletonBinning
    Base Classes: GeneralBinningBase, IntervalBinningBase, FlexibleBinningBase
    Utilities: Data handling, bin operations, error management
    Integration: Feature selection, pipeline utilities, scoring functions

Example:
    >>> from binlearn import EqualWidthBinning
    >>> import numpy as np
    >>> X = np.random.rand(100, 3)
    >>> binner = EqualWidthBinning(n_bins=5)
    >>> X_binned = binner.fit_transform(X)
"""

from typing import Any, Optional

# Version information
try:
    from ._version import __version__
except ImportError:
    # Fallback for development or when setuptools_scm hasn't run yet
    __version__ = "unknown"

# Configuration management
# Base classes and utilities
# Integration tools
from . import tools
from .base import (
    Array1D,
    Array2D,
    ArrayLike,
    BinCountDict,
    BinEdges,
    BinEdgesDict,
    BinReps,
    BinRepsDict,
    BooleanMask,
    # Type aliases
    ColumnId,
    ColumnList,
    FitParams,
    FlexibleBinCalculationResult,
    FlexibleBinDef,
    FlexibleBinDefs,
    FlexibleBinningBase,
    FlexibleBinSpec,
    # Base classes
    GeneralBinningBase,
    GuidanceColumns,
    IntervalBinCalculationResult,
    IntervalBinningBase,
    JointParams,
    OptionalBinEdgesDict,
    OptionalBinRepsDict,
    OptionalColumnList,
    OptionalFlexibleBinSpec,
    SupervisedBinningBase,
    # Utility functions
    prepare_array,
    prepare_input_with_columns,
    return_like_input,
)
from .config import get_config, load_config, reset_config, set_config

# Concrete binning methods
from .methods import (
    Chi2Binning,
    DBSCANBinning,
    EqualFrequencyBinning,
    EqualWidthBinning,
    EqualWidthMinimumWeightBinning,
    GaussianMixtureBinning,
    IsotonicBinning,
    KMeansBinning,
    ManualFlexibleBinning,
    ManualIntervalBinning,
    SingletonBinning,
    TreeBinning,
)

# Error handling
# Parameter conversion utilities
from .utils import (
    ABOVE_RANGE,
    BELOW_RANGE,
    # Constants
    MISSING_VALUE,
    BinningError,
    ConfigurationError,
    FittingError,
    InvalidDataError,
    TransformationError,
    ValidationError,
    create_bin_masks,
    default_representatives,
    resolve_n_bins_parameter,
    resolve_string_parameter,
    validate_bin_edges_format,
    validate_bin_representatives_format,
    validate_bins,
    validate_numeric_parameter,
)

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
    "validate_bin_edges_format",
    "validate_bin_representatives_format",
    "validate_bins",
    "default_representatives",
    "create_bin_masks",
    # Parameter conversion utilities
    "resolve_n_bins_parameter",
    "resolve_string_parameter",
    "validate_numeric_parameter",
    # Concrete methods
    "Chi2Binning",
    "DBSCANBinning",
    "EqualFrequencyBinning",
    "EqualWidthBinning",
    "EqualWidthMinimumWeightBinning",
    "GaussianMixtureBinning",
    "IsotonicBinning",
    "KMeansBinning",
    "ManualFlexibleBinning",
    "ManualIntervalBinning",
    "SingletonBinning",
    "TreeBinning",
    # Integration tools
    "tools",
    # Optional dependencies
    "PANDAS_AVAILABLE",
    "pd",
    "POLARS_AVAILABLE",
    "pl",
]
