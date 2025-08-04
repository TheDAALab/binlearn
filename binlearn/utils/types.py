"""
Type aliases for the binning framework.

This module provides standardized type aliases used throughout the binlearn library
to improve code readability and maintainability.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# =============================================================================
# COLUMN AND DATA TYPES
# =============================================================================

# Column identifiers can be any type (int, str, etc.)
ColumnId = Any
ColumnList = list[ColumnId]
OptionalColumnList = ColumnList | None

# Flexible guidance columns - can be single value or list
GuidanceColumns = ColumnList | ColumnId | None

# Array-like data types
ArrayLike = Any  # Could be np.ndarray, pandas.DataFrame, polars.DataFrame, list, etc.

# =============================================================================
# INTERVAL BINNING TYPES
# =============================================================================

# Bin edges for interval binning (e.g., [0, 1, 2, 3] for 3 bins)
BinEdges = list[float]
BinEdgesDict = dict[ColumnId, BinEdges]

# Bin representatives for interval binning (e.g., [0.5, 1.5, 2.5] for 3 bins)
BinReps = list[float]
BinRepsDict = dict[ColumnId, BinReps]

# Optional versions for parameters
OptionalBinEdgesDict = BinEdgesDict | Any | None
OptionalBinRepsDict = BinRepsDict | Any | None

# =============================================================================
# FLEXIBLE BINNING TYPES
# =============================================================================

# Single flexible bin definition - either a scalar (singleton) or tuple (interval)
FlexibleBinDef = Any  # Union[int, float, Tuple[Union[int, float], Union[int, float]]]

# List of flexible bin definitions for a column
FlexibleBinDefs = list[FlexibleBinDef]

# Dictionary mapping columns to their flexible bin definitions
FlexibleBinSpec = dict[ColumnId, FlexibleBinDefs]

# Optional versions for parameters
OptionalFlexibleBinSpec = FlexibleBinSpec | Any | None

# =============================================================================
# CALCULATION RETURN TYPES
# =============================================================================

# Return type for bin calculation methods in interval binning
IntervalBinCalculationResult = tuple[BinEdges, BinReps]

# Return type for bin calculation methods in flexible binning
FlexibleBinCalculationResult = tuple[FlexibleBinDefs, BinReps]

# =============================================================================
# COUNT AND VALIDATION TYPES
# =============================================================================

# Dictionary mapping columns to number of bins
BinCountDict = dict[ColumnId, int]

# =============================================================================
# NUMPY ARRAY TYPES (for better clarity)
# =============================================================================

# Common numpy array shapes used in binning
Array1D = np.ndarray  # 1D array
Array2D = np.ndarray  # 2D array
BooleanMask = np.ndarray  # Boolean array for masking

# =============================================================================
# PARAMETER TYPES FOR FUNCTIONS
# =============================================================================

# Common fit_params type
FitParams = dict[str, Any]

# Common joint parameters type
JointParams = dict[str, Any]
