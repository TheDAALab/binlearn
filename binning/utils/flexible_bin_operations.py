"""
Flexible binning utilities for working with singleton and interval bins.

This module provides utility functions for working with flexible bins that can
contain both singleton values and interval ranges.
"""

from __future__ import annotations
from typing import Any

import numpy as np
from .constants import MISSING_VALUE
from .types import (
    ColumnId,
    BinReps,
    BinRepsDict,
    FlexibleBinDef,
    FlexibleBinDefs,
    FlexibleBinSpec,
    BinCountDict,
)


def ensure_flexible_bin_spec(bin_spec: Any) -> FlexibleBinSpec:
    """Ensure bin_spec is in the correct flexible bin format.

    Parameters
    ----------
    bin_spec : Any
        Input bin specification to validate and convert.

    Returns
    -------
    Dict[Any, List[Any]]
        Dictionary mapping columns to lists of bin definitions in simplified format.

    Raises
    ------
    ValueError
        If bin_spec is not a valid format.
    """
    if bin_spec is None:
        return {}

    if isinstance(bin_spec, dict):
        # Convert any old dict format to new simplified format
        converted_spec = {}
        for col, bin_defs in bin_spec.items():
            converted_bin_defs = []
            for bin_def in bin_defs:
                if isinstance(bin_def, (int, float)):
                    # Already in simplified format
                    converted_bin_defs.append(bin_def)
                elif isinstance(bin_def, tuple):
                    # Already in simplified format
                    converted_bin_defs.append(bin_def)
                else:
                    raise ValueError(f"Invalid bin definition: {bin_def}")
            converted_spec[col] = converted_bin_defs
        return converted_spec

    # Handle other formats if needed
    raise ValueError("bin_spec must be a dictionary mapping columns to bin definitions")


def generate_default_flexible_representatives(bin_defs: FlexibleBinDefs) -> BinReps:
    """Generate default representatives for flexible bins.

    Parameters
    ----------
    bin_defs : FlexibleBinDefs
        List of bin definitions, each being either a scalar (singleton) or tuple (interval).

    Returns
    -------
    BinReps
        List of representative values for each bin.

    Raises
    ------
    ValueError
        If a bin definition has unknown format.
    """
    reps = []
    for bin_def in bin_defs:
        if isinstance(bin_def, (int, float)):
            # Singleton bin
            reps.append(float(bin_def))
        elif isinstance(bin_def, tuple) and len(bin_def) == 2:
            # Interval bin
            left, right = bin_def
            reps.append((left + right) / 2)  # Midpoint
        else:
            raise ValueError(f"Unknown bin definition: {bin_def}")
    return reps


def validate_flexible_bins(bin_spec: FlexibleBinSpec, bin_reps: BinRepsDict) -> None:
    """Validate flexible bin specifications and representatives.

    Parameters
    ----------
    bin_spec : FlexibleBinSpec
        Dictionary mapping columns to lists of bin definitions.
    bin_reps : BinRepsDict
        Dictionary mapping columns to lists of representatives.

    Raises
    ------
    ValueError
        If bin specifications are invalid.
    """
    for col in bin_spec:
        bin_defs = bin_spec[col]
        reps = bin_reps.get(col, [])

        if len(bin_defs) != len(reps):
            raise ValueError(
                f"Column {col}: Number of bin definitions ({len(bin_defs)}) "
                f"must match number of representatives ({len(reps)})"
            )

        # Validate bin definition format
        for bin_idx, bin_def in enumerate(bin_defs):
            _validate_single_flexible_bin_def(bin_def, col, bin_idx)


def _validate_single_flexible_bin_def(bin_def: FlexibleBinDef, col: ColumnId, bin_idx: int) -> None:
    """Validate a single flexible bin definition.

    Parameters
    ----------
    bin_def : FlexibleBinDef
        Single bin definition to validate - either scalar (singleton) or tuple (interval).
    col : ColumnId
        Column identifier for error messages.
    bin_idx : int
        Bin index for error messages.

    Raises
    ------
    ValueError
        If bin definition is invalid.
    """
    if isinstance(bin_def, (int, float)):
        # Singleton bin - no additional validation needed
        return
    if isinstance(bin_def, tuple):
        if len(bin_def) != 2:
            raise ValueError(f"Column {col}, bin {bin_idx}: Interval must be (min, max)")

        left, right = bin_def
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            raise ValueError(f"Column {col}, bin {bin_idx}: Interval values must be numeric")

        if left > right:
            raise ValueError(f"Column {col}, bin {bin_idx}: Interval min must be <= max")
    else:
        raise ValueError(
            f"Column {col}, bin {bin_idx}: Bin must be either a scalar (singleton) or "
            f"tuple (interval)"
        )


def is_missing_value(value: Any) -> bool:
    """Check if a value represents a missing value (numeric only).

    Parameters
    ----------
    value : Any
        Value to check.

    Returns
    -------
    bool
        True if the value is considered missing (NaN), False otherwise.
        Non-numeric values are treated as missing.
    """
    try:
        # Convert to float and check for NaN only (not inf)
        float_val = float(value)
        if np.isnan(float_val):
            return True
    except (TypeError, ValueError):
        # Cannot convert to numeric - treat as missing for our numeric-only approach
        return True

    return False


def find_flexible_bin_for_value(value: float, bin_defs: FlexibleBinDefs) -> int:
    """Find the bin index for a given value in flexible bin definitions.

    Parameters
    ----------
    value : float
        Value to find bin for.
    bin_defs : FlexibleBinDefs
        List of bin definitions to search through.

    Returns
    -------
    int
        Bin index if found, MISSING_VALUE if no match.
    """
    for bin_idx, bin_def in enumerate(bin_defs):
        if isinstance(bin_def, (int, float)):
            # Singleton bin
            if value == bin_def:
                return bin_idx
        elif isinstance(bin_def, tuple) and len(bin_def) == 2:
            # Interval bin
            left, right = bin_def
            if left <= value <= right:
                return bin_idx

    # Value doesn't match any bin - treat as missing
    return MISSING_VALUE


def calculate_flexible_bin_width(bin_def: FlexibleBinDef) -> float:
    """Calculate width of a flexible bin definition.

    Parameters
    ----------
    bin_def : FlexibleBinDef
        Bin definition - either scalar (singleton) or tuple (interval).

    Returns
    -------
    float
        Width of the bin (0.0 for singleton bins).

    Raises
    ------
    ValueError
        If bin definition has unknown format.
    """
    if isinstance(bin_def, (int, float)):
        # Singleton bin has zero width
        return 0.0
    if isinstance(bin_def, tuple) and len(bin_def) == 2:
        # Interval bin
        left, right = bin_def
        return right - left

    raise ValueError(f"Unknown bin definition: {bin_def}")


def transform_value_to_flexible_bin(value: Any, bin_defs: FlexibleBinDefs) -> int:
    """Transform a single value to its flexible bin index.

    Parameters
    ----------
    value : Any
        Value to transform.
    bin_defs : FlexibleBinDefs
        List of bin definitions.

    Returns
    -------
    int
        Bin index or MISSING_VALUE.
    """
    # Robust missing value check
    if is_missing_value(value):
        return MISSING_VALUE

    # Find matching bin
    return find_flexible_bin_for_value(value, bin_defs)


def get_flexible_bin_count(bin_spec: FlexibleBinSpec) -> BinCountDict:
    """Get number of bins for each column in flexible bin specification.

    Parameters
    ----------
    bin_spec : FlexibleBinSpec
        Dictionary mapping columns to bin definitions.

    Returns
    -------
    BinCountDict
        Dictionary mapping columns to number of bins.
    """
    return {col: len(bin_defs) for col, bin_defs in bin_spec.items()}
