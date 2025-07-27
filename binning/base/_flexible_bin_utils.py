"""
Utilities for flexible binning (singleton and interval bins).

This module provides utility functions for working with flexible bin specifications
that support both singleton bins {"singleton": value} and interval bins 
{"interval": [min, max]}.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from ._constants import MISSING_VALUE

# Type aliases for flexible binning
FlexibleBinSpec = Dict[Any, List[Dict[str, Any]]]  # col_id -> list of bin definitions
FlexibleBinReps = Dict[Any, List[float]]  # col_id -> list of representatives


def ensure_flexible_bin_spec(bin_spec: Any) -> FlexibleBinSpec:
    """Ensure bin_spec is in the correct flexible bin format.
    
    Parameters
    ----------
    bin_spec : Any
        Input bin specification to validate and convert.
        
    Returns
    -------
    FlexibleBinSpec
        Dictionary mapping columns to lists of bin definitions.
        
    Raises
    ------
    ValueError
        If bin_spec is not a valid format.
    """
    if bin_spec is None:
        return {}
        
    if isinstance(bin_spec, dict):
        return bin_spec
    else:
        # Handle other formats if needed
        raise ValueError("bin_spec must be a dictionary mapping columns to bin definitions")


def generate_default_flexible_representatives(bin_defs: List[Dict[str, Any]]) -> List[float]:
    """Generate default representatives for flexible bins.
    
    Parameters
    ----------
    bin_defs : List[Dict[str, Any]]
        List of bin definitions, each containing either 'singleton' or 'interval' key.
        
    Returns
    -------
    List[float]
        List of representative values for each bin.
        
    Raises
    ------
    ValueError
        If a bin definition has unknown format.
    """
    reps = []
    for bin_def in bin_defs:
        if "singleton" in bin_def:
            reps.append(float(bin_def["singleton"]))
        elif "interval" in bin_def:
            a, b = bin_def["interval"]
            reps.append((a + b) / 2)  # Midpoint
        else:
            raise ValueError(f"Unknown bin definition: {bin_def}")
    return reps


def validate_flexible_bins(bin_spec: FlexibleBinSpec, bin_reps: FlexibleBinReps) -> None:
    """Validate flexible bin specifications and representatives.
    
    Parameters
    ----------
    bin_spec : FlexibleBinSpec
        Dictionary mapping columns to lists of bin definitions.
    bin_reps : FlexibleBinReps  
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
        for i, bin_def in enumerate(bin_defs):
            if not isinstance(bin_def, dict):
                raise ValueError(f"Column {col}, bin {i}: Bin definition must be a dictionary")

            if "singleton" in bin_def:
                if len(bin_def) != 1:
                    raise ValueError(
                        f"Column {col}, bin {i}: Singleton bin must have only 'singleton' key"
                    )
            elif "interval" in bin_def:
                if len(bin_def) != 1:
                    raise ValueError(
                        f"Column {col}, bin {i}: Interval bin must have only 'interval' key"
                    )

                interval = bin_def["interval"]
                if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                    raise ValueError(f"Column {col}, bin {i}: Interval must be [min, max]")

                if interval[0] > interval[1]:
                    raise ValueError(f"Column {col}, bin {i}: Interval min must be <= max")
            else:
                raise ValueError(
                    f"Column {col}, bin {i}: Bin must have 'singleton' or 'interval' key"
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


def find_flexible_bin_for_value(value: float, bin_defs: List[Dict[str, Any]]) -> int:
    """Find the bin index for a given value in flexible bin definitions.
    
    Parameters
    ----------
    value : float
        Value to find bin for.
    bin_defs : List[Dict[str, Any]]
        List of bin definitions to search through.
        
    Returns
    -------
    int
        Bin index if found, MISSING_VALUE if no match.
    """
    for bin_idx, bin_def in enumerate(bin_defs):
        if "singleton" in bin_def:
            if value == bin_def["singleton"]:
                return bin_idx
        elif "interval" in bin_def:
            a, b = bin_def["interval"]
            if a <= value <= b:
                return bin_idx

    # Value doesn't match any bin - treat as missing
    return MISSING_VALUE


def calculate_flexible_bin_width(bin_def: Dict[str, Any]) -> float:
    """Calculate width of a flexible bin definition.
    
    Parameters
    ----------
    bin_def : Dict[str, Any]
        Bin definition containing either 'singleton' or 'interval' key.
        
    Returns
    -------
    float
        Width of the bin (0.0 for singleton bins).
        
    Raises
    ------
    ValueError
        If bin definition has unknown format.
    """
    if "singleton" in bin_def:
        return 0.0  # Singleton has zero width
    elif "interval" in bin_def:
        a, b = bin_def["interval"]
        return b - a
    else:
        raise ValueError(f"Unknown bin definition: {bin_def}")


def transform_value_to_flexible_bin(
    value: Any, 
    bin_defs: List[Dict[str, Any]]
) -> int:
    """Transform a single value to its flexible bin index.
    
    Parameters
    ----------
    value : Any
        Value to transform.
    bin_defs : List[Dict[str, Any]]
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


def get_flexible_bin_count(bin_spec: FlexibleBinSpec) -> Dict[Any, int]:
    """Get number of bins for each column in flexible bin specification.
    
    Parameters
    ----------
    bin_spec : FlexibleBinSpec
        Dictionary mapping columns to bin definitions.
        
    Returns
    -------
    Dict[Any, int]
        Dictionary mapping columns to number of bins.
    """
    return {col: len(bin_defs) for col, bin_defs in bin_spec.items()}


def validate_single_flexible_bin_def(bin_def: Dict[str, Any], col: Any, bin_idx: int) -> None:
    """Validate a single flexible bin definition.
    
    Parameters
    ----------
    bin_def : Dict[str, Any]
        Single bin definition to validate.
    col : Any
        Column identifier for error messages.
    bin_idx : int
        Bin index for error messages.
        
    Raises
    ------
    ValueError
        If bin definition is invalid.
    """
    if not isinstance(bin_def, dict):
        raise ValueError(f"Column {col}, bin {bin_idx}: Bin definition must be a dictionary")

    if "singleton" in bin_def:
        if len(bin_def) != 1:
            raise ValueError(
                f"Column {col}, bin {bin_idx}: Singleton bin must have only 'singleton' key"
            )
    elif "interval" in bin_def:
        if len(bin_def) != 1:
            raise ValueError(
                f"Column {col}, bin {bin_idx}: Interval bin must have only 'interval' key"
            )

        interval = bin_def["interval"]
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError(f"Column {col}, bin {bin_idx}: Interval must be [min, max]")

        if interval[0] > interval[1]:
            raise ValueError(f"Column {col}, bin {bin_idx}: Interval min must be <= max")
    else:
        raise ValueError(
            f"Column {col}, bin {bin_idx}: Bin must have 'singleton' or 'interval' key"
        )
