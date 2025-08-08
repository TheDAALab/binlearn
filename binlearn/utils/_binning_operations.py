"""
Binning operations utilities for interval and flexible binning.

This module provides utility functions for working with both traditional interval bins
and flexible bins that can contain singleton values and interval ranges.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._types import (
    ABOVE_RANGE,
    BELOW_RANGE,
    MISSING_VALUE,
    BinCountDict,
    BinEdges,
    BinEdgesDict,
    BinReps,
    BinRepsDict,
    BooleanMask,
    ColumnId,
    FlexibleBinDef,
    FlexibleBinDefs,
    FlexibleBinSpec,
)

# =============================================================================
# INTERVAL BINNING OPERATIONS
# =============================================================================


def validate_bin_edges_format(bin_edges: Any) -> None:
    """Validate bin edges format without transformation.

    Args:
        bin_edges: Input bin edges to validate.

    Raises:
        ValueError: If format is invalid.
    """
    if bin_edges is None:
        return

    if not isinstance(bin_edges, dict):
        raise ValueError("bin_edges must be a dictionary mapping column identifiers to edge lists")

    for col_id, edges in bin_edges.items():
        if not hasattr(edges, "__iter__") or isinstance(edges, str | bytes):
            raise ValueError(
                f"Edges for column {col_id} must be array-like (list, tuple, or numpy array)"
            )

        edges_list = list(edges)
        if len(edges_list) < 2:
            raise ValueError(f"Column {col_id} needs at least 2 bin edges")

        # Check if all values are numeric
        try:
            float_edges = [float(x) for x in edges_list]
        except (ValueError, TypeError) as exc:
            raise ValueError(f"All edges for column {col_id} must be numeric") from exc

        # Check if edges are sorted
        if not all(float_edges[i] <= float_edges[i + 1] for i in range(len(float_edges) - 1)):
            raise ValueError(f"Bin edges for column {col_id} must be sorted in ascending order")

        # Check for invalid values
        if any(not np.isfinite(x) for x in float_edges):
            raise ValueError(f"Bin edges for column {col_id} must be finite values")


def validate_bin_representatives_format(bin_representatives: Any, bin_edges: Any = None) -> None:
    """Validate bin representatives format without transformation.

    Args:
        bin_representatives: Input bin representatives to validate.
        bin_edges: Optional bin edges to check compatibility.

    Raises:
        ValueError: If format is invalid.
    """
    if bin_representatives is None:
        return

    if not isinstance(bin_representatives, dict):
        raise ValueError(
            "bin_representatives must be a dictionary mapping column identifiers to"
            " representative lists"
        )

    for col_id, reps in bin_representatives.items():
        if not hasattr(reps, "__iter__") or isinstance(reps, str | bytes):
            raise ValueError(
                f"Representatives for column {col_id} must be array-like (list, tuple,"
                " or numpy array)"
            )

        reps_list = list(reps)

        # Check if all values are numeric
        try:
            float_reps = [float(x) for x in reps_list]
        except (ValueError, TypeError) as exc:
            raise ValueError(f"All representatives for column {col_id} must be numeric") from exc

        # Check for invalid values
        if any(not np.isfinite(x) for x in float_reps):
            raise ValueError(f"Representatives for column {col_id} must be finite values")

        # Check compatibility with bin edges if provided
        if bin_edges is not None and col_id in bin_edges:
            expected_bins = len(list(bin_edges[col_id])) - 1
            if len(reps_list) != expected_bins:
                raise ValueError(
                    f"Column {col_id}: {len(reps_list)} representatives provided, but "
                    f"{expected_bins} expected"
                )


def validate_bins(bin_spec: BinEdgesDict | None, bin_reps: BinRepsDict | None) -> None:
    """Validate bin specifications and representatives.

    Args:
        bin_spec: Dictionary of bin edges.
        bin_reps: Dictionary of bin representatives.

    Raises:
        ValueError: If bins are invalid.
    """
    if bin_spec is None:
        return

    for col, edges in bin_spec.items():
        edges_list = list(edges)
        if len(edges_list) < 2:
            raise ValueError(f"Column {col} needs at least 2 bin edges")

        # Check if edges are sorted
        float_edges = [float(x) for x in edges_list]
        if not all(float_edges[i] <= float_edges[i + 1] for i in range(len(float_edges) - 1)):
            raise ValueError(f"Bin edges for column {col} must be non-decreasing")

        # Check representatives match
        if bin_reps is not None and col in bin_reps:
            n_bins = len(edges_list) - 1
            reps_list = list(bin_reps[col])
            if len(reps_list) != n_bins:
                raise ValueError(
                    f"Column {col}: {len(reps_list)} representatives for {n_bins} bins"
                )


def default_representatives(edges: BinEdges) -> BinReps:
    """Compute default bin representatives (centers)."""
    reps = []
    for i in range(len(edges) - 1):
        left, right = edges[i], edges[i + 1]
        if np.isneginf(left) and np.isposinf(right):
            reps.append(0.0)
        elif np.isneginf(left):
            reps.append(float(right) - 1.0)
        elif np.isposinf(right):
            reps.append(float(left) + 1.0)
        else:
            reps.append((left + right) / 2.0)
    return reps


def create_bin_masks(
    bin_indices: np.ndarray[Any, Any], n_bins: int
) -> tuple[BooleanMask, BooleanMask, BooleanMask, BooleanMask]:
    """Create boolean masks for different bin index types.

    Args:
        bin_indices: Array of bin indices.
        n_bins: Number of valid bins.

    Returns:
        Tuple of (valid_mask, nan_mask, below_mask, above_mask).
    """
    # Create masks for special values first
    nan_mask = bin_indices == MISSING_VALUE
    below_mask = bin_indices == BELOW_RANGE
    above_mask = bin_indices == ABOVE_RANGE

    # Valid indices are non-negative, less than n_bins, and not special values
    valid = (bin_indices >= 0) & (bin_indices < n_bins) & ~nan_mask & ~below_mask & ~above_mask

    return valid, nan_mask, below_mask, above_mask


# =============================================================================
# FLEXIBLE BINNING OPERATIONS
# =============================================================================


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
        if isinstance(bin_def, int | float):
            # Numeric singleton bin
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
            _validate_single_flexible_bin_def(
                bin_def, col, bin_idx, check_finite_bounds=False, strict=True
            )


def validate_flexible_bin_spec_format(
    bin_spec: FlexibleBinSpec, check_finite_bounds: bool = False, strict: bool = True
) -> None:
    """Validate the format and content of flexible bin specifications.

    Parameters
    ----------
    bin_spec : FlexibleBinSpec
        Dictionary mapping columns to lists of bin definitions to validate.
    check_finite_bounds : bool, optional
        Whether to check that interval bounds are finite (not inf or -inf).
        Defaults to False for backwards compatibility.
    strict : bool, optional
        Whether to perform strict validation. When False, allows empty bin
        definitions and equal interval bounds. Defaults to True.

    Raises
    ------
    ValueError
        If bin specifications are invalid.
    """
    if not isinstance(bin_spec, dict):
        raise ValueError("bin_spec must be a dictionary mapping columns to bin definitions")

    for col, bin_defs in bin_spec.items():
        if not isinstance(bin_defs, list | tuple):
            raise ValueError(f"Bin definitions for column {col} must be a list or tuple")

        if strict and len(bin_defs) == 0:
            raise ValueError(f"Bin specifications for column {col} cannot be empty")

        # Validate each bin definition
        for bin_idx, bin_def in enumerate(bin_defs):
            _validate_single_flexible_bin_def(bin_def, col, bin_idx, check_finite_bounds, strict)


def _validate_single_flexible_bin_def(
    bin_def: FlexibleBinDef,
    col: ColumnId,
    bin_idx: int,
    check_finite_bounds: bool = False,
    strict: bool = True,
) -> None:
    """Validate a single flexible bin definition.

    Parameters
    ----------
    bin_def : FlexibleBinDef
        Single bin definition to validate - either scalar (singleton) or tuple (interval).
    col : ColumnId
        Column identifier for error messages.
    bin_idx : int
        Bin index for error messages.
    check_finite_bounds : bool, optional
        Whether to check that interval bounds are finite (not inf or -inf).
        Defaults to False for backwards compatibility.
    strict : bool, optional
        Whether to perform strict validation. When False, allows equal interval bounds.
        Defaults to True.

    Raises
    ------
    ValueError
        If bin definition is invalid.
    """
    if isinstance(bin_def, int | float):
        # Numeric singleton bin - optionally check if finite
        if check_finite_bounds and not np.isfinite(bin_def):
            raise ValueError(f"Column {col}, bin {bin_idx}: Singleton value must be finite")
        return
    if isinstance(bin_def, tuple):
        if len(bin_def) != 2:
            raise ValueError(f"Column {col}, bin {bin_idx}: Interval must be (min, max)")

        left, right = bin_def
        if not isinstance(left, int | float) or not isinstance(right, int | float):
            raise ValueError(f"Column {col}, bin {bin_idx}: Interval values must be numeric")

        # Check for finite bounds if required
        if check_finite_bounds:
            if not (np.isfinite(left) and np.isfinite(right)):
                raise ValueError(f"Column {col}, bin {bin_idx}: Interval bounds must be finite")

        # Check for proper ordering - be less strict when not in strict mode
        if strict and left >= right:
            raise ValueError(
                f"Column {col}, bin {bin_idx}: Interval min ({left}) must be < max ({right})"
            )
        if not strict and left > right:
            raise ValueError(
                f"Column {col}, bin {bin_idx}: Interval min ({left}) must be <= max ({right})"
            )
    else:
        raise ValueError(
            f"Column {col}, bin {bin_idx}: Bin must be either a numeric scalar (singleton) or "
            f"tuple (interval)"
        )


def is_missing_value(value: Any) -> bool:
    """Check if a value represents a missing value.

    Parameters
    ----------
    value : Any
        Value to check.

    Returns
    -------
    bool
        True if the value is considered missing (None or NaN for numeric types), False otherwise.
    """
    if value is None:
        return True

    if isinstance(value, int | float):
        return bool(np.isnan(value))

    return False


def find_flexible_bin_for_value(value: Any, bin_defs: FlexibleBinDefs) -> int:
    """Find the bin index for a given value in flexible bin definitions.

    Parameters
    ----------
    value : Any
        Value to find bin for (must be numeric).
    bin_defs : FlexibleBinDefs
        List of bin definitions to search through.

    Returns
    -------
    int
        Bin index if found, MISSING_VALUE if no match.
    """
    for bin_idx, bin_def in enumerate(bin_defs):
        if isinstance(bin_def, int | float):
            # Singleton bin - direct comparison
            if value == bin_def:
                return bin_idx
        elif isinstance(bin_def, tuple) and len(bin_def) == 2:
            # Interval bin - only for numeric values
            left, right = bin_def
            if isinstance(value, int | float) and left <= value <= right:
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
    if isinstance(bin_def, int | float):
        # Singleton bin has zero width
        return 0.0
    if isinstance(bin_def, tuple) and len(bin_def) == 2:
        # Interval bin
        left, right = bin_def
        return right - left  # type: ignore[no-any-return]

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
