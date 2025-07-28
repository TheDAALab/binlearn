"""
Bin operations utilities for interval binning.

This module provides utility functions for working with traditional interval bins.
"""

from __future__ import annotations
from typing import Any, Tuple

import numpy as np
from .constants import MISSING_VALUE, ABOVE_RANGE, BELOW_RANGE
from .types import (
    ColumnId,
    BinEdges,
    BinEdgesDict,
    BinReps,
    BinRepsDict,
    BooleanMask,
)


def ensure_bin_dict(data: Any) -> BinEdgesDict:
    """Convert any input to a dictionary of bin edges/representatives.

    Args:
        data: Input data (dict, array-like, or None).

    Returns:
        Dictionary mapping column keys to lists of values.
    """
    if data is None:
        return {}

    if isinstance(data, dict):
        return {k: list(np.asarray(v, dtype=float)) for k, v in data.items()}

    # Convert array-like to dict with integer keys
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 0:
        return {0: [float(arr)]}
    elif arr.ndim == 1:
        return {0: arr.tolist()}
    else:
        return {i: arr[i].tolist() for i in range(arr.shape[0])}


def validate_bins(bin_spec: BinEdgesDict, bin_reps: BinRepsDict) -> None:
    """Validate bin specifications and representatives.

    Args:
        bin_spec: Dictionary of bin edges.
        bin_reps: Dictionary of bin representatives.

    Raises:
        ValueError: If bins are invalid.
    """
    for col, edges in bin_spec.items():
        if len(edges) < 2:
            raise ValueError(f"Column {col} needs at least 2 bin edges")

        # Check if edges are sorted
        if not all(edges[i] <= edges[i + 1] for i in range(len(edges) - 1)):
            raise ValueError(f"Bin edges for column {col} must be non-decreasing")

        # Check representatives match
        if col in bin_reps:
            n_bins = len(edges) - 1
            if len(bin_reps[col]) != n_bins:
                raise ValueError(
                    f"Column {col}: {len(bin_reps[col])} representatives " f"for {n_bins} bins"
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
    bin_indices: np.ndarray, n_bins: int
) -> Tuple[BooleanMask, BooleanMask, BooleanMask, BooleanMask]:
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
