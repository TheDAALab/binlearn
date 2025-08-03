"""
Bin operations utilities for interval binning.

This module provides utility functions for working with traditional interval bins.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .constants import ABOVE_RANGE, BELOW_RANGE, MISSING_VALUE
from .types import (
    BinEdges,
    BinEdgesDict,
    BinReps,
    BinRepsDict,
    BooleanMask,
)


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
        if not hasattr(edges, "__iter__") or isinstance(edges, (str, bytes)):
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
        if not hasattr(reps, "__iter__") or isinstance(reps, (str, bytes)):
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
                    f"Column {col}: {len(reps_list)} representatives " f"for {n_bins} bins"
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
