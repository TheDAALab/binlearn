"""
Equal-width binning utilities for fallback implementations.

This module provides standardized equal-width binning functionality used as
fallback across multiple binning methods to reduce code duplication.
"""

from __future__ import annotations

import warnings

import numpy as np


def create_equal_width_bins(
    data: np.ndarray,
    n_bins: int,
    data_range: tuple[float, float] | None = None,
    add_epsilon: bool = True,
) -> np.ndarray:
    """Create equal-width bin edges for given data.

    This function implements the standard equal-width binning algorithm
    used as fallback in many binning methods. It ensures consistent
    behavior across different implementations.

    Args:
        data: Input data array (1D)
        n_bins: Number of bins to create
        data_range: Optional (min, max) range. If None, computed from data
        add_epsilon: Whether to add small epsilon to max edge to ensure
            all data points fall within bins

    Returns:
        Array of bin edges with length n_bins + 1

    Example:
        >>> import numpy as np
        >>> data = np.array([1, 2, 3, 4, 5])
        >>> edges = create_equal_width_bins(data, 3)
        >>> print(edges)
        [1.   2.33 3.67 5.01]
    """
    if data_range is None:
        min_val, max_val = float(np.min(data)), float(np.max(data))
    else:
        min_val, max_val = data_range

    # Handle edge case where min == max
    if min_val == max_val:
        if add_epsilon:
            max_val += np.finfo(float).eps
        else:
            # Create bins around the single value
            epsilon = 1e-8 if min_val == 0 else abs(min_val) * 1e-8
            min_val -= epsilon
            max_val += epsilon

    # Create equal-width bin edges
    edges = np.linspace(min_val, max_val, n_bins + 1)

    # Add epsilon to the maximum edge to ensure all points are included
    if add_epsilon and data_range is None:
        edges[-1] += np.finfo(float).eps

    return edges


def apply_equal_width_fallback(
    data: np.ndarray, n_bins: int, method_name: str = "method", warn_on_fallback: bool = True
) -> np.ndarray:
    """Apply equal-width binning as fallback with optional warning.

    This function provides a standardized way to fall back to equal-width
    binning when other methods fail, with consistent warning messages.

    Args:
        data: Input data to bin
        n_bins: Number of bins to create
        method_name: Name of the original method (for warning message)
        warn_on_fallback: Whether to issue warning about fallback

    Returns:
        Array of equal-width bin edges

    Example:
        >>> import numpy as np
        >>> data = np.array([1, 1, 1, 2, 2, 3])
        >>> edges = apply_equal_width_fallback(data, 3, "kmeans", warn_on_fallback=True)
        # UserWarning: kmeans binning failed, falling back to equal-width binning
    """
    if warn_on_fallback:
        warnings.warn(
            f"{method_name} binning failed, falling back to equal-width binning. "
            f"This may happen with insufficient data or inappropriate parameters.",
            category=UserWarning,
            stacklevel=3,
        )

    return create_equal_width_bins(data, n_bins)


def validate_binning_input(data: np.ndarray, n_bins: int) -> None:
    """Validate input data and parameters for binning.

    Args:
        data: Input data array
        n_bins: Number of bins requested

    Raises:
        ValueError: If input is invalid for binning
    """
    if data.size == 0:
        raise ValueError("Cannot bin empty data array")

    if n_bins <= 0:
        raise ValueError(f"Number of bins must be positive, got {n_bins}")

    if n_bins >= data.size:
        warnings.warn(
            f"Number of bins ({n_bins}) is greater than or equal to "
            f"data size ({data.size}). This may result in many empty bins.",
            category=UserWarning,
            stacklevel=3,
        )


def ensure_monotonic_edges(edges: np.ndarray) -> np.ndarray:
    """Ensure bin edges are strictly monotonic.

    This function fixes potential issues with bin edges that may not be
    strictly increasing due to numerical precision or algorithm limitations.

    Args:
        edges: Array of bin edges

    Returns:
        Array of strictly monotonic bin edges
    """
    # Ensure edges are strictly increasing
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            # Use a more practical epsilon that's relative to the value magnitude
            if abs(edges[i - 1]) == 0:
                epsilon = 1e-10
            else:
                epsilon = max(np.finfo(float).eps, abs(edges[i - 1]) * 1e-10)
            edges[i] = edges[i - 1] + epsilon

    return edges
