"""
Data handling utilities for binning operations.

This module provides utility functions for handling data inputs and outputs,
with support for pandas and polars DataFrames.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from binlearn import _pandas_config, _polars_config

from .types import ArrayLike, OptionalColumnList


def _is_pandas_df(obj: Any) -> bool:
    """Check if object is a pandas DataFrame."""
    pandas_module = _pandas_config.pd
    return pandas_module is not None and isinstance(obj, pandas_module.DataFrame)


def _is_polars_df(obj: Any) -> bool:
    """Check if object is a polars DataFrame."""
    polars_module = _polars_config.pl
    return polars_module is not None and isinstance(obj, polars_module.DataFrame)


def prepare_array(X: ArrayLike) -> tuple[np.ndarray, OptionalColumnList, Any]:
    """Convert input to numpy array and extract metadata.

    Args:
        X: Input data (array-like, pandas DataFrame, or polars DataFrame).

    Returns:
        Tuple of (numpy_array, column_names, index).
    """
    if _is_pandas_df(X):
        return np.asarray(X), list(X.columns), X.index
    if _is_polars_df(X):
        return X.to_numpy(), list(X.columns), None

    arr = np.asarray(X)
    # Ensure at least 2D
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr, None, None


def return_like_input(
    arr: np.ndarray,
    original_input: ArrayLike,
    columns: OptionalColumnList = None,
    preserve_dataframe: bool = False,
) -> ArrayLike:
    """Return array in same format as original input if requested.

    Args:
        arr: Numpy array to return.
        original_input: Original input data.
        columns: Column names for DataFrame output.
        preserve_dataframe: Whether to preserve DataFrame format.

    Returns:
        Array or DataFrame depending on settings and input type.
    """
    if not preserve_dataframe:
        return arr

    if _is_pandas_df(original_input):
        pandas_module = _pandas_config.pd
        if pandas_module is not None:
            cols = columns if columns is not None else list(original_input.columns)
            return pandas_module.DataFrame(arr, columns=cols, index=original_input.index)
    elif _is_polars_df(original_input):
        polars_module = _polars_config.pl
        if polars_module is not None:
            cols = columns if columns is not None else list(original_input.columns)
            return polars_module.DataFrame(arr, schema=cols)

    return arr


def _determine_columns(
    X: Any,
    col_names: list[Any] | None,
    fitted: bool,
    original_columns: list[Any] | None,
    arr_shape: tuple[int, ...],
) -> list[Any]:
    """Helper function to determine column identifiers.

    Parameters
    ----------
    X : ArrayLike
        Input data
    col_names : List or None
        Column names from prepare_array
    fitted : bool
        Whether this is for a fitted estimator
    original_columns : List or None
        Original column identifiers
    arr_shape : tuple
        Shape of the prepared array

    Returns
    -------
    columns : List[Any]
        Column identifiers
    """
    if col_names is not None:
        return col_names
    if hasattr(X, "shape") and len(X.shape) == 2:
        # For numpy arrays, use the actual number of columns
        return list(range(X.shape[1]))
    if fitted and original_columns is not None:
        # Use stored columns from fitting
        return original_columns

    # Fallback
    return list(range(arr_shape[1]))


def prepare_input_with_columns(
    X: ArrayLike, fitted: bool = False, original_columns: OptionalColumnList = None
) -> tuple[np.ndarray, list[Any]]:
    """Prepare input data and determine column identifiers.

    Parameters
    ----------
    X : ArrayLike
        Input data (pandas DataFrame, numpy array, etc.)
    fitted : bool, default=False
        Whether this is being called on a fitted estimator
    original_columns : List[Any] or None, default=None
        Original column identifiers from fitting

    Returns
    -------
    arr : np.ndarray
        Prepared array
    columns : List[Any]
        Column identifiers
    """
    arr, col_names, _ = prepare_array(X)

    # Determine column identifiers using helper function
    columns = _determine_columns(X, col_names, fitted, original_columns, arr.shape)

    return arr, columns
