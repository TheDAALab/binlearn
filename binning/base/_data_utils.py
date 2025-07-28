"""
Simplified data handling utilities for binning operations.
"""

from __future__ import annotations
from typing import Any, Optional, Tuple

import numpy as np
from binning import _pandas_config, _polars_config
from ._types import ColumnList, OptionalColumnList, ArrayLike


def _is_pandas_df(obj: Any) -> bool:
    """Check if object is a pandas DataFrame."""
    pd = _pandas_config.pd
    return pd is not None and isinstance(obj, pd.DataFrame)


def _is_polars_df(obj: Any) -> bool:
    """Check if object is a polars DataFrame."""
    pl = _polars_config.pl
    return pl is not None and isinstance(obj, pl.DataFrame)


def prepare_array(X: ArrayLike) -> Tuple[np.ndarray, OptionalColumnList, Any]:
    """Convert input to numpy array and extract metadata.

    Args:
        X: Input data (array-like, pandas DataFrame, or polars DataFrame).

    Returns:
        Tuple of (numpy_array, column_names, index).
    """
    if _is_pandas_df(X):
        return np.asarray(X), list(X.columns), X.index
    elif _is_polars_df(X):
        return X.to_numpy(), list(X.columns), None
    else:
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
        pd = _pandas_config.pd
        if pd is not None:
            cols = columns if columns is not None else list(original_input.columns)
            return pd.DataFrame(arr, columns=cols, index=original_input.index)
    elif _is_polars_df(original_input):
        pl = _polars_config.pl
        if pl is not None:
            cols = columns if columns is not None else list(original_input.columns)
            return pl.DataFrame(arr, schema=cols)

    return arr


def prepare_input_with_columns(
    X: ArrayLike, fitted: bool = False, original_columns: OptionalColumnList = None
) -> Tuple[np.ndarray, ColumnList]:
    """
    Prepare input array and determine column identifiers.

    Parameters
    ----------
    X : Any
        Input data (numpy array, pandas DataFrame, etc.)
    fitted : bool, default=False
        Whether the estimator is fitted
    original_columns : List[Any], optional
        Original column names from fitting

    Returns
    -------
    arr : np.ndarray
        Prepared array
    columns : List[Any]
        Column identifiers
    """
    arr, col_names, _ = prepare_array(X)

    # Determine column identifiers
    if col_names is not None:
        columns = col_names
    elif hasattr(X, "shape") and len(X.shape) == 2:
        # For numpy arrays, use the actual number of columns
        columns = list(range(X.shape[1]))
    elif fitted and original_columns is not None:
        # Use stored columns from fitting, but only if dimensions match
        if hasattr(X, "shape") and len(X.shape) == 2 and X.shape[1] <= len(original_columns):
            columns = list(range(X.shape[1]))
        else:
            columns = original_columns
    else:
        # Fallback
        columns = list(range(arr.shape[1]))

    return arr, columns
