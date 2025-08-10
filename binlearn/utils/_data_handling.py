"""
Data handling utilities for binning operations.

This module provides utility functions for handling data inputs and outputs,
with support for pandas and polars DataFrames.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from binlearn import _pandas_config, _polars_config

from ._types import ArrayLike, OptionalColumnList


# pylint: disable=too-many-return-statements
def convert_to_python_types(value: Any) -> Any:
    """Convert numpy types to pure Python types recursively for serialization.

    This function recursively processes nested data structures (dicts, lists, tuples)
    and converts numpy data types to equivalent Python built-in types. This is
    particularly useful for JSON serialization and ensuring compatibility with
    systems that don't handle numpy types directly.

    Args:
        value: Input value to convert. Can be any type including nested structures
            containing numpy arrays, numpy scalars, or mixed data types.

    Returns:
        The input value with all numpy types converted to Python equivalents:
        - numpy arrays become lists
        - numpy scalars become Python int, float, or bool
        - numpy bool_ becomes Python bool
        - numpy integers become Python int
        - numpy floating types become Python float
        - Other types are preserved unchanged

    Example:
        >>> import numpy as np
        >>> data = {
        ...     'array': np.array([1, 2, 3]),
        ...     'scalar': np.int64(42),
        ...     'bool': np.bool_(True),
        ...     'nested': [np.float32(3.14), {'inner': np.array([4, 5])}]
        ... }
        >>> converted = convert_to_python_types(data)
        >>> type(converted['array'])
        <class 'list'>
        >>> type(converted['scalar'])
        <class 'int'>

    Note:
        This function handles nested structures recursively, so very deeply nested
        data structures may cause recursion limits to be reached.
    """
    if isinstance(value, dict):
        return {k: convert_to_python_types(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        converted = [convert_to_python_types(item) for item in value]
        return type(value)(converted) if isinstance(value, tuple) else converted
    if isinstance(value, np.ndarray):
        return convert_to_python_types(value.tolist())
    if isinstance(value, np.number | np.bool_):
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        return value.item()
    return value


def _is_pandas_df(obj: Any) -> bool:
    """Check if object is a pandas DataFrame.

    This is a safe check that handles the case where pandas is not installed
    or not available in the current environment.

    Args:
        obj: Object to check for pandas DataFrame type.

    Returns:
        True if the object is a pandas DataFrame and pandas is available,
        False otherwise.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'a': [1, 2, 3]})
        >>> _is_pandas_df(df)
        True
        >>> _is_pandas_df([1, 2, 3])
        False
    """
    pandas_module = _pandas_config.pd
    return pandas_module is not None and isinstance(obj, pandas_module.DataFrame)


def _is_polars_df(obj: Any) -> bool:
    """Check if object is a polars DataFrame.

    This is a safe check that handles the case where polars is not installed
    or not available in the current environment.

    Args:
        obj: Object to check for polars DataFrame type.

    Returns:
        True if the object is a polars DataFrame and polars is available,
        False otherwise.

    Example:
        >>> import polars as pl
        >>> df = pl.DataFrame({'a': [1, 2, 3]})
        >>> _is_polars_df(df)
        True
        >>> _is_polars_df([1, 2, 3])
        False
    """
    polars_module = _polars_config.pl
    return polars_module is not None and isinstance(obj, polars_module.DataFrame)


def prepare_array(X: ArrayLike) -> tuple[np.ndarray[Any, Any], OptionalColumnList, Any]:
    """Convert input to numpy array and extract metadata.

    This function standardizes input data by converting various formats (numpy arrays,
    pandas DataFrames, polars DataFrames, lists) to numpy arrays while preserving
    important metadata like column names and index information.

    Args:
        X: Input data in various formats:
            - numpy.ndarray: Used directly with shape normalization
            - pandas.DataFrame: Converted to numpy array, columns and index preserved
            - polars.DataFrame: Converted to numpy array, columns preserved
            - list/tuple: Converted to numpy array with shape normalization
            - scalar: Converted to 1x1 numpy array

    Returns:
        A tuple containing:
        - numpy_array: The input data as a numpy array, guaranteed to be at least 2D
        - column_names: List of column names (for DataFrames) or None (for arrays)
        - index: Index information (for pandas DataFrames) or None (for others)

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # With pandas DataFrame
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> arr, cols, idx = prepare_array(df)
        >>> print(arr.shape, cols)
        (2, 2) ['A', 'B']
        >>>
        >>> # With numpy array
        >>> X = np.array([[1, 2], [3, 4]])
        >>> arr, cols, idx = prepare_array(X)
        >>> print(arr.shape, cols)
        (2, 2) None
        >>>
        >>> # With 1D array (gets reshaped)
        >>> X = [1, 2, 3]
        >>> arr, cols, idx = prepare_array(X)
        >>> print(arr.shape)
        (3, 1)

    Note:
        - Input arrays are always normalized to at least 2D
        - Scalar inputs become 1x1 arrays
        - 1D inputs become Nx1 arrays
        - DataFrame metadata is preserved for later reconstruction
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
    arr: np.ndarray[Any, Any],
    original_input: ArrayLike,
    columns: OptionalColumnList = None,
    preserve_dataframe: bool = False,
) -> ArrayLike:
    """Return array in same format as original input if requested.

    This function allows returning processed data in the same format as the original
    input when preserve_dataframe is True. This is useful for maintaining workflow
    consistency when users expect the same data type they provided.

    Args:
        arr: Processed numpy array to return or convert.
        original_input: Original input data that determines the return format.
            Used to detect whether input was pandas DataFrame, polars DataFrame,
            or other array-like format.
        columns: Column names to use when creating DataFrame output. If None,
            attempts to use columns from original_input for DataFrames.
        preserve_dataframe: Whether to preserve DataFrame format when the original
            input was a DataFrame. If False, always returns numpy array.

    Returns:
        The processed data in the requested format:
        - If preserve_dataframe is False: Always returns numpy array
        - If preserve_dataframe is True and original was pandas DataFrame:
          Returns pandas DataFrame with same index and specified/original columns
        - If preserve_dataframe is True and original was polars DataFrame:
          Returns polars DataFrame with specified/original columns
        - Otherwise: Returns numpy array

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Original DataFrame preserved
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> arr = np.array([[10, 20], [30, 40]])
        >>> result = return_like_input(arr, df, preserve_dataframe=True)
        >>> type(result)
        <class 'pandas.core.frame.DataFrame'>
        >>>
        >>> # Array format when preserve_dataframe=False
        >>> result = return_like_input(arr, df, preserve_dataframe=False)
        >>> type(result)
        <class 'numpy.ndarray'>

    Note:
        - Requires pandas/polars to be installed when working with respective DataFrame types
        - Column names and index are preserved from original DataFrames when possible
        - If preserve_dataframe=False, this function simply returns the input array
    """
    if not preserve_dataframe:
        return arr

    if _is_pandas_df(original_input):
        # pandas_module is guaranteed to be not None if _is_pandas_df returns True
        pandas_module = _pandas_config.pd
        assert pandas_module is not None  # This should always be true
        cols = columns if columns is not None else list(original_input.columns)
        return pandas_module.DataFrame(arr, columns=cols, index=original_input.index)
    if _is_polars_df(original_input):
        # polars_module is guaranteed to be not None if _is_polars_df returns True
        polars_module = _polars_config.pl
        assert polars_module is not None  # This should always be true
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
    """Helper function to determine column identifiers from various sources.

    This function implements a priority system for determining column identifiers
    when working with different data formats. It ensures consistent column handling
    across different input types and usage contexts.

    Args:
        X: Input data object, used for shape inference when needed.
        col_names: Column names extracted from input data (e.g., DataFrame columns).
            Takes highest priority when available.
        fitted: Whether this is being called on a fitted estimator. Affects
            priority of original_columns parameter.
        original_columns: Column identifiers from the original training data.
            Used when fitted=True and col_names is not available.
        arr_shape: Shape tuple of the prepared array, used as fallback for
            generating numeric column indices.

    Returns:
        List of column identifiers determined by priority:
        1. Column names from input data itself (DataFrame columns)
        2. For fitted estimators, stored original columns
        3. For numpy arrays with shape info, numeric indices
        4. Fallback to numeric indices based on arr_shape

    Example:
        >>> import pandas as pd
        >>> # DataFrame with column names (highest priority)
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> cols = _determine_columns(df, ['A', 'B'], False, None, (2, 2))
        >>> print(cols)
        ['A', 'B']
        >>>
        >>> # Fitted estimator with stored columns
        >>> cols = _determine_columns(None, None, True, ['X', 'Y'], (2, 2))
        >>> print(cols)
        ['X', 'Y']
        >>>
        >>> # Fallback to numeric indices
        >>> cols = _determine_columns(None, None, False, None, (2, 3))
        >>> print(cols)
        [0, 1, 2]

    Note:
        This is a helper function used internally by prepare_input_with_columns
        to maintain consistent column identification logic.
    """
    # Priority order for column determination:
    # 1. Column names from input data itself (DataFrame columns)
    if col_names is not None:
        return col_names
    # 2. For fitted estimators, use stored original columns
    if fitted and original_columns is not None:
        return original_columns
    # 3. For numpy arrays, use numeric indices
    if hasattr(X, "shape") and len(X.shape) == 2:
        return list(range(X.shape[1]))
    # 4. Fallback to numeric indices
    return list(range(arr_shape[1]))


def prepare_input_with_columns(
    X: ArrayLike, fitted: bool = False, original_columns: OptionalColumnList = None
) -> tuple[np.ndarray[Any, Any], list[Any]]:
    """Prepare input data and determine column identifiers.

    This function combines array preparation with intelligent column identification,
    making it the primary entry point for processing input data in binning methods.
    It handles various input formats and maintains column consistency between
    fitting and transformation phases.

    Args:
        X: Input data in any supported format:
            - pandas.DataFrame: Columns names are extracted and preserved
            - polars.DataFrame: Column names are extracted and preserved
            - numpy.ndarray: Numeric column indices are generated
            - list/tuple: Converted to array with numeric column indices
        fitted: Whether this is being called on a fitted estimator. When True,
            the function will try to use original_columns to maintain consistency
            with the training phase.
        original_columns: Column identifiers from the original training data.
            Only used when fitted=True and input data doesn't provide column names.

    Returns:
        A tuple containing:
        - arr: Prepared numpy array (guaranteed to be at least 2D)
        - columns: List of column identifiers for consistent data handling

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Training phase with DataFrame
        >>> df_train = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
        >>> arr, cols = prepare_input_with_columns(df_train, fitted=False)
        >>> print(cols)
        ['feature1', 'feature2']
        >>>
        >>> # Transform phase - maintains column consistency
        >>> df_test = pd.DataFrame({'feature1': [5, 6], 'feature2': [7, 8]})
        >>> arr, cols = prepare_input_with_columns(df_test, fitted=True, original_columns=cols)
        >>> print(cols)
        ['feature1', 'feature2']
        >>>
        >>> # With numpy array
        >>> X = np.array([[1, 2, 3], [4, 5, 6]])
        >>> arr, cols = prepare_input_with_columns(X)
        >>> print(cols)
        [0, 1, 2]

    Raises:
        ValueError: If input data format is unsupported or incompatible.

    Note:
        This function is the recommended way to prepare input data for binning
        operations as it handles both data conversion and column management
        in a single step.
    """
    arr, col_names, _ = prepare_array(X)

    # Determine column identifiers using helper function
    columns = _determine_columns(X, col_names, fitted, original_columns, arr.shape)

    return arr, columns
