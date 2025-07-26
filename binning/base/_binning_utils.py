"""
Utility functions for binning base classes.
"""

from __future__ import annotations
from collections.abc import Hashable
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence, TYPE_CHECKING, cast

import numpy as np

from binning import _pandas_config
from binning import _polars_config

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

#
# Common type aliases for DataFrame-like objects and binning dicts
ArrayLike = Union[np.ndarray, "pd.DataFrame", "pl.DataFrame"]
BinEdgesDict = Dict[Hashable, List[float]]
BinSpecDict = Dict[Hashable, List[dict]]
OptBinEdgesDict = Optional[BinEdgesDict]
OptBinSpecDict = Optional[BinSpecDict]
BinSpecLike = Union[BinSpecDict, List[dict], None]
BinEdgesLike = Union[BinEdgesDict, Sequence[float], np.ndarray, None]
OptDictAny = Optional[Dict[Hashable, Any]]
OptListAny = Optional[List[Any]]
OptDictAnyInt = Optional[Dict[Any, int]]
OptDictIntAny = Optional[Dict[int, Any]]
PreparedArrayReturn = Tuple[np.ndarray, OptListAny, Optional[Any], OptDictAnyInt, OptDictIntAny]


def ensure_dict_format(
    data: Union[None, BinEdgesDict, Sequence[Any], np.ndarray],
) -> BinEdgesDict:
    """Convert input data to a dict-of-lists format for bin edges or representatives.

    Args:
        data: None, dict, or array-like.

    Returns:
        Dictionary mapping column indices/names to lists of bin edges or representatives.

    Raises:
        ValueError: If the input data is not in a supported format.
    """
    if data is None:
        return {}
    if isinstance(data, dict):
        out: Dict[Hashable, List[float]] = {}
        for k, v in data.items():
            arr = np.asarray(v) if v is not None and len(v) > 0 else np.array([-np.inf, np.inf])
            out[k] = arr.tolist()
        return out
    arr = np.asarray(data)
    if arr.ndim == 1:
        arr_out = arr if arr.size > 0 else np.array([-np.inf, np.inf])
        return {0: arr_out.tolist()}
    if arr.ndim == 2:
        return {
            i: arr[i].tolist() if arr[i].size > 0 else [-np.inf, np.inf]
            for i in range(arr.shape[0])
        }
    raise ValueError("Unsupported data format for bin edges/representatives.")


def ensure_dict_of_list_of_dicts_format(
    data: BinSpecLike,
) -> OptBinSpecDict:
    """Ensure data is a dict of lists of dicts for flexible binning.

    Args:
        data: None, dict, or list.

    Returns:
        Dict[Hashable, List[dict]] or None.

    Raises:
        ValueError: If a value in the dict is not a list of dicts, or if input is not a dict/list.
    """
    if data is None:
        return None
    if isinstance(data, dict):
        out: Dict[Hashable, List[dict]] = {}
        for k, v in data.items():
            if v is None or len(v) == 0:
                out[k] = []
            elif isinstance(v, list):
                out[k] = v
            else:
                raise ValueError("Each value in dict must be a list of dicts.")
        return out
    if isinstance(data, list):
        return {0: data}
    raise ValueError("Flexible bin spec must be a dict of lists or a list.")


def get_from_dict(
    dct: OptDictAny,
    col_idx: Any,
    X: Any = None,
    col_pos: Optional[int] = None,
) -> Any:
    """Robustly get a value from a dict by int, str, or column name.

    Args:
        dct: Dictionary to search.
        col_idx: Key to look up.
        X: Optional DataFrame-like object with columns.
        col_pos: Optional column position.

    Returns:
        Value from dict or None.
    """
    if dct is None:
        return None
    if col_idx in dct:
        return dct[col_idx]
    if str(col_idx) in dct:
        return dct[str(col_idx)]
    if X is not None and hasattr(X, "columns") and col_pos is not None:
        col_name = X.columns[col_pos]
        if col_name in dct:
            return dct[col_name]
    return None


def has_in_dict(
    dct: OptDictAny,
    col_idx: Any,
    X: Any = None,
    col_pos: Optional[int] = None,
) -> bool:
    """Check if a dict has a key by int, str, or column name.

    Args:
        dct: Dictionary to search.
        col_idx: Key to look up.
        X: Optional DataFrame-like object with columns.
        col_pos: Optional column position.

    Returns:
        True if key is present, False otherwise.
    """
    if dct is None:
        return False
    if col_idx in dct or str(col_idx) in dct:
        return True
    if X is not None and hasattr(X, "columns") and col_pos is not None:
        col_name = X.columns[col_pos]
        return col_name in dct
    return False


def default_representatives(bin_edges: List[float]) -> np.ndarray:
    """Compute default bin representatives (typically bin centers).

    Args:
        bin_edges: List of bin edges.

    Returns:
        Array of representatives.
    """
    reps: List[float] = []
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        if np.isneginf(left) and np.isposinf(right):
            reps.append(0.0)
        elif np.isneginf(left):
            reps.append(-np.inf)
        elif np.isposinf(right):
            reps.append(np.inf)
        else:
            reps.append(float(0.5 * (left + right)))
    return np.array(reps)


# pylint: disable=too-many-arguments
def maybe_return_dataframe(
    *,
    arr: np.ndarray,
    columns: Optional[List[Any]] = None,
    index: Any = None,
    pandas_module: Any = None,
    polars_module: Any = None,
    return_dataframe: Optional[str] = None,
) -> Any:
    """Return a pandas or polars DataFrame if requested, else return the array as is.

    Args:
        arr: Numpy array to convert.
        columns: Optional column names.
        index: Optional index.
        pandas_module: pandas module or None.
        polars_module: polars module or None.
        return_dataframe: "pandas", "polars", or None.

    Returns:
        DataFrame or ndarray.

    Raises:
        RuntimeError: If requested DataFrame type is not available.
    """
    if return_dataframe == "pandas":
        if pandas_module is None:
            raise RuntimeError("pandas is required for DataFrame output but is not installed.")
        return pandas_module.DataFrame(
            arr, columns=list(columns) if columns is not None else None, index=index
        )
    if return_dataframe == "polars":
        if polars_module is None:
            raise RuntimeError("polars is required for DataFrame output but is not installed.")
        return polars_module.DataFrame(arr, schema=columns)
    return arr


def lookup_edges(
    bin_edges_dict: BinEdgesDict,
    arr: np.ndarray,
    edge: str = "left",
) -> np.ndarray:
    """Vectorized lookup of bin edges for given bin indices.

    Args:
        bin_edges_dict: Dict of bin edges per column.
        arr: Array of bin indices.
        edge: "left" or "right".

    Returns:
        Array of bin edges.

    Raises:
        ValueError: If edge is not "left" or "right".
    """
    out = np.full(arr.shape, np.nan, dtype=float)
    sorted_keys = sorted(bin_edges_dict.keys(), key=str)
    for i, col_idx in enumerate(sorted_keys):
        edges = np.asarray(bin_edges_dict[col_idx])
        idx = arr[:, i]
        valid = (idx >= 0) & (idx < len(edges) - 1)
        if edge == "left":
            out[valid, i] = edges[idx[valid]]
        elif edge == "right":
            out[valid, i] = edges[idx[valid] + 1]
        else:
            raise ValueError("edge must be 'left' or 'right'")
    return out


def bin_index_masks(
    col_data: np.ndarray, n_bins: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute boolean masks for bin index interpretation.

    Args:
        col_data: 1D numpy array of bin indices.
        n_bins: Number of bins.

    Returns:
        Tuple of boolean masks:
            - mask_valid: True where bin index is valid (0 <= idx < n_bins)
            - mask_nan: True where bin index is a NaN marker
            - mask_below: True where bin index is a below-range marker (-3)
            - mask_above: True where bin index is an above-range marker (-4)
    """
    col_data = np.asarray(col_data)
    mask_valid = (col_data >= 0) & (col_data < n_bins)
    mask_below = col_data == -3
    mask_above = col_data == -4
    mask_nan = (col_data < 0) & (~mask_below) & (~mask_above)
    return mask_valid, mask_nan, mask_below, mask_above


def prepare_input_array(
    X: ArrayLike,
    require_fitted: bool = True,
    fitted_col_labels: OptListAny = None,
) -> PreparedArrayReturn:
    """Convert input data to a 2D numpy array and extract feature names and index.

    Args:
        X: Input data (numpy array, pandas DataFrame, or polars DataFrame).
        require_fitted: If True, raises ValueError if fitted_col_labels is None and
            X is not a DataFrame.
        fitted_col_labels: Column labels from a previously fitted estimator.

    Returns:
        Tuple:
            - x_array: 2D numpy array.
            - columns: List of feature names, if available.
            - index: Index of the input, if available (None for polars).
            - col_label_to_idx: Mapping from label to index.
            - col_idx_to_label: Mapping from index to label.

    Raises:
        ValueError: If require_fitted is True and neither X nor fitted_col_labels
            provide column labels.
    """
    if is_pandas_df(X):
        x_array = np.asarray(X)
        X = cast("pd.DataFrame", X)
        columns = list(X.columns)
        index = X.index
        col_label_to_idx = {label: idx for idx, label in enumerate(columns)}
        col_idx_to_label = dict(enumerate(columns))
    elif is_polars_df(X):
        X = cast("pl.DataFrame", X)
        x_array = X.to_numpy()
        columns = list(X.columns)
        index = None
        col_label_to_idx = {label: idx for idx, label in enumerate(columns)}
        col_idx_to_label = dict(enumerate(columns))
    else:
        x_array = np.asarray(X)
        index = None
        columns = None
        col_label_to_idx = None
        col_idx_to_label = None
        if fitted_col_labels is not None:
            columns = list(fitted_col_labels)
            col_label_to_idx = {label: idx for idx, label in enumerate(columns)}
            col_idx_to_label = dict(enumerate(columns))
        elif require_fitted:
            raise ValueError("Column labels are required but not available.")
    if x_array.ndim == 0:
        x_array = np.array([[x_array]])
    elif x_array.ndim == 1:
        x_array = x_array.reshape(-1, 1)
    return x_array, columns, index, col_label_to_idx, col_idx_to_label


def normalize_param_dict(
    param_dict: Optional[dict],
    col_label_to_idx: OptDictAnyInt,
    col_idx_to_label: OptDictIntAny = None,
) -> Optional[dict]:
    """Convert a user-supplied parameter dictionary to one keyed by integer column indices.

    Args:
        param_dict: The user-supplied parameter dictionary.
        col_label_to_idx: Mapping from column label to index.
        col_idx_to_label: Mapping from index to label (optional, for error checking).

    Returns:
        Dictionary keyed by integer column indices, or None if param_dict is None.

    Raises:
        KeyError: If a key cannot be mapped.
    """
    if param_dict is None or not isinstance(param_dict, dict):
        return param_dict
    if col_label_to_idx is None:
        return {int(k): v for k, v in param_dict.items()}
    out: Dict[int, Any] = {}
    for k, v in param_dict.items():
        if k in col_label_to_idx:
            idx = col_label_to_idx[k]
        elif isinstance(k, int) and (col_idx_to_label is None or k in col_idx_to_label):
            idx = k
        else:
            raise KeyError(f"Unknown column key: {k}")
        out[idx] = v
    return out


def process_interval_params(
    bin_spec: BinEdgesLike,
    bin_reps: BinEdgesLike,
) -> Tuple[OptBinEdgesDict, OptBinEdgesDict]:
    """Process and validate bin_spec and bin_reps for interval binning.

    Args:
        bin_spec: User-supplied bin specification (dict, array-like, or None).
        bin_reps: User-supplied bin representatives (dict, array-like, or None).

    Returns:
        Tuple[dict, dict or None]: (bin_spec_, bin_reps_) processed and validated.

    Raises:
        ValueError: If bin edges are not strictly increasing, too short, or reps length mismatch.
    """
    bin_spec_: OptBinEdgesDict = ensure_dict_format(bin_spec) if bin_spec is not None else None
    if bin_reps is not None:
        tmp = ensure_dict_format(bin_reps)
        bin_reps_: OptBinEdgesDict = {k: [float(x) for x in v] for k, v in tmp.items()}
    else:
        bin_reps_ = {}
    if bin_spec_ is not None:
        for col, edges in bin_spec_.items():
            if edges is None or len(edges) < 2:
                raise ValueError(
                    f"Bin edges for column {col} must have at least 2 elements "
                    "(e.g. [-np.inf, np.inf] for a single bin)"
                )
            if not all(edges[i] < edges[i + 1] for i in range(len(edges) - 1)):
                raise ValueError(f"Bin edges for column {col} must be strictly increasing")
            reps = bin_reps_.get(col)
            n_bins = len(edges) - 1
            if reps is None:
                bin_reps_[col] = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]
            elif len(reps) != n_bins:
                raise ValueError(
                    f"Number of bin representatives ({len(reps)}) does not match "
                    f"number of bins ({n_bins}) for column {col}"
                )
            bin_spec_[col] = edges
    else:
        bin_reps_ = None
    return bin_spec_, bin_reps_


def process_flexible_params(
    bin_spec: BinSpecLike,
    bin_reps: BinEdgesLike,
) -> Tuple[OptBinSpecDict, OptBinEdgesDict]:
    """Process and validate bin_spec and bin_reps for flexible binning.

    Args:
        bin_spec: User-supplied bin specification (dict, list, or None).
        bin_reps: User-supplied bin representatives (dict, array-like, or None).

    Returns:
        Tuple[dict, dict or None]: (bin_spec_, bin_reps_) processed and validated.

    Raises:
        ValueError: If bin_spec is not a non-empty list of dicts, or reps length mismatch.
    """
    bin_spec_: OptBinSpecDict = (
        ensure_dict_of_list_of_dicts_format(bin_spec) if bin_spec is not None else None
    )
    bin_reps_: OptBinEdgesDict = (
        {k: [float(x) for x in v] for k, v in ensure_dict_format(bin_reps).items()}
        if bin_reps is not None
        else {}
    )
    if bin_spec_ is not None:
        for col, bins in bin_spec_.items():
            if not isinstance(bins, list) or len(bins) == 0:
                raise ValueError(
                    f"bin_spec for column {col} must be a non-empty list of bin definitions."
                )
            reps = bin_reps_.get(col) if bin_reps_ is not None else None
            if reps is not None and len(reps) != len(bins):
                raise ValueError(
                    f"Number of bin representatives ({len(reps)}) does not match "
                    f"number of bins ({len(bins)}) for column {col}"
                )
    else:
        bin_reps_ = None
    return bin_spec_, bin_reps_


def is_pandas_df(obj: Any) -> bool:
    """Check if object is a pandas DataFrame.

    Args:
        obj: Object to check.

    Returns:
        True if obj is a pandas DataFrame, False otherwise.
    """
    pd = _pandas_config.pd
    return pd is not None and isinstance(obj, pd.DataFrame)


def is_polars_df(obj: Any) -> bool:
    """Check if object is a polars DataFrame.

    Args:
        obj: Object to check.

    Returns:
        True if obj is a polars DataFrame, False otherwise.
    """
    pl = _polars_config.pl
    return pl is not None and isinstance(obj, pl.DataFrame)
