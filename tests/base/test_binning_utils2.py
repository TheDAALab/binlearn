"""
Minimal, comprehensive tests for all functions in _binning_utils.py.
Each function and code path is covered, including pandas/polars branches.
"""

import numpy as np
import pytest
from collections.abc import Hashable
from typing import Any, Dict, List, Sequence, TYPE_CHECKING, Union

from binning.base._binning_utils import (
    ensure_dict_format,
    ensure_dict_of_list_of_dicts_format,
    get_from_dict,
    has_in_dict,
    default_representatives,
    maybe_return_dataframe,
    lookup_edges,
    bin_index_masks,
    prepare_input_array,
    normalize_param_dict,
    process_interval_params,
    process_flexible_params,
    is_pandas_df,
    is_polars_df,
)
from binning import _pandas_config, _polars_config

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

ArrayLike = Union[np.ndarray, "pd.DataFrame", "pl.DataFrame"]
PANDAS_AVAILABLE = _pandas_config.PANDAS_AVAILABLE
POLARS_AVAILABLE = _polars_config.POLARS_AVAILABLE


def test_ensure_dict_format_and_of_list_of_dicts():
    # None, dict, 1d, 2d, empty, error
    assert ensure_dict_format(None) == {}
    assert ensure_dict_format({0: [1, 2]}) == {0: [1, 2]}
    assert ensure_dict_format(np.array([1, 2])) == {0: [1, 2]}
    assert ensure_dict_format(np.array([[1, 2], [3, 4]])) == {0: [1, 2], 1: [3, 4]}
    assert ensure_dict_format(np.array([])) == {0: [-np.inf, np.inf]}
    with pytest.raises(ValueError):
        ensure_dict_format(np.zeros((2, 2, 2)))
    # of_list_of_dicts: None, dict, list, error
    assert ensure_dict_of_list_of_dicts_format(None) is None
    assert ensure_dict_of_list_of_dicts_format({0: [{"a": 1}]}) == {0: [{"a": 1}]}
    assert ensure_dict_of_list_of_dicts_format([{"a": 1}]) == {0: [{"a": 1}]}
    with pytest.raises(ValueError):
        ensure_dict_of_list_of_dicts_format({0: "bad"})
    with pytest.raises(ValueError):
        ensure_dict_of_list_of_dicts_format("bad")  # type: ignore


def test_get_and_has_from_dict():
    d = {0: "a", "1": "b", "foo": 123}

    class DummyDF:
        columns = ["foo", "bar"]

    assert get_from_dict(d, 0) == "a"
    assert get_from_dict(d, 1) == "b"
    assert get_from_dict(d, 2) is None
    assert get_from_dict(d, 0, X=DummyDF, col_pos=2) is None
    assert get_from_dict(d, 0, X=DummyDF, col_pos=0) == 123
    assert has_in_dict(d, 0)
    assert has_in_dict(d, 1)
    assert not has_in_dict(d, 2)
    assert has_in_dict(d, 0, X=DummyDF, col_pos=0)
    assert not has_in_dict(d, 1, X=DummyDF, col_pos=1)


def test_default_representatives():
    assert np.allclose(default_representatives([0, 1, 2]), [0.5, 1.5])
    assert np.allclose(default_representatives([-np.inf, 0, 1]), [-np.inf, 0.5])
    assert np.allclose(default_representatives([0, 1, np.inf]), [0.5, np.inf])
    assert np.allclose(default_representatives([-np.inf, np.inf]), [0.0])


def test_maybe_return_dataframe_numpy():
    arr = np.array([[1, 2]])
    assert np.all(maybe_return_dataframe(arr=arr) == arr)


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
def test_maybe_return_dataframe_pandas():
    import pandas as pd

    arr = np.array([[1, 2]])
    df = maybe_return_dataframe(
        arr=arr, columns=["a", "b"], index=[10], pandas_module=pd, return_dataframe="pandas"
    )
    assert hasattr(df, "values") and list(df.columns) == ["a", "b"]


@pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
def test_maybe_return_dataframe_polars():
    import polars as pl

    arr = np.array([[1, 2]])
    df = maybe_return_dataframe(
        arr=arr, columns=["a", "b"], polars_module=pl, return_dataframe="polars"
    )
    assert hasattr(df, "to_numpy") and df.columns == ["a", "b"]


def test_lookup_edges_and_bin_index_masks():
    bed = {0: [0, 1, 2], 1: [10, 20, 30]}
    arr = np.array([[0, 1], [1, 0], [-1, 2]])
    left = lookup_edges(bed, arr, edge="left")
    right = lookup_edges(bed, arr, edge="right")
    assert left.shape == arr.shape and right.shape == arr.shape
    with pytest.raises(ValueError):
        lookup_edges(bed, arr, edge="bad")
    col_data = np.array([0, 1, -3, -4, -2])
    mask_valid, mask_nan, mask_below, mask_above = bin_index_masks(col_data, 2)
    assert (
        mask_valid.sum() == 2
        and mask_nan.sum() == 1
        and mask_below.sum() == 1
        and mask_above.sum() == 1
    )


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
def test_prepare_input_array_pandas():
    import pandas as pd

    df = pd.DataFrame([[1, 2]], columns=["a", "b"])
    arr, columns, index, col_label_to_idx, col_idx_to_label = prepare_input_array(df)
    assert arr.shape == (1, 2) and columns == ["a", "b"] and index is not None


@pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
def test_prepare_input_array_polars():
    import polars as pl

    df = pl.DataFrame({"a": [1], "b": [2]})
    arr, columns, index, col_label_to_idx, col_idx_to_label = prepare_input_array(df)
    assert arr.shape == (1, 2) and columns == ["a", "b"] and index is None


def test_prepare_input_array_numpy_and_errors():
    arr = np.array([[1, 2]])
    arr2, columns, index, col_label_to_idx, col_idx_to_label = prepare_input_array(
        arr, require_fitted=True, fitted_col_labels=["x", "y"]
    )
    assert arr2.shape == (1, 2) and columns == ["x", "y"]
    arr3, *_ = prepare_input_array(np.array(42), require_fitted=False)
    assert arr3.shape == (1, 1)
    with pytest.raises(ValueError):
        prepare_input_array(np.array([[1, 2]]), require_fitted=True)


def test_normalize_param_dict():
    d = {0: "a", 1: "b"}
    assert normalize_param_dict(d, None) == d
    col_label_to_idx = {"foo": 0, "bar": 1}
    d2 = {"foo": 123, "bar": 456}
    assert normalize_param_dict(d2, col_label_to_idx) == {0: 123, 1: 456}
    with pytest.raises(KeyError):
        normalize_param_dict({"baz": 999}, col_label_to_idx)
    assert normalize_param_dict(None, col_label_to_idx) is None


def test_process_interval_params_and_flexible_params():
    # interval: valid, errors
    spec, reps = process_interval_params({0: [0, 1, 2]}, {0: [0.5, 1.5]})
    assert spec == {0: [0, 1, 2]} and reps == {0: [0.5, 1.5]}
    spec, reps = process_interval_params([0, 1, 2], None)
    assert 0 in spec and reps is not None
    with pytest.raises(ValueError):
        process_interval_params({0: [0, 2, 1]}, None)
    with pytest.raises(ValueError):
        process_interval_params({0: [0]}, None)
    with pytest.raises(ValueError):
        process_interval_params({0: [0, 1, 2]}, {0: [0.5]})
    assert process_interval_params(None, None) == (None, None)
    # flexible: valid, errors
    spec, reps = process_flexible_params({0: [{"a": 1}, {"b": 2}]}, {0: [1.0, 2.0]})
    assert spec == {0: [{"a": 1}, {"b": 2}]} and reps == {0: [1.0, 2.0]}
    spec, reps = process_flexible_params([{"a": 1}], None)
    assert 0 in spec and reps == {}
    with pytest.raises(ValueError):
        process_flexible_params({0: "bad"}, None)
    with pytest.raises(ValueError):
        process_flexible_params({0: [{"a": 1}]}, {0: [1.0]})
    assert process_flexible_params(None, None) == (None, None)
    with pytest.raises(ValueError):
        process_flexible_params({0: []}, None)


def test_is_pandas_df_and_is_polars_df():
    arr = np.array([1, 2])
    assert not is_pandas_df(arr)
    assert not is_polars_df(arr)
    if PANDAS_AVAILABLE:
        import pandas as pd

        df = pd.DataFrame([[1, 2]])
        assert is_pandas_df(df)
    if POLARS_AVAILABLE:
        import polars as pl

        df = pl.DataFrame({"a": [1, 2]})
        assert is_polars_df(df)
