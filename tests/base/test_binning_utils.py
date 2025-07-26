"""
Unit tests for the binning utility functions.
"""

from collections.abc import Hashable
from typing import Any, Dict, Sequence, TYPE_CHECKING, Union

import numpy as np
import pytest

from binning.base._binning_utils import (
    prepare_input_array,
    normalize_param_dict,
    process_interval_params,
    process_flexible_params,
)
from binning import (
    ensure_dict_format,
    ensure_dict_of_list_of_dicts_format,
    get_from_dict,
    has_in_dict,
    default_representatives,
    maybe_return_dataframe,
    lookup_edges,
    bin_index_masks,
)
from binning import _pandas_config, _polars_config

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

# Common type aliases for DataFrame-like objects
ArrayLike = Union[np.ndarray, "pd.DataFrame", "pl.DataFrame"]

PANDAS_AVAILABLE = _pandas_config.PANDAS_AVAILABLE
POLARS_AVAILABLE = _polars_config.POLARS_AVAILABLE


def test_ensure_dict_format_all_paths():
    """Test ensure_dict_format for all supported input types and error handling."""
    assert ensure_dict_format(None) == {}

    d: Dict[Hashable, Any] = {0: [1, 2, 3], 1: None, 2: []}
    out = ensure_dict_format(d)
    assert np.allclose(out[0], [1, 2, 3])
    assert np.allclose(out[1], [-np.inf, np.inf])
    assert np.allclose(out[2], [-np.inf, np.inf])

    arr = np.array([1, 2, 3])
    out = ensure_dict_format(arr)
    assert 0 in out and np.allclose(out[0], [1, 2, 3])

    arr = np.array([])
    out = ensure_dict_format(arr)
    assert 0 in out and np.allclose(out[0], [-np.inf, np.inf])

    arr = np.array([[1, 2], [3, 4]])
    out = ensure_dict_format(arr)
    assert 0 in out and np.allclose(out[0], [1, 2])
    assert 1 in out and np.allclose(out[1], [3, 4])

    arr = np.array([[], []])
    out = ensure_dict_format(arr)
    assert 0 in out and np.allclose(out[0], [-np.inf, np.inf])
    assert 1 in out and np.allclose(out[1], [-np.inf, np.inf])

    arr = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="Unsupported data format for bin edges/representatives."):
        ensure_dict_format(arr)


def test_ensure_dict_of_list_of_dicts_format_all_paths():
    """Test ensure_dict_of_list_of_dicts_format for all supported input types and error handling."""
    assert ensure_dict_of_list_of_dicts_format(None) is None

    d: Dict[Hashable, Any] = {0: [{"singleton": 1.0}], 1: None, 2: []}
    out = ensure_dict_of_list_of_dicts_format(d)
    assert out is not None
    assert out[0] == [{"singleton": 1.0}]
    assert out[1] == []
    assert out[2] == []

    d_bad = {0: "notalist"}
    with pytest.raises(ValueError, match="Each value in dict must be a list of dicts."):
        ensure_dict_of_list_of_dicts_format(d_bad)  # type: ignore

    l = [{"singleton": 1.0}, {"interval": (2.0, 3.0)}]
    out = ensure_dict_of_list_of_dicts_format(l)
    assert out is not None
    assert 0 in out and out[0] == l

    with pytest.raises(ValueError, match="Flexible bin spec must be a dict of lists or a list."):
        ensure_dict_of_list_of_dicts_format("notalist")  # type: ignore


def test_get_from_dict_all_paths():
    """Test get_from_dict for all lookup paths: int, str, column name, and missing."""
    assert get_from_dict(None, 0) is None

    d: Dict[Hashable, Any] = {0: "a", 1: "b"}
    assert get_from_dict(d, 0) == "a"
    assert get_from_dict(d, 1) == "b"

    d_str: Dict[Hashable, Any] = {"0": "a", "1": "b"}
    assert get_from_dict(d_str, 0) == "a"
    assert get_from_dict(d_str, 1) == "b"
    assert get_from_dict(d_str, 2) is None

    class DummyDF:
        columns = ["foo", "bar"]

    d_col: Dict[Hashable, Any] = {"foo": 123}
    assert get_from_dict(d_col, 0, X=DummyDF, col_pos=0) == 123
    assert get_from_dict(d_col, 1, X=DummyDF, col_pos=1) is None
    assert get_from_dict(d_col, 0, X=DummyDF, col_pos=None) is None


def test_has_in_dict_all_paths():
    """Test has_in_dict for all lookup paths: int, str, column name, and missing."""
    assert not has_in_dict(None, 0)

    d: Dict[Hashable, Any] = {0: "a", 1: "b"}
    assert has_in_dict(d, 0)
    assert has_in_dict(d, 1)
    assert not has_in_dict(d, 2)

    d_str: Dict[Hashable, Any] = {"0": "a", "1": "b"}
    assert has_in_dict(d_str, 0)
    assert has_in_dict(d_str, 1)
    assert not has_in_dict(d_str, 2)

    class DummyDF:
        columns = ["foo", "bar"]

    d_col: Dict[Hashable, Any] = {"foo": 123}
    assert has_in_dict(d_col, 0, X=DummyDF, col_pos=0)
    assert not has_in_dict(d_col, 1, X=DummyDF, col_pos=1)
    assert not has_in_dict(d_col, 0, X=DummyDF, col_pos=None)


def test_default_representatives_all_paths():
    """Test default_representatives for all edge cases: finite, -inf, +inf, both inf."""
    edges = np.array([0, 1, 2])
    reps = default_representatives(edges.tolist())
    assert np.allclose(reps, [0.5, 1.5])

    edges = np.array([-np.inf, 0, 1])
    reps = default_representatives(edges.tolist())
    assert np.allclose(reps, [-np.inf, 0.5])

    edges = np.array([0, 1, np.inf])
    reps = default_representatives(edges.tolist())
    assert np.allclose(reps, [0.5, np.inf])

    edges = np.array([-np.inf, np.inf])
    reps = default_representatives(edges.tolist())
    assert np.allclose(reps, [0.0])


def test_maybe_return_dataframe_numpy_paths():
    """Test maybe_return_dataframe returns ndarray by default."""
    arr = np.array([[1, 2], [3, 4]])
    columns = ["a", "b"]
    index = [10, 11]
    out = maybe_return_dataframe(arr=arr, columns=columns, index=index)
    assert isinstance(out, np.ndarray)
    assert np.all(out == arr)


def test_maybe_return_dataframe_pandas_required_but_not_installed():
    """Test maybe_return_dataframe raises if pandas output requested but pandas is not installed."""
    arr = np.array([[1, 2], [3, 4]])
    columns = ["a", "b"]
    index = [10, 11]
    with pytest.raises(RuntimeError, match="pandas is required for DataFrame output"):
        maybe_return_dataframe(
            arr=arr,
            columns=columns,
            index=index,
            pandas_module=None,
            return_dataframe="pandas",
        )


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
def test_maybe_return_dataframe_with_pandas():
    """Test maybe_return_dataframe returns a pandas DataFrame when requested."""
    import pandas as pd

    arr = np.array([[1, 2], [3, 4]])
    columns = ["a", "b"]
    index = [10, 11]
    out = maybe_return_dataframe(
        arr=arr,
        columns=columns,
        index=index,
        pandas_module=pd,
        return_dataframe="pandas",
    )
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == columns
    assert list(out.index) == index
    assert np.all(out.values == arr)


@pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
def test_prepare_input_array_polars():
    """Test prepare_input_array with polars DataFrame input."""
    import polars as pl

    df: "pl.DataFrame" = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    arr, columns, index, col_label_to_idx, col_idx_to_label = prepare_input_array(df)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2)
    assert columns == ["a", "b"]
    assert index is None
    assert col_label_to_idx == {"a": 0, "b": 1}
    assert col_idx_to_label == {0: "a", 1: "b"}


@pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
def test_maybe_return_dataframe_with_polars():
    """Test maybe_return_dataframe returns a polars DataFrame when requested."""
    import polars as pl

    arr: np.ndarray = np.array([[1, 2], [3, 4]])
    columns = ["a", "b"]
    out = maybe_return_dataframe(
        arr=arr,
        columns=columns,
        polars_module=pl,
        return_dataframe="polars",
    )
    assert isinstance(out, pl.DataFrame)
    assert out.columns == columns
    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out.to_numpy(), arr)


def test_lookup_edges_left_and_right():
    """Test lookup_edges for both left and right edges, including invalid indices and
    error handling."""
    bin_edges_dict: dict[Hashable, list[float]] = {
        0: [0.0, 1.0, 2.0, 3.0],
        1: [10.0, 20.0, 30.0, 40.0],
    }
    arr = np.array(
        [
            [0, 1],
            [2, 0],
            [-1, 2],
            [1, 3],
        ]
    )
    lefts = lookup_edges(bin_edges_dict, arr, edge="left")
    assert np.allclose(lefts[0], [0, 20])
    assert np.allclose(lefts[1], [2, 10])
    assert np.isnan(lefts[2, 0])
    assert np.allclose(lefts[2, 1], 30)
    assert np.allclose(lefts[3, 0], 1)
    assert np.isnan(lefts[3, 1])

    rights = lookup_edges(bin_edges_dict, arr, edge="right")
    assert np.allclose(rights[0], [1, 30])
    assert np.allclose(rights[1], [3, 20])
    assert np.isnan(rights[2, 0])
    assert np.allclose(rights[2, 1], 40)
    assert np.allclose(rights[3, 0], 2)
    assert np.isnan(rights[3, 1])

    with pytest.raises(ValueError, match="edge must be 'left' or 'right'"):
        lookup_edges(bin_edges_dict, arr, edge="center")


def test_bin_index_masks_all_cases():
    """Test bin_index_masks for all special codes: valid, nan, below, and above bin indices."""
    col_data = np.array([0, 1, 2, -2, -3, -4, 3, -1])
    n_bins = 3
    mask_valid, mask_nan, mask_below, mask_above = bin_index_masks(col_data, n_bins)
    assert np.all(mask_valid == np.array([True, True, True, False, False, False, False, False]))
    assert np.all(mask_nan == np.array([False, False, False, True, False, False, False, True]))
    assert np.all(mask_below == np.array([False, False, False, False, True, False, False, False]))
    assert np.all(mask_above == np.array([False, False, False, False, False, True, False, False]))


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
def test_prepare_input_array_dataframe_and_numpy():
    """Test prepare_input_array with both DataFrame and ndarray inputs."""
    import pandas as pd

    df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    arr, columns, index, col_label_to_idx, col_idx_to_label = prepare_input_array(df)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2)
    assert columns == ["a", "b"]
    assert index is not None and list(index) == [0, 1]
    assert col_label_to_idx == {"a": 0, "b": 1}
    assert col_idx_to_label == {0: "a", 1: "b"}

    arr_np = np.array([[5, 6], [7, 8]])
    arr2, columns2, index2, col_label_to_idx2, col_idx_to_label2 = prepare_input_array(
        arr_np, require_fitted=True, fitted_col_labels=["x", "y"]
    )
    assert isinstance(arr2, np.ndarray)
    assert arr2.shape == (2, 2)
    assert columns2 == ["x", "y"]
    assert index2 is None
    assert col_label_to_idx2 == {"x": 0, "y": 1}
    assert col_idx_to_label2 == {0: "x", 1: "y"}

    arr3, columns3, index3, col_label_to_idx3, col_idx_to_label3 = prepare_input_array(
        arr_np, require_fitted=False
    )
    assert isinstance(arr3, np.ndarray)
    assert arr3.shape == (2, 2)
    assert columns3 is None
    assert index3 is None
    assert col_label_to_idx3 is None
    assert col_idx_to_label3 is None

    with pytest.raises(ValueError, match="Column labels are required but not available."):
        prepare_input_array(arr_np, require_fitted=True)

    arr4, *_ = prepare_input_array(np.array(42), require_fitted=False)
    assert arr4.shape == (1, 1)

    arr5, *_ = prepare_input_array(np.array([1, 2, 3]), require_fitted=False)
    assert arr5.shape == (3, 1)


def test_normalize_param_dict_all_paths():
    """Test normalize_param_dict for all mapping and error cases."""
    d = {0: "a", 1: "b"}
    out = normalize_param_dict(d, None)
    assert out == {0: "a", 1: "b"}

    d2 = {"0": "a", "1": "b"}
    out2 = normalize_param_dict(d2, None)
    assert out2 == {0: "a", 1: "b"}

    col_label_to_idx = {"foo": 0, "bar": 1}
    d3 = {"foo": 123, "bar": 456}
    out3 = normalize_param_dict(d3, col_label_to_idx)
    assert out3 == {0: 123, 1: 456}

    d4 = {0: 111, 1: 222}
    out4 = normalize_param_dict(d4, col_label_to_idx, {0: "foo", 1: "bar"})
    assert out4 == {0: 111, 1: 222}

    d5 = {"baz": 999}
    with pytest.raises(KeyError):
        normalize_param_dict(d5, col_label_to_idx)

    assert normalize_param_dict(None, col_label_to_idx) is None


def test_process_interval_params_all_paths():
    """Test process_interval_params for all valid and error paths."""
    bin_spec: dict[Hashable, Sequence[float]] = {0: [0.0, 1.0, 2.0], 1: [10.0, 20.0, 30.0]}
    bin_reps: dict[Hashable, Sequence[float]] = {0: [0.5, 1.5], 1: [15.0, 25.0]}
    spec_out, reps_out = process_interval_params(bin_spec, bin_reps)
    assert spec_out == bin_spec
    assert reps_out == bin_reps

    bin_spec = [0.0, 1.0, 2.0]  # type: ignore
    spec_out, reps_out = process_interval_params(bin_spec, None)
    assert spec_out is not None
    assert reps_out is not None
    assert 0 in spec_out
    assert np.allclose(reps_out[0], [0.5, 1.5])

    bin_spec: dict[Hashable, Sequence[float]] = {0: [0.0, 1.0, 2.0], 1: [10.0, 20.0, 30.0]}
    bin_reps: dict[Hashable, Sequence[float]] = {0: [0.5, 1.5]}
    spec_out, reps_out = process_interval_params(bin_spec, bin_reps)
    assert reps_out is not None and np.allclose(reps_out[1], [15.0, 25.0])

    bin_spec_bad: dict[Hashable, Sequence[float]] = {0: [0.0, 2.0, 1.0]}
    with pytest.raises(ValueError, match="strictly increasing"):
        process_interval_params(bin_spec_bad, None)

    bin_spec_short: dict[Hashable, Sequence[float]] = {0: [0.0]}
    with pytest.raises(ValueError, match="at least 2 elements"):
        process_interval_params(bin_spec_short, None)

    bin_spec: dict[Hashable, Sequence[float]] = {0: [0.0, 1.0, 2.0]}
    bin_reps: dict[Hashable, Sequence[float]] = {0: [0.5]}
    with pytest.raises(ValueError, match="does not match"):
        process_interval_params(bin_spec, bin_reps)

    spec_out, reps_out = process_interval_params(None, None)
    assert spec_out is None and reps_out is None


def test_process_flexible_params_all_paths():
    """Test process_flexible_params for all valid and error paths."""
    bin_spec1: dict[Hashable, list[dict[str, int]]] | list[dict[str, int]] = {
        0: [{"a": 1}, {"b": 2}],
        1: [{"c": 3}],
    }
    bin_reps1: dict[Hashable, Sequence[float]] = {0: [1.0, 2.0], 1: [3.0]}
    spec_out, reps_out = process_flexible_params(bin_spec1, bin_reps1)
    assert spec_out == bin_spec1
    assert reps_out == bin_reps1

    bin_spec2 = [{"a": 1}, {"b": 2}]
    spec_out, reps_out = process_flexible_params(bin_spec2, None)
    assert spec_out is not None and 0 in spec_out
    assert reps_out == {}

    bin_spec3: dict[Hashable, list[dict[str, int]]] = {0: [{"a": 1}, {"b": 2}], 1: [{"c": 3}]}
    bin_reps3: dict[Hashable, Sequence[float]] = {0: [1.0, 2.0]}
    spec_out, reps_out = process_flexible_params(bin_spec3, bin_reps3)
    assert spec_out is not None and 1 in spec_out
    assert reps_out is None or 1 not in reps_out or reps_out[1] == []

    bin_spec_bad: dict[Hashable, Any] = {0: "notalist"}
    with pytest.raises(ValueError, match="Each value in dict must be a list of dicts."):
        process_flexible_params(bin_spec_bad, None)

    bin_spec4: dict[Hashable, list[dict[str, int]]] = {0: [{"a": 1}, {"b": 2}]}
    bin_reps4: dict[Hashable, Sequence[float]] = {0: [1.0]}
    with pytest.raises(ValueError, match="does not match"):
        process_flexible_params(bin_spec4, bin_reps4)

    spec_out, reps_out = process_flexible_params(None, None)
    assert spec_out is None

    bin_spec_empty: dict[Hashable, list[dict[str, int]]] = {0: []}
    with pytest.raises(
        ValueError, match="bin_spec for column 0 must be a non-empty list of bin definitions."
    ):
        process_flexible_params(bin_spec_empty, None)


def test_maybe_return_dataframe_polars_required_but_not_installed():
    """Test maybe_return_dataframe raises if polars output requested but polars is not installed."""
    arr = np.array([[1, 2], [3, 4]])
    columns = ["a", "b"]
    with pytest.raises(RuntimeError, match="polars is required for DataFrame output"):
        maybe_return_dataframe(
            arr=arr,
            columns=columns,
            polars_module=None,
            return_dataframe="polars",
        )
