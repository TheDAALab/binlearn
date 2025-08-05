import numpy as np
import pytest

from binlearn.base._supervised_binning_base import SupervisedBinningBase


class DummySupervisedBinning(SupervisedBinningBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _fit_per_column(self, X, columns, guidance_data=None, **fit_params) -> None:
        return self

    def _fit_jointly(self, X, columns, **fit_params) -> None:
        return None

    def _transform_columns(self, X, columns) -> None:
        return np.zeros_like(X, dtype=int)

    def _inverse_transform_columns(self, X, columns) -> None:
        return np.ones_like(X, dtype=float)

    def _calculate_bins(self, x_col, col_id, guidance_data=None) -> None:
        # Return dummy bin edges and representatives
        return [0.0, 1.0, 2.0], [0.5, 1.5]


def test_init_default() -> None:
    """Test initialization with default parameters."""
    obj = DummySupervisedBinning()
    # Base class should not have task_type or tree_params
    assert not hasattr(obj, "task_type")
    assert not hasattr(obj, "tree_params")
    assert not hasattr(obj, "_tree_template")


def test_init_guidance_columns() -> None:
    """Test initialization with guidance columns."""
    obj = DummySupervisedBinning(guidance_columns=[2])
    assert obj.guidance_columns  # type: ignore[attr-defined] == [2]


def test_validate_guidance_data_valid() -> None:
    """Test validate_guidance_data with valid data."""
    obj = DummySupervisedBinning()
    guidance_data = np.array([[1], [2], [1], [2]])

    result = obj.validate_guidance_data(guidance_data)
    np.testing.assert_array_equal(result, guidance_data.flatten())


def test_validate_guidance_data_1d() -> None:
    """Test validate_guidance_data with 1D data."""
    obj = DummySupervisedBinning()
    guidance_data = np.array([1, 2, 1, 2])

    result = obj.validate_guidance_data(guidance_data)
    np.testing.assert_array_equal(result, guidance_data)


def test_require_guidance_data_none() -> None:
    """Test require_guidance_data with None (should raise ValueError)."""
    obj = DummySupervisedBinning()

    with pytest.raises(ValueError):
        obj.require_guidance_data(None, "test method")
