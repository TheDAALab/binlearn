"""
Comprehensive test suite for GeneralBinningBase abstract base class.

This module tests the core functionality of the GeneralBinningBase class,
which provides the foundation for all binning transformers in the package.
It covers initialization, configuration handling, abstract method implementation,
guidance column processing, fitting workflows, and error handling.

Classes:
    DummyGeneralBinning: Concrete implementation of GeneralBinningBase for testing.

Functions:
    test_*: Individual test functions covering various aspects of the base class
    functionality including initialization, parameter validation, fitting,
    transformation, guidance handling, and sklearn compatibility.
"""

from unittest.mock import patch

import numpy as np
import pytest

from binlearn import PANDAS_AVAILABLE, pd
from binlearn.base._general_binning_base import GeneralBinningBase
from binlearn.utils.errors import BinningError


class DummyGeneralBinning(GeneralBinningBase):
    """Concrete implementation of GeneralBinningBase for testing purposes.

    This dummy class implements all abstract methods of GeneralBinningBase
    with minimal functionality to enable testing of the base class behavior
    without requiring a full binning implementation.
    """

    def __init__(self, preserve_dataframe=None, fit_jointly=None, guidance_columns=None, **kwargs):
        super().__init__(preserve_dataframe, fit_jointly, guidance_columns, **kwargs)

    def _fit_per_column(self, X, columns, guidance_data=None, **fit_params):
        return self

    def _fit_jointly(self, X, columns, **fit_params):
        return None

    def _transform_columns(self, X, columns):
        return np.zeros_like(X, dtype=int)

    def _inverse_transform_columns(self, X, columns):
        return np.ones_like(X, dtype=float)


def test_init_default_config():
    """Test initialization with default configuration values.

    Verifies that the GeneralBinningBase correctly loads default values
    from the global configuration when no explicit parameters are provided.
    """
    obj = DummyGeneralBinning()
    assert obj.preserve_dataframe is not None  # Should get from config
    assert obj.fit_jointly is not None  # Should get from config
    assert obj.guidance_columns is None
    assert obj._fitted is False


def test_init_explicit_params():
    """Test initialization with explicit parameters.

    Verifies that explicitly provided parameters override default
    configuration values and are stored correctly in the instance.
    """
    obj = DummyGeneralBinning(preserve_dataframe=True, fit_jointly=False, guidance_columns=["col1"])
    assert obj.preserve_dataframe is True
    assert obj.fit_jointly is False
    assert obj.guidance_columns == ["col1"]


def test_init_incompatible_params():
    """Test initialization with incompatible parameter combinations.

    Verifies that the base class correctly rejects incompatible parameter
    combinations like guidance_columns with fit_jointly=True.
    """
    with pytest.raises(ValueError, match="guidance_columns and fit_jointly=True are incompatible"):
        DummyGeneralBinning(guidance_columns=["col1"], fit_jointly=True)


def test_prepare_input():
    """Test _prepare_input method."""
    obj = DummyGeneralBinning()
    X = np.array([[1, 2], [3, 4]])
    arr, columns = obj._prepare_input(X)
    assert isinstance(arr, np.ndarray)
    assert isinstance(columns, list)


def test_check_fitted():
    """Test _check_fitted method."""
    obj = DummyGeneralBinning()
    with pytest.raises(RuntimeError, match="This estimator is not fitted yet"):
        obj._check_fitted()

    obj._fitted = True
    obj._check_fitted()  # Should not raise


def test_separate_columns_no_guidance():
    """Test _separate_columns with no guidance columns."""
    obj = DummyGeneralBinning()
    obj._fitted = True

    X = np.array([[1, 2, 3], [4, 5, 6]])
    X_bin, X_guide, bin_cols, guide_cols = obj._separate_columns(X)

    assert X_bin.shape == (2, 3)
    assert X_guide is None
    assert bin_cols == [0, 1, 2]
    assert guide_cols is None


def test_separate_columns_with_guidance():
    """Test _separate_columns with guidance columns."""
    obj = DummyGeneralBinning(guidance_columns=[1])
    obj._fitted = True

    X = np.array([[1, 2, 3], [4, 5, 6]])
    X_bin, X_guide, bin_cols, guide_cols = obj._separate_columns(X)

    assert X_bin.shape == (2, 2)  # Columns 0 and 2
    assert X_guide is not None
    assert X_guide.shape == (2, 1)  # Column 1
    assert bin_cols == [0, 2]
    assert guide_cols == [1]


def test_separate_columns_guidance_list():
    """Test _separate_columns with guidance columns as list."""
    obj = DummyGeneralBinning(guidance_columns=[0, 2])
    obj._fitted = True

    X = np.array([[1, 2, 3], [4, 5, 6]])
    X_bin, X_guide, bin_cols, guide_cols = obj._separate_columns(X)

    assert X_bin.shape == (2, 1)  # Column 1
    assert X_guide is not None
    assert X_guide.shape == (2, 2)  # Columns 0 and 2
    assert bin_cols == [1]
    assert guide_cols == [0, 2]


def test_fit_per_column():
    """Test fit method with per-column fitting."""
    obj = DummyGeneralBinning(fit_jointly=False)
    X = np.array([[1, 2], [3, 4]])

    result = obj.fit(X)
    assert result is obj
    assert obj._fitted is True
    assert obj._n_features_in == 2


def test_fit_jointly():
    """Test fit method with joint fitting."""
    obj = DummyGeneralBinning(fit_jointly=True)
    X = np.array([[1, 2], [3, 4]])

    result = obj.fit(X)
    assert result is obj
    assert obj._fitted is True


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
def test_fit_with_dataframe():
    """Test fit method with pandas DataFrame."""
    obj = DummyGeneralBinning()
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])

    _ = obj.fit(df)
    assert obj._feature_names_in == ["A", "B"]


def test_fit_with_feature_names():
    """Test fit with object having feature_names attribute."""
    obj = DummyGeneralBinning()

    class MockArray:
        def __init__(self):
            self.shape = (2, 2)
            self.feature_names = ["feat1", "feat2"]

        def __array__(self):
            return np.array([[1, 2], [3, 4]])

    X = MockArray()
    obj.fit(X)
    assert obj._feature_names_in == ["feat1", "feat2"]


def test_fit_numpy_array():
    """Test fit with numpy array (no column names)."""
    obj = DummyGeneralBinning()
    X = np.array([[1, 2], [3, 4]])

    obj.fit(X)
    assert obj._feature_names_in == [0, 1]


def test_fit_error_handling():
    """Test fit method error handling."""

    class ErrorBinning(DummyGeneralBinning):
        def _fit_per_column(self, X, columns, guidance_data=None, **fit_params):
            raise ValueError("Test error")

    obj = ErrorBinning()
    X = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError, match="Test error"):
        obj.fit(X)


def test_fit_generic_error():
    """Test fit method with generic error handling."""

    class ErrorBinning(DummyGeneralBinning):
        def _fit_per_column(self, X, columns, guidance_data=None, **fit_params):
            # Throw a generic exception that should be wrapped
            raise OSError("Generic OS error")

    obj = ErrorBinning()
    X = np.array([[1, 2], [3, 4]])

    # The OSError should be wrapped in ValueError
    with pytest.raises(ValueError, match="Failed to fit binning model"):
        obj.fit(X)


def test_fit_binning_error():
    """Test fit method re-raises BinningError unchanged."""

    class BinningErrorBinning(DummyGeneralBinning):
        def _fit_per_column(self, X, columns, guidance_data=None, **fit_params):
            # Throw a BinningError that should be re-raised unchanged
            raise BinningError("Specific binning error")

    obj = BinningErrorBinning()
    X = np.array([[1, 2], [3, 4]])

    # BinningError should be re-raised unchanged
    with pytest.raises(BinningError, match="Specific binning error"):
        obj.fit(X)


def test_transform():
    """Test transform method."""
    obj = DummyGeneralBinning()
    X = np.array([[1, 2], [3, 4]])
    obj.fit(X)

    result = obj.transform(X)
    assert result.shape == (2, 2)
    assert (result == 0).all()


def test_transform_with_guidance():
    """Test transform with guidance columns."""
    obj = DummyGeneralBinning(guidance_columns=[1])
    X = np.array([[1, 2], [3, 4]])
    obj.fit(X)

    result = obj.transform(X)
    assert result.shape == (2, 1)  # Only binning columns


def test_transform_not_fitted():
    """Test transform when not fitted."""
    obj = DummyGeneralBinning()
    X = np.array([[1, 2], [3, 4]])

    with pytest.raises(RuntimeError):
        obj.transform(X)


def test_transform_error_handling():
    """Test transform error handling."""

    class ErrorBinning(DummyGeneralBinning):
        def _transform_columns(self, X, columns):
            raise Exception("Transform error")

    obj = ErrorBinning()
    X = np.array([[1, 2], [3, 4]])
    obj.fit(X)

    with pytest.raises(ValueError, match="Failed to transform data"):
        obj.transform(X)


def test_inverse_transform():
    """Test inverse_transform method."""
    obj = DummyGeneralBinning()
    X = np.array([[1, 2], [3, 4]])
    obj.fit(X)

    X_binned = obj.transform(X)
    result = obj.inverse_transform(X_binned)
    assert result.shape == (2, 2)
    assert (result == 1).all()


def test_inverse_transform_with_guidance():
    """Test inverse_transform with guidance columns."""
    obj = DummyGeneralBinning(guidance_columns=[1])
    X = np.array([[1, 2], [3, 4]])
    obj.fit(X)

    X_binned = obj.transform(X)  # Shape (2, 1)
    result = obj.inverse_transform(X_binned)
    assert result.shape == (2, 1)


def test_inverse_transform_wrong_columns():
    """Test inverse_transform with wrong number of columns."""
    obj = DummyGeneralBinning(guidance_columns=[1])
    X = np.array([[1, 2], [3, 4]])
    obj.fit(X)

    # Try to inverse transform with wrong number of columns
    wrong_X = np.array([[0, 0], [1, 1]])  # 2 columns instead of 1
    with pytest.raises(ValueError, match="Input for inverse_transform should have"):
        obj.inverse_transform(wrong_X)


def test_inverse_transform_not_fitted():
    """Test inverse_transform when not fitted."""
    obj = DummyGeneralBinning()
    X = np.array([[1, 2], [3, 4]])

    with pytest.raises(RuntimeError):
        obj.inverse_transform(X)


def test_inverse_transform_error_handling():
    """Test inverse_transform error handling."""

    class ErrorBinning(DummyGeneralBinning):
        def _inverse_transform_columns(self, X, columns):
            raise Exception("Inverse transform error")

    obj = ErrorBinning()
    X = np.array([[1, 2], [3, 4]])
    obj.fit(X)

    X_binned = obj.transform(X)
    with pytest.raises(ValueError, match="Failed to inverse transform data"):
        obj.inverse_transform(X_binned)


def test_get_params():
    """Test get_params method."""
    obj = DummyGeneralBinning(preserve_dataframe=True)
    params = obj.get_params()

    assert "preserve_dataframe" in params
    assert params["preserve_dataframe"] is True


def test_get_params_missing_attributes():
    """Test get_params when object is missing some class-specific attributes."""
    from unittest.mock import patch

    obj = DummyGeneralBinning()

    # Mock safe_get_class_parameters to return a parameter that doesn't exist on the object
    with patch("binlearn.base._general_binning_base.safe_get_class_parameters") as mock_params:
        mock_params.return_value = ["preserve_dataframe", "nonexistent_param"]

        params = obj.get_params()

        # Should include preserve_dataframe (it exists) but not nonexistent_param
        assert "preserve_dataframe" in params
        assert "nonexistent_param" not in params


def test_set_params():
    """Test set_params method."""
    obj = DummyGeneralBinning(preserve_dataframe=True)
    result = obj.set_params(preserve_dataframe=False)

    assert result is obj
    assert obj.preserve_dataframe is False


def test_set_params_incompatible():
    """Test set_params with incompatible parameters."""
    obj = DummyGeneralBinning()

    with pytest.raises(ValueError, match="guidance_columns and fit_jointly=True are incompatible"):
        obj.set_params(guidance_columns=["col1"], fit_jointly=True)


def test_validate_params():
    """Test _validate_params method."""
    obj = DummyGeneralBinning()
    obj._validate_params()  # Should not raise

    # Test invalid preserve_dataframe
    obj.preserve_dataframe = "invalid"  # Bypass type checking for test
    with pytest.raises(TypeError, match="preserve_dataframe must be a boolean"):
        obj._validate_params()

    # Test invalid fit_jointly
    obj.preserve_dataframe = True
    obj.fit_jointly = "invalid"  # Bypass type checking for test
    with pytest.raises(TypeError, match="fit_jointly must be a boolean"):
        obj._validate_params()

    # Test invalid guidance_columns
    obj.fit_jointly = True
    obj.guidance_columns = 123.45  # Invalid type
    with pytest.raises(TypeError, match="guidance_columns must be list, tuple, int, str"):
        obj._validate_params()


def test_get_fitted_params():
    """Test _get_fitted_params method."""
    obj = DummyGeneralBinning()
    params = obj._get_fitted_params()
    assert isinstance(params, dict)


def test_get_fitted_params_missing_attributes():
    """Test _get_fitted_params when object is missing some expected attributes."""
    from unittest.mock import patch

    obj = DummyGeneralBinning()

    # Mock safe_get_class_parameters to return a parameter that doesn't exist on the object
    with patch("binlearn.base._general_binning_base.safe_get_class_parameters") as mock_params:
        mock_params.return_value = ["preserve_dataframe", "nonexistent_fitted_param"]

        params = obj._get_fitted_params()

        # Should include existing attributes but not nonexistent ones
        assert isinstance(params, dict)
        assert "nonexistent_fitted_param" not in params


def test_handle_bin_params():
    """Test _handle_bin_params method."""
    obj = DummyGeneralBinning()
    reset = obj._handle_bin_params({})
    assert isinstance(reset, bool)


def test_empty_binning_columns():
    """Test handling of empty binning columns."""
    obj = DummyGeneralBinning(guidance_columns=[0, 1])
    obj._fitted = True
    obj._original_columns = [0, 1]

    X = np.array([[1, 2], [3, 4]])
    X_bin, X_guide, bin_cols, guide_cols = obj._separate_columns(X)

    assert X_bin.shape == (2, 0)  # No binning columns
    assert X_guide is not None
    assert X_guide.shape == (2, 2)  # All guidance
    assert bin_cols == []
    assert guide_cols == [0, 1]


def test_transform_empty_binning_columns():
    """Test transform with no binning columns."""
    obj = DummyGeneralBinning(guidance_columns=[0, 1])
    X = np.array([[1, 2], [3, 4]])
    obj.fit(X)

    result = obj.transform(X)
    assert result.shape == (2, 0)  # No columns to transform


@patch("binlearn.base._general_binning_base.prepare_input_with_columns")
def test_prepare_input_mock(mock_prepare):
    """Test _prepare_input calls correct function."""
    mock_prepare.return_value = (np.array([[1, 2]]), [0, 1])

    obj = DummyGeneralBinning()
    X = np.array([[1, 2]])

    arr, cols = obj._prepare_input(X)

    mock_prepare.assert_called_once_with(X, fitted=False, original_columns=None)
    assert arr.shape == (1, 2)
    assert cols == [0, 1]


def test_fitted_params_override():
    """Test that fitted parameters override constructor params in get_params."""
    binning = DummyGeneralBinning()
    X = np.array([[1], [2], [3]])

    # Mock _get_fitted_params to return specific fitted values
    fitted_params = {"some_fitted_param": "fitted_value"}
    with patch.object(binning, "_get_fitted_params", return_value=fitted_params):
        # Fit the binning
        binning.fit(X)

        # Get params after fitting - should include fitted params
        params_after = binning.get_params()
        assert "some_fitted_param" in params_after
        assert params_after["some_fitted_param"] == "fitted_value"


def test_set_params_resets_fitted():
    """Test that set_params resets fitted state when _handle_bin_params returns True."""
    binning = DummyGeneralBinning()
    X = np.array([[1], [2], [3]])

    # Fit the binning first
    binning.fit(X)
    assert binning._fitted is True

    # Mock _handle_bin_params to return True (indicating reset needed)
    with patch.object(binning, "_handle_bin_params", return_value=True):
        binning.set_params(fit_jointly=True)  # Use valid parameter
        assert binning._fitted is False


def test_get_params_with_fitted_attributes():
    """Test get_params method includes fitted attributes when they exist."""
    obj = DummyGeneralBinning()

    # Mark object as fitted so _get_fitted_params() is called
    obj._fitted = True

    # Add some fitted attributes to test lines 294-298
    obj.bin_spec_ = {0: [1]}  # New simplified format
    obj.bin_representatives_ = {0: [1.0]}
    obj.bin_edges_ = {0: [0, 1, 2]}

    params = obj.get_params()

    # Should include fitted parameters without trailing underscores
    assert "bin_spec" in params
    assert "bin_representatives" in params
    assert "bin_edges" in params
    assert params["bin_spec"] == {0: [1]}  # New simplified format
    assert params["bin_representatives"] == {0: [1.0]}
    assert params["bin_edges"] == {0: [0, 1, 2]}


def test_get_params_with_none_fitted_attributes():
    """Test get_params method skips None fitted attributes."""
    obj = DummyGeneralBinning()

    # Mark object as fitted so _get_fitted_params() is called
    obj._fitted = True

    # Add fitted attributes with None values
    obj.bin_spec_ = None
    obj.bin_representatives_ = {0: [1.0]}  # This one is not None

    params = obj.get_params()

    # Should include non-None fitted parameters
    assert "bin_representatives" in params
    assert params["bin_representatives"] == {0: [1.0]}

    # Should not include None fitted parameters
    assert "bin_spec" not in params or params.get("bin_spec") is None
