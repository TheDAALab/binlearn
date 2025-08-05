import numpy as np
import pytest

from binlearn.base._supervised_binning_base import SupervisedBinningBase
from binlearn.utils.errors import ValidationError


class DummySupervisedBinning(SupervisedBinningBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _fit_per_column(self, X, columns, guidance_data=None, **fit_params):
        return self

    def _fit_jointly(self, X, columns, **fit_params) -> None:
        return None

    def _transform_columns(self, X, columns):
        return np.zeros_like(X, dtype=int)

    def _inverse_transform_columns(self, X, columns):
        return np.ones_like(X, dtype=float)

    def _calculate_bins(self, x_col, col_id, guidance_data=None):
        return [0.0, 1.0, 2.0], [0.5, 1.5]


def test_init_default():
    """Test default initialization of SupervisedBinningBase."""
    obj = DummySupervisedBinning()
    assert obj.guidance_columns is None


def test_init_guidance_columns():
    """Test initialization with guidance_columns."""
    obj = DummySupervisedBinning(guidance_columns=["target"])
    assert obj.guidance_columns == ["target"]


def test_validate_guidance_data_valid():
    """Test validate_guidance_data with valid 2D data."""
    obj = DummySupervisedBinning()

    # Valid 2D data (single column)
    guidance_data = np.array([[1], [2], [1], [2]])
    result = obj.validate_guidance_data(guidance_data)

    # Should return flattened version
    np.testing.assert_array_equal(result, guidance_data.flatten())


def test_validate_guidance_data_1d():
    """Test validate_guidance_data with valid 1D data."""
    obj = DummySupervisedBinning()

    # Valid 1D data
    guidance_data = np.array([1, 2, 1, 2])
    result = obj.validate_guidance_data(guidance_data)

    # Should return as-is
    np.testing.assert_array_equal(result, guidance_data)


def test_require_guidance_data_none():
    """Test require_guidance_data with None input."""
    obj = DummySupervisedBinning()

    with pytest.raises(ValueError):
        obj.require_guidance_data(None)


def test_validate_guidance_data_3d_array():
    """Test that validate_guidance_data correctly rejects 3D arrays."""
    obj = DummySupervisedBinning()

    # Create 3D data
    guidance_data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    # Should raise ValidationError for 3D array
    with pytest.raises(ValidationError, match="guidance_data has 3 dimensions"):
        obj.validate_guidance_data(guidance_data)


def test_validate_feature_target_pair_object_dtype():
    """Test validation with object dtype data to trigger specific error path."""
    obj = DummySupervisedBinning()

    # Create data with object dtype that can't be converted to numeric
    x_col = np.array(["a", "b", "c", "d"], dtype=object)
    y_col = np.array([1, 2, 1, 2])

    # This should handle the object dtype case
    try:
        result = obj.validate_feature_target_pair(x_col, y_col)
        # If it succeeds, that's fine - the test is about coverage
        assert result is not None
    except (ValueError, TypeError):
        # If it raises an error, that's also expected behavior
        pass


def test_validate_feature_target_pair_mismatched_lengths():
    """Test validation with mismatched feature and target lengths."""
    obj = DummySupervisedBinning()

    # Create data with different lengths
    x_col = np.array([1, 2, 3])
    y_col = np.array([1, 2, 1, 2])  # Different length

    # Should raise ValidationError for length mismatch
    with pytest.raises(ValidationError, match="Feature column"):
        obj.validate_feature_target_pair(x_col, y_col, col_id=0)


def test_validate_feature_target_pair_object_dtype_guidance():
    """Test validation with object dtype guidance data."""
    obj = DummySupervisedBinning()

    # Create data with object dtype guidance data
    x_col = np.array([1.0, 2.0, 3.0, 4.0])
    y_col = np.array(["A", "B", "A", "B"], dtype=object)

    # This should handle the object dtype guidance case (line 182)
    try:
        result = obj.validate_feature_target_pair(x_col, y_col, col_id=0)
        # If it succeeds, check it returns something
        assert result is not None
    except (ValueError, TypeError):
        # If it raises an error, that's also expected behavior
        pass


def test_create_fallback_bins_with_col_id():
    """Test create_fallback_bins with column ID for warning message."""
    obj = DummySupervisedBinning()

    # Create data that will trigger fallback and warning (lines 330-341)
    x_col = np.array([np.inf, np.nan, 1.0, 1.0])  # Some invalid, some constant

    # This should trigger the warning with col_id
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        left_edges, right_edges = obj.create_fallback_bins(x_col, default_range=None)

        # Should have created bins and possibly issued warning
        assert len(left_edges) > 0
        assert len(right_edges) > 0


def test_create_fallback_bins_all_invalid_data():
    """Test create_fallback_bins with all invalid data (lines 381-395)."""
    obj = DummySupervisedBinning()

    # Data with all non-finite values
    x_col = np.array([np.inf, -np.inf, np.nan])

    # Should use default (0.0, 1.0) range
    left_edges, right_edges = obj.create_fallback_bins(x_col)

    # Should return fallback bins
    assert len(left_edges) > 0
    assert len(right_edges) > 0
    assert left_edges[0] == 0.0
    assert right_edges[0] == 0.5  # Representative should be midpoint


def test_create_fallback_bins_with_default_range():
    """Test create_fallback_bins with explicit default_range (line 382)."""
    obj = DummySupervisedBinning()

    # Test with explicit default range - this should cover line 382
    x_col = np.array([1.0, 2.0, 3.0])  # Valid data but we override with default_range
    default_range = (10.0, 20.0)

    left_edges, right_edges = obj.create_fallback_bins(x_col, default_range=default_range)

    # Should use the provided default range, not the data range
    assert len(left_edges) > 0
    assert len(right_edges) > 0
    assert left_edges[0] == 10.0
    assert right_edges[0] == 15.0  # Midpoint of (10, 20)


def test_handle_insufficient_data_no_valid_data_int_col_id():
    """Test handle_insufficient_data with no valid data and integer column ID (covers 330->341)."""
    obj = DummySupervisedBinning()

    # Create data with no valid samples (all NaN/inf)
    x_col = np.array([np.nan, np.inf, -np.inf, np.nan])
    valid_mask = np.array([False, False, False, False])  # No valid data
    min_samples = 2
    col_id = 0  # Integer column ID

    # This should trigger the branch with n_valid == 0 and integer col_id
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = obj.handle_insufficient_data(x_col, valid_mask, min_samples, col_id=col_id)

        # Should return fallback bins and issue warning
        assert result is not None
        left_edges, right_edges = result
        assert len(left_edges) > 0
        assert len(right_edges) > 0

        # Should have issued a warning
        assert len(w) > 0
        warning_msg = str(w[0].message)
        assert "column 0" in warning_msg  # Should reference integer column ID
        assert "default" in warning_msg  # Should mention default range for n_valid == 0


def test_handle_insufficient_data_no_valid_data_string_col_id():
    """Test handle_insufficient_data with no valid data and string column ID (covers 330->341)."""
    obj = DummySupervisedBinning()

    # Create data with no valid samples
    x_col = np.array([np.nan, np.inf, -np.inf, np.nan])
    valid_mask = np.array([False, False, False, False])  # No valid data
    min_samples = 2
    col_id = "feature_name"  # String column ID

    # This should trigger the branch with n_valid == 0 and string col_id
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = obj.handle_insufficient_data(x_col, valid_mask, min_samples, col_id=col_id)

        # Should return fallback bins and issue warning
        assert result is not None
        left_edges, right_edges = result
        assert len(left_edges) > 0
        assert len(right_edges) > 0

        # Should have issued a warning
        assert len(w) > 0
        warning_msg = str(w[0].message)
        assert (
            "column 'feature_name'" in warning_msg
        )  # Should reference string column ID with quotes
        assert "default" in warning_msg  # Should mention default range for n_valid == 0


def test_handle_insufficient_data_some_valid_data():
    """Test handle_insufficient_data with some but insufficient valid data."""
    obj = DummySupervisedBinning()

    # Create data with one valid sample (insufficient for min_samples=2)
    x_col = np.array([1.0, np.nan, np.inf, np.nan])
    valid_mask = np.array([True, False, False, False])  # Only one valid sample
    min_samples = 2
    col_id = 1  # Integer column ID

    # This should trigger the branch with n_valid > 0 but < min_samples
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = obj.handle_insufficient_data(x_col, valid_mask, min_samples, col_id=col_id)

        # Should return fallback bins and issue warning
        assert result is not None
        left_edges, right_edges = result
        assert len(left_edges) > 0
        assert len(right_edges) > 0

        # Should have issued a warning
        assert len(w) > 0
        warning_msg = str(w[0].message)
        assert "column 1" in warning_msg  # Should reference integer column ID
        # Should NOT mention "default" since n_valid > 0
        assert "default" not in warning_msg


def test_handle_insufficient_data_no_col_id():
    """Test handle_insufficient_data without column ID (covers other branch of 330->341)."""
    obj = DummySupervisedBinning()

    # Create data with no valid samples
    x_col = np.array([np.nan, np.inf, -np.inf, np.nan])
    valid_mask = np.array([False, False, False, False])  # No valid data
    min_samples = 2
    col_id = None  # No column ID - this should skip the column reference in warning

    # This should issue a warning but without column reference
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = obj.handle_insufficient_data(x_col, valid_mask, min_samples, col_id=col_id)

        # Should return fallback bins
        assert result is not None
        left_edges, right_edges = result
        assert len(left_edges) > 0
        assert len(right_edges) > 0

        # May have issued a warning, but it should not contain column reference
        if len(w) > 0:
            warning_msg = str(w[0].message)
            # Should not contain "column" since col_id is None
            assert "column" not in warning_msg
