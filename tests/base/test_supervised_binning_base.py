import warnings
from unittest.mock import patch

import numpy as np
import pytest

from binlearn.base._supervised_binning_base import SupervisedBinningBase
from binlearn.utils.errors import ConfigurationError, DataQualityWarning, ValidationError


class DummySupervisedBinning(SupervisedBinningBase):
    def __init__(self, task_type="classification", tree_params=None, **kwargs):
        super().__init__(task_type=task_type, tree_params=tree_params, **kwargs)

    def _fit_per_column(self, X, columns, guidance_data=None, **fit_params):
        return self

    def _fit_jointly(self, X, columns, **fit_params):
        return None

    def _transform_columns(self, X, columns):
        return np.zeros_like(X, dtype=int)

    def _inverse_transform_columns(self, X, columns):
        return np.ones_like(X, dtype=float)

    def _calculate_bins(self, x_col, col_id, guidance_data=None):
        # Return dummy bin edges and representatives
        return [0.0, 1.0, 2.0], [0.5, 1.5]


def test_init_default():
    """Test initialization with default parameters."""
    obj = DummySupervisedBinning()
    assert obj.task_type == "classification"
    assert obj.tree_params is None
    assert hasattr(obj, "_tree_template")


def test_init_classification():
    """Test initialization for classification task."""
    tree_params = {"max_depth": 5, "random_state": 42}
    obj = DummySupervisedBinning(task_type="classification", tree_params=tree_params)

    assert obj.task_type == "classification"
    assert obj.tree_params == tree_params
    # Tree template should be created with these parameters
    obj._create_tree_template()
    assert obj._tree_template is not None


def test_init_regression():
    """Test initialization for regression task."""
    obj = DummySupervisedBinning(task_type="regression")
    assert obj.task_type == "regression"


def test_init_invalid_task_type():
    """Test initialization with invalid task type."""
    with pytest.raises(ConfigurationError):  # Should raise ConfigurationError
        DummySupervisedBinning(task_type="invalid")


def test_init_tree_params_none():
    """Test initialization with tree_params=None."""
    obj = DummySupervisedBinning(tree_params=None)
    assert obj.tree_params is None
    # Tree template should still be created with defaults
    obj._create_tree_template()
    assert obj._tree_template is not None


def test_create_tree_template_early_return():
    """Test _create_tree_template early return when template already exists (line 72)."""
    obj = DummySupervisedBinning()

    # First call should create the template
    obj._create_tree_template()
    first_template = obj._tree_template
    assert first_template is not None

    # Second call should return early (line 72) without changing the template
    obj._create_tree_template()
    assert obj._tree_template is first_template  # Should be the same object


def test_validate_guidance_data_valid():
    """Test validate_guidance_data with valid data."""
    obj = DummySupervisedBinning()
    guidance_data = np.array([[1], [2], [1], [2]])

    result = obj.validate_guidance_data(guidance_data)
    np.testing.assert_array_equal(result, guidance_data.flatten())


def test_validate_guidance_data_1d():
    """Test validate_guidance_data with 1D data."""
    obj = DummySupervisedBinning()
    guidance_data = np.array([1, 2, 1, 2])

    result = obj.validate_guidance_data(guidance_data)
    np.testing.assert_array_equal(result, guidance_data)


def test_validate_guidance_data_multiple_columns():
    """Test validate_guidance_data with multiple columns."""
    obj = DummySupervisedBinning()
    guidance_data = np.array([[1, 3], [2, 4], [1, 3]])

    with pytest.raises(ValidationError):  # Should raise ValidationError
        obj.validate_guidance_data(guidance_data)


def test_validate_guidance_data_empty():
    """Test validate_guidance_data with empty data."""
    obj = DummySupervisedBinning()
    guidance_data = np.array([])

    # Empty arrays should be validated but may trigger warnings
    result = obj.validate_guidance_data(guidance_data)
    assert len(result) == 0


def test_validate_guidance_data_all_missing():
    """Test validate_guidance_data with all missing values."""
    obj = DummySupervisedBinning()
    guidance_data = np.array([np.nan, np.nan, np.nan])

    # Should validate but will trigger data quality warnings about missing values
    with pytest.warns(DataQualityWarning, match="contains 100.0% missing values"):
        result = obj.validate_guidance_data(guidance_data)
    assert len(result) == 3


def test_validate_guidance_data_with_missing():
    """Test validate_guidance_data with some missing values."""
    obj = DummySupervisedBinning()
    guidance_data = np.array([1, np.nan, 2])

    result = obj.validate_guidance_data(guidance_data)
    # Should handle missing values appropriately
    assert len(result) == 3


def test_validate_guidance_data_2d_single_column():
    """Test validate_guidance_data with 2D single column."""
    obj = DummySupervisedBinning()
    guidance_data = np.array([[1], [2], [3]])

    result = obj.validate_guidance_data(guidance_data)
    expected = np.array([1, 2, 3])
    np.testing.assert_array_equal(result, expected)


def test_validate_feature_target_pairs_valid():
    """Test validate_feature_target_pair with valid data."""
    obj = DummySupervisedBinning()
    features = np.array([1.0, 2.0, 3.0])
    targets = np.array([0, 1, 0])

    feat_clean, targ_clean, valid_mask = obj.validate_feature_target_pair(features, targets, "col1")

    assert len(feat_clean) == 3
    assert len(targ_clean) == 3
    assert np.all(valid_mask)


def test_validate_feature_target_pairs_with_missing():
    """Test validate_feature_target_pair with missing values."""
    obj = DummySupervisedBinning()
    features = np.array([1.0, np.nan, 3.0])
    targets = np.array([0, 1, np.nan])

    feat_clean, targ_clean, valid_mask = obj.validate_feature_target_pair(features, targets, "col1")

    assert len(feat_clean) == 3
    assert len(targ_clean) == 3
    # Only first element should be valid
    assert valid_mask[0]
    assert not valid_mask[1]  # features has NaN
    assert not valid_mask[2]  # targets has NaN


def test_validate_feature_target_pairs_insufficient_data():
    """Test validate_feature_target_pair with insufficient data after cleaning."""
    obj = DummySupervisedBinning()
    features = np.array([np.nan, np.nan])
    targets = np.array([np.nan, np.nan])

    # Should not raise an error - validation just creates the mask
    # But it will warn about the data quality issues
    with pytest.warns(DataQualityWarning):
        feat_clean, targ_clean, valid_mask = obj.validate_feature_target_pair(
            features, targets, "col1"
        )
    assert not np.any(valid_mask)  # No valid pairs


def test_validate_feature_target_pairs_single_class():
    """Test validate_feature_target_pair with single class."""
    obj = DummySupervisedBinning()
    features = np.array([1.0, 2.0, 3.0])
    targets = np.array([0, 0, 0])  # All same class

    # Should not raise an error during validation - that's handled elsewhere
    feat_clean, targ_clean, valid_mask = obj.validate_feature_target_pair(features, targets, "col1")
    assert np.all(valid_mask)


def test_validate_feature_target_pairs_regression_constant():
    """Test validate_feature_target_pair with constant targets in regression."""
    obj = DummySupervisedBinning(task_type="regression")
    features = np.array([1.0, 2.0, 3.0])
    targets = np.array([1.0, 1.0, 1.0])  # Constant targets

    # Should not raise an error during validation - that's handled elsewhere
    feat_clean, targ_clean, valid_mask = obj.validate_feature_target_pair(features, targets, "col1")
    assert np.all(valid_mask)


def test_create_fallback_bins_classification():
    """Test create_fallback_bins for classification."""
    obj = DummySupervisedBinning(task_type="classification")
    features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    edges, reps = obj.create_fallback_bins(features)

    assert isinstance(edges, list)
    assert isinstance(reps, list)
    assert len(edges) >= 2  # At least 2 edges for 1 bin
    assert len(reps) == len(edges) - 1  # Representatives for each bin


def test_create_fallback_bins_regression():
    """Test create_fallback_bins for regression."""
    obj = DummySupervisedBinning(task_type="regression")
    features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    edges, reps = obj.create_fallback_bins(features)

    assert isinstance(edges, list)
    assert isinstance(reps, list)
    assert len(edges) >= 2
    assert len(reps) == len(edges) - 1


def test_extract_valid_pairs():
    """Test extract_valid_pairs method."""
    obj = DummySupervisedBinning()
    features = np.array([1.0, np.nan, 3.0, 4.0])
    targets = np.array([0, 1, np.nan, 1])
    valid_mask = np.array([True, False, False, True])

    feat_valid, targ_valid = obj.extract_valid_pairs(features, targets, valid_mask)

    assert len(feat_valid) == 2
    assert len(targ_valid) == 2
    assert feat_valid[0] == 1.0
    assert feat_valid[1] == 4.0
    assert targ_valid[0] == 0
    assert targ_valid[1] == 1


def test_data_quality_warnings():
    """Test data quality warning generation."""
    obj = DummySupervisedBinning()

    # Test with features that should trigger warnings
    features = np.array([np.inf, 2.0, 3.0])
    targets = np.array([0, 1, 0])

    with patch("warnings.warn"):
        feat_clean, targ_clean, mask = obj.validate_feature_target_pair(features, targets, "col1")

        # Should handle infinite values appropriately


def test_feature_target_length_mismatch():
    """Test feature-target length mismatch handling."""
    obj = DummySupervisedBinning()
    features = np.array([1.0, 2.0])
    targets = np.array([0, 1, 0])  # Different length

    with pytest.raises(ValidationError):  # Should raise ValidationError
        obj.validate_feature_target_pair(features, targets, "col1")


def test_tree_template_types():
    """Test that correct tree types are created."""
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    clf_obj = DummySupervisedBinning(task_type="classification")
    clf_obj._create_tree_template()  # Need to call this explicitly
    assert isinstance(clf_obj._tree_template, DecisionTreeClassifier)

    reg_obj = DummySupervisedBinning(task_type="regression")
    reg_obj._create_tree_template()  # Need to call this explicitly
    assert isinstance(reg_obj._tree_template, DecisionTreeRegressor)


def test_tree_params_merging():
    """Test that tree parameters are properly stored and used."""
    custom_params = {"max_depth": 10, "min_samples_leaf": 5}
    obj = DummySupervisedBinning(tree_params=custom_params)

    # Test that custom parameters were stored
    assert obj.tree_params is not None
    assert obj.tree_params["max_depth"] == 10
    assert obj.tree_params["min_samples_leaf"] == 5

    # Test that tree template is created successfully with these parameters
    obj._create_tree_template()
    assert obj._tree_template is not None


def test_guidance_data_validation_name():
    """Test validate_guidance_data with custom name."""
    obj = DummySupervisedBinning()
    guidance_data = np.array([[1, 2]])  # Invalid: multiple columns

    with pytest.raises(ValidationError):  # Should mention custom name in error
        obj.validate_guidance_data(guidance_data, name="custom_target")


def test_edge_case_single_sample():
    """Test behavior with single sample."""
    obj = DummySupervisedBinning()
    features = np.array([1.0])
    targets = np.array([0])

    # Should return results but may have limited validity
    result = obj.validate_feature_target_pair(features, targets, "col1")
    assert len(result) == 3  # Should return (features, targets, valid_mask)
    features_out, targets_out, valid_mask = result
    assert len(features_out) == 1
    assert len(targets_out) == 1
    assert len(valid_mask) == 1


def test_edge_case_all_nan_features():
    """Test behavior when all features are NaN."""
    obj = DummySupervisedBinning()
    features = np.array([np.nan, np.nan, np.nan])
    targets = np.array([0, 1, 0])

    # Should return results but all features will be marked as invalid
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = obj.validate_feature_target_pair(features, targets, "col1")

        # Should have issued a warning about missing values
        assert len(w) > 0
        assert any("missing values" in str(warning.message) for warning in w)

    features_out, targets_out, valid_mask = result
    assert len(features_out) == 3
    assert len(targets_out) == 3
    assert len(valid_mask) == 3
    # All features should be marked as invalid due to NaN
    assert not valid_mask.any()


def test_validate_guidance_data_wrong_dimensions():
    """Test validate_guidance_data with wrong dimensions."""
    obj = DummySupervisedBinning()

    # Test 3D array - should raise ValidationError
    guidance_3d = np.random.rand(2, 3, 4)
    with pytest.raises(ValidationError, match="has 3 dimensions"):
        obj.validate_guidance_data(guidance_3d, "test_data")


def test_validate_guidance_data_wrong_2d_shape():
    """Test validate_guidance_data with 2D array having multiple columns."""
    obj = DummySupervisedBinning()

    # Test 2D array with multiple columns - should raise ValidationError
    guidance_2d_multi = np.random.rand(10, 3)
    with pytest.raises(ValidationError, match="Please specify the correct guidance column"):
        obj.validate_guidance_data(guidance_2d_multi, "test_data")


def test_validate_guidance_data_none():
    """Test require_guidance_data with None guidance_data."""
    obj = DummySupervisedBinning()

    with pytest.raises(ValueError, match="requires guidance_data"):
        obj.require_guidance_data(None, "test_data")


def test_validate_task_type_invalid():
    """Test validate_task_type with invalid task type."""
    obj = DummySupervisedBinning()

    with pytest.raises(ValueError, match="task_type.*not supported"):
        obj.validate_task_type("invalid_task", ["classification", "regression"])


def test_validate_task_type_valid():
    """Test validate_task_type with valid task type."""
    obj = DummySupervisedBinning()

    # Should not raise any exception
    obj.validate_task_type("classification", ["classification", "regression"])
    obj.validate_task_type("regression", ["classification", "regression"])


def test_handle_insufficient_data():
    """Test handle_insufficient_data method."""
    obj = DummySupervisedBinning()

    # Test with insufficient data
    x_col = np.array([1.0, 2.0])
    valid_mask = np.array([False, False])  # No valid data
    min_samples = 5

    with pytest.warns(DataQualityWarning, match="has no valid data points"):
        result = obj.handle_insufficient_data(x_col, valid_mask, min_samples, col_id="test_col")
    # Should return some fallback bins
    assert result is not None


def test_create_fallback_bins():
    """Test create_fallback_bins method."""
    obj = DummySupervisedBinning()

    # Test fallback bin creation
    x_col = np.array([1.0, 2.0, 3.0])

    result = obj.create_fallback_bins(x_col)
    # Should return fallback bins
    assert result is not None
    edges, reps = result
    assert len(edges) >= 2  # At least 2 edges for 1 bin


def test_validate_feature_target_pairs_length_mismatch():
    """Test validate_feature_target_pair with length mismatch."""
    obj = DummySupervisedBinning()

    features = np.array([1.0, 2.0])
    targets = np.array([0, 1, 0])  # Different length

    with pytest.raises(ValidationError, match="Both must have the same number of samples"):
        obj.validate_feature_target_pair(features, targets, "col1")


def test_validate_feature_target_pairs_insufficient_finite_data():
    """Test validate_feature_target_pair with insufficient finite data."""
    obj = DummySupervisedBinning()

    # Most values are NaN/inf
    features = np.array([np.nan, np.inf, 1.0])
    targets = np.array([0, 1, 0])

    # Should warn about infinite values
    with pytest.warns(DataQualityWarning, match="contains infinite values"):
        feat_clean, targ_clean, mask = obj.validate_feature_target_pair(features, targets, "col1")
    # At least we get some output, the exact behavior depends on implementation


def test_validate_feature_target_pair_object_dtype():
    """Test validate_feature_target_pair with object dtype guidance data to cover line 174."""
    obj = DummySupervisedBinning()

    # Create guidance data with object dtype (mixed types)
    features = np.array([1.0, 2.0, 3.0, 4.0])
    targets = np.array(["A", "B", None, "A"], dtype=object)  # Object dtype with None

    feat_clean, targ_clean, mask = obj.validate_feature_target_pair(features, targets, "col1")

    # Should handle object dtype correctly
    assert len(feat_clean) == len(features)
    assert len(targ_clean) == len(targets)
    assert isinstance(mask, np.ndarray)


def test_handle_insufficient_data_string_column():
    """Test handle_insufficient_data with string and integer column IDs to cover lines 422-435."""
    obj = DummySupervisedBinning()

    # Test with string column ID - should trigger the else branch (f"column '{col_id}'")
    x_col = np.array([np.nan, np.nan, np.nan])
    valid_mask = np.array([False, False, False])

    with pytest.warns(
        DataQualityWarning, match="Data in column 'string_col' has no valid data points"
    ):
        result = obj.handle_insufficient_data(x_col, valid_mask, min_samples=2, col_id="string_col")

    # Should return fallback bins
    assert result is not None
    edges, reps = result
    assert len(edges) == 2
    assert len(reps) == 1

    # Test with Python integer column ID - should trigger the if branch (f"column {col_id}")
    with pytest.warns(DataQualityWarning, match="Data in column 42 has no valid data points"):
        result = obj.handle_insufficient_data(x_col, valid_mask, min_samples=2, col_id=42)

    # Should return fallback bins
    assert result is not None
    edges, reps = result
    assert len(edges) == 2
    assert len(reps) == 1

    # Test with numpy integer column ID to cover the np.integer type check
    np_int_col_id = np.int32(99)
    with pytest.warns(DataQualityWarning, match="Data in column 99 has no valid data points"):
        result = obj.handle_insufficient_data(
            x_col, valid_mask, min_samples=2, col_id=np_int_col_id
        )

    # Should return fallback bins
    assert result is not None
    edges, reps = result
    assert len(edges) == 2
    assert len(reps) == 1

    # Test with col_id=None (no warning issued, but completes branch coverage)
    result = obj.handle_insufficient_data(x_col, valid_mask, min_samples=2, col_id=None)
    assert result is not None
    edges, reps = result
    assert len(edges) == 2
    assert len(reps) == 1
    edges, reps = result
    assert len(edges) == 2
    assert len(reps) == 1


def test_handle_insufficient_data_min_samples():
    """Test handle_insufficient_data with min_samples threshold to cover lines 446-458."""
    obj = DummySupervisedBinning()

    # Only 2 valid samples, but need minimum 5 - test with string column ID
    x_col = np.array([1.0, 2.0, np.nan, np.nan, np.nan])
    valid_mask = np.array([True, True, False, False, False])

    with pytest.warns(
        DataQualityWarning,
        match="Data in column 'test_col' has only 2 valid samples.*minimum 5 required",
    ):
        result = obj.handle_insufficient_data(x_col, valid_mask, min_samples=5, col_id="test_col")

    # Should return fallback bins due to insufficient samples
    assert result is not None
    edges, reps = result
    assert len(edges) == 2
    assert len(reps) == 1

    # Test with Python integer column ID to cover the integer branch
    with pytest.warns(
        DataQualityWarning, match="Data in column 123 has only 2 valid samples.*minimum 5 required"
    ):
        result = obj.handle_insufficient_data(x_col, valid_mask, min_samples=5, col_id=123)

    # Test with numpy integer column ID to cover the np.integer branch
    np_int_col_id = np.int64(456)
    with pytest.warns(
        DataQualityWarning, match="Data in column 456 has only 2 valid samples.*minimum 5 required"
    ):
        result = obj.handle_insufficient_data(
            x_col, valid_mask, min_samples=5, col_id=np_int_col_id
        )

    # Test with col_id=None (no warning issued, but completes branch coverage)
    result = obj.handle_insufficient_data(x_col, valid_mask, min_samples=5, col_id=None)
    assert result is not None
    edges, reps = result
    assert len(edges) == 2
    assert len(reps) == 1

    # Should return fallback bins due to insufficient samples
    assert result is not None
    edges, reps = result
    assert len(edges) == 2
    assert len(reps) == 1


def test_handle_insufficient_data_sufficient_samples():
    """Test handle_insufficient_data when there ARE sufficient samples (should return None)."""
    obj = DummySupervisedBinning()

    # Test with sufficient samples - should return None to continue normal processing
    x_col = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    valid_mask = np.array([True, True, True, True, True])

    result = obj.handle_insufficient_data(x_col, valid_mask, min_samples=3, col_id=0)

    # Should return None since we have sufficient data (5 >= 3)
    assert result is None


def test_create_fallback_bins_with_default_range():
    """Test create_fallback_bins with default_range to cover line 347."""
    obj = DummySupervisedBinning()

    x_col = np.array([1.0, 2.0, 3.0])
    default_range = (10.0, 20.0)

    edges, reps = obj.create_fallback_bins(x_col, default_range)

    assert edges == [10.0, 20.0]
    assert reps == [15.0]


def test_create_fallback_bins_no_finite_data():
    """Test create_fallback_bins with no finite data to cover line 355."""
    obj = DummySupervisedBinning()

    # All infinite or NaN data
    x_col = np.array([np.inf, np.nan, -np.inf])

    edges, reps = obj.create_fallback_bins(x_col)

    # Should use default 0.0, 1.0 range
    assert edges == [0.0, 1.0]
    assert reps == [0.5]


def test_create_fallback_bins_same_min_max():
    """Test create_fallback_bins when min equals max to cover line 358."""
    obj = DummySupervisedBinning()

    # All same value
    x_col = np.array([5.0, 5.0, 5.0])

    edges, reps = obj.create_fallback_bins(x_col)

    # Should adjust max to be min + 1
    assert edges == [5.0, 6.0]
    assert reps == [5.5]


def test_handle_bin_params():
    """Test _handle_bin_params with automatic discovery."""
    obj = DummySupervisedBinning()

    # Test updating task_type and tree_params
    params = {
        "task_type": "regression",
        "tree_params": {"max_depth": 10},
    }

    reset_fitted = obj._handle_bin_params(params)

    # Should return True due to parameter changes
    assert reset_fitted
    assert obj.task_type == "regression"
    assert obj.tree_params == {"max_depth": 10}
    assert obj.task_type == "regression"
    assert obj.tree_params == {"max_depth": 10}


def test_dummy_methods_coverage():
    """Test the dummy methods to get coverage on lines 12, 15, 18, 21, 25."""
    obj = DummySupervisedBinning()

    # Test _fit_per_column (line 12)
    X = np.array([[1, 2], [3, 4]])
    result = obj._fit_per_column(X, [0, 1])
    assert result is obj

    # Test _fit_jointly (line 15)
    result = obj._fit_jointly(X, [0, 1])
    assert result is None

    # Test _transform_columns (line 18)
    result = obj._transform_columns(X, [0, 1])
    expected = np.zeros_like(X, dtype=int)
    np.testing.assert_array_equal(result, expected)

    # Test _inverse_transform_columns (line 21)
    result = obj._inverse_transform_columns(X, [0, 1])
    expected = np.ones_like(X, dtype=float)
    np.testing.assert_array_equal(result, expected)

    # Test _calculate_bins (line 25)
    x_col = np.array([1.0, 2.0, 3.0])
    edges, reps = obj._calculate_bins(x_col, 0)
    assert edges == [0.0, 1.0, 2.0]
    assert reps == [0.5, 1.5]


def test_create_tree_template_invalid_params():
    """Test _create_tree_template with invalid tree_params that cause TypeError."""
    from binlearn.utils.errors import ConfigurationError

    # Create object with invalid tree parameters that will cause TypeError
    obj = DummySupervisedBinning(task_type="classification")
    obj.tree_params = {"invalid_param": "invalid_value", "max_depth": "not_an_int"}

    with pytest.raises(ConfigurationError, match="Invalid tree_params"):
        obj._create_tree_template()
