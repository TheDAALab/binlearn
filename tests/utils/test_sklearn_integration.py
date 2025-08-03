"""Tests for sklearn_integration module."""

# pylint: disable=protected-access
from unittest.mock import Mock

import pytest

from binlearn.utils.sklearn_integration import SklearnCompatibilityMixin


class TestSklearnCompatibilityMixin:
    """Test SklearnCompatibilityMixin."""

    def test_more_tags(self):
        """Test _more_tags method."""
        mixin = SklearnCompatibilityMixin()
        tags = mixin._more_tags()

        assert isinstance(tags, dict)
        assert tags["requires_fit"] is True
        assert tags["requires_y"] is False
        assert tags["allow_nan"] is True
        assert tags["stateless"] is False
        assert "2darray" in tags["X_types"]

    def test_check_feature_names_with_columns(self):
        """Test _check_feature_names with DataFrame-like object."""
        mixin = SklearnCompatibilityMixin()

        # Mock object with columns
        mock_X = Mock()
        mock_X.columns = ["col1", "col2", "col3"]

        result = mixin._check_feature_names(mock_X, reset=True)

        assert result == ["col1", "col2", "col3"]
        assert hasattr(mixin, "feature_names_in_")
        assert mixin.feature_names_in_ == ["col1", "col2", "col3"]

    def test_check_feature_names_with_feature_names_attr(self):
        """Test _check_feature_names with feature_names attribute."""
        mixin = SklearnCompatibilityMixin()

        # Mock object with feature_names
        mock_X = Mock()
        del mock_X.columns  # Remove columns attribute
        mock_X.feature_names = ["feat1", "feat2"]

        result = mixin._check_feature_names(mock_X, reset=True)

        assert result == ["feat1", "feat2"]
        assert mixin.feature_names_in_ == ["feat1", "feat2"]

    def test_check_feature_names_with_underscore_feature_names_attr(self):
        """Test _check_feature_names with _feature_names attribute."""
        mixin = SklearnCompatibilityMixin()

        # Mock object with _feature_names
        mock_X = Mock()
        del mock_X.columns
        del mock_X.feature_names
        mock_X._feature_names = ["_feat1", "_feat2", "_feat3"]

        result = mixin._check_feature_names(mock_X, reset=True)

        assert result == ["_feat1", "_feat2", "_feat3"]
        assert mixin.feature_names_in_ == ["_feat1", "_feat2", "_feat3"]

    def test_check_feature_names_with_generic_array(self):
        """Test _check_feature_names with generic array (no feature names)."""
        mixin = SklearnCompatibilityMixin()

        # Mock object with shape but no feature names
        mock_X = Mock()
        mock_X.shape = (100, 4)
        del mock_X.columns
        del mock_X.feature_names
        del mock_X._feature_names

        result = mixin._check_feature_names(mock_X, reset=True)

        expected = ["feature_0", "feature_1", "feature_2", "feature_3"]
        assert result == expected
        assert mixin.feature_names_in_ == expected

    def test_check_feature_names_with_list_like(self):
        """Test _check_feature_names with list-like object."""
        mixin = SklearnCompatibilityMixin()

        # Mock object without shape but with length
        mock_X = [["a", "b", "c"], ["d", "e", "f"]]

        result = mixin._check_feature_names(mock_X, reset=True)

        expected = ["feature_0", "feature_1", "feature_2"]
        assert result == expected
        assert mixin.feature_names_in_ == expected

    def test_check_feature_names_no_reset(self):
        """Test _check_feature_names without reset when feature_names_in_ exists."""
        mixin = SklearnCompatibilityMixin()
        mixin.feature_names_in_ = ["existing1", "existing2"]

        mock_X = Mock()
        mock_X.columns = ["new1", "new2", "new3"]

        result = mixin._check_feature_names(mock_X, reset=False)

        # Should return new feature names but keep existing feature_names_in_
        assert result == ["new1", "new2", "new3"]
        assert mixin.feature_names_in_ == ["existing1", "existing2"]

    def test_get_feature_names_out_not_fitted(self):
        """Test get_feature_names_out when not fitted."""
        mixin = SklearnCompatibilityMixin()

        with pytest.raises(ValueError, match="This estimator is not fitted yet"):
            mixin.get_feature_names_out()

    def test_get_feature_names_out_with_input_features(self):
        """Test get_feature_names_out with explicit input_features."""
        mixin = SklearnCompatibilityMixin()
        mixin._fitted = True

        input_features = ["in1", "in2", "in3"]
        result = mixin.get_feature_names_out(input_features)

        assert result == input_features

    def test_get_feature_names_out_with_stored_feature_names(self):
        """Test get_feature_names_out with stored feature_names_in_."""
        mixin = SklearnCompatibilityMixin()
        mixin._fitted = True
        mixin.feature_names_in_ = ["stored1", "stored2"]

        result = mixin.get_feature_names_out()

        assert result == ["stored1", "stored2"]

    def test_get_feature_names_out_with_n_features_in(self):
        """Test get_feature_names_out with n_features_in_."""
        mixin = SklearnCompatibilityMixin()
        mixin._fitted = True
        mixin.n_features_in_ = 3

        result = mixin.get_feature_names_out()

        assert result == ["x0", "x1", "x2"]

    def test_get_feature_names_out_with_underscore_n_features_in(self):
        """Test get_feature_names_out with _n_features_in."""
        mixin = SklearnCompatibilityMixin()
        mixin._fitted = True
        mixin._n_features_in = 2

        result = mixin.get_feature_names_out()

        assert result == ["x0", "x1"]

    def test_get_feature_names_out_with_guidance_columns_list(self):
        """Test get_feature_names_out with guidance_columns as list."""
        mixin = SklearnCompatibilityMixin()
        mixin._fitted = True
        mixin.feature_names_in_ = ["col1", "col2", "col3", "col4"]
        mixin.guidance_columns = ["col2", "col4"]

        result = mixin.get_feature_names_out()

        # Should exclude guidance columns
        assert result == ["col1", "col3"]

    def test_get_feature_names_out_with_guidance_columns_indices(self):
        """Test get_feature_names_out with guidance_columns as indices."""
        mixin = SklearnCompatibilityMixin()
        mixin._fitted = True
        mixin.feature_names_in_ = ["col1", "col2", "col3", "col4"]
        mixin.guidance_columns = [1, 3]

        result = mixin.get_feature_names_out()

        # Should exclude columns at indices 1 and 3
        assert result == ["col1", "col3"]

    def test_get_feature_names_out_with_guidance_columns_non_list(self):
        """Test get_feature_names_out with guidance_columns not as a list."""
        mixin = SklearnCompatibilityMixin()
        mixin._fitted = True
        mixin.feature_names_in_ = ["col1", "col2", "col3", "col4"]
        mixin.guidance_columns = "col2"  # Single string, not a list

        result = mixin.get_feature_names_out()

        # Should exclude the single guidance column
        assert result == ["col1", "col3", "col4"]

    def test_validate_params(self):
        """Test _validate_params method."""
        mixin = SklearnCompatibilityMixin()
        # Should not raise any errors for base implementation
        mixin._validate_params()
