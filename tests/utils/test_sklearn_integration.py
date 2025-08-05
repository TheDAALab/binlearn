"""Tests for sklearn_integration module."""

# pylint: disable=protected-access
from typing import Any
from unittest.mock import Mock

import pytest

from binlearn.utils.sklearn_integration import SklearnCompatibilityMixin


class MockTransformer(SklearnCompatibilityMixin):
    """Mock transformer class for testing sklearn integration."""

    def __init__(self, guidance_columns: Any = None) -> None:
        super().__init__()
        self._fitted = False
        self.guidance_columns = guidance_columns
        # Add _n_features_in for backward compatibility testing
        self._n_features_in: int = 0

    def fit(self, X: Any, y: Any = None) -> "MockTransformer":  # pylint: disable=unused-argument
        """Mock fit method."""
        self._fitted = True
        if hasattr(X, "shape"):
            self.n_features_in_ = X.shape[1]
        else:
            self.n_features_in_ = len(X[0]) if X else 0
        return self


class TestSklearnCompatibilityMixin:
    """Test SklearnCompatibilityMixin."""

    def test_more_tags(self) -> None:
        """Test _more_tags method."""
        transformer = MockTransformer()
        tags = transformer._more_tags()

        assert isinstance(tags, dict)
        assert tags["requires_fit"] is True
        assert tags["allow_nan"] is True

    def test_check_feature_names_with_pandas_like(self) -> None:
        """Test _check_feature_names with pandas-like object."""
        transformer = MockTransformer()

        # Mock object with columns attribute
        mock_x = Mock()
        mock_x.columns = ["col1", "col2", "col3"]
        mock_x.shape = (100, 3)

        result = transformer._check_feature_names(mock_x)

        assert result == ["col1", "col2", "col3"]
        assert transformer.feature_names_in_ == ["col1", "col2", "col3"]

    def test_check_feature_names_with_feature_names_attribute(self) -> None:
        """Test _check_feature_names with object having feature_names attribute."""
        transformer = MockTransformer()

        # Mock object with feature_names attribute (but no columns)
        mock_x = Mock()
        mock_x.feature_names = ["feat1", "feat2", "feat3"]
        mock_x.shape = (100, 3)
        # Ensure it doesn't have columns attribute
        del mock_x.columns

        result = transformer._check_feature_names(mock_x)

        assert result == ["feat1", "feat2", "feat3"]
        assert transformer.feature_names_in_ == ["feat1", "feat2", "feat3"]

    def test_check_feature_names_with_private_feature_names_attribute(self) -> None:
        """Test _check_feature_names with object having _feature_names attribute."""
        transformer = MockTransformer()

        # Mock object with _feature_names attribute (but no columns or feature_names)
        mock_x = Mock()
        mock_x._feature_names = ["private_feat1", "private_feat2"]
        mock_x.shape = (100, 2)
        # Ensure it doesn't have columns or feature_names attributes
        del mock_x.columns
        del mock_x.feature_names

        result = transformer._check_feature_names(mock_x)

        assert result == ["private_feat1", "private_feat2"]
        assert transformer.feature_names_in_ == ["private_feat1", "private_feat2"]

    def test_check_feature_names_with_numpy_array(self) -> None:
        """Test _check_feature_names with numpy array."""
        transformer = MockTransformer()
        x = [[1, 2, 3], [4, 5, 6]]

        result = transformer._check_feature_names(x)

        assert result == ["feature_0", "feature_1", "feature_2"]

    def test_get_feature_names_out_not_fitted(self) -> None:
        """Test get_feature_names_out raises error when not fitted."""
        transformer = MockTransformer()

        with pytest.raises(ValueError, match="not fitted yet"):
            transformer.get_feature_names_out()

    def test_get_feature_names_out_with_input_features(self) -> None:
        """Test get_feature_names_out with explicit input features."""
        transformer = MockTransformer()
        transformer._fitted = True

        result = transformer.get_feature_names_out(["a", "b", "c"])

        assert result == ["a", "b", "c"]

    def test_get_feature_names_out_fitted_with_feature_names_in(self) -> None:
        """Test get_feature_names_out when fitted with feature_names_in_."""
        transformer = MockTransformer()
        transformer._fitted = True
        transformer.feature_names_in_ = ["feature1", "feature2"]

        result = transformer.get_feature_names_out()

        assert result == ["feature1", "feature2"]

    def test_get_feature_names_out_with_n_features_in(self) -> None:
        """Test get_feature_names_out with n_features_in_."""
        transformer = MockTransformer()
        transformer._fitted = True
        transformer.n_features_in_ = 2

        result = transformer.get_feature_names_out()

        assert result == ["x0", "x1"]

    def test_get_feature_names_out_with_underscore_n_features_in(self) -> None:
        """Test get_feature_names_out with _n_features_in."""
        transformer = MockTransformer()
        transformer._fitted = True
        # Add _n_features_in as a dynamic attribute for backward compatibility
        transformer._n_features_in = 2

        result = transformer.get_feature_names_out()

        assert result == ["x0", "x1"]

    def test_get_feature_names_out_with_guidance_columns_list(self) -> None:
        """Test get_feature_names_out with guidance_columns as list."""
        transformer = MockTransformer(guidance_columns=["col2", "col4"])
        transformer._fitted = True
        transformer.feature_names_in_ = ["col1", "col2", "col3", "col4"]

        result = transformer.get_feature_names_out()

        # Should exclude guidance columns
        assert result == ["col1", "col3"]

    def test_get_feature_names_out_with_guidance_columns_indices(self) -> None:
        """Test get_feature_names_out with guidance_columns as indices."""
        transformer = MockTransformer(guidance_columns=[1, 3])
        transformer._fitted = True
        transformer.feature_names_in_ = ["col1", "col2", "col3", "col4"]

        result = transformer.get_feature_names_out()

        # Should exclude columns at indices 1 and 3
        assert result == ["col1", "col3"]

    def test_get_feature_names_out_with_guidance_columns_non_list(self) -> None:
        """Test get_feature_names_out with guidance_columns not as a list."""
        transformer = MockTransformer(guidance_columns="col2")
        transformer._fitted = True
        transformer.feature_names_in_ = ["col1", "col2", "col3", "col4"]

        result = transformer.get_feature_names_out()

        # Should exclude the single guidance column
        assert result == ["col1", "col3", "col4"]

    def test_validate_params(self) -> None:
        """Test _validate_params method."""
        transformer = MockTransformer()
        # Should not raise any errors for base implementation
        transformer._validate_params()
