"""Tests for sklearn_integration module."""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from binning.utils.sklearn_integration import (
    SklearnCompatibilityMixin,
    BinningFeatureSelector,
    BinningPipeline,
    make_binning_scorer
)


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
        assert "sparse" in tags["X_types"]

    def test_check_feature_names_with_columns(self):
        """Test _check_feature_names with DataFrame-like object."""
        mixin = SklearnCompatibilityMixin()

        # Mock object with columns
        mock_X = Mock()
        mock_X.columns = ['col1', 'col2', 'col3']

        result = mixin._check_feature_names(mock_X, reset=True)

        assert result == ['col1', 'col2', 'col3']
        assert hasattr(mixin, 'feature_names_in_')
        assert mixin.feature_names_in_ == ['col1', 'col2', 'col3']

    def test_check_feature_names_with_feature_names_attr(self):
        """Test _check_feature_names with feature_names attribute."""
        mixin = SklearnCompatibilityMixin()

        # Mock object with feature_names
        mock_X = Mock()
        del mock_X.columns  # Remove columns attribute
        mock_X.feature_names = ['feat1', 'feat2']

        result = mixin._check_feature_names(mock_X, reset=True)

        assert result == ['feat1', 'feat2']
        assert mixin.feature_names_in_ == ['feat1', 'feat2']

    def test_check_feature_names_with_shape(self):
        """Test _check_feature_names with shape attribute."""
        mixin = SklearnCompatibilityMixin()

        # Mock object with shape
        mock_X = Mock()
        del mock_X.columns
        del mock_X.feature_names
        mock_X.shape = (100, 5)

        result = mixin._check_feature_names(mock_X, reset=True)

        expected = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']
        assert result == expected
        assert mixin.feature_names_in_ == expected

    def test_check_feature_names_with_list_shape(self):
        """Test _check_feature_names with list-like object."""
        mixin = SklearnCompatibilityMixin()

        # List of lists
        X = [[1, 2, 3], [4, 5, 6]]

        result = mixin._check_feature_names(X, reset=True)

        expected = ['feature_0', 'feature_1', 'feature_2']
        assert result == expected

    def test_check_feature_names_no_reset(self):
        """Test _check_feature_names without reset."""
        mixin = SklearnCompatibilityMixin()
        mixin.feature_names_in_ = ['existing1', 'existing2']

        mock_X = Mock()
        mock_X.columns = ['new1', 'new2', 'new3']

        result = mixin._check_feature_names(mock_X, reset=False)

        # Should not change existing feature names
        assert mixin.feature_names_in_ == ['existing1', 'existing2']
        assert result == ['new1', 'new2', 'new3']

    def test_get_feature_names_out_not_fitted(self):
        """Test get_feature_names_out when not fitted."""
        mixin = SklearnCompatibilityMixin()

        with pytest.raises(ValueError, match="This estimator is not fitted yet"):
            mixin.get_feature_names_out()

    def test_get_feature_names_out_fitted(self):
        """Test get_feature_names_out when fitted."""
        mixin = SklearnCompatibilityMixin()
        setattr(mixin, "_fitted", True)
        mixin.feature_names_in_ = ['col1', 'col2', 'col3']

        result = mixin.get_feature_names_out()

        assert result == ['col1', 'col2', 'col3']

    def test_get_feature_names_out_with_input_features(self):
        """Test get_feature_names_out with provided input features."""
        mixin = SklearnCompatibilityMixin()
        setattr(mixin, "_fitted", True)

        input_features = ['input1', 'input2']
        result = mixin.get_feature_names_out(input_features)

        assert result == ['input1', 'input2']

    def test_get_feature_names_out_with_guidance_columns(self):
        """Test get_feature_names_out with guidance columns."""
        mixin = SklearnCompatibilityMixin()
        setattr(mixin, "_fitted", True)
        mixin.feature_names_in_ = ['col1', 'col2', 'col3', 'col4']
        setattr(mixin, "guidance_columns", ['col2', 'col4'])

        result = mixin.get_feature_names_out()

        # Should exclude guidance columns
        assert result == ['col1', 'col3']

    def test_get_feature_names_out_with_guidance_columns_indices(self):
        """Test get_feature_names_out with guidance column indices."""
        mixin = SklearnCompatibilityMixin()
        setattr(mixin, "_fitted", True)
        mixin.feature_names_in_ = ['col1', 'col2', 'col3', 'col4']
        setattr(mixin, "guidance_columns", [1, 3])

        result = mixin.get_feature_names_out()

        # Should exclude guidance columns by index
        assert result == ['col1', 'col3']

    def test_get_feature_names_out_no_stored_features(self):
        """Test get_feature_names_out without stored feature names."""
        mixin = SklearnCompatibilityMixin()
        setattr(mixin, "_fitted", True)
        setattr(mixin, "_n_features_in", 3)

        result = mixin.get_feature_names_out()

        assert result == ['x0', 'x1', 'x2']

    def test_get_feature_names_out_with_n_features_in_attr(self):
        """Test get_feature_names_out with _n_features_in attribute."""
        mixin = SklearnCompatibilityMixin()
        setattr(mixin, "_fitted", True)
        setattr(mixin, "_n_features_in", 2)

        result = mixin.get_feature_names_out()

        assert result == ['x0', 'x1']

    def test_validate_params(self):
        """Test _validate_params method."""
        mixin = SklearnCompatibilityMixin()
        # Should not raise any errors for base implementation
        mixin._validate_params()


class TestBinningFeatureSelector:
    """Test BinningFeatureSelector."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        selector = BinningFeatureSelector()

        assert selector.binning_method == "equal_width"
        assert selector.k == 10
        assert selector.score_func == "auto"
        assert selector.binning_params == {}

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        binning_params = {'n_bins': 5}
        selector = BinningFeatureSelector(
            binning_method="supervised",
            k=15,
            score_func="mutual_info_classif",
            binning_params=binning_params
        )

        assert selector.binning_method == "supervised"
        assert selector.k == 15
        assert selector.score_func == "mutual_info_classif"
        assert selector.binning_params == binning_params

    @patch('binning.utils.sklearn_integration.SelectKBest')
    @patch('binning.methods._equal_width_binning.EqualWidthBinning')
    @patch('binning.utils.sklearn_integration.mutual_info_classif')
    def test_fit_equal_width_classification(self, mock_mutual_info, mock_binning_class, mock_select_k):
        """Test fit with equal_width binning for classification."""
        # Setup mocks
        mock_binner = Mock()
        mock_binner.fit_transform.return_value = np.array([[1, 2], [3, 4]])
        mock_binning_class.return_value = mock_binner

        mock_selector = Mock()
        mock_select_k.return_value = mock_selector

        X = np.array([[1.1, 2.2], [3.3, 4.4]])
        y = np.array([0, 1])  # Binary classification

        selector = BinningFeatureSelector(binning_method="equal_width", score_func="auto")
        selector.fit(X, y)

        # Check that binning was called correctly
        mock_binning_class.assert_called_once_with()
        mock_binner.fit_transform.assert_called_once_with(X)

        # Check that SelectKBest was called with classification function
        mock_select_k.assert_called_once_with(score_func=mock_mutual_info, k=10)
        mock_selector.fit.assert_called_once()

    @patch('binning.utils.sklearn_integration.SelectKBest')
    @patch('binning.methods._supervised_binning.SupervisedBinning')
    @patch('binning.utils.sklearn_integration.mutual_info_regression')
    def test_fit_supervised_regression(self, mock_mutual_info, mock_binning_class, mock_select_k):
        """Test fit with supervised binning for regression."""
        # Setup mocks
        mock_binner = Mock()
        mock_binner.fit_transform.return_value = np.array([[1, 2], [3, 4]])
        mock_binning_class.return_value = mock_binner

        mock_selector = Mock()
        mock_select_k.return_value = mock_selector

        X = np.array([[1.1, 2.2], [3.3, 4.4]])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5] + list(range(20)))  # Many unique values (regression)

        selector = BinningFeatureSelector(binning_method="supervised", score_func="auto")
        selector.fit(X, y)

        # Check that binning was called correctly
        mock_binning_class.assert_called_once_with()
        mock_binner.fit_transform.assert_called_once_with(X)

        # Check that SelectKBest was called with regression function
        mock_select_k.assert_called_once_with(score_func=mock_mutual_info, k=10)

    @patch('binning.methods._onehot_binning.OneHotBinning')
    def test_fit_onehot_binning(self, mock_binning_class):
        """Test fit with onehot binning."""
        mock_binner = Mock()
        mock_binner.fit_transform.return_value = np.array([[1, 2], [3, 4]])
        mock_binning_class.return_value = mock_binner

        with patch('binning.utils.sklearn_integration.SelectKBest') as mock_select_k:
            mock_selector = Mock()
            mock_select_k.return_value = mock_selector

            X = np.array([[1, 2], [3, 4]])
            y = np.array([0, 1])

            selector = BinningFeatureSelector(binning_method="onehot")
            selector.fit(X, y)

            mock_binning_class.assert_called_once_with()

    def test_fit_unknown_binning_method(self):
        """Test fit with unknown binning method."""
        selector = BinningFeatureSelector(binning_method="unknown")

        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        with pytest.raises(ValueError, match="Unknown binning method: unknown"):
            selector.fit(X, y)

    def test_fit_unknown_score_func(self):
        """Test fit with unknown score function."""
        with patch('binning.methods._equal_width_binning.EqualWidthBinning') as mock_binning_class:
            mock_binner = Mock()
            mock_binner.fit_transform.return_value = np.array([[1, 2], [3, 4]])
            mock_binning_class.return_value = mock_binner

            selector = BinningFeatureSelector(score_func="unknown_func")

            X = np.array([[1, 2], [3, 4]])
            y = np.array([0, 1])

            with pytest.raises(ValueError, match="Unknown score_func: unknown_func"):
                selector.fit(X, y)

    @patch('binning.utils.sklearn_integration.SelectKBest')
    @patch('binning.methods._equal_width_binning.EqualWidthBinning')
    @patch('binning.utils.sklearn_integration.mutual_info_classif')
    def test_fit_explicit_mutual_info_classif(self, mock_mutual_info, mock_binning_class, mock_select_k):
        """Test fit with explicit mutual_info_classif score function."""
        # Setup mocks
        mock_binner = Mock()
        mock_binner.fit_transform.return_value = np.array([[1, 2], [3, 4]])
        mock_binning_class.return_value = mock_binner

        mock_selector = Mock()
        mock_select_k.return_value = mock_selector

        X = np.array([[1.1, 2.2], [3.3, 4.4]])
        y = np.array([0, 1])

        selector = BinningFeatureSelector(score_func="mutual_info_classif")
        selector.fit(X, y)

        # Check that SelectKBest was called with explicit classif function
        mock_select_k.assert_called_once_with(score_func=mock_mutual_info, k=10)

    @patch('binning.utils.sklearn_integration.SelectKBest')
    @patch('binning.methods._equal_width_binning.EqualWidthBinning')
    @patch('binning.utils.sklearn_integration.mutual_info_regression')
    def test_fit_explicit_mutual_info_regression(self, mock_mutual_info, mock_binning_class, mock_select_k):
        """Test fit with explicit mutual_info_regression score function."""
        # Setup mocks
        mock_binner = Mock()
        mock_binner.fit_transform.return_value = np.array([[1, 2], [3, 4]])
        mock_binning_class.return_value = mock_binner

        mock_selector = Mock()
        mock_select_k.return_value = mock_selector

        X = np.array([[1.1, 2.2], [3.3, 4.4]])
        y = np.array([1.5, 2.5])

        selector = BinningFeatureSelector(score_func="mutual_info_regression")
        selector.fit(X, y)

        # Check that SelectKBest was called with explicit regression function
        mock_select_k.assert_called_once_with(score_func=mock_mutual_info, k=10)

    @patch('binning.utils.sklearn_integration.check_is_fitted')
    def test_transform(self, mock_check_fitted):
        """Test transform method."""
        selector = BinningFeatureSelector()

        # Mock fitted attributes
        mock_binner = Mock()
        binned_data = np.array([[1, 2], [3, 4]])
        mock_binner.transform.return_value = binned_data
        selector.binner_ = mock_binner

        mock_selector = Mock()
        mock_selector.transform.return_value = np.array([[1], [3]])
        selector.selector_ = mock_selector

        X = np.array([[1.1, 2.2], [3.3, 4.4]])
        result = selector.transform(X)

        mock_check_fitted.assert_called_once_with(selector)
        mock_binner.transform.assert_called_once_with(X)

        # Check that selector.transform was called with the binned data
        assert mock_selector.transform.call_count == 1
        call_args = mock_selector.transform.call_args[0]
        np.testing.assert_array_equal(call_args[0], binned_data)

        np.testing.assert_array_equal(result, np.array([[1], [3]]))

    @patch('binning.utils.sklearn_integration.check_is_fitted')
    def test_get_support(self, mock_check_fitted):
        """Test get_support method."""
        selector = BinningFeatureSelector()

        mock_selector = Mock()
        mock_selector.get_support.return_value = np.array([True, False, True])
        selector.selector_ = mock_selector

        result = selector.get_support(indices=True)

        mock_check_fitted.assert_called_once_with(selector)
        mock_selector.get_support.assert_called_once_with(indices=True)
        np.testing.assert_array_equal(result, np.array([True, False, True]))

    def test_get_feature_names_out_with_guidance_columns_non_list(self):
        """Test get_feature_names_out with guidance_columns not as a list."""
        mixin = SklearnCompatibilityMixin()
        setattr(mixin, '_fitted', True)
        mixin.feature_names_in_ = ['col1', 'col2', 'col3', 'col4']
        setattr(mixin, 'guidance_columns', 'col2')  # Single string, not a list

        result = mixin.get_feature_names_out()

        # Should exclude the single guidance column
        assert result == ['col1', 'col3', 'col4']


class TestBinningPipeline:
    """Test BinningPipeline."""

    @patch('sklearn.pipeline.Pipeline')
    @patch('binning.methods._supervised_binning.SupervisedBinning')
    def test_create_supervised_binning_pipeline_with_estimator(self, mock_binning_class, mock_pipeline):
        """Test creating supervised binning pipeline with final estimator."""
        mock_binner = Mock()
        mock_binning_class.return_value = mock_binner

        mock_estimator = Mock()

        result = BinningPipeline.create_supervised_binning_pipeline(
            guidance_column="target",
            task_type="classification",
            tree_params={"max_depth": 5},
            final_estimator=mock_estimator
        )

        # Check that SupervisedBinning was created correctly
        mock_binning_class.assert_called_once_with(
            task_type="classification",
            tree_params={"max_depth": 5},
            guidance_columns=["target"]
        )

        # Check that Pipeline was created
        mock_pipeline.assert_called_once_with([("binning", mock_binner), ("estimator", mock_estimator)])

    @patch('binning.methods._supervised_binning.SupervisedBinning')
    def test_create_supervised_binning_pipeline_without_estimator(self, mock_binning_class):
        """Test creating supervised binning pipeline without final estimator."""
        mock_binner = Mock()
        mock_binning_class.return_value = mock_binner

        result = BinningPipeline.create_supervised_binning_pipeline(
            guidance_column=0,  # Integer column index
            task_type="regression"
        )

        # Check that SupervisedBinning was created correctly
        mock_binning_class.assert_called_once_with(
            task_type="regression",
            tree_params=None,
            guidance_columns=[0]
        )

        # Should return the binner directly
        assert result == mock_binner

    def test_import_supervised_binning_success(self):
        """Test _import_supervised_binning function success."""
        from binning.utils.sklearn_integration import _import_supervised_binning

        # This should work normally
        SupervisedBinning = _import_supervised_binning()
        assert SupervisedBinning is not None

    def test_import_supervised_binning_error(self):
        """Test _import_supervised_binning function with import error."""
        from binning.utils.sklearn_integration import _import_supervised_binning

        with patch('builtins.__import__') as mock_import:
            mock_import.side_effect = ImportError("No module")

            with pytest.raises(ImportError, match="SupervisedBinning not available"):
                _import_supervised_binning()


class TestMakeBinningScorer:
    """Test make_binning_scorer function."""

    @patch('sklearn.metrics.make_scorer')
    @patch('sklearn.model_selection.cross_val_score')
    @patch('binning.methods._supervised_binning.SupervisedBinning')
    def test_make_binning_scorer_supervised(self, mock_binning_class, mock_cv_score, mock_make_scorer):
        """Test make_binning_scorer with supervised binning."""
        # Setup mocks
        mock_binner = Mock()
        mock_binner.fit_transform.return_value = np.array([[1, 2], [3, 4]])
        mock_binning_class.return_value = mock_binner

        mock_cv_score.return_value = np.array([0.8, 0.9, 0.85])

        # Create scorer
        scorer = make_binning_scorer("supervised", {"guidance_columns": [2]})

        # Test the internal scoring function
        mock_estimator = Mock()
        X = np.array([[1.1, 2.2], [3.3, 4.4]])
        y = np.array([0, 1])

        # Call the scoring function that was created
        mock_make_scorer.assert_called_once()
        scoring_func = mock_make_scorer.call_args[0][0]  # Get the scoring function

        result = scoring_func(mock_estimator, X, y)

        # Check that binning was applied correctly
        mock_binning_class.assert_called_once_with(guidance_columns=[2])

        # Check that X was augmented with y for supervised binning
        # Use a more flexible assertion that checks the call was made
        assert mock_binner.fit_transform.called
        call_args = mock_binner.fit_transform.call_args[0][0]
        expected_X_with_target = np.column_stack([X, y])
        np.testing.assert_array_equal(call_args, expected_X_with_target)

        # Check cross validation was called
        mock_cv_score.assert_called_once_with(mock_estimator, mock_binner.fit_transform.return_value, y, cv=3)

        # Check return value - use approximate comparison for floating point
        assert abs(result - 0.85) < 1e-10  # mean of [0.8, 0.9, 0.85]

    @patch('sklearn.metrics.make_scorer')
    @patch('sklearn.model_selection.cross_val_score')
    @patch('binning.methods._equal_width_binning.EqualWidthBinning')
    def test_make_binning_scorer_equal_width(self, mock_binning_class, mock_cv_score, mock_make_scorer):
        """Test make_binning_scorer with equal_width binning."""
        # Setup mocks
        mock_binner = Mock()
        mock_binner.fit_transform.return_value = np.array([[1, 2], [3, 4]])
        mock_binning_class.return_value = mock_binner

        mock_cv_score.return_value = np.array([0.7, 0.8, 0.75])

        # Create scorer
        scorer = make_binning_scorer("equal_width", {"n_bins": 5})

        # Get the scoring function
        mock_make_scorer.assert_called_once()
        scoring_func = mock_make_scorer.call_args[0][0]

        mock_estimator = Mock()
        X = np.array([[1.1, 2.2], [3.3, 4.4]])
        y = np.array([0, 1])

        result = scoring_func(mock_estimator, X, y)

        # Check that binning was applied correctly
        mock_binning_class.assert_called_once_with(n_bins=5)
        mock_binner.fit_transform.assert_called_once_with(X)  # No target augmentation for equal_width

        assert result == 0.75  # mean of [0.7, 0.8, 0.75]

    def test_make_binning_scorer_unknown_method(self):
        """Test make_binning_scorer with unknown method."""
        scorer = make_binning_scorer("unknown_method")

        # Get the scoring function
        with patch('sklearn.metrics.make_scorer') as mock_make_scorer:
            make_binning_scorer("unknown_method")
            scoring_func = mock_make_scorer.call_args[0][0]

            mock_estimator = Mock()
            X = np.array([[1, 2], [3, 4]])
            y = np.array([0, 1])

            with pytest.raises(ValueError, match="Unknown binning method: unknown_method"):
                scoring_func(mock_estimator, X, y)

    def test_make_binning_scorer_default_params(self):
        """Test make_binning_scorer with default parameters."""
        with patch('sklearn.metrics.make_scorer') as mock_make_scorer:
            scorer = make_binning_scorer()

            mock_make_scorer.assert_called_once()
            # Check that greater_is_better=True
            assert mock_make_scorer.call_args[1]["greater_is_better"] is True

    def test_make_binning_scorer_supervised_import_error(self):
        """Test make_binning_scorer with supervised import error."""
        scorer = make_binning_scorer("supervised")

        # Get the scoring function
        with patch('sklearn.metrics.make_scorer') as mock_make_scorer:
            make_binning_scorer("supervised")
            scoring_func = mock_make_scorer.call_args[0][0]

            # Mock the import to fail
            with patch('builtins.__import__') as mock_import:
                def import_side_effect(name, *args, **kwargs):
                    if name == "binning.methods._supervised_binning":
                        raise ImportError("No module")
                    # return __import__(name, *args, **kwargs)  # Removed unreachable line

                mock_import.side_effect = import_side_effect

                mock_estimator = Mock()
                X = np.array([[1, 2], [3, 4]])
                y = np.array([0, 1])

                with pytest.raises(ImportError, match="SupervisedBinning not available"):
                    scoring_func(mock_estimator, X, y)

    def test_make_binning_scorer_equal_width_import_error(self):
        """Test make_binning_scorer with equal_width import error."""
        scorer = make_binning_scorer("equal_width")

        # Get the scoring function
        with patch('sklearn.metrics.make_scorer') as mock_make_scorer:
            make_binning_scorer("equal_width")
            scoring_func = mock_make_scorer.call_args[0][0]

            # Mock the import to fail
            with patch('builtins.__import__') as mock_import:
                def import_side_effect(name, *args, **kwargs):
                    if name == "binning.methods._equal_width_binning":
                        raise ImportError("No module")
                    # return __import__(name, *args, **kwargs)  # Removed unreachable line

                mock_import.side_effect = import_side_effect

                mock_estimator = Mock()
                X = np.array([[1, 2], [3, 4]])
                y = np.array([0, 1])

                with pytest.raises(ImportError, match="EqualWidthBinning not available"):
                    scoring_func(mock_estimator, X, y)
