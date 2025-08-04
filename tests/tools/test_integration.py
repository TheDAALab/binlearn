"""
Comprehensive test suite for the binning.tools.integration module.

This module provides extensive testing for the integration utilities that
connect binning methods with scikit-learn and other ML frameworks. It covers
feature selection using binning-based mutual information, pipeline creation
utilities, and custom scoring functions that incorporate binning.

Test Classes:
    TestBinningFeatureSelector: Tests for the BinningFeatureSelector class
        including initialization, fitting with different binning methods,
        transformation, feature selection, and error handling.
    TestBinningPipeline: Tests for the BinningPipeline utility class including
        supervised binning pipeline creation with various configurations.
    TestMakeBinningScorer: Tests for the make_binning_scorer function including
        scorer creation, evaluation with different binning methods, and
        cross-validation integration.
"""

# pylint: disable=protected-access
from unittest.mock import Mock, patch

import numpy as np
import pytest

from binlearn.tools.integration import BinningFeatureSelector, BinningPipeline, make_binning_scorer


class TestBinningFeatureSelector:
    """Comprehensive test suite for BinningFeatureSelector class.

    This test class verifies the functionality of the BinningFeatureSelector,
    which combines binning methods with mutual information-based feature
    selection. Tests cover initialization, fitting with different binning
    methods, transformation, and sklearn compatibility.
    """

    def test_init_default_params(self):
        """Test initialization with default parameters.

        Verifies that the BinningFeatureSelector initializes correctly
        with default parameter values for binning method, k features,
        score function, and binning parameters.
        """
        selector = BinningFeatureSelector()

        assert selector.binning_method == "equal_width"
        assert selector.k == 10
        assert selector.score_func == "auto"
        assert selector.binning_params == {}

    def test_init_custom_params(self):
        """Test initialization with custom parameters.

        Verifies that the BinningFeatureSelector correctly accepts and
        stores custom parameters including binning method, number of features,
        score function, and binning-specific parameters.
        """
        binning_params = {"n_bins": 5}
        selector = BinningFeatureSelector(
            binning_method="supervised",
            k=15,
            score_func="mutual_info_classif",
            binning_params=binning_params,
        )

        assert selector.binning_method == "supervised"
        assert selector.k == 15
        assert selector.score_func == "mutual_info_classif"
        assert selector.binning_params == binning_params

    @patch("binlearn.tools.integration.SelectKBest")
    @patch("binlearn.tools.integration.EqualWidthBinning")
    @patch("binlearn.tools.integration.mutual_info_classif")
    def test_fit_equal_width_classification(
        self, mock_mutual_info, mock_binning_class, mock_select_k
    ):
        """Test fitting with equal_width binning for classification tasks.

        Verifies that the selector correctly uses EqualWidthBinning for
        classification tasks, applies the binning transformation, and
        sets up mutual information-based feature selection.
        """
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

    @patch("binlearn.tools.integration.SelectKBest")
    @patch("binlearn.tools.integration.SupervisedBinning")
    @patch("binlearn.tools.integration.mutual_info_regression")
    def test_fit_supervised_regression(self, mock_mutual_info, mock_binning_class, mock_select_k):
        """Test fit with supervised binning for regression."""
        # Setup mocks
        mock_binner = Mock()
        mock_binner.fit_transform.return_value = np.array([[1, 2], [3, 4]])
        mock_binning_class.return_value = mock_binner

        mock_selector = Mock()
        mock_select_k.return_value = mock_selector

        X = np.array([[1.1, 2.2], [3.3, 4.4]])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5] + list(range(20)))  # Many unique values

        selector = BinningFeatureSelector(binning_method="supervised", score_func="auto")
        selector.fit(X, y)

        # Check that binning was called correctly
        mock_binning_class.assert_called_once_with()
        mock_binner.fit_transform.assert_called_once_with(X)

        # Check that SelectKBest was called with regression function
        mock_select_k.assert_called_once_with(score_func=mock_mutual_info, k=10)

    @patch("binlearn.tools.integration.SingletonBinning")
    def test_fit_singleton_binning(self, mock_binning_class):
        """Test fit with singleton binning."""
        mock_binner = Mock()
        mock_binner.fit_transform.return_value = np.array([[1, 2], [3, 4]])
        mock_binning_class.return_value = mock_binner

        with patch("binlearn.tools.integration.SelectKBest") as mock_select_k:
            mock_selector = Mock()
            mock_select_k.return_value = mock_selector

            X = np.array([[1, 2], [3, 4]])
            y = np.array([0, 1])

            selector = BinningFeatureSelector(binning_method="singleton")
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
        with patch("binlearn.tools.integration.EqualWidthBinning") as mock_binning_class:
            mock_binner = Mock()
            mock_binner.fit_transform.return_value = np.array([[1, 2], [3, 4]])
            mock_binning_class.return_value = mock_binner

            selector = BinningFeatureSelector(score_func="unknown_func")

            X = np.array([[1, 2], [3, 4]])
            y = np.array([0, 1])

            with pytest.raises(ValueError, match="Unknown score_func: unknown_func"):
                selector.fit(X, y)

    @patch("binlearn.tools.integration.SelectKBest")
    @patch("binlearn.tools.integration.EqualWidthBinning")
    @patch("binlearn.tools.integration.mutual_info_classif")
    def test_fit_explicit_mutual_info_classif(
        self, mock_mutual_info, mock_binning_class, mock_select_k
    ):
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

    @patch("binlearn.tools.integration.SelectKBest")
    @patch("binlearn.tools.integration.EqualWidthBinning")
    @patch("binlearn.tools.integration.mutual_info_regression")
    def test_fit_explicit_mutual_info_regression(
        self, mock_mutual_info, mock_binning_class, mock_select_k
    ):
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

    @patch("binlearn.tools.integration.check_is_fitted")
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

    @patch("binlearn.tools.integration.check_is_fitted")
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


class TestBinningPipeline:
    """Test BinningPipeline."""

    @patch("binlearn.tools.integration.Pipeline")
    @patch("binlearn.tools.integration.SupervisedBinning")
    def test_create_supervised_binning_pipeline_with_estimator(
        self, mock_binning_class, mock_pipeline
    ):
        """Test creating supervised binning pipeline with final estimator."""
        mock_binner = Mock()
        mock_binning_class.return_value = mock_binner

        mock_estimator = Mock()

        _ = BinningPipeline.create_supervised_binning_pipeline(
            guidance_column="target",
            task_type="classification",
            tree_params={"max_depth": 5},
            final_estimator=mock_estimator,
        )

        # Check that SupervisedBinning was created correctly
        mock_binning_class.assert_called_once_with(
            task_type="classification", tree_params={"max_depth": 5}, guidance_columns=["target"]
        )

        # Check that Pipeline was created
        mock_pipeline.assert_called_once_with(
            [("binning", mock_binner), ("estimator", mock_estimator)]
        )

    @patch("binlearn.tools.integration.SupervisedBinning")
    def test_create_supervised_binning_pipeline_without_estimator(self, mock_binning_class):
        """Test creating supervised binning pipeline without final estimator."""
        mock_binner = Mock()
        mock_binning_class.return_value = mock_binner

        result = BinningPipeline.create_supervised_binning_pipeline(
            guidance_column=0, task_type="regression"  # Integer column index
        )

        # Check that SupervisedBinning was created correctly
        mock_binning_class.assert_called_once_with(
            task_type="regression", tree_params=None, guidance_columns=[0]
        )

        # Should return the binner directly
        assert result == mock_binner


class TestMakeBinningScorer:
    """Test make_binning_scorer function."""

    @patch("binlearn.tools.integration.make_scorer")
    @patch("binlearn.tools.integration.cross_val_score")
    @patch("binlearn.tools.integration.SupervisedBinning")
    def test_make_binning_scorer_supervised(
        self, mock_binning_class, mock_cv_score, mock_make_scorer
    ):
        """Test make_binning_scorer with supervised binning."""
        # Setup mocks
        mock_binner = Mock()
        mock_binner.fit_transform.return_value = np.array([[1, 2], [3, 4]])
        mock_binning_class.return_value = mock_binner

        mock_cv_score.return_value = np.array([0.8, 0.9, 0.85])

        # Create scorer
        _ = make_binning_scorer("supervised", {"guidance_columns": [2]})

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
        mock_cv_score.assert_called_once_with(
            mock_estimator, mock_binner.fit_transform.return_value, y, cv=3
        )

        # Check return value - use approximate comparison for floating point
        assert abs(result - 0.85) < 1e-10  # mean of [0.8, 0.9, 0.85]

    @patch("binlearn.tools.integration.make_scorer")
    @patch("binlearn.tools.integration.cross_val_score")
    @patch("binlearn.tools.integration.EqualWidthBinning")
    def test_make_binning_scorer_equal_width(
        self, mock_binning_class, mock_cv_score, mock_make_scorer
    ):
        """Test make_binning_scorer with equal_width binning."""
        # Setup mocks
        mock_binner = Mock()
        mock_binner.fit_transform.return_value = np.array([[1, 2], [3, 4]])
        mock_binning_class.return_value = mock_binner

        mock_cv_score.return_value = np.array([0.7, 0.8, 0.75])

        # Create scorer
        _ = make_binning_scorer("equal_width", {"n_bins": 5})

        # Get the scoring function
        mock_make_scorer.assert_called_once()
        scoring_func = mock_make_scorer.call_args[0][0]

        mock_estimator = Mock()
        X = np.array([[1.1, 2.2], [3.3, 4.4]])
        y = np.array([0, 1])

        result = scoring_func(mock_estimator, X, y)

        # Check that binning was applied correctly
        mock_binning_class.assert_called_once_with(n_bins=5)
        mock_binner.fit_transform.assert_called_once_with(
            X
        )  # No target augmentation for equal_width

        assert result == 0.75  # mean of [0.7, 0.8, 0.75]

    def test_make_binning_scorer_unknown_method(self):
        """Test make_binning_scorer with unknown method."""
        _ = make_binning_scorer("unknown_method")

        # Get the scoring function
        with patch("binlearn.tools.integration.make_scorer") as mock_make_scorer:
            make_binning_scorer("unknown_method")
            scoring_func = mock_make_scorer.call_args[0][0]

            mock_estimator = Mock()
            X = np.array([[1, 2], [3, 4]])
            y = np.array([0, 1])

            with pytest.raises(ValueError, match="Unknown binning method: unknown_method"):
                scoring_func(mock_estimator, X, y)

    def test_make_binning_scorer_default_params(self):
        """Test make_binning_scorer with default parameters."""
        with patch("binlearn.tools.integration.make_scorer") as mock_make_scorer:
            _ = make_binning_scorer()

            mock_make_scorer.assert_called_once()
            # Check that greater_is_better=True
            assert mock_make_scorer.call_args[1]["greater_is_better"] is True
