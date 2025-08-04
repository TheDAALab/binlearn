"""Integration utilities for binning with various frameworks.

This module provides utilities for integrating binning methods with scikit-learn
and other machine learning frameworks. It includes feature selectors, pipeline
utilities, and scoring functions that leverage binning for improved ML workflows.

Classes:
    BinningFeatureSelector: Feature selector using binning-based mutual information.
    BinningPipeline: Pipeline utilities for binning operations.

Functions:
    make_binning_scorer: Create a scorer that includes binning in evaluation.
    _import_supervised_binning: Helper function to import SupervisedBinning.
"""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from ..methods import EqualWidthBinning, SingletonBinning, SupervisedBinning


class BinningFeatureSelector(BaseEstimator, TransformerMixin):
    """Feature selector that uses binning-based mutual information.

    This transformer combines binning methods with mutual information-based
    feature selection to identify the most informative features for prediction.
    It first applies a specified binning method to discretize features, then
    uses mutual information to rank and select the top k features.

    Attributes:
        binning_method (str): The binning method to use.
        k (int): Number of top features to select.
        score_func (str): Scoring function for feature selection.
        binning_params (dict): Parameters passed to the binning method.
        selector_ (SelectKBest): Fitted feature selector (set after fit).
        binner_ (object): Fitted binning transformer (set after fit).
    """

    def __init__(
        self,
        binning_method: str = "equal_width",
        k: int = 10,
        score_func: str = "auto",
        binning_params: dict | None = None,
    ):
        """Initialize the feature selector.

        Args:
            binning_method: Binning method to use before computing mutual information.
                Options: "equal_width", "supervised", "singleton". Defaults to "equal_width".
            k: Number of top features to select. Defaults to 10.
            score_func: Scoring function for feature selection. Options:
                "mutual_info_classif", "mutual_info_regression", "auto".
                If "auto", automatically detects based on target variable.
                Defaults to "auto".
            binning_params: Additional parameters to pass to the binning method.
                Defaults to None (empty dict).
        """
        self.binning_method = binning_method
        self.k = k
        self.score_func = score_func
        self.binning_params = binning_params or {}

        # Initialize attributes that will be set during fit
        self.selector_: SelectKBest | None = None
        self.binner_: Any | None = None

    def fit(self, X: Any, y: Any) -> Any:
        """Fit the feature selector.

        Applies the specified binning method to the input data, then fits
        a mutual information-based feature selector on the binned features.

        Args:
            X: Input features. Can be array-like, pandas DataFrame, or polars DataFrame.
            y: Target values for supervised feature selection.

        Returns:
            self: Returns the fitted feature selector.

        Raises:
            ValueError: If binning_method or score_func is not recognized.
        """
        # Import here to avoid circular imports
        if self.binning_method == "equal_width":
            binner = EqualWidthBinning(**self.binning_params)
        elif self.binning_method == "supervised":
            binner = SupervisedBinning(**self.binning_params)
        elif self.binning_method == "singleton":
            binner = SingletonBinning(**self.binning_params)
        else:
            raise ValueError(f"Unknown binning method: {self.binning_method}")

        # Fit and transform data
        X_binned = binner.fit_transform(X)

        # Determine scoring function
        if self.score_func == "auto":
            # Auto-detect based on target variable
            unique_values = len(np.unique(y))
            if unique_values <= 20:  # Assume classification
                score_func = mutual_info_classif
            else:  # Assume regression
                score_func = mutual_info_regression
        elif self.score_func == "mutual_info_classif":
            score_func = mutual_info_classif
        elif self.score_func == "mutual_info_regression":
            score_func = mutual_info_regression
        else:
            raise ValueError(f"Unknown score_func: {self.score_func}")

        # Create and fit selector
        self.selector_ = SelectKBest(score_func=score_func, k=self.k)
        self.selector_.fit(X_binned, y)

        # Store binning transformer
        self.binner_ = binner

        return self

    def transform(self, X: Any) -> Any:
        """Transform the input by selecting features.

        Applies the fitted binning transformation followed by feature selection
        to return only the top k most informative features.

        Args:
            X: Input features to transform. Must have same structure as training data.

        Returns:
            Transformed data with only selected features.

        Raises:
            NotFittedError: If the selector has not been fitted yet.
        """
        check_is_fitted(self)
        assert self.binner_ is not None, "Selector must be fitted before transform"
        assert self.selector_ is not None, "Selector must be fitted before transform"
        X_binned = self.binner_.transform(X)
        return self.selector_.transform(X_binned)

    def get_support(self, indices: bool = False) -> Any:
        """Get selected feature indices or boolean mask.

        Args:
            indices: If True, return feature indices. If False, return boolean mask.
                Defaults to False.

        Returns:
            Boolean mask or integer indices of selected features.

        Raises:
            NotFittedError: If the selector has not been fitted yet.
        """
        check_is_fitted(self)
        assert self.selector_ is not None, "Selector must be fitted before get_support"
        return self.selector_.get_support(indices=indices)


# pylint: disable=too-few-public-methods
class BinningPipeline:
    """Pipeline utilities for binning operations.

    This class provides static methods for creating machine learning pipelines
    that incorporate binning methods. It simplifies the process of combining
    binning transformations with downstream estimators.
    """

    @staticmethod
    def create_supervised_binning_pipeline(
        guidance_column: str | int,
        task_type: str = "classification",
        tree_params: dict | None = None,
        final_estimator: Any = None,
    ) -> Any:
        """Create a pipeline with supervised binning.

        Creates a scikit-learn pipeline that uses supervised binning as the first
        step, optionally followed by a final estimator.

        Args:
            guidance_column: Column to use for supervised binning guidance.
                Can be column name (str) or index (int).
            task_type: Type of supervised learning task. Options: "classification"
                or "regression". Defaults to "classification".
            tree_params: Parameters to pass to the decision tree used for binning.
                Defaults to None (use default tree parameters).
            final_estimator: Optional estimator to add as final step in pipeline.
                If None, returns just the binning transformer. Defaults to None.

        Returns:
            sklearn.pipeline.Pipeline with binning and optional final estimator,
            or just the binning transformer if final_estimator is None.
        """

        binner = SupervisedBinning(
            task_type=task_type, tree_params=tree_params, guidance_columns=[guidance_column]
        )

        if final_estimator is not None:
            return Pipeline([("binning", binner), ("estimator", final_estimator)])
        return binner


def make_binning_scorer(
    binning_method: str = "supervised", binning_params: dict | None = None
) -> Any:
    """Create a scorer that includes binning in the evaluation.

    Creates a scikit-learn compatible scorer that applies binning to the data
    before evaluating a model. This allows for cross-validation and model
    selection that incorporates the binning transformation.

    Args:
        binning_method: Binning method to apply. Options: "supervised",
            "equal_width". Defaults to "supervised".
        binning_params: Parameters to pass to the binning method.
            Defaults to None (use default parameters).

    Returns:
        sklearn.metrics scorer that applies binning before model evaluation.

    Raises:
        ValueError: If binning_method is not recognized.
    """

    def binning_score(estimator: Any, X: Any, y: Any) -> Any:
        """Score function that applies binning before evaluation.

        This nested function performs the actual scoring by first applying
        the specified binning method to the data, then evaluating the
        estimator using cross-validation.

        Args:
            estimator: Machine learning estimator to evaluate.
            X: Input features.
            y: Target values.

        Returns:
            float: Mean cross-validation score after binning transformation.

        Raises:
            ValueError: If binning_method is not recognized.
        """
        # Create and fit binner based on method
        params = binning_params or {}

        if binning_method == "supervised":
            params.setdefault("guidance_columns", [-1])  # Assume last column is target
            binner = SupervisedBinning(**params)
            # For supervised binning, we need to include the target
            X_with_target = np.column_stack([X, y])
            X_binned = binner.fit_transform(X_with_target)
        elif binning_method == "equal_width":
            binner = EqualWidthBinning(**params)
            X_binned = binner.fit_transform(X)
        else:
            raise ValueError(f"Unknown binning method: {binning_method}")

        # Score the estimator on binned data
        scores = cross_val_score(estimator, X_binned, y, cv=3)
        return scores.mean()

    return make_scorer(binning_score, greater_is_better=True)
