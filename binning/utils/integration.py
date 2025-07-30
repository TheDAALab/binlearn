"""Integration utilities for binning with various frameworks."""

from typing import Dict, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.utils.validation import check_is_fitted


class BinningFeatureSelector(BaseEstimator, TransformerMixin):
    """Feature selector that uses binning-based mutual information."""

    def __init__(
        self,
        binning_method: str = "equal_width",
        k: int = 10,
        score_func: str = "auto",
        binning_params: Optional[Dict] = None,
    ):
        """
        Initialize the feature selector.

        Parameters
        ----------
        binning_method : str, default="equal_width"
            Binning method to use before computing mutual information.
        k : int, default=10
            Number of top features to select.
        score_func : str, default="auto"
            Scoring function ("mutual_info_classif", "mutual_info_regression", "auto").
        binning_params : dict, optional
            Parameters to pass to the binning method.
        """
        self.binning_method = binning_method
        self.k = k
        self.score_func = score_func
        self.binning_params = binning_params or {}

    def fit(self, X, y):
        """Fit the feature selector."""
        # Import here to avoid circular imports
        if self.binning_method == "equal_width":
            from binning.methods._equal_width_binning import EqualWidthBinning

            binner = EqualWidthBinning(**self.binning_params)
        elif self.binning_method == "supervised":
            from binning.methods._supervised_binning import SupervisedBinning

            binner = SupervisedBinning(**self.binning_params)
        elif self.binning_method == "onehot":
            from binning.methods._onehot_binning import OneHotBinning

            binner = OneHotBinning(**self.binning_params)
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

    def transform(self, X):
        """Transform the input by selecting features."""
        check_is_fitted(self)
        X_binned = self.binner_.transform(X)
        return self.selector_.transform(X_binned)

    def get_support(self, indices: bool = False):
        """Get selected feature indices or boolean mask."""
        check_is_fitted(self)
        return self.selector_.get_support(indices=indices)


def _import_supervised_binning():
    """Import SupervisedBinning with error handling."""
    try:
        from binning.methods._supervised_binning import SupervisedBinning
        return SupervisedBinning
    except ImportError:
        raise ImportError("SupervisedBinning not available")


class BinningPipeline:
    """Pipeline utilities for binning operations."""

    @staticmethod
    def create_supervised_binning_pipeline(
        guidance_column: Union[str, int],
        task_type: str = "classification",
        tree_params: Optional[Dict] = None,
        final_estimator=None,
    ):
        """Create a pipeline with supervised binning."""
        from sklearn.pipeline import Pipeline

        # Import locally to avoid issues
        SupervisedBinning = _import_supervised_binning()

        binner = SupervisedBinning(
            task_type=task_type, tree_params=tree_params, guidance_columns=[guidance_column]
        )

        if final_estimator is not None:
            return Pipeline([("binning", binner), ("estimator", final_estimator)])
        return binner


def make_binning_scorer(binning_method: str = "supervised", binning_params: Optional[Dict] = None):
    """Create a scorer that includes binning in the evaluation."""
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import cross_val_score

    def binning_score(estimator, X, y):
        """Score function that applies binning before evaluation."""
        # Create and fit binner based on method
        params = binning_params or {}

        if binning_method == "supervised":
            try:
                from binning.methods._supervised_binning import SupervisedBinning
            except ImportError as e:
                raise ImportError("SupervisedBinning not available") from e

            params.setdefault("guidance_columns", [-1])  # Assume last column is target
            binner = SupervisedBinning(**params)
            # For supervised binning, we need to include the target
            X_with_target = np.column_stack([X, y])
            X_binned = binner.fit_transform(X_with_target)
        elif binning_method == "equal_width":
            try:
                from binning.methods._equal_width_binning import EqualWidthBinning
            except ImportError as e:
                raise ImportError("EqualWidthBinning not available") from e

            binner = EqualWidthBinning(**params)
            X_binned = binner.fit_transform(X)
        else:
            raise ValueError(f"Unknown binning method: {binning_method}")

        # Score the estimator on binned data
        scores = cross_val_score(estimator, X_binned, y, cv=3)
        return scores.mean()

    return make_scorer(binning_score, greater_is_better=True)
