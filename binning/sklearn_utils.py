"""
Enhanced sklearn integration utilities for binning framework.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression


class SklearnCompatibilityMixin:
    """Mixin to enhance sklearn compatibility for binning methods."""

    def _more_tags(self) -> Dict[str, Any]:
        """Provide additional tags for sklearn compatibility."""
        return {
            "requires_fit": True,
            "requires_y": False,
            "requires_positive_X": False,
            "requires_positive_y": False,
            "X_types": ["2darray", "sparse"],
            "poor_score": True,
            "no_validation": False,
            "multioutput": False,
            "multioutput_only": False,
            "multilabel": False,
            "allow_nan": True,
            "stateless": False,
            "binary_only": False,
            "_xfail_checks": {
                "check_parameters_default_constructible": "transformer has required parameters",
                "check_estimators_dtypes": "transformer returns integers",
            },
        }

    def _check_feature_names(self, X, reset: bool = False) -> List[str]:
        """Check and store feature names from input."""
        feature_names = None

        # Try to extract feature names from pandas DataFrame
        if hasattr(X, "columns"):
            feature_names = list(X.columns)
        elif hasattr(X, "feature_names"):
            feature_names = list(X.feature_names)
        else:
            # Generate default feature names
            n_features = X.shape[1] if hasattr(X, "shape") else len(X[0])
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Store feature names in a way that's compatible with sklearn
        # Use the same attribute name as sklearn but avoid property conflicts
        if reset or not hasattr(self, "feature_names_in_"):
            self.feature_names_in_ = feature_names

        return feature_names

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Get output feature names for transformation."""
        # Note: This assumes the class using this mixin inherits from BaseEstimator
        # Check if fitted (basic check)
        if not hasattr(self, "_fitted") or not getattr(self, "_fitted", False):
            raise ValueError("This estimator is not fitted yet. Call 'fit' first.")

        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
            if input_features is None:
                n_features = getattr(self, "n_features_in_", getattr(self, "_n_features_in", 0))
                input_features = [f"x{i}" for i in range(n_features)]

        # For binning, we typically return the same feature names
        # but could be modified for guidance columns
        guidance_columns = getattr(self, "guidance_columns", None)
        if guidance_columns is not None:
            guidance_cols = guidance_columns
            if not isinstance(guidance_cols, list):
                guidance_cols = [guidance_cols]

            # Return only non-guidance column names
            output_features = []
            for i, name in enumerate(input_features):
                if name not in guidance_cols and i not in guidance_cols:
                    output_features.append(name)
            return output_features

        return input_features.copy()

    def _validate_params(self) -> None:
        """Validate parameters for sklearn compatibility."""
        # This method should be implemented by subclasses
        # to validate their specific parameters
        pass


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


class BinningPipeline:
    """Utility class to create common binning pipelines."""

    @staticmethod
    def create_supervised_binning_pipeline(
        guidance_column: Union[str, int],
        task_type: str = "classification",
        tree_params: Optional[Dict] = None,
        final_estimator: Optional[BaseEstimator] = None,
    ):
        """Create a pipeline with supervised binning."""
        from sklearn.pipeline import Pipeline

        # Import locally to avoid issues
        try:
            from binning.methods._supervised_binning import SupervisedBinning
        except ImportError:
            raise ImportError("SupervisedBinning not available")

        binner = SupervisedBinning(
            task_type=task_type, tree_params=tree_params, guidance_columns=[guidance_column]
        )

        if final_estimator is not None:
            return Pipeline([("binning", binner), ("estimator", final_estimator)])
        else:
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

                params.setdefault("guidance_columns", [-1])  # Assume last column is target
                binner = SupervisedBinning(**params)
                # For supervised binning, we need to include the target
                X_with_target = np.column_stack([X, y])
                X_binned = binner.fit_transform(X_with_target)
            except ImportError:
                raise ImportError("SupervisedBinning not available")
        elif binning_method == "equal_width":
            try:
                from binning.methods._equal_width_binning import EqualWidthBinning

                binner = EqualWidthBinning(**params)
                X_binned = binner.fit_transform(X)
            except ImportError:
                raise ImportError("EqualWidthBinning not available")
        else:
            raise ValueError(f"Unknown binning method: {binning_method}")

        # Score the estimator on binned data
        scores = cross_val_score(estimator, X_binned, y, cv=3)
        return scores.mean()

    return make_scorer(binning_score, greater_is_better=True)
