"""Sklearn integration utilities for binning methods."""

from typing import Any


# pylint: disable=too-few-public-methods
class SklearnCompatibilityMixin:
    """Mixin to enhance sklearn compatibility for binning methods."""

    def _more_tags(self) -> dict:
        """Provide additional tags for sklearn compatibility."""
        return {
            "requires_fit": True,
            "requires_y": False,
            "requires_positive_X": False,
            "requires_positive_y": False,
            "X_types": ["2darray"],
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

    def _check_feature_names(self, X: Any, reset: bool = False) -> list[str]:
        """Check and store feature names from input."""
        feature_names = None

        # Try to get feature names from pandas-like objects
        if hasattr(X, "columns"):
            feature_names = list(X.columns)
        elif hasattr(X, "feature_names"):
            feature_names = list(X.feature_names)
        elif hasattr(X, "_feature_names"):
            feature_names = list(X._feature_names)  # pylint: disable=protected-access
        else:
            # Default to generic names
            n_features = X.shape[1] if hasattr(X, "shape") else len(X[0])
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Store feature names in a way that's compatible with sklearn
        # Use the same attribute name as sklearn but avoid property conflicts
        if reset or not hasattr(self, "feature_names_in_"):
            self.feature_names_in_ = feature_names

        return feature_names

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
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
            for idx, name in enumerate(input_features):
                if name not in guidance_cols and idx not in guidance_cols:
                    output_features.append(name)
            return output_features

        return input_features.copy()

    def _validate_params(self) -> None:
        """Validate parameters for sklearn compatibility."""
        # This method should be implemented by subclasses
        # to validate their specific parameters
        # Base implementation does nothing
