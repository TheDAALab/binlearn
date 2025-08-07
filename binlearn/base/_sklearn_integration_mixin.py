"""
Sklearn integration mixin providing complete sklearn compatibility and parameter management.

This mixin handles all sklearn-related functionality including parameter discovery,
fitted parameter serialization, feature name handling, and state management.
It enables the full workflow of fit → get_params → reconstruct → transform.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from ..utils.inspection import safe_get_class_parameters, safe_get_constructor_info
from ..utils.errors import ValidationMixin


class SklearnIntegrationMixin(BaseEstimator, ValidationMixin):  # type: ignore[misc,unused-ignore]
    """Complete sklearn integration with fitted parameter reconstruction support.

    This mixin provides comprehensive sklearn compatibility including:
    - Automatic parameter discovery and management
    - Fitted parameter serialization for object reconstruction
    - Sklearn test compatibility tags
    - Enhanced repr functionality
    - Automatic state management and validation

    Key Feature: Enables fitted parameter transfer workflows where you can
    fit an estimator, extract all parameters via get_params(), create a new
    instance with those parameters, and use it for transformation without
    refitting.

    Example:
        >>> binner = SomeBinning(n_bins=5)
        >>> binner.fit(X_train)
        >>> params = binner.get_params()  # Includes fitted parameters
        >>> new_binner = SomeBinning(**params)  # No fitting needed
        >>> X_test_binned = new_binner.transform(X_test)  # Works immediately
    """

    def __init__(self, **kwargs: Any):
        """Initialize sklearn integration mixin."""
        super().__init__(**kwargs)

        # Internal state tracking
        self._fitted = False

    def _more_tags(self) -> dict[str, Any]:
        """Provide sklearn compatibility tags."""
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

    def _check_fitted(self) -> None:
        """Check if the estimator is fitted."""
        if not self._fitted:
            raise RuntimeError("This estimator is not fitted yet. Call 'fit' first.")

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Enhanced get_params with automatic fitted parameter inclusion.

        Automatically discovers and includes both initialization parameters and
        fitted parameters, enabling complete object reconstruction workflows.

        Args:
            deep: If True, return parameters for sub-estimators as well.

        Returns:
            Dictionary containing all parameters needed to reconstruct this
            estimator, including fitted state.
        """
        # Get standard sklearn parameters
        params = super().get_params(deep=deep)

        # Add class-specific constructor parameters
        class_specific_params = safe_get_class_parameters(
            self.__class__, exclude_base_class="SklearnIntegrationMixin"
        )

        for param_name in class_specific_params:
            if hasattr(self, param_name):
                params[param_name] = getattr(self, param_name)

        # Add fitted parameters if fitted
        if self._fitted:
            fitted_params = self._get_fitted_params()
            params.update(fitted_params)

        return params  # type: ignore[no-any-return]

    def _get_fitted_params(self) -> dict[str, Any]:
        """Extract fitted parameters for object reconstruction.

        Automatically discovers fitted attributes (ending with _) and maps them
        to parameter names for use in constructor-based reconstruction.

        Returns:
            Dictionary mapping parameter names to fitted values.
        """
        fitted_params = {}

        # Sklearn internal attributes to exclude
        sklearn_internal_attrs = {
            "n_features_in_",
            "feature_names_in_",
        }

        # Find all fitted attributes
        for attr_name in dir(self):
            if (
                attr_name.endswith("_")
                and not attr_name.startswith("_")  # Exclude private
                and not attr_name.endswith("__")  # Exclude dunder
                and attr_name not in sklearn_internal_attrs
                and hasattr(self, attr_name)
            ):
                value = getattr(self, attr_name)
                if value is not None:
                    # Map fitted attribute to parameter name
                    param_name = attr_name.rstrip("_")
                    fitted_params[param_name] = value

        return fitted_params

    def get_serializable_params(self, deep: bool = True) -> dict[str, Any]:
        """Get JSON-serializable parameters."""
        params = self.get_params(deep=deep)
        return self._convert_to_python_types(params)  # type: ignore[no-any-return]

    def _convert_to_python_types(self, value: Any) -> Any:
        """Convert numpy types to pure Python types recursively."""
        if isinstance(value, dict):
            return {k: self._convert_to_python_types(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            converted = [self._convert_to_python_types(item) for item in value]
            return type(value)(converted) if isinstance(value, tuple) else converted
        if isinstance(value, np.ndarray):
            return self._convert_to_python_types(value.tolist())
        if isinstance(value, np.number | np.bool_):
            if isinstance(value, np.bool_):
                return bool(value)
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, np.floating):
                return float(value)
            return value.item()

        return value

    def set_params(self, **params: Any) -> SklearnIntegrationMixin:
        """Enhanced set_params with automatic state management.

        Intelligently handles parameter changes and automatically resets
        fitted state when necessary.

        Args:
            **params: Parameters to set.

        Returns:
            Self for method chaining.
        """
        # Handle parameter-specific logic
        if self._should_reset_fitted_state(params):
            self._fitted = False

        # Set parameters using sklearn's implementation
        result = super().set_params(**params)

        # Validate parameters after setting
        self._validate_params()

        return result  # type: ignore[no-any-return]

    def _should_reset_fitted_state(self, params: dict[str, Any]) -> bool:
        """Determine if fitted state should be reset based on parameter changes.

        Args:
            params: Parameters being set.

        Returns:
            True if fitted state should be reset.
        """
        if not self._fitted:
            return False

        # Parameters that always require refitting
        always_refit = {"fit_jointly", "guidance_columns"}

        # Class-specific parameters that require refitting
        class_params = safe_get_class_parameters(
            self.__class__, exclude_base_class="SklearnIntegrationMixin"
        )

        refit_params = always_refit | set(class_params)

        # Check if any parameter requiring refitting is being changed
        for param_name in params:
            if param_name in refit_params:
                current_value = getattr(self, param_name, None)
                new_value = params[param_name]
                if current_value != new_value:
                    return True

        return False

    def _validate_params(self) -> None:
        """Validate parameters - override in subclasses for specific validation."""
        pass

    def _get_constructor_info(self) -> dict[str, Any]:
        """Get constructor parameter information for repr."""
        return safe_get_constructor_info(self.__class__, concrete_only=True)

    def __repr__(self) -> str:  # type: ignore[override]
        """Enhanced repr showing only non-default parameters.

        Provides a clean, readable representation focusing on meaningful
        configuration differences from defaults.
        """
        class_name = self.__class__.__name__

        # Get constructor info
        constructor_info = self._get_constructor_info()

        # Build parameter list
        parts = []
        for param_name, default_value in constructor_info.items():
            if not hasattr(self, param_name):
                continue

            current_value = getattr(self, param_name)

            # Skip if value matches default
            if current_value == default_value:
                continue

            # Skip None values that are defaults
            if current_value is None and default_value is None:
                continue

            # Skip empty containers unless they differ from default
            if current_value in ({}, []) and default_value in (None, {}, []):
                continue

            # Format parameter display
            if param_name in {"bin_edges", "bin_representatives", "bin_spec", "fitted_trees"}:
                parts.append(f"{param_name}=...")
            elif isinstance(current_value, str):
                parts.append(f"{param_name}='{current_value}'")
            else:
                parts.append(f"{param_name}={current_value}")

        return f"{class_name}({', '.join(parts)})" if parts else f"{class_name}()"
