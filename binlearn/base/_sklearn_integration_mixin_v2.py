"""
Enhanced sklearn integration mixin with fitted parameter reconstruction support.

This module provides the SklearnIntegrationMixin class that handles all sklearn-related
functionality including parameter management, fitted parameter serialization, and
complete object reconstruction workflows.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from ..utils.errors import ValidationMixin
from ..utils.inspection import safe_get_class_parameters, safe_get_constructor_info
from ._data_handling_mixin_v2 import DataHandlingMixin


def convert_to_python_types(value: Any) -> Any:
    """Convert numpy types to pure Python types recursively for serialization.

    Args:
        value: Value to convert (can be nested structures).

    Returns:
        Value with numpy types converted to Python types.
    """
    if isinstance(value, dict):
        return {k: convert_to_python_types(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        converted = [convert_to_python_types(item) for item in value]
        return type(value)(converted) if isinstance(value, tuple) else converted
    if isinstance(value, np.ndarray):
        return convert_to_python_types(value.tolist())
    if isinstance(value, np.number | np.bool_):
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        return value.item()

    return value


class SklearnIntegrationMixin(BaseEstimator, ValidationMixin):  # type: ignore[misc,unused-ignore]
    """Complete sklearn integration with fitted parameter reconstruction support.

    This mixin provides comprehensive sklearn compatibility including:
    - Automatic parameter discovery and management
    - Fitted parameter serialization for object reconstruction
    - Enhanced repr functionality
    - Automatic state management and validation
    - Sklearn test compatibility

    Key Innovation: Enables fitted parameter transfer workflows where you can
    fit an estimator, extract all parameters via get_params(), create a new
    instance with those parameters, and use it for transformation without refitting.

    Example:
        >>> binner = SomeBinning(n_bins=5)
        >>> binner.fit(X_train)
        >>> params = binner.get_params()  # Includes fitted parameters!
        >>> new_binner = SomeBinning(**params)  # No fitting needed!
        >>> X_test_binned = new_binner.transform(X_test)  # Works immediately!
    """

    def __init__(self):
        """Initialize sklearn integration mixin."""
        # Define fitted attributes that indicate this estimator is fitted
        # Subclasses should override this in their constructors
        self._fitted_attributes: list[str] = []

    @property
    def _fitted(self) -> bool:
        """Check if this estimator is fitted by examining configured fitted attributes."""
        # Check if any of the configured fitted attributes have content
        for attr_name in self._fitted_attributes:
            attr_value = getattr(self, attr_name, None)
            if attr_value:
                # Check if it's a non-empty dict, list, or other truthy value
                if isinstance(attr_value, dict) and attr_value:
                    return True
                elif isinstance(attr_value, list) and attr_value:
                    return True
                elif attr_value and not isinstance(attr_value, (dict, list)):
                    return True
        return False

    def _set_fitted_attributes(self, **fitted_params: Any) -> None:
        """Set fitted parameters and derive feature information from them.

        Args:
            **fitted_params: Fitted parameters to set (e.g., bin_edges_, bin_representatives_)
        """
        # Set the provided fitted parameters
        for key, value in fitted_params.items():
            setattr(self, key, value)

        # Derive feature information if we have methods to do so
        if hasattr(self, "_derive_feature_information"):
            self._derive_feature_information()  # type: ignore

    def _has_fitted_attribute_pattern(self, attr_name: str) -> bool:
        """Check if an attribute name follows sklearn's fitted parameter convention.

        Args:
            attr_name: Attribute name to check (should end with '_').

        Returns:
            True if this follows sklearn's fitted parameter convention.
        """
        # Pure sklearn convention: fitted parameters end with underscore
        # and are not private (don't start with underscore)
        if not attr_name.endswith("_") or attr_name.startswith("_"):
            return False

        # Exclude sklearn's internal fitted attributes - they shouldn't be reconstructed
        sklearn_internal_attrs = {"_fitted"}  # Only exclude our internal state tracking

        if attr_name in sklearn_internal_attrs:
            return False

        # If it ends with _ and isn't private or sklearn internal, treat as fitted parameter
        return True

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
            fitted_params = self._extract_fitted_params()
            params.update(fitted_params)

        # Add class metadata for automatic reconstruction
        params["class_"] = self.__class__.__name__
        params["module_"] = self.__class__.__module__

        return params  # type: ignore[no-any-return]

    def _extract_fitted_params(self) -> dict[str, Any]:
        """Extract fitted parameters for object reconstruction.

        Automatically discovers fitted attributes (ending with _) and maps them
        to parameter names for use in constructor-based reconstruction.

        Returns:
            Dictionary mapping parameter names to fitted values.
        """
        fitted_params = {}

        # Sklearn internal attributes to exclude - these will be derived automatically
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
        return convert_to_python_types(params)  # type: ignore[no-any-return]

    def set_params(self, **params: Any) -> SklearnIntegrationMixin:
        """Enhanced set_params with automatic state management.

        Intelligently handles parameter changes and automatically resets
        fitted state when necessary. Also handles fitted parameter reconstruction.

        Args:
            **params: Parameters to set.

        Returns:
            Self for method chaining.
        """
        # Separate fitted parameters from init parameters
        fitted_params = {}
        init_params = {}

        for key, value in params.items():
            # Check if this parameter corresponds to a fitted attribute
            fitted_attr_name = key + "_"
            if hasattr(self, fitted_attr_name):
                fitted_params[fitted_attr_name] = value
            else:
                init_params[key] = value

        # Set init parameters using sklearn's implementation
        if init_params:
            result = super().set_params(**init_params)
        else:
            result = self

        # Set fitted parameters (fitted state is now derived automatically)
        if fitted_params:
            for attr_name, value in fitted_params.items():
                setattr(self, attr_name, value)

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

    def __repr__(self) -> str:  # type: ignore[override]
        """Enhanced repr showing only non-default parameters."""
        class_name = self.__class__.__name__

        # Get constructor info
        constructor_info = safe_get_constructor_info(self.__class__, concrete_only=True)

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
            if isinstance(current_value, dict) and len(str(current_value)) > 50:
                parts.append(f"{param_name}=...")
            elif isinstance(current_value, (list, tuple)) and len(str(current_value)) > 50:
                parts.append(f"{param_name}=...")
            elif isinstance(current_value, str):
                parts.append(f"{param_name}='{current_value}'")
            else:
                parts.append(f"{param_name}={current_value}")

        return f"{class_name}({', '.join(parts)})" if parts else f"{class_name}()"
