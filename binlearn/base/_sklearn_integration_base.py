"""
Clean sklearn integration mixin for V2 architecture.

This module provides the core sklearn compatibility layer that handles parameter
management, fitted state tracking, and reconstruction workflows.
"""

from __future__ import annotations

import inspect
from typing import Any

from sklearn.base import BaseEstimator


class SklearnIntegrationBase(BaseEstimator):  # type: ignore[misc,unused-ignore]
    """Clean sklearn integration for V2 architecture.

    Provides core sklearn compatibility without assumptions about specific fitted attributes.
    Subclasses configure which attributes indicate fitted state.
    """

    def __init__(self) -> None:
        """Initialize sklearn integration mixin."""
        BaseEstimator.__init__(self)
        # Fitted attributes to check - subclasses configure this
        self._fitted_attributes: list[str] = []

    @property
    def _fitted(self) -> bool:
        """Check if this estimator is fitted by examining configured fitted attributes."""
        if not hasattr(self, "_fitted_attributes") or not self._fitted_attributes:
            return False

        # Check if any configured fitted attributes have content
        for attr_name in self._fitted_attributes:
            attr_value = getattr(self, attr_name, None)
            if attr_value is not None:
                if isinstance(attr_value, dict):
                    if attr_value:  # Non-empty dict
                        return True
                elif isinstance(attr_value, list):
                    if attr_value:  # Non-empty list
                        return True
                else:  # Not dict or list
                    if attr_value:  # Truthy value
                        return True
        return False

    def _set_fitted_attributes(self, **fitted_params: Any) -> None:
        """Set fitted parameters.

        Args:
            **fitted_params: Fitted parameters to set
        """
        for key, value in fitted_params.items():
            setattr(self, key, value)

    def _check_fitted(self) -> None:
        """Check if the estimator is fitted."""
        if not self._fitted:
            raise RuntimeError("This estimator is not fitted yet. Call 'fit' first.")

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Enhanced get_params with automatic fitted parameter inclusion."""

        # Get the constructor signature
        init_signature = inspect.signature(self.__class__.__init__)

        # Get base parameters from constructor signature, excluding self and special params
        params = {}
        for param_name in init_signature.parameters:
            if param_name == "self":
                continue

            # Try to get attribute value, use default if not found
            if hasattr(self, param_name):
                params[param_name] = getattr(self, param_name)
            else:
                # Use parameter default if available
                param_obj = init_signature.parameters[param_name]
                if param_obj.default != inspect.Parameter.empty:
                    params[param_name] = param_obj.default
                else:
                    params[param_name] = None

        # Remove class_ and module_ parameters - they are swallowed but not stored
        params.pop("class_", None)
        params.pop("module_", None)

        # Add fitted parameters if fitted
        if self._fitted:
            fitted_params = self._extract_fitted_params()
            params.update(fitted_params)

        # Add class metadata for automatic reconstruction
        params["class_"] = self.__class__.__name__
        params["module_"] = self.__class__.__module__

        return params

    def _extract_fitted_params(self) -> dict[str, Any]:
        """Extract fitted parameters for object reconstruction."""
        fitted_params = {}

        # Find all fitted attributes
        for attr_name in dir(self):
            if (
                attr_name.endswith("_")
                and not attr_name.startswith("_")
                and not attr_name.endswith("__")
                and attr_name not in {"n_features_in_", "feature_names_in_"}  # These are derived
                and hasattr(self, attr_name)
            ):
                value = getattr(self, attr_name)
                if value is not None:
                    # Map fitted attribute to parameter name
                    param_name = attr_name.rstrip("_")
                    fitted_params[param_name] = value

        return fitted_params

    def set_params(self, **params: Any) -> SklearnIntegrationBase:
        """Set the parameters of this estimator.

        This method supports reconstruction workflows by handling fitted parameters
        that come from get_params() output (without underscores) and setting them
        as fitted attributes (with underscores).

        Args:
            **params: Parameters to set. Can include:
                - Regular constructor parameters (n_bins, clip, etc.)
                - Fitted parameters from get_params (bin_edges, bin_representatives)
                - Class metadata (ignored during reconstruction)

        Returns:
            self: Returns the instance itself.
        """
        # Handle class metadata parameters (ignore them during reconstruction)
        cleaned_params = {k: v for k, v in params.items() if k not in {"class_", "module_"}}

        # Handle fitted parameters that need to be set with underscores
        fitted_params_to_set = {}
        regular_params = {}

        for param_name, param_value in cleaned_params.items():
            # Check if this is a fitted parameter (from get_params output)
            fitted_attr_name = f"{param_name}_"

            # Special case: bin_edges, bin_representatives, and bin_spec are constructor params
            # but when set via set_params during reconstruction, they should be treated as fitted params
            is_reconstruction_param = (
                param_name in {"bin_edges", "bin_representatives", "bin_spec"}
                and param_value is not None
            )

            if is_reconstruction_param or (
                param_name not in self.get_params(deep=False) and hasattr(self, fitted_attr_name)
            ):
                # This is a fitted parameter - set it with underscore
                fitted_params_to_set[fitted_attr_name] = param_value
            else:
                # Regular parameter - handle normally
                regular_params[param_name] = param_value

        # Set regular parameters through sklearn mechanism (excluding reconstruction params)
        if regular_params:
            # Filter out bin_edges/bin_representatives/bin_spec if they're None (don't override constructor defaults)
            filtered_regular_params = {
                k: v
                for k, v in regular_params.items()
                if not (k in {"bin_edges", "bin_representatives", "bin_spec"} and v is None)
            }
            if filtered_regular_params:
                BaseEstimator.set_params(self, **filtered_regular_params)

        # Set fitted parameters directly
        for attr_name, attr_value in fitted_params_to_set.items():
            setattr(self, attr_name, attr_value)

        # If we set fitted parameters, also set sklearn attributes if available
        if fitted_params_to_set:
            sklearn_setter = getattr(self, "_set_sklearn_attributes_from_specs", None)
            if sklearn_setter:
                sklearn_setter()

        return self

    def _validate_params(self) -> None:
        """Validate parameters - override in subclasses."""
        pass

    def _more_tags(self) -> dict[str, Any]:
        """Provide sklearn compatibility tags."""
        return {
            "requires_fit": True,
            "requires_y": False,
            "X_types": ["2darray"],
            "allow_nan": True,
            "stateless": False,
        }
