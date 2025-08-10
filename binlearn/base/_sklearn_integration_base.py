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
    """Base mixin providing sklearn compatibility and integration features.

    This class serves as the foundation for sklearn integration across all binlearn
    estimators. It provides essential compatibility features including parameter
    management, fitted state tracking, and reconstruction workflows that align with
    sklearn's estimator interface conventions.

    The mixin follows sklearn's design patterns while remaining flexible enough to
    support different binning paradigms (interval-based, flexible, supervised, etc.).
    It delegates fitted attribute management to subclasses, allowing each binning
    type to define its own fitted state indicators.

    Key Features:
    - Automatic sklearn parameter introspection and management
    - Configurable fitted state tracking based on subclass-defined attributes
    - Reconstruction support for pipeline persistence and model serialization
    - BaseEstimator inheritance for sklearn ecosystem compatibility
    - Clean separation between sklearn interface and binning implementation

    Attributes:
        _fitted_attributes: List of attribute names that indicate fitted state.
            Configured by subclasses to define what constitutes a fitted estimator.
            Common examples: ['bin_edges_', 'bin_representatives_', 'bin_spec_']

    Example:
        >>> class MyBinner(SklearnIntegrationBase, SomeOtherBase):
        ...     def __init__(self, n_bins=5):
        ...         SklearnIntegrationBase.__init__(self)
        ...         self.n_bins = n_bins
        ...         self._fitted_attributes = ['bin_edges_']
        ...
        ...     def fit(self, X):
        ...         self.bin_edges_ = compute_edges(X, self.n_bins)
        ...         return self
        >>>
        >>> binner = MyBinner(n_bins=10)
        >>> print(binner._fitted)  # False
        >>> binner.fit(X)
        >>> print(binner._fitted)  # True

    Note:
        - This is a mixin class designed to be combined with other base classes
        - Subclasses must configure _fitted_attributes to enable proper fitted state detection
        - Follows sklearn's estimator interface conventions (fit, transform, etc.)
        - Provides foundation for sklearn pipeline integration and model persistence
        - Does not implement binning logic itself - purely handles sklearn integration
    """

    def __init__(self) -> None:
        """Initialize sklearn integration mixin with default settings.

        Sets up the foundation for sklearn compatibility by initializing the parent
        BaseEstimator and configuring the fitted attribute tracking system. Subclasses
        should call this method and then configure their specific fitted attributes.

        Note:
            - Initializes _fitted_attributes as empty list - subclasses must configure this
            - Must be called by subclasses during their initialization
            - Sets up BaseEstimator functionality for sklearn ecosystem compatibility
        """
        BaseEstimator.__init__(self)
        # Fitted attributes to check - subclasses configure this
        self._fitted_attributes: list[str] = []

    @property
    def _fitted(self) -> bool:
        """Check if this estimator is fitted by examining configured fitted attributes.

        This property provides a robust way to determine if the estimator has been
        fitted by checking whether any of the configured fitted attributes contain
        meaningful data. The check handles different attribute types appropriately.

        Returns:
            True if the estimator is fitted (at least one fitted attribute has content),
            False otherwise.

        Note:
            - Returns False if _fitted_attributes is not configured or empty
            - For dict attributes: considers non-empty dicts as fitted
            - For list attributes: considers non-empty lists as fitted
            - For other attributes: considers truthy values as fitted
            - Subclasses must configure _fitted_attributes for this to work properly
            - Used internally by _check_fitted() to validate estimator state
        """
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
        """Set fitted parameters as instance attributes.

        This method provides a convenient way to set multiple fitted parameters
        at once. It's typically used during the fitting process to store computed
        binning parameters as instance attributes.

        Args:
            **fitted_params: Key-value pairs of fitted parameters to set as
                instance attributes. Keys become attribute names, values become
                attribute values.

        Example:
            >>> binner._set_fitted_attributes(
            ...     bin_edges_={'col1': [0, 1, 2]},
            ...     bin_representatives_={'col1': [0.5, 1.5]}
            ... )
            >>> # Equivalent to:
            >>> # binner.bin_edges_ = {'col1': [0, 1, 2]}
            >>> # binner.bin_representatives_ = {'col1': [0.5, 1.5]}

        Note:
            - Simply uses setattr to assign each parameter as an instance attribute
            - Commonly used with fitted attribute names ending in underscore
            - Does not validate parameter names or values
            - Part of the internal fitting workflow
        """
        for key, value in fitted_params.items():
            setattr(self, key, value)

    def _check_fitted(self) -> None:
        """Check if the estimator is fitted and raise error if not.

        This method enforces the sklearn convention that estimators must be fitted
        before they can be used for transformation. It should be called at the
        beginning of methods like transform() and inverse_transform().

        Raises:
            RuntimeError: If the estimator has not been fitted yet (no fitted
                attributes contain meaningful data).

        Example:
            >>> def transform(self, X):
            ...     self._check_fitted()  # Ensure estimator is fitted
            ...     # ... proceed with transformation logic
            >>>
            >>> binner = MyBinner()
            >>> binner.transform(X)  # RuntimeError: not fitted
            >>> binner.fit(X)
            >>> binner.transform(X)  # OK, now fitted

        Note:
            - Uses the _fitted property to determine fitted state
            - Should be called at the start of all transformation methods
            - Provides clear error message to guide users
            - Part of sklearn's estimator interface best practices
        """
        if not self._fitted:
            raise RuntimeError("This estimator is not fitted yet. Call 'fit' first.")

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator, including fitted parameters.

        This method extends sklearn's standard get_params to include fitted parameters
        when the estimator is fitted, enabling complete object reconstruction through
        the get_params/set_params interface. This is essential for pipeline persistence
        and model serialization.

        Args:
            deep: If True, returns parameters for sub-estimators (not applicable here
                but maintained for sklearn compatibility).

        Returns:
            Dictionary of parameter names mapped to their values, including:
            - Constructor parameters extracted from __init__ signature
            - Fitted parameters (if estimator is fitted) mapped from attributes
            - Class metadata (class_, module_) for automatic reconstruction

        Example:
            >>> binner = EqualWidthBinning(n_bins=5)
            >>> params = binner.get_params()
            >>> print(params)
            {'n_bins': 5, 'clip': None, ..., 'class_': 'EqualWidthBinning', 'module_': '...'}
            >>>
            >>> binner.fit(X)
            >>> fitted_params = binner.get_params()
            >>> # Now includes: {'bin_edges': {...}, 'bin_representatives': {...}, ...}

        Note:
            - Automatically extracts constructor parameters from __init__ signature
            - Includes fitted parameters only when estimator is fitted
            - Adds class metadata for reconstruction workflows
            - Excludes internal sklearn attributes like n_features_in_
            - class_ and module_ parameters are handled specially during set_params
        """

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
        """Extract fitted parameters from instance attributes for reconstruction.

        This method scans the instance for fitted attributes (following the sklearn
        convention of ending with underscore) and prepares them for inclusion in
        get_params output. It maps fitted attribute names to parameter names for
        reconstruction compatibility.

        Returns:
            Dictionary mapping parameter names (without underscores) to their
            fitted values. Only includes non-None attributes that follow the
            fitted attribute naming convention.

        Example:
            >>> # Instance has: bin_edges_ = {...}, bin_representatives_ = {...}
            >>> fitted_params = binner._extract_fitted_params()
            >>> # Returns: {'bin_edges': {...}, 'bin_representatives': {...}}

        Note:
            - Only includes attributes ending with single underscore
            - Excludes private attributes (starting with underscore)
            - Excludes special sklearn attributes (n_features_in_, feature_names_in_)
            - Maps attribute names by removing the trailing underscore
            - Used internally by get_params when estimator is fitted
            - Enables complete object reconstruction through set_params
        """
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
            # but when set via set_params during reconstruction, they should be treated as
            # fitted params
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
            # Filter out bin_edges/bin_representatives/bin_spec if they're None
            # (don't override constructor defaults)
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
        """Validate parameters - should be overridden in subclasses.

        This method provides a hook for parameter validation that subclasses can
        override to implement their specific validation logic. The base implementation
        does nothing, making validation optional for subclasses.

        Note:
            - Called during initialization to validate parameter settings
            - Should raise appropriate exceptions for invalid parameter combinations
            - Subclasses should call super()._validate_params() if they override this
            - Used to catch configuration errors early in the initialization process
        """

    def _more_tags(self) -> dict[str, Any]:
        """Provide sklearn compatibility tags for metadata and testing.

        This method returns metadata tags that inform sklearn about the capabilities
        and requirements of this estimator type. These tags are used by sklearn's
        testing framework and other compatibility tools.

        Returns:
            Dictionary of capability tags:
            - requires_fit: True (must call fit before transform)
            - requires_y: False (doesn't require target data for basic fitting)
            - X_types: ["2darray"] (expects 2D numpy arrays)
            - allow_nan: True (can handle NaN values in input)
            - stateless: False (maintains fitted state)

        Note:
            - Used by sklearn's check_estimator and testing utilities
            - Helps ensure proper integration with sklearn ecosystem
            - Some binning methods may override requires_y (e.g., supervised binning)
            - Tags inform sklearn about estimator capabilities and constraints
        """
        return {
            "requires_fit": True,
            "requires_y": False,
            "X_types": ["2darray"],
            "allow_nan": True,
            "stateless": False,
        }
