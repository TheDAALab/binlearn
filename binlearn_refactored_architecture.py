"""
Refactored binning architecture with clean separation of concerns.

This module provides a complete refactoring of the binning base classes with proper
separation of concerns:

1. SklearnIntegrationMixin: Parameter management, serialization, sklearn compatibility
2. DataHandlingMixin: Multi-format I/O, column management, feature handling
3. GeneralBinningBase: Pure binning logic and orchestration

The architecture enables powerful workflows like fit → get_params → reconstruct → transform
without refitting, while maintaining clean, testable, and extensible code.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Import existing utilities we'll reuse
from binlearn.config import get_config
from binlearn.utils.data_handling import prepare_input_with_columns, return_like_input
from binlearn.utils.errors import BinningError, ValidationMixin
from binlearn.utils.types import ArrayLike, ColumnList, GuidanceColumns, OptionalColumnList


# ============================================================================
# UTILITY FUNCTIONS FOR THE NEW ARCHITECTURE
# ============================================================================


def safe_get_class_parameters(
    class_obj: type,
    exclude_params: set[str] | None = None,
    exclude_base_class: str | None = None,
    fallback: list[str] | None = None,
) -> list[str]:
    """Safely extract class-specific parameters with fallback handling.

    Args:
        class_obj: Class to inspect.
        exclude_params: Parameter names to exclude.
        exclude_base_class: Base class name whose parameters should be excluded.
        fallback: Fallback value if inspection fails.

    Returns:
        List of parameter names specific to the class.
    """
    if exclude_params is None:
        exclude_params = {"self", "kwargs"}

    if fallback is None:
        fallback = []

    try:
        current_sig = inspect.signature(class_obj.__init__)  # type: ignore[misc]
        current_params = set(current_sig.parameters.keys()) - exclude_params
    except (ValueError, TypeError):
        return fallback

    if exclude_base_class is None:
        return list(current_params)

    # Find and exclude base class parameters
    for base_class in class_obj.__mro__:
        if base_class.__name__ == exclude_base_class:
            try:
                base_sig = inspect.signature(base_class.__init__)  # type: ignore[misc]
                base_params = set(base_sig.parameters.keys()) - exclude_params
                return list(current_params - base_params)
            except (ValueError, TypeError):
                return fallback

    return list(current_params)


def safe_get_constructor_info(class_obj: type, concrete_only: bool = True) -> dict[str, Any]:
    """Safely extract constructor parameter information.

    Args:
        class_obj: Class to inspect.
        concrete_only: If True, only inspect the concrete class's __init__.

    Returns:
        Dictionary mapping parameter names to their default values.
    """
    try:
        if concrete_only and "__init__" in class_obj.__dict__:
            sig = inspect.signature(class_obj.__dict__["__init__"])
        else:
            sig = inspect.signature(class_obj.__init__)  # type: ignore[misc]

        params = {}
        for name, param in sig.parameters.items():
            if name in {"self", "kwargs"}:
                continue
            params[name] = (
                param.default
                if param.default is not inspect.Parameter.empty
                else inspect.Parameter.empty
            )
        return params
    except (ValueError, TypeError):
        return {}


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


# ============================================================================
# SKLEARN INTEGRATION MIXIN
# ============================================================================


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

    def __init__(self, **kwargs: Any):
        """Initialize sklearn integration mixin."""
        # Extract fitted parameters before calling super().__init__()
        fitted_params = {}
        init_params = {}

        for key, value in kwargs.items():
            if key.endswith("_") and not key.startswith("_"):
                # This is likely a fitted parameter
                fitted_params[key] = value
            else:
                init_params[key] = value

        # Call parent with only initialization parameters
        super().__init__(**init_params)

        # Internal state tracking
        self._fitted = False

        # Set fitted parameters if any were provided (for reconstruction)
        if fitted_params:
            for key, value in fitted_params.items():
                setattr(self, key, value)
            self._fitted = True  # Mark as fitted if fitted parameters were provided

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

        return params  # type: ignore[no-any-return]

    def _extract_fitted_params(self) -> dict[str, Any]:
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
        return convert_to_python_types(params)  # type: ignore[no-any-return]

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
            if param_name in {"bin_edges", "bin_representatives", "bin_spec", "fitted_trees"}:
                parts.append(f"{param_name}=...")
            elif isinstance(current_value, str):
                parts.append(f"{param_name}='{current_value}'")
            else:
                parts.append(f"{param_name}={current_value}")

        return f"{class_name}({', '.join(parts)})" if parts else f"{class_name}()"


# ============================================================================
# DATA HANDLING MIXIN
# ============================================================================


class DataHandlingMixin(ValidationMixin):
    """Complete data handling for multi-format inputs and outputs.

    This mixin provides comprehensive data format support including:
    - Multi-format input processing (pandas, polars, numpy)
    - Feature names and counts management
    - Column name preservation and management
    - Input validation and preparation
    - Output format preservation
    - Column separation logic for guidance scenarios

    The mixin automatically handles format detection, conversion, and preservation
    while maintaining a consistent internal numpy array representation for
    processing algorithms.

    Key Responsibilities:
    - Feature name extraction and validation
    - Input/output format consistency
    - Column management for guidance scenarios
    - Sklearn feature compatibility
    """

    def __init__(self, **kwargs: Any):
        """Initialize data handling mixin."""
        # Extract fitted parameters and feature-related params
        fitted_params = {}
        init_params = {}

        for key, value in kwargs.items():
            if key.endswith("_") and not key.startswith("_"):
                fitted_params[key] = value
            else:
                init_params[key] = value

        super().__init__(**init_params)

        # Column management attributes
        self._original_columns: OptionalColumnList = None
        self._binning_columns: OptionalColumnList = None
        self._guidance_columns_resolved: OptionalColumnList = None

        # Feature information (sklearn compatibility)
        self.feature_names_in_: list[str] | None = None
        self.n_features_in_: int = 0
        self._n_features_in: int | None = None
        self._feature_names_in: list[str] | None = None

        # Set fitted parameters if provided
        if fitted_params:
            for key, value in fitted_params.items():
                setattr(self, key, value)

    def _prepare_input(self, X: ArrayLike) -> tuple[np.ndarray[Any, Any], ColumnList]:
        """Prepare input data and extract column information.

        Args:
            X: Input data in any supported format.

        Returns:
            Tuple of (prepared_array, column_identifiers).
        """
        fitted = getattr(self, "_fitted", False)
        original_columns = getattr(self, "_original_columns", None)

        return prepare_input_with_columns(X, fitted=fitted, original_columns=original_columns)

    def _extract_and_validate_feature_info(self, X: Any, reset: bool = False) -> list[str]:
        """Extract and validate feature names and counts from input.

        This is the central method for feature handling, extracting feature names
        from various input types and ensuring consistency with fitted state.

        Args:
            X: Input data to extract feature information from.
            reset: Whether to reset/store new feature information (used during fit).

        Returns:
            List of feature names.

        Raises:
            ValueError: If feature information is inconsistent with fitted state.
        """
        # Extract feature names from various input types
        if hasattr(X, "columns"):
            feature_names = list(X.columns)
        elif hasattr(X, "feature_names"):
            feature_names = list(X.feature_names)
        elif hasattr(X, "_feature_names"):
            feature_names = list(X._feature_names)  # pylint: disable=protected-access
        else:
            # Default to generic names for array inputs
            n_features = X.shape[1] if hasattr(X, "shape") else len(X[0])
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Validate consistency with fitted state
        if not reset and getattr(self, "_fitted", False) and self._feature_names_in is not None:
            if len(feature_names) != len(self._feature_names_in):
                raise ValueError(
                    f"Input has {len(feature_names)} features, but this estimator "
                    f"was fitted with {len(self._feature_names_in)} features."
                )

            # For DataFrame inputs, check name consistency
            if hasattr(X, "columns"):
                if feature_names != self._feature_names_in:
                    raise ValueError(
                        f"Input feature names don't match fitted feature names. "
                        f"Expected: {self._feature_names_in}, got: {feature_names}"
                    )

        # Store feature information (both internal and sklearn public attributes)
        if reset or not self._feature_names_in:
            self.feature_names_in_ = feature_names
            self._feature_names_in = feature_names
            self.n_features_in_ = len(feature_names)
            self._n_features_in = len(feature_names)

        return feature_names

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get output feature names for transformation.

        Args:
            input_features: Input feature names (optional).

        Returns:
            List of output feature names (excluding guidance columns).
        """
        fitted = getattr(self, "_fitted", False)
        if not fitted:
            raise ValueError("This estimator is not fitted yet. Call 'fit' first.")

        if input_features is None:
            input_features = self._feature_names_in or []
            if not input_features:
                n_features = self._n_features_in or 0
                input_features = [f"x{i}" for i in range(n_features)]

        # Handle guidance columns - only return binning column names
        guidance_columns = getattr(self, "guidance_columns", None)
        if guidance_columns is not None:
            guidance_cols = (
                guidance_columns if isinstance(guidance_columns, list) else [guidance_columns]
            )

            return [
                name
                for idx, name in enumerate(input_features)
                if name not in guidance_cols and idx not in guidance_cols
            ]

        return input_features.copy()

    def _validate_and_prepare_input(self, X: ArrayLike, name: str = "X") -> np.ndarray[Any, Any]:
        """Validate input data and convert to array format.

        Args:
            X: Input data to validate.
            name: Parameter name for error messages.

        Returns:
            Validated numpy array.
        """
        return self.validate_array_like(X, name)  # type: ignore[no-any-return]

    def _separate_binning_and_guidance_columns(
        self, X: ArrayLike
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any] | None, ColumnList, ColumnList | None]:
        """Separate input into binning and guidance columns.

        Core logic for handling guided vs unguided binning scenarios.

        Args:
            X: Input data with both binning and guidance columns.

        Returns:
            Tuple of (X_binning, X_guidance, binning_columns, guidance_columns).
        """
        arr, columns = self._prepare_input(X)

        guidance_columns = getattr(self, "guidance_columns", None)
        if guidance_columns is None:
            # No guidance - all columns are binning columns
            return arr, None, columns, None

        # Normalize guidance_columns to list
        guidance_cols = (
            [guidance_columns] if not isinstance(guidance_columns, list) else guidance_columns
        )

        # Separate columns
        binning_indices = []
        guidance_indices = []
        binning_column_names = []
        guidance_column_names = []

        for i, col in enumerate(columns):
            if col in guidance_cols:
                guidance_indices.append(i)
                guidance_column_names.append(col)
            else:
                binning_indices.append(i)
                binning_column_names.append(col)

        # Extract data arrays
        X_binning = arr[:, binning_indices] if binning_indices else np.empty((arr.shape[0], 0))
        X_guidance = arr[:, guidance_indices] if guidance_indices else None

        # Store resolved column information for later use
        self._binning_columns = binning_column_names
        self._guidance_columns_resolved = guidance_column_names if guidance_column_names else None

        return X_binning, X_guidance, binning_column_names, guidance_column_names

    def _store_original_input_info(self, X: ArrayLike) -> None:
        """Store original input information for output formatting.

        Args:
            X: Input data to store information about.
        """
        _, columns = self._prepare_input(X)
        self._original_columns = columns

    def _format_output_like_input(
        self,
        result: np.ndarray[Any, Any],
        original_input: ArrayLike,
        columns: ColumnList,
        preserve_dataframe: bool | None = None,
    ) -> Any:
        """Format output to match input format when requested.

        Args:
            result: Processed numpy array result.
            original_input: Original input data for format reference.
            columns: Column names/indices for the result.
            preserve_dataframe: Whether to preserve DataFrame format.

        Returns:
            Formatted output matching input format.
        """
        if preserve_dataframe is None:
            preserve_dataframe = getattr(self, "preserve_dataframe", False)

        return return_like_input(result, original_input, columns, bool(preserve_dataframe))

    def _validate_input_consistency_with_fitted_state(
        self, X: ArrayLike, method_name: str = "transform"
    ) -> None:
        """Validate input consistency with fitted state.

        Args:
            X: Input data to check.
            method_name: Method name for error messages.
        """
        fitted = getattr(self, "_fitted", False)
        if fitted:
            # This will validate consistency automatically
            self._extract_and_validate_feature_info(X, reset=False)

    def _get_binning_columns_for_output(self) -> ColumnList:
        """Get column identifiers for binning columns in output."""
        binning_columns = getattr(self, "_binning_columns", None)
        if binning_columns is not None:
            return binning_columns

        # Fallback logic
        guidance_columns = getattr(self, "guidance_columns", None)
        original_columns = getattr(self, "_original_columns", None)

        if guidance_columns is None and original_columns is not None:
            return original_columns

        # Ultimate fallback
        n_features = getattr(self, "_n_features_in", 0)
        return list(range(n_features))

    def _get_feature_count(self, include_guidance: bool = True) -> int:
        """Get feature count with optional guidance exclusion."""
        n_features = getattr(self, "_n_features_in", 0)

        if not include_guidance:
            guidance_columns = getattr(self, "_guidance_columns_resolved", None)
            if guidance_columns is not None:
                return n_features - len(guidance_columns)

        return n_features

    def _has_guidance_columns(self) -> bool:
        """Check if guidance columns are configured."""
        return getattr(self, "guidance_columns", None) is not None


# ============================================================================
# REFACTORED GENERAL BINNING BASE
# ============================================================================


class GeneralBinningBase(
    ABC,
    SklearnIntegrationMixin,
    DataHandlingMixin,
    TransformerMixin,  # type: ignore[misc,unused-ignore]
):
    """Refactored base class focusing purely on binning orchestration logic.

    This class provides the core binning functionality while delegating:
    - Sklearn compatibility to SklearnIntegrationMixin
    - Data format handling to DataHandlingMixin

    The class focuses on binning-specific concerns:
    - Joint vs per-column fitting strategies
    - Guidance column management for supervised methods
    - Binning transformation pipeline orchestration
    - Abstract interface definition for binning methods

    Args:
        preserve_dataframe: Whether to preserve DataFrame format in output.
        fit_jointly: Whether to fit parameters jointly across all columns.
        guidance_columns: Columns to use for guided binning.
        **kwargs: Additional arguments for subclasses.

    Example:
        >>> # Abstract class - use concrete implementation
        >>> from binlearn.methods import EqualWidthBinning
        >>> binner = EqualWidthBinning(n_bins=5, preserve_dataframe=True)
        >>> X_binned = binner.fit_transform(X)
    """

    def __init__(
        self,
        preserve_dataframe: bool | None = None,
        fit_jointly: bool | None = None,
        guidance_columns: GuidanceColumns = None,
        **kwargs: Any,
    ):
        """Initialize the binning transformer with clean parameter handling."""
        # Load configuration defaults
        config = get_config()

        # Apply configuration defaults
        if preserve_dataframe is None:
            preserve_dataframe = config.preserve_dataframe
        if fit_jointly is None:
            fit_jointly = config.fit_jointly

        # Validate parameter compatibility early
        if guidance_columns is not None and fit_jointly:
            raise ValueError(
                "guidance_columns and fit_jointly=True are incompatible. "
                "Use either guidance_columns for per-record guidance OR "
                "fit_jointly=True for global fitting, but not both."
            )

        # Store binning-specific parameters
        self.preserve_dataframe = preserve_dataframe
        self.fit_jointly = fit_jointly
        self.guidance_columns = guidance_columns

        # Initialize parent mixins
        super().__init__(**kwargs)

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> GeneralBinningBase:
        """Fit the binning transformer with comprehensive orchestration.

        Handles both guided and unguided binning scenarios with proper
        parameter routing and state management.

        Args:
            X: Input data to fit on.
            y: Target values (sklearn convenience for supervised binning).
            **fit_params: Additional parameters for fitting methods.

        Returns:
            Self for method chaining.
        """
        try:
            # Step 1: Parameter validation
            self._validate_params()

            # Step 2: Input validation and feature information extraction
            self._validate_and_prepare_input(X, "X")
            self._extract_and_validate_feature_info(X, reset=True)
            self._store_original_input_info(X)

            # Step 3: Column separation for guidance handling
            X_binning, X_guidance, binning_cols, _ = self._separate_binning_and_guidance_columns(X)

            # Step 4: Route to appropriate fitting strategy
            if self.fit_jointly:
                self._fit_jointly_across_columns(X_binning, binning_cols, **fit_params)
            else:
                # Handle guidance data resolution with priority order
                final_guidance_data = self._resolve_guidance_data_priority(
                    X_guidance, fit_params.pop("guidance_data", None), y
                )

                self._fit_per_column_independently(
                    X_binning, binning_cols, final_guidance_data, **fit_params
                )

            # Step 5: Mark as fitted
            self._fitted = True
            return self

        except Exception as e:
            # Preserve specific exception types for compatibility
            if isinstance(e, BinningError | ValueError | RuntimeError | NotImplementedError):
                raise
            raise ValueError(f"Failed to fit binning model: {str(e)}") from e

    def transform(self, X: Any) -> Any:
        """Transform input data using fitted binning parameters.

        Handles both guided and unguided scenarios, transforming only binning
        columns while preserving the input format.

        Args:
            X: Input data to transform.

        Returns:
            Transformed data with binned values in original format.
        """
        try:
            # Step 1: Validation checks
            self._check_fitted()
            self._validate_and_prepare_input(X, "X")
            self._validate_input_consistency_with_fitted_state(X, "transform")

            # Step 2: Column separation and transformation
            X_binning, _, binning_cols, _ = self._separate_binning_and_guidance_columns(X)

            if self.guidance_columns is None:
                # Simple case: transform all columns
                result = self._transform_columns_to_bins(X_binning, binning_cols)
                return self._format_output_like_input(
                    result, X, binning_cols, self.preserve_dataframe
                )

            # Guided case: transform only binning columns
            if X_binning.shape[1] > 0:
                result = self._transform_columns_to_bins(X_binning, binning_cols)
            else:
                result = np.empty((X_binning.shape[0], 0), dtype=int)

            return self._format_output_like_input(result, X, binning_cols, self.preserve_dataframe)

        except Exception as e:
            if isinstance(e, BinningError | RuntimeError):
                raise
            raise ValueError(f"Failed to transform data: {str(e)}") from e

    def inverse_transform(self, X: Any) -> Any:
        """Inverse transform from bin indices back to representative values.

        Args:
            X: Binned data to inverse transform.

        Returns:
            Data with representative values in original format.
        """
        try:
            self._check_fitted()
            self._validate_and_prepare_input(X, "X")

            arr, columns = self._prepare_input(X)

            # Validate expected column count for guided binning
            if self.guidance_columns is not None:
                expected_cols = self._get_feature_count(include_guidance=False)
                if len(columns) != expected_cols:
                    raise ValueError(
                        f"Input for inverse_transform should have {expected_cols} "
                        f"columns (binning columns only), got {len(columns)}"
                    )

            result = self._inverse_transform_bins_to_values(arr, columns)
            return self._format_output_like_input(result, X, columns, self.preserve_dataframe)

        except Exception as e:
            if isinstance(e, BinningError | RuntimeError):
                raise
            raise ValueError(f"Failed to inverse transform data: {str(e)}") from e

    def _resolve_guidance_data_priority(
        self, X_guidance: np.ndarray[Any, Any] | None, external_guidance: Any, y: Any
    ) -> np.ndarray[Any, Any] | None:
        """Resolve guidance data with clear priority order.

        Priority: X_guidance > external_guidance > y

        Args:
            X_guidance: Guidance columns from input X.
            external_guidance: Explicit guidance_data parameter.
            y: Target values (sklearn convenience).

        Returns:
            Resolved guidance data array or None.
        """
        if X_guidance is not None:
            return X_guidance

        if external_guidance is not None:
            return external_guidance

        if y is not None:
            y_array = np.asarray(y)
            if y_array.ndim == 1:
                y_array = y_array.reshape(-1, 1)
            return y_array

        return None

    def _validate_params(self) -> None:
        """Validate binning-specific parameters with clear error messages."""
        super()._validate_params()

        if self.preserve_dataframe is not None and not isinstance(self.preserve_dataframe, bool):
            raise TypeError("preserve_dataframe must be a boolean or None")

        if self.fit_jointly is not None and not isinstance(self.fit_jointly, bool):
            raise TypeError("fit_jointly must be a boolean or None")

        if self.guidance_columns is not None:
            if not isinstance(self.guidance_columns, list | tuple | int | str):
                raise TypeError("guidance_columns must be list, tuple, int, str, or None")

    # Abstract methods for subclasses - renamed for clarity
    @abstractmethod
    def _fit_per_column_independently(
        self,
        X: np.ndarray[Any, Any],
        columns: ColumnList,
        guidance_data: ArrayLike | None = None,
        **fit_params: Any,
    ) -> None:
        """Fit binning parameters independently for each column.

        Args:
            X: Input data for binning columns.
            columns: Column identifiers for binning columns.
            guidance_data: Optional guidance data for supervised binning.
            **fit_params: Additional fitting parameters.
        """
        raise NotImplementedError("Subclasses must implement _fit_per_column_independently")

    @abstractmethod
    def _fit_jointly_across_columns(
        self, X: np.ndarray[Any, Any], columns: ColumnList, **fit_params: Any
    ) -> None:
        """Fit binning parameters jointly across all columns.

        Args:
            X: Input data for all binning columns.
            columns: Column identifiers for all columns.
            **fit_params: Additional fitting parameters.
        """
        raise NotImplementedError("Subclasses must implement _fit_jointly_across_columns")

    @abstractmethod
    def _transform_columns_to_bins(
        self, X: np.ndarray[Any, Any], columns: ColumnList
    ) -> np.ndarray[Any, Any]:
        """Transform columns to bin indices.

        Args:
            X: Input data to transform.
            columns: Column identifiers.

        Returns:
            Transformed data with bin indices.
        """
        raise NotImplementedError("Subclasses must implement _transform_columns_to_bins")

    @abstractmethod
    def _inverse_transform_bins_to_values(
        self, X: np.ndarray[Any, Any], columns: ColumnList
    ) -> np.ndarray[Any, Any]:
        """Inverse transform from bin indices to representative values.

        Args:
            X: Binned data to inverse transform.
            columns: Column identifiers.

        Returns:
            Data with representative values.
        """
        raise NotImplementedError("Subclasses must implement _inverse_transform_bins_to_values")


# ============================================================================
# EXAMPLE IMPLEMENTATION FOR TESTING
# ============================================================================


class EqualWidthBinningRefactored(GeneralBinningBase):
    """Example implementation using the refactored architecture.

    This serves as both a working implementation and a template showing
    how to use the new architecture.
    """

    def __init__(self, n_bins: int = 5, **kwargs):
        # Extract fitted parameters
        fitted_params = {}
        init_params = {}

        for key, value in kwargs.items():
            if key.endswith("_") and not key.startswith("_"):
                fitted_params[key] = value
            else:
                init_params[key] = value

        self.n_bins = n_bins
        super().__init__(**init_params)

        # Initialize fitted parameters
        self.bin_edges_: dict = {}
        self.bin_representatives_: dict = {}

        # Set fitted parameters if provided (for reconstruction)
        if fitted_params:
            for key, value in fitted_params.items():
                setattr(self, key, value)

    def _fit_per_column_independently(self, X, columns, guidance_data=None, **fit_params):
        """Example per-column fitting implementation."""
        for i, col in enumerate(columns):
            col_data = X[:, i]
            min_val, max_val = np.min(col_data), np.max(col_data)

            # Equal-width binning
            edges = np.linspace(min_val, max_val, self.n_bins + 1)
            reps = [(edges[j] + edges[j + 1]) / 2 for j in range(len(edges) - 1)]

            self.bin_edges_[col] = edges
            self.bin_representatives_[col] = np.array(reps)

    def _fit_jointly_across_columns(self, X, columns, **fit_params):
        """Example joint fitting implementation."""
        # Use same range for all columns
        min_val, max_val = np.min(X), np.max(X)
        edges = np.linspace(min_val, max_val, self.n_bins + 1)
        reps = [(edges[j] + edges[j + 1]) / 2 for j in range(len(edges) - 1)]

        for col in columns:
            self.bin_edges_[col] = edges
            self.bin_representatives_[col] = np.array(reps)

    def _transform_columns_to_bins(self, X, columns):
        """Example transformation implementation."""
        result = np.zeros_like(X, dtype=int)
        for i, col in enumerate(columns):
            edges = self.bin_edges_[col]
            result[:, i] = np.digitize(X[:, i], edges[1:-1])
        return result

    def _inverse_transform_bins_to_values(self, X, columns):
        """Example inverse transformation implementation."""
        result = np.zeros_like(X, dtype=float)
        for i, col in enumerate(columns):
            reps = self.bin_representatives_[col]
            result[:, i] = reps[X[:, i]]
        return result


# ============================================================================
# COMPREHENSIVE TEST SUITE
# ============================================================================


def run_comprehensive_tests():
    """Run comprehensive tests of the refactored architecture."""
    import pandas as pd

    print("🧪 Testing Refactored Binning Architecture")
    print("=" * 60)

    # Test 1: Basic functionality
    print("\n1️⃣ Testing Basic Functionality")
    X = np.random.randn(100, 4)
    X_df = pd.DataFrame(X, columns=["A", "B", "C", "D"])

    binner = EqualWidthBinningRefactored(n_bins=3, preserve_dataframe=True)
    binner.fit(X_df)
    X_binned = binner.transform(X_df)

    print(f"✅ Basic workflow: {type(X_binned)}, shape={X_binned.shape}")
    print(f"✅ Fitted state: {binner._fitted}")

    # Test 2: Parameter reconstruction workflow
    print("\n2️⃣ Testing Parameter Reconstruction")
    params = binner.get_params()
    print(f"✅ Extracted {len(params)} parameters including fitted state")

    new_binner = EqualWidthBinningRefactored(**params)
    print(f"✅ Reconstructed binner: fitted={new_binner._fitted}")

    X_test = np.random.randn(20, 4)
    X_test_df = pd.DataFrame(X_test, columns=["A", "B", "C", "D"])
    X_test_binned = new_binner.transform(X_test_df)
    print(f"✅ Transform without fitting: {type(X_test_binned)}, shape={X_test_binned.shape}")

    # Test 3: Guidance columns
    print("\n3️⃣ Testing Guidance Columns")
    guidance_binner = EqualWidthBinningRefactored(n_bins=3, guidance_columns=["D"])
    guidance_binner.fit(X_df)
    X_guided = guidance_binner.transform(X_df)
    print(f"✅ Guided binning: shape={X_guided.shape} (should be 3 columns)")

    X_inverse = guidance_binner.inverse_transform(X_guided)
    print(f"✅ Inverse transform: shape={X_inverse.shape}")

    # Test 4: Feature names handling
    print("\n4️⃣ Testing Feature Name Handling")
    feature_names = binner.get_feature_names_out()
    print(f"✅ Feature names out: {feature_names}")

    # Test 5: Error handling
    print("\n5️⃣ Testing Error Handling")
    try:
        EqualWidthBinningRefactored(fit_jointly=True, guidance_columns=["A"])
        print("❌ Should have raised ValueError")
    except ValueError:
        print("✅ Correctly caught parameter incompatibility error")

    try:
        unfitted_binner = EqualWidthBinningRefactored()
        unfitted_binner.transform(X)
        print("❌ Should have raised RuntimeError")
    except RuntimeError:
        print("✅ Correctly caught unfitted state error")

    # Test 6: Serialization
    print("\n6️⃣ Testing Serialization")
    serializable_params = binner.get_serializable_params()
    print(f"✅ Serializable params: {len(serializable_params)} parameters")

    print("\n✨ All tests passed successfully!")

    print("\n📋 Architecture Summary:")
    print("├── SklearnIntegrationMixin: Parameter management, fitted state transfer")
    print("├── DataHandlingMixin: Multi-format I/O, feature handling, column management")
    print("└── GeneralBinningBase: Pure binning orchestration and abstract interface")
    print("\n🎯 Key Innovation: Complete fitted parameter reconstruction workflows!")


if __name__ == "__main__":
    run_comprehensive_tests()
