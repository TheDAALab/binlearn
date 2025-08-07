"""
Data handling mixin for multi-format input/output processing.

This mixin handles all data format management including pandas, polars, and numpy
support, column management, input validation, and output format preservation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..utils.data_handling import prepare_input_with_columns, return_like_input
from ..utils.errors import ValidationMixin
from ..utils.types import ArrayLike, ColumnList, OptionalColumnList


class DataHandlingMixin(ValidationMixin):
    """Complete data handling for multi-format inputs and outputs.

    This mixin provides comprehensive data format support including:
    - Multi-format input processing (pandas, polars, numpy)
    - Column name preservation and management
    - Input validation and preparation
    - Output format preservation
    - Column separation logic for guidance scenarios

    The mixin automatically handles format detection, conversion, and preservation
    while maintaining a consistent internal numpy array representation for
    processing algorithms.

    Features:
    - Automatic format detection and conversion
    - Column name/index preservation
    - Guidance vs binning column separation
    - Consistent input validation
    - Format-preserving output generation
    """

    def __init__(self, **kwargs: Any):
        """Initialize data handling mixin."""
        # DataHandlingMixin inherits from ValidationMixin
        ValidationMixin.__init__(self)

        # Column management attributes
        self._original_columns: OptionalColumnList = None
        self._binning_columns: OptionalColumnList = None
        self._guidance_columns_resolved: OptionalColumnList = None

        # Feature information (sklearn compatibility)
        self.feature_names_in_: list[str] | None = None
        self.n_features_in_: int = 0
        self._n_features_in: int | None = None
        self._feature_names_in: list[str] | None = None

    def _prepare_input(self, X: ArrayLike) -> tuple[np.ndarray[Any, Any], ColumnList]:
        """Prepare input data and extract column information.

        Converts input data to standardized numpy array format while preserving
        column information for later output formatting. Handles pandas DataFrames,
        polars DataFrames, and numpy arrays consistently.

        Args:
            X: Input data in any supported format.

        Returns:
            Tuple containing:
            - Prepared numpy array with standardized format
            - Column identifiers (names for DataFrames, indices for arrays)
        """
        fitted = getattr(self, "_fitted", False)
        original_columns = getattr(self, "_original_columns", None)

        return prepare_input_with_columns(X, fitted=fitted, original_columns=original_columns)

    def _check_feature_names(self, X: Any, reset: bool = False) -> list[str]:
        """Check and validate feature names from input.

        Extracts feature names from various input types and validates consistency
        with previously fitted state.

        Args:
            X: Input data to extract feature names from.
            reset: Whether to reset/store new feature names (used during fit).

        Returns:
            List of feature names.

        Raises:
            ValueError: If feature names are inconsistent with fitted state.
        """
        feature_names = None

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

        # Store for sklearn compatibility (public attribute) and internal use
        if reset or not hasattr(self, "feature_names_in_") or self.feature_names_in_ is None:
            self.feature_names_in_ = feature_names
            self._feature_names_in = feature_names

        # Store n_features_in for sklearn compatibility
        if reset or not hasattr(self, "n_features_in_") or self.n_features_in_ == 0:
            self.n_features_in_ = len(feature_names)
            self._n_features_in = len(feature_names)

        return feature_names

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get output feature names for transformation.

        Args:
            input_features: Input feature names (optional).

        Returns:
            List of output feature names.

        Raises:
            ValueError: If estimator is not fitted.
        """
        fitted = getattr(self, "_fitted", False)
        if not fitted:
            raise ValueError("This estimator is not fitted yet. Call 'fit' first.")

        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
            if input_features is None:
                n_features = getattr(self, "_n_features_in", 0) or getattr(
                    self, "n_features_in_", 0
                )
                input_features = [f"x{i}" for i in range(n_features)]

        # Handle guidance columns - only return binning column names
        guidance_columns = getattr(self, "guidance_columns", None)
        if guidance_columns is not None:
            guidance_cols = (
                guidance_columns if isinstance(guidance_columns, list) else [guidance_columns]
            )

            output_features = []
            for idx, name in enumerate(input_features):
                if name not in guidance_cols and idx not in guidance_cols:
                    output_features.append(name)
            return output_features

        return input_features.copy()

    def _validate_input(self, X: ArrayLike, name: str = "X") -> np.ndarray[Any, Any]:
        """Validate input data and convert to array format.

        Args:
            X: Input data to validate.
            name: Name of the parameter for error messages.

        Returns:
            Validated numpy array.

        Raises:
            InvalidDataError: If input validation fails.
        """

    def _validate_input(self, X: ArrayLike, name: str = "X") -> np.ndarray[Any, Any]:
        """Validate input data and convert to array format.

        Args:
            X: Input data to validate.
            name: Name of the parameter for error messages.

        Returns:
            Validated numpy array.

        Raises:
            InvalidDataError: If input validation fails.
        """
        return self.validate_array_like(X, name)  # type: ignore[no-any-return]

    def _separate_columns(
        self, X: ArrayLike
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any] | None, ColumnList, ColumnList | None]:
        """Separate input into binning and guidance columns.

        Core logic for handling guided vs unguided binning scenarios. When
        guidance_columns is specified, splits data into separate arrays for
        binning and guidance. Otherwise, treats all columns as binning columns.

        Args:
            X: Input data with both binning and guidance columns.

        Returns:
            Tuple containing:
            - X_binning: Data for columns to be binned
            - X_guidance: Data for guidance columns (None if no guidance)
            - binning_columns: Names/indices of binning columns
            - guidance_columns: Names/indices of guidance columns (None if no guidance)
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

        # Store resolved column information
        self._binning_columns = binning_column_names
        self._guidance_columns_resolved = guidance_column_names if guidance_column_names else None

        return X_binning, X_guidance, binning_column_names, guidance_column_names

    def _store_input_info(self, X: ArrayLike) -> None:
        """Store input information for later use in output formatting.

        Args:
            X: Input data to store information about.
        """
        _, columns = self._prepare_input(X)
        self._original_columns = columns

    def _format_output(
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
            Formatted output matching input format when preserve_dataframe=True.
        """
        if preserve_dataframe is None:
            preserve_dataframe = getattr(self, "preserve_dataframe", False)

        return return_like_input(result, original_input, columns, bool(preserve_dataframe))

    def _check_input_consistency(self, X: ArrayLike, method_name: str = "transform") -> None:
        """Check input consistency with fitted state.

        Validates that input data is consistent with the data used during fitting
        in terms of number of features and feature names.

        Args:
            X: Input data to check.
            method_name: Name of the method being called (for error messages).

        Raises:
            ValueError: If input is inconsistent with fitted state.
        """
        fitted = getattr(self, "_fitted", False)
        if not fitted:
            return  # No consistency check needed for unfitted estimator

        # Use feature name checking which includes consistency validation
        self._check_feature_names(X, reset=False)

    def _get_binning_columns_for_output(self) -> ColumnList:
        """Get column identifiers for binning columns in output.

        Returns:
            Column identifiers for binning columns only.
        """
        binning_columns = getattr(self, "_binning_columns", None)
        if binning_columns is not None:
            return binning_columns

        # Fallback: if no guidance columns, use all original columns
        guidance_columns = getattr(self, "guidance_columns", None)
        original_columns = getattr(self, "_original_columns", None)

        if guidance_columns is None and original_columns is not None:
            return original_columns

        # Ultimate fallback
        n_features = getattr(self, "_n_features_in", 0)
        return list(range(n_features))

    def _validate_column_specification(
        self, columns: Any, data_shape: tuple[int, ...]
    ) -> list[Any]:
        """Validate column specification against data shape.

        Args:
            columns: Column specification to validate.
            data_shape: Shape of the data for validation.

        Returns:
            Validated list of column identifiers.

        Raises:
            ValueError: If column specification is invalid.
        """
        return self.validate_column_specification(columns, data_shape)  # type: ignore[no-any-return]

    def _get_column_count(self, include_guidance: bool = True) -> int:
        """Get total number of columns.

        Args:
            include_guidance: Whether to include guidance columns in count.

        Returns:
            Number of columns.
        """
        n_features = getattr(self, "_n_features_in", 0)

        if not include_guidance:
            guidance_columns = getattr(self, "_guidance_columns_resolved", None)
            if guidance_columns is not None:
                return n_features - len(guidance_columns)

        return n_features

    def _has_guidance_columns(self) -> bool:
        """Check if guidance columns are configured.

        Returns:
            True if guidance columns are specified.
        """
        return getattr(self, "guidance_columns", None) is not None

    def _get_guidance_info(self) -> tuple[bool, int]:
        """Get guidance column information.

        Returns:
            Tuple of (has_guidance, n_guidance_columns).
        """
        guidance_columns = getattr(self, "_guidance_columns_resolved", None)
        if guidance_columns is not None:
            return True, len(guidance_columns)
        return False, 0
