"""
Enhanced data handling mixin with comprehensive multi-format support.

This module provides the DataHandlingMixin class that handles all data format
management including feature names, column handling, input/output processing,
and format preservation across pandas, polars, and numpy.
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
    - Feature names and counts management
    - Column name preservation and management
    - Input validation and preparation
    - Output format preservation

    The mixin automatically handles format detection, conversion, and preservation
    while maintaining a consistent internal numpy array representation for
    processing algorithms.

    Key Responsibilities:
    - Feature name extraction and validation
    - Input/output format consistency
    - Column management
    - Sklearn feature compatibility
    """

    def __init__(self):
        """Initialize data handling mixin."""
        # Column management attributes
        self._original_columns: OptionalColumnList = None

    @property
    def feature_names_in_(self) -> list[str] | None:
        """Get feature names derived from fitted data."""
        # Try to get from bin_edges_ keys
        bin_edges = getattr(self, "bin_edges_", {})
        if bin_edges:
            # Get all columns (binning + guidance)
            binning_columns = list(bin_edges.keys())
            guidance_columns = getattr(self, "guidance_columns", None)

            if guidance_columns is None:
                return binning_columns

            # Add guidance columns if they exist
            guidance_cols = (
                guidance_columns if isinstance(guidance_columns, list) else [guidance_columns]
            )
            all_columns = binning_columns + guidance_cols
            return all_columns

        # Fallback to stored value if available
        return getattr(self, "_feature_names_in", None)

    @feature_names_in_.setter
    def feature_names_in_(self, value: list[str] | None) -> None:
        """Set feature names (used during fitting)."""
        self._feature_names_in = value

    @property
    def n_features_in_(self) -> int:
        """Get number of features derived from fitted data."""
        feature_names = self.feature_names_in_
        return len(feature_names) if feature_names else 0

    @n_features_in_.setter
    def n_features_in_(self, value: int) -> None:
        """Set number of features (used during fitting)."""
        self._n_features_in = value

    @property
    def _feature_names_in(self) -> list[str] | None:
        """Internal feature names storage."""
        return getattr(self, "_internal_feature_names_in", None)

    @_feature_names_in.setter
    def _feature_names_in(self, value: list[str] | None) -> None:
        """Set internal feature names."""
        self._internal_feature_names_in = value

    @property
    def _n_features_in(self) -> int | None:
        """Internal feature count storage."""
        return getattr(self, "_internal_n_features_in", None)

    @_n_features_in.setter
    def _n_features_in(self, value: int | None) -> None:
        """Set internal feature count."""
        self._internal_n_features_in = value

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
            List of output feature names.
        """
        fitted = getattr(self, "_fitted", False)
        if not fitted:
            raise ValueError("This estimator is not fitted yet. Call 'fit' first.")

        if input_features is None:
            input_features = self._feature_names_in or []
            if not input_features:
                n_features = self._n_features_in or 0
                input_features = [f"x{i}" for i in range(n_features)]

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
