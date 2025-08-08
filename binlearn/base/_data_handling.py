"""
Clean data handling for V2 architecture with straight inheritance.

This module provides pure data format handling without binning-specific concerns.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._sklearn_integration import SklearnIntegration
from ._validation_mixin import ValidationMixin
from ..utils.data_handling import prepare_input_with_columns, return_like_input
from ..utils.types import ArrayLike, ColumnList, OptionalColumnList


class DataHandling(SklearnIntegration, ValidationMixin):
    """Pure data handling for multi-format inputs and outputs.

    Handles data format conversion, validation, and preservation without
    assumptions about binning, guidance, or fitted state.
    """

    def __init__(self):
        """Initialize data handling mixin."""
        SklearnIntegration.__init__(self)
        ValidationMixin.__init__(self)
        # Column management for input/output format preservation
        self._original_columns: OptionalColumnList = None

    def _prepare_input(self, X: ArrayLike) -> tuple[np.ndarray[Any, Any], ColumnList]:
        """Prepare input data and extract column information."""
        fitted = getattr(self, "_fitted", False)

        # For fitted transformers, try to use stored feature names for inverse transformation
        original_columns = None
        if fitted and hasattr(self, "_get_binning_columns"):
            # Use dynamic binning columns computation
            original_columns = getattr(self, "_get_binning_columns")()

        return prepare_input_with_columns(X, fitted=fitted, original_columns=original_columns)

    def _validate_and_prepare_input(self, X: ArrayLike, name: str = "X") -> np.ndarray[Any, Any]:
        """Validate input data and convert to array format."""
        return self.validate_array_like(X, name)  # type: ignore[no-any-return]

    def _format_output_like_input(
        self,
        result: np.ndarray[Any, Any],
        original_input: ArrayLike,
        columns: ColumnList,
        preserve_dataframe: bool | None = None,
    ) -> Any:
        """Format output to match input format when requested."""
        if preserve_dataframe is None:
            preserve_dataframe = getattr(self, "preserve_dataframe", False)
        return return_like_input(result, original_input, columns, bool(preserve_dataframe))

    def _extract_and_validate_feature_info(self, X: Any, reset: bool = False) -> list[str]:
        """Extract and validate feature names and counts from input."""
        # Extract feature names from various input types
        if hasattr(X, "columns"):
            feature_names = list(X.columns)
        elif hasattr(X, "feature_names"):
            feature_names = list(X.feature_names)
        else:
            # Default to generic names for array inputs
            n_features = X.shape[1] if hasattr(X, "shape") else len(X[0])
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Store feature information if resetting
        if reset:
            self._feature_names_in = feature_names
            self._n_features_in = len(feature_names)

        return feature_names

    @property
    def feature_names_in_(self) -> list[str] | None:
        """Get feature names."""
        return getattr(self, "_feature_names_in", None)

    @feature_names_in_.setter
    def feature_names_in_(self, value: list[str] | None) -> None:
        """Set feature names."""
        self._feature_names_in = value

    @property
    def n_features_in_(self) -> int:
        """Get number of features."""
        feature_names = self.feature_names_in_
        return len(feature_names) if feature_names else 0

    @n_features_in_.setter
    def n_features_in_(self, value: int) -> None:
        """Set number of features."""
        self._n_features_in = value

    def _validate_numeric_data(self, x_col: np.ndarray[Any, Any], col_id: Any) -> None:
        """Validate that column contains only numeric data.

        Args:
            x_col: Column data to validate
            col_id: Column identifier for error messages

        Raises:
            ValueError: If column contains non-numeric data
        """
        # Check if data is numeric type (allow NaN and inf values)
        if x_col.dtype.kind not in {"i", "u", "f", "c"}:  # integer, unsigned, float, complex
            raise ValueError(
                f"Column {col_id} contains non-numeric data (dtype: {x_col.dtype}). "
                "Only numeric data is supported."
            )

    def _validate_all_numeric_data(
        self, X: np.ndarray[Any, Any], column_names: list[Any] | None = None
    ) -> None:
        """Validate that all columns contain only numeric data.

        Args:
            X: Input data array to validate
            column_names: Optional column names for better error messages

        Raises:
            ValueError: If any column contains non-numeric data
        """
        if column_names is None:
            column_names = [f"column_{i}" for i in range(X.shape[1])]

        for i in range(X.shape[1]):
            col_name = column_names[i] if i < len(column_names) else f"column_{i}"
            self._validate_numeric_data(X[:, i], col_name)
