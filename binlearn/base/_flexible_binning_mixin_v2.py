"""Flexible binning mixin with comprehensive functionality.

This module provides the FlexibleBinningMixin class that serves as a proper
base class for binning methods using flexible bin definitions that can contain
both singleton bins (exact matches) and interval bins (range matches).
It includes constructor parameter handling and comprehensive transformation logic.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np

from ..config import get_config
from ..utils.constants import MISSING_VALUE
from ..utils.flexible_bin_operations import (
    find_flexible_bin_for_value,
    generate_default_flexible_representatives,
    get_flexible_bin_count,
    validate_flexible_bin_spec_format,
)
from ..utils.types import FlexibleBinSpec, ColumnList


class FlexibleBinningMixin:
    """Base class for binning methods that use flexible bin definitions.

    Provides comprehensive functionality for flexible binning transformers where bins
    can be either singleton bins (exact value matches) or interval bins (range matches).
    This mixin handles the flexible binning-specific logic while the parent class
    handles orchestration and infrastructure.

    Features:
    - Constructor parameters for bin_spec, bin_representatives
    - Flexible transform method supporting both singleton and interval bins
    - Comprehensive inverse_transform with representative value mapping
    - Compatibility with sklearn parameter reconstruction workflows
    - Integration with V2 architecture

    Args:
        bin_spec (FlexibleBinSpec, optional): Pre-computed bin specifications for each column.
            If provided, these specifications are used instead of calculating from data.
        bin_representatives (dict, optional): Pre-computed representative values
            for each bin. If provided along with bin_spec, these representatives are used.
        **kwargs: Additional arguments passed to parent classes.
    """

    def __init__(
        self,
        bin_spec: FlexibleBinSpec | None = None,
        bin_representatives: dict[Any, list[float]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the flexible binning base class."""
        # Store constructor parameters for sklearn compatibility
        self.bin_spec = bin_spec
        self.bin_representatives = bin_representatives

        # Initialize fitted parameters
        self.bin_spec_: dict[Any, list[Any]] = {}
        self.bin_representatives_: dict[Any, list[float]] = {}

        # Initialize parent classes with only kwargs (no flexible-specific params)
        super().__init__(**kwargs)

    def _transform_columns_to_bins(
        self, X: np.ndarray[Any, Any], columns: ColumnList
    ) -> np.ndarray[Any, Any]:
        """Transform columns to bin indices using flexible bin definitions.

        Converts values in each column to discrete bin indices using
        the fitted flexible bin specifications. Handles both singleton bins
        (exact matches) and interval bins (range matches) seamlessly.

        Args:
            X: Input data array with shape (n_samples, n_features).
            columns: List of column identifiers corresponding to X columns.

        Returns:
            Array of bin indices with same shape as input, where each
            value represents the bin index for the corresponding input value.
            Uses MISSING_VALUE for NaN inputs.
        """
        result = np.zeros(X.shape, dtype=int)
        available_keys = list(self.bin_spec_.keys())

        for i, col in enumerate(columns):
            # Find the right bin specification
            key = self._get_column_key(col, available_keys, i)
            bin_defs = self.bin_spec_[key]

            # Transform this column using flexible bin definitions
            col_data = X[:, i]
            for j, value in enumerate(col_data):
                if np.isnan(value):
                    result[j, i] = MISSING_VALUE
                else:
                    result[j, i] = find_flexible_bin_for_value(value, bin_defs)

        return result

    def _inverse_transform_bins_to_values(
        self, X: np.ndarray[Any, Any], columns: ColumnList
    ) -> np.ndarray[Any, Any]:
        """Inverse transform from bin indices to representative values.

        Converts bin indices back to continuous values using the fitted bin
        representatives. Handles special indices for missing values appropriately.

        Args:
            X: Array of bin indices with shape (n_samples, n_features).
            columns: List of column identifiers corresponding to X columns.

        Returns:
            Array of representative values with same shape as input,
            where each bin index is replaced by its corresponding representative
            value. NaN for missing values.
        """
        result = np.zeros(X.shape, dtype=float)
        available_keys = list(self.bin_representatives_.keys())

        for i, col in enumerate(columns):
            # Find the right bin specification
            key = self._get_column_key(col, available_keys, i)
            representatives = self.bin_representatives_[key]

            col_data = X[:, i]

            # Transform each value
            for j, bin_idx in enumerate(col_data):
                if bin_idx == MISSING_VALUE:
                    result[j, i] = np.nan
                else:
                    # Clip bin index to valid range and get representative
                    clipped_idx = max(0, min(int(bin_idx), len(representatives) - 1))
                    result[j, i] = representatives[clipped_idx]

        return result

    def _get_column_key(self, target_col: Any, available_keys: ColumnList, col_index: int) -> Any:
        """Get the appropriate key for looking up bin specifications.

        Handles column key resolution with fallback strategies for
        different column identifier formats (names vs indices).

        Args:
            target_col: The target column identifier to find.
            available_keys: List of available keys in bin specifications.
            col_index: Index position of the column.

        Returns:
            The key to use for bin specification lookup.

        Raises:
            ValueError: If no matching key can be found.
        """
        # First try exact match
        if target_col in available_keys:
            return target_col

        # Try index-based fallback
        if col_index < len(available_keys):
            return available_keys[col_index]

        # No match found
        raise ValueError(f"No bin specification found for column {target_col} (index {col_index})")

    def get_bin_count(self, column: Any = None) -> Union[int, dict[Any, int]]:
        """Get the number of bins for a column or all columns.

        Args:
            column: Specific column to get bin count for. If None, returns all.

        Returns:
            Bin count(s).

        Raises:
            RuntimeError: If not fitted.
            ValueError: If column was not fitted.
        """
        # Check if fitted by verifying bin_spec_ exists and is not empty
        if not hasattr(self, "bin_spec_") or not self.bin_spec_:
            raise RuntimeError("This estimator is not fitted yet. Call 'fit' first.")

        if column is not None:
            if column not in self.bin_spec_:
                raise ValueError(f"Column {column} was not fitted")
            bin_defs = self.bin_spec_[column]
            return get_flexible_bin_count(bin_defs)

        # Return counts for all columns
        counts = {}
        for col, bin_defs in self.bin_spec_.items():
            counts[col] = get_flexible_bin_count(bin_defs)
        return counts

    def get_bin_definitions(self, column: Any = None) -> Union[list[Any], dict[Any, list[Any]]]:
        """Get the flexible bin definitions for a column or all columns.

        Args:
            column: Specific column to get definitions for. If None, returns all.

        Returns:
            Bin definition(s).

        Raises:
            RuntimeError: If not fitted.
            ValueError: If column was not fitted.
        """
        # Check if fitted by verifying bin_spec_ exists and is not empty
        if not hasattr(self, "bin_spec_") or not self.bin_spec_:
            raise RuntimeError("This estimator is not fitted yet. Call 'fit' first.")

        if column is not None:
            if column not in self.bin_spec_:
                raise ValueError(f"Column {column} was not fitted")
            return self.bin_spec_[column].copy()

        # Return definitions for all columns
        definitions = {}
        for col, bin_defs in self.bin_spec_.items():
            definitions[col] = bin_defs.copy()
        return definitions
