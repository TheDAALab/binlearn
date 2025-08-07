"""
Clean interval binning base class for V2 architecture.

This module provides interval-based binning functionality that inherits from GeneralBinningBase.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from ..config import get_config
from ..utils.errors import ConfigurationError, FittingError
from ..utils.types import ArrayLike, BinEdgesDict, ColumnList
from ..utils import (
    default_representatives,
    validate_bin_edges_format,
    validate_bin_representatives_format,
    validate_bins,
)
from ._general_binning_base import GeneralBinningBase


class IntervalBinningBase(GeneralBinningBase):
    """Interval-based binning functionality inheriting from GeneralBinningBase.

    Provides:
    - Interval-based transformation logic
    - Bin edge and representative management
    - Clipping and special value handling
    """

    def __init__(
        self,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        fit_jointly: bool | None = None,
        guidance_columns: Any = None,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
    ):
        """Initialize interval binning base."""
        # Initialize parent
        GeneralBinningBase.__init__(
            self,
            preserve_dataframe=preserve_dataframe,
            fit_jointly=fit_jointly,
            guidance_columns=guidance_columns,
        )

        # Load configuration defaults
        config = get_config()
        if clip is None:
            clip = config.default_clip

        # Store interval-specific parameters
        self.clip = clip
        self.bin_edges = bin_edges
        self.bin_representatives = bin_representatives

        # Working fitted attributes
        self.bin_edges_: BinEdgesDict = {}
        self.bin_representatives_: BinEdgesDict = {}

        # Configure fitted attributes for the base class
        self._fitted_attributes = ["bin_edges_", "bin_representatives_"]

        # Validate parameters early
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate interval binning parameters."""
        # Call parent validation
        GeneralBinningBase._validate_params(self)

        # Validate clip parameter
        if not isinstance(self.clip, bool):
            raise TypeError("clip must be a boolean")

        # Process provided bin specifications
        try:
            if self.bin_edges is not None:
                validate_bin_edges_format(self.bin_edges)
                self.bin_edges_ = self.bin_edges

                if self.bin_representatives is not None:
                    validate_bin_representatives_format(self.bin_representatives, self.bin_edges)
                    self.bin_representatives_ = self.bin_representatives

                    # Validate compatibility
                    validate_bins(self.bin_edges_, self.bin_representatives_)
                elif self.bin_edges_:
                    # Generate default representatives
                    self.bin_representatives_ = {}
                    for col, edges in self.bin_edges_.items():
                        edges_list = list(edges)
                        self.bin_representatives_[col] = default_representatives(edges_list)

                # If we have complete specifications, mark as fitted and set sklearn attributes
                if self.bin_edges_ and self.bin_representatives_:
                    self._set_sklearn_attributes_from_specs()

        except ValueError as e:
            raise ConfigurationError(str(e)) from e

    def _set_sklearn_attributes_from_specs(self) -> None:
        """Set sklearn attributes from bin specifications."""
        if self.bin_edges_ is not None:
            # Get column names/indices from bin_edges
            binning_columns = list(self.bin_edges_.keys())

            # Add guidance columns if specified
            all_features = binning_columns.copy()
            if self.guidance_columns is not None:
                guidance_cols = (
                    [self.guidance_columns]
                    if not isinstance(self.guidance_columns, list)
                    else self.guidance_columns
                )
                # Add guidance columns that aren't already in binning columns
                for col in guidance_cols:
                    if col not in all_features:
                        all_features.append(col)

            # Set sklearn attributes
            self._feature_names_in = all_features
            self._n_features_in = len(all_features)

    def _fit_per_column_independently(
        self,
        X: np.ndarray[Any, Any],
        columns: ColumnList,
        guidance_data: np.ndarray[Any, Any] | None = None,
        **fit_params: Any,
    ) -> None:
        """Fit binning parameters independently for each column."""
        self.bin_edges_ = {}
        self.bin_representatives_ = {}

        for i, col in enumerate(columns):
            x_col = X[:, i]

            # Validate and preprocess numeric data
            x_col_processed = self._validate_and_preprocess_column(x_col, col)

            # Use the same guidance_data for all columns (not indexed per column)
            edges, representatives = self._calculate_bins(x_col_processed, col, guidance_data)
            self.bin_edges_[col] = edges
            self.bin_representatives_[col] = representatives

    def _fit_jointly_across_columns(
        self, X: np.ndarray[Any, Any], columns: ColumnList, **fit_params: Any
    ) -> None:
        """Fit binning parameters jointly across all columns."""
        # For interval binning, joint fitting is the same as per-column fitting
        # since intervals don't depend on other columns
        self._fit_per_column_independently(X, columns, None, **fit_params)

    def _transform_columns_to_bins(
        self, X: np.ndarray[Any, Any], columns: ColumnList
    ) -> np.ndarray[Any, Any]:
        """Transform columns to bin indices."""
        if X.size == 0:
            return np.empty((X.shape[0], 0))

        result = np.empty_like(X, dtype=int)
        available_keys = list(self.bin_edges_.keys())

        for i, col in enumerate(columns):
            # Get the right bin specification using column key resolution
            key = self._get_column_key(col, available_keys, i)
            edges = np.array(self.bin_edges_[key])
            column_data = X[:, i]

            # Handle special values (NaN, inf)
            is_special = np.isnan(column_data) | np.isinf(column_data)

            # Apply clipping if enabled
            if self.clip:
                column_data = np.clip(column_data, edges[0], edges[-1])

            # Digitize to get bin indices
            bin_indices = np.digitize(column_data, edges) - 1

            # Ensure bin indices are in valid range
            bin_indices = np.clip(bin_indices, 0, len(edges) - 2)

            # Handle special values - assign to last bin
            bin_indices[is_special] = len(edges) - 2

            result[:, i] = bin_indices

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

        # Handle feature_N -> N mapping for numpy array inputs
        if isinstance(target_col, str) and target_col.startswith("feature_"):
            try:
                feature_index = int(target_col.split("_")[1])
                if feature_index in available_keys:
                    return feature_index
            except (ValueError, IndexError):
                pass

        # Handle N -> feature_N mapping
        if isinstance(target_col, int):
            feature_name = f"feature_{target_col}"
            if feature_name in available_keys:
                return feature_name

        # Try index-based fallback
        if col_index < len(available_keys):
            return available_keys[col_index]

        # No match found
        raise ValueError(f"No bin specification found for column {target_col} (index {col_index})")

    def _inverse_transform_bins_to_values(
        self, X: np.ndarray[Any, Any], columns: ColumnList
    ) -> np.ndarray[Any, Any]:
        """Transform bin indices to representative values."""
        if X.size == 0:
            return np.empty((X.shape[0], 0))

        result = np.empty_like(X, dtype=float)
        available_keys = list(self.bin_representatives_.keys())

        for i, col in enumerate(columns):
            # Get the right bin specification using column key resolution
            key = self._get_column_key(col, available_keys, i)
            representatives = np.array(self.bin_representatives_[key])
            bin_indices = X[:, i].astype(int)

            # Clip indices to valid range
            bin_indices = np.clip(bin_indices, 0, len(representatives) - 1)

            result[:, i] = representatives[bin_indices]

        return result

    def _validate_and_preprocess_column(
        self, x_col: np.ndarray[Any, Any], col_id: Any
    ) -> np.ndarray[Any, Any]:
        """Validate and preprocess column data for interval binning.

        Handles common data quality issues:
        - Throws error for all-NaN columns
        - Converts inf/-inf to finite boundary values
        - Handles constant columns by adding small epsilon
        - Returns preprocessed column ready for binning

        Args:
            x_col: Raw column data
            col_id: Column identifier for error messages

        Returns:
            Preprocessed column data ready for binning

        Raises:
            FittingError: If column contains only NaN values
        """
        # Check for all-NaN column
        if np.all(np.isnan(x_col)):
            raise FittingError(f"Column {col_id} contains only NaN values. Cannot perform binning.")

        # Work with a copy to avoid modifying input data
        x_processed = x_col.copy()

        # Handle inf/-inf values by replacing with finite boundary values
        if np.any(np.isinf(x_processed)):
            # Find finite values to determine reasonable replacement values
            finite_mask = np.isfinite(x_processed)

            if np.any(finite_mask):
                # Use finite values to determine range
                finite_values = x_processed[finite_mask]
                finite_min = np.min(finite_values)
                finite_max = np.max(finite_values)

                # Calculate a reasonable extension beyond the finite range
                if finite_min == finite_max:
                    # All finite values are the same
                    range_extension = max(abs(finite_min) * 0.1, 1.0)
                else:
                    range_extension = (finite_max - finite_min) * 0.1

                # Replace inf/-inf with extended boundary values
                x_processed[x_processed == np.inf] = finite_max + range_extension
                x_processed[x_processed == -np.inf] = finite_min - range_extension
            else:
                # Only inf/-inf values exist, use default range
                x_processed[x_processed == np.inf] = 1.0
                x_processed[x_processed == -np.inf] = -1.0

        # Handle constant columns (after inf handling)
        finite_mask = np.isfinite(x_processed)
        if np.any(finite_mask):
            finite_values = x_processed[finite_mask]
            if len(np.unique(finite_values)) == 1:
                # Column is constant - add small epsilon
                constant_value = finite_values[0]
                epsilon = max(abs(constant_value) * 1e-8, 1e-8)

                # Create small variation around the constant
                # Half the values get -epsilon, half get +epsilon
                n_values = len(finite_values)
                variations = np.full(n_values, epsilon)
                variations[: n_values // 2] = -epsilon

                # Apply variations only to finite values
                x_processed[finite_mask] = constant_value + variations

        return x_processed

    @abstractmethod
    def _calculate_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        guidance_data: np.ndarray[Any, Any] | None = None,
    ) -> tuple[list[float], list[float]]:
        """Calculate bin edges and representatives for a column.

        Subclasses must implement this method to define their binning strategy.
        """
        ...
