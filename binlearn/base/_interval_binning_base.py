"""
Clean interval binning base class for V2 architecture.

This module provides interval-based binning functionality that inherits from GeneralBinningBase.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from ..config import get_config
from ..utils import (
    BinEdgesDict,
    ColumnList,
    ConfigurationError,
    FittingError,
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
        """Validate column data for interval binning.

        Args:
            x_col: Raw column data
            col_id: Column identifier for error messages

        Returns:
            The original column data (unchanged)

        Raises:
            FittingError: If column contains only NaN values
        """
        # Check for all-NaN column
        if np.all(np.isnan(x_col)):
            raise FittingError(f"Column {col_id} contains only NaN values. Cannot perform binning.")

        return x_col

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
