"""
Clean flexible binning base class for V2 architecture.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from ..utils import (
    BinEdgesDict,
    ColumnList,
    ConfigurationError,
    FlexibleBinSpec,
    transform_value_to_flexible_bin,
    validate_bin_representatives_format,
)
from ._general_binning_base import GeneralBinningBase


class FlexibleBinningBase(GeneralBinningBase):
    """Flexible binning functionality inheriting from GeneralBinningBase.

    For binning methods that use flexible, non-interval-based binning strategies.

    Provides:
    - Flexible bin specification and management
    - Custom transformation logic for non-interval methods
    - Bin mapping and lookup functionality
    """

    def __init__(
        self,
        preserve_dataframe: bool | None = None,
        fit_jointly: bool | None = None,
        guidance_columns: Any = None,
        bin_spec: FlexibleBinSpec | None = None,
        bin_representatives: BinEdgesDict | None = None,
    ):
        """Initialize flexible binning base."""
        # Initialize parent
        GeneralBinningBase.__init__(
            self,
            preserve_dataframe=preserve_dataframe,
            fit_jointly=fit_jointly,
            guidance_columns=guidance_columns,
        )

        # Store flexible-specific parameters
        self.bin_spec = bin_spec
        self.bin_representatives = bin_representatives

        # Working fitted attributes
        self.bin_spec_: FlexibleBinSpec = {}
        self.bin_representatives_: BinEdgesDict = {}

        # Configure fitted attributes for the base class
        self._fitted_attributes = ["bin_spec_", "bin_representatives_"]

        # Validate parameters early
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate flexible binning parameters."""
        # Call parent validation
        GeneralBinningBase._validate_params(self)

        # Process provided bin specifications
        try:
            if self.bin_spec is not None:
                # For now, just check it's a dictionary
                if not isinstance(self.bin_spec, dict):
                    raise ValueError("bin_spec must be a dictionary")
                self.bin_spec_ = self.bin_spec

                if self.bin_representatives is not None:
                    validate_bin_representatives_format(self.bin_representatives)
                    self.bin_representatives_ = self.bin_representatives
                elif self.bin_spec_:
                    # For flexible binning, auto-generate proper numeric representatives
                    # bin_spec contains mixed values (singletons and tuples for intervals)
                    # but representatives must be all numeric
                    self.bin_representatives_ = {}
                    for col, spec in self.bin_spec_.items():
                        if isinstance(spec, list):
                            representatives = []
                            for spec_item in spec:
                                if isinstance(spec_item, tuple) and len(spec_item) == 2:
                                    # Interval bin: use midpoint as representative
                                    representatives.append(float((spec_item[0] + spec_item[1]) / 2))
                                elif not isinstance(spec_item, tuple):
                                    # Singleton bin: use the value itself as representative
                                    try:
                                        representatives.append(float(spec_item))
                                    except (ValueError, TypeError):
                                        # For non-numeric singleton bins, use a placeholder
                                        representatives.append(0.0)
                                else:
                                    # Fallback for unexpected formats
                                    representatives.append(0.0)
                            self.bin_representatives_[col] = representatives

                # If we have complete specifications, mark as fitted and set sklearn attributes
                if self.bin_spec_ and self.bin_representatives_:
                    self._set_sklearn_attributes_from_specs()

        except ValueError as e:
            raise ConfigurationError(str(e)) from e

    def _set_sklearn_attributes_from_specs(self) -> None:
        """Set sklearn attributes from bin specifications."""
        if self.bin_spec_ is not None:
            # Get column names/indices from bin_spec
            binning_columns = list(self.bin_spec_.keys())

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
        self.bin_spec_ = {}
        self.bin_representatives_ = {}

        for i, col in enumerate(columns):
            x_col = X[:, i]

            # Validate numeric data
            self._validate_numeric_data(x_col, col)

            # Use the same guidance_data for all columns (not indexed per column)
            edges, representatives = self._calculate_flexible_bins(x_col, col, guidance_data)
            self.bin_spec_[col] = edges
            self.bin_representatives_[col] = representatives

    def _fit_jointly_across_columns(
        self, X: np.ndarray[Any, Any], columns: ColumnList, **fit_params: Any
    ) -> None:
        """Fit binning parameters jointly across all columns."""
        # For flexible binning, joint fitting is typically the same as per-column fitting
        # unless overridden by specific implementations
        self._fit_per_column_independently(X, columns, None, **fit_params)

    def _transform_columns_to_bins(
        self, X: np.ndarray[Any, Any], columns: ColumnList
    ) -> np.ndarray[Any, Any]:
        """Transform columns to bin indices using flexible mapping."""
        if X.size == 0:
            return np.empty((X.shape[0], 0))

        # Validate that input has same number of columns as bin specifications
        if X.shape[1] != len(self.bin_spec_):
            raise ValueError(
                f"Input data has {X.shape[1]} columns but bin specifications "
                f"are provided for {len(self.bin_spec_)} columns"
            )

        result = np.empty_like(X, dtype=int)
        available_keys = list(self.bin_spec_.keys())

        for i, col in enumerate(columns):
            # Find the right bin specification - this will raise ValueError for missing columns
            key = self._get_column_key(col, available_keys, i)
            bin_defs = self.bin_spec_[key]

            # Transform this column
            col_data = X[:, i]

            for row_idx, value in enumerate(col_data):
                # Use utility function for transformation
                result[row_idx, i] = transform_value_to_flexible_bin(value, bin_defs)

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

    @abstractmethod
    def _calculate_flexible_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        guidance_data: np.ndarray[Any, Any] | None = None,
    ) -> tuple[list[Any], list[Any]]:
        """Calculate flexible bin values and representatives for a column.

        For flexible binning, this typically identifies unique values or patterns
        rather than creating fixed intervals.

        Args:
            x_col: Column data to analyze
            col_id: Column identifier
            guidance_data: Optional guidance data for this column

        Returns:
            Tuple of (bin_values, representatives) where:
            - bin_values: List of values that define the bins
            - representatives: List of representative values for each bin
        """
        ...
