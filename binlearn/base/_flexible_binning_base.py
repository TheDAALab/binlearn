"""
Clean flexible binning base class for V2 architecture.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from ..config import get_config
from ..utils._constants import MISSING_VALUE
from ..utils._errors import ConfigurationError
from ..utils._types import ArrayLike, FlexibleBinSpec, FlexibleBinDefs, BinEdgesDict, ColumnList
from ..utils import (
    validate_bin_representatives_format,
    validate_flexible_bin_spec_format,
    transform_value_to_flexible_bin,
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
                    # For flexible binning, representatives are typically the unique values themselves
                    self.bin_representatives_ = {}
                    for col, spec in self.bin_spec_.items():
                        # For flexible binning, spec might be the actual values
                        # Representatives are usually the same as the bin spec values
                        if isinstance(spec, list):
                            self.bin_representatives_[col] = spec.copy()

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

        result = np.full(X.shape, MISSING_VALUE, dtype=int)
        available_keys = list(self.bin_spec_.keys())

        for i, col in enumerate(columns):
            # Find the right bin specification
            key = self._get_column_key(col, available_keys, i)
            bin_defs = self.bin_spec_[key]

            # Transform this column
            col_data = X[:, i]

            for row_idx, value in enumerate(col_data):
                # Use utility function for transformation
                result[row_idx, i] = transform_value_to_flexible_bin(value, bin_defs)

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
