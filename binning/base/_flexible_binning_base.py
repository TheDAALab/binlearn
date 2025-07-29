"""Flexible binning base class with unified joint/per-column logic."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import abstractmethod

import numpy as np

from ..utils.types import (
    BinEdges, BinEdgesDict, FlexibleBinSpec, FlexibleBinDefs,
    ColumnId, ColumnList, OptionalColumnList, GuidanceColumns, ArrayLike
)
from ._general_binning_base import GeneralBinningBase
from ..utils.bin_operations import ensure_bin_dict
from ..utils.flexible_binning import (
    ensure_flexible_bin_spec,
    generate_default_flexible_representatives,
    validate_flexible_bins,
    is_missing_value,
    find_flexible_bin_for_value,
    calculate_flexible_bin_width,
    transform_value_to_flexible_bin,
    get_flexible_bin_count,
)
from ..utils.data_handling import return_like_input
from ..utils.constants import MISSING_VALUE


class FlexibleBinningBase(GeneralBinningBase):
    """Base class for flexible binning methods supporting singleton and interval bins.

    This class handles binning where bins can be:
    - Singleton bins: {"singleton": value} - exact value matches
    - Interval bins: {"interval": [min, max]} - range matches
    """

    def __init__(
        self,
        preserve_dataframe: Optional[bool] = None,
        bin_spec: Optional[FlexibleBinSpec] = None,
        bin_representatives: Optional[BinEdgesDict] = None,
        fit_jointly: Optional[bool] = None,
        guidance_columns: Optional[GuidanceColumns] = None,
        **kwargs,
    ):
        super().__init__(
            preserve_dataframe=preserve_dataframe,
            fit_jointly=fit_jointly,
            guidance_columns=guidance_columns,
            **kwargs,
        )

        # Store parameters as expected by sklearn
        self.bin_spec = bin_spec
        self.bin_representatives = bin_representatives

        # Working specifications (fitted or user-provided)
        self._bin_spec: FlexibleBinSpec = {}
        self._bin_reps: BinEdgesDict = {}

        # If bin_spec is provided, process it immediately to enable transform without fit
        if bin_spec is not None:
            self._process_provided_flexible_bins()

    def _process_provided_flexible_bins(self) -> None:
        """Process user-provided flexible bin specifications and mark as fitted if complete."""
        try:
            if self.bin_spec is not None:
                self._bin_spec = ensure_flexible_bin_spec(self.bin_spec)

            if self.bin_representatives is not None:
                self._bin_reps = ensure_bin_dict(self.bin_representatives)
            else:
                # Generate default representatives for provided specs
                for col in self._bin_spec:
                    if col not in self._bin_reps:
                        bin_defs = self._bin_spec[col]
                        self._bin_reps[col] = generate_default_flexible_representatives(bin_defs)

            # Validate the bins
            if self._bin_spec:
                validate_flexible_bins(self._bin_spec, self._bin_reps)
                # Mark as fitted since we have complete bin specifications
                self._fitted = True
                # Store columns for later reference
                self._original_columns = list(self._bin_spec.keys())

        except Exception as e:
            raise ValueError(f"Failed to process provided flexible bin specifications: {str(e)}") from e

    @property
    def bin_spec(self):
        """Bin specification property."""
        return getattr(self, '_bin_spec_param', None)

    @bin_spec.setter
    def bin_spec(self, value):
        """Set bin specification and update internal state."""
        self._bin_spec_param = value
        if hasattr(self, '_fitted') and self._fitted:
            self._fitted = False  # Reset fitted state when bin_spec changes

    @property
    def bin_representatives(self):
        """Bin representatives property."""
        return getattr(self, '_bin_representatives_param', None)

    @bin_representatives.setter
    def bin_representatives(self, value):
        """Set bin representatives and update internal state."""
        self._bin_representatives_param = value
        if hasattr(self, '_fitted') and self._fitted:
            self._fitted = False  # Reset fitted state when bin_representatives change

    def _fit_per_column(
        self,
        X: np.ndarray,
        columns: ColumnList,
        guidance_data: Optional[np.ndarray] = None,
        **fit_params,
    ) -> 'FlexibleBinningBase':
        """Fit flexible bins per column with optional guidance data."""
        try:
            self._process_user_specifications(columns)

            # Calculate bins from data for columns that don't have user-provided specs
            for i, col in enumerate(columns):
                # Skip column entirely if it already has bin specs or representatives
                if col not in self._bin_spec and col not in self._bin_reps:
                    bin_defs, reps = self._calculate_flexible_bins(X[:, i], col, guidance_data)
                    self._bin_spec[col] = bin_defs
                    self._bin_reps[col] = reps

            self._finalize_fitting()
            return self
        except (ValueError, RuntimeError, NotImplementedError):
            # Let these pass through unchanged for test compatibility
            raise
        except Exception as e:
            raise ValueError(f"Failed to fit per-column bins: {str(e)}") from e

    def _fit_jointly(self, X: np.ndarray, columns: ColumnList, **fit_params) -> None:
        """Fit flexible bins jointly across all columns."""
        try:
            self._process_user_specifications(columns)

            # Calculate bins from all flattened data if needed
            if any(col not in self._bin_spec for col in columns):
                # For true joint binning, flatten all data together
                all_data = X.ravel()
                
                # Calculate bins once from all flattened data
                bin_defs, reps = self._calculate_flexible_bins_jointly(all_data, columns)
                
                # Apply the same bins to all columns that don't have user-provided specs
                for col in columns:
                    if col not in self._bin_spec:
                        self._bin_spec[col] = bin_defs
                        self._bin_reps[col] = reps

            self._finalize_fitting()
        except Exception as e:
            raise ValueError(f"Failed to fit joint bins: {str(e)}") from e

    def _process_user_specifications(self, columns: ColumnList) -> None:
        """Process user-provided flexible bin specifications."""
        if self.bin_spec is not None:
            self._bin_spec = ensure_flexible_bin_spec(self.bin_spec)
        else:
            # Reset _bin_spec if no user specs provided to allow refitting
            self._bin_spec = {}

        if self.bin_representatives is not None:
            self._bin_reps = ensure_bin_dict(self.bin_representatives)
        else:
            # Reset _bin_reps if no user reps provided to allow refitting
            self._bin_reps = {}

    def _finalize_fitting(self) -> None:
        """Finalize the fitting process."""
        # Generate default representatives for any missing ones
        for col in self._bin_spec:
            if col not in self._bin_reps:
                self._bin_reps[col] = generate_default_flexible_representatives(self._bin_spec[col])

        # Validate the bins
        validate_flexible_bins(self._bin_spec, self._bin_reps)

    def _calculate_flexible_bins_jointly(
        self, all_data: np.ndarray, columns: ColumnList
    ) -> Tuple[FlexibleBinDefs, BinEdges]:
        """Calculate flexible bins from all flattened data for joint binning.

        Default implementation falls back to regular _calculate_flexible_bins using first column.
        """
        return self._calculate_flexible_bins(all_data, columns[0] if columns else 0)

    def _ensure_flexible_bin_dict(self, bin_spec: Any) -> FlexibleBinSpec:
        """Ensure bin_spec is in the correct dictionary format.

        DEPRECATED: Use ensure_flexible_bin_spec from _flexible_bin_utils instead.
        """
        return ensure_flexible_bin_spec(bin_spec)

    def _generate_default_flexible_representatives(
        self, bin_defs: FlexibleBinDefs
    ) -> BinEdges:
        """Generate default representatives for flexible bins.

        DEPRECATED: Use generate_default_flexible_representatives from _flexible_bin_utils instead.
        """
        return generate_default_flexible_representatives(bin_defs)

    def _validate_flexible_bins(self, bin_spec: FlexibleBinSpec, bin_reps: BinEdgesDict) -> None:
        """Validate flexible bin specifications.

        DEPRECATED: Use validate_flexible_bins from _flexible_bin_utils instead.
        """
        return validate_flexible_bins(bin_spec, bin_reps)

    def _is_missing_value(self, value: Any) -> bool:
        """Check if a value should be considered missing.

        DEPRECATED: Use is_missing_value from _flexible_bin_utils instead.
        """
        return is_missing_value(value)

    def _find_bin_for_value(self, value: float, bin_defs: FlexibleBinDefs) -> int:
        """Find the bin index for a given value.

        DEPRECATED: Use find_flexible_bin_for_value from _flexible_bin_utils instead.
        """
        return find_flexible_bin_for_value(value, bin_defs)

    @abstractmethod
    def _calculate_flexible_bins(
        self, x_col: np.ndarray, col_id: Any, guidance_data: Optional[np.ndarray] = None
    ) -> Tuple[FlexibleBinDefs, BinEdges]:
        """Calculate flexible bin definitions and representatives for a column.

        Parameters
        ----------
        x_col : np.ndarray
            Data for a single column.
        col_id : Any
            Column identifier.
        guidance_data : Optional[np.ndarray], default=None
            Optional guidance data that can influence bin calculation.

        Returns
        -------
        Tuple[FlexibleBinDefs, BinEdges]
            A tuple of (bin_definitions, bin_representatives).
            Bin definitions are dicts with 'singleton' or 'interval' keys.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Must be implemented by subclasses.")

    def _get_column_key(self, target_col: ColumnId, available_keys: ColumnList, col_index: int) -> ColumnId:
        """Find the right key for a column, handling mismatches between fit and transform."""
        # Direct match
        if target_col in available_keys:
            return target_col

        # Try index-based fallback
        if col_index < len(available_keys):
            return available_keys[col_index]

        # No match found
        raise ValueError(f"No bin specification found for column {target_col} (index {col_index})")

    def _transform_columns(self, X: np.ndarray, columns: ColumnList) -> np.ndarray:
        """Transform columns to bin indices using flexible bins."""
        result = np.full(X.shape, MISSING_VALUE, dtype=int)
        available_keys = list(self._bin_spec.keys())

        for i, col in enumerate(columns):
            # Find the right bin specification
            key = self._get_column_key(col, available_keys, i)
            bin_defs = self._bin_spec[key]

            # Transform this column
            col_data = X[:, i]

            for row_idx, value in enumerate(col_data):
                # Use utility function for transformation
                result[row_idx, i] = transform_value_to_flexible_bin(value, bin_defs)

        return result

    def _inverse_transform_columns(self, X: np.ndarray, columns: ColumnList) -> np.ndarray:
        """Transform bin indices back to representative values for flexible bins."""
        result = np.full(X.shape, np.nan, dtype=float)
        available_keys = list(self._bin_reps.keys())

        for i, col in enumerate(columns):
            key = self._get_column_key(col, available_keys, i)
            reps = self._bin_reps[key]

            col_data = X[:, i]

            # Handle missing values
            missing_mask = col_data == MISSING_VALUE

            # Everything else gets clipped to valid range and mapped
            regular_indices = ~missing_mask
            if regular_indices.any():
                clipped_indices = np.clip(col_data[regular_indices].astype(int), 0, len(reps) - 1)
                result[regular_indices, i] = np.array(reps)[clipped_indices]

            # Handle missing values
            result[missing_mask, i] = np.nan

        return result

    def inverse_transform(self, X: Any) -> Any:
        """Transform bin indices back to representative values."""
        self._check_fitted()
        arr, columns = self._prepare_input(X)
        result = self._inverse_transform_columns(arr, columns)
        return return_like_input(result, X, columns, self.preserve_dataframe)

    def lookup_bin_widths(self, bin_indices: Any) -> Any:
        """Look up bin widths for given bin indices."""
        self._check_fitted()
        arr, columns = self._prepare_input(bin_indices)
        result = np.full(arr.shape, np.nan, dtype=float)
        available_keys = list(self._bin_spec.keys())

        for i, col in enumerate(columns):
            key = self._get_column_key(col, available_keys, i)
            bin_defs = self._bin_spec[key]

            col_data = arr[:, i]

            for row_idx, bin_idx in enumerate(col_data):
                # Only handle missing values specially
                if bin_idx == MISSING_VALUE:
                    continue

                bin_idx_int = int(bin_idx)
                if 0 <= bin_idx_int < len(bin_defs):
                    bin_def = bin_defs[bin_idx_int]
                    result[row_idx, i] = calculate_flexible_bin_width(bin_def)

        return return_like_input(result, bin_indices, columns, self.preserve_dataframe)

    def lookup_bin_ranges(self) -> Dict[ColumnId, int]:
        """Return number of bins for each column."""
        self._check_fitted()
        return get_flexible_bin_count(self._bin_spec)

    def _get_fitted_params(self) -> Dict[str, Any]:
        """Get fitted parameter values for FlexibleBinningBase."""
        return {
            "bin_spec": self._bin_spec,
            "bin_representatives": self._bin_reps,
        }

    # Properties for sklearn compatibility
    @property
    def bin_spec_(self) -> FlexibleBinSpec:
        """Fitted bin specifications (sklearn style)."""
        return self._bin_spec

    @property
    def bin_representatives_(self) -> BinEdgesDict:
        """Fitted bin representatives (sklearn style)."""
        return self._bin_reps
