"""
Interval binning base class with unified joint/per-column logic.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import abstractmethod
import warnings

import numpy as np

from ..utils.types import (
    BinEdges, BinEdgesDict, ColumnId, ColumnList, 
    OptionalColumnList, GuidanceColumns, ArrayLike
)
from ._general_binning_base import GeneralBinningBase
from ..utils.bin_operations import ensure_bin_dict, validate_bins, default_representatives, create_bin_masks
from ..utils.data_handling import return_like_input
from ..utils.constants import MISSING_VALUE, ABOVE_RANGE, BELOW_RANGE
from ..utils.errors import (
    BinningError,
    ConfigurationError,
    DataQualityWarning,
)


class IntervalBinningBase(GeneralBinningBase):
    """Base class for interval binning methods."""

    def __init__(
        self,
        clip: Optional[bool] = None,
        preserve_dataframe: Optional[bool] = None,
        bin_edges: Optional[BinEdgesDict] = None,
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

        # Load configuration defaults
        from ..config import get_config
        config = get_config()

        # Apply defaults from configuration
        if clip is None:
            clip = config.default_clip

        self.clip = clip

        # Store parameters as expected by sklearn
        self.bin_edges = bin_edges
        self.bin_representatives = bin_representatives

        # Working specifications (fitted or user-provided)
        self._bin_edges: BinEdgesDict = {}
        self._bin_reps: BinEdgesDict = {}

        # If bin_edges are provided, process them immediately to enable transform without fit
        if bin_edges is not None:
            self._process_provided_bins()

    def _process_provided_bins(self) -> None:
        """Process user-provided bin specifications and mark as fitted if complete."""
        try:
            if self.bin_edges is not None:
                self._bin_edges = ensure_bin_dict(self.bin_edges)

            if self.bin_representatives is not None:
                self._bin_reps = ensure_bin_dict(self.bin_representatives)
            else:
                # Generate default representatives for provided edges
                for col in self._bin_edges:
                    if col not in self._bin_reps:
                        edges = self._bin_edges[col]
                        self._bin_reps[col] = default_representatives(edges)

            # Validate the bins
            if self._bin_edges:
                validate_bins(self._bin_edges, self._bin_reps)
                # Mark as fitted since we have complete bin specifications
                self._fitted = True
                # Store columns for later reference
                self._original_columns = list(self._bin_edges.keys())

        except Exception as e:
            if isinstance(e, BinningError):
                raise
            raise ConfigurationError(f"Failed to process provided bin specifications: {str(e)}") from e

    @property
    def bin_edges(self):
        """Bin edges property."""
        return getattr(self, '_bin_edges_param', None)

    @bin_edges.setter
    def bin_edges(self, value):
        """Set bin edges and update internal state."""
        self._bin_edges_param = value
        if hasattr(self, '_fitted') and self._fitted:
            self._fitted = False  # Reset fitted state when bin_edges change

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
    ) -> 'IntervalBinningBase':
        """Fit bins per column with optional guidance data."""
        try:
            self._process_user_specifications(columns)

            if not self.bin_edges:
                # Calculate bins from data
                for i, col in enumerate(columns):
                    if col not in self._bin_edges:
                        # Validate column data
                        col_data = X[:, i]
                        if np.all(np.isnan(col_data)):
                            # Create a more descriptive column reference
                            if isinstance(col, (int, np.integer)):
                                col_ref = f"column {col} (index {i})"
                            else:
                                col_ref = f"column '{col}'"
                            warnings.warn(
                                f"Data in {col_ref} contains only NaN values", DataQualityWarning
                            )

                        edges, reps = self._calculate_bins(col_data, col, guidance_data)
                        self._bin_edges[col] = edges
                        if col not in self._bin_reps:
                            self._bin_reps[col] = reps

            self._finalize_fitting()
            return self

        except Exception as e:
            if isinstance(e, BinningError):
                raise
            raise ValueError(f"Failed to fit per-column bins: {str(e)}") from e

    def _fit_jointly(self, X: np.ndarray, columns: ColumnList, **fit_params) -> None:
        """Fit bins jointly across all columns."""
        try:
            self._process_user_specifications(columns)

            if not self.bin_edges:
                # For true joint binning, flatten all data together
                all_data = X.ravel()
                
                # Check if all data is NaN
                if np.all(np.isnan(all_data)):
                    warnings.warn("All data contains only NaN values", DataQualityWarning)
                
                # Calculate bins once from all flattened data
                edges, reps = self._calculate_bins_jointly(all_data, columns)
                
                # Apply the same bins to all columns
                for col in columns:
                    self._bin_edges[col] = edges
                    self._bin_reps[col] = reps

            self._finalize_fitting()

        except Exception as e:
            if isinstance(e, BinningError):
                raise
            raise ValueError(f"Failed to fit joint bins: {str(e)}") from e

    def _process_user_specifications(self, columns: ColumnList) -> None:
        """Process user-provided bin specifications."""
        try:
            if self.bin_edges is not None:
                self._bin_edges = ensure_bin_dict(self.bin_edges)
            else:
                self._bin_edges = {}

            if self.bin_representatives is not None:
                self._bin_reps = ensure_bin_dict(self.bin_representatives)
            else:
                self._bin_reps = {}

        except Exception as e:
            if isinstance(e, BinningError):
                raise
            raise ConfigurationError(f"Failed to process bin specifications: {str(e)}") from e

    def _finalize_fitting(self) -> None:
        """Finalize the fitting process."""
        # Generate default representatives for any missing ones
        for col in self._bin_edges:
            if col not in self._bin_reps:
                edges = self._bin_edges[col]
                self._bin_reps[col] = default_representatives(edges)

        # Validate the bins
        validate_bins(self._bin_edges, self._bin_reps)

    def _calculate_bins_jointly(
        self, all_data: np.ndarray, columns: ColumnList
    ) -> Tuple[BinEdges, BinEdges]:
        """Calculate bins from all flattened data for joint binning.

        Default implementation falls back to regular _calculate_bins using first column.
        """
        return self._calculate_bins(all_data, columns[0] if columns else 0)

    @abstractmethod
    def _calculate_bins(
        self, x_col: np.ndarray, col_id: Any, guidance_data: Optional[np.ndarray] = None
    ) -> Tuple[BinEdges, BinEdges]:
        """Calculate bin edges and representatives for a column.

        Parameters
        ----------
        x_col : np.ndarray
            The data for the column being binned.
        col_id : Any
            The identifier for the column.
        guidance_data : Optional[np.ndarray], default=None
            Optional guidance data that can influence bin calculation.

        Returns
        -------
        Tuple[BinEdges, BinEdges]
            A tuple of (bin_edges, bin_representatives).

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
        """Transform columns to bin indices."""
        result = np.zeros(X.shape, dtype=int)
        available_keys = list(self._bin_edges.keys())

        for i, col in enumerate(columns):
            # Find the right bin specification
            key = self._get_column_key(col, available_keys, i)
            edges = self._bin_edges[key]

            # Transform this column
            col_data = X[:, i]
            bin_indices = np.digitize(col_data, edges) - 1

            # Handle special cases
            nan_mask = np.isnan(col_data)
            below_mask = (col_data < edges[0]) & ~nan_mask
            above_mask = (col_data >= edges[-1]) & ~nan_mask

            if self.clip:
                bin_indices = np.clip(bin_indices, 0, len(edges) - 2)
            else:
                bin_indices[below_mask] = BELOW_RANGE
                bin_indices[above_mask] = ABOVE_RANGE

            bin_indices[nan_mask] = MISSING_VALUE
            result[:, i] = bin_indices

        return result

    def _inverse_transform_columns(self, X: np.ndarray, columns: ColumnList) -> np.ndarray:
        """Inverse transform from bin indices to representative values."""
        result = np.zeros(X.shape, dtype=float)
        available_keys = list(self._bin_reps.keys())

        for i, col in enumerate(columns):
            key = self._get_column_key(col, available_keys, i)
            reps = self._bin_reps[key]

            col_data = X[:, i]

            # Handle special values first
            nan_mask = col_data == MISSING_VALUE
            below_mask = col_data == BELOW_RANGE
            above_mask = col_data == ABOVE_RANGE

            # Everything else gets clipped to valid range and mapped
            regular_indices = ~nan_mask & ~below_mask & ~above_mask
            if regular_indices.any():
                clipped_indices = np.clip(col_data[regular_indices].astype(int), 0, len(reps) - 1)
                result[regular_indices, i] = np.array(reps)[clipped_indices]

            # Handle special values
            result[nan_mask, i] = np.nan
            result[below_mask, i] = -np.inf
            result[above_mask, i] = np.inf

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
        result = np.zeros(arr.shape, dtype=float)
        available_keys = list(self._bin_edges.keys())

        for i, col in enumerate(columns):
            key = self._get_column_key(col, available_keys, i)
            edges = self._bin_edges[key]

            col_data = arr[:, i]
            valid, _, _, _ = create_bin_masks(col_data, len(edges) - 1)

            if valid.any():
                valid_indices = np.clip(col_data[valid].astype(int), 0, len(edges) - 2)
                widths = np.array([edges[j + 1] - edges[j] for j in range(len(edges) - 1)])
                result[valid, i] = widths[valid_indices]

        return return_like_input(result, bin_indices, columns, self.preserve_dataframe)

    def lookup_bin_ranges(self) -> Dict[ColumnId, int]:
        """Return number of bins for each column."""
        self._check_fitted()
        return {col: len(edges) - 1 for col, edges in self._bin_edges.items()}

    def _get_fitted_params(self) -> Dict[str, Any]:
        """Get fitted parameter values for IntervalBinningBase."""
        return {
            "bin_edges": self._bin_edges,
            "bin_representatives": self._bin_reps,
        }
