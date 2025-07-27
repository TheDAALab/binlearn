"""
Interval binning base class with unified joint/per-column logic.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np

from ._general_binning_base import GeneralBinningBase
from ._bin_utils import ensure_bin_dict, validate_bins, default_representatives, create_bin_masks
from ._data_utils import return_like_input
from ._constants import MISSING_VALUE, ABOVE_RANGE, BELOW_RANGE


class IntervalBinningBase(GeneralBinningBase):
    """Base class for interval binning methods."""

    def __init__(
        self,
        clip: bool = True,
        preserve_dataframe: bool = False,
        bin_edges: Optional[Union[Dict[Any, List[float]], Any]] = None,
        bin_representatives: Optional[Union[Dict[Any, List[float]], Any]] = None,
        fit_jointly: bool = False,
        guidance_columns: Optional[Union[List[Any], Any]] = None,
        **kwargs,
    ):
        super().__init__(
            preserve_dataframe=preserve_dataframe, 
            fit_jointly=fit_jointly, 
            guidance_columns=guidance_columns,
            **kwargs
        )
        self.clip = clip

        # Store parameters as expected by sklearn
        self.bin_edges = bin_edges
        self.bin_representatives = bin_representatives

        # Store user-provided specs (for internal use)
        self._user_bin_edges = bin_edges
        self._user_bin_reps = bin_representatives

        # Fitted specifications
        self._bin_edges: Dict[Any, List[float]] = {}
        self._bin_reps: Dict[Any, List[float]] = {}

    def _fit_per_column(
        self, 
        X: np.ndarray, 
        columns: List[Any], 
        guidance_data: Optional[np.ndarray] = None,
        **fit_params
    ) -> None:
        """Fit bins per column with optional guidance data."""
        self._process_user_specifications(columns)

        if not self._user_bin_edges:
            # Calculate bins from data
            for i, col in enumerate(columns):
                if col not in self._bin_edges:
                    edges, reps = self._calculate_bins(X[:, i], col, guidance_data)
                    self._bin_edges[col] = edges
                    if col not in self._bin_reps:
                        self._bin_reps[col] = reps

        self._finalize_fitting()

    def _fit_jointly(self, X: np.ndarray, columns: List[Any], **fit_params) -> None:
        """Fit bins jointly across all columns."""
        self._process_user_specifications(columns)

        if not self._user_bin_edges:
            # Calculate joint parameters and apply to each column
            joint_params = self._calculate_joint_parameters(X, columns)

            for i, col in enumerate(columns):
                edges, reps = self._calculate_bins_jointly(X[:, i], col, joint_params)
                self._bin_edges[col] = edges
                self._bin_reps[col] = reps

        self._finalize_fitting()

    def _process_user_specifications(self, columns: List[Any]) -> None:
        """Process user-provided bin specifications."""
        if self._user_bin_edges is not None:
            self._bin_edges = ensure_bin_dict(self._user_bin_edges)
        else:
            self._bin_edges = {}

        if self._user_bin_reps is not None:
            self._bin_reps = ensure_bin_dict(self._user_bin_reps)
        else:
            self._bin_reps = {}

    def _finalize_fitting(self) -> None:
        """Finalize the fitting process."""
        # Generate default representatives for any missing ones
        for col in self._bin_edges:
            if col not in self._bin_reps:
                edges = self._bin_edges[col]
                self._bin_reps[col] = default_representatives(edges)

        # Validate the bins
        validate_bins(self._bin_edges, self._bin_reps)

    def _calculate_joint_parameters(self, X: np.ndarray, columns: List[Any]) -> Dict[str, Any]:
        """Calculate parameters shared across all columns.

        Default implementation returns empty dict. Subclasses override for specific logic.
        """
        return {}

    def _calculate_bins_jointly(
        self, x_col: np.ndarray, col_id: Any, joint_params: Dict[str, Any]
    ) -> Tuple[List[float], List[float]]:
        """Calculate bins for a column using joint parameters.

        Default implementation falls back to regular _calculate_bins.
        """
        return self._calculate_bins(x_col, col_id)

    def _calculate_bins(
        self, 
        x_col: np.ndarray, 
        col_id: Any, 
        guidance_data: Optional[np.ndarray] = None
    ) -> Tuple[List[float], List[float]]:
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
        Tuple[List[float], List[float]]
            A tuple of (bin_edges, bin_representatives).

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Must be implemented by subclasses.")

    def _get_column_key(self, target_col: Any, available_keys: List[Any], col_index: int) -> Any:
        """Find the right key for a column, handling mismatches between fit and transform."""
        # Direct match
        if target_col in available_keys:
            return target_col

        # Try index-based fallback
        if col_index < len(available_keys):
            return available_keys[col_index]

        # No match found
        raise ValueError(f"No bin specification found for column {target_col} (index {col_index})")

    def _transform_columns(self, X: np.ndarray, columns: List[Any]) -> np.ndarray:
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

    def _inverse_transform_columns(self, X: np.ndarray, columns: List[Any]) -> np.ndarray:
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

    def lookup_bin_ranges(self) -> Dict[Any, int]:
        """Return number of bins for each column."""
        self._check_fitted()
        return {col: len(edges) - 1 for col, edges in self._bin_edges.items()}

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        params = super().get_params(deep=deep)

        # Always include these parameters
        params.update(
            {
                "clip": self.clip,
                "bin_edges": self.bin_edges,
                "bin_representatives": self.bin_representatives,
                "fit_jointly": self.fit_jointly,
                "guidance_columns": self.guidance_columns,
            }
        )

        # Override with fitted values if fitted, otherwise keep constructor values
        if self._fitted:
            params["bin_edges"] = self._bin_edges
            params["bin_representatives"] = self._bin_reps

        return params

    def set_params(self, **params) -> "IntervalBinningBase":
        """Set parameters and reset fitted state if bin specs change."""
        # Handle bin specifications first (before calling super)
        reset_fitted = False

        if "bin_edges" in params:
            self.bin_edges = params["bin_edges"]
            self._user_bin_edges = params["bin_edges"]
            reset_fitted = True

        if "bin_representatives" in params:
            self.bin_representatives = params["bin_representatives"]
            self._user_bin_reps = params["bin_representatives"]
            reset_fitted = True

        if "clip" in params:
            self.clip = params["clip"]

        if "fit_jointly" in params:
            self.fit_jointly = params["fit_jointly"]
            reset_fitted = True

        if "guidance_columns" in params:
            self.guidance_columns = params["guidance_columns"]
            reset_fitted = True

        if reset_fitted:
            self._fitted = False

        super().set_params(**params)
        return self
