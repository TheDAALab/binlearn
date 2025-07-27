"""Flexible binning base class with unified joint/per-column logic."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np

from ._general_binning_base import GeneralBinningBase
from ._repr_mixin import ReprMixin
from ._bin_utils import ensure_bin_dict, validate_bins
from ._data_utils import return_like_input
from ._constants import MISSING_VALUE, ABOVE_RANGE, BELOW_RANGE
from ._flexible_bin_utils import (
    FlexibleBinSpec,
    FlexibleBinReps,
    ensure_flexible_bin_spec,
    generate_default_flexible_representatives,
    validate_flexible_bins,
    is_missing_value,
    find_flexible_bin_for_value,
    calculate_flexible_bin_width,
    transform_value_to_flexible_bin,
    get_flexible_bin_count,
)
from ..config import get_config
from ..errors import ValidationMixin, BinningError, InvalidDataError, ConfigurationError, FittingError, DataQualityWarning


class FlexibleBinningBase(GeneralBinningBase, ReprMixin):
    """Base class for flexible binning methods supporting singleton and interval bins.

    This class handles binning where bins can be:
    - Singleton bins: {"singleton": value} - exact value matches
    - Interval bins: {"interval": [min, max]} - range matches
    """

    def __init__(
        self,
        preserve_dataframe: Optional[bool] = None,
        bin_spec: Optional[Union[FlexibleBinSpec, Any]] = None,
        bin_representatives: Optional[Union[FlexibleBinReps, Any]] = None,
        fit_jointly: Optional[bool] = None,
        guidance_columns: Optional[Union[List[Any], Any]] = None,
        **kwargs,
    ):
        super().__init__(
            preserve_dataframe=preserve_dataframe, 
            fit_jointly=fit_jointly, 
            guidance_columns=guidance_columns,
            **kwargs
        )

        # Store parameters as expected by sklearn
        self.bin_spec = bin_spec
        self.bin_representatives = bin_representatives

        # Store user-provided specs (for internal use)
        self._user_bin_spec = bin_spec
        self._user_bin_reps = bin_representatives

        # Fitted specifications
        self._bin_spec: FlexibleBinSpec = {}
        self._bin_reps: FlexibleBinReps = {}
        
    def _fit_per_column(
        self, 
        X: np.ndarray, 
        columns: List[Any], 
        guidance_data: Optional[np.ndarray] = None,
        **fit_params
    ) -> None:
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
        except (ValueError, RuntimeError) as e:
            # Let these pass through unchanged for test compatibility
            raise
        except NotImplementedError:
            # Let NotImplementedError pass through unchanged
            raise
        except Exception as e:
            raise ValueError(f"Failed to fit per-column bins: {str(e)}") from e

    def _fit_jointly(self, X: np.ndarray, columns: List[Any], **fit_params) -> None:
        """Fit flexible bins jointly across all columns."""
        try:
            self._process_user_specifications(columns)

            # Calculate joint parameters and apply to columns without user-provided specs
            if any(col not in self._bin_spec for col in columns):
                joint_params = self._calculate_joint_parameters(X, columns)

                for i, col in enumerate(columns):
                    if col not in self._bin_spec:
                        bin_defs, reps = self._calculate_flexible_bins_jointly(X[:, i], col, joint_params)
                        self._bin_spec[col] = bin_defs
                        self._bin_reps[col] = reps

            self._finalize_fitting()
        except Exception as e:
            raise ValueError(f"Failed to fit joint bins: {str(e)}") from e

    def _process_user_specifications(self, columns: List[Any]) -> None:
        """Process user-provided flexible bin specifications."""
        if self._user_bin_spec is not None:
            self._bin_spec = ensure_flexible_bin_spec(self._user_bin_spec)
        # If no user specs provided, keep existing _bin_spec (don't reset to {})

        if self._user_bin_reps is not None:
            self._bin_reps = ensure_bin_dict(self._user_bin_reps)
        # If no user reps provided, keep existing _bin_reps (don't reset to {})

    def _finalize_fitting(self) -> None:
        """Finalize the fitting process."""
        # Generate default representatives for any missing ones
        for col in self._bin_spec:
            if col not in self._bin_reps:
                self._bin_reps[col] = generate_default_flexible_representatives(
                    self._bin_spec[col]
                )

        # Validate the bins
        validate_flexible_bins(self._bin_spec, self._bin_reps)

    def _calculate_joint_parameters(self, X: np.ndarray, columns: List[Any]) -> Dict[str, Any]:
        """Calculate parameters shared across all columns for flexible binning.

        Default implementation returns empty dict. Subclasses override for specific logic.
        """
        return {}

    def _calculate_flexible_bins_jointly(
        self, 
        x_col: np.ndarray, 
        col_id: Any, 
        joint_params: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Calculate flexible bins for a column using joint parameters.

        Default implementation falls back to regular _calculate_flexible_bins.
        """
        return self._calculate_flexible_bins(x_col, col_id)

    def _ensure_flexible_bin_dict(self, bin_spec: Any) -> FlexibleBinSpec:
        """Ensure bin_spec is in the correct dictionary format.
        
        DEPRECATED: Use ensure_flexible_bin_spec from _flexible_bin_utils instead.
        """
        return ensure_flexible_bin_spec(bin_spec)

    def _generate_default_flexible_representatives(
        self, bin_defs: List[Dict[str, Any]]
    ) -> List[float]:
        """Generate default representatives for flexible bins.
        
        DEPRECATED: Use generate_default_flexible_representatives from _flexible_bin_utils instead.
        """
        return generate_default_flexible_representatives(bin_defs)

    def _validate_flexible_bins(self, bin_spec: FlexibleBinSpec, bin_reps: FlexibleBinReps) -> None:
        """Validate flexible bin specifications.
        
        DEPRECATED: Use validate_flexible_bins from _flexible_bin_utils instead.
        """
        return validate_flexible_bins(bin_spec, bin_reps)

    def _is_missing_value(self, value: Any) -> bool:
        """Check if a value should be considered missing.
        
        DEPRECATED: Use is_missing_value from _flexible_bin_utils instead.
        """
        return is_missing_value(value)

    def _find_bin_for_value(self, value: float, bin_defs: List[Dict[str, Any]]) -> int:
        """Find the bin index for a given value.
        
        DEPRECATED: Use find_flexible_bin_for_value from _flexible_bin_utils instead.
        """
        return find_flexible_bin_for_value(value, bin_defs)

    def _calculate_flexible_bins(
        self, 
        x_col: np.ndarray, 
        col_id: Any, 
        guidance_data: Optional[np.ndarray] = None
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
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
        Tuple[List[Dict[str, Any]], List[float]]
            A tuple of (bin_definitions, bin_representatives).
            Bin definitions are dicts with 'singleton' or 'interval' keys.

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

    def inverse_transform(self, X: Any) -> Any:
        """Transform bin indices back to representative values."""
        self._check_fitted()
        arr, columns = self._prepare_input(X)
        result = np.full(arr.shape, np.nan, dtype=float)
        available_keys = list(self._bin_reps.keys())

        for i, col in enumerate(columns):
            key = self._get_column_key(col, available_keys, i)
            reps = self._bin_reps[key]

            col_data = arr[:, i]

            # Handle missing values
            missing_mask = col_data == MISSING_VALUE

            # Everything else gets clipped to valid range and mapped
            regular_indices = ~missing_mask
            if regular_indices.any():
                clipped_indices = np.clip(col_data[regular_indices].astype(int), 0, len(reps) - 1)
                result[regular_indices, i] = np.array(reps)[clipped_indices]

            # Handle missing values
            result[missing_mask, i] = np.nan

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

    def lookup_bin_ranges(self) -> Dict[Any, int]:
        """Return number of bins for each column."""
        self._check_fitted()
        return get_flexible_bin_count(self._bin_spec)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        params = super().get_params(deep=deep)

        # Always include these parameters
        params.update(
            {
                "preserve_dataframe": self.preserve_dataframe,
                "fit_jointly": self.fit_jointly,
                "guidance_columns": self.guidance_columns,
                "bin_spec": self.bin_spec,
                "bin_representatives": self.bin_representatives,
            }
        )

        # Override with fitted values if fitted, otherwise keep constructor values
        if self._fitted:
            params["bin_spec"] = self._bin_spec
            params["bin_representatives"] = self._bin_reps

        return params

    def set_params(self, **params) -> "FlexibleBinningBase":
        """Set parameters and reset fitted state if bin specs change."""
        # Handle bin specifications first (before calling super)
        reset_fitted = False

        if "bin_spec" in params:
            self.bin_spec = params["bin_spec"]
            self._user_bin_spec = params["bin_spec"]
            reset_fitted = True

        if "bin_representatives" in params:
            self.bin_representatives = params["bin_representatives"]
            self._user_bin_reps = params["bin_representatives"]
            reset_fitted = True

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
