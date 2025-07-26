"""Flexible binning base class with unified joint/per-column logic."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import abstractmethod

import numpy as np

from ._general_binning_base import GeneralBinningBase
from ._bin_utils import ensure_bin_dict, validate_bins
from ._data_utils import return_like_input
from ._constants import MISSING_VALUE, ABOVE_RANGE, BELOW_RANGE

# Type aliases for flexible binning
FlexibleBinSpec = Dict[Any, List[Dict[str, Any]]]  # col_id -> list of bin definitions
FlexibleBinReps = Dict[Any, List[float]]  # col_id -> list of representatives


class FlexibleBinningBase(GeneralBinningBase):
    """Base class for flexible binning methods supporting singleton and interval bins.

    This class handles binning where bins can be:
    - Singleton bins: {"singleton": value} - exact value matches
    - Interval bins: {"interval": [min, max]} - range matches
    """

    def __init__(
        self,
        preserve_dataframe: bool = False,
        bin_spec: Optional[Union[FlexibleBinSpec, Any]] = None,
        bin_representatives: Optional[Union[FlexibleBinReps, Any]] = None,
        fit_jointly: bool = False,
        **kwargs,
    ):
        super().__init__(preserve_dataframe=preserve_dataframe, fit_jointly=fit_jointly, **kwargs)

        # Store parameters as expected by sklearn
        self.bin_spec = bin_spec
        self.bin_representatives = bin_representatives

        # Store user-provided specs (for internal use)
        self._user_bin_spec = bin_spec
        self._user_bin_reps = bin_representatives

        # Fitted specifications
        self._bin_spec: FlexibleBinSpec = {}
        self._bin_reps: FlexibleBinReps = {}

    def _fit_per_column(self, X: np.ndarray, columns: List[Any]) -> None:
        """Fit flexible bins per column (original logic)."""
        self._process_user_specifications(columns)

        if not self._user_bin_spec:
            # Calculate bins from data
            for i, col in enumerate(columns):
                if col not in self._bin_spec:
                    bin_defs, reps = self._calculate_flexible_bins(X[:, i], col)
                    self._bin_spec[col] = bin_defs
                    if col not in self._bin_reps:
                        self._bin_reps[col] = reps

        self._finalize_fitting()

    def _fit_jointly(self, X: np.ndarray, columns: List[Any]) -> None:
        """Fit flexible bins jointly across all columns."""
        self._process_user_specifications(columns)

        if not self._user_bin_spec:
            # Calculate joint parameters and apply to each column
            joint_params = self._calculate_joint_parameters(X, columns)

            for i, col in enumerate(columns):
                bin_defs, reps = self._calculate_flexible_bins_jointly(X[:, i], col, joint_params)
                self._bin_spec[col] = bin_defs
                self._bin_reps[col] = reps

        self._finalize_fitting()

    def _process_user_specifications(self, columns: List[Any]) -> None:
        """Process user-provided flexible bin specifications."""
        if self._user_bin_spec is not None:
            self._bin_spec = self._ensure_flexible_bin_dict(self._user_bin_spec)
        else:
            self._bin_spec = {}

        if self._user_bin_reps is not None:
            self._bin_reps = ensure_bin_dict(self._user_bin_reps)
        else:
            self._bin_reps = {}

    def _finalize_fitting(self) -> None:
        """Finalize the fitting process."""
        # Generate default representatives for any missing ones
        for col in self._bin_spec:
            if col not in self._bin_reps:
                self._bin_reps[col] = self._generate_default_flexible_representatives(
                    self._bin_spec[col]
                )

        # Validate the bins
        self._validate_flexible_bins(self._bin_spec, self._bin_reps)

    def _ensure_flexible_bin_dict(self, bin_spec: Any) -> FlexibleBinSpec:
        """Ensure bin_spec is in the correct dictionary format."""
        if isinstance(bin_spec, dict):
            return bin_spec
        else:
            # Handle other formats if needed
            raise ValueError("bin_spec must be a dictionary mapping columns to bin definitions")

    def _generate_default_flexible_representatives(
        self, bin_defs: List[Dict[str, Any]]
    ) -> List[float]:
        """Generate default representatives for flexible bins."""
        reps = []
        for bin_def in bin_defs:
            if "singleton" in bin_def:
                reps.append(float(bin_def["singleton"]))
            elif "interval" in bin_def:
                a, b = bin_def["interval"]
                reps.append((a + b) / 2)  # Midpoint
            else:
                raise ValueError(f"Unknown bin definition: {bin_def}")
        return reps

    def _validate_flexible_bins(self, bin_spec: FlexibleBinSpec, bin_reps: FlexibleBinReps) -> None:
        """Validate flexible bin specifications."""
        for col in bin_spec:
            bin_defs = bin_spec[col]
            reps = bin_reps.get(col, [])

            if len(bin_defs) != len(reps):
                raise ValueError(
                    f"Column {col}: Number of bin definitions ({len(bin_defs)}) "
                    f"must match number of representatives ({len(reps)})"
                )

            # Validate bin definition format
            for i, bin_def in enumerate(bin_defs):
                if not isinstance(bin_def, dict):
                    raise ValueError(f"Column {col}, bin {i}: Bin definition must be a dictionary")

                if "singleton" in bin_def:
                    if len(bin_def) != 1:
                        raise ValueError(
                            f"Column {col}, bin {i}: Singleton bin must have only 'singleton' key"
                        )
                elif "interval" in bin_def:
                    if len(bin_def) != 1:
                        raise ValueError(
                            f"Column {col}, bin {i}: Interval bin must have only 'interval' key"
                        )

                    interval = bin_def["interval"]
                    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                        raise ValueError(f"Column {col}, bin {i}: Interval must be [min, max]")

                    if interval[0] > interval[1]:
                        raise ValueError(f"Column {col}, bin {i}: Interval min must be <= max")
                else:
                    raise ValueError(
                        f"Column {col}, bin {i}: Bin must have 'singleton' or 'interval' key"
                    )

    def _calculate_joint_parameters(self, X: np.ndarray, columns: List[Any]) -> Dict[str, Any]:
        """Calculate parameters shared across all columns for flexible binning.

        Default implementation returns empty dict. Subclasses override for specific logic.
        """
        return {}

    def _calculate_flexible_bins_jointly(
        self, x_col: np.ndarray, col_id: Any, joint_params: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Calculate flexible bins for a column using joint parameters.

        Default implementation falls back to regular _calculate_flexible_bins.
        """
        return self._calculate_flexible_bins(x_col, col_id)

    @abstractmethod
    def _calculate_flexible_bins(
        self, x_col: np.ndarray, col_id: Any
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Calculate flexible bin definitions and representatives for a column.

        Args:
            x_col: Data for a single column.
            col_id: Column identifier.

        Returns:
            Tuple containing:
            - List of bin definitions (dicts with 'singleton' or 'interval' keys)
            - List of representative values

        Must be implemented by subclasses.
        """
        pass

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
                # Robust missing value check
                if self._is_missing_value(value):
                    result[row_idx, i] = MISSING_VALUE
                    continue

                # Find matching bin
                bin_idx = self._find_bin_for_value(value, bin_defs)
                result[row_idx, i] = bin_idx

        return result

    def _is_missing_value(self, value: Any) -> bool:
        """Check if a value should be considered missing."""
        if value is None:
            return True

        # For numeric types, check for NaN
        try:
            if np.isnan(value):
                return True
        except (TypeError, ValueError):
            # Not a numeric type or can't convert to check NaN
            pass

        # For string types, check common missing representations
        if isinstance(value, str) and value.lower() in ["nan", "none", "", "null"]:
            return True

        return False

    def _find_bin_for_value(self, value: float, bin_defs: List[Dict[str, Any]]) -> int:
        """Find the bin index for a given value."""
        for bin_idx, bin_def in enumerate(bin_defs):
            if "singleton" in bin_def:
                if value == bin_def["singleton"]:
                    return bin_idx
            elif "interval" in bin_def:
                a, b = bin_def["interval"]
                if a <= value <= b:
                    return bin_idx

        # Value doesn't match any bin - treat as missing
        return MISSING_VALUE

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

                    if "singleton" in bin_def:
                        result[row_idx, i] = 0.0  # Singleton has zero width
                    elif "interval" in bin_def:
                        a, b = bin_def["interval"]
                        result[row_idx, i] = b - a

        return return_like_input(result, bin_indices, columns, self.preserve_dataframe)

    def lookup_bin_ranges(self) -> Dict[Any, int]:
        """Return number of bins for each column."""
        self._check_fitted()
        return {col: len(bin_defs) for col, bin_defs in self._bin_spec.items()}

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        params = super().get_params(deep=deep)

        # Ensure all our parameters are included
        params.update(
            {
                "preserve_dataframe": self.preserve_dataframe,
                "fit_jointly": self.fit_jointly,
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

        if reset_fitted:
            self._fitted = False

        return super().set_params(**params)

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """String representation of the estimator."""
        params = []

        if self.bin_spec is not None:
            params.append(f"bin_spec=...")  # Abbreviated since it can be large
        if self.bin_representatives is not None:
            params.append(f"bin_representatives=...")
        if self.preserve_dataframe:
            params.append(f"preserve_dataframe={self.preserve_dataframe}")
        if self.fit_jointly:
            params.append(f"fit_jointly={self.fit_jointly}")

        param_str = ", ".join(params)
        return f"FlexibleBinningBase({param_str})"
