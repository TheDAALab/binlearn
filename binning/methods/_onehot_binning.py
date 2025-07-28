"""
OneHotBinning transformer - creates a singleton bin for each unique value in the data.
"""

from typing import Any, Dict, List, Tuple, Optional
import numpy as np

from ..base._types import (
    BinEdges, BinEdgesDict, FlexibleBinSpec, FlexibleBinDefs,
    ColumnId, ColumnList, OptionalColumnList, GuidanceColumns, ArrayLike
)
from ..base._flexible_binning_base import FlexibleBinningBase
from ..base._repr_mixin import ReprMixin


class OneHotBinning(ReprMixin, FlexibleBinningBase):
    """
    Creates a singleton bin for each unique value in numeric data.

    This is NOT one-hot encoding that expands columns. Instead, it's a binning
    method that creates one bin per unique value, where each bin is defined as
    {"singleton": value}. The output has the same shape as the input.

    **Important**: This method only supports numeric data. Non-numeric data will
    raise a ValueError during fitting.

    For example:
    - Input: [[1.0, 10.0], [2.0, 20.0], [1.0, 10.0]]
    """

    def __init__(
        self,
        preserve_dataframe: bool = False,
        fit_jointly: bool = False,
        bin_spec: Any = None,
        bin_representatives: Any = None,
        max_unique_values: int = 100,
        **kwargs,
    ):
        """
        Initialize the OneHotBinning transformer.

        Parameters
        ----------
        preserve_dataframe : bool, default=False
            If True, preserve DataFrame structure in output.

        fit_jointly : bool, default=False
            If True, find unique values across all columns and create
            the same singleton bins for each column.

        bin_spec : dict or None, default=None
            Pre-defined bin specification.

        bin_representatives : dict or None, default=None
            Pre-defined bin representatives.

        max_unique_values : int, default=100
            Maximum number of unique values per column to prevent
            memory issues with high-cardinality data.
        """
        super().__init__(
            bin_spec=bin_spec,
            bin_representatives=bin_representatives,
            preserve_dataframe=preserve_dataframe,
            fit_jointly=fit_jointly,
            **kwargs,
        )
        self.max_unique_values = max_unique_values

    def _calculate_flexible_bins(
        self, x_col: np.ndarray, col_id: ColumnId, guidance_data: Optional[np.ndarray] = None
    ) -> Tuple[FlexibleBinDefs, BinEdges]:
        """
        Calculate singleton bins for each unique value in the column.

        Note: This method only supports numeric data. Non-numeric data will raise an error.

        Args:
            x_col: Numeric data for a single column.
            col_id: Column identifier.
            guidance_data: Optional guidance data (not used in one-hot binning).

        Returns:
            Tuple containing:
            - List of singleton bin definitions: [{"singleton": val1}, {"singleton": val2}, ...]
            - List of representative values: [val1, val2, ...]
        """
        # Convert to numeric array - this will raise an error for non-numeric data
        try:
            x_col = np.asarray(x_col, dtype=float)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"OneHotBinning only supports numeric data. "
                f"Column {col_id} contains non-numeric values. "
                f"Original error: {str(e)}"
            ) from e

        # Remove NaN/inf values for finding unique values
        finite_mask = np.isfinite(x_col)
        if not finite_mask.any():
            # All values are NaN/inf - create a default bin
            return [{"singleton": 0.0}], [0.0]

        finite_values = x_col[finite_mask]
        unique_values = np.unique(finite_values)

        # Check if we have too many unique values
        if len(unique_values) > self.max_unique_values:
            raise ValueError(
                f"Column {col_id} has {len(unique_values)} unique values, "
                f"which exceeds max_unique_values={self.max_unique_values}. "
                f"Consider using a different binning method for high-cardinality data."
            )

        # Create singleton bins for each unique value
        bin_defs = []
        representatives = []

        for val in unique_values:
            val = float(val)  # Convert to Python float
            bin_defs.append({"singleton": val})
            representatives.append(val)

        return bin_defs, representatives

    def _calculate_joint_parameters(self, X: np.ndarray, columns: ColumnList) -> Dict[str, Any]:
        """
        Calculate joint parameters for one-hot binning.

        For joint fitting, we find all unique values across all columns
        and use the same set of singleton bins for each column.
        """
        # Collect all finite values from all columns
        all_finite_values = []

        for i in range(X.shape[1]):
            col_data = np.asarray(X[:, i], dtype=float)
            finite_mask = np.isfinite(col_data)
            if finite_mask.any():
                all_finite_values.extend(col_data[finite_mask])

        if not all_finite_values:
            # No valid values across any column
            global_unique = np.array([0.0])
        else:
            global_unique = np.unique(all_finite_values)

        # Check global unique value limit
        if len(global_unique) > self.max_unique_values:
            raise ValueError(
                f"Joint fitting found {len(global_unique)} unique values across all columns, "
                f"which exceeds max_unique_values={self.max_unique_values}. "
                f"Consider using fit_jointly=False or increasing max_unique_values."
            )

        return {"global_unique_values": global_unique}

    def _calculate_flexible_bins_jointly(
        self, x_col: np.ndarray, col_id: ColumnId, joint_params: Dict[str, Any]
    ) -> Tuple[FlexibleBinDefs, BinEdges]:
        """
        Calculate singleton bins using joint parameters.

        In joint mode, all columns get the same set of singleton bins based on
        the unique values found across all columns.
        """
        global_unique = joint_params["global_unique_values"]

        # Create singleton bins for all global unique values
        bin_defs = []
        representatives = []

        for val in global_unique:
            val = float(val)  # Convert to Python float
            bin_defs.append({"singleton": val})
            representatives.append(val)

        return bin_defs, representatives

    def _get_binning_params(self) -> Dict[str, Any]:
        """Get one-hot binning specific parameters."""
        params = super()._get_binning_params()
        params.update({
            "max_unique_values": self.max_unique_values,
        })
        return params

    def _handle_bin_params(self, params: Dict[str, Any]) -> bool:
        """Handle one-hot binning specific parameter changes."""
        reset_fitted = super()._handle_bin_params(params)

        if "max_unique_values" in params:
            self.max_unique_values = params.pop("max_unique_values")
            reset_fitted = True

        return reset_fitted
