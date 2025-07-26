"""
OneHotBinning transformer - creates a singleton bin for each unique value in the data.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from ..base._flexible_binning_base import FlexibleBinningBase


class OneHotBinning(FlexibleBinningBase):
    """
    Creates a singleton bin for each unique value in the data.

    This is NOT one-hot encoding that expands columns. Instead, it's a binning
    method that creates one bin per unique value, where each bin is defined as
    {"singleton": value}. The output has the same shape as the input.

    For example:
    - Input: [[1, 'A'], [2, 'B'], [1, 'A']]
    - Bins created: {0: [{"singleton": 1}, {"singleton": 2}],
                     1: [{"singleton": 'A'}, {"singleton": 'B'}]}
    - Transform output: [[0, 0], [1, 1], [0, 0]]  # Same shape as input
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
        self, x_col: np.ndarray, col_id: Any
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Calculate singleton bins for each unique value in the column.

        Args:
            x_col: Data for a single column.
            col_id: Column identifier.

        Returns:
            Tuple containing:
            - List of singleton bin definitions: [{"singleton": val1}, {"singleton": val2}, ...]
            - List of representative values: [val1, val2, ...]
        """
        # Convert to appropriate type and handle NaNs
        x_col = np.asarray(x_col)

        # Remove NaNs/infinites for finding unique values
        if np.issubdtype(x_col.dtype, np.floating):
            finite_mask = np.isfinite(x_col)
            if not finite_mask.any():
                # All values are NaN/inf - create a default bin
                return [{"singleton": 0.0}], [0.0]
            finite_values = x_col[finite_mask]
        else:
            # For non-floating types, just remove NaNs if they exist
            if x_col.dtype == object:
                finite_values = x_col[x_col != None]  # Remove None values
            else:
                finite_values = x_col

        unique_values = np.unique(finite_values)

        # Check if we have too many unique values
        if len(unique_values) > self.max_unique_values:
            raise ValueError(
                f"Column {col_id} has {len(unique_values)} unique values, "
                f"which exceeds max_unique_values={self.max_unique_values}. "
                f"Consider using a different binning method for high-cardinality data."
            )

        # Create singleton bins for each unique value
        # Convert to Python native types to avoid JSON serialization issues
        bin_defs = []
        representatives = []

        for val in unique_values:
            if isinstance(val, (np.integer, np.floating)):
                val = float(val)
            elif isinstance(val, np.str_):
                val = str(val)

            bin_defs.append({"singleton": val})
            representatives.append(val)

        return bin_defs, representatives

    def _calculate_joint_parameters(self, X: np.ndarray, columns: List[Any]) -> Dict[str, Any]:
        """
        Calculate joint parameters for one-hot binning.

        For joint fitting, we find all unique values across all columns
        and use the same set of singleton bins for each column.
        """
        # Collect all finite values from all columns
        all_finite_values = []

        for i in range(X.shape[1]):
            col_data = X[:, i]

            if np.issubdtype(col_data.dtype, np.floating):
                finite_mask = np.isfinite(col_data)
                if finite_mask.any():
                    all_finite_values.extend(col_data[finite_mask])
            else:
                # Handle non-floating types
                if col_data.dtype == object:
                    valid_values = col_data[col_data != None]
                else:
                    valid_values = col_data
                all_finite_values.extend(valid_values)

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
        self, x_col: np.ndarray, col_id: Any, joint_params: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
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
            if isinstance(val, (np.integer, np.floating)):
                val = float(val)
            elif isinstance(val, np.str_):
                val = str(val)

            bin_defs.append({"singleton": val})
            representatives.append(val)

        return bin_defs, representatives

    def __repr__(self) -> str:
        """String representation of the estimator."""
        params = []

        if self.max_unique_values != 100:
            params.append(f"max_unique_values={self.max_unique_values}")
        if self.preserve_dataframe:
            params.append(f"preserve_dataframe={self.preserve_dataframe}")
        if self.fit_jointly:
            params.append(f"fit_jointly={self.fit_jointly}")
        if self.bin_spec is not None:
            params.append("bin_spec=...")
        if self.bin_representatives is not None:
            params.append("bin_representatives=...")

        param_str = ", ".join(params)
        return f"OneHotBinning({param_str})"
