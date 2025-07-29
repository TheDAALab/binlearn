"""
OneHotBinning transformer - creates a singleton bin for each unique value in the data.
"""

from typing import Any, Dict, List, Tuple, Optional
import numpy as np

from ..utils.types import (
    FlexibleBinSpec, FlexibleBinDefs, FlexibleBinDef, BinEdges, BinEdgesDict,
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
        preserve_dataframe: Optional[bool] = None,
        bin_spec: Optional[FlexibleBinSpec] = None,
        bin_representatives: Optional[BinEdgesDict] = None,
        max_unique_values: int = 100,
        **kwargs,
    ):
        """
        Initialize OneHotBinning.

        Creates singleton bins for each unique value in the data.
        This is NOT traditional one-hot encoding - instead, it creates
        bins where each bin contains exactly one unique value.

        Parameters
        ----------
        preserve_dataframe : bool, optional
            Whether to preserve pandas DataFrame structure in output.

        bin_spec : dict or None, default=None
            Pre-defined bin specification.

        bin_representatives : dict or None, default=None
            Pre-defined bin representatives.

        max_unique_values : int, default=100
            Maximum number of unique values per column to prevent
            memory issues with high-cardinality data.
        """
        # Remove fit_jointly from kwargs if present to avoid conflicts
        kwargs.pop('fit_jointly', None)
        
        super().__init__(
            bin_spec=bin_spec,
            bin_representatives=bin_representatives,
            preserve_dataframe=preserve_dataframe,
            fit_jointly=False,  # Always use per-column fitting
            **kwargs,
        )
        self.max_unique_values = max_unique_values

    def _validate_params(self) -> None:
        """Validate OneHotBinning specific parameters."""
        super()._validate_params()
        
        if not isinstance(self.max_unique_values, int) or self.max_unique_values <= 0:
            raise ValueError("max_unique_values must be a positive integer")

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




