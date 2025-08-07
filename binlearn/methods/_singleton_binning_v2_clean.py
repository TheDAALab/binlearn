"""
Clean singleton binning implementation for V2 architecture.

This module provides SingletonBinningV2 that inherits from FlexibleBinningBaseV2.
Each unique value gets its own bin.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import get_config
from ..utils.types import BinEdgesDict
from ..base._flexible_binning_base_v2 import FlexibleBinningBaseV2


class SingletonBinningV2(FlexibleBinningBaseV2):
    """Singleton binning implementation using V2 architecture.

    Creates one bin for each unique value in the data.
    Each unique value becomes both a bin edge and its own representative.
    """

    def __init__(
        self,
        preserve_dataframe: bool | None = None,
        fit_jointly: bool | None = None,
        guidance_columns: Any = None,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
    ):
        """Initialize singleton binning."""
        # Initialize parent
        FlexibleBinningBaseV2.__init__(
            self,
            preserve_dataframe=preserve_dataframe,
            fit_jointly=fit_jointly,
            guidance_columns=guidance_columns,
            bin_edges=bin_edges,
            bin_representatives=bin_representatives,
        )

    def _validate_params(self) -> None:
        """Validate singleton binning parameters."""
        # Call parent validation
        FlexibleBinningBaseV2._validate_params(self)
        # No additional validation needed for singleton binning

    def _calculate_flexible_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        guidance_data: np.ndarray[Any, Any] | None = None,
    ) -> tuple[list[Any], list[Any]]:
        """Calculate singleton bins - one bin per unique value.

        Args:
            x_col: Column data to analyze
            col_id: Column identifier (unused for singleton)
            guidance_data: Optional guidance data (unused for singleton)

        Returns:
            Tuple of (unique_values, unique_values) - both are the same for singleton binning
        """
        # Find unique values, excluding NaN
        mask_valid = ~(np.isnan(x_col) | np.isinf(x_col))
        valid_data = x_col[mask_valid]

        if len(valid_data) == 0:
            # Handle case where all values are NaN/inf
            unique_values = [0.0]  # Default fallback
        else:
            unique_values = np.unique(valid_data).tolist()

        # For singleton binning, representatives are the same as the unique values
        representatives = unique_values.copy()

        return unique_values, representatives

    def _match_values_to_bin(
        self, column_data: np.ndarray[Any, Any], bin_value: Any, bin_idx: int, col_id: Any
    ) -> np.ndarray[Any, Any]:
        """Match values exactly to their singleton bins.

        Args:
            column_data: The column data to match
            bin_value: The exact value that defines this bin
            bin_idx: The index of this bin
            col_id: Column identifier (unused)

        Returns:
            Boolean mask of exact matches
        """
        # Handle NaN values specially
        if np.isnan(bin_value) if isinstance(bin_value, (int, float)) else False:
            return np.isnan(column_data)

        # Exact match for singleton binning
        return column_data == bin_value
