"""
Clean singleton binning implementation for  architecture.

This module provides SingletonBinning that inherits from FlexibleBinningBase.
Each unique value gets its own bin.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ..base import FlexibleBinningBase
from ..config import apply_config_defaults
from ..utils import DataQualityWarning


# pylint: disable=too-many-ancestors
class SingletonBinning(FlexibleBinningBase):
    """Singleton binning implementation using  architecture.

    Creates one bin for each unique value in the numeric data. Only supports numeric data.
    Each unique numeric value becomes both a bin edge and its own representative.
    This is useful for discrete numeric variables or when preserving all distinct values.

    This implementation follows the clean  architecture with straight inheritance,
    dynamic column resolution, and parameter reconstruction capabilities.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        preserve_dataframe: bool | None = None,
        fit_jointly: bool | None = None,
        *,
        bin_spec: Any | None = None,  # FlexibleBinSpec
        bin_representatives: Any | None = None,  # BinEdgesDict
        class_: str | None = None,  # For reconstruction compatibility
        module_: str | None = None,  # For reconstruction compatibility
    ):
        """Initialize singleton binning."""
        # Prepare user parameters for config integration (exclude never-configurable params)
        user_params = {
            "preserve_dataframe": preserve_dataframe,
            "fit_jointly": fit_jointly,
        }
        # Remove None values to allow config defaults to take effect
        user_params = {k: v for k, v in user_params.items() if v is not None}

        # Apply configuration defaults for singleton method
        params = apply_config_defaults("singleton", user_params)

        # Initialize parent with resolved config parameters (no guidance_columns for singleton binning)
        # Note: bin_spec and bin_representatives are never set from config
        FlexibleBinningBase.__init__(
            self,
            preserve_dataframe=params.get("preserve_dataframe"),
            fit_jointly=params.get("fit_jointly"),
            guidance_columns=None,  # Singleton binning doesn't use guidance
            bin_spec=bin_spec,
            bin_representatives=bin_representatives,
        )

    def _validate_params(self) -> None:
        """Validate singleton binning parameters."""
        # Call parent validation
        FlexibleBinningBase._validate_params(self)
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
        # Find unique values, excluding NaN and inf
        mask_valid = np.isfinite(x_col)
        valid_data = x_col[mask_valid]

        if len(valid_data) == 0:
            # Handle case where all values are NaN/inf - create a minimal valid bin
            warnings.warn(
                f"Column {col_id} contains only NaN/inf values. Creating default bin.",
                DataQualityWarning,
                stacklevel=2,
            )
            unique_values = [0.0]  # Default fallback
        else:
            unique_values = np.unique(valid_data).tolist()

        # For singleton binning, representatives are the same as the unique values
        representatives = unique_values.copy()

        return unique_values, representatives
