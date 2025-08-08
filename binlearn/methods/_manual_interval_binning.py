"""
Clean Manual Interval binning implementation for  architecture.

This module provides ManualIntervalBinning that inherits from IntervalBinningBase.
Uses user-provided bin edges rather than inferring them from data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base._interval_binning_base import IntervalBinningBase
from ..config import apply_config_defaults
from ..utils import ArrayLike, BinEdgesDict, BinningError, ConfigurationError


class ManualIntervalBinning(IntervalBinningBase):
    """Manual interval binning implementation using  architecture.

    Creates bins using explicitly provided bin edges, giving users complete control
    over binning boundaries. Unlike automatic binning methods, this transformer
    never infers bin edges from data - they must always be provided by the user.

    This implementation follows the clean  architecture with straight inheritance,
    dynamic column resolution, and parameter reconstruction capabilities.
    """

    def __init__(
        self,
        bin_edges: BinEdgesDict,
        bin_representatives: BinEdgesDict | None = None,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        class_: str | None = None,  # For reconstruction compatibility
        module_: str | None = None,  # For reconstruction compatibility
    ):
        """Initialize Manual Interval binning."""
        # For manual binning, bin_edges is required and passed directly
        if bin_edges is None:
            raise ConfigurationError(
                "bin_edges must be provided for ManualIntervalBinning",
                suggestions=[
                    "Provide bin_edges as a dictionary mapping columns to edge lists",
                    "Example: bin_edges={0: [0, 10, 20, 30], 1: [0, 5, 15, 25]}",
                ],
            )

        # Prepare user parameters for config integration (exclude never-configurable params)
        user_params = {
            "clip": clip,
            "preserve_dataframe": preserve_dataframe,
        }
        # Remove None values to allow config defaults to take effect
        user_params = {k: v for k, v in user_params.items() if v is not None}

        # Apply configuration defaults for manual_interval method
        resolved_params = apply_config_defaults("manual_interval", user_params)

        # Initialize parent with resolved parameters (never-configurable params passed as-is)
        # Manual binning doesn't need fit_jointly or guidance_columns
        IntervalBinningBase.__init__(
            self,
            clip=resolved_params.get("clip"),
            preserve_dataframe=resolved_params.get("preserve_dataframe"),
            fit_jointly=False,  # Manual binning doesn't fit from data
            guidance_columns=None,  # Not needed for unsupervised manual binning
            bin_edges=bin_edges,  # Required for manual binning
            bin_representatives=bin_representatives,  # Never configurable
        )

    def fit(
        self, X: ArrayLike, y: ArrayLike | None = None, **fit_params: Any
    ) -> ManualIntervalBinning:
        """Fit the Manual Interval binning (no-op since bins are pre-defined).

        For manual binning, no fitting is required since bin edges are provided
        by the user. This method performs validation and returns self.

        Args:
            X: Input data (used only for validation)
            y: Target values (ignored for manual binning)
            **fit_params: Additional fit parameters (ignored)

        Returns:
            Self (fitted transformer)
        """
        # Validate parameters but don't actually fit anything
        self._validate_params()

        # Manual binning is always "fitted" since bins are pre-defined
        self._is_fitted = True
        return self

    def _validate_params(self) -> None:
        """Validate Manual Interval binning parameters."""
        # Call parent validation
        IntervalBinningBase._validate_params(self)

        # ManualIntervalBinning specific validation: bin_edges is required
        if self.bin_edges is None:
            raise ConfigurationError(
                "bin_edges must be provided for ManualIntervalBinning",
                suggestions=[
                    "Provide bin_edges as a dictionary mapping columns to edge lists",
                    "Example: bin_edges={0: [0, 10, 20, 30], 1: [0, 5, 15, 25]}",
                ],
            )

        if not self.bin_edges:  # Empty dict
            raise ConfigurationError(
                "bin_edges cannot be empty for ManualIntervalBinning",
                suggestions=[
                    "Provide at least one column with bin edges",
                    "Example: bin_edges={0: [0, 10, 20, 30]}",
                ],
            )

    def _calculate_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        guidance_data: np.ndarray[Any, Any] | None = None,
    ) -> tuple[list[float], list[float]]:
        """Return pre-defined bins without calculation.

        Since ManualIntervalBinning uses user-provided bin edges, this method
        simply returns the pre-specified edges and representatives without
        performing any data-based calculations.

        Args:
            x_col: Input data column (ignored in manual binning)
            col_id: Column identifier to retrieve pre-defined bins
            guidance_data: Not used for manual binning

        Returns:
            Tuple of (bin_edges, bin_representatives)

        Raises:
            BinningError: If no bin edges are defined for the specified column
        """
        # Handle column name mapping for numpy arrays
        # The  architecture uses feature_N names internally, but users provide 0, 1, etc.
        actual_col_key = col_id

        # If col_id is like 'feature_N' and not found, try mapping to integer N
        if (self.bin_edges is None or col_id not in self.bin_edges) and isinstance(col_id, str):
            if col_id.startswith("feature_") and self.bin_edges is not None:
                try:
                    # Extract the number from 'feature_N'
                    col_idx = int(col_id.replace("feature_", ""))
                    if col_idx in self.bin_edges:
                        actual_col_key = col_idx
                except ValueError:
                    pass

        # If original integer key and not found, try mapping to feature_N
        elif (self.bin_edges is None or col_id not in self.bin_edges) and isinstance(col_id, int):
            if self.bin_edges is not None:
                feature_name = f"feature_{col_id}"
                if feature_name in self.bin_edges:
                    actual_col_key = feature_name

        # Get pre-defined edges for this column
        if self.bin_edges is None or actual_col_key not in self.bin_edges:
            raise BinningError(
                f"No bin edges defined for column {col_id}",
                suggestions=[
                    f"Add bin edges for column {col_id} in the bin_edges dictionary",
                    "For numpy arrays, use integer keys (0, 1, 2, ...) in bin_edges",
                    "For DataFrames, use column names as keys in bin_edges",
                    "Ensure all data columns have corresponding bin edge definitions",
                ],
            )

        edges = list(self.bin_edges[actual_col_key])

        # Get or generate representatives
        if self.bin_representatives is not None and actual_col_key in self.bin_representatives:
            representatives = list(self.bin_representatives[actual_col_key])
        else:
            # Auto-generate representatives as bin centers
            representatives = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]

        return edges, representatives
