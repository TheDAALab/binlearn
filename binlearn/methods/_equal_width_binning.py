"""
Clean equal width binning implementation for  architecture.

This module provides EqualWidthBinning that inherits from IntervalBinningBase.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base import IntervalBinningBase
from ..config import apply_config_defaults
from ..utils import BinEdgesDict


class EqualWidthBinning(IntervalBinningBase):
    """Equal width binning implementation using  architecture.

    Creates bins with equal width intervals between the minimum and maximum
    values of each feature. Only supports numeric data.

    This implementation follows the clean  architecture with straight inheritance,
    dynamic column resolution, and parameter reconstruction capabilities.
    """

    def __init__(
        self,
        n_bins: int | None = None,
        bin_range: tuple[float, float] | None = None,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        fit_jointly: bool | None = None,
        guidance_columns: Any = None,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
        class_: str | None = None,  # For reconstruction compatibility
        module_: str | None = None,  # For reconstruction compatibility
    ):
        """Initialize equal width binning."""
        # Prepare user parameters for config integration (exclude never-configurable params)
        user_params = {
            "n_bins": n_bins,
            "bin_range": bin_range,
            "clip": clip,
            "preserve_dataframe": preserve_dataframe,
            "fit_jointly": fit_jointly,
        }
        # Remove None values to allow config defaults to take effect
        user_params = {k: v for k, v in user_params.items() if v is not None}

        # Apply configuration defaults for equal_width method
        params = apply_config_defaults("equal_width", user_params)

        # Store equal width specific parameters
        self.n_bins = params.get("n_bins", 5)  # Fallback to 5 if not in config
        self.bin_range = params.get("bin_range", bin_range)

        # Initialize parent with resolved config parameters
        # Note: bin_edges, bin_representatives, guidance_columns are never set from config
        IntervalBinningBase.__init__(
            self,
            clip=params.get("clip"),
            preserve_dataframe=params.get("preserve_dataframe"),
            fit_jointly=params.get("fit_jointly"),
            guidance_columns=guidance_columns,
            bin_edges=bin_edges,
            bin_representatives=bin_representatives,
        )

    def _validate_params(self) -> None:
        """Validate equal width specific parameters."""
        # Call parent validation
        IntervalBinningBase._validate_params(self)

        # Validate n_bins
        if not isinstance(self.n_bins, int) or self.n_bins < 1:
            raise ValueError("n_bins must be a positive integer")

        # Validate bin_range
        if self.bin_range is not None:
            if (
                not isinstance(self.bin_range, tuple)
                or len(self.bin_range) != 2
                or self.bin_range[0] >= self.bin_range[1]
            ):
                raise ValueError("bin_range must be a tuple (min, max) with min < max")

    def _calculate_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        guidance_data: np.ndarray[Any, Any] | None = None,
    ) -> tuple[list[float], list[float]]:
        """Calculate equal width bin edges and representatives."""
        # Get range for this data
        if self.bin_range is not None:
            min_val, max_val = self.bin_range
        else:
            min_val, max_val = self._get_data_range(x_col, col_id)

        return self._create_equal_width_bins(min_val, max_val, self.n_bins)

    def _get_data_range(self, x_col: np.ndarray[Any, Any], col_id: Any) -> tuple[float, float]:
        """Get the data range from preprocessed data.

        The data has already been validated and preprocessed by the base class,
        so we can directly compute the range.
        """
        min_val: float = np.min(x_col)
        max_val: float = np.max(x_col)

        return float(min_val), float(max_val)

    def _create_equal_width_bins(
        self, min_val: float, max_val: float, n_bins: int
    ) -> tuple[list[float], list[float]]:
        """Create equal-width bins given range and number of bins.

        The range comes from preprocessed data, so constant data
        has already been handled by the base class.
        """
        # Create equal-width bin edges
        edges = np.linspace(min_val, max_val, n_bins + 1)

        # Create representatives as bin centers
        reps = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]

        return list(edges), reps
