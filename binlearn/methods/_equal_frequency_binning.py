"""
Clean equal frequency binning implementation for  architecture.

This module provides EqualFrequencyBinning that inherits from IntervalBinningBase.
Creates bins containing approximately equal numbers of observations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import get_config, apply_config_defaults
from ..utils.errors import ConfigurationError
from ..utils.parameter_conversion import (
    resolve_n_bins_parameter,
    validate_bin_number_for_calculation,
    validate_bin_number_parameter,
)
from ..utils.types import BinEdgesDict
from ..base._interval_binning_base import IntervalBinningBase


class EqualFrequencyBinning(IntervalBinningBase):
    """Equal frequency binning implementation using  architecture.

    Creates bins containing approximately equal numbers of observations across
    each feature. Each bin contains roughly the same number of data points,
    making this method useful when you want balanced bin populations regardless
    of the underlying data distribution.

    This implementation follows the clean  architecture with straight inheritance,
    dynamic column resolution, and parameter reconstruction capabilities.
    """

    def __init__(
        self,
        n_bins: int | str | None = None,
        quantile_range: tuple[float, float] | None = None,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        fit_jointly: bool | None = None,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
        class_: str | None = None,  # For reconstruction compatibility
        module_: str | None = None,  # For reconstruction compatibility
    ):
        """Initialize equal frequency binning."""
        # Prepare user parameters for config integration (exclude never-configurable params)
        user_params = {
            "n_bins": n_bins,
            "quantile_range": quantile_range,
            "clip": clip,
            "preserve_dataframe": preserve_dataframe,
            "fit_jointly": fit_jointly,
        }
        # Remove None values to allow config defaults to take effect
        user_params = {k: v for k, v in user_params.items() if v is not None}

        # Apply configuration defaults for equal_frequency method
        resolved_params = apply_config_defaults("equal_frequency", user_params)

        # Store method-specific parameters
        self.n_bins = resolved_params.get("n_bins", 10)
        self.quantile_range = resolved_params.get("quantile_range", None)

        # Initialize parent with resolved parameters (never-configurable params passed as-is)
        IntervalBinningBase.__init__(
            self,
            clip=resolved_params.get("clip"),
            preserve_dataframe=resolved_params.get("preserve_dataframe"),
            fit_jointly=resolved_params.get("fit_jointly"),
            guidance_columns=None,  # Not needed for unsupervised binning
            bin_edges=bin_edges,  # Never configurable
            bin_representatives=bin_representatives,  # Never configurable
        )

    def _validate_params(self) -> None:
        """Validate equal frequency binning parameters."""
        # Call parent validation
        IntervalBinningBase._validate_params(self)

        # Validate n_bins using centralized utility
        validate_bin_number_parameter(self.n_bins, param_name="n_bins")

        # Validate quantile_range if provided
        if self.quantile_range is not None:
            if not isinstance(self.quantile_range, tuple) or len(self.quantile_range) != 2:
                raise ConfigurationError(
                    "quantile_range must be a tuple (min_quantile, max_quantile)",
                    suggestions=["Example: quantile_range=(0.1, 0.9)"],
                )

            min_q, max_q = self.quantile_range
            if (
                not isinstance(min_q, (int, float))
                or not isinstance(max_q, (int, float))
                or min_q < 0
                or max_q > 1
                or min_q >= max_q
            ):
                raise ConfigurationError(
                    "quantile_range values must be numbers between 0 and 1 with min < max",
                    suggestions=["Example: quantile_range=(0.1, 0.9)"],
                )

    def _calculate_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        guidance_data: np.ndarray[Any, Any] | None = None,
    ) -> tuple[list[float], list[float]]:
        """Calculate equal-frequency bins for a single column.

        Computes bin edges and representatives using quantiles to ensure
        approximately equal numbers of observations in each bin.

        Args:
            x_col: Preprocessed column data (from base class)
            col_id: Column identifier for error reporting
            guidance_data: Not used for equal-frequency binning (unsupervised)

        Returns:
            Tuple of (bin_edges, bin_representatives)

        Raises:
            ValueError: If n_bins is invalid or insufficient data for calculation
        """
        # Validate n_bins for calculation
        validate_bin_number_for_calculation(self.n_bins, param_name="n_bins")

        resolved_n_bins = resolve_n_bins_parameter(
            self.n_bins, data_shape=(len(x_col), 1), param_name="n_bins"
        )

        # Get quantile range for this data
        if self.quantile_range is not None:
            min_quantile, max_quantile = self.quantile_range
        else:
            min_quantile, max_quantile = 0.0, 1.0

        return self._create_equal_frequency_bins(
            x_col, col_id, min_quantile, max_quantile, resolved_n_bins
        )

    def _create_equal_frequency_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        min_quantile: float,
        max_quantile: float,
        n_bins: int,
    ) -> tuple[list[float], list[float]]:
        """Create equal-frequency bins using quantiles.

        Args:
            x_col: Preprocessed column data (no NaN/inf values)
            col_id: Column identifier for error reporting
            min_quantile: Minimum quantile (0.0 to 1.0)
            max_quantile: Maximum quantile (0.0 to 1.0)
            n_bins: Number of bins to create

        Returns:
            Tuple of (bin_edges, bin_representatives)

        Note:
            The data is already preprocessed by the base class, so we don't need
            to handle NaN/inf values or constant data here.
        """
        if len(x_col) < n_bins:
            raise ValueError(
                f"Column {col_id}: Insufficient values ({len(x_col)}) "
                f"for {n_bins} bins. Need at least {n_bins} values."
            )

        # Create quantile points from min_quantile to max_quantile
        quantile_points = np.linspace(min_quantile, max_quantile, n_bins + 1)

        # Calculate quantile values
        try:
            edges = np.quantile(x_col, quantile_points)
        except (ValueError, IndexError) as e:
            raise ValueError(f"Column {col_id}: Error calculating quantiles: {e}") from e

        # Convert to list and ensure edges are strictly increasing
        edges = list(edges)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-8

        # Create representatives as bin centers based on quantiles
        reps = []
        for i in range(n_bins):
            # Calculate representative as the median of values in this bin
            bin_mask = (x_col >= edges[i]) & (x_col <= edges[i + 1])
            bin_data = x_col[bin_mask]

            if len(bin_data) > 0:
                # Use median of bin data as representative
                rep = float(np.median(bin_data))
            else:
                # Fallback to bin center if no data in bin
                rep = (edges[i] + edges[i + 1]) / 2
            reps.append(rep)

        return edges, reps
