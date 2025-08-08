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
from ..utils import ArrayLike, BinEdgesDict, ConfigurationError


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

        For manual binning, the object is already fitted during initialization.
        This method only performs validation.

        Args:
            X: Input data (used only for validation)
            y: Target values (ignored for manual binning)
            **fit_params: Additional fit parameters (ignored)

        Returns:
            Self (fitted transformer)
        """
        # Just validate parameters - object is already fitted
        self._validate_params()
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
        """Should never be called for manual binning.

        Manual binning uses pre-defined specifications, so this method
        should never be invoked. If called, it indicates a logic error.
        """
        raise NotImplementedError(
            "Manual binning uses pre-defined specifications. "
            "_calculate_bins should never be called for ManualIntervalBinning."
        )
