"""
Clean Manual Flexible binning implementation for  architecture.

This module provides ManualFlexibleBinning that inherits from FlexibleBinningBase.
Uses user-provided flexible bin specifications with both singleton and interval bins.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base import FlexibleBinningBase
from ..config import apply_config_defaults
from ..utils import (
    ArrayLike,
    BinEdgesDict,
    BinReps,
    ConfigurationError,
    FlexibleBinDefs,
    FlexibleBinSpec,
)


class ManualFlexibleBinning(FlexibleBinningBase):
    """Manual flexible binning implementation using  architecture.

    Creates bins using explicitly provided bin specifications that can include both:
    - Singleton bins: exact numeric value matches (e.g., specific values or outliers)
    - Interval bins: numeric range matches (e.g., [min, max) intervals)

    This implementation follows the clean  architecture with straight inheritance,
    dynamic column resolution, and parameter reconstruction capabilities.
    """

    def __init__(
        self,
        bin_spec: FlexibleBinSpec,
        bin_representatives: BinEdgesDict | None = None,
        preserve_dataframe: bool | None = None,
        class_: str | None = None,  # For reconstruction compatibility
        module_: str | None = None,  # For reconstruction compatibility
    ):
        """Initialize Manual Flexible binning."""
        # For manual flexible binning, bin_spec is required and passed directly
        if bin_spec is None:
            raise ConfigurationError(
                "bin_spec must be provided for ManualFlexibleBinning",
                suggestions=[
                    "Provide bin_spec as a dictionary mapping columns to flexible bin lists",
                    "Example: bin_spec={0: [1.5, (2, 5), (5, 10)], 1: [(0, 25), (25, 50)]}",
                ],
            )

        # Prepare user parameters for config integration (exclude never-configurable params)
        user_params = {
            "preserve_dataframe": preserve_dataframe,
        }
        # Remove None values to allow config defaults to take effect
        user_params = {k: v for k, v in user_params.items() if v is not None}

        # Apply configuration defaults for manual_flexible method
        resolved_params = apply_config_defaults("manual_flexible", user_params)

        # Initialize parent with resolved parameters (never-configurable params passed as-is)
        # Manual flexible binning doesn't need guidance_columns
        FlexibleBinningBase.__init__(
            self,
            preserve_dataframe=resolved_params.get("preserve_dataframe"),
            guidance_columns=None,  # Not needed for unsupervised manual binning
            bin_spec=bin_spec,  # Required for manual flexible binning
            bin_representatives=bin_representatives,  # Never configurable
        )

    def fit(
        self, X: ArrayLike, y: ArrayLike | None = None, **fit_params: Any
    ) -> ManualFlexibleBinning:
        """Fit the Manual Flexible binning (no-op since bin specs are pre-defined).

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
        """Validate Manual Flexible binning parameters."""
        # ManualFlexibleBinning specific validation: bin_spec is required
        if self.bin_spec is None or len(self.bin_spec) == 0:
            raise ConfigurationError(
                "bin_spec must be provided and non-empty for ManualFlexibleBinning",
                suggestions=[
                    "Provide bin_spec as a dictionary: {column: [spec1, spec2, ...]}",
                    "Example: bin_spec={0: [1.5, (2, 5)], 1: [(0, 25), (25, 50)]}",
                ],
            )

        # Call parent validation for common checks
        FlexibleBinningBase._validate_params(self)

    def _calculate_flexible_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        guidance_data: np.ndarray[Any, Any] | None = None,
    ) -> tuple[FlexibleBinDefs, BinReps]:
        """Return pre-defined flexible bin specifications without calculation.

        Since ManualFlexibleBinning uses user-provided bin specifications, this method
        simply returns the pre-specified bins and representatives without performing
        any data-based calculations.

        Args:
            x_col: Input data column (ignored in manual binning)
            col_id: Column identifier to retrieve pre-defined bin specifications
            guidance_data: Not used for manual flexible binning

        Returns:
            Tuple of (bin_specs, bin_representatives)

        Raises:
            BinningError: If no bin specifications are defined for the specified column
        """
        raise NotImplementedError(
            "Manual binning uses pre-defined specifications. "
            "_calculate_bins should never be called for ManualIntervalBinning."
        )
