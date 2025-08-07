"""
Clean Manual Flexible binning implementation for  architecture.

This module provides ManualFlexibleBinning that inherits from FlexibleBinningBase.
Uses user-provided flexible bin specifications with both singleton and interval bins.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import apply_config_defaults
from ..utils.errors import BinningError, ConfigurationError
from ..utils.types import FlexibleBinDefs, BinReps, FlexibleBinSpec, BinEdgesDict, ArrayLike
from ..base._flexible_binning_base import FlexibleBinningBase


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
    ) -> "ManualFlexibleBinning":
        """Fit the Manual Flexible binning (no-op since bin specs are pre-defined).

        For manual flexible binning, no fitting is required since bin specifications
        are provided by the user. This method performs validation and returns self.

        Args:
            X: Input data (used only for validation)
            y: Target values (ignored for manual binning)
            **fit_params: Additional fit parameters (ignored)

        Returns:
            Self (fitted transformer)
        """
        # Validate parameters but don't actually fit anything
        self._validate_params()

        # Manual flexible binning is always "fitted" since bin specs are pre-defined
        self._is_fitted = True
        return self

    def _validate_params(self) -> None:
        """Validate Manual Flexible binning parameters."""
        # Call parent validation
        FlexibleBinningBase._validate_params(self)

        # ManualFlexibleBinning specific validation: bin_spec is required
        if self.bin_spec is None or len(self.bin_spec) == 0:
            raise ConfigurationError(
                "bin_spec must be provided and non-empty for ManualFlexibleBinning",
                suggestions=[
                    "Provide bin_spec as a dictionary: {column: [spec1, spec2, ...]}",
                    "Example: bin_spec={0: [1.5, (2, 5)], 1: [(0, 25), (25, 50)]}",
                ],
            )

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
        # Handle column name mapping for numpy arrays
        # The  architecture uses feature_N names internally, but users provide 0, 1, etc.
        actual_col_key = col_id

        # If col_id is like 'feature_N' and not found, try mapping to integer N
        if (self.bin_spec is None or col_id not in self.bin_spec) and isinstance(col_id, str):
            if col_id.startswith("feature_") and self.bin_spec is not None:
                try:
                    # Extract the number from 'feature_N'
                    col_idx = int(col_id.replace("feature_", ""))
                    if col_idx in self.bin_spec:
                        actual_col_key = col_idx
                except ValueError:
                    pass

        # If original integer key and not found, try mapping to feature_N
        elif (self.bin_spec is None or col_id not in self.bin_spec) and isinstance(col_id, int):
            if self.bin_spec is not None:
                feature_name = f"feature_{col_id}"
                if feature_name in self.bin_spec:
                    actual_col_key = feature_name

        # Get pre-defined bin specifications for this column
        if self.bin_spec is None or actual_col_key not in self.bin_spec:
            raise BinningError(
                f"No bin specifications defined for column {col_id}",
                suggestions=[
                    f"Add bin specifications for column {col_id} in the bin_spec dictionary",
                    "For numpy arrays, use integer keys (0, 1, 2, ...) in bin_spec",
                    "For DataFrames, use column names as keys in bin_spec",
                    "Ensure all data columns have corresponding bin specification definitions",
                ],
            )

        specs = list(self.bin_spec[actual_col_key])

        # Get or generate representatives
        if self.bin_representatives is not None and actual_col_key in self.bin_representatives:
            representatives = list(self.bin_representatives[actual_col_key])
        else:
            # Auto-generate representatives based on bin type
            representatives = []
            for spec in specs:
                if isinstance(spec, tuple) and len(spec) == 2:
                    # Interval bin: use midpoint as representative
                    representatives.append(float((spec[0] + spec[1]) / 2))
                elif not isinstance(spec, tuple):
                    # Singleton bin: use the value itself as representative
                    # Ensure we can convert to float
                    try:
                        representatives.append(float(spec))
                    except (ValueError, TypeError):
                        # For non-numeric singleton bins, use a placeholder
                        representatives.append(0.0)
                else:
                    # Fallback for unexpected formats
                    representatives.append(0.0)

        return specs, representatives
