"""
Clean Equal Width Minimum Weight binning implementation for  architecture.

This module provides EqualWidthMinimumWeightBinning that inherits from SupervisedBinningBase.
Uses equal-width bins with minimum weight constraints from guidance data.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ..base import SupervisedBinningBase
from ..config import apply_config_defaults
from ..utils import (
    BinEdgesDict,
    ConfigurationError,
    DataQualityWarning,
    FittingError,
    resolve_n_bins_parameter,
    validate_bin_number_for_calculation,
    validate_bin_number_parameter,
)


# pylint: disable=too-many-ancestors
class EqualWidthMinimumWeightBinning(SupervisedBinningBase):
    """Equal-width binning with minimum weight constraint implementation using  architecture.

    Creates bins of equal width across the range of each feature, but adjusts the
    number of bins to ensure each bin contains at least the specified minimum total
    weight from the guidance column. This method combines the interpretability of
    equal-width binning with weight-based constraints for more balanced bins.

    This implementation follows the clean  architecture with straight inheritance,
    dynamic column resolution, and parameter reconstruction capabilities.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        n_bins: int | str | None = None,
        minimum_weight: float | None = None,
        bin_range: tuple[float, float] | None = None,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        guidance_columns: Any = None,
        *,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
        class_: str | None = None,  # For reconstruction compatibility
        module_: str | None = None,  # For reconstruction compatibility
    ):
        """Initialize Equal Width Minimum Weight binning."""
        # Prepare user parameters for config integration (exclude never-configurable params)
        user_params = {
            "n_bins": n_bins,
            "minimum_weight": minimum_weight,
            "bin_range": bin_range,
            "clip": clip,
            "preserve_dataframe": preserve_dataframe,
        }
        # Remove None values to allow config defaults to take effect
        user_params = {k: v for k, v in user_params.items() if v is not None}

        # Apply configuration defaults for equal_width_minimum_weight method
        resolved_params = apply_config_defaults("equal_width_minimum_weight", user_params)

        # Store method-specific parameters
        self.n_bins = resolved_params.get("n_bins", 10)
        self.minimum_weight = resolved_params.get("minimum_weight", 1.0)
        self.bin_range = resolved_params.get("bin_range", None)

        # Initialize parent with resolved parameters (never-configurable params passed as-is)
        SupervisedBinningBase.__init__(
            self,
            clip=resolved_params.get("clip"),
            preserve_dataframe=resolved_params.get("preserve_dataframe"),
            guidance_columns=guidance_columns,  # Never configurable
            bin_edges=bin_edges,  # Never configurable
            bin_representatives=bin_representatives,  # Never configurable
        )

    def _validate_params(self) -> None:
        """Validate Equal Width Minimum Weight binning parameters."""
        # Call parent validation
        SupervisedBinningBase._validate_params(self)

        # Validate n_bins using centralized utility
        validate_bin_number_parameter(self.n_bins, param_name="n_bins")

        # Validate minimum_weight parameter
        if not isinstance(self.minimum_weight, int | float) or self.minimum_weight <= 0:
            raise ConfigurationError(
                "minimum_weight must be a positive number",
                suggestions=["Example: minimum_weight=1.0"],
            )

        # Validate bin_range parameter
        if self.bin_range is not None:
            if not isinstance(self.bin_range, list | tuple) or len(self.bin_range) != 2:
                raise ConfigurationError(
                    "bin_range must be a tuple/list of two numbers (min, max)",
                    suggestions=["Example: bin_range=(0, 100)"],
                )

            min_val, max_val = self.bin_range
            if not isinstance(min_val, int | float) or not isinstance(max_val, int | float):
                raise ConfigurationError(
                    "bin_range values must be numbers",
                    suggestions=["Example: bin_range=(0.0, 100.0)"],
                )

            if min_val >= max_val:
                raise ConfigurationError(
                    "bin_range minimum must be less than maximum",
                    suggestions=["Example: bin_range=(0, 100) where 0 < 100"],
                )

    def _calculate_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        guidance_data: np.ndarray[Any, Any] | None = None,
    ) -> tuple[list[float], list[float]]:
        """Calculate equal-width bins with minimum weight constraint for a single column.

        Computes bin edges and representatives starting with equal-width bins and then
        merging adjacent bins that don't meet the minimum weight requirement from the
        guidance data.

        Args:
            x_col: Preprocessed column data (from base class)
            col_id: Column identifier for error reporting
            guidance_data: Weight values for each data point (required)

        Returns:
            Tuple of (bin_edges, bin_representatives)

        Raises:
            FittingError: If guidance_data is None or insufficient data for binning
        """
        # Require guidance data for supervised binning
        if guidance_data is None:
            raise FittingError(
                f"Column {col_id}: EqualWidthMinimumWeightBinning requires guidance_data "
                "to calculate weights for minimum weight constraint"
            )

        # Validate n_bins for calculation
        validate_bin_number_for_calculation(self.n_bins, param_name="n_bins")

        resolved_n_bins = resolve_n_bins_parameter(
            self.n_bins, data_shape=(len(x_col), 1), param_name="n_bins"
        )

        # Extract the single weight column (guaranteed to have shape (n_samples, 1) by SupervisedBinningBase)
        weights = guidance_data[:, 0]

        return self._create_equal_width_minimum_weight_bins(x_col, weights, col_id, resolved_n_bins)

    def _create_equal_width_minimum_weight_bins(
        self,
        x_col: np.ndarray[Any, Any],
        weights: np.ndarray[Any, Any],
        col_id: Any,
        n_bins: int,
    ) -> tuple[list[float], list[float]]:
        """Create equal-width bins with minimum weight constraints.

        Args:
            x_col: Preprocessed column data (no NaN/inf values)
            weights: Weight values for each data point
            col_id: Column identifier for error reporting
            n_bins: Number of initial bins to create

        Returns:
            Tuple of (bin_edges, bin_representatives)

        Note:
            The data is already preprocessed by the base class, so we don't need
            to handle NaN/inf values here.
        """
        # Check for negative weights
        if np.any(weights < 0):
            raise ValueError(
                f"Column {col_id}: guidance_data contains negative values. "
                "All weights must be non-negative."
            )

        # Determine the range for binning
        if self.bin_range is not None:
            min_val, max_val = self.bin_range
        else:
            min_val, max_val = float(np.min(x_col)), float(np.max(x_col))

        # Handle constant data
        if min_val == max_val:
            # Create a single bin with small extension
            epsilon = 1e-8 if min_val != 0 else 1e-8
            edges = [min_val - epsilon, min_val + epsilon]
            representatives = [min_val]
            return edges, representatives

        # Create initial equal-width bins
        initial_edges = np.linspace(min_val, max_val, n_bins + 1)

        # Calculate weights in each initial bin
        bin_weights = []
        for i in range(n_bins):
            left_edge = initial_edges[i]
            right_edge = initial_edges[i + 1]

            # Include left edge, exclude right edge (except for last bin)
            if i == n_bins - 1:  # Last bin includes right edge
                mask = (x_col >= left_edge) & (x_col <= right_edge)
            else:
                mask = (x_col >= left_edge) & (x_col < right_edge)

            total_weight = np.sum(weights[mask])
            bin_weights.append(total_weight)

        # Merge bins with insufficient weight
        merged_edges, merged_weights = self._merge_underweight_bins(
            list(initial_edges), bin_weights, col_id
        )

        # Create representatives as bin centers
        representatives = []
        for i in range(len(merged_edges) - 1):
            center = (merged_edges[i] + merged_edges[i + 1]) / 2
            representatives.append(center)

        return merged_edges, representatives

    def _merge_underweight_bins(
        self,
        edges: list[float],
        bin_weights: list[float],
        col_id: Any,
    ) -> tuple[list[float], list[float]]:
        """Merge adjacent bins that don't meet minimum weight requirement.

        Args:
            edges: Initial bin edges
            bin_weights: Weight in each bin
            col_id: Column identifier for warnings

        Returns:
            Tuple of (merged_edges, merged_weights)
        """
        if len(edges) <= 2:  # Only one bin, can't merge further
            return edges, bin_weights

        merged_edges = [edges[0]]  # Start with first edge
        merged_weights = []
        current_weight = 0.0

        for i, weight in enumerate(bin_weights):
            current_weight += weight

            # Check if we've reached minimum weight or this is the last bin
            if current_weight >= self.minimum_weight or i == len(bin_weights) - 1:
                # Close current merged bin
                merged_edges.append(edges[i + 1])
                merged_weights.append(current_weight)
                current_weight = 0.0

        # Check if we ended up with no bins due to all weights being too small
        if len(merged_weights) == 0:
            warnings.warn(
                f"Column {col_id}: No bins meet minimum weight requirement "
                f"({self.minimum_weight}). Creating single bin with total weight "
                f"{sum(bin_weights)}.",
                DataQualityWarning,
                stacklevel=2,
            )
            # Return single bin with all data
            return [edges[0], edges[-1]], [sum(bin_weights)]

        return merged_edges, merged_weights
