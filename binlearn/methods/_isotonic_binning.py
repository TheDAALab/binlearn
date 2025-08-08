"""
Clean Isotonic binning implementation for  architecture.

This module provides IsotonicBinning that inherits from SupervisedBinningBase.
Uses isotonic regression to find optimal cut points that preserve monotonic relationships.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ..config import apply_config_defaults
from ..utils._errors import ConfigurationError, FittingError
from ..utils._parameter_conversion import (
    validate_bin_number_parameter,
)
from ..utils._types import BinEdgesDict
from ..base._supervised_binning_base import SupervisedBinningBase


class IsotonicBinning(SupervisedBinningBase):
    """Isotonic regression-based monotonic binning implementation using  architecture.

    Creates bins using isotonic regression to find optimal cut points that preserve
    monotonic relationships between features and targets. The transformer fits an
    isotonic (non-decreasing) function to the data and identifies significant changes
    in this function to determine bin boundaries.

    This implementation follows the clean  architecture with straight inheritance,
    dynamic column resolution, and parameter reconstruction capabilities.
    """

    def __init__(
        self,
        max_bins: int | str | None = None,
        min_samples_per_bin: int | None = None,
        increasing: bool | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        min_change_threshold: float | None = None,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        guidance_columns: Any = None,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
        class_: str | None = None,  # For reconstruction compatibility
        module_: str | None = None,  # For reconstruction compatibility
    ):
        """Initialize Isotonic binning."""
        # Prepare user parameters for config integration (exclude never-configurable params)
        user_params = {
            "max_bins": max_bins,
            "min_samples_per_bin": min_samples_per_bin,
            "increasing": increasing,
            "y_min": y_min,
            "y_max": y_max,
            "min_change_threshold": min_change_threshold,
            "clip": clip,
            "preserve_dataframe": preserve_dataframe,
        }
        # Remove None values to allow config defaults to take effect
        user_params = {k: v for k, v in user_params.items() if v is not None}

        # Apply configuration defaults for isotonic method
        resolved_params = apply_config_defaults("isotonic", user_params)

        # Store method-specific parameters
        self.max_bins = resolved_params.get("max_bins", 10)
        self.min_samples_per_bin = resolved_params.get("min_samples_per_bin", 5)
        self.increasing = resolved_params.get("increasing", True)
        self.y_min = resolved_params.get("y_min", None)
        self.y_max = resolved_params.get("y_max", None)
        self.min_change_threshold = resolved_params.get("min_change_threshold", 0.01)

        # Dictionary to store fitted isotonic models for each feature
        self._isotonic_models: dict[Any, IsotonicRegression] = {}

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
        """Validate Isotonic binning parameters."""
        # Call parent validation
        SupervisedBinningBase._validate_params(self)

        # Validate max_bins using centralized utility
        validate_bin_number_parameter(self.max_bins, param_name="max_bins")

        # Validate min_samples_per_bin parameter
        if not isinstance(self.min_samples_per_bin, int) or self.min_samples_per_bin < 1:
            raise ConfigurationError(
                "min_samples_per_bin must be a positive integer",
                suggestions=["Example: min_samples_per_bin=5"],
            )

        # Validate increasing parameter
        if not isinstance(self.increasing, bool):
            raise ConfigurationError(
                "increasing must be a boolean value",
                suggestions=["Use increasing=True for increasing monotonicity"],
            )

        # Validate y_min and y_max parameters
        if self.y_min is not None and not isinstance(self.y_min, (int, float)):
            raise ConfigurationError(
                "y_min must be a number or None",
                suggestions=["Example: y_min=0.0"],
            )

        if self.y_max is not None and not isinstance(self.y_max, (int, float)):
            raise ConfigurationError(
                "y_max must be a number or None",
                suggestions=["Example: y_max=1.0"],
            )

        if self.y_min is not None and self.y_max is not None and self.y_min >= self.y_max:
            raise ConfigurationError(
                "y_min must be less than y_max",
                suggestions=["Example: y_min=0.0, y_max=1.0"],
            )

        # Validate min_change_threshold parameter
        if (
            not isinstance(self.min_change_threshold, (int, float))
            or self.min_change_threshold <= 0
        ):
            raise ConfigurationError(
                "min_change_threshold must be a positive number",
                suggestions=["Example: min_change_threshold=0.01"],
            )

    def _calculate_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        guidance_data: np.ndarray[Any, Any] | None = None,
    ) -> tuple[list[float], list[float]]:
        """Calculate isotonic regression-based bins for a single column.

        Uses isotonic regression to fit a monotonic function to the feature-target
        relationship, then identifies cut points based on significant changes in
        the fitted function.

        Args:
            x_col: Preprocessed column data (from base class)
            col_id: Column identifier for error reporting
            guidance_data: Target/guidance data for supervised binning (required)

        Returns:
            Tuple of (bin_edges, bin_representatives)

        Raises:
            FittingError: If guidance_data is None or insufficient data for binning
        """
        # Require guidance data for supervised binning
        if guidance_data is None:
            raise FittingError(f"Column {col_id}: guidance_data is required for isotonic binning")

        # Convert categorical guidance data to numeric before processing
        guidance_data_numeric = self._prepare_target_values(guidance_data)

        # Check if we have sufficient data
        if len(x_col) < self.min_samples_per_bin:
            raise FittingError(
                f"Column {col_id}: Insufficient data points ({len(x_col)}) "
                f"for isotonic binning. Need at least {self.min_samples_per_bin}."
            )

        # Create isotonic binning
        return self._create_isotonic_bins(x_col, guidance_data_numeric, col_id)

    def _create_isotonic_bins(
        self, x_col: np.ndarray[Any, Any], y_col: np.ndarray[Any, Any], col_id: Any
    ) -> tuple[list[float], list[float]]:
        """Create bins using isotonic regression.

        Fits an isotonic regression model to the feature-target relationship and
        identifies optimal cut points based on changes in the fitted function.

        Args:
            x_col: Clean feature data (no NaN values)
            y_col: Clean target data (no NaN values)
            col_id: Column identifier for model storage

        Returns:
            Tuple of (bin_edges, bin_representatives)

        Note:
            The data is already preprocessed by the base class, so we don't need
            to handle NaN/inf values or constant data here.
        """
        # Handle constant feature data
        if len(np.unique(x_col)) == 1:
            x_val = float(x_col[0])
            return ([x_val - 0.1, x_val + 0.1], [x_val])

        # Sort data by feature values for isotonic regression
        sort_indices = np.argsort(x_col)
        x_sorted = x_col[sort_indices]
        y_sorted = y_col[sort_indices]

        # Ensure both arrays are 1D for sklearn's IsotonicRegression
        x_sorted = x_sorted.flatten()
        y_sorted = y_sorted.flatten()

        # Verify shapes match
        if len(x_sorted) != len(y_sorted):
            raise ValueError(
                f"Column {col_id}: Feature and target arrays have mismatched lengths: "
                f"{len(x_sorted)} vs {len(y_sorted)}"
            )

        # Fit isotonic regression
        try:
            isotonic_model = IsotonicRegression(
                increasing=self.increasing,
                y_min=self.y_min,
                y_max=self.y_max,
                out_of_bounds="clip",
            )
            y_fitted = isotonic_model.fit_transform(x_sorted, y_sorted)
        except Exception as e:
            raise ValueError(f"Column {col_id}: Isotonic regression failed: {e}") from e

        # Store the fitted model
        self._isotonic_models[col_id] = isotonic_model

        # Find cut points based on fitted function changes
        cut_points = self._find_cut_points(x_sorted, y_fitted)

        # Create bin edges and representatives
        return self._create_bins_from_cuts(x_sorted, y_fitted, cut_points, col_id)

    def _prepare_target_values(self, y_values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Prepare target values for isotonic regression.

        Converts categorical targets to numeric values and applies bounds if specified.

        Args:
            y_values: Raw target values

        Returns:
            Processed target values suitable for isotonic regression
        """
        # Handle object/categorical data
        if y_values.dtype == object or not np.issubdtype(y_values.dtype, np.number):
            # Convert categorical to numeric (for classification)
            unique_values = np.unique(y_values)
            value_mapping = {val: i for i, val in enumerate(unique_values)}
            y_processed = np.array([value_mapping[val] for val in y_values], dtype=float)
        else:
            y_processed = y_values.astype(float)

        return y_processed

    def _find_cut_points(
        self, x_sorted: np.ndarray[Any, Any], y_fitted: np.ndarray[Any, Any]
    ) -> list[int]:
        """Find cut points based on changes in fitted isotonic function.

        Identifies locations where the fitted function has significant changes
        that warrant creating new bin boundaries.

        Args:
            x_sorted: Sorted feature values
            y_fitted: Fitted isotonic regression values

        Returns:
            Indices of cut points in the sorted arrays
        """
        cut_indices = [0]  # Always start with first point

        if len(y_fitted) <= 1:
            return cut_indices

        # Calculate relative changes in fitted values
        y_range = np.max(y_fitted) - np.min(y_fitted)
        if y_range == 0:
            return cut_indices

        # Find points with significant changes
        for i in range(1, len(y_fitted)):
            # Check if there's a significant change from the last cut point
            last_cut_idx = cut_indices[-1]
            y_change = abs(y_fitted[i] - y_fitted[last_cut_idx])
            relative_change = y_change / y_range

            # Check if we have enough samples since last cut
            samples_since_cut = i - last_cut_idx

            if (
                relative_change >= self.min_change_threshold
                and samples_since_cut >= self.min_samples_per_bin
                and len(cut_indices) < self.max_bins
            ):
                cut_indices.append(i)

        return cut_indices

    def _create_bins_from_cuts(
        self,
        x_sorted: np.ndarray[Any, Any],
        y_fitted: np.ndarray[Any, Any],
        cut_indices: list[int],
        col_id: Any,
    ) -> tuple[list[float], list[float]]:
        """Create bin edges and representatives from cut points.

        Args:
            x_sorted: Sorted feature values
            y_fitted: Fitted isotonic regression values
            cut_indices: Indices of cut points
            col_id: Column identifier for error reporting

        Returns:
            Tuple of (bin_edges, bin_representatives)
        """
        if len(cut_indices) == 1:
            # Only one cut point - create single bin
            x_min, x_max = float(np.min(x_sorted)), float(np.max(x_sorted))
            if x_min == x_max:
                x_max = x_min + 1.0
            return [x_min, x_max], [(x_min + x_max) / 2]

        # Create bin edges
        bin_edges = []
        bin_representatives = []

        for i, cut_idx in enumerate(cut_indices):
            if i == 0:
                # First bin edge
                bin_edges.append(float(x_sorted[cut_idx]))
            else:
                # Find midpoint between consecutive cut points for bin boundary
                prev_cut_idx = cut_indices[i - 1]
                if cut_idx > prev_cut_idx:
                    midpoint = (x_sorted[prev_cut_idx] + x_sorted[cut_idx]) / 2
                    bin_edges.append(float(midpoint))

                    # Representative is the mean of feature values in this bin
                    bin_x_values = x_sorted[prev_cut_idx:cut_idx]
                    bin_representative = float(np.mean(bin_x_values))
                    bin_representatives.append(bin_representative)

        # Add final bin edge and representative
        bin_edges.append(float(x_sorted[-1]))
        if len(cut_indices) > 1:
            final_bin_x = x_sorted[cut_indices[-1] :]
            final_representative = float(np.mean(final_bin_x))
            bin_representatives.append(final_representative)
        else:
            bin_representatives.append(float(np.mean(x_sorted)))

        return bin_edges, bin_representatives
