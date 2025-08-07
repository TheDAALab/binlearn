"""
Clean Chi-square binning implementation.

This module provides Chi2Binning that inherits from SupervisedBinningBase.
Uses chi-square statistic to find optimal bin boundaries for classification tasks.
"""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
from scipy.stats import chi2_contingency

from ..config import get_config, apply_config_defaults
from ..utils.types import BinEdgesDict
from ..utils.errors import ConfigurationError, FittingError, DataQualityWarning
from ..base._supervised_binning_base import SupervisedBinningBase


class Chi2Binning(SupervisedBinningBase):
    """Chi-square binning implementation.

    Creates bins using the chi-square statistic to find optimal split points that
    maximize the association between numeric features and target variables. Only supports numeric data.
    The method starts with an initial discretization and then iteratively merges adjacent
    intervals that have the smallest chi-square statistic until a stopping criterion is met.

    This approach creates bins that are optimized for classification tasks.
    This implementation follows the clean  architecture with straight inheritance,
    dynamic column resolution, and parameter reconstruction capabilities.
    """

    def __init__(
        self,
        max_bins: int | None = None,
        min_bins: int | None = None,
        alpha: float | None = None,
        initial_bins: int | None = None,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        guidance_columns: Any = None,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
        class_: str | None = None,  # For reconstruction compatibility
        module_: str | None = None,  # For reconstruction compatibility
    ):
        """Initialize Chi-square binning."""
        # Prepare user parameters for config integration (exclude never-configurable params)
        user_params = {
            "max_bins": max_bins,
            "min_bins": min_bins,
            "alpha": alpha,
            "initial_bins": initial_bins,
            "clip": clip,
            "preserve_dataframe": preserve_dataframe,
        }
        # Remove None values to allow config defaults to take effect
        user_params = {k: v for k, v in user_params.items() if v is not None}

        # Apply configuration defaults for chi2 method
        params = apply_config_defaults("chi2", user_params)

        # Store chi-square specific parameters with config defaults
        self.max_bins = params.get("max_bins", max_bins if max_bins is not None else 10)
        self.min_bins = params.get("min_bins", min_bins if min_bins is not None else 2)
        self.alpha = params.get("alpha", alpha if alpha is not None else 0.05)
        self.initial_bins = params.get(
            "initial_bins", initial_bins if initial_bins is not None else 20
        )

        # Initialize parent with resolved config parameters (no fit_jointly for supervised)
        # Note: guidance_columns, bin_edges, bin_representatives are never set from config
        SupervisedBinningBase.__init__(
            self,
            clip=params.get("clip"),
            preserve_dataframe=params.get("preserve_dataframe"),
            guidance_columns=guidance_columns,
            bin_edges=bin_edges,
            bin_representatives=bin_representatives,
        )

    def _validate_params(self) -> None:
        """Validate Chi-square binning specific parameters."""
        # Call parent validation
        SupervisedBinningBase._validate_params(self)

        # Validate max_bins
        if not isinstance(self.max_bins, int) or self.max_bins < 1:
            raise ValueError("max_bins must be a positive integer")

        # Validate min_bins
        if not isinstance(self.min_bins, int) or self.min_bins < 1:
            raise ValueError("min_bins must be a positive integer")

        # Validate bin constraints
        if self.min_bins > self.max_bins:
            raise ValueError("min_bins must be <= max_bins")

        # Validate alpha
        if not isinstance(self.alpha, (int, float)) or not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be a float between 0 and 1")

        # Validate initial_bins
        if not isinstance(self.initial_bins, int) or self.initial_bins < self.max_bins:
            raise ValueError("initial_bins must be an integer >= max_bins")

    def _calculate_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        guidance_data: np.ndarray[Any, Any] | None = None,
    ) -> tuple[list[float], list[float]]:
        """Calculate bin edges and representatives using chi-square optimization.

        Chi2 binning is a supervised method and requires guidance data.

        Args:
            x_col: Clean feature data (no missing values)
            col_id: Column identifier
            guidance_data: Target data with shape (n_samples, 1). Required.

        Returns:
            Tuple of (bin_edges, bin_representatives)

        Raises:
            ValueError: If guidance_data is None (supervised method requires targets)
        """
        if guidance_data is None:
            raise ValueError(
                "Chi2 binning is a supervised method and requires guidance data (targets)"
            )

        # Extract the single target column (guaranteed to have shape (n_samples, 1) by SupervisedBinningBase)
        y_col = guidance_data[:, 0]

        return self._calculate_chi2_bins(x_col, y_col, col_id)

    def _calculate_chi2_bins(
        self,
        x_col: np.ndarray[Any, Any],
        y_col: np.ndarray[Any, Any],
        col_id: Any,
    ) -> tuple[list[float], list[float]]:
        """Calculate chi-square optimized bin edges and representatives.

        Args:
            x_col: Preprocessed feature data (already handled by base class)
            y_col: Target data - 1D array (may have been filtered by SupervisedBinningBase)
            col_id: Column identifier

        Returns:
            Tuple of (bin_edges, bin_representatives)

        Raises:
            FittingError: If data is insufficient for chi-square binning
        """
        # The feature data (x_col) is already preprocessed by IntervalBinningBase
        # The target data (y_col) has been handled by SupervisedBinningBase
        # We can now work directly with the data

        if len(x_col) < 2:
            raise FittingError(
                f"Column {col_id} has too few data points ({len(x_col)}). "
                "Chi2 binning requires at least 2 data points."
            )

        # Get unique target classes
        unique_classes = np.unique(y_col)
        if len(unique_classes) < 2:
            raise FittingError(
                f"Column {col_id} target has insufficient class diversity ({len(unique_classes)} classes). "
                "Chi2 binning requires at least 2 target classes."
            )

        # Step 1: Create initial equal-width binning
        data_min = float(np.min(x_col))
        data_max = float(np.max(x_col))

        initial_edges = np.linspace(data_min, data_max, self.initial_bins + 1)

        # Step 2: Create contingency tables for initial bins
        bin_indices = np.digitize(x_col, initial_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(initial_edges) - 2)

        # Build initial contingency table
        intervals = self._build_intervals(bin_indices, y_col, initial_edges, unique_classes)

        if not intervals:
            raise FittingError(
                f"Failed to create initial intervals for column {col_id}. "
                "Data distribution may be unsuitable for chi2 binning."
            )

        # Step 3: Iteratively merge intervals with smallest chi-square
        final_intervals = self._merge_intervals(intervals, unique_classes)

        # Step 4: Extract edges and representatives
        edges = [final_intervals[0]["min"]]
        representatives = []

        for interval in final_intervals:
            edges.append(interval["max"])
            # Representative is the midpoint of the interval
            representatives.append((interval["min"] + interval["max"]) / 2)

        return edges, representatives

    def _build_intervals(
        self,
        bin_indices: np.ndarray[Any, Any],
        y_col: np.ndarray[Any, Any],
        initial_edges: np.ndarray[Any, Any],
        unique_classes: np.ndarray[Any, Any],
    ) -> list[dict[str, Any]]:
        """Build initial intervals with contingency information."""
        intervals = []

        for i in range(len(initial_edges) - 1):
            mask = bin_indices == i
            if not np.any(mask):
                continue  # Skip empty intervals

            # Count occurrences of each class in this interval
            y_interval = y_col[mask]
            class_counts = {}
            for cls in unique_classes:
                class_counts[cls] = int(np.sum(y_interval == cls))

            interval = {
                "min": float(initial_edges[i]),
                "max": float(initial_edges[i + 1]),
                "class_counts": class_counts,
                "total_count": int(np.sum(mask)),
            }

            if interval["total_count"] > 0:  # Only add non-empty intervals
                intervals.append(interval)

        return intervals

    def _merge_intervals(
        self,
        intervals: list[dict[str, Any]],
        unique_classes: np.ndarray[Any, Any],
    ) -> list[dict[str, Any]]:
        """Iteratively merge intervals to optimize chi-square statistic."""
        current_intervals = intervals.copy()

        while len(current_intervals) > self.max_bins:
            # Find the pair of adjacent intervals with smallest chi-square
            min_chi2 = float("inf")
            merge_idx = -1

            for i in range(len(current_intervals) - 1):
                chi2_stat = self._calculate_chi2_for_merge(
                    current_intervals[i], current_intervals[i + 1], unique_classes
                )
                if chi2_stat < min_chi2:
                    min_chi2 = chi2_stat
                    merge_idx = i

            # Check if we should stop merging based on significance
            if len(current_intervals) <= self.min_bins:
                break

            # If chi-square is significant and we have more than min_bins, stop
            if min_chi2 > self._get_chi2_critical_value(len(unique_classes) - 1):
                if len(current_intervals) >= self.min_bins:
                    break

            # Merge the intervals
            if merge_idx >= 0:
                merged_interval = self._merge_two_intervals(
                    current_intervals[merge_idx], current_intervals[merge_idx + 1]
                )
                current_intervals = (
                    current_intervals[:merge_idx]
                    + [merged_interval]
                    + current_intervals[merge_idx + 2 :]
                )

        return current_intervals

    def _calculate_chi2_for_merge(
        self,
        interval1: dict[str, Any],
        interval2: dict[str, Any],
        unique_classes: np.ndarray[Any, Any],
    ) -> float:
        """Calculate chi-square statistic for merging two intervals."""
        try:
            # Build contingency table for the two intervals
            contingency_table = []

            for cls in unique_classes:
                row = [interval1["class_counts"].get(cls, 0), interval2["class_counts"].get(cls, 0)]
                contingency_table.append(row)

            contingency_table = np.array(contingency_table)

            # Remove empty rows/columns
            row_sums = contingency_table.sum(axis=1)
            col_sums = contingency_table.sum(axis=0)

            valid_rows = row_sums > 0
            valid_cols = col_sums > 0

            if not np.any(valid_rows) or not np.any(valid_cols):
                return 0.0

            contingency_table = contingency_table[valid_rows][:, valid_cols]

            if (
                contingency_table.size == 0
                or contingency_table.shape[0] < 2
                or contingency_table.shape[1] < 2
            ):
                return 0.0

            # Calculate chi-square statistic
            try:
                chi2_stat, _, _, _ = chi2_contingency(contingency_table)
                return float(chi2_stat) if isinstance(chi2_stat, (int, float, np.number)) else 0.0
            except Exception:
                return 0.0

        except (ValueError, ZeroDivisionError):
            return 0.0

    def _merge_two_intervals(
        self,
        interval1: dict[str, Any],
        interval2: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge two adjacent intervals."""
        merged_class_counts = {}

        # Combine class counts
        all_classes = set(interval1["class_counts"].keys()) | set(interval2["class_counts"].keys())
        for cls in all_classes:
            merged_class_counts[cls] = interval1["class_counts"].get(cls, 0) + interval2[
                "class_counts"
            ].get(cls, 0)

        return {
            "min": interval1["min"],
            "max": interval2["max"],
            "class_counts": merged_class_counts,
            "total_count": interval1["total_count"] + interval2["total_count"],
        }

    def _get_chi2_critical_value(self, dof: int) -> float:
        """Get critical chi-square value for given degrees of freedom and alpha."""
        # Approximation for common alpha values
        # This could be made more precise with scipy.stats.chi2.ppf
        if self.alpha >= 0.1:
            return 2.706  # Very lenient
        elif self.alpha >= 0.05:
            return 3.841 if dof == 1 else 5.991  # Standard
        else:
            return 6.635 if dof == 1 else 9.210  # Strict
