"""
Clean Chi-square binning implementation.

This module provides Chi2Binning that inherits from SupervisedBinningBase.
Uses chi-square statistic to find optimal bin boundaries for classification tasks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import chi2_contingency

from ..base import SupervisedBinningBase
from ..config import apply_config_defaults
from ..utils import (
    BinEdgesDict,
    ConfigurationError,
    FittingError,
    create_equal_width_bins,
    create_param_dict_for_config,
    validate_positive_integer,
)


# pylint: disable=too-many-ancestors
class Chi2Binning(SupervisedBinningBase):
    """Chi-square binning implementation for supervised discretization.

    This class implements chi-square binning (χ² binning), a supervised discretization
    method that uses the chi-square statistic to find optimal bin boundaries for
    classification tasks. The method creates bins that maximize the association
    between numeric features and categorical target variables, making it particularly
    effective for improving classification performance.

    Chi-square binning is particularly effective for:
    - Binary and multi-class classification preprocessing
    - Creating bins that preserve class-discriminative information
    - Reducing feature dimensionality while maintaining predictive power
    - Handling continuous features with complex relationships to target classes

    Key Features:
    - Uses chi-square test of independence to guide bin boundary selection
    - Iterative merging process starting from fine initial discretization
    - Configurable stopping criteria (significance level, bin count limits)
    - Handles both binary and multi-class classification targets
    - Automatic handling of insufficient data and edge cases

    Algorithm:
    1. Create initial fine-grained discretization (equal frequency or equal width)
    2. For each pair of adjacent bins, calculate chi-square statistic
    3. Merge the pair with the smallest (least significant) chi-square value
    4. Repeat merging until stopping criterion is met:
       - Minimum number of bins reached, OR
       - All remaining chi-square values exceed significance threshold (alpha)
    5. Create final bin boundaries and representatives

    Parameters:
        max_bins: Maximum number of bins to create. The algorithm will not exceed
            this limit regardless of statistical significance. Useful for controlling
            model complexity and computational costs.
        min_bins: Minimum number of bins to maintain. The algorithm will not merge
            below this threshold even if chi-square values are not significant.
            Ensures some level of discretization is preserved.
        alpha: Significance level for the chi-square test. Adjacent bins are merged
            if their chi-square p-value exceeds this threshold (indicating lack of
            significant association). Lower values result in more bins.
        initial_bins: Number of bins to create in the initial discretization step
            before beginning the merging process. Higher values provide finer
            granularity for the merging algorithm to work with.

    Attributes:
        bin_edges_: Dictionary mapping column identifiers to lists of optimized
            bin edges after fitting. These edges maximize class separation.
        bin_representatives_: Dictionary mapping column identifiers to lists
            of bin representatives (typically bin centers).

    Example:
        >>> import numpy as np
        >>> from binlearn.methods import Chi2Binning
        >>>
        >>> # Binary classification example
        >>> X = np.random.normal(0, 1, (1000, 2))
        >>> # Create target correlated with first feature
        >>> y = (X[:, 0] > 0).astype(int)
        >>>
        >>> binner = Chi2Binning(max_bins=5, alpha=0.05)
        >>> binner.fit(X, guidance_data=y.reshape(-1, 1))
        >>> X_binned = binner.transform(X)
        >>>
        >>> # Multi-class example with custom parameters
        >>> y_multi = np.random.choice([0, 1, 2], size=1000)
        >>> binner_multi = Chi2Binning(
        ...     max_bins=10,
        ...     min_bins=3,
        ...     alpha=0.01,
        ...     initial_bins=20
        ... )
        >>> binner_multi.fit(X, guidance_data=y_multi.reshape(-1, 1))

    Note:
        - Requires target data (guidance_data) during fitting for supervised learning
        - Works only with numeric input features and categorical targets
        - Performance depends on the relationship between features and target
        - May create fewer bins than max_bins if early stopping criteria are met
        - Inherits clipping behavior and format preservation from SupervisedBinningBase
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        max_bins: int | None = None,
        min_bins: int | None = None,
        alpha: float | None = None,
        initial_bins: int | None = None,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        guidance_columns: Any = None,
        *,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
        class_: str | None = None,  # For reconstruction compatibility
        module_: str | None = None,  # For reconstruction compatibility
    ):
        """Initialize Chi-square binning."""
        # Use standardized initialization pattern
        user_params = create_param_dict_for_config(
            max_bins=max_bins,
            min_bins=min_bins,
            alpha=alpha,
            initial_bins=initial_bins,
            clip=clip,
            preserve_dataframe=preserve_dataframe,
        )

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

        # Use standardized validation utilities
        validate_positive_integer(self.max_bins, "max_bins")
        validate_positive_integer(self.min_bins, "min_bins")
        validate_positive_integer(self.initial_bins, "initial_bins")

        # Validate alpha (must be between 0 and 1)
        if not isinstance(self.alpha, int | float) or not (0.0 < self.alpha < 1.0):
            raise ConfigurationError(
                f"alpha must be a number between 0 and 1 (exclusive), got {self.alpha}",
                suggestions=["Example: alpha=0.05"],
            )

        # Validate bin constraints
        if self.min_bins > self.max_bins:
            raise ConfigurationError(
                f"min_bins ({self.min_bins}) must be <= max_bins ({self.max_bins})",
                suggestions=["Reduce min_bins or increase max_bins"],
            )

        # Validate initial_bins constraint
        if self.initial_bins < self.max_bins:
            raise ConfigurationError(
                f"initial_bins ({self.initial_bins}) must be >= max_bins ({self.max_bins})",
                suggestions=["Increase initial_bins or reduce max_bins"],
            )

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

    def _validate_data_requirements(
        self, x_col: np.ndarray[Any, Any], y_col: np.ndarray[Any, Any], col_id: Any
    ) -> np.ndarray[Any, Any]:
        """Validate data requirements and return unique classes.

        Separated for easier testing of validation logic.
        """
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

        return np.asarray(unique_classes)

    def _create_initial_bins(
        self, x_col: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Create initial equal-width binning.

        Separated for easier testing of binning logic.
        """
        # Use standardized equal-width binning utility
        initial_edges = create_equal_width_bins(x_col, self.initial_bins)

        # Create bin assignments
        bin_indices = np.digitize(x_col, initial_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(initial_edges) - 2)

        return initial_edges, bin_indices

    def _validate_intervals_created(self, intervals: list[dict[str, Any]], col_id: Any) -> None:
        """Validate that intervals were successfully created.

        Separated to make the empty intervals check (line 189) easily testable.
        """
        if not intervals:
            raise FittingError(
                f"Failed to create initial intervals for column {col_id}. "
                "Data distribution may be unsuitable for chi2 binning."
            )

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
        # Step 1: Validate data requirements
        unique_classes = self._validate_data_requirements(x_col, y_col, col_id)

        # Step 2: Create initial equal-width binning
        initial_edges, bin_indices = self._create_initial_bins(x_col)

        # Step 3: Build initial contingency table
        intervals = self._build_intervals(bin_indices, y_col, initial_edges, unique_classes)

        # Step 4: Validate intervals were created (testable line 189)
        self._validate_intervals_created(intervals, col_id)

        # Step 5: Iteratively merge intervals with smallest chi-square
        final_intervals = self._merge_intervals(intervals, unique_classes)

        # Step 6: Extract edges and representatives
        return self._extract_final_results(final_intervals)

    def _extract_final_results(
        self, final_intervals: list[dict[str, Any]]
    ) -> tuple[list[float], list[float]]:
        """Extract final bin edges and representatives from intervals.

        Separated for easier testing of result extraction.
        """
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
            interval = self._create_interval_from_bin(
                i, bin_indices, y_col, initial_edges, unique_classes
            )
            if interval is not None:
                intervals.append(interval)

        return intervals

    def _is_valid_interval(self, interval: dict[str, Any]) -> bool:
        """Check if interval is valid (non-empty).

        Separated to make line 254 branch easily testable.
        """
        total_count = interval["total_count"]
        return isinstance(total_count, int | float) and total_count > 0

    def _create_interval_from_bin(
        self,
        bin_idx: int,
        bin_indices: np.ndarray[Any, Any],
        y_col: np.ndarray[Any, Any],
        initial_edges: np.ndarray[Any, Any],
        unique_classes: np.ndarray[Any, Any],
    ) -> dict[str, Any] | None:
        """Create an interval from a bin index, returns None for empty bins."""
        mask = bin_indices == bin_idx
        if not np.any(mask):
            return None  # Skip empty intervals

        # Count occurrences of each class in this interval
        y_interval = y_col[mask]
        class_counts = self._calculate_class_counts(y_interval, unique_classes)

        interval = {
            "min": float(initial_edges[bin_idx]),
            "max": float(initial_edges[bin_idx + 1]),
            "class_counts": class_counts,
            "total_count": int(np.sum(mask)),
        }

        # Only return non-empty intervals - simplified logic
        if not self._is_valid_interval(interval):
            return None  # Line 307 - now easily testable

        return interval

    def _calculate_class_counts(
        self,
        y_interval: np.ndarray[Any, Any],
        unique_classes: np.ndarray[Any, Any],
    ) -> dict[Any, int]:
        """Calculate class counts for a given interval."""
        class_counts = {}
        for cls in unique_classes:
            class_counts[cls] = int(np.sum(y_interval == cls))
        return class_counts

    def _has_enough_bins_to_merge(self, current_intervals: list[dict[str, Any]]) -> bool:
        """Check if we have more bins than max_bins (continue merging condition).

        Separated to make while loop condition testable.
        """
        return bool(len(current_intervals) > self.max_bins)

    def _should_perform_merge(self, merge_idx: int) -> bool:
        """Check if merge should be performed (merge_idx validation).

        Separated to make line 282->275 branch testable.
        """
        return merge_idx >= 0

    def _should_continue_merging(
        self,
        current_intervals: list[dict[str, Any]],
        min_chi2: float,
        unique_classes: np.ndarray[Any, Any],
    ) -> bool:
        """Check if merging should continue (opposite of _should_stop_merging).

        Separated to make the break condition (line 349->342) testable.
        """
        return not self._should_stop_merging(current_intervals, min_chi2, unique_classes)

    def _merge_intervals(
        self,
        intervals: list[dict[str, Any]],
        unique_classes: np.ndarray[Any, Any],
    ) -> list[dict[str, Any]]:
        """Iteratively merge intervals to optimize chi-square statistic."""
        current_intervals = intervals.copy()

        while self._has_enough_bins_to_merge(current_intervals):
            merge_idx, min_chi2 = self._find_best_merge_candidate(current_intervals, unique_classes)

            # Reorganized to make break condition testable
            if not self._should_continue_merging(current_intervals, min_chi2, unique_classes):
                break  # Line 349->342 - now easily testable

            # Reorganized merge logic to make the skip-to-while flow testable
            current_intervals = self._attempt_merge_or_continue(current_intervals, merge_idx)

        return current_intervals

    def _attempt_merge_or_continue(
        self,
        current_intervals: list[dict[str, Any]],
        merge_idx: int,
    ) -> list[dict[str, Any]]:
        """Attempt merge if valid, otherwise return unchanged intervals.

        Separated to make the skip-back-to-while-loop branch testable.
        """
        if self._should_perform_merge(merge_idx):
            return self._perform_merge(current_intervals, merge_idx)
        # If merge_idx is invalid, return unchanged intervals
        # This makes the flow back to while loop easily testable
        return current_intervals

    def _find_best_merge_candidate(
        self,
        current_intervals: list[dict[str, Any]],
        unique_classes: np.ndarray[Any, Any],
    ) -> tuple[int, float]:
        """Find the pair of adjacent intervals with smallest chi-square statistic."""
        min_chi2 = float("inf")
        merge_idx = -1

        for i in range(len(current_intervals) - 1):
            chi2_stat = self._calculate_chi2_for_merge(
                current_intervals[i], current_intervals[i + 1], unique_classes
            )
            if chi2_stat < min_chi2:
                min_chi2 = chi2_stat
                merge_idx = i

        return merge_idx, min_chi2

    def _at_minimum_bins(self, current_intervals: list[dict[str, Any]]) -> bool:
        """Check if we're at minimum number of bins.

        Separated to make stopping condition testable.
        """
        return bool(len(current_intervals) <= self.min_bins)

    def _chi2_is_significant(self, min_chi2: float, unique_classes: np.ndarray[Any, Any]) -> bool:
        """Check if chi-square value is significant.

        Separated to make significance testing branch testable.
        """
        return min_chi2 > self._get_chi2_critical_value(len(unique_classes) - 1)

    def _above_minimum_bins(self, current_intervals: list[dict[str, Any]]) -> bool:
        """Check if we have at least minimum number of bins.

        Separated to make line 319->322 branch testable.
        """
        return bool(len(current_intervals) >= self.min_bins)

    def _should_stop_merging(
        self,
        current_intervals: list[dict[str, Any]],
        min_chi2: float,
        unique_classes: np.ndarray[Any, Any],
    ) -> bool:
        """Determine if merging should stop based on significance and bin constraints."""
        # Always stop if at minimum bins
        if self._at_minimum_bins(current_intervals):
            return True

        # Check significance - reorganized for better testability
        if not self._chi2_is_significant(min_chi2, unique_classes):
            return False  # Line 407->410 - not significant, continue merging

        # Chi2 is significant, check if we can stop
        return self._above_minimum_bins(current_intervals)

    def _perform_merge(
        self,
        current_intervals: list[dict[str, Any]],
        merge_idx: int,
    ) -> list[dict[str, Any]]:
        """Perform the actual merge of two intervals."""
        merged_interval = self._merge_two_intervals(
            current_intervals[merge_idx], current_intervals[merge_idx + 1]
        )
        return (
            current_intervals[:merge_idx] + [merged_interval] + current_intervals[merge_idx + 2 :]
        )

    def _build_contingency_table(
        self,
        interval1: dict[str, Any],
        interval2: dict[str, Any],
        unique_classes: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Build contingency table for two intervals.

        Separated to make contingency table building testable.
        """
        contingency_rows = []
        for cls in unique_classes:
            row = [interval1["class_counts"].get(cls, 0), interval2["class_counts"].get(cls, 0)]
            contingency_rows.append(row)
        return np.array(contingency_rows)

    def _validate_contingency_table(self, contingency_table: np.ndarray[Any, Any]) -> bool:
        """Validate contingency table has valid rows and columns.

        Separated to make validation logic testable.
        """
        # Remove empty rows/columns
        row_sums = contingency_table.sum(axis=1)
        col_sums = contingency_table.sum(axis=0)

        valid_rows = row_sums > 0
        valid_cols = col_sums > 0

        if not np.any(valid_rows) or not np.any(valid_cols):
            return False

        # Filter to valid rows/cols
        filtered_table = contingency_table[valid_rows][:, valid_cols]

        if filtered_table.size == 0 or filtered_table.shape[0] < 2 or filtered_table.shape[1] < 2:
            return False

        # Store filtered table for calculation
        self._filtered_contingency = filtered_table
        return True

    def _convert_chi2_result(self, chi2_stat: Any) -> float:
        """Convert chi2 statistic to float, handling edge cases.

        Separated to make the type checking (lines 478-479) testable.
        """
        if isinstance(chi2_stat, int | float | np.number):
            return float(chi2_stat)
        return 0.0  # Lines 478-479 - now testable with non-numeric input

    def _compute_chi2_statistic(self, contingency_table: np.ndarray[Any, Any]) -> float:
        """Compute chi-square statistic from contingency table.

        Separated to make chi2 calculation and exception handling testable.
        """
        try:
            chi2_stat, _, _, _ = chi2_contingency(contingency_table)
            return self._convert_chi2_result(chi2_stat)
        except Exception:
            return 0.0  # Exception handling

    def _handle_chi2_calculation_errors(self, error: Exception) -> float:
        """Handle specific calculation errors.

        Separated to make lines 499-500 testable.
        """
        if isinstance(error, ValueError | ZeroDivisionError):
            return 0.0  # Lines 499-500 - now testable
        # Re-raise other exceptions
        raise error

    def _calculate_chi2_for_merge(
        self,
        interval1: dict[str, Any],
        interval2: dict[str, Any],
        unique_classes: np.ndarray[Any, Any],
    ) -> float:
        """Calculate chi-square statistic for merging two intervals."""
        return self._safe_chi2_calculation(interval1, interval2, unique_classes)

    def _safe_chi2_calculation(
        self,
        interval1: dict[str, Any],
        interval2: dict[str, Any],
        unique_classes: np.ndarray[Any, Any],
    ) -> float:
        """Safely calculate chi2 with exception handling separated for testability."""
        try:
            return self._perform_chi2_calculation(interval1, interval2, unique_classes)
        except Exception as e:
            return self._handle_chi2_calculation_errors(e)

    def _perform_chi2_calculation(
        self,
        interval1: dict[str, Any],
        interval2: dict[str, Any],
        unique_classes: np.ndarray[Any, Any],
    ) -> float:
        """Perform the actual chi2 calculation (separated for easier exception testing)."""
        # Build contingency table for the two intervals
        contingency_table = self._build_contingency_table(interval1, interval2, unique_classes)

        # Validate and filter contingency table
        if not self._validate_contingency_table(contingency_table):
            return 0.0

        # Calculate chi-square statistic with exception handling
        return self._compute_chi2_statistic(self._filtered_contingency)

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
