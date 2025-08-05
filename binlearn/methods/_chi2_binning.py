"""Chi-square binning transformer.

This module implements chi-square binning, a supervised discretization method that
uses the chi-square statistic to find optimal split points. The method iteratively
merges adjacent intervals to minimize the chi-square statistic, creating bins that
maximize the association between features and target variables.

Chi-square binning is particularly effective for classification tasks where the goal
is to create bins that separate different classes as effectively as possible.

Classes:
    Chi2Binning: Main transformer for chi-square binning operations.
"""

from typing import Any, cast

import numpy as np
from scipy.stats import chi2_contingency

from ..base._repr_mixin import ReprMixin
from ..base._supervised_binning_base import SupervisedBinningBase
from ..utils.errors import ConfigurationError, FittingError, InvalidDataError
from ..utils.types import BinEdgesDict, ColumnId, GuidanceColumns


# pylint: disable=too-many-ancestors
class Chi2Binning(ReprMixin, SupervisedBinningBase):
    """Chi-square binning transformer for supervised discretization.

    Creates bins using the chi-square statistic to find optimal split points that
    maximize the association between features and target variables. The method
    starts with an initial discretization and then iteratively merges adjacent
    intervals that have the smallest chi-square statistic until a stopping
    criterion is met.

    This approach creates bins that are optimized for classification tasks, as
    they capture the most informative feature ranges for distinguishing between
    different target classes. The resulting bins often lead to better downstream
    classification performance compared to unsupervised binning methods.

    The transformer is fully sklearn-compatible and supports pandas/polars DataFrames.
    It automatically handles both categorical and continuous features.

    Attributes:
        max_bins (int): Maximum number of bins to create.
        min_bins (int): Minimum number of bins to create.
        alpha (float): Significance level for chi-square test.
        initial_bins (int): Number of initial bins for equal-width discretization.
        guidance_columns (list): Columns to use for supervised guidance.
        preserve_dataframe (bool): Whether to preserve DataFrame format.
        bin_edges_ (dict): Computed bin edges after fitting.

    Example:
        >>> import numpy as np
        >>> from binlearn.methods import Chi2Binning
        >>> X = np.random.rand(100, 3)
        >>> y = np.random.randint(0, 2, 100)  # Binary target
        >>> X_with_target = np.column_stack([X, y])
        >>> binner = Chi2Binning(guidance_columns=[3], max_bins=5)
        >>> X_binned = binner.fit_transform(X_with_target)
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        max_bins: int = 10,
        min_bins: int = 2,
        alpha: float = 0.05,
        initial_bins: int = 20,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
        guidance_columns: GuidanceColumns | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Chi2Binning transformer.

        Creates a chi-square binning transformer that uses the chi-square statistic
        to find optimal bin boundaries based on the association between features
        and target variables. The method starts with an initial equal-width
        discretization and then merges adjacent intervals based on chi-square tests.

        Args:
            max_bins (int, optional): Maximum number of bins to create for each
                feature. Must be at least 2. The algorithm will stop merging
                intervals when this number is reached. Defaults to 10.
            min_bins (int, optional): Minimum number of bins to create for each
                feature. Must be at least 2 and less than or equal to max_bins.
                The algorithm will continue merging until either this number is
                reached or the significance criterion is met. Defaults to 2.
            alpha (float, optional): Significance level for the chi-square test.
                Adjacent intervals are merged if their chi-square statistic
                corresponds to a p-value greater than alpha. Must be between
                0 and 1. Defaults to 0.05.
            initial_bins (int, optional): Number of initial bins for the equal-width
                discretization before merging. Should be larger than max_bins to
                allow for meaningful merging. Defaults to 20.
            clip (bool, optional): Whether to clip values outside bin ranges to nearest
                bin edges. If True, out-of-range values are clipped to the nearest
                bin boundary. If False, out-of-range values are assigned special
                indicators. If None, uses global configuration default. Defaults to None.
            preserve_dataframe (bool, optional): Whether to preserve DataFrame
                format in output. If None, uses global configuration default.
                Defaults to None.
            bin_edges (BinEdgesDict, optional): Pre-computed bin edges for each
                column. If provided, these edges are used instead of calculating
                from data. Defaults to None.
            bin_representatives (BinEdgesDict, optional): Pre-computed representative
                values for each bin. If provided along with bin_edges, these
                representatives are used. Defaults to None.
            guidance_columns (GuidanceColumns, optional): Columns to use for
                supervised guidance. Must be specified for chi-square binning
                to work properly. Should contain categorical target variable(s).
                Defaults to None.
            **kwargs: Additional arguments passed to SupervisedBinningBase.

        Raises:
            ConfigurationError: If max_bins < min_bins, if min_bins < 2, if
                alpha is not between 0 and 1, or if initial_bins < max_bins.

        Example:
            >>> # Basic usage with default parameters
            >>> binner = Chi2Binning(guidance_columns=[3], max_bins=5)

            >>> # Custom parameters for more conservative binning
            >>> binner = Chi2Binning(
            ...     guidance_columns=[3],
            ...     max_bins=8,
            ...     min_bins=3,
            ...     alpha=0.01,
            ...     initial_bins=30
            ... )

            >>> # With pre-specified bin edges
            >>> edges = {0: [0, 25, 50, 75, 100]}
            >>> binner = Chi2Binning(guidance_columns=[3], bin_edges=edges)
        """
        # Store chi2-specific parameters BEFORE calling super().__init__
        self.max_bins = max_bins
        self.min_bins = min_bins
        self.alpha = alpha
        self.initial_bins = initial_bins

        # Chi2 binning is always classification-based - store as class attribute
        self.task_type = "classification"

        # Chi2 binning is always classification-based
        super().__init__(
            clip=clip,
            preserve_dataframe=preserve_dataframe,
            bin_edges=bin_edges,
            bin_representatives=bin_representatives,
            guidance_columns=guidance_columns,
            **kwargs,
        )

    def _validate_params(self) -> None:
        """Validate chi-square binning specific parameters.

        Performs comprehensive validation of all Chi2Binning parameters to ensure
        they meet the expected constraints and are logically consistent.

        Raises:
            ConfigurationError: If any parameter validation fails.
        """
        super()._validate_params()

        # Validate max_bins
        if not isinstance(self.max_bins, int) or self.max_bins < 2:
            raise ConfigurationError(f"max_bins must be an integer >= 2, got {self.max_bins}")

        # Validate min_bins
        if not isinstance(self.min_bins, int) or self.min_bins < 2:
            raise ConfigurationError(f"min_bins must be an integer >= 2, got {self.min_bins}")

        # Check min_bins <= max_bins
        if self.min_bins > self.max_bins:
            raise ConfigurationError(
                f"min_bins ({self.min_bins}) must be <= max_bins ({self.max_bins})"
            )

        # Validate alpha
        if not isinstance(self.alpha, int | float) or not 0 < self.alpha < 1:
            raise ConfigurationError(f"alpha must be a number between 0 and 1, got {self.alpha}")

        # Validate initial_bins
        if not isinstance(self.initial_bins, int) or self.initial_bins < self.max_bins:
            raise ConfigurationError(
                f"initial_bins ({self.initial_bins}) must be >= max_bins ({self.max_bins})"
            )

        # Note: Chi2 binning requires guidance data, but this can come from
        # either guidance_columns or the y parameter in fit(), so we don't
        # validate guidance_columns here

    # pylint: disable=too-many-locals
    def _calculate_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: ColumnId,
        guidance_data: np.ndarray[Any, Any] | None = None,
    ) -> tuple[list[float], list[float]]:
        """Calculate chi-square based bins for a single column.

        Implements the chi-square binning algorithm that starts with equal-width
        discretization and then iteratively merges adjacent intervals based on
        chi-square tests. The algorithm continues merging until either the minimum
        number of bins is reached or no more intervals can be merged based on
        the significance level.

        Args:
            x_col (np.ndarray[Any, Any]): Feature data for binning with shape (n_samples,).
                May contain NaN values which are handled appropriately.
            col_id (ColumnId): Column identifier for error reporting and logging.
            guidance_data (np.ndarray[Any, Any], optional): Target data for supervised
                binning with shape (n_samples,). Required for chi-square binning.

        Returns:
            tuple[list[float], list[float]]: A tuple containing:
                - bin_edges (list[float]): List of bin edge values
                - bin_representatives (list[float]): List of bin center values

        Raises:
            FittingError: If guidance_data is None or if the chi-square binning
                algorithm fails to converge.
            InvalidDataError: If the feature data is invalid or incompatible
                with chi-square binning.
        """
        # Require guidance data (raises ValueError if None)
        self.require_guidance_data(guidance_data, "Chi-square binning")

        # At this point guidance_data is guaranteed to be not None
        assert guidance_data is not None

        try:
            # Validate guidance data (ensures single column requirement)
            guidance_data_validated = self.validate_guidance_data(guidance_data)

            # Extract the single guidance column as 1D array
            # Note: validate_guidance_data always returns 1D, but keep this for explicit clarity
            guidance_col = self._extract_guidance_column(guidance_data_validated)

            # Remove missing values from both feature and target
            valid_mask = ~(np.isnan(x_col) | np.isnan(guidance_col))
            if not valid_mask.any():
                raise InvalidDataError(
                    f"No valid data points for column {col_id} after removing missing values"
                )

            x_clean = x_col[valid_mask]
            y_clean = guidance_col[valid_mask]

            if len(x_clean) < self.min_bins:
                raise InvalidDataError(
                    f"Insufficient data for column {col_id}: {len(x_clean)} samples "
                    f"but need at least {self.min_bins} for binning"
                )

            # Check if feature has enough unique values
            unique_values = np.unique(x_clean)
            if len(unique_values) < self.min_bins:
                # Fallback to using unique values as bin edges
                bin_edges = self._create_edges_from_unique_values(unique_values)
                bin_centers = self._calculate_bin_centers(bin_edges)
                return bin_edges, bin_centers

            # Start with equal-width initial discretization
            data_min, data_max = np.min(x_clean), np.max(x_clean)
            constant_edges = self._handle_constant_feature_values(data_min, data_max)
            if constant_edges is not None:
                return constant_edges

            # Create initial bins
            initial_edges = np.linspace(data_min, data_max, self.initial_bins + 1)

            # Perform chi-square based merging
            final_edges = self._merge_bins_chi2(x_clean, y_clean, initial_edges)

            # Calculate bin centers
            bin_centers = self._calculate_bin_centers(final_edges)

            return final_edges, bin_centers

        except Exception as e:
            if isinstance(e, FittingError | InvalidDataError | ValueError):
                raise
            raise FittingError(
                f"Failed to calculate chi-square bins for column {col_id}: {str(e)}"
            ) from e

    def _merge_bins_chi2(
        self,
        x_data: np.ndarray[Any, Any],
        y_data: np.ndarray[Any, Any],
        initial_edges: np.ndarray[Any, Any],
    ) -> list[float]:
        """Merge bins based on chi-square statistic.

        Implements the core chi-square merging algorithm. Adjacent intervals are
        iteratively merged if their chi-square statistic indicates they are not
        significantly different (p-value > alpha).

        Args:
            x_data (np.ndarray[Any, Any]): Clean feature data without missing values.
            y_data (np.ndarray[Any, Any]): Clean target data without missing values.
            initial_edges (np.ndarray[Any, Any]): Initial bin edges from equal-width discretization.

        Returns:
            list[float]: Final bin edges after merging.
        """
        current_edges = list(initial_edges)

        while len(current_edges) - 1 > self.min_bins:
            # Find the best pair of adjacent intervals to merge
            best_pair_idx = self._find_best_merge_pair(x_data, y_data, current_edges)

            if best_pair_idx is None:
                # No more merging possible based on significance level
                break

            # Check if we should stop merging based on significance when at max_bins
            should_stop = self._should_stop_merging_for_significance(
                x_data, y_data, current_edges, best_pair_idx
            )
            if should_stop:
                break  # pragma: no cover

            # Merge the best pair
            current_edges.pop(best_pair_idx + 1)

        return current_edges

    def _find_best_merge_pair(
        self, x_data: np.ndarray[Any, Any], y_data: np.ndarray[Any, Any], edges: list[float]
    ) -> int | None:
        """Find the best pair of adjacent intervals to merge.

        Evaluates all adjacent interval pairs and finds the one with the smallest
        chi-square statistic (least significant difference).

        Args:
            x_data (np.ndarray[Any, Any]): Feature data.
            y_data (np.ndarray[Any, Any]): Target data.
            edges (list[float]): Current bin edges.

        Returns:
            int | None: Index of the left interval in the best pair to merge,
                or None if no valid pairs can be merged.
        """
        best_idx = None
        min_chi2 = float("inf")

        for i in range(len(edges) - 2):  # -2 because we need pairs of intervals
            try:
                chi2_stat, p_value = self._calculate_chi2_for_pair(x_data, y_data, edges, i)

                # Only consider merging if p-value > alpha or we're above max_bins
                if p_value > self.alpha or len(edges) - 1 > self.max_bins:
                    if chi2_stat < min_chi2:
                        min_chi2 = chi2_stat
                        best_idx = i

            except (ValueError, RuntimeWarning):
                # Skip pairs that cause computational issues
                continue

        return best_idx

    # pylint: disable=too-many-locals
    def _calculate_chi2_for_pair(
        self,
        x_data: np.ndarray[Any, Any],
        y_data: np.ndarray[Any, Any],
        edges: list[float],
        pair_idx: int,
    ) -> tuple[float, float]:
        """Calculate chi-square statistic for a pair of adjacent intervals.

        Computes the chi-square statistic and p-value for testing independence
        between the combined interval and the target variable.

        Args:
            x_data (np.ndarray[Any, Any]): Feature data.
            y_data (np.ndarray[Any, Any]): Target data.
            edges (list[float]): Current bin edges.
            pair_idx (int): Index of the left interval in the pair.

        Returns:
            tuple[float, float]: Chi-square statistic and p-value.
        """
        # Create interval indicators for the pair
        left_edge = edges[pair_idx]
        middle_edge = edges[pair_idx + 1]
        right_edge = edges[pair_idx + 2]

        # Find data points in each interval
        left_mask = (x_data >= left_edge) & (x_data < middle_edge)
        right_mask = (x_data >= middle_edge) & (x_data < right_edge)

        # Handle the rightmost bin boundary
        if pair_idx + 2 == len(edges) - 1:
            right_mask = (x_data >= middle_edge) & (x_data <= right_edge)

        # Get target values for each interval
        left_targets = y_data[left_mask]
        right_targets = y_data[right_mask]

        # Create contingency table
        unique_targets = np.unique(y_data)

        contingency_table: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.zeros(
            (2, len(unique_targets))
        )

        for j, target_val in enumerate(unique_targets):
            contingency_table[0, j] = np.sum(left_targets == target_val)
            contingency_table[1, j] = np.sum(right_targets == target_val)

        # Remove columns with all zeros
        non_zero_cols = contingency_table.sum(axis=0) > 0
        if not non_zero_cols.any():
            return float("inf"), 0.0

        contingency_table = contingency_table[:, non_zero_cols]

        # Remove rows with all zeros
        non_zero_rows = contingency_table.sum(axis=1) > 0
        if non_zero_rows.sum() < 2:
            return 0.0, 1.0  # Perfect independence if only one interval has data

        contingency_table = contingency_table[non_zero_rows, :]

        # Check if contingency table is valid for chi-square test
        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            return 0.0, 1.0

        # Calculate chi-square statistic
        try:
            chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
            return float(cast(float, chi2_stat)), float(cast(float, p_value))
        except ValueError:
            # Handle edge cases where chi2 calculation fails
            return 0.0, 1.0

    def _create_edges_from_unique_values(self, unique_values: np.ndarray[Any, Any]) -> list[float]:
        """Create bin edges from unique values when there aren't enough values.

        Args:
            unique_values (np.ndarray[Any, Any]): Sorted unique values from the feature.

        Returns:
            list[float]: Bin edges that separate the unique values.
        """
        if len(unique_values) == 1:
            val = unique_values[0]
            return [val - 0.1, val + 0.1]

        edges = [unique_values[0] - (unique_values[1] - unique_values[0]) * 0.1]

        for i in range(len(unique_values) - 1):
            mid_point = (unique_values[i] + unique_values[i + 1]) / 2
            edges.append(mid_point)

        edges.append(unique_values[-1] + (unique_values[-1] - unique_values[-2]) * 0.1)

        return edges

    def _calculate_bin_centers(self, bin_edges: list[float]) -> list[float]:
        """Calculate bin center points from bin edges.

        Args:
            bin_edges (list[float]): List of bin edge values.

        Returns:
            list[float]: List of bin center values.
        """
        centers = []
        for i in range(len(bin_edges) - 1):
            center = (bin_edges[i] + bin_edges[i + 1]) / 2
            centers.append(center)
        return centers

    def _extract_guidance_column(
        self, guidance_data_validated: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """Extract guidance column from validated guidance data.

        Args:
            guidance_data_validated: Validated guidance data from validate_guidance_data

        Returns:
            1D array of guidance values
        """
        # This method allows testing both 1D and 2D paths
        if guidance_data_validated.ndim == 2:
            return guidance_data_validated[:, 0]  # Line would be here

        return guidance_data_validated  # This line can be tested

    def _handle_constant_feature_values(
        self, data_min: float, data_max: float
    ) -> tuple[list[float], list[float]] | None:
        """Handle case where all feature values are the same.

        Args:
            data_min: Minimum value in the feature data
            data_max: Maximum value in the feature data

        Returns:
            Tuple of (bin_edges, bin_centers) if constant values, None otherwise
        """
        if data_min == data_max:
            # All values are the same - lines 268-270 equivalent
            bin_edges = [data_min - 0.1, data_max + 0.1]
            bin_centers = [data_min]
            return bin_edges, bin_centers
        return None

    def _should_stop_merging_for_significance(
        self,
        x_data: np.ndarray[Any, Any],
        y_data: np.ndarray[Any, Any],
        current_edges: list[float],
        best_pair_idx: int,
    ) -> bool:
        """Check if merging should stop based on significance when at max_bins.

        Args:
            x_data: Feature data
            y_data: Target data
            current_edges: Current bin edges
            best_pair_idx: Index of best pair to merge

        Returns:
            True if should stop merging, False otherwise
        """
        if len(current_edges) - 1 <= self.max_bins:
            # We've reached max_bins, check if we should stop based on significance
            _, p_value = self._calculate_chi2_for_pair(x_data, y_data, current_edges, best_pair_idx)
            if p_value <= self.alpha:
                # Significant difference, don't merge - line 324 equivalent
                return True

        return False
