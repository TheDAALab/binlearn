"""Isotonic regression-based monotonic binning transformer.

This module implements monotonic binning using isotonic regression, where bin boundaries
are determined by finding optimal cut points that preserve monotonic relationships
between features and targets. This method is particularly useful when you know there
should be a monotonic relationship between the feature and target.

Classes:
    IsotonicBinning: Main transformer for isotonic regression-based binning operations.
"""

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ..base._repr_mixin import ReprMixin
from ..base._supervised_binning_base import SupervisedBinningBase
from ..utils.errors import ConfigurationError, FittingError


# pylint: disable=too-many-ancestors
class IsotonicBinning(ReprMixin, SupervisedBinningBase):
    """Isotonic regression-based monotonic binning transformer.

    Creates bins using isotonic regression to find optimal cut points that preserve
    monotonic relationships between features and targets. The transformer fits an
    isotonic (non-decreasing) function to the data and identifies significant changes
    in this function to determine bin boundaries.

    This method is particularly effective when:
    - There's a known monotonic relationship between feature and target
    - You want bins that respect this monotonic ordering
    - Traditional tree-based methods might create non-monotonic splits

    The transformer supports both classification and regression tasks and automatically
    handles the conversion of target values for isotonic regression fitting.

    Attributes:
        max_bins (int): Maximum number of bins to create per feature.
        min_samples_per_bin (int): Minimum samples required per bin.
        increasing (bool): Whether to enforce increasing monotonicity.
        y_min (float): Lower bound for target values in isotonic regression.
        y_max (float): Upper bound for target values in isotonic regression.
        min_change_threshold (float): Minimum change in fitted values to create new bin.
        preserve_dataframe (bool): Whether to preserve DataFrame format.
        _isotonic_models (dict): Fitted isotonic regression models per feature.

    Example:
        >>> import numpy as np
        >>> from binlearn.methods import IsotonicBinning
        >>> X = np.random.rand(100, 3)
        >>> y = X[:, 0] + 0.1 * np.random.randn(100)  # Monotonic relationship
        >>> binner = IsotonicBinning(max_bins=5)
        >>> X_binned = binner.fit_transform(X, guidance_data=y)
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        max_bins: int = 10,
        min_samples_per_bin: int = 5,
        increasing: bool = True,
        y_min: float | None = None,
        y_max: float | None = None,
        min_change_threshold: float = 0.01,
        clip: bool | None = None,
        preserve_dataframe: bool = False,
        guidance_columns: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize IsotonicBinning transformer.

        Creates an isotonic regression-based binning transformer that finds optimal
        cut points preserving monotonic relationships between features and targets.

        Args:
            max_bins (int, optional): Maximum number of bins to create for each feature.
                The actual number of bins may be smaller if the data doesn't support
                that many distinct bins. Defaults to 10.
            min_samples_per_bin (int, optional): Minimum number of samples required
                per bin. Bins with fewer samples will be merged with adjacent bins.
                Defaults to 5.
            increasing (bool, optional): Whether to enforce increasing monotonicity
                (True) or decreasing monotonicity (False). If True, higher feature
                values should correspond to higher target values. Defaults to True.
            y_min (Optional[float], optional): Lower bound for target values used
                in isotonic regression. If None, uses the minimum target value.
                Defaults to None.
            y_max (Optional[float], optional): Upper bound for target values used
                in isotonic regression. If None, uses the maximum target value.
                Defaults to None.
            min_change_threshold (float, optional): Minimum relative change in fitted
                isotonic values required to create a new bin boundary. Smaller values
                create more bins, larger values create fewer bins. Defaults to 0.01.
            clip (bool, optional): Whether to clip values outside bin ranges to nearest
                bin edges. If True, out-of-range values are clipped to the nearest
                bin boundary. If False, out-of-range values are assigned special
                indicators. If None, uses global configuration default. Defaults to None.
            preserve_dataframe (bool, optional): Whether to preserve pandas/polars
                DataFrame format in the output. Defaults to False.
            guidance_columns (Any, optional): Column identifier(s) to use as guidance/target
                for supervised binning. Can be a single column identifier or list.
                Defaults to None.
            **kwargs: Additional keyword arguments passed to the parent SupervisedBinningBase.

        Raises:
            ConfigurationError: If parameters are invalid (e.g., max_bins < 1,
                min_samples_per_bin < 1, invalid y_min/y_max bounds).

        Example:
            >>> # Basic usage with default parameters
            >>> binner = IsotonicBinning(max_bins=5)

            >>> # For decreasing monotonic relationships
            >>> binner = IsotonicBinning(increasing=False, max_bins=8)

            >>> # With custom bounds and change threshold
            >>> binner = IsotonicBinning(
            ...     y_min=0.0, y_max=1.0,
            ...     min_change_threshold=0.05,
            ...     min_samples_per_bin=10
            ... )
        """
        # Store isotonic-specific parameters
        self.max_bins = max_bins
        self.min_samples_per_bin = min_samples_per_bin
        self.increasing = increasing
        self.y_min = y_min
        self.y_max = y_max
        self.min_change_threshold = min_change_threshold

        # Dictionary to store fitted isotonic models for each feature
        self._isotonic_models: dict[Any, IsotonicRegression] = {}

        super().__init__(
            clip=clip,
            preserve_dataframe=preserve_dataframe,
            guidance_columns=guidance_columns,
            **kwargs,
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
            x_col (np.ndarray[Any, Any]): Feature data for binning with shape (n_samples,).
                May contain NaN values which will be handled appropriately.
            col_id (Any): Column identifier for error reporting and model storage.
            guidance_data (Optional[np.ndarray[Any, Any]], optional): Target/guidance data
                for supervised binning. Must be provided for isotonic binning.

        Returns:
            Tuple[List[float], List[float]]: A tuple containing:
                - bin_edges (List[float]): List of bin boundary values
                - bin_representatives (List[float]): List of representative values for each bin

        Raises:
            FittingError: If guidance_data is None or if there's insufficient valid data.
            ValueError: If isotonic regression fitting fails.

        Note:
            - Stores the fitted isotonic model in self._isotonic_models[col_id]
            - Handles both classification and regression targets
            - Automatically determines appropriate bin boundaries based on fitted function
        """
        # Require guidance data for supervised binning
        self.require_guidance_data(guidance_data, "isotonic binning")

        # At this point, guidance_data is guaranteed to be not None
        assert (
            guidance_data is not None
        ), "guidance_data should not be None after require_guidance_data check"

        # Convert categorical guidance data to numeric before validation
        if guidance_data.dtype == object or not np.issubdtype(guidance_data.dtype, np.number):
            # Pre-process categorical targets
            unique_values = np.unique(guidance_data)
            if len(unique_values) == 0:
                raise FittingError(f"Column {col_id}: No valid guidance data found")
            value_mapping = {val: i for i, val in enumerate(unique_values)}
            guidance_data_numeric = np.array(
                [value_mapping[val] for val in guidance_data], dtype=float
            )
        else:
            guidance_data_numeric = guidance_data

        # Validate and clean feature-target pairs
        x_validated, y_validated, valid_mask = self.validate_feature_target_pair(
            x_col, guidance_data_numeric, col_id
        )

        # Extract valid pairs
        x_clean, y_clean = self.extract_valid_pairs(x_validated, y_validated, valid_mask)
        n_valid = len(x_clean)

        # Check if we have sufficient data
        if n_valid < self.min_samples_per_bin:
            result = self.handle_insufficient_data(
                x_validated, valid_mask, self.min_samples_per_bin, col_id
            )
            if result is not None:
                return result
            raise FittingError(
                f"Column {col_id}: Insufficient valid data points ({n_valid}) "
                f"for isotonic binning. Need at least {self.min_samples_per_bin}."
            )

        # Create isotonic binning
        return self._create_isotonic_bins(x_clean, y_clean, col_id)

    def _create_isotonic_bins(
        self, x_col: np.ndarray[Any, Any], y_col: np.ndarray[Any, Any], col_id: Any
    ) -> tuple[list[float], list[float]]:
        """Create bins using isotonic regression.

        Fits an isotonic regression model to the feature-target relationship and
        identifies optimal cut points based on changes in the fitted function.

        Args:
            x_col (np.ndarray[Any, Any]): Clean feature data (no NaN values).
            y_col (np.ndarray[Any, Any]): Clean target data (no NaN values).
            col_id (Any): Column identifier for model storage.

        Returns:
            Tuple[List[float], List[float]]: Bin edges and representatives.

        Raises:
            ValueError: If isotonic regression fitting fails.
        """
        # Handle constant feature data
        if len(np.unique(x_col)) == 1:
            return self.handle_insufficient_data(
                x_col, np.ones(len(x_col), dtype=bool), self.min_samples_per_bin, col_id
            ) or ([float(x_col[0]) - 0.1, float(x_col[0]) + 0.1], [float(x_col[0])])

        # Sort data by feature values for isotonic regression
        sort_indices = np.argsort(x_col)
        x_sorted = x_col[sort_indices]
        y_sorted = y_col[sort_indices]

        # Prepare target values for isotonic regression
        y_processed = self._prepare_target_values(y_sorted)

        # Fit isotonic regression
        try:
            isotonic_model = IsotonicRegression(
                increasing=self.increasing,
                y_min=self.y_min,
                y_max=self.y_max,
                out_of_bounds="clip",
            )
            y_fitted = isotonic_model.fit_transform(x_sorted, y_processed)
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
            y_values (np.ndarray[Any, Any]): Raw target values.

        Returns:
            np.ndarray[Any, Any]: Processed target values suitable for isotonic regression.
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
        self, _: np.ndarray[Any, Any], y_fitted: np.ndarray[Any, Any]
    ) -> list[int]:
        """Find cut points based on changes in fitted isotonic function.

        Identifies locations where the fitted function has significant changes
        that warrant creating new bin boundaries.

        Args:
            x_sorted (np.ndarray[Any, Any]): Sorted feature values.
            y_fitted (np.ndarray[Any, Any]): Fitted isotonic regression values.

        Returns:
            List[int]: Indices of cut points in the sorted arrays.
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
        _: np.ndarray[Any, Any],
        cut_indices: list[int],
        __: Any,
    ) -> tuple[list[float], list[float]]:
        """Create bin edges and representatives from cut points.

        Args:
            x_sorted (np.ndarray[Any, Any]): Sorted feature values.
            y_fitted (np.ndarray[Any, Any]): Fitted isotonic regression values.
            cut_indices (List[int]): Indices of cut points.
            col_id (Any): Column identifier for error reporting.

        Returns:
            Tuple[List[float], List[float]]: Bin edges and representatives.
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

    def _validate_params(self) -> None:
        """Validate isotonic binning parameters.

        Raises:
            ConfigurationError: If any parameter validation fails.
        """
        # Call parent validation
        super()._validate_params()

        # Validate max_bins
        if not isinstance(self.max_bins, int) or self.max_bins < 1:
            raise ConfigurationError(
                "max_bins must be a positive integer",
                suggestions=["Set max_bins to a positive integer (e.g., max_bins=10)"],
            )

        # Validate min_samples_per_bin
        if not isinstance(self.min_samples_per_bin, int) or self.min_samples_per_bin < 1:
            raise ConfigurationError(
                "min_samples_per_bin must be a positive integer",
                suggestions=[
                    "Set min_samples_per_bin to a positive integer (e.g., min_samples_per_bin=5)"
                ],
            )

        # Validate increasing parameter
        if not isinstance(self.increasing, bool):
            raise ConfigurationError(
                "increasing must be a boolean",
                suggestions=["Set increasing=True for increasing monotonicity or increasing=False"],
            )

        # Validate y bounds if provided
        if self.y_min is not None and self.y_max is not None:
            if self.y_min >= self.y_max:
                raise ConfigurationError(
                    "y_min must be less than y_max",
                    suggestions=["Ensure y_min < y_max, e.g., y_min=0.0, y_max=1.0"],
                )

        # Validate min_change_threshold
        if not isinstance(self.min_change_threshold, int | float) or self.min_change_threshold <= 0:
            raise ConfigurationError(
                "min_change_threshold must be a positive number",
                suggestions=[
                    "Set min_change_threshold to a positive value (e.g., min_change_threshold=0.01)"
                ],
            )
