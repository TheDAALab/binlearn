"""Equal-width binning with minimum weight constraint transformer.

This module implements equal-width binning with a minimum weight constraint, where
continuous data is divided into bins of equal width, but the number of bins is adjusted
to ensure each bin contains at least a minimum total weight from the guidance column.

Classes:
    EqualWidthMinimumWeightBinning: Main transformer for equal-width minimum weight binning.
"""

import warnings
from typing import Any

import numpy as np

from ..base._interval_binning_base import IntervalBinningBase
from ..base._repr_mixin import ReprMixin
from ..utils.errors import ConfigurationError, DataQualityWarning, FittingError
from ..utils.types import BinEdgesDict, ColumnList, GuidanceColumns


# pylint: disable=too-many-ancestors
class EqualWidthMinimumWeightBinning(ReprMixin, IntervalBinningBase):
    """Equal-width binning with minimum weight constraint transformer.

    Creates bins of equal width across the range of each feature, but adjusts the
    number of bins to ensure each bin contains at least the specified minimum total
    weight from the guidance column. This method combines the interpretability of
    equal-width binning with weight-based constraints for more balanced bins.

    The guidance column is used to calculate weights within each bin, and bins are
    merged if they don't meet the minimum weight requirement. This is not supervised
    learning - the guidance column provides weights but doesn't influence the
    initial bin placement strategy.

    This transformer is sklearn-compatible and supports pandas/polars DataFrames.
    It includes comprehensive error handling and validation.

    Attributes:
        n_bins (int): Initial number of bins per feature.
        minimum_weight (float): Minimum total weight required per bin.
        bin_range (tuple, optional): Custom range for binning.
        clip (bool, optional): Whether to clip values outside bin range.
        columns (list, optional): Specific columns to bin.
        guidance_columns (GuidanceColumns): Column providing weights for binning.
        preserve_dataframe (bool): Whether to preserve DataFrame format.
        _bin_edges (dict): Computed bin edges after fitting.

    Example:
        >>> import numpy as np
        >>> from binlearn.methods import EqualWidthMinimumWeightBinning
        >>> X = np.random.rand(100, 2)
        >>> weights = np.random.rand(100)  # guidance column with weights
        >>> binner = EqualWidthMinimumWeightBinning(n_bins=5, minimum_weight=10.0)
        >>> X_binned = binner.fit_transform(X, guidance_data=weights)
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        n_bins: int = 10,
        minimum_weight: float = 1.0,
        bin_range: tuple[float, float] | None = None,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
        fit_jointly: bool | None = None,
        guidance_columns: GuidanceColumns | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize EqualWidthMinimumWeightBinning transformer.

        Creates an equal-width binning transformer with minimum weight constraints.
        The transformer creates equal-width bins but ensures each bin contains at
        least the specified minimum total weight from the guidance column.

        Args:
            n_bins (int, optional): Initial number of bins to create for each feature.
                Must be a positive integer. The actual number of bins may be lower
                due to minimum weight constraints. Defaults to 10.
            minimum_weight (float, optional): Minimum total weight required per bin
                from the guidance column. Must be positive. Bins with less weight
                will be merged with adjacent bins. Defaults to 1.0.
            bin_range (Optional[Tuple[float, float]], optional): Custom range for
                binning as (min, max). If None, uses the actual data range.
                Useful for ensuring consistent binning across datasets. Defaults to None.
            clip (Optional[bool], optional): Whether to clip out-of-range values
                to the nearest bin edge. If None, uses global configuration.
                Defaults to None.
            preserve_dataframe (Optional[bool], optional): Whether to return
                DataFrames when input is DataFrame. If None, uses global
                configuration. Defaults to None.
            bin_edges (Optional[BinEdgesDict], optional): Pre-specified bin edges
                for each column. If provided, these edges are used instead of
                calculating from data. Defaults to None.
            bin_representatives (Optional[BinEdgesDict], optional): Pre-specified
                representative values for each bin. If provided along with bin_edges,
                these representatives are used. Defaults to None.
            fit_jointly (Optional[bool], optional): Whether to fit parameters
                jointly across all columns using the same global range. If None,
                uses global configuration. Defaults to None.
            guidance_columns (Optional[GuidanceColumns], optional): Columns providing
                weights for minimum weight constraint. If None, defaults to the first
                column not being binned. Defaults to None.
            **kwargs: Additional arguments passed to parent IntervalBinningBase.

        Raises:
            ConfigurationError: If n_bins is not a positive integer, minimum_weight
                is not positive, or if bin_range is invalid.

        Example:
            >>> # Basic usage with default parameters
            >>> binner = EqualWidthMinimumWeightBinning(n_bins=5, minimum_weight=2.0)

            >>> # Custom range for consistent binning
            >>> binner = EqualWidthMinimumWeightBinning(
            ...     n_bins=10, minimum_weight=5.0, bin_range=(0, 100)
            ... )

            >>> # Pre-specified guidance columns
            >>> binner = EqualWidthMinimumWeightBinning(
            ...     n_bins=8, minimum_weight=3.0, guidance_columns=['weight_col']
            ... )
        """

        # Store specific parameters BEFORE calling super().__init__
        # because parent class calls _validate_params() which needs these attributes
        self.n_bins = n_bins
        self.minimum_weight = minimum_weight
        self.bin_range = bin_range

        super().__init__(
            clip=clip,
            preserve_dataframe=preserve_dataframe,
            bin_edges=bin_edges,
            bin_representatives=bin_representatives,
            fit_jointly=fit_jointly,
            guidance_columns=guidance_columns,
            **kwargs,
        )

    def _calculate_bins(
        self, x_col: np.ndarray, col_id: Any, guidance_data: np.ndarray | None = None
    ) -> tuple[list[float], list[float]]:
        """Calculate equal-width bins with minimum weight constraint for a single column.

        Computes bin edges and representatives starting with equal-width bins and then
        merging adjacent bins that don't meet the minimum weight requirement from the
        guidance data. The guidance data provides weights for each data point.

        Args:
            x_col (np.ndarray): Data for binning with shape (n_samples,).
                May contain NaN values which are excluded from binlearn calculations.
            col_id (Any): Column identifier (name or index) for error reporting
                and logging purposes.
            guidance_data (Optional[np.ndarray], optional): Weight values for each
                data point with shape (n_samples,). Must be provided for this binning
                method. Non-negative weights are expected. Defaults to None.

        Returns:
            Tuple[List[float], List[float]]: A tuple containing:
                - bin_edges (List[float]): List of bin edge values with length n_bins+1
                - bin_representatives (List[float]): List of representative values
                  (bin centers) with length n_bins

        Raises:
            ValueError: If n_bins <= 0, if guidance_data is None, if guidance_data
                contains negative values, or if data is insufficient for binning.
            FittingError: If no valid bins can be created due to weight constraints.
            DataQualityWarning: If guidance_data contains NaN values or if all
                weights are zero.

        Example:
            >>> binner = EqualWidthMinimumWeightBinning(n_bins=5, minimum_weight=2.0)
            >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            >>> weights = np.array([0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 0.5, 0.5, 2.0, 2.0])
            >>> edges, reps = binner._calculate_bins(data, 0, weights)
        """
        # Validate inputs
        if self.n_bins <= 0:
            raise ValueError("n_bins must be >= 1")

        if guidance_data is None:
            raise ValueError(
                f"Column {col_id}: EqualWidthMinimumWeightBinning requires guidance_data "
                "to calculate weights for minimum weight constraint"
            )

        # Remove NaN values from both data and guidance
        valid_mask = ~(np.isnan(x_col) | np.isnan(guidance_data))
        if not np.any(valid_mask):
            # Handle all-NaN case gracefully - return default bins

            warnings.warn(
                f"Column {col_id}: No valid (non-NaN) data points available for binning. "
                "Creating default bins.",
                DataQualityWarning,
                stacklevel=3,
            )
            # Return default single bin that covers the range
            return [-1e10, 1e10], [0.0]

        x_valid = x_col[valid_mask]
        weights_valid = guidance_data[valid_mask]

        # Check for negative weights
        if np.any(weights_valid < 0):
            raise ValueError(
                f"Column {col_id}: Guidance data contains negative weights. "
                "Only non-negative weights are supported."
            )

        # Warn about zero weights
        if np.all(weights_valid == 0):

            warnings.warn(
                f"Column {col_id}: All weights are zero. Binning may not work as expected.",
                DataQualityWarning,
                stacklevel=2,
            )

        # Check for sufficient data
        if len(x_valid) < 2:
            raise ValueError(
                f"Column {col_id}: Insufficient non-NaN values for binning. "
                f"Need at least 2 values, got {len(x_valid)}"
            )

        return self._create_weight_constrained_bins(x_valid, weights_valid, col_id)

    # pylint: disable=too-many-locals
    def _create_weight_constrained_bins(
        self, x_data: np.ndarray, weights: np.ndarray, col_id: Any
    ) -> tuple[list[float], list[float]]:
        """Create equal-width bins with minimum weight constraint.

        Args:
            x_data (np.ndarray): Valid (non-NaN) data values.
            weights (np.ndarray): Valid (non-NaN) weight values.
            col_id (Any): Column identifier for error reporting.

        Returns:
            Tuple[List[float], List[float]]: Bin edges and representatives.

        Raises:
            FittingError: If no valid bins can be created.
        """
        try:
            # Determine data range
            if self.bin_range is not None:
                data_min, data_max = self.bin_range
            else:
                data_min, data_max = np.min(x_data), np.max(x_data)

            # Handle edge case where all values are the same
            if data_max <= data_min:
                # Create a single bin that spans the value
                epsilon = 1e-8 if data_min != 0 else 1e-8
                bin_edges = [data_min - epsilon, data_max + epsilon]
                bin_representatives = [(data_min + data_max) / 2]
                return bin_edges, bin_representatives

            # Create initial equal-width bins
            initial_edges = np.linspace(data_min, data_max, self.n_bins + 1)

            # Assign data points to initial bins and calculate weights per bin
            bin_indices = np.digitize(x_data, initial_edges) - 1

            # Handle edge case: values equal to data_max get assigned to bin n_bins
            bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

            # Calculate total weight in each bin
            bin_weights = np.zeros(self.n_bins)
            for i in range(self.n_bins):
                mask = bin_indices == i
                if np.any(mask):
                    bin_weights[i] = np.sum(weights[mask])

            # Merge bins that don't meet minimum weight requirement
            merged_edges = self._merge_underweight_bins(initial_edges, bin_weights, col_id)

            # Calculate representatives as bin centers
            n_final_bins = len(merged_edges) - 1
            representatives = []
            for i in range(n_final_bins):
                center = (merged_edges[i] + merged_edges[i + 1]) / 2
                representatives.append(center)

            return list(merged_edges), representatives

        except Exception as e:
            raise FittingError(
                f"Column {col_id}: Error in equal-width minimum weight binning: {str(e)}"
            ) from e

    def _merge_underweight_bins(
        self, edges: np.ndarray, bin_weights: np.ndarray, col_id: Any
    ) -> np.ndarray:
        """Merge adjacent bins that don't meet minimum weight requirement.

        Args:
            edges (np.ndarray): Initial bin edges.
            bin_weights (np.ndarray): Weight totals for each initial bin.
            col_id (Any): Column identifier for error reporting.

        Returns:
            np.ndarray: Merged bin edges.

        Raises:
            FittingError: If total weight is insufficient for even one bin.
        """

        # Check if all weights are zero
        total_weight: float = np.sum(bin_weights)
        if total_weight == 0:
            warnings.warn(
                f"Column {col_id}: All weights are zero. Creating default equal-width bins.",
                DataQualityWarning,
                stacklevel=3,
            )
            # Return original edges for equal-width binning when all weights are zero
            return edges

        # Check if total weight is sufficient for minimum constraint
        if total_weight < self.minimum_weight:
            raise FittingError(
                f"Column {col_id}: Total weight ({total_weight:.3f}) is less than "
                f"minimum_weight ({self.minimum_weight}). Cannot create any valid bins."
            )

        # Perform the actual merging logic
        merged_edges = self._perform_bin_merging(edges, bin_weights)

        # Defensive check - ensure we have at least one valid bin
        if len(merged_edges) < 2:
            merged_edges = [float(edges[0]), float(edges[-1])]

        return np.array(merged_edges)

    def _perform_bin_merging(self, edges: np.ndarray, bin_weights: np.ndarray) -> list[float]:
        """Perform the actual bin merging logic.

        Separated from _merge_underweight_bins to make testing easier.

        Args:
            edges (np.ndarray): Initial bin edges.
            bin_weights (np.ndarray): Weight totals for each initial bin.

        Returns:
            List[float]: Merged bin edges (may have < 2 elements in edge cases).
        """
        # Start with first bin
        merged_edges = [float(edges[0])]
        current_weight = 0

        for i, bin_weight in enumerate(bin_weights):
            current_weight += bin_weight

            # If this is the last bin or weight meets minimum, close current bin
            if i == len(bin_weights) - 1 or current_weight >= self.minimum_weight:
                merged_edges.append(float(edges[i + 1]))
                current_weight = 0

        return merged_edges

    def _validate_params(self) -> None:
        """Validate initialization parameters.

        Raises:
            ConfigurationError: If any parameter is invalid.
        """
        super()._validate_params()

        # Validate n_bins
        if not isinstance(self.n_bins, int) or self.n_bins <= 0:
            raise ConfigurationError("n_bins must be a positive integer")

        # Validate minimum_weight
        if not isinstance(self.minimum_weight, int | float) or self.minimum_weight <= 0:
            raise ConfigurationError("minimum_weight must be a positive number")

        # Validate bin_range if provided
        if self.bin_range is not None:
            if (
                not isinstance(self.bin_range, tuple | list)
                or len(self.bin_range) != 2
                or not all(isinstance(x, int | float) for x in self.bin_range)
                or self.bin_range[0] >= self.bin_range[1]
            ):
                raise ConfigurationError(
                    "bin_range must be a tuple/list of two numbers (min, max) with min < max"
                )

    def requires_guidance_columns(self) -> bool:
        """Check if this binning method requires guidance columns.

        Returns:
            bool: True, as this method requires guidance data for weight calculations.
        """
        return True

    def _fit_jointly(self, X: np.ndarray, columns: ColumnList, **fit_params: Any) -> None:
        """Override joint fitting to handle guidance data properly.

        Joint fitting doesn't make conceptual sense for weight-constrained binning
        since weights are per-sample, not per-flattened-value. We'll use per-column
        fitting instead.

        Args:
            X (np.ndarray): Input data array.
            columns (ColumnList): Column identifiers.
            **fit_params: Additional parameters including guidance_data.
        """

        warnings.warn(
            "Joint fitting is not recommended for EqualWidthMinimumWeightBinning "
            "as weights are per-sample. Using per-column fitting instead.",
            DataQualityWarning,
            stacklevel=2,
        )

        # Extract guidance data from fit_params
        guidance_data = fit_params.get("guidance_data", None)

        # Use per-column fitting instead
        self._fit_per_column(X, columns, guidance_data)
