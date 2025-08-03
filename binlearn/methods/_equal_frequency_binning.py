"""Equal-frequency binning transformer.

This module implements equal-frequency binning (also known as quantile binning),
where continuous data is divided into bins containing approximately equal numbers
of observations. This binning strategy ensures balanced bin populations but may
result in unequal bin widths.

Classes:
    EqualFrequencyBinning: Main transformer for equal-frequency binning operations.
"""

from typing import Any, List, Optional, Tuple

import numpy as np

from ..base._interval_binning_base import IntervalBinningBase
from ..base._repr_mixin import ReprMixin
from ..utils.errors import ConfigurationError
from ..utils.types import BinEdgesDict


# pylint: disable=too-many-ancestors
class EqualFrequencyBinning(ReprMixin, IntervalBinningBase):
    """Equal-frequency (quantile) binning transformer.

    Creates bins containing approximately equal numbers of observations across
    each feature. Each bin contains roughly the same number of data points,
    making this method useful when you want balanced bin populations regardless
    of the underlying data distribution.

    This transformer is sklearn-compatible and supports pandas/polars DataFrames.
    It includes advanced features like custom quantile ranges, clipping, and
    comprehensive error handling.

    Attributes:
        n_bins (int): Number of bins per feature.
        quantile_range (tuple, optional): Custom quantile range for binning.
        clip (bool, optional): Whether to clip values outside bin range.
        columns (list, optional): Specific columns to bin.
        guidance_columns (list, optional): Columns to exclude from binlearn.
        preserve_dataframe (bool): Whether to preserve DataFrame format.
        _bin_edges (dict): Computed bin edges after fitting.

    Example:
        >>> import numpy as np
        >>> from binlearn.methods import EqualFrequencyBinning
        >>> X = np.random.rand(100, 3)
        >>> binner = EqualFrequencyBinning(n_bins=5)
        >>> X_binned = binner.fit_transform(X)
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        n_bins: int = 10,
        quantile_range: Optional[Tuple[float, float]] = None,
        clip: Optional[bool] = None,
        preserve_dataframe: Optional[bool] = None,
        bin_edges: Optional[BinEdgesDict] = None,
        bin_representatives: Optional[BinEdgesDict] = None,
        fit_jointly: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize EqualFrequencyBinning transformer.

        Creates an equal-frequency binning transformer that divides each feature
        into bins containing approximately equal numbers of observations. This
        ensures balanced bin populations but may result in unequal bin widths.

        Args:
            n_bins (int, optional): Number of bins to create for each feature.
                Must be a positive integer. Defaults to 10.
            quantile_range (Optional[Tuple[float, float]], optional): Custom quantile
                range for binning as (min_quantile, max_quantile). Values should be
                between 0 and 1. If None, uses the full data range (0, 1).
                Useful for excluding outliers. Defaults to None.
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
                jointly across all columns using the same global quantiles. If None,
                uses global configuration. Defaults to None.
            **kwargs: Additional arguments passed to parent IntervalBinningBase.

        Raises:
            ConfigurationError: If n_bins is not a positive integer or if quantile_range
                is invalid (not a tuple or values outside [0, 1] or min >= max).

        Example:
            >>> # Basic usage with default parameters
            >>> binner = EqualFrequencyBinning(n_bins=5)

            >>> # Custom quantile range to exclude outliers
            >>> binner = EqualFrequencyBinning(n_bins=10, quantile_range=(0.1, 0.9))

            >>> # With pre-specified bin edges
            >>> edges = {0: [0, 25, 50, 75, 100]}
            >>> binner = EqualFrequencyBinning(bin_edges=edges)
        """
        # Store equal-frequency specific parameters BEFORE calling super().__init__
        # because parent class calls _validate_params() which needs these attributes
        self.n_bins = n_bins
        self.quantile_range = quantile_range

        super().__init__(
            clip=clip,
            preserve_dataframe=preserve_dataframe,
            bin_edges=bin_edges,
            bin_representatives=bin_representatives,
            fit_jointly=fit_jointly,
            **kwargs,
        )

    def _calculate_bins(
        self, x_col: np.ndarray, col_id: Any, guidance_data: Optional[np.ndarray] = None
    ) -> Tuple[List[float], List[float]]:
        """Calculate equal-frequency bins for a single column or joint binning data.

        Computes bin edges and representatives for either a single feature (per-column
        fitting) or from all flattened data (joint fitting). Uses quantiles to ensure
        approximately equal numbers of observations in each bin.

        Args:
            x_col (np.ndarray): Data for binning. For per-column fitting, this is
                data for a single column with shape (n_samples,). For joint fitting,
                this is flattened data from all columns. May contain NaN values.
            col_id (Any): Column identifier (name or index) for error reporting
                and logging purposes. For joint fitting, this is typically the
                first column identifier.
            guidance_data (Optional[np.ndarray], optional): Guidance data for
                supervised binning. Not used in equal-frequency binning as it's an
                unsupervised method. Defaults to None.

        Returns:
            Tuple[List[float], List[float]]: A tuple containing:
                - bin_edges (List[float]): List of bin edge values with length n_bins+1
                - bin_representatives (List[float]): List of representative values
                  (quantile centers) with length n_bins

        Raises:
            ValueError: If n_bins is less than 1 or if the data contains insufficient
                non-NaN values for quantile calculation.

        Note:
            - For per-column fitting: uses column-specific quantiles
            - For joint fitting: uses global quantiles from all flattened data
            - Handles all-NaN data by creating a default [0, 1] range
            - Guidance data is ignored as equal-frequency binning is unsupervised
            - May create fewer than n_bins if data has many duplicate values
        """
        if self.n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {self.n_bins}")

        # Get quantile range for this data
        if self.quantile_range is not None:
            min_quantile, max_quantile = self.quantile_range
        else:
            min_quantile, max_quantile = 0.0, 1.0

        return self._create_equal_frequency_bins(
            x_col, col_id, min_quantile, max_quantile, self.n_bins
        )

    # pylint: disable=too-many-locals
    def _create_equal_frequency_bins(
        self, x_col: np.ndarray, col_id: Any, min_quantile: float, max_quantile: float, n_bins: int
    ) -> Tuple[List[float], List[float]]:
        """Create equal-frequency bins using quantiles.

        Generates bin edges and representative values for equal-frequency binning
        using quantiles to ensure approximately equal numbers of observations in each bin.

        Args:
            x_col (np.ndarray): Data to bin. May contain NaN values.
            col_id (Any): Column identifier for error reporting.
            min_quantile (float): Minimum quantile (0.0 to 1.0).
            max_quantile (float): Maximum quantile (0.0 to 1.0).
            n_bins (int): Number of bins to create. Must be positive.

        Returns:
            Tuple[List[float], List[float]]: A tuple containing:
                - bin_edges (List[float]): List of n_bins+1 edge values that
                  define the bin boundaries based on quantiles
                - bin_representatives (List[float]): List of n_bins representative
                  values (quantile centers) that represent each bin

        Raises:
            ValueError: If data contains insufficient non-NaN values for quantile calculation.

        Example:
            >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            >>> binner._create_equal_frequency_bins(data, 'col1', 0.0, 1.0, 4)
            ([1.0, 3.25, 5.5, 7.75, 10.0], [2.125, 4.375, 6.625, 8.875])

        Note:
            - Uses np.nanquantile to handle NaN values
            - Handles constant data by adding small epsilon
            - May create fewer bins if data has many duplicate values
            - Representatives are calculated as quantile centers
        """
        # Remove NaN values for quantile calculation
        clean_data = x_col[~np.isnan(x_col)]

        if len(clean_data) == 0:
            # All NaN data - create default range
            edges = np.linspace(0.0, 1.0, n_bins + 1)
            reps = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]
            return list(edges), reps

        if len(clean_data) < n_bins:
            raise ValueError(
                f"Column {col_id}: Insufficient non-NaN values ({len(clean_data)}) "
                f"for {n_bins} bins. Need at least {n_bins} values."
            )

        # Create quantile points from min_quantile to max_quantile
        quantile_points = np.linspace(min_quantile, max_quantile, n_bins + 1)

        # Calculate quantile values
        try:
            edges = np.quantile(clean_data, quantile_points)
        except (ValueError, IndexError) as e:
            raise ValueError(f"Column {col_id}: Error calculating quantiles: {e}") from e

        # Handle case where quantiles result in duplicate edges (constant regions)
        edges = np.array(edges)
        if edges[0] == edges[-1]:
            # All data points are the same - add small epsilon
            epsilon = 1e-8
            edges[0] -= epsilon
            edges[-1] += epsilon

        # Ensure edges are strictly increasing
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-8

        # Create representatives as bin centers based on quantiles
        reps = []
        for i in range(n_bins):
            # Calculate representative as the median of values in this bin
            bin_mask = (clean_data >= edges[i]) & (clean_data <= edges[i + 1])
            bin_data = clean_data[bin_mask]

            if len(bin_data) > 0:
                # Use median of bin data as representative
                rep = np.median(bin_data)
            else:
                # Fallback to bin center if no data in bin
                rep = (edges[i] + edges[i + 1]) / 2
            reps.append(rep)

        return list(edges), reps

    def _validate_params(self) -> None:
        """Validate parameters for sklearn compatibility and logical consistency.

        Performs comprehensive validation of all EqualFrequencyBinning parameters
        to ensure they meet the expected types, ranges, and logical constraints.
        This method provides early error detection and clear error messages
        for common configuration mistakes.

        Raises:
            ConfigurationError: If any parameter validation fails:
                - n_bins must be a positive integer
                - quantile_range must be a valid tuple (min, max) with 0 <= min < max <= 1

        Example:
            >>> # This will raise ConfigurationError
            >>> binner = EqualFrequencyBinning(n_bins=0)  # n_bins must be positive
            >>> binner = EqualFrequencyBinning(quantile_range=(0.9, 0.1))  # min >= max
            >>> binner = EqualFrequencyBinning(quantile_range=(-0.1, 0.5))  # out of range

        Note:
            - Called automatically during fit() for early error detection
            - Provides helpful suggestions in error messages
            - Focuses on parameter validation, not data validation
            - Part of sklearn-compatible parameter validation pattern
        """
        # Call parent validation first (handles bin edges and representatives)
        super()._validate_params()

        # Validate n_bins
        if not isinstance(self.n_bins, int) or self.n_bins < 1:
            raise ConfigurationError(
                "n_bins must be a positive integer",
                suggestions=["Set n_bins to a positive integer (e.g., n_bins=10)"],
            )

        # Validate quantile_range if provided
        if self.quantile_range is not None:
            if not isinstance(self.quantile_range, tuple) or len(self.quantile_range) != 2:
                raise ConfigurationError(
                    "quantile_range must be a tuple (min_quantile, max_quantile)",
                    suggestions=["Example: quantile_range=(0.1, 0.9)"],
                )

            min_q, max_q = self.quantile_range
            if (
                not isinstance(min_q, (int, float))
                or not isinstance(max_q, (int, float))
                or min_q < 0
                or max_q > 1
                or min_q >= max_q
            ):
                raise ConfigurationError(
                    "quantile_range values must be numbers between 0 and 1 with min < max",
                    suggestions=["Example: quantile_range=(0.1, 0.9)"],
                )
