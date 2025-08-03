"""Equal-width binning transformer.

This module implements equal-width binning, where continuous data is divided
into bins of equal width across the range of each feature. This is one of the
most common and straightforward binning strategies.

Classes:
    EqualWidthBinning: Main transformer for equal-width binning operations.
"""

from typing import Any, List, Optional, Tuple

import numpy as np

from ..base._interval_binning_base import IntervalBinningBase
from ..base._repr_mixin import ReprMixin
from ..utils.errors import ConfigurationError
from ..utils.types import BinEdgesDict


# pylint: disable=too-many-ancestors
class EqualWidthBinning(ReprMixin, IntervalBinningBase):
    """Classic equal-width binning transformer.

    Creates bins of equal width across the range of each feature. Each bin
    spans the same numeric range, making this method intuitive and easy to
    interpret. The bins are determined solely by the minimum and maximum
    values in each feature, without considering the target variable.

    This transformer is sklearn-compatible and supports pandas/polars DataFrames.
    It includes advanced features like custom bin ranges, clipping, and
    comprehensive error handling.

    Attributes:
        n_bins (int): Number of bins per feature.
        bin_range (tuple, optional): Custom range for binning.
        clip (bool, optional): Whether to clip values outside bin range.
        columns (list, optional): Specific columns to bin.
        guidance_columns (list, optional): Columns to exclude from binlearn.
        preserve_dataframe (bool): Whether to preserve DataFrame format.
        bin_edges_ (dict): Computed bin edges after fitting.

    Example:
        >>> import numpy as np
        >>> from binlearn.methods import EqualWidthBinning
        >>> X = np.random.rand(100, 3)
        >>> binner = EqualWidthBinning(n_bins=5)
        >>> X_binned = binner.fit_transform(X)
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        n_bins: int = 10,
        bin_range: Optional[Tuple[float, float]] = None,
        clip: Optional[bool] = None,
        preserve_dataframe: Optional[bool] = None,
        bin_edges: Optional[BinEdgesDict] = None,
        bin_representatives: Optional[BinEdgesDict] = None,
        fit_jointly: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize EqualWidthBinning transformer.

        Creates an equal-width binning transformer that divides the range of each
        feature into bins of equal width. This is one of the most intuitive binning
        strategies where each bin spans the same numeric range.

        Args:
            n_bins (int, optional): Number of bins to create for each feature.
                Must be a positive integer. Defaults to 10.
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
            **kwargs: Additional arguments passed to parent IntervalBinningBase.

        Raises:
            ConfigurationError: If n_bins is not a positive integer or if bin_range
                is invalid (not a tuple or min >= max).

        Example:
            >>> # Basic usage with default parameters
            >>> binner = EqualWidthBinning(n_bins=5)

            >>> # Custom range for consistent binning
            >>> binner = EqualWidthBinning(n_bins=10, bin_range=(0, 100))

            >>> # With pre-specified bin edges
            >>> edges = {0: [0, 25, 50, 75, 100]}
            >>> binner = EqualWidthBinning(bin_edges=edges)
        """
        # Store equal-width specific parameters BEFORE calling super().__init__
        # because parent class calls _validate_params() which needs these attributes
        self.n_bins = n_bins
        self.bin_range = bin_range

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
        """Calculate equal-width bins for a single column or joint binning data.

        Computes bin edges and representatives for either a single feature (per-column
        fitting) or from all flattened data (joint fitting). Uses either the
        specified bin_range or the actual data range to determine bin boundaries.

        Args:
            x_col (np.ndarray): Data for binning. For per-column fitting, this is
                data for a single column with shape (n_samples,). For joint fitting,
                this is flattened data from all columns. May contain NaN values.
            col_id (Any): Column identifier (name or index) for error reporting
                and logging purposes. For joint fitting, this is typically the
                first column identifier.
            guidance_data (Optional[np.ndarray], optional): Guidance data for
                supervised binning. Not used in equal-width binning as it's an
                unsupervised method. Defaults to None.

        Returns:
            Tuple[List[float], List[float]]: A tuple containing:
                - bin_edges (List[float]): List of bin edge values with length n_bins+1
                - bin_representatives (List[float]): List of representative values
                  (bin centers) with length n_bins

        Raises:
            ValueError: If n_bins is less than 1 or if the data contains only
                infinite values making binning impossible.

        Note:
            - For per-column fitting: uses column-specific data range when bin_range is None
            - For joint fitting: uses global range from all flattened data
            - Handles all-NaN data by creating a default [0, 1] range
            - Guidance data is ignored as equal-width binning is unsupervised
        """
        if self.n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {self.n_bins}")

        # Get range for this data
        if self.bin_range is not None:
            min_val, max_val = self.bin_range
        else:
            # For both per-column and joint fitting, use the same range calculation
            min_val, max_val = self._get_data_range(x_col, col_id)

        return self._create_equal_width_bins(min_val, max_val, self.n_bins)

    def _get_data_range(self, x_col: np.ndarray, col_id: Any) -> Tuple[float, float]:
        """Get the data range with robust handling of edge cases.

        Determines the minimum and maximum values for the provided data while handling
        special cases like all-NaN data and infinite values. This method works for
        both single-column data and flattened multi-column data (joint fitting).

        Args:
            x_col (np.ndarray): Data to analyze. For per-column fitting, this is
                data for a single column with shape (n_samples,). For joint fitting,
                this is flattened data from all columns. May contain NaN or infinite values.
            col_id (Any): Column identifier used for error reporting and logging.

        Returns:
            Tuple[float, float]: A tuple containing (min_val, max_val) representing
                the range of the data. For all-NaN data, returns (0.0, 1.0)
                as a sensible default.

        Raises:
            ValueError: If the data contains only infinite values or if min/max
                values are not finite after excluding NaN values.

        Example:
            >>> col_data = np.array([1.0, 2.5, 3.0, np.nan, 4.5])
            >>> binner._get_data_range(col_data, 'feature1')
            (1.0, 4.5)

        Note:
            - Uses np.nanmin and np.nanmax to ignore NaN values
            - Provides default range (0.0, 1.0) for all-NaN data
            - Validates that computed min/max values are finite
            - Works identically for single-column and joint fitting scenarios
        """
        # Check if all values are NaN
        if np.all(np.isnan(x_col)):
            # Create a default range for all-NaN columns
            return 0.0, 1.0

        min_val: float = np.nanmin(x_col)
        max_val: float = np.nanmax(x_col)

        if not (np.isfinite(min_val) and np.isfinite(max_val)):
            # This can happen if there are inf values
            raise ValueError(f"Cannot create bins for column {col_id}: min and max must be finite.")

        return float(min_val), float(max_val)

    def _create_equal_width_bins(
        self, min_val: float, max_val: float, n_bins: int
    ) -> Tuple[List[float], List[float]]:
        """Create equal-width bins given range and number of bins.

        Generates bin edges and representative values for equal-width binning
        given the data range and desired number of bins. This is the core
        algorithm that creates evenly spaced bins across the specified range.

        Args:
            min_val (float): Minimum value of the range for binning.
            max_val (float): Maximum value of the range for binning.
            n_bins (int): Number of bins to create. Must be positive.

        Returns:
            Tuple[List[float], List[float]]: A tuple containing:
                - bin_edges (List[float]): List of n_bins+1 edge values that
                  define the bin boundaries, from min_val to max_val
                - bin_representatives (List[float]): List of n_bins representative
                  values (bin centers) that represent each bin

        Example:
            >>> binner._create_equal_width_bins(0.0, 10.0, 5)
            ([0.0, 2.0, 4.0, 6.0, 8.0, 10.0], [1.0, 3.0, 5.0, 7.0, 9.0])

        Note:
            - Handles constant data (min_val == max_val) by adding small epsilon
            - Uses np.linspace for precise edge calculation
            - Representatives are calculated as bin centers (midpoints)
            - Ensures equal width across all bins
        """
        if min_val == max_val:
            # Handle constant data
            epsilon = 1e-8
            min_val -= epsilon
            max_val += epsilon

        # Create equal-width bin edges
        edges = np.linspace(min_val, max_val, n_bins + 1)

        # Create representatives as bin centers
        reps = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]

        return list(edges), reps

    def _validate_params(self) -> None:
        """Validate parameters for sklearn compatibility and logical consistency.

        Performs comprehensive validation of all EqualWidthBinning parameters
        to ensure they meet the expected types, ranges, and logical constraints.
        This method provides early error detection and clear error messages
        for common configuration mistakes.

        Raises:
            ConfigurationError: If any parameter validation fails:
                - n_bins must be a positive integer
                - bin_range must be a valid tuple (min, max) with min < max

        Example:
            >>> # This will raise ConfigurationError
            >>> binner = EqualWidthBinning(n_bins=0)  # n_bins must be positive
            >>> binner = EqualWidthBinning(bin_range=(10, 5))  # min >= max

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

        # Validate bin_range if provided
        if self.bin_range is not None:
            if (
                not isinstance(self.bin_range, tuple)
                or len(self.bin_range) != 2
                or self.bin_range[0] >= self.bin_range[1]
            ):
                raise ConfigurationError(
                    "bin_range must be a tuple (min, max) with min < max",
                    suggestions=["Example: bin_range=(0, 100)"],
                )
