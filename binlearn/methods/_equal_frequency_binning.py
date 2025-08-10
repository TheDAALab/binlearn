"""
Clean equal frequency binning implementation for  architecture.

This module provides EqualFrequencyBinning that inherits from IntervalBinningBase.
Creates bins containing approximately equal numbers of observations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base import IntervalBinningBase
from ..config import apply_config_defaults
from ..utils import (
    BinEdgesDict,
    ConfigurationError,
    create_param_dict_for_config,
    resolve_n_bins_parameter,
    validate_bin_number_for_calculation,
    validate_bin_number_parameter,
    validate_range_parameter,
)


# pylint: disable=too-many-ancestors
class EqualFrequencyBinning(IntervalBinningBase):
    """Equal frequency binning implementation for creating balanced population bins.

    This class implements equal frequency binning (also known as quantile binning),
    which creates bins containing approximately equal numbers of observations. Unlike
    equal width binning that creates uniform intervals, equal frequency binning
    adapts bin boundaries to the data distribution, ensuring each bin has roughly
    the same number of data points.

    Equal frequency binning is particularly effective for:
    - Skewed distributions where equal width bins would create imbalanced populations
    - Creating bins with consistent statistical power for analysis
    - Preprocessing for algorithms that perform better with balanced bin sizes
    - Ordinal discretization that preserves ranking relationships

    Key Features:
    - Creates bins with approximately equal observation counts
    - Handles duplicate values intelligently by adjusting bin boundaries
    - Supports custom quantile ranges for focusing on specific data portions
    - Automatic handling of edge cases (insufficient unique values, duplicates)
    - Inherits all interval binning capabilities (clipping, format preservation, etc.)

    Algorithm:
    1. Sort the data values for each column
    2. Calculate quantile positions for n_bins equal-frequency intervals
    3. Find actual data values at or near these quantile positions
    4. Adjust bin boundaries to handle duplicate values properly
    5. Create representative values (typically quantile midpoints)

    Parameters:
        n_bins: Number of bins to create, or string specification like 'sqrt', 'log2'.
            Can be an integer for exact count or string for automatic calculation.
            Default value can be configured globally via binlearn.config.
        quantile_range: Optional (min_quantile, max_quantile) tuple to focus binning
            on a specific portion of the data distribution. For example, (0.1, 0.9)
            creates bins only within the 10th-90th percentile range, ignoring outliers.

    Attributes:
        bin_edges_: Dictionary mapping column identifiers to lists of bin edges
            after fitting. Edges correspond to quantile boundaries in the data.
        bin_representatives_: Dictionary mapping column identifiers to lists
            of bin representatives (typically quantile midpoints).

    Example:
        >>> import numpy as np
        >>> from binlearn.methods import EqualFrequencyBinning
        >>>
        >>> # Skewed data - exponential distribution
        >>> X = np.random.exponential(2, (1000, 1))
        >>> binner = EqualFrequencyBinning(n_bins=4)
        >>> binner.fit(X)
        >>> X_binned = binner.transform(X)
        >>> # Each bin contains approximately 250 observations
        >>>
        >>> # Focus on middle 80% of data, ignoring extreme outliers
        >>> binner_focused = EqualFrequencyBinning(n_bins=5, quantile_range=(0.1, 0.9))
        >>> binner_focused.fit(X)

    Note:
        - Works best with continuous numeric data having many unique values
        - May create fewer bins than requested if data has insufficient unique values
        - Duplicate values at quantile boundaries are handled by boundary adjustment
        - Consider the trade-off between bin count and meaningful bin boundaries
        - Inherits clipping behavior and DataFrame preservation from IntervalBinningBase
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        n_bins: int | str | None = None,
        quantile_range: tuple[float, float] | None = None,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        fit_jointly: bool | None = None,
        *,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
        class_: (
            str | None
        ) = None,  # For reconstruction compatibility  # pylint: disable=unused-argument
        module_: (
            str | None
        ) = None,  # For reconstruction compatibility  # pylint: disable=unused-argument
    ):
        """Initialize equal frequency binning with configuration and parameters.

        Sets up the equal frequency binning method with user-specified parameters
        and configuration defaults. The method integrates with binlearn's global
        configuration system and supports flexible bin count specification.

        Args:
            n_bins: Number of equal-frequency bins to create, or string specification
                for automatic calculation. Can be:
                - Integer: exact number of bins to create
                - 'sqrt': number of bins = sqrt(n_samples)
                - 'log2': number of bins = log2(n_samples)
                - 'sturges': Sturges' rule for histogram bins
                If None, uses global configuration default for 'equal_frequency' method.
            quantile_range: Optional (min_quantile, max_quantile) tuple to focus
                binning on a specific portion of the data distribution. Values
                should be between 0.0 and 1.0 with min_quantile < max_quantile.
                For example, (0.05, 0.95) creates bins within the 5th-95th
                percentile range, effectively ignoring extreme outliers.
            clip: Whether to clip out-of-range values to the nearest bin boundary
                during transformation. If True, values outside the range are assigned
                to the nearest edge bin. If False, they receive special out-of-range
                indices. If None, uses global configuration default.
            preserve_dataframe: Whether to preserve DataFrame format in outputs when
                input is a DataFrame. If None, uses global configuration default.
            fit_jointly: Whether to fit all columns together (not applicable for
                equal frequency binning as it's inherently per-column). If None,
                uses global configuration default.
            bin_edges: Pre-computed bin edges for reconstruction/deserialization.
                If provided, no fitting is performed and these edges are used
                directly. Should be a dictionary mapping column identifiers to
                lists of edge values.
            bin_representatives: Pre-computed bin representatives for reconstruction.
                Must be provided together with bin_edges. Should be a dictionary
                mapping column identifiers to lists of representative values.
            class_: Class name for reconstruction compatibility (ignored during
                normal initialization).
            module_: Module name for reconstruction compatibility (ignored during
                normal initialization).

        Example:
            >>> # Basic initialization with automatic bin count
            >>> binner = EqualFrequencyBinning(n_bins='sqrt')
            >>>
            >>> # Fixed number of bins with outlier handling
            >>> binner = EqualFrequencyBinning(n_bins=10, quantile_range=(0.1, 0.9))
            >>>
            >>> # Reconstruction from saved parameters
            >>> binner = EqualFrequencyBinning(
            ...     n_bins=4,
            ...     bin_edges={'col1': [0.1, 0.3, 0.6, 0.8, 0.95]},
            ...     bin_representatives={'col1': [0.2, 0.45, 0.7, 0.875]}
            ... )

        Note:
            - String n_bins values are resolved during fitting based on data size
            - quantile_range is useful for handling outliers and focusing analysis
            - Pre-computed edges and representatives enable object reconstruction
            - Parameters integrate with binlearn's global configuration system
        """
        # Use standardized initialization pattern
        user_params = create_param_dict_for_config(
            n_bins=n_bins,
            quantile_range=quantile_range,
            clip=clip,
            preserve_dataframe=preserve_dataframe,
            fit_jointly=fit_jointly,
        )

        # Apply configuration defaults for equal_frequency method
        resolved_params = apply_config_defaults("equal_frequency", user_params)

        # Store method-specific parameters
        self.n_bins = resolved_params.get("n_bins", 10)
        self.quantile_range = resolved_params.get("quantile_range", None)

        # Initialize parent with resolved parameters (never-configurable params passed as-is)
        IntervalBinningBase.__init__(
            self,
            clip=resolved_params.get("clip"),
            preserve_dataframe=resolved_params.get("preserve_dataframe"),
            fit_jointly=resolved_params.get("fit_jointly"),
            guidance_columns=None,  # Not needed for unsupervised binning
            bin_edges=bin_edges,  # Never configurable
            bin_representatives=bin_representatives,  # Never configurable
        )

    def _validate_params(self) -> None:
        """Validate equal frequency binning parameters."""
        # Call parent validation
        IntervalBinningBase._validate_params(self)

        # Validate n_bins using centralized utility
        validate_bin_number_parameter(self.n_bins, param_name="n_bins")

        # Validate quantile_range using standardized utility
        if self.quantile_range is not None:
            validate_range_parameter(self.quantile_range, "quantile_range")

            min_q, max_q = self.quantile_range
            # Additional validation for quantile range constraints
            if min_q < 0 or max_q > 1 or min_q >= max_q:
                raise ConfigurationError(
                    "quantile_range values must be numbers between 0 and 1 with min < max",
                    suggestions=["Example: quantile_range=(0.1, 0.9)"],
                )

    def _calculate_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        guidance_data: np.ndarray[Any, Any] | None = None,
    ) -> tuple[list[float], list[float]]:
        """Calculate equal-frequency bins for a single column.

        Computes bin edges and representatives using quantiles to ensure
        approximately equal numbers of observations in each bin.

        Args:
            x_col: Preprocessed column data (from base class)
            col_id: Column identifier for error reporting
            guidance_data: Not used for equal-frequency binning (unsupervised)

        Returns:
            Tuple of (bin_edges, bin_representatives)

        Raises:
            ValueError: If n_bins is invalid or insufficient data for calculation
        """
        # Validate n_bins for calculation
        validate_bin_number_for_calculation(self.n_bins, param_name="n_bins")

        resolved_n_bins = resolve_n_bins_parameter(
            self.n_bins, data_shape=(len(x_col), 1), param_name="n_bins"
        )

        # Get quantile range for this data
        if self.quantile_range is not None:
            min_quantile, max_quantile = self.quantile_range
        else:
            min_quantile, max_quantile = 0.0, 1.0

        return self._create_equal_frequency_bins(
            x_col, col_id, min_quantile, max_quantile, resolved_n_bins
        )

    def _create_equal_frequency_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        min_quantile: float,
        max_quantile: float,
        n_bins: int,
    ) -> tuple[list[float], list[float]]:
        """Create equal-frequency bins using quantiles.

        Args:
            x_col: Preprocessed column data (no NaN/inf values)
            col_id: Column identifier for error reporting
            min_quantile: Minimum quantile (0.0 to 1.0)
            max_quantile: Maximum quantile (0.0 to 1.0)
            n_bins: Number of bins to create

        Returns:
            Tuple of (bin_edges, bin_representatives)

        Note:
            The data is already preprocessed by the base class, so we don't need
            to handle NaN/inf values or constant data here.
        """
        if len(x_col) < n_bins:
            raise ValueError(
                f"Column {col_id}: Insufficient values ({len(x_col)}) "
                f"for {n_bins} bins. Need at least {n_bins} values."
            )

        # Create quantile points from min_quantile to max_quantile
        quantile_points = np.linspace(min_quantile, max_quantile, n_bins + 1)

        # Calculate quantile values
        try:
            edges = np.quantile(x_col, quantile_points)
        except (ValueError, IndexError) as e:
            raise ValueError(f"Column {col_id}: Error calculating quantiles: {e}") from e

        # Convert to list and ensure edges are strictly increasing
        edges = list(edges)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-8

        # Create representatives as bin centers based on quantiles
        reps = []
        for i in range(n_bins):
            # Calculate representative as the median of values in this bin
            bin_mask = (x_col >= edges[i]) & (x_col <= edges[i + 1])
            bin_data = x_col[bin_mask]

            if len(bin_data) > 0:
                # Use median of bin data as representative
                rep = float(np.median(bin_data))
            else:
                # Fallback to bin center if no data in bin
                rep = (edges[i] + edges[i + 1]) / 2
            reps.append(rep)

        return edges, reps
