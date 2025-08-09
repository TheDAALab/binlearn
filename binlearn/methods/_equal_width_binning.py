"""
Clean equal width binning implementation for  architecture.

This module provides EqualWidthBinning that inherits from IntervalBinningBase.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base import IntervalBinningBase
from ..config import apply_config_defaults
from ..utils import BinEdgesDict


# pylint: disable=too-many-ancestors
class EqualWidthBinning(IntervalBinningBase):
    """Equal width binning implementation for creating uniform interval bins.

    This class implements equal width binning, one of the most fundamental and
    widely-used discretization methods. It divides the range of continuous values
    into a specified number of bins, each having the same width (range span).
    The method is simple, interpretable, and works well when data is uniformly
    distributed across the value range.

    Equal width binning is particularly effective for:
    - Uniformly distributed data where equal intervals make intuitive sense
    - Creating interpretable bins with consistent spacing
    - Baseline discretization before trying more sophisticated methods
    - Applications where maintaining consistent bin widths is important

    Key Features:
    - Creates bins with identical width (max_value - min_value) / n_bins
    - Handles custom value ranges via bin_range parameter
    - Automatic bin representative calculation as interval midpoints
    - Inherits all interval binning capabilities (clipping, format preservation, etc.)
    - Supports both automatic range detection and user-specified ranges

    Algorithm:
    1. Determine value range (from data or user-specified bin_range)
    2. Create n_bins equally-spaced intervals across the range
    3. Assign representative values as bin centers
    4. Transform values by finding their containing interval

    Parameters:
        n_bins: Number of bins to create. Must be a positive integer.
            Default value can be configured globally via binlearn.config.
        bin_range: Optional custom range as (min, max) tuple. If provided,
            bins are created within this range regardless of actual data range.
            Useful for ensuring consistent binning across datasets.

    Attributes:
        bin_edges_: Dictionary mapping column identifiers to lists of bin edges
            after fitting. Each edge list contains n_bins + 1 values.
        bin_representatives_: Dictionary mapping column identifiers to lists
            of bin representatives (typically bin centers).

    Example:
        >>> import numpy as np
        >>> from binlearn.methods import EqualWidthBinning
        >>>
        >>> # Basic equal width binning
        >>> X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        >>> binner = EqualWidthBinning(n_bins=3)
        >>> binner.fit(X)
        >>> X_binned = binner.transform(X)
        >>> print(X_binned)  # [[0, 0], [1, 1], [2, 2], [2, 2]]
        >>>
        >>> # With custom range
        >>> binner_custom = EqualWidthBinning(n_bins=2, bin_range=(0.0, 10.0))
        >>> binner_custom.fit(X)
        >>> # Bins are created in [0, 10] range regardless of actual data range

    Note:
        - Only works with numeric data - non-numeric columns will raise errors
        - Constant columns (same value everywhere) are handled by creating single bin
        - Outliers can significantly affect bin boundaries in automatic range mode
        - Consider using bin_range parameter for consistent binning across datasets
        - Inherits clipping behavior and DataFrame preservation from IntervalBinningBase
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        n_bins: int | None = None,
        bin_range: tuple[float, float] | None = None,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        fit_jointly: bool | None = None,
        guidance_columns: Any = None,
        *,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
        class_: str | None = None,  # For reconstruction compatibility
        module_: str | None = None,  # For reconstruction compatibility
    ):
        """Initialize equal width binning with configuration and parameters.

        Sets up the equal width binning method with user-specified parameters
        and configuration defaults. The method integrates with binlearn's global
        configuration system while allowing parameter-specific overrides.

        Args:
            n_bins: Number of equal-width bins to create for each column. Must be
                a positive integer. If None, uses the global configuration default
                for the 'equal_width' method, typically 5.
            bin_range: Optional custom range as (min, max) tuple within which to
                create bins. If provided, bins are created within this range
                regardless of the actual data range. Useful for ensuring consistent
                binning across different datasets. If None, range is determined
                automatically from the data during fitting.
            clip: Whether to clip out-of-range values to the nearest bin boundary
                during transformation. If True, values outside the range are assigned
                to the nearest edge bin. If False, they receive special out-of-range
                indices. If None, uses global configuration default.
            preserve_dataframe: Whether to preserve DataFrame format in outputs when
                input is a DataFrame. If None, uses global configuration default.
            fit_jointly: Whether to fit all columns together (not applicable for
                equal width binning as it's inherently per-column). If None, uses
                global configuration default.
            guidance_columns: Additional columns to include in input validation but
                not to bin. Not typically used for equal width binning. If None,
                no guidance columns are expected.
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
            >>> # Basic initialization with defaults
            >>> binner = EqualWidthBinning()
            >>>
            >>> # Custom number of bins and range
            >>> binner = EqualWidthBinning(n_bins=10, bin_range=(0.0, 100.0))
            >>>
            >>> # Reconstruction from saved parameters
            >>> binner = EqualWidthBinning(
            ...     n_bins=5,
            ...     bin_edges={'col1': [0, 1, 2, 3, 4, 5]},
            ...     bin_representatives={'col1': [0.5, 1.5, 2.5, 3.5, 4.5]}
            ... )

        Note:
            - Parameters integrate with binlearn's global configuration system
            - None values allow configuration defaults to take effect
            - Pre-computed edges and representatives enable object reconstruction
            - Class follows sklearn's estimator interface conventions
        """
        # Prepare user parameters for config integration (exclude never-configurable params)
        user_params = {
            "n_bins": n_bins,
            "bin_range": bin_range,
            "clip": clip,
            "preserve_dataframe": preserve_dataframe,
            "fit_jointly": fit_jointly,
        }
        # Remove None values to allow config defaults to take effect
        user_params = {k: v for k, v in user_params.items() if v is not None}

        # Apply configuration defaults for equal_width method
        params = apply_config_defaults("equal_width", user_params)

        # Store equal width specific parameters
        self.n_bins = params.get("n_bins", 5)  # Fallback to 5 if not in config
        self.bin_range = params.get("bin_range", bin_range)

        # Initialize parent with resolved config parameters
        # Note: bin_edges, bin_representatives, guidance_columns are never set from config
        IntervalBinningBase.__init__(
            self,
            clip=params.get("clip"),
            preserve_dataframe=params.get("preserve_dataframe"),
            fit_jointly=params.get("fit_jointly"),
            guidance_columns=guidance_columns,
            bin_edges=bin_edges,
            bin_representatives=bin_representatives,
        )

    def _validate_params(self) -> None:
        """Validate equal width binning specific parameters.

        Performs comprehensive validation of equal width binning parameters
        to ensure they meet the method's requirements. This includes checking
        parameter types, ranges, and logical consistency.

        Raises:
            ValueError: If any parameter is invalid:
                - n_bins is not a positive integer
                - bin_range is not a valid (min, max) tuple with min < max

        Note:
            - Calls parent class validation first for inherited parameters
            - Validates that n_bins is appropriate for creating meaningful bins
            - Ensures bin_range, if provided, defines a valid interval
            - Called automatically during initialization
        """
        # Call parent validation
        IntervalBinningBase._validate_params(self)

        # Validate n_bins
        if not isinstance(self.n_bins, int) or self.n_bins < 1:
            raise ValueError("n_bins must be a positive integer")

        # Validate bin_range
        if self.bin_range is not None:
            if (
                not isinstance(self.bin_range, tuple)
                or len(self.bin_range) != 2
                or self.bin_range[0] >= self.bin_range[1]
            ):
                raise ValueError("bin_range must be a tuple (min, max) with min < max")

    def _calculate_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        guidance_data: np.ndarray[Any, Any] | None = None,
    ) -> tuple[list[float], list[float]]:
        """Calculate equal width bin edges and representatives for a single column.

        This method implements the core equal width binning algorithm by determining
        the appropriate value range and creating evenly-spaced intervals within
        that range.

        Args:
            x_col: Preprocessed column data as numpy array. The data has been
                validated and cleaned by the parent class, containing only finite
                numeric values.
            col_id: Column identifier for error messages and logging. Can be
                string, integer, or other hashable type.
            guidance_data: Optional guidance data (not used in equal width binning
                but included for interface compatibility). Ignored in this method.

        Returns:
            Tuple containing:
            - bin_edges: List of n_bins + 1 edge values defining the bin boundaries
            - bin_representatives: List of n_bins representative values (bin centers)

        Example:
            >>> x_col = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            >>> edges, reps = binner._calculate_bins(x_col, 'feature1')
            >>> # With n_bins=3: edges might be [1.0, 2.33, 3.67, 5.0]
            >>> #                  reps might be [1.67, 3.0, 4.33]

        Note:
            - Uses bin_range parameter if provided, otherwise determines range from data
            - Creates exactly n_bins intervals with equal width
            - Representatives are calculated as interval midpoints
            - Handles edge cases like constant data (handled by parent class)
        """
        # Get range for this data
        if self.bin_range is not None:
            min_val, max_val = self.bin_range
        else:
            min_val, max_val = self._get_data_range(x_col, col_id)

        return self._create_equal_width_bins(min_val, max_val, self.n_bins)

    def _get_data_range(self, x_col: np.ndarray[Any, Any], col_id: Any) -> tuple[float, float]:
        """Get the data range from preprocessed column data.

        Computes the minimum and maximum values from the preprocessed column data
        to determine the range for bin creation. The data has already been validated
        and cleaned by the parent class.

        Args:
            x_col: Preprocessed column data containing only finite numeric values.
                Missing values, infinities, and non-numeric data have been handled
                by the parent class validation.
            col_id: Column identifier used for error reporting and debugging.

        Returns:
            Tuple of (min_value, max_value) as floats representing the data range
            that will be used for creating equal-width bins.

        Example:
            >>> x_col = np.array([1.5, 2.3, 4.7, 3.1])
            >>> min_val, max_val = binner._get_data_range(x_col, 'feature1')
            >>> # Returns: (1.5, 4.7)

        Note:
            - Data has been preprocessed so no additional validation is needed
            - Returns exact min/max values from the data
            - Used when bin_range parameter is not provided
            - Constant data (min == max) is handled by parent class before this method
        """
        min_val: float = np.min(x_col)
        max_val: float = np.max(x_col)

        return float(min_val), float(max_val)

    def _create_equal_width_bins(
        self, min_val: float, max_val: float, n_bins: int
    ) -> tuple[list[float], list[float]]:
        """Create equal-width bins given range and number of bins.

        This method implements the core equal width binning algorithm by creating
        evenly-spaced intervals within the specified range and computing appropriate
        representative values for each bin.

        Args:
            min_val: Minimum value of the range within which to create bins.
                Can come from data or user-specified bin_range parameter.
            max_val: Maximum value of the range within which to create bins.
                Must be greater than min_val.
            n_bins: Number of equal-width bins to create. Must be positive integer.

        Returns:
            Tuple containing:
            - edges: List of n_bins + 1 equally-spaced edge values from min_val to max_val
            - representatives: List of n_bins representative values (bin centers)

        Example:
            >>> edges, reps = binner._create_equal_width_bins(0.0, 10.0, 4)
            >>> # edges = [0.0, 2.5, 5.0, 7.5, 10.0]
            >>> # reps = [1.25, 3.75, 6.25, 8.75]

        Note:
            - Uses numpy.linspace for precise equal spacing
            - Representatives are calculated as arithmetic mean of bin boundaries
            - The range has been validated to ensure min_val < max_val
            - Creates exactly n_bins intervals with width = (max_val - min_val) / n_bins
        """
        # Create equal-width bin edges
        edges = np.linspace(min_val, max_val, n_bins + 1)

        # Create representatives as bin centers
        reps = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]

        return list(edges), reps
