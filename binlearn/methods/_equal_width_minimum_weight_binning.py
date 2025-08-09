"""
Clean Equal Width Minimum Weight binning implementation for  architecture.

This module provides EqualWidthMinimumWeightBinning that inherits from SupervisedBinningBase.
Uses equal-width bins with minimum weight constraints from guidance data.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ..base import SupervisedBinningBase
from ..config import apply_config_defaults
from ..utils import (
    BinEdgesDict,
    ConfigurationError,
    DataQualityWarning,
    FittingError,
    resolve_n_bins_parameter,
    validate_bin_number_for_calculation,
    validate_bin_number_parameter,
)


# pylint: disable=too-many-ancestors
class EqualWidthMinimumWeightBinning(SupervisedBinningBase):
    """Equal-width binning with minimum weight constraint implementation using clean architecture.

    Creates bins of equal width across the range of each feature, but adjusts the
    number of bins to ensure each bin contains at least the specified minimum total
    weight from the guidance column. This method combines the interpretability of
    equal-width binning with weight-based constraints for more balanced bins.

    This approach is particularly valuable when working with weighted data where
    statistical significance or minimum sample requirements must be maintained within
    each bin. The algorithm starts with equal-width bins and then merges adjacent
    underweight bins until all remaining bins meet the minimum weight requirement.

    The weight constraint helps ensure that:
    - Each bin has sufficient statistical power for analysis
    - Bins are meaningful for weighted modeling or evaluation
    - Sparse regions in the data don't create unreliable bins
    - The resulting binning respects both spatial (equal-width) and statistical (weight) considerations

    When no bins can meet the minimum weight requirement individually, the algorithm
    creates a single bin containing all data to maintain functionality.

    This implementation follows the clean binlearn architecture with straight inheritance,
    dynamic column resolution, and parameter reconstruction capabilities.

    Args:
        n_bins: Initial number of equal-width bins to create before weight-based merging.
            Controls the granularity of the initial binning. Can be an integer or a
            string expression like 'sqrt', 'log2', etc. for dynamic calculation.
            Final number of bins may be smaller due to merging. If None, uses
            configuration default.
        minimum_weight: Minimum total weight required per bin. Bins with lower total
            weight will be merged with adjacent bins until this requirement is met.
            Must be positive. If None, uses configuration default.
        bin_range: Optional tuple specifying (min, max) range for binning. If provided,
            bins are created within this range rather than the data's natural range.
            Useful for ensuring consistent binning across datasets. If None, uses
            data's min/max values.
        clip: Whether to clip values outside the fitted range to the nearest bin edge.
            If None, uses configuration default.
        preserve_dataframe: Whether to preserve pandas DataFrame structure in transform
            operations. If None, uses configuration default.
        guidance_columns: Column specification for weight/guidance data used in
            supervised binning. Should point to weight values for each sample.
        bin_edges: Pre-computed bin edges for reconstruction. Should not be provided
            during normal usage.
        bin_representatives: Pre-computed bin representatives for reconstruction.
            Should not be provided during normal usage.
        class_: Class name for reconstruction compatibility. Internal use only.
        module_: Module name for reconstruction compatibility. Internal use only.

    Attributes:
        n_bins: Initial number of bins before merging
        minimum_weight: Minimum weight requirement per bin
        bin_range: Optional fixed range for binning

    Example:
        >>> import numpy as np
        >>> from binlearn.methods import EqualWidthMinimumWeightBinning
        >>>
        >>> # Create sample data with weights
        >>> np.random.seed(42)
        >>> X = np.random.uniform(0, 100, 1000).reshape(-1, 1)
        >>> weights = np.random.exponential(2.0, 1000)  # Exponentially distributed weights
        >>>
        >>> # Initialize with minimum weight constraint
        >>> binner = EqualWidthMinimumWeightBinning(
        ...     n_bins=10,
        ...     minimum_weight=50.0,
        ...     guidance_columns='weight'
        ... )
        >>>
        >>> # Fit with weight data
        >>> binner.fit(X, weights.reshape(-1, 1))
        >>> X_binned = binner.transform(X)
        >>>
        >>> # Check bin weights
        >>> for i, edges in enumerate(zip(binner.bin_edges_[0][:-1], binner.bin_edges_[0][1:])):
        ...     left, right = edges
        ...     mask = (X >= left) & (X < right) if i < len(binner.bin_edges_[0]) - 2 else (X >= left) & (X <= right)
        ...     bin_weight = np.sum(weights[mask.flatten()])
        ...     print(f"Bin {i}: [{left:.1f}, {right:.1f}] weight: {bin_weight:.1f}")

    Note:
        - Requires guidance data containing weight values for each sample
        - Final number of bins may be less than n_bins due to merging underweight bins
        - All weights must be non-negative (negative weights raise ValueError)
        - Bins are merged by combining adjacent underweight bins
        - Creates a single bin if no individual bins can meet the weight requirement
        - Each column is processed independently with its corresponding weight data
        - Weight-based merging preserves the equal-width property where possible

    See Also:
        EqualWidthBinning: Standard equal-width binning without weight constraints
        EqualFrequencyBinning: Equal-frequency binning for balanced sample counts
        SupervisedBinningBase: Base class for supervised binning methods

    References:
        This method extends standard equal-width binning with statistical adequacy constraints
        commonly used in risk modeling and weighted analysis scenarios.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        n_bins: int | str | None = None,
        minimum_weight: float | None = None,
        bin_range: tuple[float, float] | None = None,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        guidance_columns: Any = None,
        *,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
        class_: str | None = None,  # For reconstruction compatibility
        module_: str | None = None,  # For reconstruction compatibility
    ):
        """Initialize Equal Width Minimum Weight binning with weight constraints.

        Sets up equal-width binning with minimum weight constraints, combining spatial
        and statistical adequacy requirements. Applies configuration defaults for any
        unspecified parameters and validates the resulting configuration.

        Args:
            n_bins: Initial number of equal-width bins to create before weight-based
                merging. Controls the granularity of the initial binning. Can be:
                - Integer: Exact initial number of bins
                - String: Dynamic calculation expression ('sqrt', 'log2', etc.)
                Final number of bins may be smaller due to merging. Must be positive.
                If None, uses configuration default.
            minimum_weight: Minimum total weight required per bin. Bins with total
                weight below this threshold will be merged with adjacent bins until
                the requirement is met. Must be positive. If None, uses configuration
                default.
            bin_range: Optional tuple specifying (min_value, max_value) range for
                binning. If provided, equal-width bins are created within this range
                regardless of the actual data range. Useful for:
                - Consistent binning across multiple datasets
                - Excluding outliers from bin range calculation
                - Domain-specific range constraints
                Must be (min, max) where min < max. If None, uses data's actual range.
            clip: Whether to clip transformed values outside the fitted range to the
                nearest bin edge. If None, uses configuration default.
            preserve_dataframe: Whether to preserve pandas DataFrame structure in
                transform operations. If None, uses configuration default.
            guidance_columns: Column specification for weight/guidance data. Should
                point to columns containing weight values for each sample. Required
                for supervised binning during fit operations.
            bin_edges: Pre-computed bin edges dictionary for reconstruction. Internal
                use only - should not be provided during normal initialization.
            bin_representatives: Pre-computed representatives dictionary for
                reconstruction. Internal use only.
            class_: Class name string for reconstruction compatibility. Internal use only.
            module_: Module name string for reconstruction compatibility. Internal use only.

        Example:
            >>> # Standard initialization with weight constraints
            >>> binner = EqualWidthMinimumWeightBinning(
            ...     n_bins=8,
            ...     minimum_weight=100.0,
            ...     guidance_columns='sample_weight'
            ... )
            >>>
            >>> # Custom range with tighter weight requirements
            >>> binner = EqualWidthMinimumWeightBinning(
            ...     n_bins=12,
            ...     minimum_weight=50.0,
            ...     bin_range=(0, 1000),
            ...     guidance_columns=['weight_column']
            ... )
            >>>
            >>> # Use configuration defaults
            >>> binner = EqualWidthMinimumWeightBinning(
            ...     guidance_columns='weights'
            ... )

        Note:
            - Parameter validation occurs during initialization
            - Configuration defaults are applied for None parameters
            - The minimum_weight parameter is crucial for determining bin merging behavior
            - bin_range allows for consistent binning across datasets with different ranges
            - Guidance columns must point to weight data for the minimum weight constraint to work
            - Reconstruction parameters should not be provided during normal usage
        """
        # Prepare user parameters for config integration (exclude never-configurable params)
        user_params = {
            "n_bins": n_bins,
            "minimum_weight": minimum_weight,
            "bin_range": bin_range,
            "clip": clip,
            "preserve_dataframe": preserve_dataframe,
        }
        # Remove None values to allow config defaults to take effect
        user_params = {k: v for k, v in user_params.items() if v is not None}

        # Apply configuration defaults for equal_width_minimum_weight method
        resolved_params = apply_config_defaults("equal_width_minimum_weight", user_params)

        # Store method-specific parameters
        self.n_bins = resolved_params.get("n_bins", 10)
        self.minimum_weight = resolved_params.get("minimum_weight", 1.0)
        self.bin_range = resolved_params.get("bin_range", None)

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
        """Validate Equal Width Minimum Weight binning parameters."""
        # Call parent validation
        SupervisedBinningBase._validate_params(self)

        # Validate n_bins using centralized utility
        validate_bin_number_parameter(self.n_bins, param_name="n_bins")

        # Validate minimum_weight parameter
        if not isinstance(self.minimum_weight, int | float) or self.minimum_weight <= 0:
            raise ConfigurationError(
                "minimum_weight must be a positive number",
                suggestions=["Example: minimum_weight=1.0"],
            )

        # Validate bin_range parameter
        if self.bin_range is not None:
            if not isinstance(self.bin_range, list | tuple) or len(self.bin_range) != 2:
                raise ConfigurationError(
                    "bin_range must be a tuple/list of two numbers (min, max)",
                    suggestions=["Example: bin_range=(0, 100)"],
                )

            min_val, max_val = self.bin_range
            if not isinstance(min_val, int | float) or not isinstance(max_val, int | float):
                raise ConfigurationError(
                    "bin_range values must be numbers",
                    suggestions=["Example: bin_range=(0.0, 100.0)"],
                )

            if min_val >= max_val:
                raise ConfigurationError(
                    "bin_range minimum must be less than maximum",
                    suggestions=["Example: bin_range=(0, 100) where 0 < 100"],
                )

    def _calculate_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        guidance_data: np.ndarray[Any, Any] | None = None,
    ) -> tuple[list[float], list[float]]:
        """Calculate equal-width bins with minimum weight constraint for a single column.

        Computes bin edges and representatives starting with equal-width bins and then
        merging adjacent bins that don't meet the minimum weight requirement from the
        guidance data.

        Args:
            x_col: Preprocessed column data (from base class)
            col_id: Column identifier for error reporting
            guidance_data: Weight values for each data point (required)

        Returns:
            Tuple of (bin_edges, bin_representatives)

        Raises:
            FittingError: If guidance_data is None or insufficient data for binning
        """
        # Require guidance data for supervised binning
        if guidance_data is None:
            raise FittingError(
                f"Column {col_id}: EqualWidthMinimumWeightBinning requires guidance_data "
                "to calculate weights for minimum weight constraint"
            )

        # Validate n_bins for calculation
        validate_bin_number_for_calculation(self.n_bins, param_name="n_bins")

        resolved_n_bins = resolve_n_bins_parameter(
            self.n_bins, data_shape=(len(x_col), 1), param_name="n_bins"
        )

        # Extract the single weight column (guaranteed to have shape (n_samples, 1) by SupervisedBinningBase)
        weights = guidance_data[:, 0]

        return self._create_equal_width_minimum_weight_bins(x_col, weights, col_id, resolved_n_bins)

    def _create_equal_width_minimum_weight_bins(
        self,
        x_col: np.ndarray[Any, Any],
        weights: np.ndarray[Any, Any],
        col_id: Any,
        n_bins: int,
    ) -> tuple[list[float], list[float]]:
        """Create equal-width bins with minimum weight constraints.

        Args:
            x_col: Preprocessed column data (no NaN/inf values)
            weights: Weight values for each data point
            col_id: Column identifier for error reporting
            n_bins: Number of initial bins to create

        Returns:
            Tuple of (bin_edges, bin_representatives)

        Note:
            The data is already preprocessed by the base class, so we don't need
            to handle NaN/inf values here.
        """
        # Check for negative weights
        if np.any(weights < 0):
            raise ValueError(
                f"Column {col_id}: guidance_data contains negative values. "
                "All weights must be non-negative."
            )

        # Determine the range for binning
        if self.bin_range is not None:
            min_val, max_val = self.bin_range
        else:
            min_val, max_val = float(np.min(x_col)), float(np.max(x_col))

        # Handle constant data
        if min_val == max_val:
            # Create a single bin with small extension
            epsilon = 1e-8 if min_val != 0 else 1e-8
            edges = [min_val - epsilon, min_val + epsilon]
            representatives = [min_val]
            return edges, representatives

        # Create initial equal-width bins
        initial_edges = np.linspace(min_val, max_val, n_bins + 1)

        # Calculate weights in each initial bin
        bin_weights = []
        for i in range(n_bins):
            left_edge = initial_edges[i]
            right_edge = initial_edges[i + 1]

            # Include left edge, exclude right edge (except for last bin)
            if i == n_bins - 1:  # Last bin includes right edge
                mask = (x_col >= left_edge) & (x_col <= right_edge)
            else:
                mask = (x_col >= left_edge) & (x_col < right_edge)

            total_weight = np.sum(weights[mask])
            bin_weights.append(total_weight)

        # Merge bins with insufficient weight
        merged_edges, merged_weights = self._merge_underweight_bins(
            list(initial_edges), bin_weights, col_id
        )

        # Create representatives as bin centers
        representatives = []
        for i in range(len(merged_edges) - 1):
            center = (merged_edges[i] + merged_edges[i + 1]) / 2
            representatives.append(center)

        return merged_edges, representatives

    def _merge_underweight_bins(
        self,
        edges: list[float],
        bin_weights: list[float],
        col_id: Any,
    ) -> tuple[list[float], list[float]]:
        """Merge adjacent bins that don't meet minimum weight requirement.

        Args:
            edges: Initial bin edges
            bin_weights: Weight in each bin
            col_id: Column identifier for warnings

        Returns:
            Tuple of (merged_edges, merged_weights)
        """
        if len(edges) <= 2:  # Only one bin, can't merge further
            return edges, bin_weights

        merged_edges = [edges[0]]  # Start with first edge
        merged_weights = []
        current_weight = 0.0

        for i, weight in enumerate(bin_weights):
            current_weight += weight

            # Check if we've reached minimum weight or this is the last bin
            if current_weight >= self.minimum_weight or i == len(bin_weights) - 1:
                # Close current merged bin
                merged_edges.append(edges[i + 1])
                merged_weights.append(current_weight)
                current_weight = 0.0

        # Check if we ended up with no bins due to all weights being too small
        if len(merged_weights) == 0:
            warnings.warn(
                f"Column {col_id}: No bins meet minimum weight requirement "
                f"({self.minimum_weight}). Creating single bin with total weight "
                f"{sum(bin_weights)}.",
                DataQualityWarning,
                stacklevel=2,
            )
            # Return single bin with all data
            return [edges[0], edges[-1]], [sum(bin_weights)]

        return merged_edges, merged_weights
