"""
Clean Manual Interval binning implementation for  architecture.

This module provides ManualIntervalBinning that inherits from IntervalBinningBase.
Uses user-provided bin edges rather than inferring them from data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base._interval_binning_base import IntervalBinningBase
from ..config import apply_config_defaults
from ..utils import ArrayLike, BinEdgesDict, ConfigurationError


# pylint: disable=too-many-ancestors
class ManualIntervalBinning(IntervalBinningBase):
    """Manual interval binning implementation for user-defined bin boundaries.

    This class provides complete control over binning boundaries by allowing users
    to specify exact bin edges for each column. Unlike automatic binning methods
    that infer boundaries from data, manual interval binning uses pre-defined
    edges, making it ideal for standardized binning schemes, domain-specific
    requirements, or ensuring consistent binning across different datasets.

    Manual interval binning is particularly useful for:
    - Implementing domain-specific binning rules (e.g., age groups, income brackets)
    - Ensuring consistent binning across training and test sets
    - Regulatory or business requirements with fixed bin boundaries
    - Comparative analysis requiring standardized bins across datasets

    Key Features:
    - Complete user control over bin boundaries for each column
    - No data-dependent bin edge calculation - uses provided edges exactly
    - Support for different binning schemes per column
    - Automatic generation of bin representatives if not provided
    - Integration with binlearn's clipping and format preservation features

    Algorithm:
    1. Validate and store user-provided bin edges
    2. Generate default representatives (bin centers) if not provided
    3. During transformation, assign values to bins based on interval membership
    4. Handle out-of-range values according to clipping configuration

    Parameters:
        bin_edges: Required dictionary mapping column identifiers to lists of bin
            edge values. Each edge list must contain at least 2 values and be
            sorted in ascending order. For example: {0: [0, 10, 20], 'age': [0, 18, 65, 100]}
        bin_representatives: Optional dictionary mapping column identifiers to
            lists of representative values for each bin. If not provided,
            representatives are automatically calculated as bin midpoints.

    Attributes:
        bin_edges_: Dictionary containing the provided bin edges (same as input)
        bin_representatives_: Dictionary containing bin representatives (provided
            or auto-generated)

    Example:
        >>> import numpy as np
        >>> from binlearn.methods import ManualIntervalBinning
        >>>
        >>> # Define custom bin edges for different features
        >>> bin_edges = {
        ...     'age': [0, 18, 35, 50, 65, 100],          # Age groups
        ...     'income': [0, 30000, 60000, 100000, float('inf')]  # Income brackets
        ... }
        >>>
        >>> # Create binner with custom edges
        >>> binner = ManualIntervalBinning(bin_edges=bin_edges)
        >>>
        >>> # Sample data
        >>> X = np.array([[25, 45000], [60, 80000], [30, 25000]])
        >>> X_binned = binner.fit_transform(X)  # fit() is no-op, transform() uses edges
        >>>
        >>> # With custom representatives
        >>> bin_reps = {
        ...     'age': [9, 26.5, 42.5, 57.5, 82.5],      # Custom age representatives
        ...     'income': [15000, 45000, 80000, 150000]    # Custom income representatives
        ... }
        >>> binner_custom = ManualIntervalBinning(
        ...     bin_edges=bin_edges,
        ...     bin_representatives=bin_reps
        ... )

    Note:
        - bin_edges is required and cannot be None
        - fit() method is essentially a no-op since edges are predefined
        - Each column can have different numbers of bins and edge values
        - Out-of-range values are handled according to the clip parameter
        - Inherits all interval binning capabilities from IntervalBinningBase
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        bin_edges: BinEdgesDict,
        bin_representatives: BinEdgesDict | None = None,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        *,
        class_: str | None = None,  # For reconstruction compatibility
        module_: str | None = None,  # For reconstruction compatibility
    ):
        """Initialize manual interval binning with user-defined bin edges.

        Sets up manual interval binning with explicitly provided bin boundaries
        and optional representatives. This method requires complete bin edge
        specification upfront and integrates with binlearn's configuration system
        for other parameters.

        Args:
            bin_edges: Required dictionary mapping column identifiers to lists of
                bin edge values. Each edge list must:
                - Contain at least 2 values (to define at least 1 bin)
                - Be sorted in ascending order
                - Contain only finite numeric values
                For example: {0: [0, 10, 20, 30], 'feature1': [0.0, 0.5, 1.0]}
            bin_representatives: Optional dictionary mapping column identifiers to
                lists of representative values for each bin. If provided, must have
                the same column keys as bin_edges and appropriate counts (one
                representative per bin). If None, representatives are automatically
                generated as bin midpoints.
            clip: Whether to clip out-of-range values to the nearest bin boundary
                during transformation. If True, values outside the defined range
                are assigned to the nearest edge bin. If False, they receive
                special out-of-range indices. If None, uses global configuration default.
            preserve_dataframe: Whether to preserve DataFrame format in outputs when
                input is a DataFrame. If None, uses global configuration default.
            class_: Class name for reconstruction compatibility (ignored during
                normal initialization).
            module_: Module name for reconstruction compatibility (ignored during
                normal initialization).

        Raises:
            ConfigurationError: If bin_edges is None or not provided, with helpful
                suggestions for proper usage.

        Example:
            >>> # Basic manual binning with auto-generated representatives
            >>> bin_edges = {
            ...     'feature1': [0, 10, 20, 30, 40],
            ...     'feature2': [-1.0, 0.0, 1.0, 2.0]
            ... }
            >>> binner = ManualIntervalBinning(bin_edges=bin_edges)
            >>>
            >>> # With custom representatives
            >>> bin_reps = {
            ...     'feature1': [5, 15, 25, 35],           # Custom values
            ...     'feature2': [-0.5, 0.5, 1.5]          # Custom values
            ... }
            >>> binner_custom = ManualIntervalBinning(
            ...     bin_edges=bin_edges,
            ...     bin_representatives=bin_reps
            ... )
            >>>
            >>> # With clipping enabled
            >>> binner_clip = ManualIntervalBinning(
            ...     bin_edges=bin_edges,
            ...     clip=True
            ... )

        Note:
            - bin_edges is the only required parameter and cannot be None
            - Validation of bin_edges format occurs during initialization
            - The fit() method will be essentially a no-op since edges are predefined
            - Each column can have different numbers of bins
            - Integration with global configuration for clip and preserve_dataframe
        """
        # For manual binning, bin_edges is required and passed directly
        if bin_edges is None:
            raise ConfigurationError(
                "bin_edges must be provided for ManualIntervalBinning",
                suggestions=[
                    "Provide bin_edges as a dictionary mapping columns to edge lists",
                    "Example: bin_edges={0: [0, 10, 20, 30], 1: [0, 5, 15, 25]}",
                ],
            )

        # Prepare user parameters for config integration (exclude never-configurable params)
        user_params = {
            "clip": clip,
            "preserve_dataframe": preserve_dataframe,
        }
        # Remove None values to allow config defaults to take effect
        user_params = {k: v for k, v in user_params.items() if v is not None}

        # Apply configuration defaults for manual_interval method
        resolved_params = apply_config_defaults("manual_interval", user_params)

        # Initialize parent with resolved parameters (never-configurable params passed as-is)
        # Manual binning doesn't need fit_jointly or guidance_columns
        IntervalBinningBase.__init__(
            self,
            clip=resolved_params.get("clip"),
            preserve_dataframe=resolved_params.get("preserve_dataframe"),
            fit_jointly=False,  # Manual binning doesn't fit from data
            guidance_columns=None,  # Not needed for unsupervised manual binning
            bin_edges=bin_edges,  # Required for manual binning
            bin_representatives=bin_representatives,  # Never configurable
        )

    def fit(
        self, X: ArrayLike, y: ArrayLike | None = None, **fit_params: Any
    ) -> ManualIntervalBinning:
        """Fit the Manual Interval binning (no-op since bins are pre-defined).

        For manual binning, the object is already fitted during initialization.
        This method only performs validation.

        Args:
            X: Input data (used only for validation)
            y: Target values (ignored for manual binning)
            **fit_params: Additional fit parameters (ignored)

        Returns:
            Self (fitted transformer)
        """
        # Just validate parameters - object is already fitted
        self._validate_params()
        return self

    def _validate_params(self) -> None:
        """Validate Manual Interval binning parameters."""
        # Call parent validation
        IntervalBinningBase._validate_params(self)

        # ManualIntervalBinning specific validation: bin_edges is required
        if self.bin_edges is None:
            raise ConfigurationError(
                "bin_edges must be provided for ManualIntervalBinning",
                suggestions=[
                    "Provide bin_edges as a dictionary mapping columns to edge lists",
                    "Example: bin_edges={0: [0, 10, 20, 30], 1: [0, 5, 15, 25]}",
                ],
            )

        if not self.bin_edges:  # Empty dict
            raise ConfigurationError(
                "bin_edges cannot be empty for ManualIntervalBinning",
                suggestions=[
                    "Provide at least one column with bin edges",
                    "Example: bin_edges={0: [0, 10, 20, 30]}",
                ],
            )

    def _calculate_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        guidance_data: np.ndarray[Any, Any] | None = None,
    ) -> tuple[list[float], list[float]]:
        """Should never be called for manual binning.

        Manual binning uses pre-defined specifications, so this method
        should never be invoked. If called, it indicates a logic error.
        """
        raise NotImplementedError(
            "Manual binning uses pre-defined specifications. "
            "_calculate_bins should never be called for ManualIntervalBinning."
        )
