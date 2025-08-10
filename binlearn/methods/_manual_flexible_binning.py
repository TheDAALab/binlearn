"""
Clean Manual Flexible binning implementation for  architecture.

This module provides ManualFlexibleBinning that inherits from FlexibleBinningBase.
Uses user-provided flexible bin specifications with both singleton and interval bins.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base import FlexibleBinningBase
from ..config import apply_config_defaults
from ..utils import (
    ArrayLike,
    BinEdgesDict,
    BinReps,
    ConfigurationError,
    FlexibleBinDefs,
    FlexibleBinSpec,
    create_param_dict_for_config,
)


# pylint: disable=too-many-ancestors
class ManualFlexibleBinning(FlexibleBinningBase):
    """Manual flexible binning implementation for user-defined mixed bin types.

    This class provides complete control over flexible binning by allowing users
    to specify bin definitions that can include both singleton bins (exact value
    matching) and interval bins (range matching) within the same feature. This
    flexibility makes it ideal for domain-specific binning requirements, handling
    special values, and creating custom discretization schemes.

    Manual flexible binning is particularly useful for:
    - Mixed data types requiring both exact and range-based binning
    - Handling special values (outliers, missing indicators) as singleton bins
    - Domain-specific requirements with irregular bin boundaries
    - Creating bins that combine categorical-like values with continuous ranges

    Key Features:
    - Support for mixed bin types within the same feature
    - Singleton bins for exact value matching
    - Interval bins for range-based matching
    - No data-dependent bin calculation - uses provided specifications exactly
    - Automatic generation of representatives if not provided
    - Integration with binlearn's format preservation features

    Algorithm:
    1. Validate and store user-provided flexible bin specifications
    2. Generate default representatives if not provided:
       - For singleton bins: use the singleton value itself
       - For interval bins: use the interval midpoint
    3. During transformation, match values against bin definitions:
       - Check singleton bins for exact matches
       - Check interval bins for range membership
       - Return index of first matching bin

    Parameters:
        bin_spec: Required dictionary mapping column identifiers to lists of
            flexible bin definitions. Each bin definition can be either:
            - Scalar value: singleton bin matching exactly that value
            - Tuple (start, end): interval bin matching values in [start, end]
            For example: {0: [42, (10, 20), 'special'], 'age': [(0, 18), (18, 65), (65, 100)]}
        bin_representatives: Optional dictionary mapping column identifiers to
            lists of representative values for each bin. If not provided,
            representatives are automatically generated.

    Attributes:
        bin_spec_: Dictionary containing the provided flexible bin specifications
        bin_representatives_: Dictionary containing bin representatives (provided
            or auto-generated)

    Example:
        >>> import numpy as np
        >>> from binlearn.methods import ManualFlexibleBinning
        >>>
        >>> # Define mixed bin types for different features
        >>> bin_spec = {
        ...     'numeric_feature': [
        ...         0,              # Singleton: exactly zero
        ...         (1, 10),        # Interval: 1 to 10
        ...         (10, 100),      # Interval: 10 to 100
        ...         999             # Singleton: exactly 999 (outlier)
        ...     ],
        ...     'mixed_feature': [
        ...         'special',      # Singleton: exactly 'special'
        ...         (0, 50),        # Interval: 0 to 50
        ...         (50, 100)       # Interval: 50 to 100
        ...     ]
        ... }
        >>>
        >>> # Create binner with flexible specifications
        >>> binner = ManualFlexibleBinning(bin_spec=bin_spec)
        >>>
        >>> # Sample data with mixed types
        >>> X = np.array([[0, 25], [5, 75], [999, 'special']], dtype=object)
        >>> X_binned = binner.fit_transform(X)
        >>> # Results: [[0, 1], [1, 2], [3, 0]]
        >>>
        >>> # With custom representatives
        >>> bin_reps = {
        ...     'numeric_feature': [0, 5.5, 55, 999],    # Custom representatives
        ...     'mixed_feature': ['special', 25, 75]      # Mixed type representatives
        ... }
        >>> binner_custom = ManualFlexibleBinning(
        ...     bin_spec=bin_spec,
        ...     bin_representatives=bin_reps
        ... )

    Note:
        - bin_spec is required and cannot be None
        - fit() method is essentially a no-op since specifications are predefined
        - Values are matched against bins in order - first match wins
        - Singleton bins support any hashable type (numeric, string, etc.)
        - Interval bins only work with numeric values
        - Unmatched values receive MISSING_VALUE (-1) bin index
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        bin_spec: FlexibleBinSpec,
        bin_representatives: BinEdgesDict | None = None,
        preserve_dataframe: bool | None = None,
        *,
        class_: (  # pylint: disable=unused-argument
            str | None
        ) = None,  # For reconstruction compatibility
        module_: (  # pylint: disable=unused-argument
            str | None
        ) = None,  # For reconstruction compatibility
    ):
        """Initialize manual flexible binning with user-defined bin specifications.

        Sets up manual flexible binning with explicitly provided bin definitions
        that can include both singleton and interval bins. This method requires
        complete bin specification upfront and integrates with binlearn's
        configuration system for other parameters.

        Args:
            bin_spec: Required dictionary mapping column identifiers to lists of
                flexible bin definitions. Each bin definition can be either:
                - Scalar value (any type): singleton bin matching exactly that value
                - Tuple (start, end): interval bin matching numeric values in [start, end]
                Mixed types are allowed within the same feature. For example:
                {0: [42, (10, 20), 'special'], 'col': [(0, 50), (50, 100)]}
            bin_representatives: Optional dictionary mapping column identifiers to
                lists of representative values for each bin. If provided, must have
                the same column keys as bin_spec and appropriate counts (one
                representative per bin). If None, representatives are automatically
                generated:
                - For singleton bins: the singleton value itself
                - For interval bins: the interval midpoint (start + end) / 2
            preserve_dataframe: Whether to preserve DataFrame format in outputs when
                input is a DataFrame. If None, uses global configuration default.
            class_: Class name for reconstruction compatibility (ignored during
                normal initialization).
            module_: Module name for reconstruction compatibility (ignored during
                normal initialization).

        Raises:
            ConfigurationError: If bin_spec is None or not provided, with helpful
                suggestions for proper usage including example formats.

        Example:
            >>> # Basic flexible binning with auto-generated representatives
            >>> bin_spec = {
            ...     'feature1': [0, (1, 10), (10, 100), 999],     # Mixed types
            ...     'feature2': [(0, 25), 'special', (50, 100)]   # Mixed types
            ... }
            >>> binner = ManualFlexibleBinning(bin_spec=bin_spec)
            >>>
            >>> # With custom representatives
            >>> bin_reps = {
            ...     'feature1': [0, 5.5, 55, 999],      # Custom values
            ...     'feature2': [12.5, 'special', 75]   # Mixed representatives
            ... }
            >>> binner_custom = ManualFlexibleBinning(
            ...     bin_spec=bin_spec,
            ...     bin_representatives=bin_reps
            ... )
            >>>
            >>> # Single feature with intervals only
            >>> simple_spec = {'price': [(0, 100), (100, 500), (500, float('inf'))]}
            >>> binner_simple = ManualFlexibleBinning(bin_spec=simple_spec)

        Note:
            - bin_spec is the only required parameter and cannot be None
            - Validation of bin_spec format occurs during initialization
            - The fit() method will be essentially a no-op since specs are predefined
            - Each column can have different numbers and types of bins
            - Singleton bins can be any hashable type (numbers, strings, etc.)
            - Interval bins must have numeric start and end values
        """
        # For manual flexible binning, bin_spec is required and passed directly
        if bin_spec is None:
            raise ConfigurationError(
                "bin_spec must be provided for ManualFlexibleBinning",
                suggestions=[
                    "Provide bin_spec as a dictionary mapping columns to flexible bin lists",
                    "Example: bin_spec={0: [1.5, (2, 5), (5, 10)], 1: [(0, 25), (25, 50)]}",
                ],
            )

        # Use standardized initialization pattern
        user_params = create_param_dict_for_config(
            preserve_dataframe=preserve_dataframe,
        )

        # Apply configuration defaults for manual_flexible method
        resolved_params = apply_config_defaults("manual_flexible", user_params)

        # Initialize parent with resolved parameters (never-configurable params passed as-is)
        # Manual flexible binning doesn't need guidance_columns
        FlexibleBinningBase.__init__(
            self,
            preserve_dataframe=resolved_params.get("preserve_dataframe"),
            guidance_columns=None,  # Not needed for unsupervised manual binning
            bin_spec=bin_spec,  # Required for manual flexible binning
            bin_representatives=bin_representatives,  # Never configurable
        )

    def fit(
        self, X: ArrayLike, y: ArrayLike | None = None, **fit_params: Any
    ) -> ManualFlexibleBinning:
        """Fit the Manual Flexible binning (no-op since bin specs are pre-defined).

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
        """Validate Manual Flexible binning parameters."""
        # ManualFlexibleBinning specific validation: bin_spec is required
        if self.bin_spec is None or len(self.bin_spec) == 0:
            raise ConfigurationError(
                "bin_spec must be provided and non-empty for ManualFlexibleBinning",
                suggestions=[
                    "Provide bin_spec as a dictionary: {column: [spec1, spec2, ...]}",
                    "Example: bin_spec={0: [1.5, (2, 5)], 1: [(0, 25), (25, 50)]}",
                ],
            )

        # Call parent validation for common checks
        FlexibleBinningBase._validate_params(self)

    def _calculate_flexible_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        guidance_data: np.ndarray[Any, Any] | None = None,
    ) -> tuple[FlexibleBinDefs, BinReps]:
        """Return pre-defined flexible bin specifications without calculation.

        Since ManualFlexibleBinning uses user-provided bin specifications, this method
        simply returns the pre-specified bins and representatives without performing
        any data-based calculations.

        Args:
            x_col: Input data column (ignored in manual binning)
            col_id: Column identifier to retrieve pre-defined bin specifications
            guidance_data: Not used for manual flexible binning

        Returns:
            Tuple of (bin_specs, bin_representatives)

        Raises:
            BinningError: If no bin specifications are defined for the specified column
        """
        raise NotImplementedError(
            "Manual binning uses pre-defined specifications. "
            "_calculate_bins should never be called for ManualIntervalBinning."
        )
