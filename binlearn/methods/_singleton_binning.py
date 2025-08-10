"""
Clean singleton binning implementation for  architecture.

This module provides SingletonBinning that inherits from FlexibleBinningBase.
Each unique value gets its own bin.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ..base import FlexibleBinningBase
from ..config import apply_config_defaults
from ..utils import (
    DataQualityWarning,
    create_param_dict_for_config,
)


# pylint: disable=too-many-ancestors
class SingletonBinning(FlexibleBinningBase):
    """Singleton binning implementation using clean architecture.

    Creates one bin for each unique value in the numeric data. This method preserves
    all distinct values by creating individual bins for each unique data point,
    making it ideal for discrete numeric variables or cases where no information
    loss is acceptable.

    Each unique numeric value becomes both a bin definition and its own representative.
    This creates a one-to-one mapping between unique values in the data and bin
    definitions, effectively implementing an identity transformation for unique values.

    The method only supports numeric data and automatically filters out NaN and
    infinite values during bin creation. When all values are invalid (NaN/inf),
    it falls back to creating a single default bin to maintain functionality.

    This is particularly useful for:
    - Discrete numeric variables (e.g., counts, ratings, categories encoded as numbers)
    - Preserving exact values in downstream processing
    - Creating lookup tables for categorical encoding
    - Cases where data aggregation is not desired

    This implementation follows the clean binlearn architecture with straight inheritance,
    dynamic column resolution, and parameter reconstruction capabilities.

    Args:
        preserve_dataframe: Whether to preserve pandas DataFrame structure in transform
            operations. If None, uses configuration default.
        fit_jointly: Whether to fit all columns together (True) or independently (False).
            For singleton binning, this typically doesn't affect results since each
            unique value gets its own bin regardless. If None, uses configuration default.
        bin_spec: Pre-computed flexible bin specification for reconstruction. Internal
            use only - should not be provided during normal initialization.
        bin_representatives: Pre-computed representatives dictionary for reconstruction.
            Internal use only.
        class_: Class name for reconstruction compatibility. Internal use only.
        module_: Module name for reconstruction compatibility. Internal use only.

    Example:
        >>> import numpy as np
        >>> from binlearn.methods import SingletonBinning
        >>>
        >>> # Create discrete numeric data
        >>> data = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5]).reshape(-1, 1)
        >>>
        >>> # Initialize singleton binning
        >>> binner = SingletonBinning()
        >>>
        >>> # Fit and transform
        >>> binner.fit(data)
        >>> data_binned = binner.transform(data)
        >>>
        >>> # Check unique bins created
        >>> print(f"Original unique values: {np.unique(data)}")
        >>> print(f"Bin specifications: {binner.bin_spec_[0]}")
        >>> print(f"Number of bins: {len(binner.bin_spec_[0])}")
        >>>
        >>> # Verify identity mapping
        >>> for i, unique_val in enumerate(np.unique(data)):
        ...     assert binner.bin_spec_[0][i] == unique_val
        ...     assert binner.bin_representatives_[0][i] == unique_val

    Note:
        - Creates exactly as many bins as there are unique values in the data
        - Only processes finite numeric values (filters out NaN and infinite values)
        - Representatives are identical to the bin definitions (unique values)
        - Preserves all information from original discrete numeric data
        - Falls back gracefully when all data is invalid (creates single default bin)
        - Does not require guidance data (unsupervised method)
        - Each column is processed independently

    See Also:
        ManualFlexibleBinning: User-defined flexible bin specifications
        EqualWidthBinning: Fixed-width interval binning for continuous data
        EqualFrequencyBinning: Equal-frequency interval binning
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        preserve_dataframe: bool | None = None,
        fit_jointly: bool | None = None,
        *,
        bin_spec: Any | None = None,  # FlexibleBinSpec
        bin_representatives: Any | None = None,  # BinEdgesDict
        class_: str | None = None,  # For reconstruction compatibility
        module_: str | None = None,  # For reconstruction compatibility
    ):
        """Initialize singleton binning with basic configuration options.

        Sets up singleton binning that creates one bin per unique value. This method
        has minimal configuration since its behavior is deterministic - each unique
        value gets its own bin.

        Args:
            preserve_dataframe: Whether to preserve pandas DataFrame structure in
                transform operations. If None, uses configuration default.
            fit_jointly: Whether to fit all columns together (True) or independently
                (False). For singleton binning, this typically doesn't change the
                result since each unique value gets its own bin regardless of other
                columns. If None, uses configuration default.
            bin_spec: Pre-computed flexible bin specification dictionary for
                reconstruction. Maps column identifiers to lists of unique values.
                Internal use only - should not be provided during normal initialization.
            bin_representatives: Pre-computed representatives dictionary for
                reconstruction. For singleton binning, this is identical to bin_spec.
                Internal use only.
            class_: Class name string for reconstruction compatibility. Internal use only.
            module_: Module name string for reconstruction compatibility. Internal use only.

        Example:
            >>> # Standard initialization with default settings
            >>> binner = SingletonBinning()
            >>>
            >>> # Preserve DataFrame structure
            >>> binner = SingletonBinning(preserve_dataframe=True)
            >>>
            >>> # Fit all columns together (though doesn't affect singleton results)
            >>> binner = SingletonBinning(fit_jointly=True, preserve_dataframe=True)

        Note:
            - Minimal configuration needed since behavior is deterministic
            - Configuration defaults are applied for None parameters
            - fit_jointly parameter exists for consistency but doesn't change results
            - Reconstruction parameters (bin_spec, bin_representatives, class_, module_)
              are used internally for object reconstruction and should not be provided
              during normal usage
            - No guidance_columns parameter since this is an unsupervised method
        """
        # Use standardized initialization pattern
        user_params = create_param_dict_for_config(
            preserve_dataframe=preserve_dataframe,
            fit_jointly=fit_jointly,
        )

        # Apply configuration defaults for singleton method
        params = apply_config_defaults("singleton", user_params)

        # Initialize parent with resolved config parameters (no guidance_columns for singleton binning)
        # Note: bin_spec and bin_representatives are never set from config
        FlexibleBinningBase.__init__(
            self,
            preserve_dataframe=params.get("preserve_dataframe"),
            fit_jointly=params.get("fit_jointly"),
            guidance_columns=None,  # Singleton binning doesn't use guidance
            bin_spec=bin_spec,
            bin_representatives=bin_representatives,
        )

    def _validate_params(self) -> None:
        """Validate singleton binning parameters."""
        # Call parent validation
        FlexibleBinningBase._validate_params(self)
        # No additional validation needed for singleton binning

    def _calculate_flexible_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        guidance_data: np.ndarray[Any, Any] | None = None,
    ) -> tuple[list[Any], list[Any]]:
        """Calculate singleton bins - one bin per unique value.

        Args:
            x_col: Column data to analyze
            col_id: Column identifier (unused for singleton)
            guidance_data: Optional guidance data (unused for singleton)

        Returns:
            Tuple of (unique_values, unique_values) - both are the same for singleton binning
        """
        # Find unique values, excluding NaN and inf
        mask_valid = np.isfinite(x_col)
        valid_data = x_col[mask_valid]

        if len(valid_data) == 0:
            # Handle case where all values are NaN/inf - create a minimal valid bin
            warnings.warn(
                f"Column {col_id} contains only NaN/inf values. Creating default bin.",
                DataQualityWarning,
                stacklevel=2,
            )
            unique_values = [0.0]  # Default fallback
        else:
            unique_values = np.unique(valid_data).tolist()

        # For singleton binning, representatives are the same as the unique values
        representatives = unique_values.copy()

        return unique_values, representatives
