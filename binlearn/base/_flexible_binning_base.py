"""
Clean flexible binning base class for V2 architecture.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from ..utils import (
    BinEdgesDict,
    ColumnList,
    ConfigurationError,
    FlexibleBinSpec,
    transform_value_to_flexible_bin,
    validate_bin_representatives_format,
)
from ._general_binning_base import GeneralBinningBase


# pylint: disable=too-many-ancestors
class FlexibleBinningBase(GeneralBinningBase):
    """Base class for flexible binning methods that support mixed bin types.

    This class extends GeneralBinningBase to provide specialized functionality for
    flexible binning methods. Unlike traditional interval-based binning, flexible
    binning supports mixed bin types within the same feature, including singleton
    bins (exact value matching) and interval bins (range matching) in any combination.

    Flexible binning is particularly useful for:
    - Categorical features with numeric representations
    - Mixed data types requiring different binning strategies per value
    - Custom binning schemes that don't fit traditional interval patterns
    - Data with important singleton values that should be preserved exactly

    Key Features:
    - Mixed bin types: Combine singleton and interval bins in the same feature
    - Custom bin specifications: Define bins as either exact values or ranges
    - Automatic representative generation: Creates numeric representatives for mixed bins
    - Flexible transformation: Handles both numeric and non-numeric data appropriately

    Attributes:
        bin_spec: Dictionary mapping column identifiers to lists of flexible bin
            definitions. Each bin can be either a scalar (singleton) or tuple (interval).
        bin_representatives: Dictionary mapping column identifiers to lists of
            numeric representative values for each bin. Auto-generated if not provided.

    Example:
        >>> # Example of flexible bin specification
        >>> bin_spec = {
        ...     'mixed_feature': [
        ...         42,           # Singleton bin: exactly value 42
        ...         (10, 20),     # Interval bin: range [10, 20]
        ...         'special',    # Categorical singleton
        ...         (100, 200)    # Another interval
        ...     ]
        ... }

    Note:
        - This is an abstract base class - use concrete implementations like ManualFlexibleBinning
        - Bin representatives are automatically generated as midpoints for intervals and
          preserved values for singletons (with numeric conversion where possible)
        - Inherits all functionality from GeneralBinningBase including fit/transform interface
        - Subclasses must implement the abstract _do_fit_single_column method
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        preserve_dataframe: bool | None = None,
        fit_jointly: bool | None = None,
        guidance_columns: Any = None,
        *,
        bin_spec: FlexibleBinSpec | None = None,
        bin_representatives: BinEdgesDict | None = None,
    ):
        """Initialize flexible binning base class.

        Args:
            preserve_dataframe: Whether to preserve the original DataFrame format
                during transformation. If True, returns DataFrame when input is DataFrame.
                If False, returns numpy array. If None, uses the global configuration
                default from binlearn.config.preserve_dataframe.
            fit_jointly: Whether to fit all columns together using shared information.
                If True, performs joint fitting across columns. If False, fits each
                column independently. If None, uses method-specific default behavior.
            guidance_columns: Additional columns to use as guidance for binning decisions.
                These columns are not binned themselves but can influence the binning
                of other columns. Can be column names/indices or None for no guidance.
            bin_spec: Pre-defined flexible bin specification as a dictionary mapping
                column identifiers to lists of bin definitions. Each bin definition
                can be either a scalar value (singleton bin) or a tuple (interval bin).
                If provided, no fitting is performed and this specification is used directly.
            bin_representatives: Pre-defined representative values for each bin as a
                dictionary mapping column identifiers to lists of numeric values.
                Must match the structure of bin_spec if provided. If None, representatives
                are automatically generated from bin_spec.

        Raises:
            ConfigurationError: If bin specifications are invalid or incompatible.

        Example:
            >>> # Initialize with custom bin specification
            >>> bin_spec = {
            ...     'feature1': [10, (20, 30), 40],
            ...     'feature2': ['A', 'B', (1, 5)]
            ... }
            >>> binner = ConcreteFlexibleBinner(bin_spec=bin_spec)
            >>>
            >>> # Initialize for automatic fitting
            >>> binner = ConcreteFlexibleBinner(fit_jointly=True)

        Note:
            - When bin_spec is provided, the binning is pre-configured and fit() becomes a no-op
            - bin_representatives will be auto-generated if not provided with bin_spec
            - guidance_columns feature may not be supported by all flexible binning methods
            - All parameters are passed to the parent GeneralBinningBase constructor
        """
        # Initialize parent
        GeneralBinningBase.__init__(
            self,
            preserve_dataframe=preserve_dataframe,
            fit_jointly=fit_jointly,
            guidance_columns=guidance_columns,
        )

        # Store flexible-specific parameters
        self.bin_spec = bin_spec
        self.bin_representatives = bin_representatives

        # Working fitted attributes
        self.bin_spec_: FlexibleBinSpec = {}
        self.bin_representatives_: BinEdgesDict = {}

        # Initialize sklearn attributes to avoid W0201 warnings
        self._feature_names_in: list[Any] | None = None
        self._n_features_in: int = 0

        # Configure fitted attributes for the base class
        self._fitted_attributes = ["bin_spec_", "bin_representatives_"]

        # Validate parameters early
        self._validate_params()

    # pylint: disable=too-many-branches
    def _validate_params(self) -> None:
        """Validate flexible binning parameters and initialize fitted attributes.

        This method performs comprehensive validation of the flexible binning parameters
        and pre-processes any provided bin specifications. It handles auto-generation
        of bin representatives and sets up the fitted state when complete specifications
        are provided.

        Raises:
            ConfigurationError: If bin specifications are invalid, bin_spec is not a
                dictionary, or bin_representatives format is invalid.

        Note:
            - Calls parent class parameter validation first
            - Auto-generates numeric representatives for bin specifications when not provided
            - For singleton bins, uses the value itself (converted to float if possible)
            - For interval bins, uses the midpoint as the representative
            - Non-numeric singleton values get a placeholder representative (0.0)
            - Sets sklearn attributes automatically when complete specifications are provided
        """
        # Call parent validation
        GeneralBinningBase._validate_params(self)

        # Process provided bin specifications
        # pylint: disable=too-many-nested-blocks
        try:
            if self.bin_spec is not None:
                # For now, just check it's a dictionary
                if not isinstance(self.bin_spec, dict):
                    raise ValueError("bin_spec must be a dictionary")
                self.bin_spec_ = self.bin_spec

                if self.bin_representatives is not None:
                    validate_bin_representatives_format(self.bin_representatives)
                    self.bin_representatives_ = self.bin_representatives
                elif self.bin_spec_:
                    # For flexible binning, auto-generate proper numeric representatives
                    # bin_spec contains mixed values (singletons and tuples for intervals)
                    # but representatives must be all numeric
                    self.bin_representatives_ = {}
                    for col, spec in self.bin_spec_.items():
                        if isinstance(spec, list):
                            representatives = []
                            for spec_item in spec:
                                if isinstance(spec_item, tuple) and len(spec_item) == 2:
                                    # Interval bin: use midpoint as representative
                                    representatives.append(float((spec_item[0] + spec_item[1]) / 2))
                                elif not isinstance(spec_item, tuple):
                                    # Singleton bin: use the value itself as representative
                                    try:
                                        representatives.append(float(spec_item))
                                    except (ValueError, TypeError):
                                        # For non-numeric singleton bins, use a placeholder
                                        representatives.append(0.0)
                                else:
                                    # Fallback for unexpected formats
                                    representatives.append(0.0)
                            self.bin_representatives_[col] = representatives

                # If we have complete specifications, mark as fitted and set sklearn attributes
                if self.bin_spec_ and self.bin_representatives_:
                    self._set_sklearn_attributes_from_specs()

        except ValueError as e:
            raise ConfigurationError(str(e)) from e

    def _set_sklearn_attributes_from_specs(self) -> None:
        """Set sklearn-compatible attributes from flexible bin specifications.

        This method configures the sklearn-compatible attributes (_feature_names_in
        and _n_features_in) based on the provided bin specifications and guidance
        columns. It ensures compatibility with sklearn's metadata routing and
        feature inspection APIs.

        Note:
            - Extracts column identifiers from bin_spec_ as primary features
            - Adds guidance_columns to the feature list if provided
            - Handles both single guidance column and list of guidance columns
            - Avoids duplicate columns when guidance columns overlap with binning columns
            - Sets _feature_names_in to the complete list of input columns
            - Sets _n_features_in to the total number of expected input features
        """
        if self.bin_spec_ is not None:
            # Get column names/indices from bin_spec
            binning_columns = list(self.bin_spec_.keys())

            # Add guidance columns if specified
            all_features = binning_columns.copy()
            if self.guidance_columns is not None:
                guidance_cols = (
                    [self.guidance_columns]
                    if not isinstance(self.guidance_columns, list)
                    else self.guidance_columns
                )
                # Add guidance columns that aren't already in binning columns
                for col in guidance_cols:
                    if col not in all_features:
                        all_features.append(col)

            # Set sklearn attributes
            self._feature_names_in = all_features
            self._n_features_in = len(all_features)

    def _fit_per_column_independently(
        self,
        X: np.ndarray[Any, Any],
        columns: ColumnList,
        guidance_data: np.ndarray[Any, Any] | None = None,
        **fit_params: Any,
    ) -> None:
        """Fit flexible binning parameters independently for each column.

        This method implements independent column fitting for flexible binning,
        where each column is analyzed separately to determine its optimal bin
        structure. This is the default fitting strategy for most flexible
        binning methods.

        Args:
            X: Input data array with shape (n_samples, n_features) where each
                column will be fitted independently.
            columns: List of column identifiers corresponding to the columns in X.
                Used for bin specification keys and error messages.
            guidance_data: Optional guidance data array that can influence binning
                decisions but is not binned itself. Same guidance data is provided
                to all columns during fitting.
            **fit_params: Additional parameters passed to the underlying bin
                calculation method (_calculate_flexible_bins).

        Note:
            - Creates separate bin specifications for each column in self.bin_spec_
            - Creates separate representatives for each column in self.bin_representatives_
            - Validates that each column contains numeric data before processing
            - Delegates actual bin calculation to the abstract _calculate_flexible_bins method
            - Each column is processed independently without sharing information
        """
        self.bin_spec_ = {}
        self.bin_representatives_ = {}

        for i, col in enumerate(columns):
            x_col = X[:, i]

            # Validate numeric data
            self._validate_numeric_data(x_col, col)

            # Use the same guidance_data for all columns (not indexed per column)
            edges, representatives = self._calculate_flexible_bins(x_col, col, guidance_data)
            self.bin_spec_[col] = edges
            self.bin_representatives_[col] = representatives

    def _fit_jointly_across_columns(
        self, X: np.ndarray[Any, Any], columns: ColumnList, **fit_params: Any
    ) -> None:
        """Fit flexible binning parameters jointly across all columns.

        This method implements joint fitting for flexible binning, where information
        from all columns is considered together during bin determination. For most
        flexible binning methods, this defaults to independent fitting unless
        overridden by specific implementations.

        Args:
            X: Input data array with shape (n_samples, n_features) where all
                columns will be considered together during fitting.
            columns: List of column identifiers corresponding to the columns in X.
                Used for bin specification keys and error messages.
            **fit_params: Additional parameters passed to the underlying fitting logic.

        Note:
            - Default implementation delegates to _fit_per_column_independently
            - Subclasses can override this method to implement true joint fitting
            - Joint fitting might consider correlations or dependencies between columns
            - The choice between joint and independent fitting is controlled by fit_jointly
                parameter
        """
        # For flexible binning, joint fitting is typically the same as per-column fitting
        # unless overridden by specific implementations
        self._fit_per_column_independently(X, columns, None, **fit_params)

    def _transform_columns_to_bins(
        self, X: np.ndarray[Any, Any], columns: ColumnList
    ) -> np.ndarray[Any, Any]:
        """Transform data columns to bin indices using flexible bin mapping.

        This method performs the core transformation operation for flexible binning,
        mapping each value in the input data to its corresponding bin index. It
        handles both singleton and interval bins within the same column.

        Args:
            X: Input data array with shape (n_samples, n_features) to transform.
                Each column should correspond to a column that was fitted.
            columns: List of column identifiers corresponding to the columns in X.
                Used to match with the fitted bin specifications.

        Returns:
            Transformed array with same shape as input, containing integer bin indices.
            Values that don't match any bin are assigned MISSING_VALUE (-1).

        Raises:
            ValueError: If the number of input columns doesn't match the number of
                fitted bin specifications.

        Note:
            - Uses transform_value_to_flexible_bin utility for individual value transformation
            - Handles missing values and out-of-range values by assigning MISSING_VALUE (-1)
            - Each value is compared against all bin definitions for its column
            - For singleton bins, uses exact equality comparison
            - For interval bins, uses inclusive range comparison [start, end]
            - Returns the first matching bin index if multiple bins could match
        """
        if X.size == 0:
            return np.empty((X.shape[0], 0))

        # Validate that input has same number of columns as bin specifications
        if X.shape[1] != len(self.bin_spec_):
            raise ValueError(
                f"Input data has {X.shape[1]} columns but bin specifications "
                f"are provided for {len(self.bin_spec_)} columns"
            )

        result = np.empty_like(X, dtype=int)
        available_keys = list(self.bin_spec_.keys())

        for i, col in enumerate(columns):
            # Find the right bin specification - this will raise ValueError for missing columns
            key = self._get_column_key(col, available_keys, i)
            bin_defs = self.bin_spec_[key]

            # Transform this column
            col_data = X[:, i]

            for row_idx, value in enumerate(col_data):
                # Use utility function for transformation
                result[row_idx, i] = transform_value_to_flexible_bin(value, bin_defs)

        return result

    def _inverse_transform_bins_to_values(
        self, X: np.ndarray[Any, Any], columns: ColumnList
    ) -> np.ndarray[Any, Any]:
        """Transform bin indices back to representative values.

        This method performs the inverse transformation for flexible binning,
        mapping bin indices back to their corresponding representative values.
        It's used to reconstruct approximate original values from bin indices.

        Args:
            X: Input array with shape (n_samples, n_features) containing integer
                bin indices to be transformed back to representative values.
            columns: List of column identifiers corresponding to the columns in X.
                Used to match with the fitted bin representatives.

        Returns:
            Array with same shape as input, containing float representative values
            for each bin index. Invalid indices are clipped to valid range.

        Note:
            - Uses the bin_representatives_ fitted during training
            - Invalid bin indices (< 0 or >= n_bins) are clipped to valid range
            - For flexible binning, representatives are typically:
              - The original singleton value for singleton bins
              - The interval midpoint for interval bins
            - Output is always float type for consistency, even if original values were integers
            - Missing values (MISSING_VALUE indices) are clipped to first bin representative
        """
        if X.size == 0:
            return np.empty((X.shape[0], 0))

        result = np.empty_like(X, dtype=float)
        available_keys = list(self.bin_representatives_.keys())

        for i, col in enumerate(columns):
            # Get the right bin specification using column key resolution
            key = self._get_column_key(col, available_keys, i)
            representatives = np.array(self.bin_representatives_[key])
            bin_indices = X[:, i].astype(int)

            # Clip indices to valid range
            bin_indices = np.clip(bin_indices, 0, len(representatives) - 1)

            result[:, i] = representatives[bin_indices]

        return result

    @abstractmethod
    def _calculate_flexible_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        guidance_data: np.ndarray[Any, Any] | None = None,
    ) -> tuple[list[Any], list[Any]]:
        """Calculate flexible bin values and representatives for a column.

        For flexible binning, this typically identifies unique values or patterns
        rather than creating fixed intervals.

        Args:
            x_col: Column data to analyze
            col_id: Column identifier
            guidance_data: Optional guidance data for this column

        Returns:
            Tuple of (bin_values, representatives) where:
            - bin_values: List of values that define the bins
            - representatives: List of representative values for each bin
        """
