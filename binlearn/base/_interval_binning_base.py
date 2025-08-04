"""
Interval binning base class with comprehensive edge and clipping support.

This module provides the foundational IntervalBinningBase class for all interval-based
binning transformers. It handles bin edge management, value clipping, joint/per-column
fitting strategies, and supports both guided and unguided binning approaches.

The class provides robust handling of out-of-range values with configurable clipping
behavior and comprehensive validation of bin edges and representatives.
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import Any

import numpy as np

from ..config import get_config
from ..utils.bin_operations import (
    create_bin_masks,
    default_representatives,
    validate_bin_edges_format,
    validate_bin_representatives_format,
    validate_bins,
)
from ..utils.constants import ABOVE_RANGE, BELOW_RANGE, MISSING_VALUE
from ..utils.data_handling import return_like_input
from ..utils.errors import (
    BinningError,
    ConfigurationError,
    DataQualityWarning,
)
from ..utils.types import BinEdges, BinEdgesDict, ColumnId, ColumnList, GuidanceColumns
from ._general_binning_base import GeneralBinningBase


# pylint: disable=too-many-ancestors,too-many-instance-attributes
class IntervalBinningBase(GeneralBinningBase):
    """Base class for interval-based binning methods with edge and clipping support.

    This abstract base class provides the foundation for all interval-based binning
    transformers such as equal-width, equal-frequency, and supervised binning methods.
    It handles bin edge computation, value clipping, and provides both joint and
    per-column fitting strategies.

    The class supports comprehensive out-of-range value handling with configurable
    clipping behavior, robust bin validation, and automatic representative value
    computation for inverse transformations.

    Args:
        clip (bool, optional): Whether to clip values outside bin ranges to nearest
            bin edges. If None, uses global configuration default.
        preserve_dataframe (bool, optional): Whether to preserve DataFrame format in output.
            If None, uses global configuration default.
        bin_edges (BinEdgesDict, optional): Pre-computed bin edges for each column.
            If provided, skips edge computation during fitting.
        bin_representatives (BinEdgesDict, optional): Pre-computed representative values
            for each bin, used in inverse transformation.
        fit_jointly (bool, optional): Whether to fit parameters jointly across all columns.
            If None, uses global configuration default.
        guidance_columns (GuidanceColumns, optional): Columns to use for guided binning.
            Cannot be used with fit_jointly=True.
        **kwargs: Additional arguments passed to GeneralBinningBase.

    Attributes:
        clip (bool): Whether values outside bin ranges are clipped.
        bin_edges_ (BinEdgesDict): Computed bin edges after fitting.
        bin_representatives_ (BinEdgesDict): Computed bin representatives after fitting.

    Example:
        >>> # This is an abstract class, use a concrete implementation
        >>> from binlearn.methods import EqualWidthBinning
        >>> binner = EqualWidthBinning(n_bins=5, clip=True)
        >>> X_binned = binner.fit_transform(X)
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
        fit_jointly: bool | None = None,
        guidance_columns: GuidanceColumns | None = None,
        **kwargs: Any,
    ):
        """Initialize the interval binning base class.

        Args:
            clip (bool, optional): Whether to clip values outside bin ranges to nearest
                bin edges. If None, uses global configuration default.
            preserve_dataframe (bool, optional): Whether to preserve DataFrame format in output.
                If None, uses global configuration default.
            bin_edges (BinEdgesDict, optional): Pre-computed bin edges for each column.
                If provided, skips edge computation during fitting.
            bin_representatives (BinEdgesDict, optional): Pre-computed representative values
                for each bin, used in inverse transformation.
            fit_jointly (bool, optional): Whether to fit parameters jointly across all columns.
                If None, uses global configuration default.
            guidance_columns (GuidanceColumns, optional): Columns to use for guided binning.
                Cannot be used with fit_jointly=True.
            **kwargs: Additional arguments passed to GeneralBinningBase parent class.
        """
        super().__init__(
            preserve_dataframe=preserve_dataframe,
            fit_jointly=fit_jointly,
            guidance_columns=guidance_columns,
            **kwargs,
        )

        # Load configuration defaults
        config = get_config()

        # Apply defaults from configuration
        if clip is None:
            clip = config.default_clip

        self.clip = clip

        # Store parameters as expected by sklearn
        self.bin_edges = bin_edges
        self.bin_representatives = bin_representatives

        # Working specifications (fitted or user-provided)
        self.bin_edges_: BinEdgesDict = {}
        self.bin_representatives_: BinEdgesDict = {}

        # Validate parameters early
        # This will also process any provided bins
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate parameters for interval binning.

        Performs comprehensive validation of all IntervalBinningBase parameters
        to ensure they meet the expected format and constraints. This includes
        validating bin edge format, ordering, and compatibility with representatives.
        Also processes any provided bins to enable transform without fit.

        Raises:
            ConfigurationError: If any parameter validation fails.

        Note:
            - Called automatically during initialization for early error detection
            - Can be overridden in subclasses for additional validation
            - Should only validate, not transform parameters
            - Also processes provided bin specifications
        """
        try:
            # Call parent validation first
            super()._validate_params()

            # Process and validate bin edges if provided
            if self.bin_edges is not None:
                validate_bin_edges_format(self.bin_edges)
                self.bin_edges_ = self.bin_edges

            # Process and validate bin representatives if provided
            if self.bin_representatives is not None:
                validate_bin_representatives_format(self.bin_representatives, self.bin_edges)
                self.bin_representatives_ = self.bin_representatives

                # Validate compatibility with bin_edges if both are provided
                if self.bin_edges is not None:
                    validate_bins(self.bin_edges, self.bin_representatives)
            elif self.bin_edges is not None and self.bin_edges_:
                # Generate default representatives for provided edges
                self.bin_representatives_ = {}
                for col, edges in self.bin_edges.items():
                    edges_list = list(edges)
                    self.bin_representatives_[col] = default_representatives(edges_list)

            # If we have complete specifications, mark as fitted
            if self.bin_edges is not None and self.bin_edges_ and self.bin_representatives_:
                # Validate the complete bins
                validate_bins(self.bin_edges_, self.bin_representatives_)
                # Mark as fitted since we have complete bin specifications
                self._fitted = True
                # Store columns for later reference
                self._original_columns = list(self.bin_edges_.keys())

                # Set sklearn attributes based on bin_edges and guidance_columns
                self._set_sklearn_attributes_from_specs()

        except ValueError as e:
            raise ConfigurationError(str(e)) from e

    def _set_sklearn_attributes_from_specs(self) -> None:
        """Set sklearn attributes (n_features_in_, feature_names_in_) from bin specifications.

        Derives the sklearn-compatible attributes from the provided bin specifications
        and guidance columns. This enables parameter transfer workflows where an instance
        is created with pre-computed bins.
        """
        if self.bin_edges is not None:
            # Get column names/indices from bin_edges
            binning_columns = list(self.bin_edges.keys())

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

    @property
    def bin_edges(self) -> BinEdgesDict | None:
        """Get the pre-provided bin edges parameter.

        Returns:
            BinEdgesDict or None: Pre-provided bin edges for each column, or None
                if no bin edges were provided during initialization.
        """
        return getattr(self, "_bin_edges_param", None)

    @bin_edges.setter
    def bin_edges(self, value: BinEdgesDict | None) -> None:
        """Set bin edges and update internal state.

        Args:
            value (BinEdgesDict or None): New bin edges to set. If None, clears
                the current bin edges parameter.

        Note:
            Setting bin edges resets the fitted state if the transformer was
            previously fitted, requiring a new fit call before transform.
        """
        self._bin_edges_param = value
        if hasattr(self, "_fitted") and self._fitted:
            self._fitted = False  # Reset fitted state when bin_edges change

    @property
    def bin_representatives(self) -> BinEdgesDict | None:
        """Get the pre-provided bin representatives parameter.

        Returns:
            BinEdgesDict or None: Pre-provided representative values for each bin,
                or None if no representatives were provided during initialization.
        """
        return getattr(self, "_bin_representatives_param", None)

    @bin_representatives.setter
    def bin_representatives(self, value: BinEdgesDict | None) -> None:
        """Set bin representatives and update internal state.

        Args:
            value (BinEdgesDict or None): New bin representatives to set. If None,
                clears the current bin representatives parameter.

        Note:
            Setting bin representatives resets the fitted state if the transformer
            was previously fitted, requiring a new fit call before transform.
        """
        self._bin_representatives_param = value
        if hasattr(self, "_fitted") and self._fitted:
            self._fitted = False  # Reset fitted state when bin_representatives change

    def _fit_per_column(
        self,
        X: np.ndarray,
        columns: ColumnList,
        guidance_data: np.ndarray | None = None,
        **fit_params: Any,
    ) -> IntervalBinningBase:
        """Fit bins per column with optional guidance data.

        Processes each column independently to calculate bin edges and representatives.
        Always performs fitting from the data, even if user-provided bin edges exist.
        User-provided specifications serve as parameters or starting points but do not
        skip the fitting process.

        Args:
            X (np.ndarray): Input data array with shape (n_samples, n_features).
            columns (ColumnList): List of column identifiers corresponding to X columns.
            guidance_data (np.ndarray, optional): Optional guidance data that can
                influence bin calculation. Defaults to None.
            **fit_params: Additional parameters passed to fitting methods.

        Returns:
            IntervalBinningBase: Self, for method chaining.

        Raises:
            ValueError: If per-column bin fitting fails.
            BinningError: If bin processing or validation fails.
        """
        try:
            # Handle potential duplicate guidance_data from fit_params
            # The positional guidance_data takes precedence over fit_params
            fit_params_clean = fit_params.copy()
            fit_params_clean.pop("guidance_data", None)  # Remove duplicate if present

            self._process_user_specifications(columns)

            # Always calculate bins from data for all columns
            # User-provided specifications are used as starting points or defaults
            self._calculate_bins_for_columns(X, columns, guidance_data)
            self._finalize_fitting()
            return self

        except Exception as e:
            if isinstance(e, BinningError):
                raise
            raise ValueError(f"Failed to fit per-column bins: {str(e)}") from e

    def _calculate_bins_for_columns(
        self, X: np.ndarray, columns: ColumnList, guidance_data: np.ndarray | None
    ) -> None:
        """Calculate bins for each column.

        Iterates through all columns and calculates bin edges and representatives.
        This method always calculates bins from the data, even if user-provided
        specifications exist (user specs serve as starting points or validation).

        Args:
            X (np.ndarray): Input data array with shape (n_samples, n_features).
            columns (ColumnList): List of column identifiers corresponding to X columns.
            guidance_data (np.ndarray, optional): Optional guidance data that can
                influence bin calculation for each column.
        """
        for i, col in enumerate(columns):
            self._calculate_bins_for_single_column(X, i, col, guidance_data)

    def _calculate_bins_for_single_column(
        self, X: np.ndarray, col_index: int, col: ColumnId, guidance_data: np.ndarray | None
    ) -> None:
        """Calculate bins for a single column.

        Extracts data for the specified column, validates it for quality issues,
        and calls the abstract _calculate_bins method to compute bin edges and
        representatives for this column. Always overwrites any existing bin
        specifications for this column.

        Args:
            X (np.ndarray): Input data array with shape (n_samples, n_features).
            col_index (int): Index of the column in the data array.
            col (ColumnId): Identifier for the column being processed.
            guidance_data (np.ndarray, optional): Optional guidance data that can
                influence bin calculation for this column.
        """
        col_data = X[:, col_index]

        # Check for all-NaN data and warn if needed
        self._validate_column_data(col_data, col, col_index)

        # Always calculate bins from data, overwriting any user-provided specs
        edges, reps = self._calculate_bins(col_data, col, guidance_data)
        self.bin_edges_[col] = edges
        self.bin_representatives_[col] = reps

    def _validate_column_data(self, col_data: np.ndarray, col: ColumnId, col_index: int) -> None:
        """Validate column data and issue warnings for all-NaN columns.

        Checks if the column contains only NaN values and issues a data quality
        warning if so. This helps identify potential data quality issues early
        in the binning process.

        Args:
            col_data (np.ndarray): Data array for the specific column being validated.
            col (ColumnId): Identifier for the column (name or index).
            col_index (int): Numeric index of the column in the data array.

        Warns:
            DataQualityWarning: If the column contains only NaN values.
        """
        if not np.all(np.isnan(col_data)):
            return

        # Create a more descriptive column reference
        if isinstance(col, int | np.integer):
            col_ref = f"column {col} (index {col_index})"
        else:
            col_ref = f"column '{col}'"

        warnings.warn(
            f"Data in {col_ref} contains only NaN values", DataQualityWarning, stacklevel=2
        )

    def _fit_jointly(self, X: np.ndarray, columns: ColumnList, **fit_params: Any) -> None:
        """Fit bins jointly across all columns.

        Computes bins using all data across all columns simultaneously, creating
        the same bin structure for every column. This is useful when you want
        consistent binning across features.

        Args:
            X (np.ndarray): Input data array with shape (n_samples, n_features).
            columns (ColumnList): List of column identifiers corresponding to X columns.
            **fit_params: Additional parameters passed to fitting methods.

        Raises:
            ValueError: If joint bin fitting fails.
            BinningError: If bin processing or validation fails.

        Warns:
            DataQualityWarning: If all data contains only NaN values.
        """
        try:
            self._process_user_specifications(columns)

            # Always perform joint binning calculation
            # For true joint binning, flatten all data together
            all_data = X.ravel()

            # Check if all data is NaN
            if np.all(np.isnan(all_data)):
                warnings.warn("All data contains only NaN values", DataQualityWarning, stacklevel=2)

            # Calculate bins once from all flattened data
            edges, reps = self._calculate_bins_jointly(all_data, columns)

            # Apply the same bins to all columns, overwriting any user-provided specs
            for col in columns:
                self.bin_edges_[col] = edges
                self.bin_representatives_[col] = reps

            self._finalize_fitting()

        except Exception as e:
            if isinstance(e, BinningError):
                raise
            raise ValueError(f"Failed to fit joint bins: {str(e)}") from e

    def _process_user_specifications(self, columns: ColumnList) -> None:
        """Process user-provided bin specifications.

        Validates and processes any pre-provided bin edges and representatives
        from the user. Ensures they are in the correct dictionary format and
        initializes internal storage structures.

        Args:
            columns (ColumnList): List of column identifiers, currently unused
                but kept for interface compatibility.

        Raises:
            ConfigurationError: If provided bin specifications are invalid or
                cannot be processed.
            BinningError: If bin validation fails.
        """
        _ = columns

        try:
            if self.bin_edges is not None:
                # Validate format but don't transform - store as-is
                validate_bin_edges_format(self.bin_edges)
                self.bin_edges_ = self.bin_edges
            else:
                self.bin_edges_ = {}

            if self.bin_representatives is not None:
                # Validate format but don't transform - store as-is
                validate_bin_representatives_format(self.bin_representatives, self.bin_edges)
                self.bin_representatives_ = self.bin_representatives
            else:
                self.bin_representatives_ = {}

        except Exception as e:
            raise ConfigurationError(f"Failed to process bin specifications: {str(e)}") from e

    def _finalize_fitting(self) -> None:
        """Finalize the fitting process.

        Completes the fitting process by generating default representatives for
        any bin edges that don't have explicit representatives, and validates
        the final bin specifications to ensure they are consistent and valid.

        Raises:
            BinningError: If bin validation fails or if the final bin
                specifications are inconsistent.
        """
        # Generate default representatives for any missing ones
        for col in self.bin_edges_:
            if col not in self.bin_representatives_:
                edges = self.bin_edges_[col]
                self.bin_representatives_[col] = default_representatives(edges)

        # Validate the bins
        validate_bins(self.bin_edges_, self.bin_representatives_)

    def _calculate_bins_jointly(
        self, all_data: np.ndarray, columns: ColumnList
    ) -> tuple[BinEdges, BinEdges]:
        """Calculate bins from all flattened data for joint binning.

        This method provides a default implementation for joint binning that
        falls back to the regular _calculate_bins method using the first column
        identifier. Subclasses can override this for more sophisticated joint
        binning strategies.

        Args:
            all_data (np.ndarray): Flattened data array containing all values
                from all columns concatenated together.
            columns (ColumnList): List of column identifiers for reference.

        Returns:
            Tuple[BinEdges, BinEdges]: A tuple containing (bin_edges, bin_representatives)
                that will be applied to all columns.

        Note:
            Default implementation falls back to regular _calculate_bins using first column.
        """
        return self._calculate_bins(all_data, columns[0] if columns else 0)

    @abstractmethod
    def _calculate_bins(
        self, x_col: np.ndarray, col_id: Any, guidance_data: np.ndarray | None = None
    ) -> tuple[BinEdges, BinEdges]:
        """Calculate bin edges and representatives for a column.

        Parameters
        ----------
        x_col : np.ndarray
            The data for the column being binned.
        col_id : Any
            The identifier for the column.
        guidance_data : Optional[np.ndarray], default=None
            Optional guidance data that can influence bin calculation.

        Returns
        -------
        Tuple[BinEdges, BinEdges]
            A tuple of (bin_edges, bin_representatives).

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Must be implemented by subclasses.")

    def _get_column_key(
        self, target_col: ColumnId, available_keys: ColumnList, col_index: int
    ) -> ColumnId:
        """Find the right key for a column, handling mismatches between fit and transform.

        Resolves column identifiers when there are mismatches between the column
        identifiers used during fit and transform. Tries direct matching first,
        then falls back to index-based matching.

        Args:
            target_col (ColumnId): The column identifier we're looking for.
            available_keys (ColumnList): List of available keys in the fitted bins.
            col_index (int): The index position of the column in the current data.

        Returns:
            ColumnId: The matching key from available_keys that corresponds to
                the target column.

        Raises:
            ValueError: If no matching bin specification can be found for the
                target column.
        """
        # Direct match
        if target_col in available_keys:
            return target_col

        # Try index-based fallback
        if col_index < len(available_keys):
            return available_keys[col_index]

        # No match found
        raise ValueError(f"No bin specification found for column {target_col} (index {col_index})")

    def _transform_columns(self, X: np.ndarray, columns: ColumnList) -> np.ndarray:
        """Transform columns to bin indices.

        Converts continuous values in each column to discrete bin indices using
        the fitted bin edges. Handles special cases like NaN values, out-of-range
        values (with optional clipping), and maintains consistent indexing.

        Args:
            X (np.ndarray): Input data array with shape (n_samples, n_features).
            columns (ColumnList): List of column identifiers corresponding to X columns.

        Returns:
            np.ndarray: Array of bin indices with same shape as input, where each
                value represents the bin index for the corresponding input value.
                Special values: MISSING_VALUE for NaN, BELOW_RANGE/ABOVE_RANGE
                for out-of-range values when clipping is disabled.
        """
        result = np.zeros(X.shape, dtype=int)
        available_keys = list(self.bin_edges_.keys())

        for i, col in enumerate(columns):
            # Find the right bin specification
            key = self._get_column_key(col, available_keys, i)
            edges = self.bin_edges_[key]

            # Transform this column
            col_data = X[:, i]
            bin_indices = np.digitize(col_data, edges) - 1

            # Handle special cases
            nan_mask = np.isnan(col_data)
            below_mask = (col_data < edges[0]) & ~nan_mask
            above_mask = (col_data >= edges[-1]) & ~nan_mask

            if self.clip:
                bin_indices = np.clip(bin_indices, 0, len(edges) - 2)
            else:
                bin_indices[below_mask] = BELOW_RANGE
                bin_indices[above_mask] = ABOVE_RANGE

            bin_indices[nan_mask] = MISSING_VALUE
            result[:, i] = bin_indices

        return result

    def _inverse_transform_columns(self, X: np.ndarray, columns: ColumnList) -> np.ndarray:
        """Inverse transform from bin indices to representative values.

        Converts bin indices back to continuous values using the fitted bin
        representatives. Handles special indices for missing values and
        out-of-range conditions appropriately.

        Args:
            X (np.ndarray): Array of bin indices with shape (n_samples, n_features).
            columns (ColumnList): List of column identifiers corresponding to X columns.

        Returns:
            np.ndarray: Array of representative values with same shape as input,
                where each bin index is replaced by its corresponding representative
                value. Special values: NaN for missing, -inf for below range,
                +inf for above range.
        """
        result = np.zeros(X.shape, dtype=float)
        available_keys = list(self.bin_representatives_.keys())

        for i, col in enumerate(columns):
            key = self._get_column_key(col, available_keys, i)
            reps = self.bin_representatives_[key]

            col_data = X[:, i]

            # Handle special values first
            nan_mask = col_data == MISSING_VALUE
            below_mask = col_data == BELOW_RANGE
            above_mask = col_data == ABOVE_RANGE

            # Everything else gets clipped to valid range and mapped
            regular_indices = ~nan_mask & ~below_mask & ~above_mask
            if regular_indices.any():
                clipped_indices = np.clip(col_data[regular_indices].astype(int), 0, len(reps) - 1)
                result[regular_indices, i] = np.array(reps)[clipped_indices]

            # Handle special values
            result[nan_mask, i] = np.nan
            result[below_mask, i] = -np.inf
            result[above_mask, i] = np.inf

        return result

    def inverse_transform(self, X: Any) -> Any:
        """Transform bin indices back to representative values.

        Converts discrete bin indices back to continuous representative values
        using the fitted bin representatives. This is the inverse operation of
        the transform method.

        Args:
            X (array-like): Input array of bin indices with shape (n_samples, n_columns).
                Can be numpy array, pandas DataFrame, or other array-like format.

        Returns:
            array-like: Array of representative values with the same shape and
                format as input. Format preservation depends on preserve_dataframe
                setting.

        Raises:
            NotFittedError: If the transformer has not been fitted yet.

        Example:
            >>> # Assuming a fitted transformer
            >>> bin_indices = [[0, 1], [2, 0]]
            >>> representatives = transformer.inverse_transform(bin_indices)
            >>> print(representatives)  # Representative values for each bin
        """
        self._check_fitted()
        arr, columns = self._prepare_input(X)
        result = self._inverse_transform_columns(arr, columns)
        return return_like_input(result, X, columns, self.preserve_dataframe)

    def lookup_bin_widths(self, bin_indices: Any) -> Any:
        """Look up bin widths for given bin indices.

        Returns the width (difference between upper and lower edge) of each bin
        corresponding to the provided bin indices. This is useful for understanding
        the size of each bin in the original data space.

        Args:
            bin_indices (array-like): Array of bin indices with shape (n_samples, n_columns).
                Can be numpy array, pandas DataFrame, or other array-like format.

        Returns:
            array-like: Array of bin widths with the same shape and format as input.
                Each value represents the width of the corresponding bin. Invalid
                bin indices result in zero width.

        Raises:
            NotFittedError: If the transformer has not been fitted yet.

        Example:
            >>> # Assuming a fitted transformer
            >>> bin_indices = [[0, 1], [2, 0]]
            >>> widths = transformer.lookup_bin_widths(bin_indices)
            >>> print(widths)  # Width of each bin
        """
        self._check_fitted()
        arr, columns = self._prepare_input(bin_indices)
        result = np.zeros(arr.shape, dtype=float)
        available_keys = list(self.bin_edges_.keys())

        for i, col in enumerate(columns):
            key = self._get_column_key(col, available_keys, i)
            edges = self.bin_edges_[key]

            col_data = arr[:, i]
            valid, _, _, _ = create_bin_masks(col_data, len(edges) - 1)

            if valid.any():
                valid_indices = np.clip(col_data[valid].astype(int), 0, len(edges) - 2)
                widths = np.array([edges[j + 1] - edges[j] for j in range(len(edges) - 1)])
                result[valid, i] = widths[valid_indices]

        return return_like_input(result, bin_indices, columns, self.preserve_dataframe)

    def lookup_bin_ranges(self) -> dict[ColumnId, int]:
        """Return number of bins for each column.

        Provides the count of bins (ranges) created for each column after fitting.
        This is useful for understanding the binning structure and for validation.

        Returns:
            Dict[ColumnId, int]: Dictionary mapping each column identifier to its
                number of bins. The number of bins equals len(edges) - 1 for each
                column.

        Raises:
            NotFittedError: If the transformer has not been fitted yet.

        Example:
            >>> # Assuming a fitted transformer
            >>> bin_counts = transformer.lookup_bin_ranges()
            >>> print(bin_counts)  # {'col1': 5, 'col2': 3}
        """
        self._check_fitted()
        return {col: len(edges) - 1 for col, edges in self.bin_edges_.items()}
