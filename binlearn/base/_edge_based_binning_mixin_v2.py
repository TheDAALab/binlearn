"""Edge-based binning mixin with comprehensive functionality.

Th        # Initialize fitted parameters
        self.bin_edges_: BinEdgesDict = {}
        self.bin_representatives_: BinEdgesDict = {}

        # Call parent constructor with all parameters (cooperative inheritance)
        super().__init__(
            bin_edges=bin_edges,
            bin_representatives=bin_representatives,
            clip=clip,
            **kwargs
        )module provides the EdgeBasedBinningMixin class that serves as a proper
base class for binning methods using explicit bin edges and representatives.
It includes constructor parameter handling, clipping logic, and comprehensive
inverse transformation with special value handling.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np

from ..config import get_config
from ..utils.constants import ABOVE_RANGE, BELOW_RANGE, MISSING_VALUE
from ..utils.data_handling import return_like_input
from ..utils.types import BinEdgesDict, ColumnList, GuidanceColumns
from ._general_binning_base_v2 import GeneralBinningBaseV2


class EdgeBasedBinningMixin(GeneralBinningBaseV2):
    """Base class for binning methods that use explicit bin edges and representatives.

    Provides comprehensive functionality for edge-based binning transformers where bins are
    defined by explicit edge values and have representative values for inverse
    transformation. This is a proper base class that handles constructor parameters,
    clipping logic, and special value handling.

    Features:
    - Constructor parameters for bin_edges, bin_representatives, and clip
    - Proper transform method with clipping support for out-of-range values
    - Comprehensive inverse_transform with special value handling (NaN, -inf, +inf)
    - Compatibility with sklearn parameter reconstruction workflows
    - Integration with V2 architecture

    Args:
        bin_edges (BinEdgesDict, optional): Pre-computed bin edges for each column.
            If provided, these edges are used instead of calculating from data.
        bin_representatives (BinEdgesDict, optional): Pre-computed representative values
            for each bin. If provided along with bin_edges, these representatives are used.
        clip (bool, optional): Whether to clip out-of-range values to the nearest bin edge.
            If None, uses global configuration default.
        **kwargs: Additional arguments passed to parent classes.
    """

    def __init__(
        self,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        fit_jointly: bool | None = None,
        guidance_columns: GuidanceColumns = None,
    ) -> None:
        """Initialize the edge-based binning base class."""
        # Load configuration defaults
        config = get_config()

        # Apply configuration defaults
        if clip is None:
            clip = config.default_clip

        # Store constructor parameters for sklearn compatibility
        self.bin_edges = bin_edges
        self.bin_representatives = bin_representatives
        self.clip = clip

        # Initialize fitted parameters
        self.bin_edges_: BinEdgesDict = {}
        self.bin_representatives_: BinEdgesDict = {}

        # Call parent constructor
        super().__init__(
            preserve_dataframe=preserve_dataframe,
            fit_jointly=fit_jointly,
            guidance_columns=guidance_columns,
        )

        # If fitted parameters were provided in constructor, set them and mark as fitted
        if bin_edges is not None:
            self._set_fitted_attributes(
                bin_edges_=bin_edges, bin_representatives_=bin_representatives or {}
            )

    def _transform_columns_to_bins(
        self, X: np.ndarray[Any, Any], columns: ColumnList
    ) -> np.ndarray[Any, Any]:
        """Transform columns to bin indices with clipping support.

        Converts continuous values in each column to discrete bin indices using
        the fitted bin edges. Handles special cases like NaN values, out-of-range
        values (with optional clipping), and maintains consistent indexing.

        Args:
            X: Input data array with shape (n_samples, n_features).
            columns: List of column identifiers corresponding to X columns.

        Returns:
            Array of bin indices with same shape as input, where each
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

    def _inverse_transform_bins_to_values(
        self, X: np.ndarray[Any, Any], columns: ColumnList
    ) -> np.ndarray[Any, Any]:
        """Inverse transform from bin indices to representative values with special value handling.

        Converts bin indices back to continuous values using the fitted bin
        representatives. Handles special indices for missing values and
        out-of-range conditions appropriately.

        Args:
            X: Array of bin indices with shape (n_samples, n_features).
            columns: List of column identifiers corresponding to X columns.

        Returns:
            Array of representative values with same shape as input,
            where each bin index is replaced by its corresponding representative
            value. Special values: NaN for missing, -inf for below range,
            +inf for above range.
        """
        result = np.zeros(X.shape, dtype=float)
        available_keys = list(self.bin_representatives_.keys())

        for i, col in enumerate(columns):
            # Find the right bin specification
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

    def _get_column_key(self, target_col: Any, available_keys: ColumnList, col_index: int) -> Any:
        """Get the appropriate key for looking up bin specifications.

        Handles column key resolution with fallback strategies for
        different column identifier formats (names vs indices).

        Args:
            target_col: The target column identifier to find.
            available_keys: List of available keys in bin specifications.
            col_index: Index position of the column.

        Returns:
            The key to use for bin specification lookup.

        Raises:
            ValueError: If no matching key can be found.
        """
        # First try exact match
        if target_col in available_keys:
            return target_col

        # Handle feature_N -> N mapping for numpy array inputs
        if isinstance(target_col, str) and target_col.startswith("feature_"):
            try:
                feature_index = int(target_col.split("_")[1])
                if feature_index in available_keys:
                    return feature_index
            except (ValueError, IndexError):
                pass

        # Handle N -> feature_N mapping
        if isinstance(target_col, int):
            feature_name = f"feature_{target_col}"
            if feature_name in available_keys:
                return feature_name

        # Try index-based fallback
        if col_index < len(available_keys):
            return available_keys[col_index]

        # No match found
        raise ValueError(f"No bin specification found for column {target_col} (index {col_index})")

    def get_bin_width(self, column: Any = None) -> Union[float, dict[Any, float]]:
        """Get the width of bins for a column or all columns.

        Args:
            column: Specific column to get width for. If None, returns all.

        Returns:
            Bin width(s).

        Raises:
            RuntimeError: If not fitted.
            ValueError: If column was not fitted.
        """
        # Check if fitted by verifying bin_edges_ exists and is not empty
        if not hasattr(self, "bin_edges_") or not self.bin_edges_:
            raise RuntimeError("This estimator is not fitted yet. Call 'fit' first.")

        if column is not None:
            if column not in self.bin_edges_:
                raise ValueError(f"Column {column} was not fitted")
            bin_edges = self.bin_edges_[column]
            return float(bin_edges[1] - bin_edges[0])

        # Return widths for all columns
        widths = {}
        for col, bin_edges in self.bin_edges_.items():
            widths[col] = float(bin_edges[1] - bin_edges[0])
        return widths
