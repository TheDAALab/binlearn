"""
Common binning utilities that can be reused across different binning methods.

This module provides utility mixins with common implementations for binning
operations that many concrete binning methods can reuse.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..utils.flexible_bin_operations import (
    find_flexible_bin_for_value,
    generate_default_flexible_representatives,
)
from ..utils.types import ColumnList, FlexibleBinDefs


class EdgeBasedBinningMixin:
    """Mixin providing common implementations for edge-based binning methods.

    This mixin provides standard implementations of transform and inverse transform
    methods for binning methods that use bin edges (like equal-width, quantile, etc.).

    Requires:
        - self.bin_edges_ dict mapping column identifiers to numpy arrays of bin edges
        - self.n_bins attribute for the number of bins
    """

    def _transform_columns_to_bins(
        self, X: np.ndarray[Any, Any], columns: ColumnList
    ) -> np.ndarray[Any, Any]:
        """Transform columns to bin indices using stored bin edges.

        Args:
            X: Input data to transform.
            columns: Column identifiers.

        Returns:
            Transformed data with bin indices.
        """
        if X.ndim == 1 and len(columns) == 1:
            # Single column case
            col = columns[0]
            bin_edges = self.bin_edges_[col]
            binned = np.digitize(X, bin_edges) - 1
            return np.clip(binned, 0, len(bin_edges) - 2)

        # Multiple columns case
        result = np.zeros_like(X, dtype=int)
        for i, col in enumerate(columns):
            bin_edges = self.bin_edges_[col]
            binned = np.digitize(X[:, i], bin_edges) - 1
            result[:, i] = np.clip(binned, 0, len(bin_edges) - 2)

        return result

    def _inverse_transform_bins_to_values(
        self, X: np.ndarray[Any, Any], columns: ColumnList
    ) -> np.ndarray[Any, Any]:
        """Inverse transform using stored bin representatives.

        Args:
            X: Binned data to inverse transform (always 2D).
            columns: Column identifiers.

        Returns:
            Data with bin representative values.
        """
        # Always expect 2D arrays
        result = np.zeros_like(X, dtype=float)
        for i, col in enumerate(columns):
            representatives = self.bin_representatives_[col]
            result[:, i] = [representatives[int(bin_idx)] for bin_idx in X[:, i]]

        return result


class CenterBasedBinningMixin:
    """Mixin providing common implementations for center-based binning methods.

    This mixin provides standard implementations for binning methods that use
    cluster centers or centroids (like k-means, Gaussian mixture, etc.).

    Requires:
        - self.cluster_centers_ dict mapping column identifiers to numpy arrays of centers
        - self.bin_edges_ dict mapping column identifiers to numpy arrays of bin edges
    """

    def _transform_columns_to_bins(
        self, X: np.ndarray[Any, Any], columns: ColumnList
    ) -> np.ndarray[Any, Any]:
        """Transform columns to bin indices using bin edges.

        Args:
            X: Input data to transform (always 2D).
            columns: Column identifiers.

        Returns:
            Transformed data with bin indices.
        """
        # Always expect 2D arrays
        result = np.zeros_like(X, dtype=int)
        for i, col in enumerate(columns):
            bin_edges = self.bin_edges_[col]
            binned = np.digitize(X[:, i], bin_edges) - 1
            result[:, i] = np.clip(binned, 0, len(bin_edges) - 2)

        return result

    def _inverse_transform_bins_to_values(
        self, X: np.ndarray[Any, Any], columns: ColumnList
    ) -> np.ndarray[Any, Any]:
        """Inverse transform using cluster centers.

        Args:
            X: Binned data to inverse transform (always 2D).
            columns: Column identifiers.

        Returns:
            Data with cluster center values.
        """
        # Always expect 2D arrays
        result = np.zeros_like(X, dtype=float)
        for i, col in enumerate(columns):
            centers = self.cluster_centers_[col]
            result[:, i] = centers[X[:, i].astype(int)]

        return result


class BinningUtilsMixin:
    """Mixin providing common utility methods for binning implementations.

    This mixin contains helper methods that are commonly used across
    different binning implementations.
    """

    def _validate_n_bins(self, n_features: int) -> None:
        """Validate the n_bins parameter.

        Args:
            n_features: Number of features in the data.

        Raises:
            ValueError: If n_bins is invalid.
        """
        if not isinstance(self.n_bins, int) or self.n_bins <= 0:
            raise ValueError(f"n_bins must be a positive integer, got {self.n_bins}")

        if self.n_bins == 1:
            import warnings

            warnings.warn(
                "n_bins=1 will result in a single bin for all values", UserWarning, stacklevel=3
            )

    def _handle_constant_column(
        self, column_data: np.ndarray[Any, Any], column_id: Any
    ) -> tuple[np.ndarray[Any, Any], int]:
        """Handle the case where a column has constant values.

        Args:
            column_data: The column data array.
            column_id: Identifier for the column.

        Returns:
            Tuple of (bin_edges, actual_n_bins) for the constant column.
        """
        constant_value = column_data[0]
        bin_edges = np.array([constant_value - 0.5, constant_value + 0.5])

        import warnings

        warnings.warn(
            f"Column {column_id} has constant values ({constant_value}). " f"Creating single bin.",
            UserWarning,
            stacklevel=4,
        )

        return bin_edges, 1


class FlexibleBinningMixin:
    """Mixin providing common implementations for flexible binning methods.

    This mixin provides standard implementations for binning methods that use
    flexible bin definitions (both singleton and interval bins).

    Requires:
        - self.bin_spec_ dict mapping column identifiers to lists of FlexibleBinDefs
        - self.bin_representatives_ dict mapping column identifiers to representative values
    """

    def _transform_columns_to_bins(
        self, X: np.ndarray[Any, Any], columns: ColumnList
    ) -> np.ndarray[Any, Any]:
        """Transform columns to bin indices using flexible bin definitions.

        Args:
            X: Input data to transform (always 2D).
            columns: Column identifiers.

        Returns:
            Transformed data with bin indices.
        """
        # Always expect 2D arrays
        result = np.zeros_like(X, dtype=int)
        for i, col in enumerate(columns):
            bin_defs = self.bin_spec_[col]
            for j, value in enumerate(X[:, i]):
                result[j, i] = find_flexible_bin_for_value(value, bin_defs)

        return result

    def _inverse_transform_bins_to_values(
        self, X: np.ndarray[Any, Any], columns: ColumnList
    ) -> np.ndarray[Any, Any]:
        """Inverse transform using bin representatives.

        Args:
            X: Binned data to inverse transform (always 2D).
            columns: Column identifiers.

        Returns:
            Data with representative values.
        """
        # Always expect 2D arrays
        result = np.zeros_like(X, dtype=float)
        for i, col in enumerate(columns):
            representatives = self.bin_representatives_[col]
            result[:, i] = [representatives[int(bin_idx)] for bin_idx in X[:, i]]

        return result
