"""
Clean K-means binning implementation for  architecture.

This module provides KMeansBinning that inherits from IntervalBinningBase.
Uses K-means clustering to find natural groupings and creates bins at cluster boundaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import kmeans1d

from ..config import apply_config_defaults
from ..utils import ConfigurationError
from ..utils import (
    resolve_n_bins_parameter,
    validate_bin_number_for_calculation,
    validate_bin_number_parameter,
)
from ..utils import BinEdgesDict
from ..base import IntervalBinningBase


class KMeansBinning(IntervalBinningBase):
    """K-means clustering-based binning implementation using  architecture.

    Creates bins based on K-means clustering of each feature. The bin edges are
    determined by finding the midpoints between adjacent cluster centroids, which
    naturally groups similar values together.

    This implementation follows the clean  architecture with straight inheritance,
    dynamic column resolution, and parameter reconstruction capabilities.
    """

    def __init__(
        self,
        n_bins: int | str | None = None,
        random_state: int | None = None,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        fit_jointly: bool | None = None,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
        class_: str | None = None,  # For reconstruction compatibility
        module_: str | None = None,  # For reconstruction compatibility
    ):
        """Initialize K-means binning."""
        # Prepare user parameters for config integration (exclude never-configurable params)
        user_params = {
            "n_bins": n_bins,
            "random_state": random_state,
            "clip": clip,
            "preserve_dataframe": preserve_dataframe,
            "fit_jointly": fit_jointly,
        }
        # Remove None values to allow config defaults to take effect
        user_params = {k: v for k, v in user_params.items() if v is not None}

        # Apply configuration defaults for kmeans method
        resolved_params = apply_config_defaults("kmeans", user_params)

        # Store method-specific parameters
        self.n_bins = resolved_params.get("n_bins", 10)
        self.random_state = resolved_params.get("random_state", None)

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
        """Validate K-means binning parameters."""
        # Call parent validation
        IntervalBinningBase._validate_params(self)

        # Validate n_bins using centralized utility
        validate_bin_number_parameter(self.n_bins, param_name="n_bins")

        # Validate random_state parameter
        if self.random_state is not None:
            if not isinstance(self.random_state, int) or self.random_state < 0:
                raise ConfigurationError(
                    "random_state must be a non-negative integer or None",
                    suggestions=["Example: random_state=42"],
                )

    def _calculate_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        guidance_data: np.ndarray[Any, Any] | None = None,
    ) -> tuple[list[float], list[float]]:
        """Calculate K-means clustering-based bins for a single column.

        Uses K-means clustering to find natural groupings in the data
        and creates bin boundaries at midpoints between cluster centroids.

        Args:
            x_col: Preprocessed column data (from base class)
            col_id: Column identifier for error reporting
            guidance_data: Not used for K-means binning (unsupervised)

        Returns:
            Tuple of (bin_edges, bin_representatives)

        Raises:
            ValueError: If n_bins is invalid or insufficient data for clustering
        """
        # Validate n_bins for calculation
        validate_bin_number_for_calculation(self.n_bins, param_name="n_bins")

        resolved_n_bins = resolve_n_bins_parameter(
            self.n_bins, data_shape=(len(x_col), 1), param_name="n_bins"
        )

        return self._create_kmeans_bins(x_col, col_id, resolved_n_bins)

    def _create_kmeans_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        n_bins: int,
    ) -> tuple[list[float], list[float]]:
        """Create K-means clustering-based bins.

        Args:
            x_col: Preprocessed column data (no NaN/inf values)
            col_id: Column identifier for error reporting
            n_bins: Number of clusters/bins to create

        Returns:
            Tuple of (bin_edges, bin_representatives)

        Note:
            The data is already preprocessed by the base class, so we don't need
            to handle NaN/inf values or constant data here.
        """
        if len(x_col) < n_bins:
            raise ValueError(
                f"Column {col_id}: Insufficient values ({len(x_col)}) "
                f"for {n_bins} clusters. Need at least {n_bins} values."
            )

        # Handle case where all values are the same
        if len(np.unique(x_col)) == 1:
            # All data points are the same - create equal-width bins around the value
            value = float(x_col[0])
            epsilon = 1e-8 if value != 0 else 1e-8
            edges_array = np.linspace(value - epsilon, value + epsilon, n_bins + 1)
            edges = list(edges_array)
            reps = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]
            return edges, reps

        # Handle case where we have fewer unique values than desired clusters
        unique_values = np.unique(x_col)
        if len(unique_values) < n_bins:
            # Create bins around each unique value
            sorted_values = np.sort(unique_values)
            unique_edges: list[float] = []

            # First edge: extend slightly below minimum
            unique_edges.append(sorted_values[0] - (sorted_values[-1] - sorted_values[0]) * 0.01)

            # Intermediate edges: midpoints between consecutive unique values
            for i in range(len(sorted_values) - 1):
                mid = (sorted_values[i] + sorted_values[i + 1]) / 2
                unique_edges.append(mid)

            # Last edge: extend slightly above maximum
            unique_edges.append(sorted_values[-1] + (sorted_values[-1] - sorted_values[0]) * 0.01)

            # Representatives are the unique values themselves
            reps = [float(val) for val in sorted_values]

            return unique_edges, reps

        # Perform K-means clustering
        try:
            # Set random seed if specified
            if self.random_state is not None:
                np.random.seed(self.random_state)

            # Convert numpy array to list for kmeans1d compatibility
            data_list = x_col.tolist()
            _, centroids = kmeans1d.cluster(data_list, n_bins)
        except Exception as e:
            raise ValueError(f"Column {col_id}: Error in K-means clustering: {e}") from e

        # Sort centroids to ensure proper ordering
        centroids = sorted(centroids)

        # Create bin edges as midpoints between adjacent centroids
        cluster_edges: list[float] = []

        # First edge: extend below the minimum centroid
        data_min: float = float(np.min(x_col))
        if centroids[0] > data_min:
            cluster_edges.append(data_min)
        else:
            # Extend slightly below the first centroid
            edge_extension = (centroids[-1] - centroids[0]) * 0.05
            cluster_edges.append(centroids[0] - edge_extension)

        # Intermediate edges: midpoints between consecutive centroids
        for i in range(len(centroids) - 1):
            midpoint = (centroids[i] + centroids[i + 1]) / 2
            cluster_edges.append(midpoint)

        # Last edge: extend above the maximum centroid
        data_max: float = float(np.max(x_col))
        if centroids[-1] < data_max:
            cluster_edges.append(data_max)
        else:
            # Extend slightly above the last centroid
            edge_extension = (centroids[-1] - centroids[0]) * 0.05
            cluster_edges.append(centroids[-1] + edge_extension)

        return cluster_edges, centroids
