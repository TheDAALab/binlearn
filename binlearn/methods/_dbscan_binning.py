"""
Clean DBSCAN binning implementation for  architecture.

This module provides DBSCANBinning that inherits from IntervalBinningBase.
Uses DBSCAN clustering to find natural density-based bin boundaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

from ..config import apply_config_defaults
from ..utils.errors import ConfigurationError
from ..utils.types import BinEdgesDict
from ..base._interval_binning_base import IntervalBinningBase


class DBSCANBinning(IntervalBinningBase):
    """DBSCAN clustering-based binning implementation using  architecture.

    Creates bins based on DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    clustering of each feature. The bin edges are determined by the natural cluster boundaries
    identified by DBSCAN, which naturally groups densely connected values together while
    treating isolated points as noise.

    This implementation follows the clean  architecture with straight inheritance,
    dynamic column resolution, and parameter reconstruction capabilities.
    """

    def __init__(
        self,
        eps: float | None = None,
        min_samples: int | None = None,
        min_bins: int | None = None,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        fit_jointly: bool | None = None,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
        class_: str | None = None,  # For reconstruction compatibility
        module_: str | None = None,  # For reconstruction compatibility
    ):
        """Initialize DBSCAN binning."""
        # Prepare user parameters for config integration (exclude never-configurable params)
        user_params = {
            "eps": eps,
            "min_samples": min_samples,
            "min_bins": min_bins,
            "clip": clip,
            "preserve_dataframe": preserve_dataframe,
            "fit_jointly": fit_jointly,
        }
        # Remove None values to allow config defaults to take effect
        user_params = {k: v for k, v in user_params.items() if v is not None}

        # Apply configuration defaults for dbscan method
        resolved_params = apply_config_defaults("dbscan", user_params)

        # Store method-specific parameters
        self.eps = resolved_params.get("eps", 0.1)
        self.min_samples = resolved_params.get("min_samples", 5)
        self.min_bins = resolved_params.get("min_bins", 2)

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
        """Validate DBSCAN binning parameters."""
        # Call parent validation
        IntervalBinningBase._validate_params(self)

        # Validate eps parameter
        if not isinstance(self.eps, (int, float)) or self.eps <= 0:
            raise ConfigurationError(
                "eps must be a positive number",
                suggestions=["Example: eps=0.1"],
            )

        # Validate min_samples parameter
        if not isinstance(self.min_samples, int) or self.min_samples <= 0:
            raise ConfigurationError(
                "min_samples must be a positive integer",
                suggestions=["Example: min_samples=5"],
            )

        # Validate min_bins parameter
        if not isinstance(self.min_bins, int) or self.min_bins < 1:
            raise ConfigurationError(
                "min_bins must be a positive integer",
                suggestions=["Example: min_bins=2"],
            )

    def _calculate_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        guidance_data: np.ndarray[Any, Any] | None = None,
    ) -> tuple[list[float], list[float]]:
        """Calculate DBSCAN clustering-based bins for a single column.

        Uses DBSCAN clustering to find natural density-based groupings
        and creates bin boundaries at cluster boundaries.

        Args:
            x_col: Preprocessed column data (from base class)
            col_id: Column identifier for error reporting
            guidance_data: Not used for DBSCAN binning (unsupervised)

        Returns:
            Tuple of (bin_edges, bin_representatives)

        Raises:
            ValueError: If insufficient data for clustering
        """
        return self._create_dbscan_bins(x_col, col_id)

    def _create_dbscan_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
    ) -> tuple[list[float], list[float]]:
        """Create DBSCAN clustering-based bins.

        Args:
            x_col: Preprocessed column data (no NaN/inf values)
            col_id: Column identifier for error reporting

        Returns:
            Tuple of (bin_edges, bin_representatives)

        Note:
            The data is already preprocessed by the base class, so we don't need
            to handle NaN/inf values or constant data here.
        """
        if len(x_col) < self.min_samples:
            raise ValueError(
                f"Column {col_id}: Insufficient values ({len(x_col)}) "
                f"for DBSCAN clustering. Need at least {self.min_samples} values."
            )

        # Reshape data for DBSCAN (expects 2D array)
        X_reshaped = x_col.reshape(-1, 1)

        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        cluster_labels = dbscan.fit_predict(X_reshaped)

        # Get unique clusters (excluding noise points labeled as -1)
        unique_clusters = np.unique(cluster_labels[cluster_labels != -1])

        if len(unique_clusters) < self.min_bins:
            # Fall back to equal-width binning if too few clusters
            return self._fallback_equal_width_bins(x_col, col_id)

        # Calculate cluster centers and boundaries
        cluster_centers = []
        cluster_boundaries = []

        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_data = x_col[cluster_mask]

            # Calculate cluster center
            center = float(np.mean(cluster_data))
            cluster_centers.append((center, np.min(cluster_data), np.max(cluster_data)))

        # Sort clusters by center
        cluster_centers.sort(key=lambda x: x[0])

        # Create bin edges from cluster boundaries
        edges = [cluster_centers[0][1]]  # Start with minimum of first cluster

        for i in range(len(cluster_centers) - 1):
            # Boundary between clusters is the midpoint
            boundary = (cluster_centers[i][2] + cluster_centers[i + 1][1]) / 2
            edges.append(boundary)

        edges.append(cluster_centers[-1][2])  # End with maximum of last cluster

        # Representatives are cluster centers
        reps = [center for center, _, _ in cluster_centers]

        return edges, reps

    def _fallback_equal_width_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
    ) -> tuple[list[float], list[float]]:
        """Fallback to equal-width binning when DBSCAN produces too few clusters.

        Args:
            x_col: Preprocessed column data
            col_id: Column identifier for error reporting

        Returns:
            Tuple of (bin_edges, bin_representatives)
        """
        min_val, max_val = float(np.min(x_col)), float(np.max(x_col))

        # Create equal-width bins
        edges = np.linspace(min_val, max_val, self.min_bins + 1)

        # Create representatives as bin centers
        reps = []
        for i in range(self.min_bins):
            rep = (edges[i] + edges[i + 1]) / 2
            reps.append(rep)

        return list(edges), reps
