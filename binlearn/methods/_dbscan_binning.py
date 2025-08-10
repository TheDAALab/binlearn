"""
Clean DBSCAN binning implementation for  architecture.

This module provides DBSCANBinning that inherits from IntervalBinningBase.
Uses DBSCAN clustering to find natural density-based bin boundaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

from ..base import IntervalBinningBase
from ..config import apply_config_defaults
from ..utils import (
    BinEdgesDict,
    ConfigurationError,
    apply_equal_width_fallback,
    create_param_dict_for_config,
    safe_sklearn_call,
    validate_positive_integer,
    validate_positive_number,
)


# pylint: disable=too-many-ancestors
class DBSCANBinning(IntervalBinningBase):
    """DBSCAN clustering-based binning implementation using clean architecture.

    Creates bins based on DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    clustering of each feature. The bin edges are determined by the natural cluster boundaries
    identified by DBSCAN, which naturally groups densely connected values together while
    treating isolated points as noise.

    The DBSCAN algorithm finds dense regions in the data and creates natural groupings that
    respect the underlying data distribution. Unlike k-means or equal-width binning, DBSCAN
    does not assume any particular shape for clusters and can identify clusters of varying
    densities. The resulting bins correspond to naturally occurring dense regions in the data.

    When DBSCAN produces fewer clusters than the minimum required bins, the algorithm falls
    back to equal-width binning to ensure the minimum bin count is satisfied.

    This implementation follows the clean binlearn architecture with straight inheritance,
    dynamic column resolution, and parameter reconstruction capabilities.

    Args:
        eps: The maximum distance between two samples for them to be considered as in the
            same neighborhood. This is the key parameter that controls cluster density.
            Smaller values create more, smaller clusters. Larger values merge clusters
            together. If None, uses configuration default.
        min_samples: The minimum number of samples in a neighborhood for a point to be
            considered as a core point (including the point itself). Controls the minimum
            cluster size. If None, uses configuration default.
        min_bins: Minimum number of bins to create. If DBSCAN produces fewer clusters,
            falls back to equal-width binning. Must be at least 1. If None, uses
            configuration default.
        allow_fallback: Whether to fall back to equal-width binning when DBSCAN produces
            fewer clusters than min_bins. If True (default), uses equal-width binning as
            fallback with a warning. If False, raises an error when insufficient clusters
            are found. If None, uses configuration default.
        clip: Whether to clip values outside the fitted range to the nearest bin edge.
            If None, uses configuration default.
        preserve_dataframe: Whether to preserve pandas DataFrame structure in transform
            operations. If None, uses configuration default.
        fit_jointly: Whether to fit all columns together (False for DBSCAN - always
            fits columns independently). If None, uses configuration default.
        bin_edges: Pre-computed bin edges for reconstruction. Should not be provided
            during normal usage.
        bin_representatives: Pre-computed bin representatives for reconstruction.
            Should not be provided during normal usage.
        class_: Class name for reconstruction compatibility. Internal use only.
        module_: Module name for reconstruction compatibility. Internal use only.

    Attributes:
        eps: Maximum distance for neighborhood definition
        min_samples: Minimum samples for core point definition
        min_bins: Minimum number of bins to ensure
        allow_fallback: Whether to fall back to equal-width binning when needed

    Example:
        >>> import numpy as np
        >>> from binlearn.methods import DBSCANBinning
        >>>
        >>> # Create sample data with natural clusters
        >>> data = np.concatenate([
        ...     np.random.normal(0, 0.5, 100),    # First cluster
        ...     np.random.normal(5, 0.8, 150),    # Second cluster
        ...     np.random.normal(10, 0.3, 80)     # Third cluster
        ... ])
        >>>
        >>> # Initialize DBSCAN binning
        >>> binner = DBSCANBinning(eps=0.8, min_samples=10, min_bins=3)
        >>>
        >>> # Fit and transform
        >>> X = data.reshape(-1, 1)
        >>> binner.fit(X)
        >>> X_binned = binner.transform(X)
        >>>
        >>> # Check identified bins
        >>> print(f"Number of bins: {len(binner.bin_edges_[0]) - 1}")
        >>> print(f"Bin edges: {binner.bin_edges_[0]}")

    Note:
        - DBSCAN is particularly effective for data with natural density-based clusters
        - The eps parameter requires careful tuning based on data scale and density
        - Noise points (outliers) identified by DBSCAN are included in boundary bins
        - Falls back to equal-width binning if insufficient clusters are found
        - Each column is processed independently (unsupervised approach)
        - Requires at least min_samples finite values per column for clustering

    See Also:
        KMeansBinning: Alternative clustering-based binning with fixed cluster count
        EqualWidthBinning: Simple equal-width interval binning
        GaussianMixtureBinning: Probabilistic clustering-based binning
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        eps: float | None = None,
        min_samples: int | None = None,
        min_bins: int | None = None,
        allow_fallback: bool | None = None,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        fit_jointly: bool | None = None,
        *,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
        class_: str | None = None,  # For reconstruction compatibility
        module_: str | None = None,  # For reconstruction compatibility
    ):
        """Initialize DBSCAN binning with clustering parameters.

        Sets up DBSCAN clustering-based binning with specified parameters. Applies
        configuration defaults for any unspecified parameters and validates the
        resulting configuration.

        Args:
            eps: Maximum distance between two samples for neighborhood definition.
                Controls cluster density - smaller values create tighter, more numerous
                clusters. Must be positive. If None, uses configuration default.
            min_samples: Minimum number of samples in a neighborhood for a core point.
                Controls minimum cluster size and noise tolerance. Must be positive
                integer. If None, uses configuration default.
            min_bins: Minimum number of bins to ensure. If DBSCAN produces fewer
                clusters, falls back to equal-width binning. Must be at least 1.
                If None, uses configuration default.
            allow_fallback: Whether to fall back to equal-width binning when DBSCAN
                produces fewer clusters than min_bins. If True (default), uses equal-width
                binning as fallback with a warning. If False, raises an error when
                insufficient clusters are found. If None, uses configuration default.
            clip: Whether to clip transformed values outside the fitted range to the
                nearest bin edge. If None, uses configuration default.
            preserve_dataframe: Whether to preserve pandas DataFrame structure in
                transform operations. If None, uses configuration default.
            fit_jointly: Whether to fit all columns together. Always False for DBSCAN
                as it processes columns independently. If None, uses configuration default.
            bin_edges: Pre-computed bin edges dictionary for reconstruction. Internal
                use only - should not be provided during normal initialization.
            bin_representatives: Pre-computed representatives dictionary for
                reconstruction. Internal use only.
            class_: Class name string for reconstruction compatibility. Internal use only.
            module_: Module name string for reconstruction compatibility. Internal use only.

        Example:
            >>> # Standard initialization with custom parameters
            >>> binner = DBSCANBinning(eps=0.5, min_samples=8, min_bins=3)
            >>>
            >>> # Use configuration defaults
            >>> binner = DBSCANBinning()
            >>>
            >>> # Custom clustering with clipping enabled
            >>> binner = DBSCANBinning(
            ...     eps=1.2,
            ...     min_samples=15,
            ...     min_bins=4,
            ...     clip=True,
            ...     preserve_dataframe=True
            ... )

        Note:
            - Parameter validation occurs during initialization
            - Configuration defaults are applied for None parameters
            - Reconstruction parameters (bin_edges, bin_representatives, class_, module_)
              are used internally for object reconstruction and should not be provided
              during normal usage
            - The eps parameter is critical for DBSCAN performance and may require
              experimentation based on data characteristics
        """
        # Use standardized initialization pattern
        user_params = create_param_dict_for_config(
            eps=eps,
            min_samples=min_samples,
            min_bins=min_bins,
            allow_fallback=allow_fallback,
            clip=clip,
            preserve_dataframe=preserve_dataframe,
            fit_jointly=fit_jointly,
        )

        # Apply configuration defaults
        resolved_params = apply_config_defaults("dbscan", user_params)

        # Store method-specific parameters
        self.eps = resolved_params.get("eps", 0.1)
        self.min_samples = resolved_params.get("min_samples", 5)
        self.min_bins = resolved_params.get("min_bins", 2)
        self.allow_fallback = resolved_params.get("allow_fallback", True)

        # Initialize parent with resolved parameters
        IntervalBinningBase.__init__(
            self,
            clip=resolved_params.get("clip"),
            preserve_dataframe=resolved_params.get("preserve_dataframe"),
            fit_jointly=resolved_params.get("fit_jointly"),
            guidance_columns=None,
            bin_edges=bin_edges,
            bin_representatives=bin_representatives,
        )

    def _validate_params(self) -> None:
        """Validate DBSCAN binning parameters."""
        # Call parent validation
        IntervalBinningBase._validate_params(self)

        # Use standardized validation utilities
        validate_positive_number(self.eps, "eps", allow_zero=False)
        validate_positive_integer(self.min_samples, "min_samples")
        validate_positive_integer(self.min_bins, "min_bins")

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
            x_col: Column data that may contain NaN/inf values
            col_id: Column identifier for error reporting

        Returns:
            Tuple of (bin_edges, bin_representatives)
        """
        # Filter out NaN and infinite values for DBSCAN fitting
        finite_mask = np.isfinite(x_col)
        x_col_clean = x_col[finite_mask]

        if len(x_col_clean) < self.min_samples:
            raise ValueError(
                f"Column {col_id}: Insufficient finite values ({len(x_col_clean)}) "
                f"for DBSCAN clustering. Need at least {self.min_samples} values."
            )

        # Reshape data for DBSCAN (expects 2D array)
        X_reshaped = x_col_clean.reshape(-1, 1)

        # Apply DBSCAN clustering using safe sklearn call
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        cluster_labels = safe_sklearn_call(dbscan.fit_predict, X_reshaped)

        # Get unique clusters (excluding noise points labeled as -1)
        unique_clusters = np.unique(cluster_labels[cluster_labels != -1])

        if len(unique_clusters) < self.min_bins:
            # Check if fallback is allowed
            if not self.allow_fallback:
                raise ConfigurationError(
                    f"DBSCAN found only {len(unique_clusters)} clusters, but min_bins={self.min_bins}",
                    suggestions=[
                        f"Reduce min_bins to {len(unique_clusters)} or lower",
                        "Adjust eps parameter to find more clusters",
                        "Reduce min_samples parameter",
                        "Set allow_fallback=True to enable equal-width fallback",
                    ],
                )

            # Fall back to equal-width binning if too few clusters
            return list(apply_equal_width_fallback(x_col_clean, self.min_bins, "DBSCAN")), [
                float(val)
                for val in np.linspace(np.min(x_col_clean), np.max(x_col_clean), self.min_bins)
            ]

        # Calculate cluster centers and boundaries
        cluster_centers = []

        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_data = x_col_clean[cluster_mask]

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
