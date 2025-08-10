"""
Clean K-means binning implementation for  architecture.

This module provides KMeansBinning that inherits from IntervalBinningBase.
Uses K-means clustering to find natural groupings and creates bins at cluster boundaries.
"""

from __future__ import annotations

import warnings
from typing import Any

import kmeans1d
import numpy as np

from ..base import IntervalBinningBase
from ..config import apply_config_defaults
from ..utils import (
    BinEdgesDict,
    ConfigurationError,
    apply_equal_width_fallback,
    create_param_dict_for_config,
    handle_insufficient_data_error,
    resolve_n_bins_parameter,
    validate_bin_number_for_calculation,
    validate_bin_number_parameter,
)


# pylint: disable=too-many-ancestors
class KMeansBinning(IntervalBinningBase):
    """K-means clustering-based binning implementation for natural data groupings.

    This class implements K-means binning, which uses K-means clustering to identify
    natural groupings in the data and creates bin boundaries at the midpoints between
    adjacent cluster centroids. This approach is data-adaptive and creates bins that
    reflect the underlying distribution of values, making it particularly effective
    for non-uniformly distributed data.

    K-means binning is particularly effective for:
    - Non-uniformly distributed data with natural clusters
    - Creating bins that preserve data density patterns
    - Multimodal distributions where clusters represent different modes
    - Cases where traditional equal-width or equal-frequency binning is inadequate

    Key Features:
    - Data-driven bin boundary selection based on clustering
    - Automatically adapts to the underlying data distribution
    - Creates bins with meaningful separation based on value similarity
    - Handles irregular data distributions better than fixed-interval methods
    - Support for flexible bin count specification (integer or string rules)

    Algorithm:
    1. Apply K-means clustering to each column independently to find n_bins centroids
    2. Sort the centroids in ascending order
    3. Create bin edges at the midpoints between consecutive centroids
    4. Add data range boundaries (min, max) as outer edges
    5. Use centroids as bin representatives

    Parameters:
        n_bins: Number of bins to create, or string specification for automatic
            calculation. Can be:
            - Integer: exact number of bins (and clusters) to create
            - 'sqrt': number of bins = sqrt(n_samples)
            - 'log2': number of bins = log2(n_samples)
            - 'sturges': Sturges' rule for histogram bins
            Default value can be configured globally via binlearn.config.
        allow_fallback: Whether to fall back to equal-width binning when K-means
            clustering fails or when data has insufficient variation. If True (default),
            uses equal-width binning as fallback with a warning. If False, raises an
            error when clustering fails. Default can be configured globally.

    Attributes:
        n_bins: Number of clusters/bins to create
        allow_fallback: Whether to fall back to equal-width binning when needed
        bin_edges_: Dictionary mapping column identifiers to lists of bin edges
            after fitting. Edges are positioned at midpoints between cluster centroids.
        bin_representatives_: Dictionary mapping column identifiers to lists
            of bin representatives (the cluster centroids).

    Example:
        >>> import numpy as np
        >>> from binlearn.methods import KMeansBinning
        >>>
        >>> # Multimodal data - mixture of two normal distributions
        >>> X1 = np.random.normal(2, 0.5, 500)    # First mode
        >>> X2 = np.random.normal(8, 0.5, 500)    # Second mode
        >>> X = np.concatenate([X1, X2]).reshape(-1, 1)
        >>>
        >>> binner = KMeansBinning(n_bins=4)
        >>> binner.fit(X)
        >>> X_binned = binner.transform(X)
        >>> # Bins naturally separate the two modes
        >>>
        >>> # Automatic bin count based on data size
        >>> binner_auto = KMeansBinning(n_bins='sqrt')
        >>> binner_auto.fit(X)  # Uses sqrt(1000) ≈ 32 bins
        >>>
        >>> # Irregular distribution
        >>> X_irregular = np.concatenate([
        ...     np.random.uniform(0, 2, 100),     # Uniform region
        ...     np.random.normal(5, 0.2, 800),   # Tight cluster
        ...     np.random.uniform(8, 10, 100)    # Another uniform region
        ... ]).reshape(-1, 1)
        >>> binner_adaptive = KMeansBinning(n_bins=6)
        >>> binner_adaptive.fit(X_irregular)  # Adapts to density variations

    Note:
        - Only works with numeric data - non-numeric columns will raise errors
        - Performance depends on the clustering quality and data separability
        - May create fewer effective bins if clusters are very close together
        - Requires the kmeans1d package for efficient 1D K-means clustering
        - Inherits clipping behavior and format preservation from IntervalBinningBase
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        n_bins: int | str | None = None,
        allow_fallback: bool | None = None,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        fit_jointly: bool | None = None,
        *,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
        class_: (
            str | None
        ) = None,  # For reconstruction compatibility  # pylint: disable=unused-argument
        module_: (
            str | None
        ) = None,  # For reconstruction compatibility  # pylint: disable=unused-argument
    ):
        """Initialize K-means binning."""
        # Use standardized initialization pattern
        user_params = create_param_dict_for_config(
            n_bins=n_bins,
            allow_fallback=allow_fallback,
            clip=clip,
            preserve_dataframe=preserve_dataframe,
            fit_jointly=fit_jointly,
        )

        # Apply configuration defaults
        resolved_params = apply_config_defaults("kmeans", user_params)

        # Store method-specific parameters
        self.n_bins = resolved_params.get("n_bins", 10)
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
        """Validate K-means binning parameters."""
        # Call parent validation
        IntervalBinningBase._validate_params(self)

        # Validate n_bins using centralized utility
        validate_bin_number_parameter(self.n_bins, param_name="n_bins")

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
        col_id: Any,  # pylint: disable=unused-argument
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
        # Check for insufficient data
        if len(x_col) < n_bins:
            raise handle_insufficient_data_error(len(x_col), n_bins, "KMeansBinning")

        # Handle case where all values are the same or very few unique values
        unique_values = np.unique(x_col)
        if len(unique_values) == 1:
            # All data points are the same - fallback to equal-width
            if not self.allow_fallback:
                raise ConfigurationError(
                    "All data values are identical - cannot create meaningful bins",
                    suggestions=[
                        "Provide data with more variation",
                        "Set allow_fallback=True to enable equal-width fallback",
                    ],
                )
            return (
                list(apply_equal_width_fallback(x_col, n_bins, "KMeans", warn_on_fallback=True)),
                [float(unique_values[0])] * n_bins,
            )

        if len(unique_values) < n_bins:
            # Fewer unique values than desired bins - fallback to equal-width
            if not self.allow_fallback:
                raise ConfigurationError(
                    f"Too few unique values ({len(unique_values)}) for {n_bins} bins",
                    suggestions=[
                        f"Reduce n_bins to {len(unique_values)} or fewer",
                        "Set allow_fallback=True to enable equal-width fallback",
                    ],
                )
            return list(
                apply_equal_width_fallback(x_col, n_bins, "KMeans", warn_on_fallback=True)
            ), [float(val) for val in np.linspace(unique_values[0], unique_values[-1], n_bins)]

        # Perform K-means clustering with error handling
        def kmeans_func(data: Any, n_clusters: int) -> list[float]:
            data_list = data.tolist()
            _, centroids = kmeans1d.cluster(data_list, n_clusters)
            return sorted(centroids)

        def fallback_func(data: Any, n_clusters: int) -> list[float]:
            return list(apply_equal_width_fallback(data, n_clusters, "KMeans"))

        try:
            if self.allow_fallback:
                # Try K-means with fallback to equal-width on failure
                try:
                    centroids = kmeans_func(x_col, n_bins)
                except Exception:
                    warnings.warn(
                        "KMeans failed with sklearn, using fallback: clustering error",
                        category=UserWarning,
                        stacklevel=3,
                    )
                    centroids = fallback_func(x_col, n_bins)
            else:
                # Don't use fallback - let exceptions propagate
                centroids = kmeans_func(x_col, n_bins)
        except (
            ValueError,
            RuntimeError,
            ImportError,
            Exception,
        ) as e:  # pylint: disable=broad-exception-caught
            # Only reached when allow_fallback=False
            raise ConfigurationError(
                f"K-means clustering failed: {str(e)}",
                suggestions=[
                    "Try reducing n_bins",
                    "Increase sample size",
                    "Check data distribution",
                    "Set allow_fallback=True to enable equal-width fallback",
                ],
            ) from e

        # Create bin edges as midpoints between adjacent centroids
        cluster_edges: list[float] = []

        # First edge: extend below the minimum centroid or use data min
        data_min: float = float(np.min(x_col))
        cluster_edges.append(data_min)

        # Intermediate edges: midpoints between consecutive centroids
        for i in range(len(centroids) - 1):
            midpoint = (centroids[i] + centroids[i + 1]) / 2
            cluster_edges.append(midpoint)

        # Last edge: extend above the maximum centroid or use data max
        data_max: float = float(np.max(x_col))
        cluster_edges.append(data_max)

        return cluster_edges, centroids
