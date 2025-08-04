"""K-means clustering-based binning transformer.

This module implements K-means binning, where continuous data is divided into bins
based on K-means clustering. The bin edges are determined by the midpoints between
adjacent cluster centroids, creating bins that naturally group similar values together.

Classes:
    KMeansBinning: Main transformer for K-means clustering-based binning operations.
"""

from typing import Any

import kmeans1d
import numpy as np

from ..base._interval_binning_base import IntervalBinningBase
from ..base._repr_mixin import ReprMixin
from ..utils.errors import ConfigurationError
from ..utils.types import BinEdgesDict


# pylint: disable=too-many-ancestors
class KMeansBinning(ReprMixin, IntervalBinningBase):
    """K-means clustering-based binning transformer.

    Creates bins based on K-means clustering of each feature. The bin edges are
    determined by finding the midpoints between adjacent cluster centroids, which
    naturally groups similar values together. This approach is particularly useful
    when the data has natural clusters or when you want bins that adapt to the
    data distribution.

    This transformer is sklearn-compatible and supports pandas/polars DataFrames.
    It includes advanced features like random state control, clipping, and
    comprehensive error handling.

    Attributes:
        n_bins (int): Number of bins (clusters) per feature.
        random_state (int, optional): Random seed for reproducible clustering.
        clip (bool, optional): Whether to clip values outside bin range.
        columns (list, optional): Specific columns to bin.
        guidance_columns (list, optional): Columns to exclude from binlearn.
        preserve_dataframe (bool): Whether to preserve DataFrame format.
        _bin_edges (dict): Computed bin edges after fitting.

    Example:
        >>> import numpy as np
        >>> from binlearn.methods import KMeansBinning
        >>> X = np.random.rand(100, 3)
        >>> binner = KMeansBinning(n_bins=5)
        >>> X_binned = binner.fit_transform(X)
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        n_bins: int = 10,
        random_state: int | None = None,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
        fit_jointly: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize KMeansBinning transformer.

        Creates a K-means clustering-based binning transformer that uses K-means
        clustering to find natural groupings in the data and creates bin edges
        at the midpoints between adjacent cluster centroids.

        Args:
            n_bins (int, optional): Number of bins (clusters) to create for each feature.
                Must be a positive integer. Defaults to 10.
            random_state (Optional[int], optional): Random seed for reproducible
                K-means clustering results. If None, clustering may produce
                different results on repeated runs. Defaults to None.
            clip (Optional[bool], optional): Whether to clip out-of-range values
                to the nearest bin edge. If None, uses global configuration.
                Defaults to None.
            preserve_dataframe (Optional[bool], optional): Whether to return
                DataFrames when input is DataFrame. If None, uses global
                configuration. Defaults to None.
            bin_edges (Optional[BinEdgesDict], optional): Pre-specified bin edges
                for each column. If provided, these edges are used instead of
                calculating from K-means clustering. Defaults to None.
            bin_representatives (Optional[BinEdgesDict], optional): Pre-specified
                representative values for each bin. If provided along with bin_edges,
                these representatives are used. Defaults to None.
            fit_jointly (Optional[bool], optional): Whether to fit parameters
                jointly across all columns using the same global clustering. If None,
                uses global configuration. Defaults to None.
            **kwargs: Additional arguments passed to parent IntervalBinningBase.

        Raises:
            ConfigurationError: If n_bins is not a positive integer or if random_state
                is not a valid integer.
            ImportError: If kmeans1d package is not available.

        Example:
            >>> # Basic usage with default parameters
            >>> binner = KMeansBinning(n_bins=5)

            >>> # With reproducible results
            >>> binner = KMeansBinning(n_bins=10, random_state=42)

            >>> # With pre-specified bin edges
            >>> edges = {0: [0, 25, 50, 75, 100]}
            >>> binner = KMeansBinning(bin_edges=edges)
        """

        # Store K-means specific parameters BEFORE calling super().__init__
        # because parent class calls _validate_params() which needs these attributes
        self.n_bins = n_bins
        self.random_state = random_state

        super().__init__(
            clip=clip,
            preserve_dataframe=preserve_dataframe,
            bin_edges=bin_edges,
            bin_representatives=bin_representatives,
            fit_jointly=fit_jointly,
            **kwargs,
        )

    def _calculate_bins(
        self, x_col: np.ndarray, col_id: Any, guidance_data: np.ndarray | None = None
    ) -> tuple[list[float], list[float]]:
        """Calculate K-means clustering-based bins for a single column or joint binning data.

        Computes bin edges and representatives for either a single feature (per-column
        fitting) or from all flattened data (joint fitting). Uses K-means clustering
        to find natural groupings and creates bin edges at midpoints between centroids.

        Args:
            x_col (np.ndarray): Data for binning. For per-column fitting, this is
                data for a single column with shape (n_samples,). For joint fitting,
                this is flattened data from all columns. May contain NaN values.
            col_id (Any): Column identifier (name or index) for error reporting
                and logging purposes. For joint fitting, this is typically the
                first column identifier.
            guidance_data (Optional[np.ndarray], optional): Guidance data for
                supervised binning. Not used in K-means binning as it's an
                unsupervised method. Defaults to None.

        Returns:
            Tuple[List[float], List[float]]: A tuple containing:
                - bin_edges (List[float]): List of bin edge values with length n_bins+1
                - bin_representatives (List[float]): List of representative values
                  (cluster centroids) with length n_bins

        Raises:
            ValueError: If n_bins is less than 1 or if the data contains insufficient
                non-NaN values for clustering.

        Note:
            - For per-column fitting: uses column-specific clustering
            - For joint fitting: uses global clustering from all flattened data
            - Handles all-NaN data by creating a default [0, 1] range
            - Guidance data is ignored as K-means binning is unsupervised
            - May create fewer than n_bins if data has insufficient unique values
        """
        if self.n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {self.n_bins}")

        return self._create_kmeans_bins(x_col, col_id, self.n_bins)

    # pylint: disable=too-many-locals
    def _create_kmeans_bins(
        self, x_col: np.ndarray, col_id: Any, n_bins: int
    ) -> tuple[list[float], list[float]]:
        """Create K-means clustering-based bins.

        Generates bin edges and representative values using K-means clustering
        to identify natural groupings in the data and creates bin boundaries
        at the midpoints between adjacent cluster centroids.

        Args:
            x_col (np.ndarray): Data to bin. May contain NaN values.
            col_id (Any): Column identifier for error reporting.
            n_bins (int): Number of bins (clusters) to create. Must be positive.

        Returns:
            Tuple[List[float], List[float]]: A tuple containing:
                - bin_edges (List[float]): List of n_bins+1 edge values that
                  define the bin boundaries based on cluster midpoints
                - bin_representatives (List[float]): List of n_bins representative
                  values (cluster centroids) that represent each bin

        Raises:
            ValueError: If data contains insufficient non-NaN values for clustering.

        Example:
            >>> data = np.array([1, 2, 3, 10, 11, 12, 20, 21, 22])
            >>> binner._create_kmeans_bins(data, 'col1', 3)
            ([1.0, 6.5, 16.0, 22.0], [2.0, 11.0, 21.0])

        Note:
            - Uses kmeans1d.cluster for 1D clustering
            - Handles constant data by adding small epsilon
            - May create fewer bins if data has insufficient unique values
            - Representatives are the actual cluster centroids
        """
        # Remove NaN values for clustering
        clean_data = x_col[~np.isnan(x_col)]

        if len(clean_data) == 0:
            # All NaN data - create default range
            edges_array = np.linspace(0.0, 1.0, n_bins + 1)
            edges = list(edges_array)
            reps = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]
            return edges, reps

        if len(clean_data) < n_bins:
            raise ValueError(
                f"Column {col_id}: Insufficient non-NaN values ({len(clean_data)}) "
                f"for {n_bins} clusters. Need at least {n_bins} values."
            )

        # Handle case where all values are the same
        if len(np.unique(clean_data)) == 1:
            # All data points are the same - create equal-width bins around the value
            value = clean_data[0]
            epsilon = 1e-8 if value != 0 else 1e-8
            edges_array = np.linspace(value - epsilon, value + epsilon, n_bins + 1)
            edges = list(edges_array)
            reps = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]
            return edges, reps

        # Handle case where we have fewer unique values than desired clusters
        unique_values = np.unique(clean_data)
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
            reps = list(sorted_values)

            return unique_edges, reps

        # Perform K-means clustering
        try:
            # Set random seed if specified
            if self.random_state is not None:
                np.random.seed(self.random_state)

            # Convert numpy array to list for kmeans1d compatibility
            data_list = clean_data.tolist()
            _, centroids = kmeans1d.cluster(data_list, n_bins)
        except Exception as e:
            raise ValueError(f"Column {col_id}: Error in K-means clustering: {e}") from e

        # Sort centroids to ensure proper ordering
        centroids = sorted(centroids)

        # Create bin edges as midpoints between adjacent centroids
        cluster_edges: list[float] = []

        # First edge: extend below the minimum centroid
        data_min: float = np.min(clean_data)
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
        data_max: float = np.max(clean_data)
        if centroids[-1] < data_max:
            cluster_edges.append(data_max)
        else:
            # Extend slightly above the last centroid
            edge_extension = (centroids[-1] - centroids[0]) * 0.05
            cluster_edges.append(centroids[-1] + edge_extension)

        return cluster_edges, centroids

    def _validate_params(self) -> None:
        """Validate parameters for sklearn compatibility and logical consistency.

        Performs comprehensive validation of all KMeansBinning parameters
        to ensure they meet the expected types, ranges, and logical constraints.
        This method provides early error detection and clear error messages
        for common configuration mistakes.

        Raises:
            ConfigurationError: If any parameter validation fails:
                - n_bins must be a positive integer
                - random_state must be a non-negative integer if provided

        Example:
            >>> # This will raise ConfigurationError
            >>> binner = KMeansBinning(n_bins=0)  # n_bins must be positive
            >>> binner = KMeansBinning(random_state=-1)  # random_state must be non-negative

        Note:
            - Called automatically during fit() for early error detection
            - Provides helpful suggestions in error messages
            - Focuses on parameter validation, not data validation
            - Part of sklearn-compatible parameter validation pattern
        """
        # Call parent validation first (handles bin edges and representatives)
        super()._validate_params()

        # Validate n_bins
        if not isinstance(self.n_bins, int) or self.n_bins < 1:
            raise ConfigurationError(
                "n_bins must be a positive integer",
                suggestions=["Set n_bins to a positive integer (e.g., n_bins=10)"],
            )

        # Validate random_state if provided
        if self.random_state is not None:
            if not isinstance(self.random_state, int) or self.random_state < 0:
                raise ConfigurationError(
                    "random_state must be a non-negative integer",
                    suggestions=[
                        "Set random_state to a non-negative integer (e.g., random_state=42)"
                    ],
                )
