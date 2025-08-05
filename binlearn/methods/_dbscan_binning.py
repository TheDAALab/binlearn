"""DBSCAN clustering-based binning transformer.

This module implements DBSCAN binning, where continuous data is divided into bins
based on DBSCAN (Density-Based Spatial Clustering of Applications with Noise) clustering.
The bin edges are determined by the natural cluster boundaries identified by DBSCAN,
creating bins that group together densely connected values while handling outliers.

Classes:
    DBSCANBinning: Main transformer for DBSCAN clustering-based binning operations.
"""

from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

from ..base._interval_binning_base import IntervalBinningBase
from ..base._repr_mixin import ReprMixin
from ..utils.errors import ConfigurationError
from ..utils.types import BinEdgesDict


# pylint: disable=too-many-ancestors
class DBSCANBinning(ReprMixin, IntervalBinningBase):
    """DBSCAN clustering-based binning transformer.

    Creates bins based on DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    clustering of each feature. The bin edges are determined by the natural cluster boundaries
    identified by DBSCAN, which naturally groups densely connected values together while
    treating isolated points as noise. This approach is particularly useful when the data
    has natural density-based clusters or when you want bins that adapt to the local
    density structure of the data distribution.

    This transformer is sklearn-compatible and supports pandas/polars DataFrames.
    It includes advanced features like epsilon and min_samples control, clipping, and
    comprehensive error handling.

    Attributes:
        eps (float): The maximum distance between two samples for one to be considered in the
            neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered
            as a core point.
        clip (bool, optional): Whether to clip values outside bin range.
        columns (list, optional): Specific columns to bin.
        guidance_columns (list, optional): Columns to exclude from binlearn.
        preserve_dataframe (bool): Whether to preserve DataFrame format.
        _bin_edges (dict): Computed bin edges after fitting.

    Example:
        >>> import numpy as np
        >>> from binlearn.methods import DBSCANBinning
        >>> X = np.random.rand(100, 3)
        >>> binner = DBSCANBinning(eps=0.1, min_samples=5)
        >>> X_binned = binner.fit_transform(X)
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        eps: float = 0.1,
        min_samples: int = 5,
        min_bins: int = 2,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
        fit_jointly: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize DBSCANBinning transformer.

        Creates a DBSCAN clustering-based binning transformer that uses DBSCAN
        clustering to find natural density-based groupings in the data and creates bin edges
        at the cluster boundaries.

        Args:
            eps (float, optional): The maximum distance between two samples for one to be
                considered in the neighborhood of the other. This is the most important
                DBSCAN parameter. Defaults to 0.1.
            min_samples (int, optional): The number of samples (or total weight) in a
                neighborhood for a point to be considered as a core point. This includes
                the point itself. Defaults to 5.
            min_bins (int, optional): Minimum number of bins to create. If DBSCAN produces
                fewer clusters, equal-width binning will be used as fallback. Defaults to 2.
            clip (Optional[bool], optional): Whether to clip out-of-range values
                to the nearest bin edge. If None, uses global configuration.
                Defaults to None.
            preserve_dataframe (Optional[bool], optional): Whether to return
                DataFrames when input is DataFrame. If None, uses global
                configuration. Defaults to None.
            bin_edges (Optional[BinEdgesDict], optional): Pre-specified bin edges
                for each column. If provided, these edges are used instead of
                calculating from DBSCAN clustering. Defaults to None.
            bin_representatives (Optional[BinEdgesDict], optional): Pre-specified
                representative values for each bin. If provided along with bin_edges,
                these representatives are used. Defaults to None.
            fit_jointly (Optional[bool], optional): Whether to fit parameters
                jointly across all columns using the same global clustering. If None,
                uses global configuration. Defaults to None.
            **kwargs: Additional arguments passed to parent IntervalBinningBase.

        Raises:
            ConfigurationError: If eps is not positive, if min_samples is not a positive integer,
                or if min_bins is not a positive integer.

        Example:
            >>> # Basic usage with default parameters
            >>> binner = DBSCANBinning(eps=0.1, min_samples=5)

            >>> # With stricter clustering requirements
            >>> binner = DBSCANBinning(eps=0.05, min_samples=10, min_bins=3)

            >>> # With pre-specified bin edges
            >>> edges = {0: [0, 25, 50, 75, 100]}
            >>> binner = DBSCANBinning(bin_edges=edges)
        """

        # Store DBSCAN specific parameters BEFORE calling super().__init__
        # because parent class calls _validate_params() which needs these attributes
        self.eps = eps
        self.min_samples = min_samples
        self.min_bins = min_bins

        super().__init__(
            clip=clip,
            preserve_dataframe=preserve_dataframe,
            bin_edges=bin_edges,
            bin_representatives=bin_representatives,
            fit_jointly=fit_jointly,
            **kwargs,
        )

    def _calculate_bins(
        self, x_col: np.ndarray[Any, Any], col_id: Any, guidance_data: np.ndarray[Any, Any] | None = None
    ) -> tuple[list[float], list[float]]:
        """Calculate DBSCAN clustering-based bins for a single column or joint binning data.

        Computes bin edges and representatives for either a single feature (per-column
        fitting) or from all flattened data (joint fitting). Uses DBSCAN clustering
        to find natural density-based groupings and creates bin edges at cluster boundaries.

        Args:
            x_col (np.ndarray[Any, Any]): Data for binning. For per-column fitting, this is
                data for a single column with shape (n_samples,). For joint fitting,
                this is flattened data from all columns. May contain NaN values.
            col_id (Any): Column identifier (name or index) for error reporting
                and logging purposes. For joint fitting, this is typically the
                first column identifier.
            guidance_data (Optional[np.ndarray[Any, Any]], optional): Guidance data for
                supervised binning. Not used in DBSCAN binning as it's an
                unsupervised method. Defaults to None.

        Returns:
            Tuple[List[float], List[float]]: A tuple containing:
                - bin_edges (List[float]): List of bin edge values
                - bin_representatives (List[float]): List of representative values
                  (cluster centers) that represent each bin

        Raises:
            ValueError: If the data contains insufficient non-NaN values for clustering.

        Note:
            - For per-column fitting: uses column-specific clustering
            - For joint fitting: uses global clustering from all flattened data
            - Handles all-NaN data by creating a default [0, 1] range
            - Guidance data is ignored as DBSCAN binning is unsupervised
            - Falls back to equal-width binning if DBSCAN produces too few clusters
        """
        return self._create_dbscan_bins(x_col, col_id)

    # pylint: disable=too-many-locals
    def _create_dbscan_bins(
        self, x_col: np.ndarray[Any, Any], col_id: Any
    ) -> tuple[list[float], list[float]]:
        """Create DBSCAN clustering-based bins.

        Generates bin edges and representative values using DBSCAN clustering
        to identify natural density-based groupings in the data and creates bin boundaries
        at the cluster boundaries.

        Args:
            x_col (np.ndarray[Any, Any]): Data to bin. May contain NaN values.
            col_id (Any): Column identifier for error reporting.

        Returns:
            Tuple[List[float], List[float]]: A tuple containing:
                - bin_edges (List[float]): List of edge values that
                  define the bin boundaries based on cluster boundaries
                - bin_representatives (List[float]): List of representative
                  values (cluster centers) that represent each bin

        Raises:
            ValueError: If data contains insufficient non-NaN values for clustering.

        Example:
            >>> data = np.array([1, 2, 3, 10, 11, 12, 20, 21, 22])
            >>> binner._create_dbscan_bins(data, 'col1')
            ([1.0, 6.5, 16.0, 22.0], [2.0, 11.0, 21.0])

        Note:
            - Uses sklearn.cluster.DBSCAN for density-based clustering
            - Handles constant data by falling back to equal-width binning
            - Falls back to equal-width binning if too few clusters are found
            - Representatives are the cluster centers (mean of cluster points)
        """
        # Remove NaN values for clustering
        clean_data = x_col[~np.isnan(x_col)]

        if len(clean_data) == 0:
            # All NaN data - create default range
            edges_array = np.linspace(0.0, 1.0, self.min_bins + 1)
            edges = list(edges_array)
            reps = [(edges[i] + edges[i + 1]) / 2 for i in range(self.min_bins)]
            return edges, reps

        if len(clean_data) < self.min_samples:
            # Insufficient data - fall back to equal-width binning
            data_min, data_max = np.min(clean_data), np.max(clean_data)
            if data_min == data_max:
                # Constant data
                epsilon = 1e-8 if data_min != 0 else 1e-8
                edges_array = np.linspace(data_min - epsilon, data_max + epsilon, self.min_bins + 1)
            else:
                edges_array = np.linspace(data_min, data_max, self.min_bins + 1)

            edges = list(edges_array)
            reps = [(edges[i] + edges[i + 1]) / 2 for i in range(self.min_bins)]
            return edges, reps

        # Handle case where all values are the same
        if len(np.unique(clean_data)) == 1:
            # All data points are the same - create equal-width bins around the value
            value = clean_data[0]
            epsilon = 1e-8 if value != 0 else 1e-8
            edges_array = np.linspace(value - epsilon, value + epsilon, self.min_bins + 1)
            edges = list(edges_array)
            reps = [(edges[i] + edges[i + 1]) / 2 for i in range(self.min_bins)]
            return edges, reps

        # Perform DBSCAN clustering
        try:
            # Reshape data for sklearn (needs 2D array)
            data_2d = clean_data.reshape(-1, 1)

            # Create and fit DBSCAN
            dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = dbscan.fit_predict(data_2d)

            # Get unique cluster labels (excluding noise points labeled as -1)
            unique_labels = np.unique(labels)
            cluster_labels = unique_labels[unique_labels >= 0]

            if len(cluster_labels) < self.min_bins:
                # Too few clusters found - fall back to equal-width binning
                data_min, data_max = np.min(clean_data), np.max(clean_data)
                edges_array = np.linspace(data_min, data_max, self.min_bins + 1)
                edges = list(edges_array)
                reps = [(edges[i] + edges[i + 1]) / 2 for i in range(self.min_bins)]
                return edges, reps

            # Calculate cluster centers (representatives)
            cluster_centers = []
            for label in cluster_labels:
                cluster_points = clean_data[labels == label]
                center = np.mean(cluster_points)
                cluster_centers.append(center)

        except Exception:
            # Fall back to equal-width binning on error
            data_min, data_max = np.min(clean_data), np.max(clean_data)
            edges_array = np.linspace(data_min, data_max, self.min_bins + 1)
            edges = list(edges_array)
            reps = [(edges[i] + edges[i + 1]) / 2 for i in range(self.min_bins)]
            return edges, reps

        # Sort cluster centers to ensure proper ordering
        cluster_centers = sorted(cluster_centers)

        # Create bin edges as midpoints between adjacent cluster centers
        dbscan_edges: list[float] = []

        # First edge: extend below the minimum center
        min_data_value: float = np.min(clean_data)
        first_edge = self._calculate_first_edge(cluster_centers, min_data_value)
        dbscan_edges.append(first_edge)

        # Intermediate edges: midpoints between consecutive centers
        for i in range(len(cluster_centers) - 1):
            midpoint = (cluster_centers[i] + cluster_centers[i + 1]) / 2
            dbscan_edges.append(midpoint)

        # Last edge: extend above the maximum center
        max_data_value: float = np.max(clean_data)
        last_edge = self._calculate_last_edge(cluster_centers, max_data_value)
        dbscan_edges.append(last_edge)

        return dbscan_edges, cluster_centers

    def _calculate_first_edge(self, cluster_centers: list[float], data_min: float) -> float:
        """Calculate the first bin edge based on cluster centers and data minimum.

        Args:
            cluster_centers: List of sorted cluster centers
            data_min: Minimum value in the data

        Returns:
            float: The first bin edge value
        """
        if cluster_centers[0] > data_min:
            return data_min

        # Extend slightly below the first center
        if len(cluster_centers) > 1:
            edge_extension = (cluster_centers[-1] - cluster_centers[0]) * 0.05
        else:
            edge_extension = abs(cluster_centers[0]) * 0.05 if cluster_centers[0] != 0 else 0.05
        return cluster_centers[0] - edge_extension

    def _calculate_last_edge(self, cluster_centers: list[float], data_max: float) -> float:
        """Calculate the last bin edge based on cluster centers and data maximum.

        Args:
            cluster_centers: List of sorted cluster centers
            data_max: Maximum value in the data

        Returns:
            float: The last bin edge value
        """
        if cluster_centers[-1] < data_max:
            return data_max

        # Extend slightly above the last center
        if len(cluster_centers) > 1:
            edge_extension = (cluster_centers[-1] - cluster_centers[0]) * 0.05
        else:
            edge_extension = abs(cluster_centers[-1]) * 0.05 if cluster_centers[-1] != 0 else 0.05
        return cluster_centers[-1] + edge_extension

    def _validate_params(self) -> None:
        """Validate parameters for sklearn compatibility and logical consistency.

        Performs comprehensive validation of all DBSCANBinning parameters
        to ensure they meet the expected types, ranges, and logical constraints.
        This method provides early error detection and clear error messages
        for common configuration mistakes.

        Raises:
            ConfigurationError: If any parameter validation fails:
                - eps must be a positive number
                - min_samples must be a positive integer
                - min_bins must be a positive integer

        Example:
            >>> # This will raise ConfigurationError
            >>> binner = DBSCANBinning(eps=0)  # eps must be positive
            >>> binner = DBSCANBinning(min_samples=0)  # min_samples must be positive

        Note:
            - Called automatically during fit() for early error detection
            - Provides helpful suggestions in error messages
            - Focuses on parameter validation, not data validation
            - Part of sklearn-compatible parameter validation pattern
        """
        # Call parent validation first (handles bin edges and representatives)
        super()._validate_params()

        # Validate eps
        if not isinstance(self.eps, int | float) or self.eps <= 0:
            raise ConfigurationError(
                "eps must be a positive number",
                suggestions=["Set eps to a positive number (e.g., eps=0.1)"],
            )

        # Validate min_samples
        if not isinstance(self.min_samples, int) or self.min_samples < 1:
            raise ConfigurationError(
                "min_samples must be a positive integer",
                suggestions=["Set min_samples to a positive integer (e.g., min_samples=5)"],
            )

        # Validate min_bins
        if not isinstance(self.min_bins, int) or self.min_bins < 1:
            raise ConfigurationError(
                "min_bins must be a positive integer",
                suggestions=["Set min_bins to a positive integer (e.g., min_bins=2)"],
            )
