"""Gaussian Mixture Model clustering-based binning transformer.

This module implements Gaussian Mixture binning, where continuous data is divided into bins
based on Gaussian Mixture Model clustering. The bin edges are determined by the decision
boundaries between adjacent mixture components, creating bins that naturally group similar
values together based on probabilistic clustering.

Classes:
    GaussianMixtureBinning: Main transformer for Gaussian Mixture clustering-based binning
        operations.
"""

from typing import Any

import numpy as np
from sklearn.mixture import GaussianMixture

from ..base._interval_binning_base import IntervalBinningBase
from ..base._repr_mixin import ReprMixin
from ..utils.errors import ConfigurationError
from ..utils.types import BinEdgesDict


# pylint: disable=too-many-ancestors
class GaussianMixtureBinning(ReprMixin, IntervalBinningBase):
    """Gaussian Mixture Model clustering-based binning transformer.

    Creates bins based on Gaussian Mixture Model (GMM) clustering of each feature. The bin edges are
    determined by finding the decision boundaries between adjacent mixture components, which
    naturally groups similar values together based on probabilistic clustering. This approach is
    particularly useful when the data has natural Gaussian-like clusters or when you want bins
    that adapt to the probabilistic structure of the data distribution.

    This transformer is sklearn-compatible and supports pandas/polars DataFrames.
    It includes advanced features like random state control, clipping, and
    comprehensive error handling.

    Attributes:
        n_components (int): Number of mixture components (bins) per feature.
        random_state (int, optional): Random seed for reproducible clustering.
        clip (bool, optional): Whether to clip values outside bin range.
        columns (list, optional): Specific columns to bin.
        guidance_columns (list, optional): Columns to exclude from binlearn.
        preserve_dataframe (bool): Whether to preserve DataFrame format.
        _bin_edges (dict): Computed bin edges after fitting.

    Example:
        >>> import numpy as np
        >>> from binlearn.methods import GaussianMixtureBinning
        >>> X = np.random.rand(100, 3)
        >>> binner = GaussianMixtureBinning(n_components=5)
        >>> X_binned = binner.fit_transform(X)
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        n_components: int = 10,
        random_state: int | None = None,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
        fit_jointly: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize GaussianMixtureBinning transformer.

        Creates a Gaussian Mixture Model clustering-based binning transformer that uses GMM
        clustering to find natural probabilistic groupings in the data and creates bin edges
        at the decision boundaries between mixture components.

        Args:
            n_components (int, optional): Number of mixture components (bins) to create for
                each feature. Must be a positive integer. Defaults to 10.
            random_state (Optional[int], optional): Random seed for reproducible
                GMM clustering results. If None, clustering may produce
                different results on repeated runs. Defaults to None.
            clip (Optional[bool], optional): Whether to clip out-of-range values
                to the nearest bin edge. If None, uses global configuration.
                Defaults to None.
            preserve_dataframe (Optional[bool], optional): Whether to return
                DataFrames when input is DataFrame. If None, uses global
                configuration. Defaults to None.
            bin_edges (Optional[BinEdgesDict], optional): Pre-specified bin edges
                for each column. If provided, these edges are used instead of
                calculating from GMM clustering. Defaults to None.
            bin_representatives (Optional[BinEdgesDict], optional): Pre-specified
                representative values for each bin. If provided along with bin_edges,
                these representatives are used. Defaults to None.
            fit_jointly (Optional[bool], optional): Whether to fit parameters
                jointly across all columns using the same global clustering. If None,
                uses global configuration. Defaults to None.
            **kwargs: Additional arguments passed to parent IntervalBinningBase.

        Raises:
            ConfigurationError: If n_components is not a positive integer or if random_state
                is not a valid integer.

        Example:
            >>> # Basic usage with default parameters
            >>> binner = GaussianMixtureBinning(n_components=5)

            >>> # With reproducible results
            >>> binner = GaussianMixtureBinning(n_components=10, random_state=42)

            >>> # With pre-specified bin edges
            >>> edges = {0: [0, 25, 50, 75, 100]}
            >>> binner = GaussianMixtureBinning(bin_edges=edges)
        """

        # Store GMM specific parameters BEFORE calling super().__init__
        # because parent class calls _validate_params() which needs these attributes
        self.n_components = n_components
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
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        guidance_data: np.ndarray[Any, Any] | None = None,
    ) -> tuple[list[float], list[float]]:
        """Calculate Gaussian Mixture Model clustering-based bins for a single column or
        joint binning data.

        Computes bin edges and representatives for either a single feature (per-column
        fitting) or from all flattened data (joint fitting). Uses GMM clustering
        to find natural probabilistic groupings and creates bin edges at decision boundaries
        between mixture components.

        Args:
            x_col (np.ndarray[Any, Any]): Data for binning. For per-column fitting, this is
                data for a single column with shape (n_samples,). For joint fitting,
                this is flattened data from all columns. May contain NaN values.
            col_id (Any): Column identifier (name or index) for error reporting
                and logging purposes. For joint fitting, this is typically the
                first column identifier.
            guidance_data (Optional[np.ndarray[Any, Any]], optional): Guidance data for
                supervised binning. Not used in GMM binning as it's an
                unsupervised method. Defaults to None.

        Returns:
            Tuple[List[float], List[float]]: A tuple containing:
                - bin_edges (List[float]): List of bin edge values with length n_components+1
                - bin_representatives (List[float]): List of representative values
                  (mixture component means) with length n_components

        Raises:
            ValueError: If n_components is less than 1 or if the data contains insufficient
                non-NaN values for clustering.

        Note:
            - For per-column fitting: uses column-specific clustering
            - For joint fitting: uses global clustering from all flattened data
            - Handles all-NaN data by creating a default [0, 1] range
            - Guidance data is ignored as GMM binning is unsupervised
            - May create fewer than n_components if data has insufficient unique values
        """
        if self.n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}")

        return self._create_gmm_bins(x_col, col_id, self.n_components)

    # pylint: disable=too-many-locals
    def _create_gmm_bins(
        self, x_col: np.ndarray[Any, Any], col_id: Any, n_components: int
    ) -> tuple[list[float], list[float]]:
        """Create Gaussian Mixture Model clustering-based bins.

        Generates bin edges and representative values using GMM clustering
        to identify natural probabilistic groupings in the data and creates bin boundaries
        at the decision boundaries between mixture components.

        Args:
            x_col (np.ndarray[Any, Any]): Data to bin. May contain NaN values.
            col_id (Any): Column identifier for error reporting.
            n_components (int): Number of mixture components to create. Must be positive.

        Returns:
            Tuple[List[float], List[float]]: A tuple containing:
                - bin_edges (List[float]): List of n_components+1 edge values that
                  define the bin boundaries based on component decision boundaries
                - bin_representatives (List[float]): List of n_components representative
                  values (mixture component means) that represent each bin

        Raises:
            ValueError: If data contains insufficient non-NaN values for clustering.

        Example:
            >>> data = np.array([1, 2, 3, 10, 11, 12, 20, 21, 22])
            >>> binner._create_gmm_bins(data, 'col1', 3)
            ([1.0, 6.5, 16.0, 22.0], [2.0, 11.0, 21.0])

        Note:
            - Uses sklearn.mixture.GaussianMixture for probabilistic clustering
            - Handles constant data by adding small epsilon
            - May create fewer bins if data has insufficient unique values
            - Representatives are the mixture component means
        """
        # Remove NaN values for clustering
        clean_data = x_col[~np.isnan(x_col)]

        if len(clean_data) == 0:
            # All NaN data - create default range
            edges_array = np.linspace(0.0, 1.0, n_components + 1)
            edges = list(edges_array)
            reps = [(edges[i] + edges[i + 1]) / 2 for i in range(n_components)]
            return edges, reps

        if len(clean_data) < n_components:
            raise ValueError(
                f"Column {col_id}: Insufficient non-NaN values ({len(clean_data)}) "
                f"for {n_components} components. Need at least {n_components} values."
            )

        # Handle case where all values are the same
        if len(np.unique(clean_data)) == 1:
            # All data points are the same - create equal-width bins around the value
            value = clean_data[0]
            epsilon = 1e-8 if value != 0 else 1e-8
            edges_array = np.linspace(value - epsilon, value + epsilon, n_components + 1)
            edges = list(edges_array)
            reps = [(edges[i] + edges[i + 1]) / 2 for i in range(n_components)]
            return edges, reps

        # Handle case where we have fewer unique values than desired components
        unique_values = np.unique(clean_data)
        if len(unique_values) < n_components:
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

        # Perform Gaussian Mixture Model clustering
        try:
            # Reshape data for sklearn (needs 2D array)
            data_2d = clean_data.reshape(-1, 1)

            # Create and fit GMM
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type="full",  # Always use "full" for maximum flexibility
                random_state=self.random_state,
            )
            gmm.fit(data_2d)

            # Get component means (representatives)
            means = np.array(gmm.means_).flatten()

        except Exception as e:
            raise ValueError(f"Column {col_id}: Error in GMM clustering: {e}") from e

        # Sort means to ensure proper ordering
        means_list = sorted(means.tolist())

        # Create bin edges as midpoints between adjacent means
        gmm_edges: list[float] = []

        # First edge: extend below the minimum mean
        data_min: float = np.min(clean_data)
        if means_list[0] > data_min:
            gmm_edges.append(data_min)
        else:
            # Extend slightly below the first mean
            edge_extension = (means_list[-1] - means_list[0]) * 0.05
            gmm_edges.append(means_list[0] - edge_extension)

        # Intermediate edges: midpoints between consecutive means
        for i in range(len(means_list) - 1):
            midpoint = (means_list[i] + means_list[i + 1]) / 2
            gmm_edges.append(midpoint)

        # Last edge: extend above the maximum mean
        data_max: float = np.max(clean_data)
        last_edge = self._calculate_gmm_last_edge(means_list, data_max)
        gmm_edges.append(last_edge)

        return gmm_edges, means_list

    def _calculate_gmm_last_edge(self, means: list[float], data_max: float) -> float:
        """Calculate the last bin edge for GMM based on means and data maximum.

        Args:
            means: List of sorted GMM component means
            data_max: Maximum value in the data

        Returns:
            float: The last bin edge value
        """
        if means[-1] < data_max:
            return data_max

        # Extend slightly above the last mean
        edge_extension = (means[-1] - means[0]) * 0.05
        return means[-1] + edge_extension

    def _validate_params(self) -> None:
        """Validate parameters for sklearn compatibility and logical consistency.

        Performs comprehensive validation of all GaussianMixtureBinning parameters
        to ensure they meet the expected types, ranges, and logical constraints.
        This method provides early error detection and clear error messages
        for common configuration mistakes.

        Raises:
            ConfigurationError: If any parameter validation fails:
                - n_components must be a positive integer
                - random_state must be a non-negative integer if provided

        Example:
            >>> # This will raise ConfigurationError
            >>> binner = GaussianMixtureBinning(n_components=0)  # n_components must be positive

        Note:
            - Called automatically during fit() for early error detection
            - Provides helpful suggestions in error messages
            - Focuses on parameter validation, not data validation
            - Part of sklearn-compatible parameter validation pattern
        """
        # Call parent validation first (handles bin edges and representatives)
        super()._validate_params()

        # Validate n_components
        if not isinstance(self.n_components, int) or self.n_components < 1:
            raise ConfigurationError(
                "n_components must be a positive integer",
                suggestions=["Set n_components to a positive integer (e.g., n_components=10)"],
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
