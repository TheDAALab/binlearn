"""
Clean Gaussian Mixture binning implementation for  architecture.

This module provides GaussianMixtureBinning that inherits from IntervalBinningBase.
Uses Gaussian Mixture Model clustering to find natural probabilistic bin boundaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from ..base import IntervalBinningBase
from ..config import apply_config_defaults
from ..utils import (
    BinEdgesDict,
    ConfigurationError,
    apply_equal_width_fallback,
    create_param_dict_for_config,
    resolve_n_bins_parameter,
    validate_bin_number_for_calculation,
    validate_bin_number_parameter,
)
from ..utils._parameter_validation import validate_random_state


# pylint: disable=too-many-ancestors
class GaussianMixtureBinning(IntervalBinningBase):
    """Gaussian Mixture Model clustering-based binning implementation using clean architecture.

    Creates bins based on Gaussian Mixture Model (GMM) clustering of each feature. The bin
    edges are determined by the decision boundaries between mixture components, creating bins
    that represent natural probabilistic groupings in the data based on underlying Gaussian
    distributions.

    The GMM algorithm assumes the data can be modeled as a mixture of Gaussian distributions
    and finds the optimal parameters (means, covariances, weights) for each component. Bin
    boundaries are placed at the midpoints between adjacent component means, creating intervals
    that correspond to regions where different Gaussian components are most likely.

    This approach is particularly effective for data with multiple modes or natural clustering,
    as it can identify and separate these distributions automatically. Unlike k-means clustering,
    GMM provides probabilistic cluster assignments and can handle clusters of different shapes
    and densities.

    When GMM fitting fails (e.g., due to numerical issues or insufficient data), the algorithm
    automatically falls back to equal-width binning to ensure robust operation.

    This implementation follows the clean binlearn architecture with straight inheritance,
    dynamic column resolution, and parameter reconstruction capabilities.

    Args:
        n_components: Number of Gaussian components (mixture components) to fit. Controls
            the number of bins created. Can be an integer or a string expression like
            'sqrt', 'log2', etc. for dynamic calculation based on data size. If None,
            uses configuration default.
        random_state: Random seed for reproducible GMM fitting. Controls the random
            initialization of component parameters. If None, results may vary between
            runs due to random initialization.
        allow_fallback: Whether to fall back to equal-width binning when GMM fitting
            fails. If True (default), uses equal-width binning as fallback with a warning.
            If False, raises an error when GMM fails. If None, uses configuration default.
        clip: Whether to clip values outside the fitted range to the nearest bin edge.
            If None, uses configuration default.
        preserve_dataframe: Whether to preserve pandas DataFrame structure in transform
            operations. If None, uses configuration default.
        fit_jointly: Whether to fit all columns together (False for GMM - always fits
            columns independently). If None, uses configuration default.
        bin_edges: Pre-computed bin edges for reconstruction. Should not be provided
            during normal usage.
        bin_representatives: Pre-computed bin representatives for reconstruction.
            Should not be provided during normal usage.
        class_: Class name for reconstruction compatibility. Internal use only.
        module_: Module name for reconstruction compatibility. Internal use only.

    Attributes:
        n_components: Number of mixture components to fit
        random_state: Random seed for reproducible results
        allow_fallback: Whether to fall back to equal-width binning when needed
        allow_fallback: Whether to fall back to equal-width binning on failure

    Example:
        >>> import numpy as np
        >>> from binlearn.methods import GaussianMixtureBinning
        >>>
        >>> # Create sample data with multiple modes
        >>> np.random.seed(42)
        >>> data = np.concatenate([
        ...     np.random.normal(-2, 0.5, 200),   # First mode
        ...     np.random.normal(1, 0.8, 300),    # Second mode
        ...     np.random.normal(4, 0.3, 150)     # Third mode
        ... ])
        >>>
        >>> # Initialize GMM binning with 3 components
        >>> binner = GaussianMixtureBinning(n_components=3, random_state=42)
        >>>
        >>> # Fit and transform
        >>> X = data.reshape(-1, 1)
        >>> binner.fit(X)
        >>> X_binned = binner.transform(X)
        >>>
        >>> # Check identified components
        >>> print(f"Number of bins: {len(binner.bin_edges_[0]) - 1}")
        >>> print(f"Bin representatives: {binner.bin_representatives_[0]}")

    Note:
        - GMM is particularly effective for data with natural multimodal distributions
        - Component means become the bin representatives (centers of identified modes)
        - Bin boundaries are placed at midpoints between adjacent component means
        - Requires sufficient data points (at least n_components) per column
        - Falls back to equal-width binning if GMM fitting fails
        - Each column is processed independently (unsupervised approach)
        - Uses full covariance type for maximum flexibility in component shapes

    See Also:
        KMeansBinning: Alternative clustering-based binning with hard cluster assignments
        DBSCANBinning: Density-based clustering for irregularly shaped clusters
        EqualWidthBinning: Simple equal-width interval binning
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        n_components: int | str | None = None,
        random_state: int | None = None,
        allow_fallback: bool | None = None,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        fit_jointly: bool | None = None,
        *,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
        class_: (  # pylint: disable=unused-argument
            str | None
        ) = None,  # For reconstruction compatibility
        module_: (  # pylint: disable=unused-argument
            str | None
        ) = None,  # For reconstruction compatibility
    ):
        """Initialize Gaussian Mixture binning with clustering parameters.

        Sets up GMM-based binning with specified parameters. Applies configuration
        defaults for any unspecified parameters and validates the resulting configuration.

        Args:
            n_components: Number of Gaussian components (mixture components) to fit.
                Controls the number of bins created. Can be:
                - Integer: Exact number of components
                - String: Dynamic calculation expression ('sqrt', 'log2', etc.)
                Must be positive. If None, uses configuration default.
            random_state: Random seed for reproducible GMM fitting. Controls the
                random initialization of component parameters. Should be a non-negative
                integer. If None, results may vary between runs.
            allow_fallback: Whether to fall back to equal-width binning when GMM
                fitting fails. If True (default), uses equal-width binning as fallback
                with a warning. If False, raises an error when GMM fails. If None,
                uses configuration default.
            clip: Whether to clip transformed values outside the fitted range to the
                nearest bin edge. If None, uses configuration default.
            preserve_dataframe: Whether to preserve pandas DataFrame structure in
                transform operations. If None, uses configuration default.
            fit_jointly: Whether to fit all columns together. Always False for GMM
                as it processes columns independently. If None, uses configuration default.
            bin_edges: Pre-computed bin edges dictionary for reconstruction. Internal
                use only - should not be provided during normal initialization.
            bin_representatives: Pre-computed representatives dictionary for
                reconstruction. Internal use only.
            class_: Class name string for reconstruction compatibility. Internal use only.
            module_: Module name string for reconstruction compatibility. Internal use only.

        Example:
            >>> # Standard initialization with 5 components
            >>> binner = GaussianMixtureBinning(n_components=5, random_state=42)
            >>>
            >>> # Dynamic component count based on data size
            >>> binner = GaussianMixtureBinning(n_components='sqrt', random_state=123)
            >>>
            >>> # Use configuration defaults
            >>> binner = GaussianMixtureBinning()
            >>>
            >>> # Custom configuration with clipping
            >>> binner = GaussianMixtureBinning(
            ...     n_components=8,
            ...     random_state=42,
            ...     clip=True,
            ...     preserve_dataframe=True
            ... )

        Note:
            - Parameter validation occurs during initialization
            - Configuration defaults are applied for None parameters
            - The random_state parameter ensures reproducible results across runs
            - n_components can use dynamic expressions for adaptive bin counts
            - Reconstruction parameters should not be provided during normal usage
        """
        # Use standardized initialization pattern
        user_params = create_param_dict_for_config(
            n_components=n_components,
            random_state=random_state,
            allow_fallback=allow_fallback,
            clip=clip,
            preserve_dataframe=preserve_dataframe,
            fit_jointly=fit_jointly,
        )

        # Apply configuration defaults
        resolved_params = apply_config_defaults("gaussian_mixture", user_params)

        # Store method-specific parameters
        self.n_components = resolved_params.get("n_components", 10)
        self.random_state = resolved_params.get("random_state", None)
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
        """Validate Gaussian Mixture binning parameters."""
        # Call parent validation
        IntervalBinningBase._validate_params(self)

        # Validate n_components using centralized utility
        validate_bin_number_parameter(self.n_components, param_name="n_components")

        # Validate random_state parameter using centralized utility
        validate_random_state(self.random_state)

    def _calculate_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        guidance_data: np.ndarray[Any, Any] | None = None,
    ) -> tuple[list[float], list[float]]:
        """Calculate Gaussian Mixture Model clustering-based bins for a single column.

        Uses GMM clustering to find natural probabilistic groupings
        and creates bin boundaries at decision boundaries between components.

        Args:
            x_col: Preprocessed column data (from base class)
            col_id: Column identifier for error reporting
            guidance_data: Not used for GMM binning (unsupervised)

        Returns:
            Tuple of (bin_edges, bin_representatives)

        Raises:
            ValueError: If n_components is invalid or insufficient data for clustering
        """
        # Validate n_components for calculation
        validate_bin_number_for_calculation(self.n_components, param_name="n_components")

        resolved_n_components = resolve_n_bins_parameter(
            self.n_components, data_shape=(len(x_col), 1), param_name="n_components"
        )

        return self._create_gmm_bins(x_col, col_id, resolved_n_components)

    # pylint: disable=too-many-locals
    def _create_gmm_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        n_components: int,
    ) -> tuple[list[float], list[float]]:
        """Create Gaussian Mixture Model clustering-based bins.

        Args:
            x_col: Column data that may contain NaN/inf values
            col_id: Column identifier for error reporting
            n_components: Number of mixture components to create

        Returns:
            Tuple of (bin_edges, bin_representatives)

        Note:
            We need to filter out NaN/inf values before GMM fitting.
        """
        # Filter out NaN and infinite values for GMM fitting
        finite_mask = np.isfinite(x_col)
        x_col_clean = x_col[finite_mask]

        if len(x_col_clean) < n_components:
            raise ValueError(
                f"Column {col_id}: Insufficient finite values ({len(x_col_clean)}) "
                f"for {n_components} components. Need at least {n_components} values."
            )

        # Reshape data for GMM (expects 2D array)
        X_reshaped = x_col_clean.reshape(-1, 1)

        try:
            # Apply Gaussian Mixture Model clustering using safe sklearn call
            gmm = GaussianMixture(
                n_components=n_components, random_state=self.random_state, covariance_type="full"
            )
            gmm.fit(X_reshaped)

            # Get component means and sort them
            means = np.array(gmm.means_).flatten()
            sorted_indices = np.argsort(means)
            sorted_means = means[sorted_indices]

            # Check if GMM produced valid means (within data range)
            min_val, max_val = float(np.min(x_col_clean)), float(np.max(x_col_clean))

            # If any means are significantly outside the data range, fall back
            tolerance = 1e-10  # Small tolerance for floating point precision
            if np.any(sorted_means < min_val - tolerance) or np.any(
                sorted_means > max_val + tolerance
            ):
                raise ValueError(
                    f"GMM produced means outside data range: {sorted_means} not "
                    f"in [{min_val}, {max_val}]"
                )

            # Calculate component boundaries
            edges = [min_val]  # Start with data minimum

            # Create boundaries between adjacent components
            for i in range(len(sorted_means) - 1):
                boundary = (sorted_means[i] + sorted_means[i + 1]) / 2
                edges.append(float(boundary))

            edges.append(max_val)  # End with data maximum

            # Representatives are the component means
            reps = [float(mean) for mean in sorted_means]

            return edges, reps

        except (
            ValueError,
            RuntimeError,
            ConvergenceWarning,
        ) as e:
            # Check if fallback is allowed
            if not self.allow_fallback:
                raise ConfigurationError(
                    f"GMM fitting failed: {str(e)}",
                    suggestions=[
                        "Try reducing n_components",
                        "Increase sample size",
                        "Check data distribution",
                        "Set allow_fallback=True to enable equal-width fallback",
                    ],
                ) from e

            # Use standardized equal-width fallback
            return list(apply_equal_width_fallback(x_col_clean, n_components, "GMM")), [
                float(val)
                for val in np.linspace(np.min(x_col_clean), np.max(x_col_clean), n_components)
            ]
