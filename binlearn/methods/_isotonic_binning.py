"""
Clean Isotonic binning implementation for  architecture.

This module provides IsotonicBinning that inherits from SupervisedBinningBase.
Uses isotonic regression to find optimal cut points that preserve monotonic relationships.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ..base import SupervisedBinningBase
from ..config import apply_config_defaults
from ..utils import (
    BinEdgesDict,
    ConfigurationError,
    FittingError,
    create_param_dict_for_config,
    resolve_n_bins_parameter,
    validate_bin_number_parameter,
)


# pylint: disable=too-many-ancestors
class IsotonicBinning(SupervisedBinningBase):
    """Isotonic regression-based monotonic binning implementation using clean architecture.

    Creates bins using isotonic regression to find optimal cut points that preserve
    monotonic relationships between features and targets. The transformer fits an
    isotonic (monotonic, non-decreasing or non-increasing) function to the data and
    identifies significant changes in this function to determine bin boundaries.

    This method is particularly valuable for cases where domain knowledge suggests
    a monotonic relationship between features and targets, such as risk modeling,
    credit scoring, or any application where preserving order relationships is critical.
    The isotonic regression ensures that the average target values within bins
    maintain the specified monotonic relationship.

    The algorithm works by:
    1. Sorting data by feature values
    2. Fitting an isotonic regression model to preserve monotonicity
    3. Identifying cut points where significant changes occur in the fitted function
    4. Creating bins that respect both the monotonic constraint and the minimum samples requirement

    When insufficient variability is found in the fitted isotonic function, the algorithm
    creates a single bin or falls back to simple boundary definitions.

    This implementation follows the clean binlearn architecture with straight inheritance,
    dynamic column resolution, and parameter reconstruction capabilities.

    Args:
        max_bins: Maximum number of bins to create. Controls the granularity of binning.
            Can be an integer or a string expression like 'sqrt', 'log2', etc. for
            dynamic calculation based on data size. If None, uses configuration default.
        min_samples_per_bin: Minimum number of samples required per bin. Ensures
            statistical significance of bins. Must be positive integer. If None,
            uses configuration default.
        increasing: Whether to enforce increasing monotonicity (True) or decreasing
            monotonicity (False). True means higher feature values correspond to
            higher target values. If None, uses configuration default.
        y_min: Minimum value for the fitted isotonic function output. Clips the
            fitted values to be at least this value. If None, no minimum constraint.
        y_max: Maximum value for the fitted isotonic function output. Clips the
            fitted values to be at most this value. If None, no maximum constraint.
        min_change_threshold: Minimum relative change in fitted values required to
            create a new bin boundary. Controls sensitivity to function changes.
            Must be positive. If None, uses configuration default.
        clip: Whether to clip values outside the fitted range to the nearest bin edge.
            If None, uses configuration default.
        preserve_dataframe: Whether to preserve pandas DataFrame structure in transform
            operations. If None, uses configuration default.
        guidance_columns: Column specification for target/guidance data used in
            supervised binning. Can be column names, indices, or callable selector.
        bin_edges: Pre-computed bin edges for reconstruction. Should not be provided
            during normal usage.
        bin_representatives: Pre-computed bin representatives for reconstruction.
            Should not be provided during normal usage.
        class_: Class name for reconstruction compatibility. Internal use only.
        module_: Module name for reconstruction compatibility. Internal use only.

    Attributes:
        max_bins: Maximum number of bins to create
        min_samples_per_bin: Minimum samples required per bin
        increasing: Whether monotonicity is increasing or decreasing
        y_min: Minimum constraint for fitted values
        y_max: Maximum constraint for fitted values
        min_change_threshold: Threshold for significant changes in fitted function

    Example:
        >>> import numpy as np
        >>> from binlearn.methods import IsotonicBinning
        >>>
        >>> # Create data with monotonic relationship
        >>> np.random.seed(42)
        >>> X = np.random.uniform(0, 10, 1000).reshape(-1, 1)
        >>> # Target increases monotonically with some noise
        >>> y = 2 * X.flatten() + np.random.normal(0, 1, 1000)
        >>>
        >>> # Initialize isotonic binning
        >>> binner = IsotonicBinning(
        ...     max_bins=5,
        ...     min_samples_per_bin=50,
        ...     increasing=True,
        ...     min_change_threshold=0.05
        ... )
        >>>
        >>> # Fit with target data
        >>> binner.fit(X, y)
        >>> X_binned = binner.transform(X)
        >>>
        >>> # Check monotonic preservation
        >>> bin_means = []
        >>> for bin_idx in range(len(binner.bin_edges_[0]) - 1):
        ...     bin_mask = X_binned[:, 0] == bin_idx
        ...     bin_means.append(np.mean(y[bin_mask]))
        >>> print("Bin target means:", bin_means)  # Should be monotonically increasing

    Note:
        - Requires target/guidance data for supervised learning of monotonic relationships
        - Preserves monotonic relationship between features and average target values within bins
        - Particularly useful for risk modeling, scoring, and ranking applications
        - Handles constant features and insufficient variability gracefully
        - Each column is processed independently with its corresponding target data
        - The fitted isotonic models are stored and can be used for analysis

    See Also:
        Chi2Binning: Statistical significance-based supervised binning
        TreeBinning: Decision tree-based supervised binning
        SupervisedBinningBase: Base class for supervised binning methods

    References:
        Robertson, T., Wright, F. T., & Dykstra, R. L. (1988). Order Restricted Statistical
            Inference.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def __init__(
        self,
        max_bins: int | str | None = None,
        min_samples_per_bin: int | None = None,
        increasing: bool | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        min_change_threshold: float | None = None,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        guidance_columns: Any = None,
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
        """Initialize Isotonic binning with monotonicity and quality parameters.

        Sets up isotonic regression-based binning with specified parameters for
        monotonicity preservation and bin quality control. Applies configuration
        defaults for any unspecified parameters and validates the resulting configuration.

        Args:
            max_bins: Maximum number of bins to create per column. Controls granularity
                of the binning. Can be:
                - Integer: Exact maximum number of bins
                - String: Dynamic calculation expression ('sqrt', 'log2', etc.)
                Must be positive. If None, uses configuration default.
            min_samples_per_bin: Minimum number of samples required per bin. Ensures
                statistical reliability of each bin. Must be positive integer.
                If None, uses configuration default.
            increasing: Whether to enforce increasing monotonicity (True) or decreasing
                monotonicity (False). True means higher feature values should correspond
                to higher average target values. If None, uses configuration default.
            y_min: Minimum value constraint for the fitted isotonic function output.
                Clips fitted values to be at least this value. Must be numeric.
                If None, no minimum constraint is applied.
            y_max: Maximum value constraint for the fitted isotonic function output.
                Clips fitted values to be at most this value. Must be numeric and
                greater than y_min if both are specified. If None, no maximum constraint.
            min_change_threshold: Minimum relative change in fitted values required
                to create a new bin boundary. Controls sensitivity to changes in the
                isotonic function. Must be positive float. If None, uses configuration default.
            clip: Whether to clip transformed values outside the fitted range to the
                nearest bin edge. If None, uses configuration default.
            preserve_dataframe: Whether to preserve pandas DataFrame structure in
                transform operations. If None, uses configuration default.
            guidance_columns: Column specification for target/guidance data. Can be
                column names, indices, or callable selector. Required for supervised
                binning during fit operations.
            bin_edges: Pre-computed bin edges dictionary for reconstruction. Internal
                use only - should not be provided during normal initialization.
            bin_representatives: Pre-computed representatives dictionary for
                reconstruction. Internal use only.
            class_: Class name string for reconstruction compatibility. Internal use only.
            module_: Module name string for reconstruction compatibility. Internal use only.

        Example:
            >>> # Standard initialization for increasing monotonic relationship
            >>> binner = IsotonicBinning(
            ...     max_bins=8,
            ...     min_samples_per_bin=30,
            ...     increasing=True,
            ...     min_change_threshold=0.02
            ... )
            >>>
            >>> # Decreasing monotonic relationship with value constraints
            >>> binner = IsotonicBinning(
            ...     max_bins=6,
            ...     min_samples_per_bin=50,
            ...     increasing=False,
            ...     y_min=0.0,
            ...     y_max=1.0,
            ...     guidance_columns=['risk_score']
            ... )
            >>>
            >>> # Use configuration defaults
            >>> binner = IsotonicBinning(guidance_columns='target')

        Note:
            - Parameter validation occurs during initialization
            - Configuration defaults are applied for None parameters
            - The increasing parameter is crucial for defining the expected relationship direction
            - y_min and y_max constraints help with numerical stability and domain knowledge
                enforcement
            - Reconstruction parameters should not be provided during normal usage
            - Guidance columns must be specified for supervised binning to work properly
        """
        # Prepare user parameters for config integration (exclude never-configurable params)
        # Use standardized initialization pattern
        user_params = create_param_dict_for_config(
            max_bins=max_bins,
            min_samples_per_bin=min_samples_per_bin,
            increasing=increasing,
            y_min=y_min,
            y_max=y_max,
            min_change_threshold=min_change_threshold,
            clip=clip,
            preserve_dataframe=preserve_dataframe,
        )

        # Apply configuration defaults for isotonic method
        resolved_params = apply_config_defaults("isotonic", user_params)

        # Store method-specific parameters
        self.max_bins = resolved_params.get("max_bins", 10)
        self.min_samples_per_bin = resolved_params.get("min_samples_per_bin", 5)
        self.increasing = resolved_params.get("increasing", True)
        self.y_min = resolved_params.get("y_min", None)
        self.y_max = resolved_params.get("y_max", None)
        self.min_change_threshold = resolved_params.get("min_change_threshold", 0.01)

        # Dictionary to store fitted isotonic models for each feature
        self._isotonic_models: dict[Any, IsotonicRegression] = {}

        # Initialize parent with resolved parameters (never-configurable params passed as-is)
        SupervisedBinningBase.__init__(
            self,
            clip=resolved_params.get("clip"),
            preserve_dataframe=resolved_params.get("preserve_dataframe"),
            guidance_columns=guidance_columns,  # Never configurable
            bin_edges=bin_edges,  # Never configurable
            bin_representatives=bin_representatives,  # Never configurable
        )

    def _validate_params(self) -> None:
        """Validate Isotonic binning parameters."""
        # Call parent validation
        SupervisedBinningBase._validate_params(self)

        # Validate max_bins using centralized utility
        validate_bin_number_parameter(self.max_bins, param_name="max_bins")

        # Validate min_samples_per_bin parameter
        if not isinstance(self.min_samples_per_bin, int) or self.min_samples_per_bin < 1:
            raise ConfigurationError(
                "min_samples_per_bin must be a positive integer",
                suggestions=["Example: min_samples_per_bin=5"],
            )

        # Validate increasing parameter
        if not isinstance(self.increasing, bool):
            raise ConfigurationError(
                "increasing must be a boolean value",
                suggestions=["Use increasing=True for increasing monotonicity"],
            )

        # Validate y_min and y_max parameters
        if self.y_min is not None and not isinstance(self.y_min, int | float):
            raise ConfigurationError(
                "y_min must be a number or None",
                suggestions=["Example: y_min=0.0"],
            )

        if self.y_max is not None and not isinstance(self.y_max, int | float):
            raise ConfigurationError(
                "y_max must be a number or None",
                suggestions=["Example: y_max=1.0"],
            )

        if self.y_min is not None and self.y_max is not None and self.y_min >= self.y_max:
            raise ConfigurationError(
                "y_min must be less than y_max",
                suggestions=["Example: y_min=0.0, y_max=1.0"],
            )

        # Validate min_change_threshold parameter
        if not isinstance(self.min_change_threshold, int | float) or self.min_change_threshold <= 0:
            raise ConfigurationError(
                "min_change_threshold must be a positive number",
                suggestions=["Example: min_change_threshold=0.01"],
            )

    def _calculate_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        guidance_data: np.ndarray[Any, Any] | None = None,
    ) -> tuple[list[float], list[float]]:
        """Calculate isotonic regression-based bins for a single column.

        Uses isotonic regression to fit a monotonic function to the feature-target
        relationship, then identifies cut points based on significant changes in
        the fitted function.

        Args:
            x_col: Preprocessed column data (from base class)
            col_id: Column identifier for error reporting
            guidance_data: Target/guidance data for supervised binning (required)

        Returns:
            Tuple of (bin_edges, bin_representatives)

        Raises:
            FittingError: If guidance_data is None or insufficient data for binning
        """
        # Require guidance data for supervised binning
        if guidance_data is None:
            raise FittingError(f"Column {col_id}: guidance_data is required for isotonic binning")

        # Prepare guidance data for processing
        guidance_data_numeric = self._prepare_target_values(guidance_data)

        # Validate guidance data shape matches feature data
        if len(guidance_data_numeric) != len(x_col):
            raise ValueError(
                f"Column {col_id}: Guidance data length ({len(guidance_data_numeric)}) "
                f"does not match feature data length ({len(x_col)})"
            )

        # Check if we have sufficient data
        if len(x_col) < self.min_samples_per_bin:
            raise FittingError(
                f"Column {col_id}: Insufficient data points ({len(x_col)}) "
                f"for isotonic binning. Need at least {self.min_samples_per_bin}."
            )

        # Create isotonic binning
        return self._create_isotonic_bins(x_col, guidance_data_numeric, col_id)

    # pylint: disable=too-many-locals
    def _create_isotonic_bins(
        self, x_col: np.ndarray[Any, Any], y_col: np.ndarray[Any, Any], col_id: Any
    ) -> tuple[list[float], list[float]]:
        """Create bins using isotonic regression.

        Fits an isotonic regression model to the feature-target relationship and
        identifies optimal cut points based on changes in the fitted function.

        Args:
            x_col: Clean feature data (no NaN values)
            y_col: Clean target data (no NaN values)
            col_id: Column identifier for model storage

        Returns:
            Tuple of (bin_edges, bin_representatives)

        Note:
            The data is already preprocessed by the base class, so we don't need
            to handle NaN/inf values or constant data here.
        """
        # Resolve max_bins parameter for this dataset
        resolved_max_bins = resolve_n_bins_parameter(
            self.max_bins, data_shape=(len(x_col), 1), param_name="max_bins"
        )

        # Handle infinity values in feature data first (before constant feature check)
        if np.any(np.isinf(x_col)):
            x_finite_mask = np.isfinite(x_col)
            if not np.any(x_finite_mask):
                raise ValueError(f"Column {col_id}: All feature values are infinite")

            # Replace inf values with finite extremes
            x_min_finite = np.min(x_col[x_finite_mask])
            x_max_finite = np.max(x_col[x_finite_mask])
            x_range = x_max_finite - x_min_finite

            # Replace -inf with minimum - 10% of range, +inf with maximum + 10% of range
            x_col = np.where(x_col == -np.inf, x_min_finite - max(abs(x_range) * 0.1, 1.0), x_col)
            x_col = np.where(x_col == np.inf, x_max_finite + max(abs(x_range) * 0.1, 1.0), x_col)

        # Handle constant feature data
        if len(np.unique(x_col)) == 1:
            x_val = float(x_col[0])
            return ([x_val - 0.1, x_val + 0.1], [x_val])

        # Sort data by feature values for isotonic regression
        sort_indices = np.argsort(x_col)
        x_sorted = x_col[sort_indices]
        y_sorted = y_col[sort_indices]

        # Ensure both arrays are 1D for sklearn's IsotonicRegression
        x_sorted = x_sorted.flatten()
        y_sorted = y_sorted.flatten()

        # Fit isotonic regression using safe sklearn call
        try:
            isotonic_model = IsotonicRegression(
                increasing=self.increasing,
                y_min=self.y_min,
                y_max=self.y_max,
                out_of_bounds="clip",
            )
            y_fitted = isotonic_model.fit_transform(x_sorted, y_sorted)
        except (
            ValueError,
            RuntimeError,
            ImportError,
            Exception,
        ) as e:  # pylint: disable=broad-exception-caught
            raise ValueError(f"Column {col_id}: Isotonic regression failed: {e}") from e

        # Store the fitted model
        self._isotonic_models[col_id] = isotonic_model

        # Find cut points based on fitted function changes
        cut_points = self._find_cut_points(x_sorted, y_fitted, resolved_max_bins)

        # Create bin edges and representatives
        return self._create_bins_from_cuts(x_sorted, y_fitted, cut_points, col_id)

    def _prepare_target_values(self, y_values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Prepare target values for isotonic regression.

        Args:
            y_values: Raw target values (may be 2D with shape (n_samples, 1))

        Returns:
            Processed target values suitable for isotonic regression (1D array)
        """
        # Flatten if 2D with single column (guidance_data format)
        if y_values.ndim == 2 and y_values.shape[1] == 1:
            y_values_flat = y_values.flatten()
        else:
            y_values_flat = y_values

        # Convert to float for isotonic regression
        y_processed = y_values_flat.astype(float)

        return y_processed

    def _find_cut_points(
        self,
        x_sorted: np.ndarray[Any, Any],
        y_fitted: np.ndarray[Any, Any],
        max_bins: int,  # pylint: disable=unused-argument
    ) -> list[int]:
        """Find cut points based on changes in fitted isotonic function.

        Identifies locations where the fitted function has significant changes
        that warrant creating new bin boundaries.

        Args:
            x_sorted: Sorted feature values
            y_fitted: Fitted isotonic regression values
            max_bins: Maximum number of bins allowed (resolved from string)

        Returns:
            Indices of cut points in the sorted arrays
        """
        _ = x_sorted

        cut_indices = [0]  # Always start with first point

        if len(y_fitted) <= 1:
            return cut_indices

        # Calculate relative changes in fitted values
        y_range = np.max(y_fitted) - np.min(y_fitted)
        if y_range == 0:
            return cut_indices

        # Find points with significant changes
        for i in range(1, len(y_fitted)):
            # Check if there's a significant change from the last cut point
            last_cut_idx = cut_indices[-1]
            y_change = abs(y_fitted[i] - y_fitted[last_cut_idx])
            relative_change = y_change / y_range

            # Check if we have enough samples since last cut
            samples_since_cut = i - last_cut_idx

            if (
                relative_change >= self.min_change_threshold
                and samples_since_cut >= self.min_samples_per_bin
                and len(cut_indices) < max_bins - 1  # Ensure we don't exceed max_bins
            ):
                cut_indices.append(i)

        return cut_indices

    def _create_bins_from_cuts(
        self,
        x_sorted: np.ndarray[Any, Any],
        y_fitted: np.ndarray[Any, Any],  # pylint: disable=unused-argument
        cut_indices: list[int],
        col_id: Any,  # pylint: disable=unused-argument
    ) -> tuple[list[float], list[float]]:
        """Create bin edges and representatives from cut points.

        Args:
            x_sorted: Sorted feature values
            y_fitted: Fitted isotonic regression values
            cut_indices: Indices of cut points
            col_id: Column identifier for error reporting

        Returns:
            Tuple of (bin_edges, bin_representatives)
        """
        if len(cut_indices) == 1:
            # Only one cut point - create single bin
            x_min, x_max = float(np.min(x_sorted)), float(np.max(x_sorted))
            if x_min == x_max:
                x_max = x_min + 1.0
            return [x_min, x_max], [(x_min + x_max) / 2]

        # Create bin edges
        bin_edges = []
        bin_representatives = []

        for i, cut_idx in enumerate(cut_indices):
            if i == 0:
                # First bin edge
                bin_edges.append(float(x_sorted[cut_idx]))
            else:
                # Find midpoint between consecutive cut points for bin boundary
                prev_cut_idx = cut_indices[i - 1]
                if cut_idx > prev_cut_idx:
                    midpoint = (x_sorted[prev_cut_idx] + x_sorted[cut_idx]) / 2
                    bin_edges.append(float(midpoint))

                    # Representative is the mean of feature values in this bin
                    bin_x_values = x_sorted[prev_cut_idx:cut_idx]
                    bin_representative = float(np.mean(bin_x_values))
                    bin_representatives.append(bin_representative)

        # Add final bin edge and representative
        bin_edges.append(float(x_sorted[-1]))
        if len(cut_indices) > 1:
            final_bin_x = x_sorted[cut_indices[-1] :]
            final_representative = float(np.mean(final_bin_x))
            bin_representatives.append(final_representative)
        else:
            bin_representatives.append(float(np.mean(x_sorted)))

        return bin_edges, bin_representatives
