"""
Clean Gaussian Mixture binning implementation for  architecture.

This module provides GaussianMixtureBinning that inherits from IntervalBinningBase.
Uses Gaussian Mixture Model clustering to find natural probabilistic bin boundaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.mixture import GaussianMixture

from ..config import apply_config_defaults
from ..utils import ConfigurationError
from ..utils import (
    resolve_n_bins_parameter,
    validate_bin_number_for_calculation,
    validate_bin_number_parameter,
)
from ..utils import BinEdgesDict
from ..base import IntervalBinningBase


class GaussianMixtureBinning(IntervalBinningBase):
    """Gaussian Mixture Model clustering-based binning implementation using  architecture.

    Creates bins based on Gaussian Mixture Model (GMM) clustering of each feature.
    The bin edges are determined by the decision boundaries between mixture components,
    creating bins that represent natural probabilistic groupings in the data.

    This implementation follows the clean  architecture with straight inheritance,
    dynamic column resolution, and parameter reconstruction capabilities.
    """

    def __init__(
        self,
        n_components: int | str | None = None,
        random_state: int | None = None,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        fit_jointly: bool | None = None,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
        class_: str | None = None,  # For reconstruction compatibility
        module_: str | None = None,  # For reconstruction compatibility
    ):
        """Initialize Gaussian Mixture binning."""
        # Prepare user parameters for config integration (exclude never-configurable params)
        user_params = {
            "n_components": n_components,
            "random_state": random_state,
            "clip": clip,
            "preserve_dataframe": preserve_dataframe,
            "fit_jointly": fit_jointly,
        }
        # Remove None values to allow config defaults to take effect
        user_params = {k: v for k, v in user_params.items() if v is not None}

        # Apply configuration defaults for gaussian_mixture method
        resolved_params = apply_config_defaults("gaussian_mixture", user_params)

        # Store method-specific parameters
        self.n_components = resolved_params.get("n_components", 10)
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
        """Validate Gaussian Mixture binning parameters."""
        # Call parent validation
        IntervalBinningBase._validate_params(self)

        # Validate n_components using centralized utility
        validate_bin_number_parameter(self.n_components, param_name="n_components")

        # Validate random_state parameter
        if self.random_state is not None and not isinstance(self.random_state, int):
            raise ConfigurationError(
                "random_state must be an integer or None",
                suggestions=["Example: random_state=42"],
            )

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

    def _create_gmm_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        n_components: int,
    ) -> tuple[list[float], list[float]]:
        """Create Gaussian Mixture Model clustering-based bins.

        Args:
            x_col: Preprocessed column data (no NaN/inf values)
            col_id: Column identifier for error reporting
            n_components: Number of mixture components to create

        Returns:
            Tuple of (bin_edges, bin_representatives)

        Note:
            The data is already preprocessed by the base class, so we don't need
            to handle NaN/inf values or constant data here.
        """
        if len(x_col) < n_components:
            raise ValueError(
                f"Column {col_id}: Insufficient values ({len(x_col)}) "
                f"for {n_components} components. Need at least {n_components} values."
            )

        # Reshape data for GMM (expects 2D array)
        X_reshaped = x_col.reshape(-1, 1)

        try:
            # Apply Gaussian Mixture Model clustering
            gmm = GaussianMixture(
                n_components=n_components, random_state=self.random_state, covariance_type="full"
            )
            gmm.fit(X_reshaped)

            # Get component means and sort them
            means = np.array(gmm.means_).flatten()
            sorted_indices = np.argsort(means)
            sorted_means = means[sorted_indices]

            # Calculate component boundaries
            edges = [float(np.min(x_col))]  # Start with data minimum

            # Create boundaries between adjacent components
            for i in range(len(sorted_means) - 1):
                boundary = (sorted_means[i] + sorted_means[i + 1]) / 2
                edges.append(float(boundary))

            edges.append(float(np.max(x_col)))  # End with data maximum

            # Representatives are the component means
            reps = [float(mean) for mean in sorted_means]

            return edges, reps

        except Exception as e:
            # Fall back to equal-width binning if GMM fails
            return self._fallback_equal_width_bins(x_col, col_id, n_components, e)

    def _fallback_equal_width_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        n_components: int,
        original_error: Exception,
    ) -> tuple[list[float], list[float]]:
        """Fallback to equal-width binning when GMM fails.

        Args:
            x_col: Preprocessed column data
            col_id: Column identifier for error reporting
            n_components: Number of bins to create
            original_error: The original GMM error

        Returns:
            Tuple of (bin_edges, bin_representatives)
        """
        import warnings
        from ..utils._errors import DataQualityWarning

        warnings.warn(
            f"Column {col_id}: GMM clustering failed ({original_error}). "
            f"Falling back to equal-width binning.",
            DataQualityWarning,
        )

        min_val, max_val = float(np.min(x_col)), float(np.max(x_col))

        # Create equal-width bins
        edges = np.linspace(min_val, max_val, n_components + 1)

        # Create representatives as bin centers
        reps = []
        for i in range(n_components):
            rep = (edges[i] + edges[i + 1]) / 2
            reps.append(rep)

        return list(edges), reps
