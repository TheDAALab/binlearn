"""Equal-frequency binning transformer (V2 Architecture).

This module implements equal-frequency binning (also known as quantile binning)
using the V2 architecture, where continuous data is divided into bins containing
approximately equal numbers of observations.

Classes:
    EqualFrequencyBinningV2: V2 architecture equal-frequency binning transformer.
"""

from typing import Any

import numpy as np

from ..base._binning_utils_mixin import EdgeBasedBinningMixin, BinningUtilsMixin
from ..base._general_binning_base_v2 import GeneralBinningBaseV2
from ..utils.validation import (
    validate_int,
    validate_float,
    validate_tuple,
    validate_bool,
    validate_n_bins,
    ParameterValidator,
)
from ..utils.types import ColumnList


class EqualFrequencyBinningV2(EdgeBasedBinningMixin, BinningUtilsMixin, GeneralBinningBaseV2):
    """Equal-frequency (quantile) binning transformer using V2 architecture.

    Creates bins containing approximately equal numbers of observations across
    each feature. Each bin contains roughly the same number of data points,
    making this method useful when you want balanced bin populations regardless
    of the underlying data distribution.

    This V2 implementation provides:
    - Clean parameter validation using utility functions
    - Multi-format I/O handling (numpy, pandas, polars)
    - Complete parameter reconstruction workflows
    - Minimal implementation using utility mixins
    - Enhanced error handling and validation

    Attributes:
        n_bins (int): Number of bins per feature.
        quantile_range (tuple, optional): Custom quantile range for binning.
        clip (bool, optional): Whether to clip values outside bin range.
        bin_edges_ (dict): Computed bin edges after fitting.

    Example:
        >>> import numpy as np
        >>> from binlearn.methods import EqualFrequencyBinningV2
        >>> X = np.random.rand(100, 3)
        >>> binner = EqualFrequencyBinningV2(n_bins=5)
        >>> X_binned = binner.fit_transform(X)
    """

    def __init__(
        self,
        n_bins: int = 10,
        quantile_range: tuple[float, float] | None = None,
        clip: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize EqualFrequencyBinningV2 transformer.

        Args:
            n_bins (int): Number of bins to create for each feature. Must be >= 1.
                Defaults to 10.
            quantile_range (Optional[Tuple[float, float]], optional): Custom quantile
                range for binning as (min_quantile, max_quantile). Values should be
                between 0 and 1. If None, uses the full data range (0, 1).
                Defaults to None.
            clip (bool): Whether to clip out-of-range values to the nearest bin edge.
                Defaults to True.
            **kwargs: Additional arguments for future extensibility.

        Raises:
            ValueError: If parameter validation fails.
        """
        # Validate parameters using V2 utilities
        validator = ParameterValidator("EqualFrequencyBinningV2")
        validated_params = validator.validate(
            n_bins=(n_bins, validate_n_bins),
            clip=(clip, validate_bool, "clip"),
        )

        if quantile_range is not None:
            validated_range = validate_tuple(
                quantile_range, "quantile_range", expected_length=2, element_type=float
            )
            if validated_range is not None:
                min_q, max_q = validated_range
                validate_float(min_q, "quantile_range[0]", min_val=0.0, max_val=1.0)
                validate_float(max_q, "quantile_range[1]", min_val=0.0, max_val=1.0)
                if min_q >= max_q:
                    raise ValueError("quantile_range[0] must be less than quantile_range[1]")
                self.quantile_range = validated_range
            else:
                self.quantile_range = None
        else:
            self.quantile_range = None

        # Store validated parameters
        self.n_bins = validated_params["n_bins"]
        self.clip = validated_params["clip"]

        # Initialize bin_edges_ for storing fitted parameters
        self.bin_edges_ = {}

        # Initialize parent class
        super().__init__(**kwargs)

    def _fit_per_column_independently(
        self, X: np.ndarray, columns: ColumnList, guidance_data=None, **fit_params
    ) -> None:
        """Fit equal-frequency bins independently for each column.

        Args:
            X: Input data for binning columns.
            columns: Column identifiers for binning columns.
            guidance_data: Ignored for equal-frequency binning.
            **fit_params: Additional fitting parameters.
        """
        for i, col in enumerate(columns):
            column_data = X[:, i]

            # Remove NaN values for quantile calculation
            clean_data = column_data[~np.isnan(column_data)]

            if len(clean_data) == 0:
                # All NaN data - create default range
                edges = np.linspace(0.0, 1.0, self.n_bins + 1)
            elif len(clean_data) < self.n_bins:
                raise ValueError(
                    f"Column {col}: Insufficient non-NaN values ({len(clean_data)}) "
                    f"for {self.n_bins} bins. Need at least {self.n_bins} values."
                )
            else:
                # Get quantile range
                if self.quantile_range is not None:
                    min_quantile, max_quantile = self.quantile_range
                else:
                    min_quantile, max_quantile = 0.0, 1.0

                # Create quantile points from min_quantile to max_quantile
                quantile_points = np.linspace(min_quantile, max_quantile, self.n_bins + 1)

                # Calculate quantile values
                try:
                    edges = np.quantile(clean_data, quantile_points)
                except (ValueError, IndexError) as e:
                    raise ValueError(f"Column {col}: Error calculating quantiles: {e}") from e

                # Handle case where quantiles result in duplicate edges (constant regions)
                if edges[0] == edges[-1]:
                    # All data points are the same - add small epsilon
                    epsilon = 1e-8
                    edges[0] -= epsilon
                    edges[-1] += epsilon

                # Ensure edges are strictly increasing
                for j in range(1, len(edges)):
                    if edges[j] <= edges[j - 1]:
                        edges[j] = edges[j - 1] + 1e-8

            self.bin_edges_[col] = edges

    def _fit_jointly_across_columns(self, X: np.ndarray, columns: ColumnList, **fit_params) -> None:
        """Fit equal-frequency parameters jointly across all columns.

        Uses the global quantiles across all columns to create uniform bins.

        Args:
            X: Input data for all binning columns.
            columns: Column identifiers for all columns.
            **fit_params: Additional fitting parameters.
        """
        # Flatten all data for joint quantile calculation
        flat_data = X.flatten()
        clean_data = flat_data[~np.isnan(flat_data)]

        if len(clean_data) == 0:
            # All NaN data - create default range
            edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        elif len(clean_data) < self.n_bins:
            raise ValueError(
                f"Insufficient non-NaN values ({len(clean_data)}) "
                f"for {self.n_bins} bins. Need at least {self.n_bins} values."
            )
        else:
            # Get quantile range
            if self.quantile_range is not None:
                min_quantile, max_quantile = self.quantile_range
            else:
                min_quantile, max_quantile = 0.0, 1.0

            # Create quantile points from min_quantile to max_quantile
            quantile_points = np.linspace(min_quantile, max_quantile, self.n_bins + 1)

            # Calculate quantile values
            try:
                edges = np.quantile(clean_data, quantile_points)
            except (ValueError, IndexError) as e:
                raise ValueError(f"Error calculating joint quantiles: {e}") from e

            # Handle case where quantiles result in duplicate edges (constant regions)
            if edges[0] == edges[-1]:
                # All data points are the same - add small epsilon
                epsilon = 1e-8
                edges[0] -= epsilon
                edges[-1] += epsilon

            # Ensure edges are strictly increasing
            for j in range(1, len(edges)):
                if edges[j] <= edges[j - 1]:
                    edges[j] = edges[j - 1] + 1e-8

        # Apply same edges to all columns
        for col in columns:
            self.bin_edges_[col] = edges
