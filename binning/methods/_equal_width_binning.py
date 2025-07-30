"""Equal-width binning transformer.

This module implements equal-width binning, where continuous data is divided
into bins of equal width across the range of each feature. This is one of the
most common and straightforward binning strategies.

Classes:
    EqualWidthBinning: Main transformer for equal-width binning operations.
"""

from typing import Any, Tuple, Optional, List
import numpy as np

from ..utils.types import (
    BinEdges, ColumnList, BinEdgesDict
)
from ..base._interval_binning_base import IntervalBinningBase
from ..base._repr_mixin import ReprMixin
from ..utils.errors import ConfigurationError


# pylint: disable=too-many-ancestors
class EqualWidthBinning(ReprMixin, IntervalBinningBase):
    """Classic equal-width binning transformer.

    Creates bins of equal width across the range of each feature. Each bin
    spans the same numeric range, making this method intuitive and easy to
    interpret. The bins are determined solely by the minimum and maximum
    values in each feature, without considering the target variable.

    This transformer is sklearn-compatible and supports pandas/polars DataFrames.
    It includes advanced features like custom bin ranges, clipping, and
    comprehensive error handling.

    Attributes:
        n_bins (int): Number of bins per feature.
        bin_range (tuple, optional): Custom range for binning.
        clip (bool, optional): Whether to clip values outside bin range.
        columns (list, optional): Specific columns to bin.
        guidance_columns (list, optional): Columns to exclude from binning.
        preserve_dataframe (bool): Whether to preserve DataFrame format.
        bin_edges_ (dict): Computed bin edges after fitting.
        
    Example:
        >>> import numpy as np
        >>> from binning.methods import EqualWidthBinning
        >>> X = np.random.rand(100, 3)
        >>> binner = EqualWidthBinning(n_bins=5)
        >>> X_binned = binner.fit_transform(X)
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        n_bins: int = 10,
        bin_range: Optional[Tuple[float, float]] = None,
        clip: Optional[bool] = None,
        preserve_dataframe: Optional[bool] = None,
        bin_edges: Optional[BinEdgesDict] = None,
        bin_representatives: Optional[BinEdgesDict] = None,
        fit_jointly: Optional[bool] = None,
        **kwargs,
    ) -> None:
        """Initialize EqualWidthBinning.

        Args:
            n_bins: Number of bins to create for all columns.
            bin_range: Range for binning. If None, uses data min/max.
            clip: Whether to clip out-of-range values to nearest bin.
            preserve_dataframe: Whether to return DataFrames when input is DataFrame.
            bin_edges: Optional pre-specified bin edges.
            bin_representatives: Optional pre-specified bin representatives.
            fit_jointly: Whether to fit joint parameters across columns.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(
            clip=clip,
            preserve_dataframe=preserve_dataframe,
            bin_edges=bin_edges,
            bin_representatives=bin_representatives,
            fit_jointly=fit_jointly,
            **kwargs,
        )

        # Store equal-width specific parameters
        self.n_bins = n_bins
        self.bin_range = bin_range

    def _calculate_bins_jointly(
        self, all_data: np.ndarray, columns: ColumnList
    ) -> Tuple[BinEdges, BinEdges]:
        """Calculate equal-width bins from all flattened data for joint binning."""
        if self.n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {self.n_bins}")

        # Determine range to use
        if self.bin_range is not None:
            min_val, max_val = self.bin_range
        else:
            # Use global min/max from all data
            min_val = np.nanmin(all_data)
            max_val = np.nanmax(all_data)

        return self._create_equal_width_bins(float(min_val), float(max_val), self.n_bins)

    def _calculate_bins(
        self, x_col: np.ndarray, col_id: Any, guidance_data: Optional[np.ndarray] = None
    ) -> Tuple[List[float], List[float]]:
        """Calculate equal-width bins for a column (per-column logic).

        Args:
            x_col: Column data to bin.
            col_id: Column identifier.
            guidance_data: Optional guidance data (not used in equal-width binning).
        """
        if self.n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {self.n_bins}")

        # Get range for this column
        if self.bin_range is not None:
            min_val, max_val = self.bin_range
        else:
            min_val, max_val = self._get_data_range(x_col, col_id)

        return self._create_equal_width_bins(min_val, max_val, self.n_bins)

    def _get_data_range(self, x_col: np.ndarray, col_id: Any) -> Tuple[float, float]:
        """Get the data range for a column."""
        # Check if all values are NaN
        if np.all(np.isnan(x_col)):
            # Create a default range for all-NaN columns
            return 0.0, 1.0

        min_val = np.nanmin(x_col)
        max_val = np.nanmax(x_col)

        if not (np.isfinite(min_val) and np.isfinite(max_val)):
            # This can happen if there are inf values
            raise ValueError(f"Cannot create bins for column {col_id}: min and max must be finite.")

        return float(min_val), float(max_val)

    def _create_equal_width_bins(
        self, min_val: float, max_val: float, n_bins: int
    ) -> Tuple[List[float], List[float]]:
        """Create equal-width bins given range and number of bins."""
        if min_val == max_val:
            # Handle constant data
            epsilon = 1e-8
            min_val -= epsilon
            max_val += epsilon

        # Create equal-width bin edges
        edges = np.linspace(min_val, max_val, n_bins + 1)

        # Create representatives as bin centers
        reps = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]

        return list(edges), reps

    def _validate_params(self) -> None:
        """Validate parameters for sklearn compatibility."""
        # Validate n_bins
        if not isinstance(self.n_bins, int) or self.n_bins < 1:
            raise ConfigurationError(
                "n_bins must be a positive integer",
                suggestions=["Set n_bins to a positive integer (e.g., n_bins=10)"],
            )

        # Validate bin_range if provided
        if self.bin_range is not None:
            if (
                not isinstance(self.bin_range, tuple)
                or len(self.bin_range) != 2
                or self.bin_range[0] >= self.bin_range[1]
            ):
                raise ConfigurationError(
                    "bin_range must be a tuple (min, max) with min < max",
                    suggestions=["Example: bin_range=(0, 100)"],
                )
