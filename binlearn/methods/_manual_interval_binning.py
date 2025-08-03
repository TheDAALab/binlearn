"""Manual interval binning transformer.

This module implements manual interval binning, where bin edges are explicitly
provided by the user rather than inferred from data. This allows for complete
control over binning boundaries and ensures consistent binning across different
datasets.

Classes:
    ManualIntervalBinning: Main transformer for user-defined interval binning.
"""

from typing import Any, List, Optional, Tuple

import numpy as np

from ..base._interval_binning_base import IntervalBinningBase
from ..base._repr_mixin import ReprMixin
from ..utils.errors import BinningError, ConfigurationError
from ..utils.types import BinEdgesDict


# pylint: disable=too-many-ancestors
class ManualIntervalBinning(ReprMixin, IntervalBinningBase):
    """Manual interval binning transformer with user-defined bin edges.

    Creates bins using explicitly provided bin edges, giving users complete control
    over binning boundaries. Unlike automatic binning methods, this transformer
    never infers bin edges from data - they must always be provided by the user.

    This approach is ideal for:
    - Standardized binning across multiple datasets
    - Domain-specific binning requirements
    - Reproducible binning with known boundaries
    - Integration with external binning specifications

    The transformer is sklearn-compatible and supports pandas/polars DataFrames.
    It includes comprehensive validation of user-provided bin edges and optional
    automatic generation of bin representatives.

    Attributes:
        bin_edges (dict): User-provided bin edges for each feature.
        bin_representatives (dict, optional): User-provided or auto-generated representatives.
        clip (bool, optional): Whether to clip values outside bin range.
        preserve_dataframe (bool): Whether to preserve DataFrame format.

    Example:
        >>> import numpy as np
        >>> from binlearn.methods import ManualIntervalBinning
        >>>
        >>> # Define custom bin edges for features
        >>> edges = {
        ...     0: [0, 25, 50, 75, 100],        # Feature 0: age groups
        ...     1: [0, 1000, 5000, 10000],      # Feature 1: income brackets
        ...     'score': [0, 60, 80, 90, 100]   # Feature 'score': grade boundaries
        ... }
        >>>
        >>> binner = ManualIntervalBinning(bin_edges=edges)
        >>> X = np.random.rand(100, 3) * 100
        >>> X_binned = binner.fit_transform(X)
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        bin_edges: BinEdgesDict,
        bin_representatives: Optional[BinEdgesDict] = None,
        clip: Optional[bool] = None,
        preserve_dataframe: Optional[bool] = None,
        fit_jointly: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ManualIntervalBinning transformer.

        Creates a manual interval binning transformer that uses user-provided bin
        edges instead of inferring them from data. This gives complete control over
        binning boundaries and ensures consistency across different datasets.

        Args:
            bin_edges (BinEdgesDict): Dictionary mapping column identifiers to
                their bin edge lists. Each edge list must be sorted in ascending
                order with at least 2 elements. Keys can be column names (str/int)
                or indices, and values are lists/arrays of bin boundaries.
            bin_representatives (Optional[BinEdgesDict], optional): Dictionary
                mapping column identifiers to their bin representative values.
                If None, representatives are auto-generated as bin centers.
                Must have compatible structure with bin_edges. Defaults to None.
            clip (Optional[bool], optional): Whether to clip out-of-range values
                to the nearest bin edge. If None, uses global configuration.
                Defaults to None.
            preserve_dataframe (Optional[bool], optional): Whether to return
                DataFrames when input is DataFrame. If None, uses global
                configuration. Defaults to None.
            fit_jointly (Optional[bool], optional): Whether to treat all columns
                with the same binning parameters. Not applicable for manual binning
                since edges are explicitly provided. Defaults to None.
            **kwargs: Additional arguments passed to parent IntervalBinningBase.

        Raises:
            ConfigurationError: If bin_edges is not provided, empty, or contains
                invalid edge specifications.

        Example:
            >>> # Basic usage with custom bin edges
            >>> edges = {0: [0, 10, 20, 30], 1: [0, 5, 15, 25, 50]}
            >>> binner = ManualIntervalBinning(bin_edges=edges)

            >>> # With custom representatives
            >>> edges = {0: [0, 10, 20, 30]}
            >>> reps = {0: [5, 15, 25]}  # Custom bin centers
            >>> binner = ManualIntervalBinning(bin_edges=edges, bin_representatives=reps)

            >>> # For DataFrame with named columns
            >>> edges = {'age': [0, 18, 35, 50, 65, 100], 'income': [0, 30000, 60000, 100000]}
            >>> binner = ManualIntervalBinning(bin_edges=edges)

        Note:
            - bin_edges is required and cannot be None
            - Each edge list must be sorted and have at least 2 elements
            - If bin_representatives is not provided, bin centers are used as representatives
            - fit_jointly parameter is ignored since bins are manually specified
        """
        super().__init__(
            clip=clip,
            preserve_dataframe=preserve_dataframe,
            bin_edges=bin_edges,
            bin_representatives=bin_representatives,
            fit_jointly=fit_jointly,
            **kwargs,
        )

    def _calculate_bins(
        self, x_col: np.ndarray, col_id: Any, guidance_data: Optional[np.ndarray] = None
    ) -> Tuple[List[float], List[float]]:
        """Return pre-defined bins without calculation.

        Since ManualIntervalBinning uses user-provided bin edges, this method
        simply returns the pre-specified edges and representatives without
        performing any data-based calculations.

        Args:
            x_col (np.ndarray): Input data column (ignored in manual binning).
            col_id (Any): Column identifier to retrieve pre-defined bins.
            guidance_data (Optional[np.ndarray], optional): Guidance data (ignored).
                Defaults to None.

        Returns:
            Tuple[List[float], List[float]]: A tuple containing:
                - bin_edges (List[float]): Pre-defined bin edges for this column
                - bin_representatives (List[float]): Pre-defined or auto-generated
                  representative values for this column

        Raises:
            BinningError: If no bin edges are defined for the specified column.

        Note:
            - Input data is ignored since bins are manually specified
            - If no representatives are provided, bin centers are auto-generated
            - This method is called during fitting but doesn't analyze data
        """
        # Get pre-defined edges for this column
        if self.bin_edges is None or col_id not in self.bin_edges:
            raise BinningError(
                f"No bin edges defined for column {col_id}",
                suggestions=[
                    f"Add bin edges for column {col_id} in the bin_edges dictionary",
                    "Ensure all data columns have corresponding bin edge definitions",
                ],
            )

        edges = list(self.bin_edges[col_id])

        # Get or generate representatives
        if self.bin_representatives is not None and col_id in self.bin_representatives:
            representatives = list(self.bin_representatives[col_id])
        else:
            # Auto-generate representatives as bin centers
            representatives = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]

        return edges, representatives

    def _validate_params(self) -> None:
        """Validate parameters for manual interval binning.

        Performs validation specific to ManualIntervalBinning - checks for presence
        and emptiness of required parameters. Content validation is handled by the
        base class IntervalBinningBase._validate_params().

        Raises:
            ConfigurationError: If any parameter validation fails:
                - bin_edges must be provided and non-empty

        Note:
            - Called automatically during initialization for early error detection
            - Only checks presence/emptiness - content validation in base class
            - Part of sklearn-compatible parameter validation pattern
        """
        # Call parent validation first (handles content validation)
        super()._validate_params()

        # ManualIntervalBinning specific validation: bin_edges is required
        if self.bin_edges is None:
            raise ConfigurationError(
                "bin_edges must be provided for ManualIntervalBinning",
                suggestions=[
                    "Provide bin_edges as a dictionary mapping columns to edge lists",
                    "Example: bin_edges={0: [0, 10, 20, 30], 1: [0, 5, 15, 25]}",
                ],
            )

        if not self.bin_edges:  # Empty dict
            raise ConfigurationError(
                "bin_edges cannot be empty for ManualIntervalBinning",
                suggestions=[
                    "Provide at least one column with bin edges",
                    "Example: bin_edges={0: [0, 10, 20, 30]}",
                ],
            )
