"""Manual flexible binning transformer.

This module implements manual flexible binning, where bin specifications are
explicitly provided by the user and can include both singleton bins (exact values)
and interval bins (ranges). This provides maximum flexibility for complex binning
requirements.

Classes:
    ManualFlexibleBinning: Main transformer for user-defined flexible binning.
"""

from typing import Any

import numpy as np

from ..base._flexible_binning_base import FlexibleBinningBase
from ..base._repr_mixin import ReprMixin
from ..utils.errors import BinningError, ConfigurationError
from ..utils.types import BinEdgesDict, BinReps, FlexibleBinDefs, FlexibleBinSpec


# pylint: disable=too-many-ancestors
class ManualFlexibleBinning(ReprMixin, FlexibleBinningBase):
    """Manual flexible binning transformer with user-defined bin specifications.

    Creates bins using explicitly provided bin specifications that can include both:
    - Singleton bins: exact numeric value matches (e.g., specific values or outliers)
    - Interval bins: numeric range matches (e.g., [min, max) intervals)

    This transformer never infers bin specifications from data - they must always
    be provided by the user. This approach offers maximum flexibility for complex
    binning scenarios that combine singleton exact matches with traditional
    interval binning.

    This is ideal for:
    - Numeric data requiring both exact and range matching
    - Complex domain-specific numeric binning rules
    - Standardized binning with both singleton and continuous elements
    - Integration with external flexible binning specifications

    The transformer is sklearn-compatible and supports pandas/polars DataFrames.
    It includes comprehensive validation of user-provided bin specifications and
    automatic generation of bin representatives when needed.

    Attributes:
        bin_specs (dict): User-provided flexible bin specifications for each feature.
        bin_representatives (dict, optional): User-provided or auto-generated representatives.
        preserve_dataframe (bool): Whether to preserve DataFrame format.

    Example:
        >>> import numpy as np
        >>> from binlearn.methods import ManualFlexibleBinning
        >>>
        >>> # Define mixed numeric bin specifications
        >>> specs = {
        ...     'grade': [
        ...         95,            # Singleton bin for high grade
        ...         85,            # Another singleton bin
        ...         (0, 60),       # Interval bin for failing grades
        ...         (60, 80),      # Interval bin for passing grades
        ...         (80, 100)      # Interval bin for high grades
        ...     ],
        ...     'age': [
        ...         (0, 18),       # Minors
        ...         (18, 35),      # Young adults
        ...         (35, 65),      # Middle-aged
        ...         65             # Seniors (singleton for exact match)
        ...     ]
        ... }
        >>>
        >>> binner = ManualFlexibleBinning(bin_specs=specs)
        >>> # Data with numeric values
        >>> X = [[95, 25], [85, 45], [75, 17], [42, 65]]
        >>> X_binned = binner.fit_transform(X)
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        bin_spec: FlexibleBinSpec,
        bin_representatives: BinEdgesDict | None = None,
        preserve_dataframe: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ManualFlexibleBinning transformer.

        Creates a manual flexible binning transformer that uses user-provided bin
        specifications instead of inferring them from data. Bin specifications can
        include both singleton bins (exact numeric values) and interval bins (numeric ranges).

        Args:
            bin_spec (FlexibleBinSpec): Dictionary mapping column identifiers to
                their flexible bin specification lists. Each specification can be:
                - Numeric scalar values for singleton bins (exact matches)
                - Tuples (min, max) for interval bins (range matches)
                Keys can be column names (str/int) or indices.
            bin_representatives (Optional[BinEdgesDict], optional): Dictionary
                mapping column identifiers to their bin representative values.
                If None, representatives are auto-generated based on bin type.
                Must have compatible structure with bin_spec. Defaults to None.
            preserve_dataframe (Optional[bool], optional): Whether to return
                DataFrames when input is DataFrame. If None, uses global
                configuration. Defaults to None.
            **kwargs: Additional arguments passed to parent FlexibleBinningBase.

        Raises:
            ConfigurationError: If bin_spec is not provided, empty, or contains
                invalid bin specifications.

        Example:
            >>> # Basic usage with mixed numeric bin types
            >>> specs = {
            ...     0: [1.5, (2, 5), (5, 10), 15],  # Mix of singletons and intervals
            ...     1: [(0, 25), (25, 50), (50, 100)]  # Only intervals
            ... }
            >>> binner = ManualFlexibleBinning(bin_spec=specs)

            >>> # With custom representatives
            >>> specs = {0: [(0, 10), (10, 20), 25]}
            >>> reps = {0: [5, 15, 25]}  # Custom representatives
            >>> binner = ManualFlexibleBinning(bin_spec=specs, bin_representatives=reps)

            >>> # For DataFrame with named columns
            >>> specs = {
            ...     'category': [1, 2, 3],  # Numeric categorical bins
            ...     'score': [(0, 60), (60, 80), (80, 100)]  # Score ranges
            ... }
            >>> binner = ManualFlexibleBinning(bin_spec=specs)

        Note:
            - bin_spec is required and cannot be None
            - Each bin can be either a scalar (singleton) or tuple (interval)
            - Singleton bins match exact values
            - Interval bins are inclusive of lower bound, exclusive of upper bound
            - If bin_representatives is not provided, they are auto-generated
        """
        # Validate that bin_spec is provided
        if bin_spec is None:
            raise ConfigurationError(
                "bin_spec must be provided for ManualFlexibleBinning",
                suggestions=[
                    "Provide bin_spec as a dictionary mapping columns to bin specification lists",
                    "Example: bin_spec={0: [1.5, (2, 5), (5, 10)], 1: [(0, 25), (25, 50)]}",
                ],
            )

        super().__init__(
            preserve_dataframe=preserve_dataframe,
            bin_spec=bin_spec,
            bin_representatives=bin_representatives,
            **kwargs,
        )

    def _calculate_flexible_bins(
        self, x_col: np.ndarray, col_id: Any, guidance_data: np.ndarray | None = None
    ) -> tuple[FlexibleBinDefs, BinReps]:
        """Return pre-defined flexible bin specifications without calculation.

        Since ManualFlexibleBinning uses user-provided bin specifications, this method
        simply returns the pre-specified bins and representatives without performing
        any data-based calculations.

        Args:
            x_col (np.ndarray): Input data column (ignored in manual binning).
            col_id (Any): Column identifier to retrieve pre-defined bin specifications.
            guidance_data (Optional[np.ndarray], optional): Guidance data (ignored).
                Defaults to None.

        Returns:
            Tuple[FlexibleBinDefs, BinReps]: A tuple containing:
                - bin_specs (FlexibleBinDefs): Pre-defined flexible bin specifications for this
                  column
                - bin_representatives (BinReps): Pre-defined or auto-generated representative
                  values for this column

        Raises:
            BinningError: If no bin specifications are defined for the specified column.

        Note:
            - Input data is ignored since bins are manually specified
            - If no representatives are provided, they are auto-generated based on bin type
            - Singleton bins get the exact value as representative
            - Interval bins get the midpoint as representative
        """
        # Get pre-defined bin specifications for this column
        if self.bin_spec is None or col_id not in self.bin_spec:
            raise BinningError(
                f"No bin specifications defined for column {col_id}",
                suggestions=[
                    f"Add bin specifications for column {col_id} in the bin_spec dictionary",
                    "Ensure all data columns have corresponding bin specification definitions",
                ],
            )

        specs = list(self.bin_spec[col_id])

        # Get or generate representatives
        if self.bin_representatives is not None and col_id in self.bin_representatives:
            representatives = list(self.bin_representatives[col_id])
        else:
            # Auto-generate representatives based on bin type
            representatives = []
            for spec in specs:
                if isinstance(spec, tuple) and len(spec) == 2:
                    # Interval bin: use midpoint as representative
                    representatives.append(float((spec[0] + spec[1]) / 2))
                elif not isinstance(spec, tuple):
                    # Singleton bin: use the value itself as representative
                    # Ensure we can convert to float
                    try:
                        representatives.append(float(spec))
                    except (ValueError, TypeError):
                        # For non-numeric singleton bins, use a placeholder
                        representatives.append(0.0)
                else:
                    # Fallback for unexpected formats
                    representatives.append(0.0)

        return specs, representatives

    def _validate_params(self) -> None:
        """Validate parameters for manual flexible binning.

        Performs validation specific to ManualFlexibleBinning - checks for presence
        and emptiness of required parameters. Content validation is handled by the
        base class FlexibleBinningBase._validate_params().

        Raises:
            ConfigurationError: If any parameter validation fails:
                - bin_spec must be provided and non-empty

        Note:
            - Called automatically during fit() for early error detection
            - Only checks presence/emptiness - content validation in base class
            - Part of sklearn-compatible parameter validation pattern
        """
        # Call parent validation first (handles content validation)
        super()._validate_params()

        # ManualFlexibleBinning specific validation: bin_spec is required
        if self.bin_spec is None or len(self.bin_spec) == 0:
            raise ConfigurationError(
                "bin_spec must be provided and non-empty for ManualFlexibleBinning",
                suggestions=[
                    "Provide bin_spec as a dictionary: {column: [spec1, spec2, ...]}",
                    "Example: bin_spec={0: [1.5, (2, 5)], 1: [(0, 25), (25, 50)]}",
                ],
            )
