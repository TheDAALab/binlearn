"""
OneHotBinning transformer for creating singleton bins from unique values.

This module implements a specialized binning method that creates one bin per unique
value in the data. Unlike traditional one-hot encoding that expands features into
multiple columns, this transformer maintains the original data shape while creating
singleton bins defined as {"singleton": value}.

The transformer is designed for scenarios where you want to treat each unique value
as its own bin, useful for categorical data represented as numbers or when you need
fine-grained binning based on actual data values.
"""

from typing import Tuple, Optional
import numpy as np

from ..utils.types import (
    FlexibleBinDefs, ColumnId, FlexibleBinSpec, BinEdges, BinEdgesDict
)
from ..base._flexible_binning_base import FlexibleBinningBase
from ..base._repr_mixin import ReprMixin


# pylint: disable=too-many-ancestors
class OneHotBinning(ReprMixin, FlexibleBinningBase):
    """Creates a singleton bin for each unique value in numeric data.
    
    This transformer creates one bin per unique value found in the data, where each
    bin is defined as {"singleton": value}. Unlike traditional one-hot encoding that
    expands columns, this maintains the original data shape while creating fine-grained
    bins based on actual data values.
    
    The method is particularly useful for:
    - Categorical data represented as numbers
    - Fine-grained binning where each unique value should be its own bin
    - Preprocessing for models that benefit from value-specific binning
    
    **Important**: This method only supports numeric data. Non-numeric data will
    raise a ValueError during fitting.
    
    Args:
        preserve_dataframe (bool, optional): Whether to preserve DataFrame format in output.
            If None, uses global configuration default.
        bin_spec (FlexibleBinSpec, optional): Pre-defined bin specifications for columns.
            If provided, skips automatic bin generation.
        bin_representatives (BinEdgesDict, optional): Pre-computed bin representatives.
        max_unique_values (int): Maximum number of unique values allowed per column
            before raising an error. Prevents memory issues with high cardinality data.
        **kwargs: Additional arguments passed to the parent FlexibleBinningBase.
        
    Attributes:
        max_unique_values (int): Maximum unique values allowed per column.
        bin_spec_ (FlexibleBinDefs): Generated bin specifications after fitting.
        bin_representatives_ (BinEdgesDict): Computed bin representatives after fitting.
        
    Raises:
        ValueError: If non-numeric data is provided or if unique values exceed max_unique_values.
        
    Example:
        >>> import numpy as np
        >>> from binning.methods import OneHotBinning
        >>> X = np.array([[1.0, 10.0], [2.0, 20.0], [1.0, 10.0]])
        >>> binner = OneHotBinning(max_unique_values=50)
        >>> X_binned = binner.fit_transform(X)
        >>> # Each unique value gets its own bin: 1.0->bin0, 2.0->bin1, etc.
    """

    def __init__(
        self,
        preserve_dataframe: Optional[bool] = None,
        bin_spec: Optional[FlexibleBinSpec] = None,
        bin_representatives: Optional[BinEdgesDict] = None,
        max_unique_values: int = 100,
        **kwargs,
    ):
        """Initialize OneHotBinning transformer.

        Creates singleton bins for each unique value in the data. This is NOT 
        traditional one-hot encoding - instead, it creates bins where each bin 
        contains exactly one unique value, maintaining the original data shape.

        Args:
            preserve_dataframe (bool, optional): Whether to preserve pandas DataFrame 
                structure in output. If None, uses global configuration default.
            bin_spec (FlexibleBinSpec, optional): Pre-defined bin specification.
                If provided, skips automatic bin generation during fitting.
            bin_representatives (BinEdgesDict, optional): Pre-defined bin representatives.
                Used for inverse transformation.
            max_unique_values (int): Maximum number of unique values per column allowed.
                Prevents memory issues with high-cardinality data. Default is 100.
            **kwargs: Additional arguments passed to FlexibleBinningBase parent class.
                
        Note:
            The fit_jointly parameter is automatically disabled for this transformer
            as it's incompatible with the one-hot binning approach.
        """
        # Remove fit_jointly from kwargs if present to avoid conflicts
        kwargs.pop('fit_jointly', None)

        super().__init__(
            bin_spec=bin_spec,
            bin_representatives=bin_representatives,
            preserve_dataframe=preserve_dataframe,
            fit_jointly=False,  # Always use per-column fitting
            **kwargs,
        )
        self.max_unique_values = max_unique_values

    def _validate_params(self) -> None:
        """Validate OneHotBinning specific parameters.
        
        Raises:
            ValueError: If max_unique_values is not a positive integer.
        """
        super()._validate_params()

        if not isinstance(self.max_unique_values, int) or self.max_unique_values <= 0:
            raise ValueError("max_unique_values must be a positive integer")

    def _calculate_flexible_bins(
        self, x_col: np.ndarray, col_id: ColumnId, guidance_data: Optional[np.ndarray] = None
    ) -> Tuple[FlexibleBinDefs, BinEdges]:
        """Calculate singleton bins for each unique value in the column.

        Creates one bin per unique value found in the data. Each bin is defined as
        {"singleton": value} and contains exactly one unique value. This method only
        supports numeric data and will raise an error for non-numeric inputs.

        Args:
            x_col (np.ndarray): Numeric data for a single column to analyze.
            col_id (ColumnId): Column identifier for error reporting.
            guidance_data (Optional[np.ndarray]): Optional guidance data, not used 
                in one-hot binning but kept for interface compatibility.

        Returns:
            Tuple[FlexibleBinDefs, BinEdges]: A tuple containing:
                - List of singleton bin definitions: [{"singleton": val1}, {"singleton": val2}, ...]
                - List of representative values: [val1, val2, ...]

        Raises:
            ValueError: If the column contains non-numeric data or if the number of
                unique values exceeds max_unique_values.
                
        Note:
            NaN and infinite values are filtered out before determining unique values.
            If all values are NaN/inf, a default bin with value 0.0 is created.
        """
        # Convert to numeric array - this will raise an error for non-numeric data
        try:
            x_col = np.asarray(x_col, dtype=float)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"OneHotBinning only supports numeric data. "
                f"Column {col_id} contains non-numeric values. "
                f"Original error: {str(e)}"
            ) from e

        # Remove NaN/inf values for finding unique values
        finite_mask = np.isfinite(x_col)
        if not finite_mask.any():
            # All values are NaN/inf - create a default bin
            return [{"singleton": 0.0}], [0.0]

        finite_values = x_col[finite_mask]
        unique_values = np.unique(finite_values)

        # Check if we have too many unique values
        if len(unique_values) > self.max_unique_values:
            raise ValueError(
                f"Column {col_id} has {len(unique_values)} unique values, "
                f"which exceeds max_unique_values={self.max_unique_values}. "
                f"Consider using a different binning method for high-cardinality data."
            )

        # Create singleton bins for each unique value
        bin_defs = []
        representatives = []

        for val in unique_values:
            val = float(val)  # Convert to Python float
            bin_defs.append({"singleton": val})
            representatives.append(val)

        return bin_defs, representatives
