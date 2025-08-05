"""
Base class for supervised binning methods with guidance data management.

This module provides the foundational SupervisedBinningBase class for all supervised
binning transformers. It handles guidance data validation, feature-target pair processing,
and automatic data quality validation.

The class supports both classification and regression tasks, with intelligent fallback
strategies for insufficient data scenarios and comprehensive data quality validation.
"""

import warnings
from typing import Any

import numpy as np

from ..utils.errors import (
    DataQualityWarning,
    ValidationError,
)
from ..utils.types import BinEdges
from ._interval_binning_base import IntervalBinningBase


# pylint: disable=too-many-ancestors
class SupervisedBinningBase(IntervalBinningBase):
    """
    Base class for supervised binning methods that use single guidance columns.

    This class provides:
    - Single guidance column validation and preprocessing
    - Feature-target pair validation and missing value handling
    - Insufficient data handling with fallback strategies
    - Data quality warnings for both features and targets
    """

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Initialize SupervisedBinningBase with guidance data management.

        Sets up the supervised binning transformer for guidance data management.
        This base class focuses purely on handling guidance columns and validation,
        without any task-specific logic like decision trees.

        Args:
            **kwargs: Additional keyword arguments passed to the parent IntervalBinningBase
                class, including clip, bin_edges, bin_representatives, preserve_dataframe,
                fit_jointly, and guidance_columns.

        Example:
            >>> # Basic usage (typically subclassed)
            >>> binner = SupervisedBinning(guidance_columns=[3])

        Note:
            - This is a base class meant to be subclassed by specific implementations
            - Guidance columns can be modified after initialization using set_params()
            - Data validation is performed during fit operations
        """
        super().__init__(**kwargs)

    def validate_guidance_data(
        self, guidance_data: np.ndarray[Any, Any], name: str = "guidance_data"
    ) -> np.ndarray[Any, Any]:
        """Validate and preprocess guidance data for supervised binning.

        Ensures that the guidance data is appropriate for supervised binning
        by validating its shape, checking for single column requirement,
        and converting to the proper format for decision tree training.

        SupervisedBinning requires exactly one guidance column, unlike other
        binning methods that may support multiple guidance columns.

        Args:
            guidance_data (np.ndarray[Any, Any]): Raw guidance/target data to validate.
                Should be a 2D array with shape (n_samples, 1) or 1D array
                with shape (n_samples,).
            name (str, optional): Name used in error messages for better
                debugging context. Defaults to "guidance_data".

        Returns:
            np.ndarray[Any, Any]: Validated 1D guidance data with shape (n_samples,).
                Ready for use with decision tree estimators.

        Raises:
            ValueError: If guidance_data has wrong shape (not single column)
                or if the data format is incompatible with supervised learning.

        Example:
            >>> guidance = np.array([[0], [1], [0], [1]])  # 2D single column
            >>> validated = binner.validate_guidance_data(guidance)
            >>> print(validated.shape)  # (4,)

        Note:
            - Converts 2D single-column arrays to 1D arrays
            - Preserves the original data values and types
            - Used internally during fit() to prepare target data
        """
        # Basic validation
        guidance_validated = self.validate_array_like(guidance_data, name)
        guidance_validated = np.asarray(guidance_validated)

        # Check data quality
        self.check_data_quality(guidance_validated, name)

        # Handle dimensionality - supervised binning expects single column
        if guidance_validated.ndim == 1:
            return guidance_validated
        if guidance_validated.ndim == 2:
            if guidance_validated.shape[1] != 1:
                raise ValidationError(
                    f"{name} has {guidance_validated.shape[1]} columns, "
                    f"expected exactly 1. Supervised binning requires a single guidance column. "
                    f"Please specify the correct guidance column."
                )
            # Flatten to 1D for easier processing
            return guidance_validated.ravel()
        raise ValidationError(
            f"{name} has {guidance_validated.ndim} dimensions, "
            f"expected 1D or 2D array with single column"
        )

    def validate_feature_target_pair(
        self, x_col: np.ndarray[Any, Any], guidance_data: np.ndarray[Any, Any], col_id: Any = None
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Validate feature-target pair and create valid data mask.

        Performs comprehensive validation of a feature-target pair for supervised
        binning, ensuring data quality and compatibility. Creates a mask to identify
        valid data points where both feature and target values are present and valid.

        This method handles missing values, data type conversions, and provides
        detailed validation for both numeric and categorical target variables.

        Args:
            x_col (np.ndarray[Any, Any]): Feature column data to be validated and converted.
                Can contain missing values (NaN) which will be identified in the mask.
            guidance_data (np.ndarray[Any, Any]): Target/guidance data that must be 1D after
                validation. Can be numeric (for regression) or categorical
                (for classification).
            col_id (Any, optional): Column identifier used in error messages for
                better debugging context. Defaults to None.

        Returns:
            Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]: A tuple containing:
                - x_col: Validated feature data as float array (shape: n_samples)
                - guidance_data: Validated guidance data (shape: n_samples)
                - valid_mask: Boolean mask indicating valid (non-missing) pairs (shape: n_samples)

        Note:
            - Feature data is converted to float64 for numeric operations
            - Valid mask identifies rows where both feature and target are non-missing
            - Categorical targets are handled appropriately without conversion
            - Used internally during fitting to prepare clean data pairs
        """
        # Validate inputs
        x_col_validated = self.validate_array_like(x_col, f"feature column {col_id}")
        guidance_data_validated = self.validate_guidance_data(guidance_data)

        # Check data quality (validate_array_like should not return None here)
        assert x_col_validated is not None, "x_col_validated should not be None"
        self.check_data_quality(x_col_validated, f"feature column {col_id}")
        self.check_data_quality(guidance_data_validated, "guidance data")

        # Convert feature to float for numeric operations
        x_col = np.asarray(x_col_validated, dtype=float)

        # Check that feature and guidance data have the same length
        if len(x_col) != len(guidance_data_validated):
            raise ValidationError(
                f"Feature column {col_id} has {len(x_col)} samples, "
                f"but guidance data has {len(guidance_data_validated)} samples. "
                f"Both must have the same number of samples."
            )

        # Create valid data mask (both feature and target must be non-missing)
        feature_finite = np.isfinite(x_col)

        if guidance_data_validated.dtype == object:
            # Handle object dtype (e.g., strings, mixed types)
            guidance_finite = np.array(
                [
                    val is not None and not (isinstance(val, float) and np.isnan(val))
                    for val in guidance_data_validated
                ]
            )
        else:
            # Numeric dtype
            guidance_finite = np.isfinite(guidance_data_validated.astype(float))

        valid_mask = feature_finite & guidance_finite

        return x_col, guidance_data_validated, valid_mask

    def extract_valid_pairs(
        self, x_col: np.ndarray[Any, Any], guidance_data: np.ndarray[Any, Any], valid_mask: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Extract valid feature-target pairs using the provided mask.

        Filters the feature and target data to include only valid (non-missing)
        pairs as identified by the provided boolean mask. This is essential for
        supervised learning algorithms that cannot handle missing values.

        Args:
            x_col (np.ndarray[Any, Any]): Feature column data with potential missing values.
                Shape should be (n_samples,).
            guidance_data (np.ndarray[Any, Any]): Target/guidance data with potential missing
                values. Shape should be (n_samples,).
            valid_mask (np.ndarray[Any, Any]): Boolean mask indicating which samples have
                valid (non-missing) feature-target pairs. Shape should be (n_samples,).

        Returns:
            Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]: A tuple containing:
                - Valid feature data (shape: n_valid_samples)
                - Valid target data (shape: n_valid_samples)

        Example:
            >>> x = np.array([1.0, np.nan, 3.0, 4.0])
            >>> y = np.array([0, 1, np.nan, 1])
            >>> mask = np.array([True, False, False, True])
            >>> x_valid, y_valid = binner.extract_valid_pairs(x, y, mask)
            >>> # x_valid = [1.0, 4.0], y_valid = [0, 1]

        Note:
            - Both arrays are filtered using the same mask for consistency
            - Result arrays have the same length (number of valid pairs)
            - Used internally after validate_feature_target_pair()
        """
        return x_col[valid_mask], guidance_data[valid_mask]

    def require_guidance_data(
        self, guidance_data: np.ndarray[Any, Any] | None, method_name: str = "supervised binning"
    ) -> None:
        """Ensure guidance data is provided for supervised methods.

        Validates that guidance data (target values) is available for supervised
        binning operations. This method provides a consistent way to enforce the
        requirement that supervised methods must have target data to guide the
        binning process.

        Args:
            guidance_data (Optional[np.ndarray[Any, Any]]): Guidance data to check for presence.
                None indicates that no guidance data was provided.
            method_name (str, optional): Name of the calling method used in error
                messages for better debugging context. Defaults to "supervised binning".

        Raises:
            ValueError: If guidance_data is None, indicating that supervised binning
                cannot proceed without target information.

        Example:
            >>> binner.require_guidance_data(None, "SupervisedBinning.fit")
            # ValueError: SupervisedBinning.fit requires guidance_data...

        Note:
            - Called early in supervised methods to provide clear error messages
            - Helps distinguish between programming errors and user errors
            - Provides guidance on how to specify guidance columns
        """
        if guidance_data is None:
            raise ValueError(
                f"{method_name.title()} requires guidance_data (target values) to be provided. "
                f"Please specify guidance_columns when creating the transformer."
            )

    def handle_insufficient_data(
        self, x_col: np.ndarray[Any, Any], valid_mask: np.ndarray[Any, Any], min_samples: int, col_id: Any = None
    ) -> tuple[BinEdges, BinEdges] | None:
        """Handle cases with insufficient valid data for supervised binning.

        Provides a fallback strategy when there are not enough valid feature-target
        pairs to perform meaningful supervised binning. This method determines
        whether to proceed with a fallback strategy or return None to indicate
        that binning is not possible.

        The method implements intelligent fallback logic that considers data
        distribution and provides appropriate warnings to inform users about
        the data quality issues.

        Args:
            x_col (np.ndarray[Any, Any]): Feature column data that may contain missing values.
                Used to compute fallback bin boundaries if needed.
            valid_mask (np.ndarray[Any, Any]): Boolean mask indicating which samples have
                valid feature-target pairs.
            min_samples (int): Minimum number of valid samples required for
                supervised binning to proceed normally.
            col_id (Any, optional): Column identifier used in warning messages
                for better debugging context. Defaults to None.

        Returns:
            Optional[Tuple[BinEdges, BinEdges]]: A tuple containing fallback bin
                edges and representatives if a fallback is possible, or None if
                no reasonable fallback can be constructed.

        Note:
            - Issues data quality warnings when fallbacks are used
            - Considers both the number of valid samples and data distribution
            - Used internally when validate_feature_target_pair finds insufficient data
            - May return None to indicate binning should be skipped entirely
        """

        n_valid = valid_mask.sum()

        # Check if we have sufficient data to continue with normal processing
        if n_valid >= min_samples:
            return None

        # Insufficient data - create fallback bins with appropriate warning
        if n_valid == 0:
            # No valid data - create default range
            min_val = np.nanmin(x_col) if not np.isnan(x_col).all() else 1.0
            max_val = np.nanmax(x_col) if not np.isnan(x_col).all() else 1.0
            warning_msg = "has no valid data points"
        else:
            # Some valid data but insufficient for complex binning
            valid_data = x_col[valid_mask]
            min_val = np.min(valid_data)
            max_val = np.max(valid_data)
            warning_msg = (
                f"has only {n_valid} valid samples (minimum {min_samples} required)."
                " Creating single bin"
            )

        # Ensure we have a valid range
        if min_val == max_val:
            max_val = min_val + 1.0

        # Issue warning with appropriate column reference
        if col_id is not None:
            col_ref = (
                f"column {col_id}" if isinstance(col_id, int | np.integer) else f"column '{col_id}'"
            )
            warnings.warn(
                f"Data in {col_ref} {warning_msg}. "
                f"Using {'default ' if n_valid == 0 else ''}bin range [{min_val}, {max_val}]",
                DataQualityWarning,
                stacklevel=2,
            )

        return [min_val, max_val], [(min_val + max_val) / 2]

    def create_fallback_bins(
        self, x_col: np.ndarray[Any, Any], default_range: tuple[float, float] | None = None
    ) -> tuple[BinEdges, BinEdges]:
        """Create fallback bins when supervised binning fails.

        Constructs simple fallback bin specifications when supervised binning
        cannot be performed due to data quality issues or insufficient samples.
        This method ensures that some form of binning can always be applied,
        even when the ideal supervised approach is not feasible.

        The fallback strategy creates a single bin that encompasses the data
        range, providing a reasonable default that preserves the data structure
        while indicating that more sophisticated binning was not possible.

        Args:
            x_col (np.ndarray[Any, Any]): Feature column data from which to infer the
                appropriate range for fallback bins.
            default_range (Optional[Tuple[float, float]], optional): Explicit
                range to use for the fallback bins as (min_val, max_val).
                If None, the range is inferred from the valid data in x_col.
                Defaults to None.

        Returns:
            Tuple[BinEdges, BinEdges]: A tuple containing:
                - Bin edges list with [min_val, max_val]
                - Bin representatives list with [midpoint]

        Example:
            >>> x = np.array([1.0, 2.0, 3.0, np.nan])
            >>> edges, reps = binner.create_fallback_bins(x)
            >>> # edges = [1.0, 3.0], reps = [2.0]

        Note:
            - Creates a single bin covering the data range
            - Handles constant data by adding a small offset
            - Uses (0.0, 1.0) as ultimate fallback for completely invalid data
            - Representative is the midpoint of the range
        """
        if default_range is not None:
            min_val, max_val = default_range
        else:
            # Infer range from valid data
            finite_mask = np.isfinite(x_col)
            if finite_mask.any():
                min_val = np.min(x_col[finite_mask])
                max_val = np.max(x_col[finite_mask])
            else:
                min_val, max_val = 0.0, 1.0

        if min_val == max_val:
            max_val = min_val + 1.0

        return [min_val, max_val], [(min_val + max_val) / 2]
