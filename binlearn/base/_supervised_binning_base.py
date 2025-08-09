"""
Clean supervised binning base class for V2 architecture.

This module provides supervised binning functionality that inherits from IntervalBinningBase.
For binning methods that use target/label information to optimize bin boundaries.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ..utils import (
    ArrayLike,
    BinEdgesDict,
    ColumnList,
    DataQualityWarning,
    ValidationError,
)
from ._interval_binning_base import IntervalBinningBase


class SupervisedBinningBase(IntervalBinningBase):
    """Base class for supervised binning methods that use target information.

    This class extends IntervalBinningBase to provide specialized functionality for
    supervised binning methods. These methods use target variable information (y)
    to optimize bin boundaries, typically aiming to create bins with homogeneous
    target distributions or maximize predictive power.

    Supervised binning is particularly effective for:
    - Binary classification with continuous predictors
    - Regression tasks where binning should preserve target relationships
    - Feature selection and engineering based on target correlation
    - Creating interpretable bins aligned with target behavior

    Key Features:
    - Target-aware bin boundary optimization
    - Built-in target data validation and preprocessing
    - Feature-target pair validation for data quality
    - Automatic handling of supervised learning constraints
    - Integration with guidance column requirements for targets

    Constraints:
    - Does not support joint fitting across multiple features (fit_jointly=False)
    - Requires exactly one guidance column to serve as the target variable
    - Target data must be provided during fit() call
    - Feature and target data must have compatible shapes and no missing values

    Attributes:
        All attributes from IntervalBinningBase plus:
        - Target-specific validation and preprocessing capabilities
        - Enhanced error handling for supervised learning scenarios

    Example:
        >>> # Supervised binning for binary classification
        >>> X = np.array([[1.2, 2.3], [3.4, 4.5], [5.6, 6.7]])
        >>> y = np.array([0, 1, 0])  # Binary target
        >>>
        >>> binner = ConcreteSupervisedBinner()
        >>> binner.fit(X, guidance_data=y)
        >>> X_binned = binner.transform(X)

    Note:
        - This is an abstract base class - use concrete implementations like Chi2Binning
        - Inherits all interval binning functionality (bin edges, representatives, etc.)
        - Target data is passed via guidance_data parameter in fit() method
        - Subclasses must implement _calculate_bin_edges with target-aware logic
    """

    def __init__(
        self,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        guidance_columns: Any = None,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
    ):
        """Initialize supervised binning base class.

        Args:
            clip: Whether to clip out-of-range values during transformation to the
                nearest bin boundary. If True, values below the minimum edge are
                assigned to the first bin, and values above the maximum edge are
                assigned to the last bin. If False, out-of-range values get special
                indices (BELOW_RANGE, ABOVE_RANGE). If None, uses global default.
            preserve_dataframe: Whether to preserve the original DataFrame format
                during transformation. If True, returns DataFrame when input is DataFrame.
                If False, returns numpy array. If None, uses global configuration default.
            guidance_columns: Column identifier for the target variable. For supervised
                binning, this should specify exactly one column that contains the target
                values. Can be column name/index or None (target passed via guidance_data).
            bin_edges: Pre-defined bin edges as a dictionary mapping column identifiers
                to lists of edge values. If provided, no fitting is performed and these
                edges are used directly. Must be compatible with supervised binning
                constraints if provided.
            bin_representatives: Pre-defined representative values for each bin as a
                dictionary mapping column identifiers to lists of values. Must match
                the structure of bin_edges if provided.

        Raises:
            ValidationError: If parameters are incompatible with supervised binning
                requirements (e.g., multiple guidance columns specified).

        Note:
            - Supervised binning does not support fit_jointly=True (always fits independently)
            - Target data should be provided via guidance_data parameter in fit() method
            - Only one guidance column is supported (the target variable)
            - Pre-defined bin edges should be optimized for the target if provided
        """
        # Initialize parent (supervised binning doesn't support fit_jointly)
        IntervalBinningBase.__init__(
            self,
            clip=clip,
            preserve_dataframe=preserve_dataframe,
            fit_jointly=False,  # Supervised binning always processes columns independently
            guidance_columns=guidance_columns,
            bin_edges=bin_edges,
            bin_representatives=bin_representatives,
        )

    def _validate_params(self) -> None:
        """Validate supervised binning parameters with additional constraints.

        Extends the parent class validation to add supervised learning specific
        constraints and warnings. This method ensures that the parameter configuration
        is appropriate for supervised binning methods.

        Warns:
            DataQualityWarning: If multiple guidance columns are specified, which
                may lead to unexpected behavior in supervised binning methods that
                typically expect a single target variable.

        Note:
            - Calls parent class parameter validation first
            - Supervised binning typically works best with exactly one target column
            - Multiple guidance columns are allowed but discouraged with a warning
            - This method is called during initialization to catch configuration issues early
        """
        # Call parent validation
        IntervalBinningBase._validate_params(self)

        # Additional validation for supervised binning
        if self.guidance_columns is not None:
            # For supervised binning, we typically expect a single guidance column (the target)
            if isinstance(self.guidance_columns, list) and len(self.guidance_columns) > 1:
                warnings.warn(
                    "Supervised binning typically works best with a single target column. "
                    "Multiple guidance columns may lead to unexpected behavior.",
                    DataQualityWarning,
                    stacklevel=2,
                )

    def validate_guidance_data(
        self, guidance_data: ArrayLike, name: str = "guidance_data"
    ) -> np.ndarray[Any, Any]:
        """Validate and preprocess guidance data for supervised binning.

        Ensures that the guidance data is appropriate for supervised binning
        by validating its shape and checking for data quality issues.

        Args:
            guidance_data: Raw guidance/target data to validate.
                Should be a 2D array with shape (n_samples, 1) or 1D array
                with shape (n_samples,).
            name: Name used in error messages for better debugging context.

        Returns:
            Validated and preprocessed guidance data with shape (n_samples, 1).

        Raises:
            ValidationError: If guidance data has invalid shape or format.
        """
        if guidance_data is None:
            raise ValidationError(f"{name} cannot be None for supervised binning")

        # Convert to numpy array if needed
        if not isinstance(guidance_data, np.ndarray):
            guidance_data = np.array(guidance_data)

        # Ensure 2D shape
        if guidance_data.ndim == 1:
            guidance_data = guidance_data.reshape(-1, 1)
        elif guidance_data.ndim > 2:
            raise ValidationError(f"{name} must be 1D or 2D array, got {guidance_data.ndim}D")

        # Check for empty data
        if guidance_data.size == 0:
            raise ValidationError(f"{name} cannot be empty")

        # Supervised binning requires exactly one guidance column
        if guidance_data.shape[1] != 1:
            raise ValidationError(
                f"Supervised binning requires exactly one target column, got {guidance_data.shape[1]} columns"
            )

        return guidance_data  # type: ignore[no-any-return]

    def _validate_feature_target_pair(
        self,
        feature_data: np.ndarray[Any, Any],
        target_data: np.ndarray[Any, Any],
        col_id: Any,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Validate and clean feature-target data pairs for supervised binning.

        This method ensures that feature and target data are compatible for supervised
        binning by checking shapes and removing rows with invalid target values. The
        feature data should already be preprocessed by the parent class, but target
        data may still contain missing or invalid values that need handling.

        Args:
            feature_data: Preprocessed feature column data that has been validated
                and cleaned by the parent class. Should contain only finite values.
            target_data: Raw target/guidance data that may contain NaN or infinite
                values. Must have the same number of rows as feature_data.
            col_id: Column identifier used for generating informative error and
                warning messages.

        Returns:
            Tuple containing:
            - cleaned_feature_data: Feature data with rows removed where target had invalid values
            - cleaned_target_data: Target data with invalid values (NaN, inf) removed
            Both arrays have the same length and correspond row-wise.

        Raises:
            ValidationError: If feature and target data have mismatched lengths.

        Warns:
            DataQualityWarning: If a significant number of rows are removed due to
                invalid target values (>5% of data OR >5 rows), or if insufficient
                valid data remains after cleaning.

        Example:
            >>> feature = np.array([1.0, 2.0, 3.0, 4.0])
            >>> target = np.array([0.0, np.nan, 1.0, 0.0])
            >>> clean_feat, clean_targ = binner._validate_feature_target_pair(
            ...     feature, target, 'col1'
            ... )
            >>> # Result: clean_feat = [1.0, 3.0, 4.0], clean_targ = [0.0, 1.0, 0.0]

        Note:
            - Feature data is assumed to be already cleaned by parent class validation
            - Only removes rows based on target data validity
            - Maintains row correspondence between feature and target data
            - Warns if data quality may be compromised by excessive missing values
            - Requires at least 2 valid samples to proceed with binning
        """
        if len(feature_data) != len(target_data):
            raise ValidationError(
                f"Feature and target data must have same length for column {col_id}. "
                f"Got feature: {len(feature_data)}, target: {len(target_data)}"
            )

        # Remove rows where target has missing values (feature data is already preprocessed)
        target_valid = ~(np.isnan(target_data.flatten()) | np.isinf(target_data.flatten()))

        cleaned_feature = feature_data[target_valid]
        cleaned_target = target_data[target_valid]

        # Warn if missing values were removed, but only if some valid data remains
        # and if a significant portion was removed (more than 5% OR more than 5 rows)
        removed_count = len(feature_data) - len(cleaned_feature)
        if removed_count > 0 and len(cleaned_feature) >= 2:
            removal_ratio = removed_count / len(feature_data)
            if removal_ratio > 0.05 or removed_count > 5:
                warnings.warn(
                    f"Column {col_id}: Removed {removed_count} rows with missing values in target data. "
                    f"Using {len(cleaned_feature)} valid samples for binning.",
                    DataQualityWarning,
                    stacklevel=2,
                )
            else:
                # Explicit else branch: missing values removed but below warning thresholds
                # This makes the 157->166 branch explicitly testable
                pass

        # Check if we have sufficient data after cleaning
        if len(cleaned_feature) < 2:
            warnings.warn(
                f"Column {col_id} has insufficient valid data points ({len(cleaned_feature)}) "
                "after removing missing values. Results may be unreliable.",
                DataQualityWarning,
                stacklevel=2,
            )

        return cleaned_feature, cleaned_target

    def _fit_per_column_independently(
        self,
        X: np.ndarray[Any, Any],
        columns: ColumnList,
        guidance_data: np.ndarray[Any, Any] | None = None,
        **fit_params: Any,
    ) -> None:
        """Fit supervised binning parameters independently for each column.

        This method implements the fitting logic for supervised binning by validating
        the guidance data (target) and delegating to the parent class implementation.
        It ensures that target information is properly formatted before bin computation.

        Args:
            X: Input feature data array with shape (n_samples, n_features) where
                each column will be binned using target information.
            columns: List of column identifiers corresponding to the columns in X.
                Used for bin specification keys and error messages.
            guidance_data: Target/label data that guides the binning process. Must
                be provided for supervised binning and should have shape (n_samples,)
                or (n_samples, 1). Cannot be None.
            **fit_params: Additional parameters passed to the underlying bin
                calculation methods.

        Raises:
            ValidationError: If guidance_data is None or has invalid format for
                supervised binning (wrong shape, multiple columns, etc.).

        Note:
            - Validates that guidance_data is provided (required for supervised binning)
            - Ensures guidance_data has exactly one column (the target variable)
            - Delegates actual fitting to parent class after target validation
            - Each feature column is fitted independently using the same target data
            - The target data is validated once and used for all feature columns
        """
        if guidance_data is None:
            raise ValidationError("Supervised binning requires guidance data (target labels)")

        # Validate that guidance data has exactly one column
        validated_guidance = self.validate_guidance_data(guidance_data, "guidance_data")

        # Call parent implementation with validated single-column guidance data
        super()._fit_per_column_independently(X, columns, validated_guidance, **fit_params)
