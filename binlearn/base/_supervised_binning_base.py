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
    """Supervised binning functionality inheriting from IntervalBinningBase.

    For binning methods that use target/label information (y) to optimize bin boundaries.

    Provides:
    - Target data validation and preprocessing
    - Feature-target pair validation
    - Single guidance column requirement for supervised learning
    - Data quality warnings for insufficient data scenarios
    """

    def __init__(
        self,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        guidance_columns: Any = None,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
    ):
        """Initialize supervised binning base."""
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
        """Validate supervised binning parameters."""
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
        """Validate and preprocess feature-target pairs for supervised binning.

        The feature data has already been preprocessed by the base class,
        but we still need to handle NaN/inf values in the target data.

        Args:
            feature_data: Preprocessed feature column data (from base class)
            target_data: Target/guidance data (may contain NaN/inf)
            col_id: Column identifier for error messages

        Returns:
            Tuple of (cleaned_feature_data, cleaned_target_data) with invalid target rows removed

        Raises:
            ValidationError: If data shapes don't match or insufficient data remains
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
        """Fit supervised binning parameters independently for each column."""
        if guidance_data is None:
            raise ValidationError("Supervised binning requires guidance data (target labels)")

        # Validate that guidance data has exactly one column
        validated_guidance = self.validate_guidance_data(guidance_data, "guidance_data")

        # Call parent implementation with validated single-column guidance data
        super()._fit_per_column_independently(X, columns, validated_guidance, **fit_params)
