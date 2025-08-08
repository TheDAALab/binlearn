"""
Clean general binning base class for V2 architecture.

This module provides the core binning orchestration logic with guidance support.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
from sklearn.base import TransformerMixin

from ..config import get_config
from ..utils import ArrayLike, BinningError, ColumnList, GuidanceColumns
from ._data_handling_base import DataHandlingBase


class GeneralBinningBase(
    ABC,
    DataHandlingBase,
    TransformerMixin,  # type: ignore[misc,unused-ignore]
):
    """Clean binning base class focusing on orchestration and guidance logic.

    Handles:
    - Joint vs per-column fitting strategies
    - Guidance column separation and management
    - Binning transformation pipeline orchestration
    - Abstract interface for binning methods
    """

    def __init__(
        self,
        preserve_dataframe: bool | None = None,
        fit_jointly: bool | None = None,
        guidance_columns: GuidanceColumns = None,
    ):
        """Initialize the binning transformer."""
        DataHandlingBase.__init__(self)
        TransformerMixin.__init__(self)

        # Load configuration defaults
        config = get_config()

        # Apply configuration defaults
        if preserve_dataframe is None:
            preserve_dataframe = config.preserve_dataframe
        if fit_jointly is None:
            fit_jointly = config.fit_jointly

        # Validate parameter compatibility
        if guidance_columns is not None and fit_jointly:
            raise ValueError(
                "guidance_columns and fit_jointly=True are incompatible. "
                "Use either guidance_columns for per-record guidance OR "
                "fit_jointly=True for global fitting, but not both."
            )

        # Store binning-specific parameters
        self.preserve_dataframe = preserve_dataframe
        self.fit_jointly = fit_jointly
        self.guidance_columns = guidance_columns

        # Note: binning and guidance columns are computed dynamically
        # from feature_names_in_ and guidance_columns when needed

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> GeneralBinningBase:
        """Fit the binning transformer with comprehensive orchestration."""
        try:
            # Step 1: Parameter validation
            self._validate_params()

            # Step 2: Runtime validation for mutually exclusive parameters
            guidance_data_provided = fit_params.get("guidance_data") is not None
            if self.fit_jointly and guidance_data_provided:
                raise ValueError(
                    "Cannot use both fit_jointly=True and guidance_data parameter. "
                    "These are mutually exclusive: fit_jointly uses all data together, "
                    "while guidance_data provides separate guidance per column."
                )

            # Step 3: Input validation and feature information extraction
            self._validate_and_prepare_input(X, "X")
            self._extract_and_validate_feature_info(X, reset=True)

            # Step 4: Column separation for guidance handling
            X_binning, X_guidance, binning_cols, _ = self._separate_binning_and_guidance_columns(X)

            # Step 5: Route to appropriate fitting strategy
            if self.fit_jointly:
                self._fit_jointly_across_columns(X_binning, binning_cols, **fit_params)
            else:
                # Handle guidance data resolution with priority order
                final_guidance_data = self._resolve_guidance_data_priority(
                    X_guidance, fit_params.pop("guidance_data", None), y
                )

                self._fit_per_column_independently(
                    X_binning, binning_cols, final_guidance_data, **fit_params
                )

            return self

        except Exception as e:
            if isinstance(e, BinningError | ValueError | RuntimeError | NotImplementedError):
                raise
            raise ValueError(f"Failed to fit binning model: {str(e)}") from e

    def transform(self, X: Any) -> Any:
        """Transform input data using fitted binning parameters."""
        try:
            # Step 1: Validation checks
            self._check_fitted()
            self._validate_and_prepare_input(X, "X")

            # Step 2: Column separation and transformation
            X_binning, _, binning_cols, _ = self._separate_binning_and_guidance_columns(X)

            if self.guidance_columns is None:
                # Simple case: transform all columns
                result = self._transform_columns_to_bins(X_binning, binning_cols)
                return self._format_output_like_input(
                    result, X, binning_cols, self.preserve_dataframe
                )

            # Guided case: transform only binning columns
            if X_binning.shape[1] > 0:
                result = self._transform_columns_to_bins(X_binning, binning_cols)
            else:
                result = np.empty((X_binning.shape[0], 0), dtype=int)

            return self._format_output_like_input(result, X, binning_cols, self.preserve_dataframe)

        except Exception as e:
            if isinstance(e, BinningError | RuntimeError):
                raise
            raise ValueError(f"Failed to transform data: {str(e)}") from e

    def inverse_transform(self, X: Any) -> Any:
        """Inverse transform from bin indices back to representative values."""
        try:
            self._check_fitted()
            self._validate_and_prepare_input(X, "X")

            arr, columns = self._prepare_input(X)

            # Validate expected column count for guided binning
            if self.guidance_columns is not None:
                expected_cols = self._get_feature_count(include_guidance=False)
                if len(columns) != expected_cols:
                    raise ValueError(
                        f"Input for inverse_transform should have {expected_cols} "
                        f"columns (binning columns only), got {len(columns)}"
                    )

            result = self._inverse_transform_bins_to_values(arr, columns)
            return self._format_output_like_input(result, X, columns, self.preserve_dataframe)

        except Exception as e:
            if isinstance(e, BinningError | RuntimeError):
                raise
            raise ValueError(f"Failed to inverse transform data: {str(e)}") from e

    def _resolve_guidance_data_priority(
        self, X_guidance: np.ndarray[Any, Any] | None, external_guidance: Any, y: Any
    ) -> np.ndarray[Any, Any] | None | Any:
        """Resolve guidance data with clear priority order.

        Priority: X_guidance > external_guidance > y

        Args:
            X_guidance: Guidance columns from input X.
            external_guidance: Explicit guidance_data parameter.
            y: Target values (sklearn convenience).

        Returns:
            Resolved guidance data array or None.
        """
        if X_guidance is not None:
            return X_guidance

        if external_guidance is not None:
            return external_guidance

        if y is not None:
            y_array = np.asarray(y)
            if y_array.ndim == 1:
                y_array = y_array.reshape(-1, 1)
            # mypy doesn't understand that np.asarray returns the right type
            return y_array

        return None

    def _separate_binning_and_guidance_columns(
        self, X: ArrayLike
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any] | None, ColumnList, ColumnList | None]:
        """Separate input into binning and guidance columns.

        Core logic for handling guided vs unguided binning scenarios.

        Args:
            X: Input data with both binning and guidance columns.

        Returns:
            Tuple of (X_binning, X_guidance, binning_columns, guidance_columns).
        """
        arr, columns = self._prepare_input(X)

        if self.guidance_columns is None:
            # No guidance - all columns are binning columns
            return arr, None, columns, None

        # Normalize guidance_columns to list
        guidance_cols = (
            [self.guidance_columns]
            if not isinstance(self.guidance_columns, list)
            else self.guidance_columns
        )

        # Separate columns
        binning_indices = []
        guidance_indices = []
        binning_column_names = []
        guidance_column_names = []

        for i, col in enumerate(columns):
            if col in guidance_cols:
                guidance_indices.append(i)
                guidance_column_names.append(col)
            else:
                binning_indices.append(i)
                binning_column_names.append(col)

        # Extract data arrays
        X_binning = arr[:, binning_indices] if binning_indices else np.empty((arr.shape[0], 0))
        X_guidance = arr[:, guidance_indices] if guidance_indices else None

        # Don't store resolved column information - compute dynamically as needed
        return X_binning, X_guidance, binning_column_names, guidance_column_names

    def _get_feature_count(self, include_guidance: bool = True) -> int:
        """Get feature count with optional guidance exclusion."""
        n_features = getattr(self, "_n_features_in", 0)

        if not include_guidance and self.guidance_columns is not None:
            # Compute guidance column count dynamically
            guidance_cols = (
                [self.guidance_columns]
                if not isinstance(self.guidance_columns, list)
                else self.guidance_columns
            )
            return n_features - len(guidance_cols)

        return n_features

    def _get_binning_columns(self) -> list[Any] | None:
        """Compute binning columns dynamically from feature_names_in_ and guidance_columns."""
        if (
            not hasattr(self, "feature_names_in_")
            or getattr(self, "feature_names_in_", None) is None
        ):
            return None

        if self.feature_names_in_ is None:
            return None
        all_features = list(self.feature_names_in_)

        if self.guidance_columns is None:
            return all_features

        # Normalize guidance_columns to list
        guidance_cols = (
            [self.guidance_columns]
            if not isinstance(self.guidance_columns, list)
            else self.guidance_columns
        )

        # Return features that are not guidance columns (guidance columns are used but not binned)
        return [col for col in all_features if col not in guidance_cols]

    def _validate_params(self) -> None:
        """Validate binning-specific parameters with clear error messages."""
        super()._validate_params()

        if self.preserve_dataframe is not None and not isinstance(self.preserve_dataframe, bool):
            raise TypeError("preserve_dataframe must be a boolean or None")

        if self.fit_jointly is not None and not isinstance(self.fit_jointly, bool):
            raise TypeError("fit_jointly must be a boolean or None")

        if self.guidance_columns is not None:
            if not isinstance(self.guidance_columns, list | tuple | int | str):
                raise TypeError("guidance_columns must be list, tuple, int, str, or None")

            # Guidance data and fit_jointly are mutually exclusive
            if self.fit_jointly:
                raise ValueError(
                    "fit_jointly=True cannot be used with guidance_columns. "
                    "Guidance-based fitting requires per-column processing."
                )

    def get_input_columns(self) -> ColumnList | None:
        """Get input columns for data preparation.

        This method should be overridden by derived classes to provide
        appropriate column information without exposing binning-specific concepts.

        Returns:
            Column information or None if not available
        """
        return self._get_binning_columns()

    # Abstract methods for subclasses - renamed for clarity
    @abstractmethod
    def _fit_per_column_independently(
        self,
        X: np.ndarray[Any, Any],
        columns: ColumnList,
        guidance_data: ArrayLike | None = None,
        **fit_params: Any,
    ) -> None:
        """Fit binning parameters independently for each column.

        Args:
            X: Input data for binning columns.
            columns: Column identifiers for binning columns.
            guidance_data: Optional guidance data for supervised binning.
            **fit_params: Additional fitting parameters.
        """
        raise NotImplementedError("Subclasses must implement _fit_per_column_independently")

    @abstractmethod
    def _fit_jointly_across_columns(
        self, X: np.ndarray[Any, Any], columns: ColumnList, **fit_params: Any
    ) -> None:
        """Fit binning parameters jointly across all columns.

        Args:
            X: Input data for all binning columns.
            columns: Column identifiers for all columns.
            **fit_params: Additional fitting parameters.
        """
        raise NotImplementedError("Subclasses must implement _fit_jointly_across_columns")

    @abstractmethod
    def _transform_columns_to_bins(
        self, X: np.ndarray[Any, Any], columns: ColumnList
    ) -> np.ndarray[Any, Any]:
        """Transform columns to bin indices.

        Args:
            X: Input data to transform.
            columns: Column identifiers.

        Returns:
            Transformed data with bin indices.
        """
        raise NotImplementedError("Subclasses must implement _transform_columns_to_bins")

    @abstractmethod
    def _inverse_transform_bins_to_values(
        self, X: np.ndarray[Any, Any], columns: ColumnList
    ) -> np.ndarray[Any, Any]:
        """Inverse transform from bin indices to representative values.

        Args:
            X: Binned data to inverse transform.
            columns: Column identifiers.

        Returns:
            Data with representative values.
        """
        raise NotImplementedError("Subclasses must implement _inverse_transform_bins_to_values")
