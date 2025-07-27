"""
Simplified base class for all binning methods.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ._data_utils import return_like_input, prepare_input_with_columns
from ..config import get_config
from ..errors import ValidationMixin, BinningError, InvalidDataError
from ..sklearn_utils import SklearnCompatibilityMixin


class GeneralBinningBase(
    ABC, BaseEstimator, TransformerMixin, ValidationMixin, SklearnCompatibilityMixin
):
    """Base binning class with universal guidance support."""

    def __init__(
        self,
        preserve_dataframe: Optional[bool] = None,
        fit_jointly: Optional[bool] = None,
        guidance_columns: Optional[Union[List[Any], Any]] = None,
        **kwargs,
    ):
        # Load configuration defaults
        config = get_config()

        # Apply defaults from configuration
        if preserve_dataframe is None:
            preserve_dataframe = config.preserve_dataframe
        if fit_jointly is None:
            fit_jointly = config.fit_jointly

        # Validate incompatible parameters
        if guidance_columns is not None and fit_jointly:
            raise ValueError(
                "guidance_columns and fit_jointly=True are incompatible. "
                "Use either guidance_columns for per-record guidance OR "
                "fit_jointly=True for global fitting, but not both."
            )

        self.preserve_dataframe = preserve_dataframe
        self.fit_jointly = fit_jointly
        self.guidance_columns = guidance_columns

        # Internal state
        self._fitted = False
        self._binning_columns = None
        self._guidance_columns = None
        self._original_columns = None
        self._n_features_in = None
        self._feature_names_in = None

    def _prepare_input(self, X: Any) -> Tuple[np.ndarray, List[Any]]:
        """Prepare input array and determine column identifiers."""
        return prepare_input_with_columns(
            X, fitted=self._fitted, original_columns=self._original_columns
        )

    def _check_fitted(self) -> None:
        """Check if the estimator is fitted."""
        if not self._fitted:
            raise RuntimeError("This estimator is not fitted yet. Call 'fit' first.")

    def _separate_columns(
        self, X: Any
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[Any], List[Any]]:
        """Universal column separation logic."""
        arr, columns = self._prepare_input(X)

        if self.guidance_columns is None:
            # No guidance - all columns are binning columns
            return arr, None, columns, []

        # Normalize guidance_columns to list
        guidance_cols = (
            [self.guidance_columns]
            if not isinstance(self.guidance_columns, list)
            else self.guidance_columns
        )

        # Separate columns
        binning_indices = []
        guidance_indices = []
        binning_columns = []
        guidance_columns = []

        for i, col in enumerate(columns):
            if col in guidance_cols:
                guidance_indices.append(i)
                guidance_columns.append(col)
            else:
                binning_indices.append(i)
                binning_columns.append(col)

        # Extract data
        X_binning = arr[:, binning_indices] if binning_indices else np.empty((arr.shape[0], 0))
        X_guidance = arr[:, guidance_indices] if guidance_indices else None

        return X_binning, X_guidance, binning_columns, guidance_columns

    def fit(self, X: Any, y: Any = None, **fit_params) -> "GeneralBinningBase":
        """Universal fit method with guidance support."""
        try:
            # Validate parameters first
            self._validate_params()

            # Validate input data using ValidationMixin
            self.validate_array_like(X, "X")

            # Store original input info for sklearn compatibility
            arr, original_columns = self._prepare_input(X)
            self._n_features_in = arr.shape[1]
            self._original_columns = original_columns

            # Handle feature names manually to avoid sklearn conflicts
            if hasattr(X, "columns"):
                self._feature_names_in = list(X.columns)
            elif hasattr(X, "feature_names"):
                self._feature_names_in = list(X.feature_names)
            else:
                # For numpy arrays without column names, use integer indices for
                # backward compatibility
                self._feature_names_in = list(range(arr.shape[1]))

            # Separate guidance and binning columns
            X_binning, X_guidance, binning_cols, guidance_cols = self._separate_columns(X)

            # Store column information
            self._binning_columns = binning_cols
            self._guidance_columns = guidance_cols

            # Route to appropriate fitting method
            if self.fit_jointly:
                self._fit_jointly(X_binning, binning_cols, **fit_params)
            else:
                self._fit_per_column(X_binning, binning_cols, X_guidance, **fit_params)

            self._fitted = True
            return self

        except Exception as e:
            if isinstance(e, BinningError):
                raise
            if isinstance(e, (ValueError, RuntimeError, NotImplementedError)):
                # Let these pass through unchanged for test compatibility
                raise
            raise ValueError(f"Failed to fit binning model: {str(e)}") from e

    def transform(self, X: Any) -> Any:
        """Universal transform with guidance column handling."""
        try:
            self._check_fitted()
            # Validate input data
            self.validate_array_like(X, "X")
            # Check feature names consistency
            self._check_feature_names(X, reset=False)
            # Separate columns
            X_binning, X_guidance, binning_cols, guidance_cols = self._separate_columns(X)
            if self.guidance_columns is None:
                # No guidance - transform all columns
                result = self._transform_columns(X_binning, binning_cols)
                return return_like_input(result, X, binning_cols, bool(self.preserve_dataframe))
            # Transform only binning columns
            if X_binning.shape[1] > 0:
                result = self._transform_columns(X_binning, binning_cols)
            else:
                result = np.empty((X_binning.shape[0], 0), dtype=int)
            return return_like_input(result, X, binning_cols, bool(self.preserve_dataframe))
        except Exception as e:
            if isinstance(e, (BinningError, RuntimeError)):
                raise
            raise ValueError(f"Failed to transform data: {str(e)}") from e

    def transform_with_guidance(self, X: Any) -> Tuple[Any, Any]:
        """Transform and return both binned and guidance data separately."""
        try:
            self._check_fitted()

            # Validate input data
            self.validate_array_like(X, "X")

            X_binning, X_guidance, binning_cols, guidance_cols = self._separate_columns(X)

            # Transform binning columns
            if X_binning.shape[1] > 0:
                binned_result = self._transform_columns(X_binning, binning_cols)
            else:
                binned_result = np.empty((X_binning.shape[0], 0), dtype=int)

            # Format outputs
            binned_output = return_like_input(
                binned_result, X, binning_cols, bool(self.preserve_dataframe)
            )

            if X_guidance is not None:
                guidance_output = return_like_input(
                    X_guidance, X, guidance_cols, bool(self.preserve_dataframe)
                )
            else:
                guidance_output = None

            return binned_output, guidance_output

        except (ValueError, RuntimeError):
            # Let these pass through unchanged for test compatibility
            raise
        except Exception as e:
            if isinstance(e, BinningError):
                raise
            raise InvalidDataError(f"Failed to transform data with guidance: {str(e)}") from e

    def inverse_transform(self, X: Any) -> Any:
        """Inverse transform from bin indices back to representative values."""
        try:
            self._check_fitted()

            # Validate input data
            self.validate_array_like(X, "X")

            # For inverse transform, we work only with binning columns
            # (guidance columns weren't transformed, so can't be inverse transformed)
            if self.guidance_columns is not None:
                # Input should only have binning columns for inverse transform
                arr, columns = self._prepare_input(X)
                if self._binning_columns is None or len(columns) != len(self._binning_columns):
                    expected_cols = (
                        len(self._binning_columns) if self._binning_columns is not None else 0
                    )
                    raise ValueError(
                        f"Input for inverse_transform should have {expected_cols} "
                        f"columns (binning columns only), got {len(columns)}"
                    )
                result = self._inverse_transform_columns(arr, self._binning_columns)
                return return_like_input(
                    result, X, self._binning_columns, bool(self.preserve_dataframe)
                )
            else:
                # No guidance - inverse transform all columns
                arr, columns = self._prepare_input(X)
                result = self._inverse_transform_columns(arr, columns)
                return return_like_input(result, X, columns, bool(self.preserve_dataframe))

        except Exception as e:
            if isinstance(e, (BinningError, RuntimeError)):
                raise
            raise ValueError(f"Failed to inverse transform data: {str(e)}") from e

    # Abstract methods to be implemented by subclasses
    @abstractmethod
    def _fit_per_column(
        self,
        X: np.ndarray,
        columns: List[Any],
        guidance_data: Optional[np.ndarray] = None,
        **fit_params,
    ) -> None:
        """Fit bins per column with optional guidance."""
        raise NotImplementedError("Subclasses must implement _fit_per_column method.")

    @abstractmethod
    def _fit_jointly(self, X: np.ndarray, columns: List[Any], **fit_params) -> None:
        """Fit bins jointly (guidance incompatible, so no guidance_data parameter)."""
        raise NotImplementedError(
            "Joint fitting not implemented. Subclasses should override this method."
        )

    @abstractmethod
    def _transform_columns(self, X: np.ndarray, columns: List[Any]) -> np.ndarray:
        """Transform columns to bin indices."""
        raise NotImplementedError("Subclasses must implement _transform_columns method.")

    @abstractmethod
    def _inverse_transform_columns(self, X: np.ndarray, columns: List[Any]) -> np.ndarray:
        """Inverse transform from bin indices to representative values."""
        raise NotImplementedError("Subclasses must implement _inverse_transform_columns method.")

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator with bin-specific handling."""
        params = super().get_params(deep=deep)

        # Let subclasses add their specific parameters
        params.update(self._get_binning_params())

        # Override with fitted values if fitted, otherwise keep constructor values
        if self._fitted:
            fitted_params = self._get_fitted_params()
            params.update(fitted_params)

        return params

    def _get_binning_params(self) -> Dict[str, Any]:
        """Get binning-specific parameters. Override in subclasses."""
        return {}

    def _get_fitted_params(self) -> Dict[str, Any]:
        """Get fitted parameter values. Override in subclasses."""
        return {}

    def set_params(self, **params) -> "GeneralBinningBase":
        """Set parameters for this estimator with bin-specific handling."""
        # Validate guidance + joint fitting before setting
        guidance_cols = params.get("guidance_columns", self.guidance_columns)
        fit_jointly = params.get("fit_jointly", self.fit_jointly)

        if guidance_cols is not None and fit_jointly:
            raise ValueError(
                "guidance_columns and fit_jointly=True are incompatible. "
                "Use either guidance_columns for per-record guidance OR "
                "fit_jointly=True for global fitting, but not both."
            )

        # Let subclasses handle bin-specific parameter changes
        reset_fitted = self._handle_bin_params(params)

        if reset_fitted:
            self._fitted = False

        return super().set_params(**params)

    def _handle_bin_params(self, params: Dict[str, Any]) -> bool:
        """Handle bin-specific parameter changes. Override in subclasses.

        Returns:
            bool: True if fitted state should be reset
        """
        return False

    def _validate_params(self) -> None:
        """Validate parameters for sklearn compatibility."""
        # Validate preserve_dataframe
        if self.preserve_dataframe is not None and not isinstance(self.preserve_dataframe, bool):
            raise TypeError("preserve_dataframe must be a boolean or None")

        # Validate fit_jointly
        if self.fit_jointly is not None and not isinstance(self.fit_jointly, bool):
            raise TypeError("fit_jointly must be a boolean or None")

        # Validate guidance_columns
        if self.guidance_columns is not None:
            if not isinstance(self.guidance_columns, (list, tuple, int, str)):
                raise TypeError("guidance_columns must be list, tuple, int, str, or None")

    # Properties for sklearn compatibility
    @property
    def is_fitted_(self) -> bool:
        """Whether the estimator is fitted."""
        return self._fitted

    @property
    def n_features_in_(self) -> Optional[int]:
        """Number of features seen during fit."""
        return self._n_features_in

    @property
    def feature_names_in_(self) -> Optional[List[Any]]:
        """Feature names seen during fit."""
        return getattr(self, "_feature_names_in", None)

    # Additional utility properties
    @property
    def binning_columns_(self) -> Optional[List[Any]]:
        """Columns that are being binned (excludes guidance columns)."""
        return self._binning_columns

    @property
    def guidance_columns_(self) -> Optional[List[Any]]:
        """Columns used for guidance."""
        return self._guidance_columns
