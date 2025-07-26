"""
Simplified base class for all binning methods.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ._data_utils import prepare_array, return_like_input, prepare_input_with_columns


class GeneralBinningBase(BaseEstimator, TransformerMixin):
    """Base binning class with universal guidance support."""

    def __init__(
        self,
        preserve_dataframe: bool = False,
        fit_jointly: bool = False,
        guidance_columns: Optional[Union[List[Any], Any]] = None,
        **kwargs,
    ):
        # Validate incompatible parameters
        if guidance_columns is not None and fit_jointly:
            raise ValueError(
                "guidance_columns and fit_jointly=True are incompatible. "
                "Guidance operates per-record while joint fitting operates globally."
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
        # Store original input info for sklearn compatibility
        arr, original_columns = self._prepare_input(X)
        self._n_features_in = arr.shape[1]
        self._original_columns = original_columns

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

    def transform(self, X: Any) -> Any:
        """Universal transform with guidance column handling."""
        self._check_fitted()

        # Separate columns
        X_binning, X_guidance, binning_cols, guidance_cols = self._separate_columns(X)

        if self.guidance_columns is None:
            # No guidance - transform all columns
            result = self._transform_columns(X_binning, binning_cols)
            return return_like_input(result, X, binning_cols, self.preserve_dataframe)

        # Transform only binning columns
        if X_binning.shape[1] > 0:
            result = self._transform_columns(X_binning, binning_cols)
        else:
            result = np.empty((X_binning.shape[0], 0), dtype=int)

        return return_like_input(result, X, binning_cols, self.preserve_dataframe)

    def transform_with_guidance(self, X: Any) -> Tuple[Any, Any]:
        """Transform and return both binned and guidance data separately."""
        self._check_fitted()

        X_binning, X_guidance, binning_cols, guidance_cols = self._separate_columns(X)

        # Transform binning columns
        if X_binning.shape[1] > 0:
            binned_result = self._transform_columns(X_binning, binning_cols)
        else:
            binned_result = np.empty((X_binning.shape[0], 0), dtype=int)

        # Format outputs
        binned_output = return_like_input(binned_result, X, binning_cols, self.preserve_dataframe)

        if X_guidance is not None:
            guidance_output = return_like_input(
                X_guidance, X, guidance_cols, self.preserve_dataframe
            )
        else:
            guidance_output = None

        return binned_output, guidance_output

    def inverse_transform(self, X: Any) -> Any:
        """Inverse transform from bin indices back to representative values."""
        self._check_fitted()

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
            return return_like_input(result, X, self._binning_columns, self.preserve_dataframe)
        else:
            # No guidance - inverse transform all columns
            arr, columns = self._prepare_input(X)
            result = self._inverse_transform_columns(arr, columns)
            return return_like_input(result, X, columns, self.preserve_dataframe)

    # Abstract methods to be implemented by subclasses
    def _fit_per_column(
        self,
        X: np.ndarray,
        columns: List[Any],
        guidance_data: Optional[np.ndarray] = None,
        **fit_params,
    ) -> None:
        """Fit bins per column with optional guidance."""
        raise NotImplementedError("Subclasses must implement _fit_per_column method.")

    def _fit_jointly(self, X: np.ndarray, columns: List[Any], **fit_params) -> None:
        """Fit bins jointly (guidance incompatible, so no guidance_data parameter)."""
        raise NotImplementedError(
            "Joint fitting not implemented. Subclasses should override this method."
        )

    def _transform_columns(self, X: np.ndarray, columns: List[Any]) -> np.ndarray:
        """Transform columns to bin indices."""
        raise NotImplementedError("Subclasses must implement _transform_columns method.")

    def _inverse_transform_columns(self, X: np.ndarray, columns: List[Any]) -> np.ndarray:
        """Inverse transform from bin indices to representative values."""
        raise NotImplementedError("Subclasses must implement _inverse_transform_columns method.")

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        params = super().get_params(deep=deep)
        return params

    def set_params(self, **params) -> "GeneralBinningBase":
        """Set parameters for this estimator."""
        # Validate guidance + joint fitting before setting
        guidance_cols = params.get("guidance_columns", self.guidance_columns)
        fit_jointly = params.get("fit_jointly", self.fit_jointly)

        if guidance_cols is not None and fit_jointly:
            raise ValueError(
                "guidance_columns and fit_jointly=True are incompatible. "
                "Guidance operates per-record while joint fitting operates globally."
            )

        return super().set_params(**params)

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
        """Names of features seen during fit."""
        return self._original_columns if self._original_columns is not None else None

    # Additional utility properties
    @property
    def binning_columns_(self) -> Optional[List[Any]]:
        """Columns that are being binned (excludes guidance columns)."""
        return self._binning_columns

    @property
    def guidance_columns_(self) -> Optional[List[Any]]:
        """Columns used for guidance."""
        return self._guidance_columns

    def __repr__(self) -> str:
        """String representation of the estimator."""
        N_CHAR_MAX = 700
        class_name = self.__class__.__name__
        params = []

        if self.preserve_dataframe:
            params.append("preserve_dataframe=True")
        if self.fit_jointly:
            params.append("fit_jointly=True")
        if self.guidance_columns is not None:
            if isinstance(self.guidance_columns, list):
                params.append(f"guidance_columns={self.guidance_columns}")
            else:
                params.append(f"guidance_columns=[{self.guidance_columns}]")

        param_str = ", ".join(params)
        result = f"{class_name}({param_str})"

        # Fix: Check length and truncate properly
        if len(result) > N_CHAR_MAX:
            result = result[: N_CHAR_MAX - 3] + "..."

        return result
