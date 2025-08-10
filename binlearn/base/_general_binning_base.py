"""
Clean general binning base class for V2 architecture.

This module provides the core binning orchestration logic with guidance support.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.base import TransformerMixin

from ..config import get_config
from ..utils import ArrayLike, BinningError, ColumnList, GuidanceColumns
from ._data_handling_base import DataHandlingBase


# pylint: disable=too-many-ancestors
class GeneralBinningBase(
    ABC,
    DataHandlingBase,
    TransformerMixin,  # type: ignore[misc,unused-ignore]
):
    """Clean binning base class focusing on orchestration and guidance logic.

    This abstract base class provides the core infrastructure for all binning
    transformers in the binlearn library. It orchestrates the binning process,
    handles guidance column separation, and manages the interaction between
    fitting and transformation phases.

    The class supports two main fitting strategies:
    - Per-column independent fitting: Each column is binned independently
    - Joint fitting: All columns are considered together for binning decisions

    Key Features:
    - Guidance column support for supervised and semi-supervised binning
    - Flexible fitting strategies (independent vs joint)
    - DataFrame format preservation
    - Comprehensive error handling and validation
    - sklearn-compatible transformer interface

    Parameters:
    -----------
    preserve_dataframe : bool, optional
        Whether to preserve the original DataFrame format in output. If None,
        uses the global configuration default. When True, pandas/polars
        DataFrames are returned as DataFrames; otherwise numpy arrays.

    fit_jointly : bool, optional
        Whether to fit all columns jointly rather than independently. If None,
        uses the global configuration default. When True, all binning columns
        are considered together; when False, each column is binned independently.

    guidance_columns : GuidanceColumns, optional
        Specification of columns to use for guidance (supervision). Can be:
        - None: No guidance columns (unsupervised binning)
        - Column identifier: Single guidance column
        - List of identifiers: Multiple guidance columns
        Incompatible with fit_jointly=True.

    Attributes:
    -----------
    preserve_dataframe : bool
        Whether to preserve DataFrame format in output.

    fit_jointly : bool
        Whether to fit columns jointly or independently.

    guidance_columns : GuidanceColumns
        Specification of guidance columns for supervision.

    Note:
    -----
    This is an abstract base class and cannot be instantiated directly.
    Concrete implementations must provide the abstract methods for specific
    binning algorithms.

    The class enforces mutual exclusivity between fit_jointly=True and
    guidance_columns to prevent conflicting binning strategies.
    """

    def __init__(
        self,
        preserve_dataframe: bool | None = None,
        fit_jointly: bool | None = None,
        guidance_columns: GuidanceColumns = None,
    ):
        """Initialize the binning transformer.

        Sets up the binning transformer with the specified configuration options,
        applying global configuration defaults where parameters are not provided.
        Validates parameter compatibility to prevent conflicting configurations.

        Args:
            preserve_dataframe: Whether to preserve DataFrame format in output.
                If None, uses global configuration default.
            fit_jointly: Whether to fit all columns together. If None, uses
                global configuration default.
            guidance_columns: Guidance column specification for supervised binning.
                Must be None if fit_jointly=True.

        Raises:
            ValueError: If guidance_columns is specified when fit_jointly=True,
                as these options are mutually exclusive.

        Note:
            The binning and guidance column lists are computed dynamically during
            fitting based on the actual input data and the guidance_columns parameter.
        """
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
        """Fit the binning transformer with comprehensive orchestration.

        This method orchestrates the complete fitting process, handling parameter
        validation, input preprocessing, column separation, and routing to the
        appropriate fitting strategy (joint vs independent).

        Args:
            X: Input data to fit the binning transformer on. Can be:
                - pandas.DataFrame: Column names are preserved
                - polars.DataFrame: Column names are preserved
                - numpy.ndarray: Numeric column indices are used
                - array-like: Converted to numpy array
            y: Target values for supervised binning methods. Ignored by
                unsupervised methods. Can be array-like or None.
            **fit_params: Additional fitting parameters passed to the specific
                binning algorithm implementation. Common parameters include:
                - guidance_data: Alternative guidance data (conflicts with fit_jointly=True)

        Returns:
            self: The fitted binning transformer instance.

        Raises:
            ValueError: If parameter validation fails, inputs are invalid, or
                conflicting parameters are provided (e.g., fit_jointly=True with
                guidance_data).
            BinningError: If the binning algorithm fails to fit the data.
            RuntimeError: If an unexpected error occurs during fitting.

        Example:
            >>> from binlearn import EqualWidthBinning
            >>> import pandas as pd
            >>> X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5], 'feature2': [10, 20, 30, 40, 50]})
            >>> binner = EqualWidthBinning(n_bins=3)
            >>> binner.fit(X)
            EqualWidthBinning(...)

        Note:
            The method automatically handles column separation when guidance_columns
            is specified, routing guidance columns separately from binning columns.
            The fitting strategy (joint vs independent) is determined by the
            fit_jointly parameter.
        """
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

            # Step 4.5: Validate that we have columns to bin
            if not binning_cols:
                if self.guidance_columns is not None:
                    raise ValueError(
                        "All columns are specified as guidance_columns. "
                        "At least one column must be available for binning."
                    )

                raise ValueError("No columns available for binning.")

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
        """Transform input data using fitted binning parameters.

        Applies the fitted binning transformation to new data, converting
        continuous values to discrete bin indices or representatives.
        Handles column separation when guidance columns are present.

        Args:
            X: Input data to transform. Must have the same structure as the
                data used during fitting (same number of columns). Can be:
                - pandas.DataFrame: Column names should match training data
                - polars.DataFrame: Column names should match training data
                - numpy.ndarray: Must have same number of columns as training
                - array-like: Converted to numpy array

        Returns:
            Transformed data where continuous values are replaced with bin
            indices or representative values. The output format depends on:
            - preserve_dataframe setting: DataFrame vs array format
            - binning method: indices vs representatives
            - guidance_columns: only binning columns are transformed

        Raises:
            RuntimeError: If the transformer has not been fitted yet.
            ValueError: If the input data has incompatible structure or format.
            BinningError: If transformation fails due to data issues.

        Example:
            >>> # After fitting
            >>> X_new = pd.DataFrame({'feature1': [1.5, 2.5], 'feature2': [15, 25]})
            >>> X_binned = binner.transform(X_new)
            >>> print(X_binned)
            [[0, 0], [1, 1]]  # Bin indices

        Note:
            When guidance_columns is specified, only the binning columns are
            transformed. Guidance columns are filtered out from the output.
            The method preserves the original data format when preserve_dataframe=True.
        """
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
        """Inverse transform from bin indices back to representative values.

        Converts discrete bin indices back to their representative values,
        effectively reversing the binning transformation. This is useful for
        interpreting results or reconstructing approximate original values.

        Args:
            X: Input data containing bin indices to inverse transform. Should
                contain only binning columns (no guidance columns). Can be:
                - pandas.DataFrame: Column names should match binning columns
                - polars.DataFrame: Column names should match binning columns
                - numpy.ndarray: Must have same number of binning columns
                - array-like: Converted to numpy array

        Returns:
            Inverse transformed data where bin indices are replaced with their
            representative values (typically bin centers). Output format matches
            the preserve_dataframe setting.

        Raises:
            RuntimeError: If the transformer has not been fitted yet.
            ValueError: If input data has wrong number of columns or invalid format.
            BinningError: If inverse transformation fails.

        Example:
            >>> # After fitting and transforming
            >>> X_binned = [[0, 1], [1, 0], [2, 2]]  # Bin indices
            >>> X_reconstructed = binner.inverse_transform(X_binned)
            >>> print(X_reconstructed)
            [[0.5, 1.5], [1.5, 0.5], [2.5, 2.5]]  # Representative values

        Note:
            For guided binning (when guidance_columns is specified), the input
            should only contain the binning columns, not the guidance columns.
            The number of input columns must match the number of binning columns.
        """
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

    def _normalize_guidance_columns(
        self, guidance_cols: list[Any], columns: ColumnList
    ) -> list[Any]:
        """Normalize guidance columns from various formats to column names.

        This method handles the conversion of integer indices to column names,
        making the logic testable and reusable.

        Args:
            guidance_cols: List of guidance column identifiers (integers or strings)
            columns: Available column names

        Returns:
            List of normalized guidance column names

        Raises:
            ValueError: If column index is out of range
        """
        normalized_guidance_cols = []
        for col in guidance_cols:
            if isinstance(col, int):
                if 0 <= col < len(columns):
                    normalized_guidance_cols.append(columns[col])
                else:
                    raise ValueError(
                        f"Column index {col} is out of range for {len(columns)} columns"
                    )
            else:
                normalized_guidance_cols.append(col)  # This is line 239 equivalent

        return normalized_guidance_cols

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

        # Convert integer indices to column names if needed - now in separate method
        normalized_guidance_cols = self._normalize_guidance_columns(guidance_cols, columns)

        # Separate columns
        binning_indices = []
        guidance_indices = []
        binning_column_names = []
        guidance_column_names = []

        for i, col in enumerate(columns):
            if col in normalized_guidance_cols:
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

        # At this point we know feature_names_in_ exists and is not None
        all_features = list(self.feature_names_in_)  # type: ignore[arg-type]

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

    def _get_column_key(self, target_col: Any, available_keys: ColumnList, col_index: int) -> Any:
        """Get the appropriate key for looking up bin specifications.

        Handles column key resolution with fallback strategies for
        different column identifier formats (names vs indices).

        Args:
            target_col: The target column identifier to find.
            available_keys: List of available keys in bin specifications.
            col_index: Index position of the column.

        Returns:
            The key to use for bin specification lookup.

        Raises:
            ValueError: If no matching key can be found.
        """
        # First try exact match
        if target_col in available_keys:
            return target_col

        # Handle feature_N -> N mapping for numpy array inputs
        if isinstance(target_col, str) and target_col.startswith("feature_"):
            try:
                feature_index = int(target_col.split("_")[1])
                if feature_index in available_keys:
                    return feature_index
            except (ValueError, IndexError):
                pass

        # Handle N -> feature_N mapping
        if isinstance(target_col, int):
            feature_name = f"feature_{target_col}"
            if feature_name in available_keys:
                return feature_name

        # Try index-based fallback
        if col_index < len(available_keys):
            return available_keys[col_index]

        # No match found
        raise ValueError(f"No bin specification found for column {target_col} (index {col_index})")

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

        This abstract method must be implemented by concrete binning classes to
        define how each column is binned independently. This is the default
        fitting strategy when fit_jointly=False.

        Args:
            X: Input data array containing only the columns to be binned.
                Shape: (n_samples, n_binning_columns).
            columns: List of column identifiers corresponding to the columns in X.
                Used for error messages and result storage.
            guidance_data: Optional guidance data for supervised binning methods.
                Can be target values (y) or additional guidance information.
                Shape should be compatible with X for supervised methods.
            **fit_params: Additional algorithm-specific fitting parameters passed
                from the fit() method.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented
                by concrete subclasses.

        Note:
            Implementations should store the fitted binning parameters (bin edges,
            representatives, etc.) in instance attributes for later use during
            transformation.
        """
        raise NotImplementedError("Subclasses must implement _fit_per_column_independently")

    @abstractmethod
    def _fit_jointly_across_columns(
        self, X: np.ndarray[Any, Any], columns: ColumnList, **fit_params: Any
    ) -> None:
        """Fit binning parameters jointly across all columns.

        This abstract method must be implemented by concrete binning classes to
        define how all columns are considered together for binning decisions.
        This enables more sophisticated binning strategies that consider
        inter-column relationships.

        Args:
            X: Input data array containing all columns to be binned together.
                Shape: (n_samples, n_binning_columns).
            columns: List of column identifiers corresponding to the columns in X.
                Used for error messages and result storage.
            **fit_params: Additional algorithm-specific fitting parameters passed
                from the fit() method.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented
                by concrete subclasses.

        Note:
            Joint fitting is incompatible with guidance_columns and guidance_data
            parameters. Implementations should consider relationships between
            columns when determining binning parameters.
        """
        raise NotImplementedError("Subclasses must implement _fit_jointly_across_columns")

    @abstractmethod
    def _transform_columns_to_bins(
        self, X: np.ndarray[Any, Any], columns: ColumnList
    ) -> np.ndarray[Any, Any]:
        """Transform columns to bin indices using fitted parameters.

        This abstract method must be implemented by concrete binning classes to
        define how continuous values are converted to discrete bin indices
        during the transformation phase.

        Args:
            X: Input data array to transform. Contains continuous values that
                need to be converted to bin indices. Shape: (n_samples, n_columns).
            columns: List of column identifiers corresponding to the columns in X.
                Used for accessing the appropriate fitted binning parameters.

        Returns:
            Transformed data array where continuous values are replaced with
            discrete bin indices. Shape: (n_samples, n_columns).
            Bin indices should be integers where:
            - 0 to n_bins-1: Valid bin indices
            - MISSING_VALUE (-1): Missing/NaN values
            - BELOW_RANGE (-3): Values below binning range
            - ABOVE_RANGE (-2): Values above binning range

        Raises:
            NotImplementedError: This is an abstract method that must be implemented
                by concrete subclasses.

        Note:
            Implementations should handle missing values and out-of-range values
            appropriately using the framework's special index constants.
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
