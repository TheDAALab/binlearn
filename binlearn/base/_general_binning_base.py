"""
General base class for all binning methods in the binlearn library.

This module provides the foundational GeneralBinningBase class that all binning
transformers inherit from. It handles common functionality like data validation,
sklearn compatibility, configuration management, and guidance column processing.

The class supports both pandas and polars DataFrames while maintaining numpy
compatibility. It provides a consistent interface for all binning methods with
configurable behavior through the global configuration system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ..config import get_config
from ..utils.data_handling import prepare_input_with_columns, return_like_input
from ..utils.errors import BinningError, ValidationMixin
from ..utils.inspection import safe_get_class_parameters
from ..utils.sklearn_integration import SklearnCompatibilityMixin
from ..utils.types import ArrayLike, ColumnList, GuidanceColumns, OptionalColumnList


# pylint: disable=too-many-ancestors
class GeneralBinningBase(
    ABC, BaseEstimator, TransformerMixin, ValidationMixin, SklearnCompatibilityMixin
):
    """Base class for all binning transformers with universal guidance support.

    This abstract base class provides the foundation for all binning methods in the
    package. It handles configuration management, data validation, sklearn integration,
    and supports both guided and unguided binning approaches.

    The class is designed to work seamlessly with pandas DataFrames, polars DataFrames,
    and numpy arrays, automatically preserving the input data format when possible.

    Args:
        preserve_dataframe (bool, optional): Whether to preserve DataFrame format in output.
            If None, uses global configuration default.
        fit_jointly (bool, optional): Whether to fit parameters jointly across all columns.
            If None, uses global configuration default. Cannot be used with guidance_columns.
        guidance_columns (GuidanceColumns, optional): Columns to use for guided binning.
            Cannot be used with fit_jointly=True.
        **kwargs: Additional keyword arguments passed to subclasses.

    Attributes:
        preserve_dataframe (bool): Whether to preserve DataFrame format.
        fit_jointly (bool): Whether parameters are fitted jointly.
        guidance_columns (GuidanceColumns): Guidance columns if specified.

    Raises:
        ValueError: If guidance_columns and fit_jointly=True are both specified.

    Example:
        >>> # This is an abstract class, use a concrete implementation
        >>> from binlearn.methods import EqualWidthBinning
        >>> binner = EqualWidthBinning(n_bins=5, preserve_dataframe=True)
        >>> X_binned = binner.fit_transform(X)
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        preserve_dataframe: bool | None = None,
        fit_jointly: bool | None = None,
        guidance_columns: GuidanceColumns = None,
        **kwargs: Any,
    ):
        """Initialize the base binning transformer.

        Args:
            preserve_dataframe (bool, optional): Whether to preserve DataFrame format in output.
                If None, uses global configuration default.
            fit_jointly (bool, optional): Whether to fit parameters jointly across all columns.
                If None, uses global configuration default. Cannot be used with guidance_columns.
            guidance_columns (GuidanceColumns, optional): Columns to use for guided binning.
                Cannot be used with fit_jointly=True.
            **kwargs: Additional keyword arguments passed to subclasses.

        Raises:
            ValueError: If guidance_columns and fit_jointly=True are both specified.
        """
        _ = kwargs

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
        self._n_features_in: int | None = None
        self._feature_names_in: OptionalColumnList = None

    def _prepare_input(self, X: ArrayLike) -> tuple[np.ndarray, ColumnList]:
        """Prepare input array and determine column identifiers.

        Converts input data to a standardized numpy array format while preserving
        column information. Handles pandas DataFrames, polars DataFrames, and numpy
        arrays consistently.

        Args:
            X (ArrayLike): Input data to prepare. Can be pandas DataFrame, polars
                DataFrame, or numpy array.

        Returns:
            Tuple[np.ndarray, ColumnList]: A tuple containing:
                - Prepared numpy array with standardized format
                - Column identifiers (names for DataFrames, indices for arrays)

        Note:
            This method leverages the utility function `prepare_input_with_columns`
            to ensure consistent handling across different input formats.
        """
        return prepare_input_with_columns(X, fitted=self._fitted, original_columns=None)

    def _check_fitted(self) -> None:
        """Check if the estimator is fitted.

        Validates that the transformer has been fitted before attempting to use it
        for transformation or other operations that require fitted state.

        Raises:
            RuntimeError: If the estimator has not been fitted yet. The error message
                instructs the user to call 'fit' first.

        Note:
            This method is called internally by transform(), inverse_transform(),
            and other methods that require a fitted estimator.
        """
        if not self._fitted:
            raise RuntimeError("This estimator is not fitted yet. Call 'fit' first.")

    def _separate_columns(
        self, X: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray | None, ColumnList, ColumnList | None]:
        """Universal column separation logic for binning and guidance columns.

        Separates the input data into binning columns (to be transformed) and
        guidance columns (used for supervised binning). This method provides
        the core logic for handling guided vs unguided binning scenarios.

        When guidance_columns is None, all columns are treated as binning columns.
        When guidance_columns is specified, the method splits the data into two
        separate arrays for binning and guidance respectively.

        Args:
            X (ArrayLike): Input data with both binning and guidance columns.
                Can be pandas DataFrame, polars DataFrame, or numpy array.

        Returns:
            Tuple containing:
                - X_binning (np.ndarray): Data for columns to be binned. Shape is
                  (n_samples, n_binning_columns).
                - X_guidance (Optional[np.ndarray]): Data for guidance columns.
                  None if no guidance columns specified. Shape is (n_samples, n_guidance_columns).
                - binning_columns (ColumnList): Names/indices of binning columns.
                - guidance_columns (ColumnList | None): Names/indices of guidance columns.
                  None if no guidance specified.

        Note:
            This method automatically handles column name/index resolution and
            ensures that each column is classified as either binning or guidance,
            but not both.
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

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> GeneralBinningBase:
        """Universal fit method with guidance support.

        Fits the binning transformer to the input data. Handles both guided and
        unguided binning scenarios, automatically separating guidance columns
        when specified.

        Args:
            X (Any): Input data (DataFrame, array-like) to fit the transformer on.
            y (Any, optional): Target values, ignored. For sklearn compatibility.
            **fit_params: Additional parameters passed to fitting methods.

        Returns:
            GeneralBinningBase: Returns self for method chaining.

        Raises:
            BinningError: If fitting fails due to binning-specific issues.
            ValueError: If input validation or parameter validation fails.
            RuntimeError: If fitting encounters runtime issues.
        """
        _ = y

        try:
            # Validate parameters first
            self._validate_params()

            # Validate input data using ValidationMixin
            self.validate_array_like(X, "X")

            # Store original input info for sklearn compatibility
            arr, original_columns = self._prepare_input(X)
            self._n_features_in = arr.shape[1]

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

            # Route to appropriate fitting method
            if self.fit_jointly:
                self._fit_jointly(X_binning, binning_cols, **fit_params)
            else:
                # Handle potential conflict between X_guidance and fit_params['guidance_data']
                fit_params_clean = fit_params.copy()
                external_guidance_data = fit_params_clean.pop("guidance_data", None)

                # Use external guidance data if no embedded guidance columns
                final_guidance_data = (
                    X_guidance if X_guidance is not None else external_guidance_data
                )

                self._fit_per_column(
                    X_binning, binning_cols, final_guidance_data, **fit_params_clean
                )

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
        """Universal transform with guidance column handling.

        Transforms the input data using the fitted binning parameters. This method
        handles the core transformation logic for both guided and unguided binning
        scenarios, ensuring that only binning columns are transformed while
        guidance columns (if present) are preserved unchanged.

        The method automatically preserves the input data format (DataFrame vs array)
        based on the preserve_dataframe setting and returns transformed data in
        the same format as the input.

        Args:
            X (Any): Input data to transform. Must be in the same format and have
                the same structure as the data used in fit(). Can be pandas DataFrame,
                polars DataFrame, or numpy array.

        Returns:
            Any: Transformed data where binning columns are converted to bin indices
                (integers) and guidance columns (if any) are preserved unchanged.
                The output format matches the input format when preserve_dataframe=True.

        Raises:
            RuntimeError: If the transformer has not been fitted yet.
            ValueError: If input validation fails or if the number/names of features
                don't match those seen during fitting.

        Example:
            >>> # After fitting
            >>> X_transformed = binner.transform(X_test)
            >>> # X_transformed has same shape as X_test but with binned values

        Note:
            - Only binning columns are transformed to bin indices
            - Guidance columns remain unchanged in the output
            - Output format is preserved based on preserve_dataframe setting
        """
        try:
            self._check_fitted()
            # Validate input data
            self.validate_array_like(X, "X")
            # Check feature names consistency
            self._check_feature_names(X, reset=False)
            # Separate columns
            X_binning, _X_guidance, binning_cols, _guidance_cols = self._separate_columns(X)
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

    def inverse_transform(self, X: Any) -> Any:
        """Inverse transform from bin indices back to representative values.

        Converts binned data (integer bin indices) back to representative values
        for each bin. This method reverses the transformation applied by transform(),
        mapping bin indices back to meaningful numeric values.

        For guided binning scenarios, only the binning columns should be provided
        as input to this method, since guidance columns were not transformed and
        therefore cannot be inverse transformed.

        Args:
            X (Any): Binned data to inverse transform. Should contain integer bin
                indices as produced by transform(). For guided binning, this should
                only include the binning columns (not guidance columns).

        Returns:
            Any: Data with representative values replacing bin indices. The output
                format matches the input format when preserve_dataframe=True.
                Values represent the center or representative value of each bin.

        Raises:
            RuntimeError: If the transformer has not been fitted yet.
            ValueError: If input validation fails or if the column count doesn't
                match expectations. For guided binning, the input must have exactly
                the same number of columns as there were binning columns during fit.

        Example:
            >>> # After fitting and transforming
            >>> X_binned = binner.transform(X)  # Get bin indices
            >>> X_recovered = binner.inverse_transform(X_binned)  # Get representative values

        Note:
            - Input should be bin indices (integers) from transform()
            - For guided binning, only provide binning columns
            - Output contains representative values, not original values
            - Guidance columns cannot be inverse transformed
        """
        try:
            self._check_fitted()

            # Validate input data
            self.validate_array_like(X, "X")

            # For inverse transform, we work only with binning columns
            # (guidance columns weren't transformed, so can't be inverse transformed)
            if self.guidance_columns is not None:
                # Calculate expected number of binning columns
                total_features = self._n_features_in or 0
                guidance_cols = (
                    [self.guidance_columns]
                    if not isinstance(self.guidance_columns, list)
                    else self.guidance_columns
                )
                expected_binning_cols = total_features - len(guidance_cols)

                # Input should only have binning columns for inverse transform
                arr, columns = self._prepare_input(X)
                if len(columns) != expected_binning_cols:
                    raise ValueError(
                        f"Input for inverse_transform should have {expected_binning_cols} "
                        f"columns (binning columns only), got {len(columns)}"
                    )
                result = self._inverse_transform_columns(arr, columns)
                return return_like_input(result, X, columns, bool(self.preserve_dataframe))
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
        self, X: Any, columns: ColumnList, guidance_data: ArrayLike | None = None, **fit_params: Any
    ) -> GeneralBinningBase:
        """Fit bins per column with optional guidance.

        Abstract method that must be implemented by subclasses to handle column-wise
        fitting of binning parameters. This method is called when fit_jointly=False
        (the default) and allows each column to be binned independently.

        Subclasses should implement this method to calculate bin edges and
        representatives for each column separately, optionally using guidance
        data for supervised binning approaches.

        Args:
            X (Any): Input data for binning columns only (guidance columns are
                passed separately). Shape is (n_samples, n_binning_columns).
            columns (ColumnList): Column identifiers for the binning columns.
                These correspond to the columns in X.
            guidance_data (Optional[ArrayLike]): Guidance data if guidance_columns
                were specified during initialization. Shape is (n_samples, n_guidance_columns).
                None if no guidance columns specified.
            **fit_params: Additional fitting parameters passed from the fit() method.

        Returns:
            GeneralBinningBase: Returns self for method chaining.

        Note:
            - This method should set internal state (fitted parameters) on self
            - It should handle both guided and unguided scenarios based on guidance_data
            - Each column should be processed independently
            - The method should be compatible with the transform() implementation
        """
        raise NotImplementedError("Subclasses must implement _fit_per_column method.")

    @abstractmethod
    def _fit_jointly(self, X: np.ndarray, columns: ColumnList, **fit_params: Any) -> None:
        """Fit bins jointly across all columns.

        Abstract method that must be implemented by subclasses to handle joint
        fitting of binning parameters across all columns simultaneously. This
        method is called when fit_jointly=True and is incompatible with guidance
        columns.

        Joint fitting allows for coordinated binning strategies that consider
        relationships between columns, such as maintaining consistent bin widths
        or ranges across features.

        Args:
            X (np.ndarray): Input data for all binning columns. Shape is
                (n_samples, n_binning_columns).
            columns (ColumnList): Column identifiers for all binning columns.
                The order corresponds to the columns in X.
            **fit_params: Additional fitting parameters passed from the fit() method.

        Returns:
            None: This method should modify the internal state of self to store
                fitted parameters but does not return a value.

        Raises:
            NotImplementedError: If the subclass doesn't support joint fitting.

        Note:
            - Joint fitting is incompatible with guidance columns
            - This method should set internal state (fitted parameters) on self
            - Implementation should consider all columns simultaneously
            - The fitted parameters should be compatible with transform() method
        """
        raise NotImplementedError(
            "Joint fitting not implemented. Subclasses should override this method."
        )

    @abstractmethod
    def _transform_columns(self, X: np.ndarray, columns: ColumnList) -> np.ndarray:
        """Transform columns to bin indices.

        Abstract method that must be implemented by subclasses to convert input
        data to bin indices using the fitted binning parameters. This is the core
        transformation logic that maps continuous or discrete values to integer
        bin indices.

        Args:
            X (np.ndarray): Input data to transform. Shape is (n_samples, n_columns).
                Contains the raw data values that need to be mapped to bin indices.
            columns (ColumnList): Column identifiers corresponding to the columns
                in X. Used to access the appropriate fitted parameters for each column.

        Returns:
            np.ndarray: Transformed data with bin indices. Shape is (n_samples, n_columns).
                Each value is an integer representing the bin index that the original
                value was assigned to.

        Note:
            - Output should contain integer bin indices
            - Bin indices typically start from 0
            - Implementation should handle edge cases (NaN, out-of-range values)
            - Must be consistent with the binning parameters fitted in _fit_per_column
                or _fit_jointly
        """
        raise NotImplementedError("Subclasses must implement _transform_columns method.")

    @abstractmethod
    def _inverse_transform_columns(self, X: np.ndarray, columns: ColumnList) -> np.ndarray:
        """Inverse transform from bin indices to representative values.

        Abstract method that must be implemented by subclasses to convert bin
        indices back to representative values. This reverses the transformation
        performed by _transform_columns, mapping integer bin indices to meaningful
        numeric values that represent each bin.

        Args:
            X (np.ndarray): Binned data to inverse transform. Shape is (n_samples, n_columns).
                Contains integer bin indices as produced by _transform_columns.
            columns (ColumnList): Column identifiers corresponding to the columns
                in X. Used to access the appropriate fitted parameters for each column.

        Returns:
            np.ndarray: Data with representative values. Shape is (n_samples, n_columns).
                Each bin index is replaced with a representative value for that bin
                (e.g., bin center, mean, or other representative statistic).

        Note:
            - Input should be integer bin indices from _transform_columns
            - Output should be representative values (typically float)
            - Representative values often represent bin centers or means
            - Must be consistent with the binning parameters from fitting
        """
        raise NotImplementedError("Subclasses must implement _inverse_transform_columns method.")

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator with automatic parameter discovery.

        Automatically discovers and returns all parameters of the estimator,
        including both initialization parameters and fitted parameters when the
        estimator has been fitted. This method provides sklearn-compatible
        parameter access with enhanced functionality for binning-specific needs.

        The method automatically detects class-specific parameters and includes
        fitted parameters that enable parameter transfer workflows (fit → get_params
        → create new instance → transform without refitting).

        Args:
            deep (bool, optional): If True, return parameters for sub-estimators
                as well. Defaults to True for sklearn compatibility.

        Returns:
            Dict[str, Any]: Parameter names mapped to their values. Includes:
                - All initialization parameters (preserve_dataframe, fit_jointly, etc.)
                - Class-specific parameters from subclasses
                - Fitted parameters (bin_edges, bin_representatives, etc.) if fitted

        Example:
            >>> binner = EqualWidthBinning(n_bins=5)
            >>> binner.fit(X)
            >>> params = binner.get_params()
            >>> # params includes both init params and fitted bin edges
            >>> new_binner = EqualWidthBinning(**params)
            >>> # new_binner can transform without fitting

        Note:
            - Automatically discovers parameters without manual specification
            - Includes fitted state for parameter transfer workflows
            - Compatible with sklearn's parameter inspection
        """
        params = super().get_params(deep=deep)

        # Add all class-specific parameters automatically
        class_specific_params = safe_get_class_parameters(
            self.__class__, exclude_base_class="GeneralBinningBase"
        )

        # Add class-specific parameters to result
        for param_name in class_specific_params:
            if hasattr(self, param_name):
                params[param_name] = getattr(self, param_name)

        # Add fitted parameters if fitted
        if self._fitted:
            fitted_params = self._get_fitted_params()
            params.update(fitted_params)

        return params  # type: ignore[no-any-return]

    def _get_fitted_params(self) -> dict[str, Any]:
        """Get fitted parameters that should be transferred to new instances.

        Automatically discovers and extracts fitted parameters (attributes ending
        with underscore) that represent the learned state of the binning transformer.
        These parameters enable the creation of new instances that can transform
        data without requiring refitting.

        The method uses introspection to find all attributes that follow the
        sklearn convention of ending fitted attributes with an underscore, while
        excluding private attributes and dunder methods.

        Returns:
            Dict[str, Any]: Fitted parameters with underscore-free names. Common
                fitted parameters include:
                - bin_edges: Learned bin boundaries for each column
                - bin_representatives: Representative values for each bin
                - fitted_trees: Decision trees for supervised methods

        Example:
            >>> binner.fit(X)
            >>> fitted_params = binner._get_fitted_params()
            >>> # fitted_params might include {'bin_edges': {...}, 'bin_representatives': {...}}

        Note:
            - Only returns fitted attributes (ending with _)
            - Excludes private attributes (_ prefix) and dunder methods
            - Maps attribute names to parameter names (removes trailing _)
            - Used for parameter transfer workflows
        """
        fitted_params = {}

        # Automatically discover fitted attributes ending with underscore
        for attr_name in dir(self):
            if (
                attr_name.endswith("_")
                and not attr_name.startswith("_")  # Exclude private attributes
                and not attr_name.endswith("__")  # Exclude dunder methods
                and hasattr(self, attr_name)
            ):

                value = getattr(self, attr_name)
                if value is not None:
                    # Map to parameter names (remove trailing underscore)
                    param_name = attr_name.rstrip("_")
                    fitted_params[param_name] = value

        return fitted_params

    def set_params(self, **params: Any) -> GeneralBinningBase:
        """Set parameters with automatic handling and validation.

        Sets parameters on the estimator with intelligent handling of parameter
        conflicts and automatic fitted state management. This method extends
        sklearn's set_params with binning-specific validation and state management.

        The method automatically detects when parameter changes require refitting
        and resets the fitted state accordingly. It also validates parameter
        compatibility (e.g., guidance_columns with fit_jointly).

        Args:
            **params: Parameter names and values to set. Can include any valid
                parameter for the estimator, including both base class and
                subclass-specific parameters.

        Returns:
            GeneralBinningBase: Returns self for method chaining.

        Raises:
            ValueError: If incompatible parameters are provided (e.g., both
                guidance_columns and fit_jointly=True).

        Example:
            >>> binner.set_params(n_bins=10, preserve_dataframe=True)
            >>> # If binner was fitted, it's now unfitted due to parameter change

        Note:
            - Validates parameter compatibility before setting
            - Automatically resets fitted state when parameters change
            - Provides method chaining capability
            - Extends sklearn's set_params with binning-specific logic
        """
        # Validate guidance + joint fitting compatibility
        guidance_cols = params.get("guidance_columns", self.guidance_columns)
        fit_jointly = params.get("fit_jointly", self.fit_jointly)

        if guidance_cols is not None and fit_jointly:
            raise ValueError(
                "guidance_columns and fit_jointly=True are incompatible. "
                "Use either guidance_columns for per-record guidance OR "
                "fit_jointly=True for global fitting, but not both."
            )

        # Handle parameter changes and reset fitted state if needed
        if self._handle_bin_params(params):
            self._fitted = False

        result = super().set_params(**params)

        # Validate all parameters after setting them
        self._validate_params()

        return result  # type: ignore[no-any-return]

    def _handle_bin_params(self, params: dict[str, Any]) -> bool:
        """Handle all parameter changes automatically.

        Processes parameter changes and determines whether the fitted state should
        be reset. This method implements intelligent parameter change detection
        by automatically identifying class-specific parameters and parameters that
        affect the binning behavior.

        The method uses introspection to automatically detect parameters that
        require refitting, eliminating the need for manual specification of
        which parameters trigger refitting.

        Args:
            params (Dict[str, Any]): Dictionary of parameter names and values
                to be set. Parameters are consumed (removed) from this dictionary
                as they are processed.

        Returns:
            bool: True if any parameter that requires refitting was changed,
                False otherwise. When True is returned, the caller should reset
                the fitted state.

        Note:
            - Automatically detects parameters that require refitting
            - Modifies the input params dictionary by removing processed parameters
            - Uses class introspection for automatic parameter discovery
            - Parameters like fit_jointly and guidance_columns always trigger refitting
        """
        reset_fitted = False

        # Get class-specific parameters
        class_specific_params = safe_get_class_parameters(
            self.__class__, exclude_base_class="GeneralBinningBase"
        )

        # Parameters that always trigger refitting
        refit_params = ["fit_jointly", "guidance_columns"] + class_specific_params

        for param_name in refit_params:
            if param_name in params:
                setattr(self, param_name, params.pop(param_name))
                reset_fitted = True

        return reset_fitted

    def _validate_params(self) -> None:
        """Validate parameters for sklearn compatibility.

        Performs comprehensive validation of all base class parameters to ensure
        they meet the expected types and constraints. This method provides early
        error detection for parameter validation issues.

        The validation covers type checking for boolean parameters and ensures
        that complex parameters like guidance_columns have appropriate types
        and formats.

        Raises:
            TypeError: If any parameter has an invalid type:
                - preserve_dataframe must be boolean or None
                - fit_jointly must be boolean or None
                - guidance_columns must be list, tuple, int, str, or None

        Note:
            - Called automatically during fit() for early error detection
            - Provides clear error messages for type validation failures
            - Focuses on type validation rather than value validation
            - Compatible with sklearn's parameter validation patterns
        """
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
