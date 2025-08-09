"""
Validation mixin for V2 architecture.

This module provides enhanced validation capabilities for binning classes.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import numpy as np

from ..config import get_config
from ..utils import DataQualityWarning, InvalidDataError


class ValidationMixin:
    """Mixin class providing enhanced validation capabilities for binning operations.

    This mixin provides common validation functionality that can be inherited by
    binning classes to ensure data quality, parameter validity, and proper input
    handling. It encapsulates validation logic to promote code reuse and
    consistent error handling across the binlearn library.

    Key Features:
    - Array-like input validation and conversion
    - Column specification validation
    - Guidance column validation for supervised binning
    - Data quality checks with warnings
    - Consistent error messaging and suggestions

    Note:
    -----
    This is a mixin class designed to be inherited alongside other base classes.
    It provides static and instance methods that can be called by binning classes
    to validate inputs and parameters.
    """

    @staticmethod
    def validate_array_like(
        data: Any, name: str = "data", allow_none: bool = False
    ) -> np.ndarray[Any, Any] | None:
        """Validate and convert array-like input to numpy array.

        This method provides robust validation and conversion of various input
        formats to numpy arrays, with comprehensive error handling and helpful
        suggestions for common issues.

        Args:
            data: Input data to validate and convert. Can be:
                - numpy.ndarray: Used directly
                - pandas.DataFrame/Series: Converted to numpy array
                - polars.DataFrame: Converted to numpy array
                - list, tuple: Converted to numpy array
                - None: Allowed only if allow_none=True
            name: Name of the data parameter for error messages. Used to provide
                context in error messages (e.g., "X", "y", "guidance_data").
            allow_none: Whether to allow None as a valid input. If True, None
                is returned unchanged; if False, None raises InvalidDataError.

        Returns:
            Validated numpy array, or None if data is None and allow_none=True.
            The returned array maintains the same data content but is guaranteed
            to be a numpy array.

        Raises:
            InvalidDataError: If validation fails:
                - data is None when allow_none=False
                - data cannot be converted to numpy array
                - Conversion process encounters errors

        Example:
            >>> # Valid inputs
            >>> arr = ValidationMixin.validate_array_like([1, 2, 3], "X")
            >>> print(type(arr))
            <class 'numpy.ndarray'>
            >>>
            >>> # Allow None
            >>> result = ValidationMixin.validate_array_like(None, "y", allow_none=True)
            >>> print(result)
            None
            >>>
            >>> # Invalid input
            >>> ValidationMixin.validate_array_like(None, "X", allow_none=False)
            InvalidDataError: X cannot be None

        Note:
            This method focuses on format validation and conversion. Content
            validation (like checking for NaN values) should be done separately
            using other validation methods.
        """
        if data is None and allow_none:
            return None

        if data is None:
            raise InvalidDataError(
                f"{name} cannot be None",
                suggestions=[
                    f"Provide a valid array-like object for {name}",
                    "Check if your data loading was successful",
                ],
            )

        try:
            arr = np.asarray(data)
        except Exception as e:
            raise InvalidDataError(
                f"Could not convert {name} to array: {str(e)}",
                suggestions=[
                    "Ensure input is array-like (list, numpy array, pandas DataFrame/Series)",
                    "Check for any invalid values in your data",
                    "Consider converting data types explicitly",
                ],
            ) from e

        # Check if array is empty - let specific methods handle this with their own error messages
        # if array.size == 0:
        #     raise ValueError(f"{name} is empty")

        return arr

    @staticmethod
    def validate_column_specification(columns: Any, data_shape: tuple[int, ...]) -> list[Any]:
        """Validate column specifications."""
        if columns is None:
            return list(range(data_shape[1]))

        # Convert single column to list
        if not isinstance(columns, list | tuple | np.ndarray):
            columns = [columns]

        # Validate each column
        validated_columns: list[Any] = []
        for col in columns:
            if isinstance(col, str):
                validated_columns.append(col)
            elif isinstance(col, int):
                if col < 0 or col >= data_shape[1]:
                    raise InvalidDataError(
                        f"Column index {col} is out of range for data with {data_shape[1]} columns",
                        suggestions=[
                            f"Use column indices between 0 and {data_shape[1] - 1}",
                            "Check if your data has the expected number of columns",
                        ],
                    )
                validated_columns.append(col)
            else:
                raise InvalidDataError(
                    f"Invalid column specification: {col} (type: {type(col)})",
                    suggestions=[
                        "Use string column names or integer indices",
                        "Ensure column specifications match your data format",
                    ],
                )

        return validated_columns

    @staticmethod
    def validate_guidance_columns(
        guidance_cols: Any, binning_cols: list[Any], data_shape: tuple[int, ...]
    ) -> list[Any]:
        """Validate guidance column specifications."""
        if guidance_cols is None:
            return []

        # Convert to list if needed
        if not isinstance(guidance_cols, list | tuple):
            guidance_cols = [guidance_cols]

        validated_guidance = ValidationMixin.validate_column_specification(
            guidance_cols, data_shape
        )

        # Check for overlap with binning columns
        overlap = set(validated_guidance) & set(binning_cols)
        if overlap:
            raise InvalidDataError(
                f"Guidance columns cannot overlap with binning columns: {overlap}",
                suggestions=[
                    "Use separate columns for guidance and binning",
                    "Consider creating a copy of the target column if needed",
                ],
            )

        return validated_guidance

    @staticmethod
    def check_data_quality(data: np.ndarray[Any, Any], name: str = "data") -> None:
        """Check data quality and issue warnings if needed."""

        config = get_config()

        if not config.show_warnings:
            return

        # Check for missing values - handle different dtypes
        # For numeric data, use np.isnan
        if np.issubdtype(data.dtype, np.number):
            missing_mask = np.isnan(data)
        else:
            # For object/string data, check for None and 'nan' strings
            missing_mask = np.array(
                [
                    x is None or (isinstance(x, str) and x.lower() in ["nan", "na", "null", ""])
                    for x in cast(Any, data.flat)
                ]
            ).reshape(data.shape)

        if missing_mask.any():
            missing_pct = missing_mask.mean() * 100
            if missing_pct > 50:
                warnings.warn(
                    f"{name} contains {missing_pct:.1f}% missing values. "
                    "This may significantly impact binning quality.",
                    DataQualityWarning,
                    stacklevel=2,
                )

        # Check for infinite values only for numeric types
        try:
            if np.issubdtype(data.dtype, np.number):
                if np.isinf(data).any():
                    warnings.warn(
                        f"{name} contains infinite values. "
                        "Consider clipping or removing these values.",
                        DataQualityWarning,
                        stacklevel=2,
                    )
        except (TypeError, ValueError):
            # Skip infinite value check if data type doesn't support it
            pass

        # Check for constant columns
        if data.ndim == 2:
            for i in range(data.shape[1]):
                col_data = data[:, i]
                finite_data = col_data[np.isfinite(col_data)]
                if len(finite_data) > 1 and np.var(finite_data) == 0:
                    warnings.warn(
                        f"Column {i} in {name} appears to be constant. "
                        "This will result in a single bin.",
                        DataQualityWarning,
                        stacklevel=2,
                    )
