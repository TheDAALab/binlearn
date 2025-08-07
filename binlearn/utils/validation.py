"""
Comprehensive parameter validation utilities for binlearn.

This module provides reusable validation functions that can be used across
all binning methods to ensure consistent parameter validation and error handling.
"""

from __future__ import annotations

import warnings
from typing import Any, Union, Optional, Sequence, Callable
from collections.abc import Iterable

import numpy as np


def validate_int(
    value: Any,
    name: str,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
    allow_none: bool = False,
) -> Optional[int]:
    """Validate integer parameters.

    Args:
        value: Value to validate.
        name: Parameter name for error messages.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).
        allow_none: Whether None is allowed.

    Returns:
        Validated integer value or None.

    Raises:
        TypeError: If value is not an integer or None (when allowed).
        ValueError: If value is outside the allowed range.
    """
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"{name} must be an integer, got None")

    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")

    value = int(value)

    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")

    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {value}")

    return value


def validate_float(
    value: Any,
    name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    allow_none: bool = False,
    allow_inf: bool = False,
) -> Optional[float]:
    """Validate float parameters.

    Args:
        value: Value to validate.
        name: Parameter name for error messages.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).
        allow_none: Whether None is allowed.
        allow_inf: Whether infinite values are allowed.

    Returns:
        Validated float value or None.

    Raises:
        TypeError: If value is not numeric or None (when allowed).
        ValueError: If value is outside the allowed range or is NaN/inf (when not allowed).
    """
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"{name} must be a number, got None")

    try:
        value = float(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} must be a number, got {type(value).__name__}") from e

    if np.isnan(value):
        raise ValueError(f"{name} cannot be NaN")

    if not allow_inf and np.isinf(value):
        raise ValueError(f"{name} cannot be infinite")

    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")

    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {value}")

    return value


def validate_bool(
    value: Any,
    name: str,
    allow_none: bool = False,
) -> Optional[bool]:
    """Validate boolean parameters.

    Args:
        value: Value to validate.
        name: Parameter name for error messages.
        allow_none: Whether None is allowed.

    Returns:
        Validated boolean value or None.

    Raises:
        TypeError: If value is not a boolean or None (when allowed).
    """
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"{name} must be a boolean, got None")

    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean, got {type(value).__name__}")

    return bool(value)


def validate_string(
    value: Any,
    name: str,
    allowed_values: Optional[Sequence[str]] = None,
    allow_none: bool = False,
    allow_empty: bool = True,
) -> Optional[str]:
    """Validate string parameters.

    Args:
        value: Value to validate.
        name: Parameter name for error messages.
        allowed_values: Sequence of allowed string values.
        allow_none: Whether None is allowed.
        allow_empty: Whether empty strings are allowed.

    Returns:
        Validated string value or None.

    Raises:
        TypeError: If value is not a string or None (when allowed).
        ValueError: If value is not in allowed_values or is empty (when not allowed).
    """
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"{name} must be a string, got None")

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")

    if not allow_empty and len(value) == 0:
        raise ValueError(f"{name} cannot be empty")

    if allowed_values is not None and value not in allowed_values:
        raise ValueError(f"{name} must be one of {list(allowed_values)}, got '{value}'")

    return value


def validate_tuple(
    value: Any,
    name: str,
    expected_length: Optional[int] = None,
    element_type: Optional[type] = None,
    allow_none: bool = False,
) -> Optional[tuple]:
    """Validate tuple parameters.

    Args:
        value: Value to validate.
        name: Parameter name for error messages.
        expected_length: Expected length of tuple.
        element_type: Expected type of tuple elements.
        allow_none: Whether None is allowed.

    Returns:
        Validated tuple or None.

    Raises:
        TypeError: If value is not a tuple or None (when allowed), or if elements are wrong type.
        ValueError: If tuple has wrong length.
    """
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"{name} must be a tuple, got None")

    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a tuple or list, got {type(value).__name__}")

    # Convert to tuple if it's a list
    value = tuple(value)

    if expected_length is not None and len(value) != expected_length:
        raise ValueError(f"{name} must have length {expected_length}, got {len(value)}")

    if element_type is not None:
        for i, element in enumerate(value):
            if not isinstance(element, element_type):
                raise TypeError(
                    f"{name}[{i}] must be of type {element_type.__name__}, "
                    f"got {type(element).__name__}"
                )

    return value


def validate_array_like(
    value: Any,
    name: str,
    dtype: Optional[type] = None,
    ndim: Optional[int] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    allow_none: bool = False,
) -> Optional[np.ndarray]:
    """Validate array-like parameters.

    Args:
        value: Value to validate.
        name: Parameter name for error messages.
        dtype: Expected data type.
        ndim: Expected number of dimensions.
        min_length: Minimum allowed length.
        max_length: Maximum allowed length.
        allow_none: Whether None is allowed.

    Returns:
        Validated numpy array or None.

    Raises:
        TypeError: If value cannot be converted to array.
        ValueError: If array doesn't meet the specified constraints.
    """
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"{name} must be array-like, got None")

    try:
        value = np.asarray(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} must be array-like, got {type(value).__name__}") from e

    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {value.ndim}")

    if min_length is not None and len(value) < min_length:
        raise ValueError(f"{name} must have at least {min_length} elements, got {len(value)}")

    if max_length is not None and len(value) > max_length:
        raise ValueError(f"{name} must have at most {max_length} elements, got {len(value)}")

    if dtype is not None:
        try:
            value = value.astype(dtype)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{name} cannot be converted to {dtype.__name__}") from e

    return value


def validate_callable(
    value: Any,
    name: str,
    allow_none: bool = False,
) -> Optional[Callable]:
    """Validate callable parameters.

    Args:
        value: Value to validate.
        name: Parameter name for error messages.
        allow_none: Whether None is allowed.

    Returns:
        Validated callable or None.

    Raises:
        TypeError: If value is not callable or None (when allowed).
    """
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"{name} must be callable, got None")

    if not callable(value):
        raise TypeError(f"{name} must be callable, got {type(value).__name__}")

    return value


def validate_random_state(
    value: Any,
    name: str = "random_state",
    allow_none: bool = True,
) -> Optional[Union[int, np.random.RandomState, np.random.Generator]]:
    """Validate random state parameters.

    Args:
        value: Value to validate (int, RandomState, Generator, or None).
        name: Parameter name for error messages.
        allow_none: Whether None is allowed.

    Returns:
        Validated random state or None.

    Raises:
        TypeError: If value is not a valid random state type.
        ValueError: If integer value is negative.
    """
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"{name} must be an integer, RandomState, or Generator, got None")

    if isinstance(value, (int, np.integer)):
        value = int(value)
        if value < 0:
            raise ValueError(f"{name} must be non-negative when integer, got {value}")
        return value

    if isinstance(value, (np.random.RandomState, np.random.Generator)):
        return value

    raise TypeError(
        f"{name} must be an integer, RandomState, or Generator, " f"got {type(value).__name__}"
    )


def validate_n_bins(
    value: Any,
    name: str = "n_bins",
    min_bins: int = 1,
    max_bins: Optional[int] = None,
    warn_single_bin: bool = True,
) -> int:
    """Validate n_bins parameter with binning-specific logic.

    Args:
        value: Value to validate.
        name: Parameter name for error messages.
        min_bins: Minimum allowed number of bins.
        max_bins: Maximum allowed number of bins.
        warn_single_bin: Whether to warn when n_bins=1.

    Returns:
        Validated number of bins.

    Raises:
        TypeError: If value is not an integer.
        ValueError: If value is outside the allowed range.
    """
    n_bins = validate_int(value, name, min_val=min_bins, max_val=max_bins, allow_none=False)

    if warn_single_bin and n_bins == 1:
        warnings.warn(
            f"{name}=1 will result in a single bin for all values", UserWarning, stacklevel=3
        )

    return n_bins  # type: ignore[return-value] # validate_int with allow_none=False returns int


def validate_binning_columns(
    value: Any,
    name: str = "binning_columns",
    allow_none: bool = True,
) -> Optional[list]:
    """Validate column specifications for binning.

    Args:
        value: Value to validate (string, int, list, or None).
        name: Parameter name for error messages.
        allow_none: Whether None is allowed.

    Returns:
        Validated list of column specifications or None.

    Raises:
        TypeError: If value is not a valid column specification.
    """
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"{name} must be a column specification, got None")

    if isinstance(value, (str, int)):
        return [value]

    if isinstance(value, (list, tuple)):
        # Validate each element
        validated = []
        for i, col in enumerate(value):
            if not isinstance(col, (str, int)):
                raise TypeError(f"{name}[{i}] must be string or integer, got {type(col).__name__}")
            validated.append(col)
        return validated

    raise TypeError(
        f"{name} must be string, integer, or list of strings/integers, "
        f"got {type(value).__name__}"
    )


class ParameterValidator:
    """Centralized parameter validation for binning methods.

    This class provides a convenient interface for validating multiple
    parameters with consistent error handling and warnings.
    """

    def __init__(self, class_name: str):
        """Initialize validator for a specific class.

        Args:
            class_name: Name of the class using this validator.
        """
        self.class_name = class_name
        self.errors = []
        self.warnings = []

    def validate(self, **validations) -> dict[str, Any]:
        """Validate multiple parameters at once.

        Args:
            **validations: Parameter validations as name=(value, validator_func, *args).

        Returns:
            Dictionary of validated parameters.

        Raises:
            ValueError: If any validation fails.
        """
        validated = {}

        for param_name, validation_spec in validations.items():
            try:
                if isinstance(validation_spec, tuple):
                    value, validator_func = validation_spec[0], validation_spec[1]
                    args = validation_spec[2:] if len(validation_spec) > 2 else ()
                    validated[param_name] = validator_func(value, param_name, *args)
                else:
                    # Assume it's just a value that doesn't need validation
                    validated[param_name] = validation_spec
            except (TypeError, ValueError) as e:
                self.errors.append(f"{self.class_name}: {str(e)}")

        if self.errors:
            error_msg = "; ".join(self.errors)
            raise ValueError(f"Parameter validation failed: {error_msg}")

        return validated

    def reset(self) -> None:
        """Reset error and warning lists."""
        self.errors = []
        self.warnings = []
