"""
Parameter validation utilities for binning methods.

This module provides standardized parameter validation patterns used across
multiple binning implementations to reduce code duplication and ensure consistency.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._errors import ConfigurationError


def validate_common_parameters(obj: Any, param_specs: dict[str, dict]) -> None:
    """Validate common parameter patterns with specifications.

    Args:
        obj: Object containing parameters to validate
        param_specs: Dictionary specifying validation rules for each parameter
            Format: {param_name: {type: type_spec, range: (min, max), ...}}

    Raises:
        ConfigurationError: If any parameter fails validation
    """
    for param_name, specs in param_specs.items():
        value = getattr(obj, param_name, None)

        if value is None:
            continue

        # Type validation
        if "type" in specs:
            expected_type = specs["type"]
            if not isinstance(value, expected_type):
                raise create_configuration_error(
                    param_name,
                    f"must be of type {expected_type.__name__}, got {type(value).__name__}",
                    [f"Example: {param_name}={specs.get('example', 'valid_value')}"],
                )

        # Numeric range validation
        if "min" in specs or "max" in specs:
            if isinstance(value, (int, float)):
                min_val = specs.get("min", float("-inf"))
                max_val = specs.get("max", float("inf"))
                if not (min_val <= value <= max_val):
                    range_desc = f"[{min_val}, {max_val}]"
                    raise create_configuration_error(
                        param_name,
                        f"must be in range {range_desc}, got {value}",
                        [f"Example: {param_name}={specs.get('example', min_val)}"],
                    )

        # Custom validator
        if "validator" in specs:
            validator = specs["validator"]
            if not validator(value):
                raise create_configuration_error(
                    param_name,
                    specs.get("validator_message", "failed custom validation"),
                    specs.get("suggestions", []),
                )


def validate_range_parameter(value: Any, param_name: str, allow_none: bool = True) -> None:
    """Validate (min, max) range parameters.

    Args:
        value: Value to validate (should be tuple/list of 2 numbers)
        param_name: Name of parameter for error messages
        allow_none: Whether None is acceptable

    Raises:
        ConfigurationError: If range parameter is invalid
    """
    if value is None and allow_none:
        return

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise create_configuration_error(
            param_name,
            "must be a tuple/list of two numbers (min, max)",
            [f"Example: {param_name}=(0, 100)"],
        )

    min_val, max_val = value
    if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
        raise create_configuration_error(
            param_name, "values must be numbers", [f"Example: {param_name}=(0.0, 100.0)"]
        )

    if min_val >= max_val:
        raise create_configuration_error(
            param_name,
            f"minimum ({min_val}) must be less than maximum ({max_val})",
            [f"Example: {param_name}=(0, 100) where 0 < 100"],
        )


def validate_positive_number(value: Any, param_name: str, allow_zero: bool = False) -> None:
    """Validate that a parameter is a positive number.

    Args:
        value: Value to validate
        param_name: Name of parameter for error messages
        allow_zero: Whether zero is acceptable

    Raises:
        ConfigurationError: If value is not a positive number
    """
    if not isinstance(value, (int, float)):
        raise create_configuration_error(
            param_name,
            f"must be a number, got {type(value).__name__}",
            [f"Example: {param_name}=1.0"],
        )

    min_threshold = 0.0 if allow_zero else 0.0
    comparison = ">=" if allow_zero else ">"

    if not (value > min_threshold if not allow_zero else value >= min_threshold):
        raise create_configuration_error(
            param_name,
            f"must be {comparison} {min_threshold}, got {value}",
            [f"Example: {param_name}=1.0"],
        )


def validate_random_state(value: Any, param_name: str = "random_state") -> None:
    """Validate random state parameter.

    Args:
        value: Value to validate (should be int or None)
        param_name: Name of parameter for error messages

    Raises:
        ConfigurationError: If random_state is invalid
    """
    if value is not None and not isinstance(value, int):
        raise create_configuration_error(
            param_name,
            f"must be an integer or None, got {type(value).__name__}",
            [f"Example: {param_name}=42"],
        )

    if isinstance(value, int) and value < 0:
        raise create_configuration_error(
            param_name,
            f"must be non-negative, got {value}",
            [f"Example: {param_name}=42"],
        )


def validate_positive_integer(value: Any, param_name: str) -> None:
    """Validate that a parameter is a positive integer.

    Args:
        value: Value to validate
        param_name: Name of parameter for error messages

    Raises:
        ConfigurationError: If value is not a positive integer
    """
    if not isinstance(value, int) or value <= 0:
        raise create_configuration_error(
            param_name, "must be a positive integer", [f"Example: {param_name}=5"]
        )


def create_configuration_error(
    param_name: str, issue: str, suggestions: list[str]
) -> ConfigurationError:
    """Standardized ConfigurationError creation.

    Args:
        param_name: Name of the problematic parameter
        issue: Description of the issue
        suggestions: List of suggested fixes or examples

    Returns:
        ConfigurationError with standardized formatting
    """
    message = f"{param_name} {issue}"
    return ConfigurationError(message, suggestions=suggestions)


# Common parameter specification templates
COMMON_PARAM_SPECS = {
    "n_bins_spec": {
        "type": (int, str),
        "min": 1,
        "example": 10,
    },
    "random_state_spec": {
        "type": (int, type(None)),
        "min": 0,
        "example": 42,
    },
    "positive_float_spec": {
        "type": (int, float),
        "min": 0,
        "example": 1.0,
    },
    "positive_int_spec": {
        "type": int,
        "min": 1,
        "example": 5,
    },
    "boolean_spec": {
        "type": bool,
        "example": True,
    },
}
