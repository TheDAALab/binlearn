"""
Standardized error handling utilities for binning methods.

This module provides common error handling patterns and messages used across
multiple binning implementations to ensure consistency.
"""

from __future__ import annotations

import warnings
from typing import Any

from ._errors import BinningError, ConfigurationError


def handle_insufficient_data_error(
    data_size: int, min_required: int, method_name: str
) -> ConfigurationError:
    """Handle insufficient data errors with standardized message.

    Args:
        data_size: Actual size of the data
        min_required: Minimum required data size
        method_name: Name of the binning method

    Returns:
        ConfigurationError with helpful message and suggestions
    """
    return ConfigurationError(
        f"{method_name} requires at least {min_required} data points, got {data_size}",
        suggestions=[
            f"Provide more data (at least {min_required} points)",
            "Use a simpler binning method like equal-width or equal-frequency",
            "Reduce the number of bins to work with smaller datasets",
        ],
    )


def handle_convergence_warning(
    method_name: str, max_iterations: int, suggest_fallback: bool = True
) -> None:
    """Issue standardized convergence warning.

    Args:
        method_name: Name of the binning method
        max_iterations: Maximum iterations reached
        suggest_fallback: Whether to suggest fallback methods
    """
    message = (
        f"{method_name} did not converge within {max_iterations} iterations. "
        f"Results may not be optimal."
    )

    if suggest_fallback:
        message += " Consider using equal-width binning as a fallback."

    warnings.warn(message, category=UserWarning, stacklevel=3)


def handle_parameter_bounds_error(
    param_name: str,
    value: Any,
    min_val: float | None = None,
    max_val: float | None = None,
    suggestions: list[str] | None = None,
) -> ConfigurationError:
    """Handle parameter bounds errors with standardized message.

    Args:
        param_name: Name of the parameter
        value: Actual value provided
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        suggestions: Custom suggestions for fixing the error

    Returns:
        ConfigurationError with helpful message
    """
    if min_val is not None and max_val is not None:
        bounds_desc = f"between {min_val} and {max_val}"
        example_val = (min_val + max_val) / 2
    elif min_val is not None:
        bounds_desc = f"at least {min_val}"
        example_val = max(min_val, 1.0)
    elif max_val is not None:
        bounds_desc = f"at most {max_val}"
        example_val = min(max_val, 1.0)
    else:
        bounds_desc = "within valid range"
        example_val = 1.0

    default_suggestions = [f"Example: {param_name}={example_val}"]
    final_suggestions = suggestions if suggestions is not None else default_suggestions

    return ConfigurationError(
        f"{param_name} must be {bounds_desc}, got {value}", suggestions=final_suggestions
    )


def validate_fitted_state(obj: Any, method_name: str = "transform") -> None:
    """Validate that an estimator has been fitted before use.

    Args:
        obj: Object to check for fitted state
        method_name: Name of method being called (for error message)

    Raises:
        BinningError: If object is not fitted
    """
    # Check for common fitted attributes
    fitted_attrs = ["bin_edges_", "bins_", "boundaries_", "fitted_"]

    is_fitted = any(hasattr(obj, attr) for attr in fitted_attrs)

    if not is_fitted:
        raise BinningError(
            f"Cannot call {method_name}() before calling fit()",
            suggestions=[
                "Call fit() method first",
                "Or use fit_transform() to fit and transform in one step",
            ],
        )
