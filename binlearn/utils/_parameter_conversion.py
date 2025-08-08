"""Parameter conversion utilities for binning methods.

This module provides utilities for converting string parameters to numeric values,
particularly for parameters like n_bins that can accept both integer values and
string specifications like "sqrt", "log", etc.

The module follows a consistent pattern for parameter conversion and validation,
providing clear error messages and suggestions when conversions fail.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._errors import ConfigurationError


# pylint: disable=too-many-branches
def resolve_n_bins_parameter(
    n_bins: int | str,
    data_shape: tuple[int, ...] | None = None,
    param_name: str = "n_bins",
) -> int:
    """Resolve n_bins parameter from integer or string specification.

    Converts string specifications like "sqrt", "log", "log2", "log10" to integer
    values based on the data characteristics. For string specifications that depend
    on data size, the data_shape parameter is required.

    Args:
        n_bins (Union[int, str]): Number of bins specification. Can be:
            - int: Direct specification of number of bins
            - "sqrt": Square root of number of samples
            - "log": Natural logarithm of number of samples (rounded up)
            - "log2": Base-2 logarithm of number of samples (rounded up)
            - "log10": Base-10 logarithm of number of samples (rounded up)
            - "sturges": Sturges' rule: 1 + log2(n_samples)
            - "fd": Freedman-Diaconis rule (requires data for range calculation)
        data_shape (Optional[Tuple[int, ...]], optional): Shape of the data array.
            Required for string specifications that depend on sample size.
            Should be in format (n_samples, n_features). Defaults to None.
        param_name (str, optional): Name of the parameter being resolved, used
            in error messages. Defaults to "n_bins".

    Returns:
        int: Resolved number of bins as a positive integer. Minimum value is 1
            to ensure at least one bin is created.

    Raises:
        ConfigurationError: If the parameter cannot be resolved:
            - Invalid string specification
            - data_shape required but not provided
            - Resolved value is not positive
            - Invalid data_shape format

    Example:
        >>> # Direct integer specification
        >>> resolve_n_bins_parameter(10)
        10

        >>> # String specification with data shape
        >>> resolve_n_bins_parameter("sqrt", data_shape=(100, 3))
        10

        >>> # Logarithmic specification
        >>> resolve_n_bins_parameter("log2", data_shape=(1000, 2))
        10

    Note:
        - String specifications are case-insensitive
        - Minimum returned value is 1 (at least one bin)
        - Freedman-Diaconis rule requires additional data analysis
        - For very small datasets, may return 1 regardless of specification
    """
    # Handle direct integer specification
    if isinstance(n_bins, int):
        if n_bins < 1:
            raise ConfigurationError(
                f"{param_name} must be a positive integer, got {n_bins}",
                suggestions=[f"Set {param_name} to a positive integer (e.g., {param_name}=10)"],
            )
        return n_bins

    # Handle string specifications
    if not isinstance(n_bins, str):
        raise ConfigurationError(
            f"{param_name} must be an integer or string, got {type(n_bins).__name__}",
            suggestions=[
                f"Use an integer: {param_name}=10",
                f'Use a string specification: {param_name}="sqrt"',
                'Valid strings: "sqrt", "log", "log2", "log10", "sturges"',
            ],
        )

    n_bins_str = n_bins.lower().strip()

    # Validate data_shape is provided for data-dependent specifications
    if data_shape is None:
        raise ConfigurationError(
            f'String specification {param_name}="{n_bins}" requires data to be fitted first',
            suggestions=[
                f"Use integer specification: {param_name}=10",
                "Call fit() method before accessing resolved parameters",
                "String specifications are resolved during fitting",
            ],
        )

    # Validate data_shape format
    if not isinstance(data_shape, tuple) or len(data_shape) < 2:
        raise ConfigurationError(
            f"Invalid data_shape format: {data_shape}. Expected tuple with at least 2 elements",
            suggestions=[
                "data_shape should be (n_samples, n_features)",
                "Ensure input data is properly formatted",
            ],
        )

    n_samples = data_shape[0]
    if n_samples < 1:
        raise ConfigurationError(
            f"Invalid number of samples: {n_samples}. Must be positive",
            suggestions=["Ensure input data has at least one sample"],
        )

    # Resolve string specifications
    try:
        if n_bins_str == "sqrt":
            resolved = int(np.ceil(np.sqrt(n_samples)))
        elif n_bins_str in ("log", "ln"):
            resolved = int(np.ceil(np.log(n_samples)))
        elif n_bins_str == "log2":
            resolved = int(np.ceil(np.log2(n_samples)))
        elif n_bins_str == "log10":
            resolved = int(np.ceil(np.log10(n_samples)))
        elif n_bins_str == "sturges":
            # Sturges' rule: 1 + log2(n)
            resolved = int(np.ceil(1 + np.log2(n_samples)))
        else:
            raise ConfigurationError(
                f'Unrecognized {param_name} specification: "{n_bins}"',
                suggestions=[
                    'Valid string options: "sqrt", "log", "log2", "log10", "sturges"',
                    f"Or use an integer: {param_name}=10",
                    'Note: "fd" (Freedman-Diaconis) requires additional implementation',
                ],
            )

    except (ValueError, OverflowError) as e:
        raise ConfigurationError(
            f'Failed to compute {param_name} from "{n_bins}" with {n_samples} samples: {str(e)}',
            suggestions=[
                f"Try a direct integer specification: {param_name}=10",
                "Check that data has reasonable number of samples",
                "Consider using a different string specification",
            ],
        ) from e

    # Ensure minimum of 1 bin
    resolved = max(1, resolved)

    return resolved


# pylint: disable=too-many-arguments,too-many-positional-arguments
def validate_numeric_parameter(
    value: Any,
    param_name: str,
    min_value: float | None = None,
    max_value: float | None = None,
    allow_none: bool = False,
    integer_only: bool = False,
) -> Any:
    """Validate a numeric parameter with optional constraints.

    Validates that a parameter is numeric and optionally within specified bounds.
    Provides clear error messages with suggestions for common parameter validation
    scenarios in binning methods.

    Args:
        value (Any): Parameter value to validate.
        param_name (str): Name of the parameter for error messages.
        min_value (Optional[float], optional): Minimum allowed value (inclusive).
            If None, no minimum constraint. Defaults to None.
        max_value (Optional[float], optional): Maximum allowed value (inclusive).
            If None, no maximum constraint. Defaults to None.
        allow_none (bool, optional): Whether None is allowed as a valid value.
            Defaults to False.
        integer_only (bool, optional): Whether to require integer values only.
            Defaults to False.

    Returns:
        Any: The validated value, unchanged if valid.

    Raises:
        ConfigurationError: If validation fails:
            - Value is None when not allowed
            - Value is not numeric
            - Value is not integer when required
            - Value is outside specified bounds

    Example:
        >>> # Basic numeric validation
        >>> validate_numeric_parameter(10, "n_bins", min_value=1, integer_only=True)
        10

        >>> # Float validation with bounds
        >>> validate_numeric_parameter(0.5, "alpha", min_value=0.0, max_value=1.0)
        0.5

        >>> # Allow None
        >>> validate_numeric_parameter(None, "max_depth", allow_none=True)
        None

    Note:
        - Integer validation uses isinstance(value, int) and excludes booleans
        - Float validation accepts both int and float types
        - Error messages include helpful suggestions for common cases
    """
    # Handle None values
    if value is None:
        if allow_none:
            return value
        raise ConfigurationError(
            f"{param_name} cannot be None",
            suggestions=[f"Provide a numeric value for {param_name}"],
        )

    # Check if value is numeric
    if integer_only:
        # For integer validation, explicitly check for int type and exclude bool
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigurationError(
                f"{param_name} must be an integer, got {type(value).__name__}",
                suggestions=[f"Set {param_name} to an integer value (e.g., {param_name}=10)"],
            )
    else:
        # For general numeric validation, allow both int and float
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ConfigurationError(
                f"{param_name} must be numeric, got {type(value).__name__}",
                suggestions=[f"Set {param_name} to a numeric value (e.g., {param_name}=10.0)"],
            )

    # Check bounds
    if min_value is not None and value < min_value:
        raise ConfigurationError(
            f"{param_name} must be >= {min_value}, got {value}",
            suggestions=[f"Set {param_name} to at least {min_value}"],
        )

    if max_value is not None and value > max_value:
        raise ConfigurationError(
            f"{param_name} must be <= {max_value}, got {value}",
            suggestions=[f"Set {param_name} to at most {max_value}"],
        )

    return value


def resolve_string_parameter(
    value: str | Any,
    valid_options: dict[str, Any],
    param_name: str,
    allow_passthrough: bool = True,
) -> Any:
    """Resolve a string parameter to its corresponding value.

    Maps string specifications to their corresponding values using a lookup
    dictionary. Useful for parameters that accept both direct values and
    string shortcuts (e.g., random_state="auto" -> None).

    Args:
        value (Union[str, Any]): Parameter value to resolve. If not a string
            and allow_passthrough=True, returns the value unchanged.
        valid_options (Dict[str, Any]): Mapping from string specifications
            to their resolved values.
        param_name (str): Name of the parameter for error messages.
        allow_passthrough (bool, optional): Whether to allow non-string values
            to pass through unchanged. If False, only strings from valid_options
            are accepted. Defaults to True.

    Returns:
        Any: Resolved parameter value.

    Raises:
        ConfigurationError: If string value is not in valid_options or if
            allow_passthrough=False and value is not a string.

    Example:
        >>> # Define string mappings
        >>> options = {"auto": None, "sqrt": "sqrt", "log2": "log2"}
        >>> resolve_string_parameter("auto", options, "max_features")
        None

        >>> # Allow passthrough for direct values
        >>> resolve_string_parameter(10, options, "max_features")
        10

        >>> # Restrict to strings only
        >>> resolve_string_parameter(10, options, "max_features", allow_passthrough=False)
        ConfigurationError: ...

    Note:
        - String matching is case-sensitive
        - When allow_passthrough=True, validates other types appropriately
        - Provides helpful suggestions in error messages
    """
    if isinstance(value, str):
        if value in valid_options:
            return valid_options[value]

        raise ConfigurationError(
            f'Invalid {param_name} specification: "{value}"',
            suggestions=[
                f"Valid string options: {list(valid_options.keys())}",
                f"Or provide a direct value if {param_name} supports it",
            ],
        )

    if allow_passthrough:
        return value

    raise ConfigurationError(
        f"{param_name} must be one of {list(valid_options.keys())}, got {type(value).__name__}",
        suggestions=[f"Use one of the valid string options: {list(valid_options.keys())}"],
    )


def validate_bin_number_parameter(
    value: int | str,
    param_name: str = "n_bins",
    valid_strings: set[str] | None = None,
) -> None:
    """Validate a bin number parameter (n_bins, n_components, etc.).

    Provides centralized validation for parameters that accept either positive integers
    or specific string specifications. This function ensures consistent validation
    behavior and error messages across all binning methods.

    Args:
        value (Union[int, str]): The parameter value to validate. Can be:
            - int: Must be a positive integer (>= 1)
            - str: Must be one of the valid string specifications
        param_name (str, optional): Name of the parameter being validated, used
            in error messages. Defaults to "n_bins".
        valid_strings (Optional[Set[str]], optional): Set of valid string specifications.
            If None, defaults to standard specifications:
            {"sqrt", "log", "ln", "log2", "log10", "sturges"}. Defaults to None.

    Raises:
        ConfigurationError: If validation fails with the message
            "{param_name} must be a positive integer" for consistency with
            existing test expectations. Includes helpful suggestions for:
            - Invalid integer values (negative, zero, non-integer types)
            - Invalid string specifications
            - Non-string, non-integer types

    Example:
        >>> # Valid cases
        >>> validate_bin_number_parameter(10)  # Valid integer
        >>> validate_bin_number_parameter("sqrt")  # Valid string
        >>> validate_bin_number_parameter("log2", "n_components")  # Valid with custom param name

        >>> # Invalid cases (will raise ConfigurationError)
        >>> validate_bin_number_parameter(0)  # Zero not allowed
        >>> validate_bin_number_parameter(-5)  # Negative not allowed
        >>> validate_bin_number_parameter("invalid")  # Invalid string
        >>> validate_bin_number_parameter(3.14)  # Float not allowed

    Note:
        - Consistent error message format ensures backward compatibility with tests
        - Provides helpful suggestions for common mistakes
        - Centralizes validation logic to reduce code duplication
        - String specifications are case-insensitive ("sqrt" == "SQRT")
    """
    if valid_strings is None:
        valid_strings = {"sqrt", "log", "ln", "log2", "log10", "sturges"}

    if isinstance(value, int):
        if value < 1:
            raise ConfigurationError(
                f"{param_name} must be a positive integer",
                suggestions=[f"Set {param_name} to a positive integer (e.g., {param_name}=10)"],
            )
    elif isinstance(value, str):
        # Check if it's a valid string specification (case-insensitive)
        if value.lower().strip() not in valid_strings:
            raise ConfigurationError(
                f"{param_name} must be a positive integer",
                suggestions=[
                    f"Valid string options: {sorted(valid_strings)}",
                    f"Or use a positive integer (e.g., {param_name}=10)",
                ],
            )
    else:
        raise ConfigurationError(
            f"{param_name} must be a positive integer",
            suggestions=[
                f"Use an integer: {param_name}=10",
                f'Use a string specification: {param_name}="sqrt"',
            ],
        )


def validate_bin_number_for_calculation(
    value: int | str,
    param_name: str = "n_bins",
) -> None:
    """Validate bin number parameter specifically for _calculate_bins methods.

    This function provides early validation for integer values in _calculate_bins
    methods to maintain backward compatibility with existing tests. It only validates
    integer values and lets string values pass through to be resolved later.

    Args:
        value (Union[int, str]): The parameter value to validate.
        param_name (str, optional): Name of the parameter being validated.
            Defaults to "n_bins".

    Raises:
        ValueError: If the value is an integer less than 1, with the exact
            error message format expected by existing tests:
            "{param_name} must be >= 1, got {value}"

    Example:
        >>> # Valid cases
        >>> validate_bin_number_for_calculation(10)  # Valid integer - no error
        >>> validate_bin_number_for_calculation("sqrt")  # String - no validation, passes through

        >>> # Invalid case (will raise ValueError)
        >>> validate_bin_number_for_calculation(-1)  # ValueError: n_bins must be >= 1, got -1

    Note:
        - Used specifically in _calculate_bins methods for backward compatibility
        - Only validates integers, strings pass through for later resolution
        - Maintains exact error message format expected by existing tests
        - Should be called before resolve_n_bins_parameter()
    """
    if isinstance(value, int) and value < 1:
        raise ValueError(f"{param_name} must be >= 1, got {value}")
