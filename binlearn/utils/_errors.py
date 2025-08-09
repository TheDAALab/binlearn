"""
Enhanced error handling for the binning framework.

This module provides custom exception classes and warning types specifically designed
for binning operations, along with utility functions to enhance error reporting
with helpful suggestions for common issues.
"""


class BinningError(Exception):
    """Base exception for all binning-related errors.

    This is the base class for all custom exceptions in the binlearn library.
    It supports providing helpful suggestions alongside error messages to guide
    users toward solutions.

    Args:
        message: The error message describing what went wrong.
        suggestions: Optional list of suggestions to help resolve the error.
            Each suggestion should be a complete, actionable statement.

    Attributes:
        suggestions: List of suggestion strings for resolving the error.

    Example:
        >>> raise BinningError(
        ...     "Invalid parameter value",
        ...     suggestions=["Use a positive integer", "Check the documentation"]
        ... )
    """

    def __init__(self, message: str, suggestions: list[str] | None = None):
        """Initialize the BinningError with message and optional suggestions.

        Args:
            message: The error message describing what went wrong.
            suggestions: Optional list of suggestions to help resolve the error.
        """
        super().__init__(message)
        self.suggestions = suggestions or []

    def __str__(self) -> str:
        """Return formatted error message with suggestions if available.

        Returns:
            Formatted error message including suggestions if provided.
        """
        msg = super().__str__()
        if self.suggestions:
            suggestions_text = "\n".join(f"  - {s}" for s in self.suggestions)
            msg += f"\n\nSuggestions:\n{suggestions_text}"
        return msg


class InvalidDataError(BinningError):
    """Raised when input data is invalid or incompatible with the binning method.

    This error is typically raised when:
    - Input data contains invalid values (NaN, infinite values where not supported)
    - Data shape is incompatible with the binning method requirements
    - Data type is not supported by the specific binning algorithm
    - Data contains values outside the expected range

    Example:
        >>> raise InvalidDataError(
        ...     "Input contains NaN values",
        ...     suggestions=["Remove NaN values with dropna()", "Use fillna() to impute values"]
        ... )
    """


class ConfigurationError(BinningError):
    """Raised when configuration parameters are invalid or incompatible.

    This error is typically raised when:
    - Parameter values are outside valid ranges
    - Parameter combinations are incompatible
    - Required parameters are missing
    - Parameter types are incorrect

    Example:
        >>> raise ConfigurationError(
        ...     "n_bins must be positive",
        ...     suggestions=["Set n_bins to a positive integer like 10"]
        ... )
    """


class FittingError(BinningError):
    """Raised when the binning algorithm fails during the fitting process.

    This error is typically raised when:
    - Algorithm cannot converge to a solution
    - Data is insufficient for the chosen method
    - Numerical issues prevent successful fitting
    - Method-specific requirements are not met

    Example:
        >>> raise FittingError(
        ...     "Algorithm failed to converge",
        ...     suggestions=["Try reducing n_bins", "Check data quality", "Use a simpler method"]
        ... )
    """


class TransformationError(BinningError):
    """Raised when transformation of data fails.

    This error is typically raised when:
    - Transform is called before fit
    - New data contains values outside the fitted range
    - Transformation logic encounters unexpected conditions
    - Memory or computational limits are exceeded during transformation

    Example:
        >>> raise TransformationError(
        ...     "Cannot transform: estimator not fitted",
        ...     suggestions=["Call fit() before transform()", "Use fit_transform() instead"]
        ... )
    """


class ValidationError(BinningError):
    """Raised when data or parameter validation fails.

    This error is typically raised when:
    - Input validation detects inconsistent data
    - Parameter validation finds invalid configurations
    - Cross-validation of results fails
    - Integrity checks detect corrupted state

    Example:
        >>> raise ValidationError(
        ...     "Bin edges are not monotonically increasing",
        ...     suggestions=["Sort bin edges in ascending order", "Check for duplicate values"]
        ... )
    """


class BinningWarning(UserWarning):
    """Base warning class for binning operations.

    This is the base class for all warning types in the binlearn library.
    Warnings are used to alert users to potential issues that don't prevent
    execution but may affect results or performance.

    Example:
        >>> import warnings
        >>> warnings.warn("Potential issue detected", BinningWarning)
    """


class DataQualityWarning(BinningWarning):
    """Warning about data quality issues that may affect binning results.

    This warning is typically issued when:
    - Data contains outliers that may skew bin boundaries
    - Data distribution is highly skewed or has unusual characteristics
    - Missing values are detected and handled automatically
    - Data sparsity may affect certain binning methods

    Example:
        >>> import warnings
        >>> warnings.warn(
        ...     "Data contains extreme outliers that may affect bin boundaries",
        ...     DataQualityWarning
        ... )
    """


class PerformanceWarning(BinningWarning):
    """Warning about potential performance issues.

    This warning is typically issued when:
    - Large datasets may cause slow performance with certain methods
    - Parameter choices may lead to excessive memory usage
    - Algorithm configurations may result in long computation times
    - More efficient alternatives are available for the current task

    Example:
        >>> import warnings
        >>> warnings.warn(
        ...     "Large number of bins may slow down computation",
        ...     PerformanceWarning
        ... )
    """


def suggest_alternatives(method_name: str) -> list[str]:
    """Suggest alternative method names for common misspellings or aliases.

    This utility function helps users find the correct method names when they
    use common alternative spellings or aliases. It's particularly useful in
    error messages to guide users toward the correct parameter values.

    Args:
        method_name: The method name to find alternatives for. Case-insensitive
            comparison is performed.

    Returns:
        List of alternative method names that are similar or equivalent to the
        input method name. Returns empty list if no alternatives are found.

    Example:
        >>> suggest_alternatives("supervised")
        ['supervised', 'tree', 'decision_tree']
        >>> suggest_alternatives("uniform")
        ['equal_width', 'uniform', 'equidistant']
        >>> suggest_alternatives("unknown_method")
        []

    Note:
        This function is primarily used internally by error handling code to
        provide helpful suggestions in error messages, but can also be used
        directly for input validation and user assistance.
    """
    alternatives = {
        "supervised": ["tree", "decision_tree"],
        "equal_width": ["uniform", "equidistant"],
        "singleton": ["categorical", "nominal"],
        "quantile": ["percentile"],
    }

    suggestions = []
    for correct, aliases in alternatives.items():
        if method_name.lower() in aliases or method_name.lower() == correct:
            suggestions.extend([correct] + aliases)

    return list(set(suggestions))
