"""
Enhanced error handling for the binning framework.
"""





class BinningError(Exception):
    """Base exception for all binning-related errors."""

    def __init__(self, message: str, suggestions: list[str] | None = None):
        super().__init__(message)
        self.suggestions = suggestions or []

    def __str__(self) -> str:
        msg = super().__str__()
        if self.suggestions:
            suggestions_text = "\n".join(f"  - {s}" for s in self.suggestions)
            msg += f"\n\nSuggestions:\n{suggestions_text}"
        return msg


class InvalidDataError(BinningError):
    """Raised when input data is invalid or incompatible."""


class ConfigurationError(BinningError):
    """Raised when configuration parameters are invalid."""


class FittingError(BinningError):
    """Raised when fitting process fails."""


class TransformationError(BinningError):
    """Raised when transformation fails."""


class ValidationError(BinningError):
    """Raised when validation fails."""


class BinningWarning(UserWarning):
    """Base warning for binning operations."""


class DataQualityWarning(BinningWarning):
    """Warning about data quality issues."""


class PerformanceWarning(BinningWarning):
    """Warning about potential performance issues."""


def suggest_alternatives(method_name: str) -> list[str]:
    """Suggest alternative method names for common misspellings."""
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
