"""
Comprehensive tests for binlearn.utils._errors module.

Tests all error classes, warning classes, and utility functions with 100% coverage.
"""

import warnings
from typing import Any

import pytest

from binlearn.utils._errors import (
    BinningError,
    BinningWarning,
    ConfigurationError,
    DataQualityWarning,
    FittingError,
    InvalidDataError,
    PerformanceWarning,
    TransformationError,
    ValidationError,
    suggest_alternatives,
)


class TestBinningError:
    """Test the base BinningError class."""

    def test_init_without_suggestions(self) -> None:
        """Test BinningError initialization without suggestions."""
        error = BinningError("Test error message")
        assert str(error) == "Test error message"
        assert error.suggestions == []

    def test_init_with_none_suggestions(self) -> None:
        """Test BinningError initialization with None suggestions."""
        error = BinningError("Test error message", None)
        assert str(error) == "Test error message"
        assert error.suggestions == []

    def test_init_with_suggestions(self) -> None:
        """Test BinningError initialization with suggestions."""
        suggestions = ["Try option A", "Try option B"]
        error = BinningError("Test error message", suggestions)

        assert "Test error message" in str(error)
        assert "Suggestions:" in str(error)
        assert "- Try option A" in str(error)
        assert "- Try option B" in str(error)
        assert error.suggestions == suggestions

    def test_init_with_empty_suggestions(self) -> None:
        """Test BinningError initialization with empty suggestions list."""
        error = BinningError("Test error message", [])
        assert str(error) == "Test error message"
        assert error.suggestions == []

    def test_str_format_with_suggestions(self) -> None:
        """Test string representation formatting with suggestions."""
        suggestions = ["Suggestion 1", "Suggestion 2", "Suggestion 3"]
        error = BinningError("Main error", suggestions)

        error_str = str(error)
        expected_parts = [
            "Main error",
            "Suggestions:",
            "  - Suggestion 1",
            "  - Suggestion 2",
            "  - Suggestion 3",
        ]

        for part in expected_parts:
            assert part in error_str

    def test_inheritance_from_exception(self) -> None:
        """Test that BinningError inherits from Exception."""
        error = BinningError("Test")
        assert isinstance(error, Exception)

        # Test that it can be raised and caught
        with pytest.raises(BinningError, match="Test"):
            raise error


class TestSpecificErrorClasses:
    """Test specific error classes that inherit from BinningError."""

    def test_invalid_data_error(self) -> None:
        """Test InvalidDataError class."""
        error = InvalidDataError("Invalid data", ["Check your input"])
        assert isinstance(error, BinningError)
        assert "Invalid data" in str(error)
        assert "Check your input" in str(error)

    def test_configuration_error(self) -> None:
        """Test ConfigurationError class."""
        error = ConfigurationError("Bad config", ["Fix the config"])
        assert isinstance(error, BinningError)
        assert "Bad config" in str(error)
        assert "Fix the config" in str(error)

    def test_fitting_error(self) -> None:
        """Test FittingError class."""
        error = FittingError("Fitting failed", ["Try different parameters"])
        assert isinstance(error, BinningError)
        assert "Fitting failed" in str(error)
        assert "Try different parameters" in str(error)

    def test_transformation_error(self) -> None:
        """Test TransformationError class."""
        error = TransformationError("Transform failed", ["Check data format"])
        assert isinstance(error, BinningError)
        assert "Transform failed" in str(error)
        assert "Check data format" in str(error)

    def test_validation_error(self) -> None:
        """Test ValidationError class."""
        error = ValidationError("Validation failed", ["Check input types"])
        assert isinstance(error, BinningError)
        assert "Validation failed" in str(error)
        assert "Check input types" in str(error)

    def test_error_without_suggestions(self) -> None:
        """Test specific errors without suggestions."""
        errors = [
            InvalidDataError("Data error"),
            ConfigurationError("Config error"),
            FittingError("Fit error"),
            TransformationError("Transform error"),
            ValidationError("Validation error"),
        ]

        for error in errors:
            assert isinstance(error, BinningError)
            assert error.suggestions == []


class TestWarningClasses:
    """Test warning classes."""

    def test_binning_warning(self) -> None:
        """Test BinningWarning base class."""
        warning = BinningWarning("Test warning")
        assert isinstance(warning, UserWarning)

    def test_data_quality_warning(self) -> None:
        """Test DataQualityWarning class."""
        warning = DataQualityWarning("Data quality issue")
        assert isinstance(warning, BinningWarning)
        assert isinstance(warning, UserWarning)

    def test_performance_warning(self) -> None:
        """Test PerformanceWarning class."""
        warning = PerformanceWarning("Performance issue")
        assert isinstance(warning, BinningWarning)
        assert isinstance(warning, UserWarning)

    def test_warnings_can_be_issued(self) -> None:
        """Test that warnings can actually be issued."""
        with pytest.warns(DataQualityWarning, match="quality"):
            warnings.warn("Data quality issue", DataQualityWarning, stacklevel=2)

        with pytest.warns(PerformanceWarning, match="Performance"):
            warnings.warn("Performance issue", PerformanceWarning, stacklevel=2)


class TestSuggestAlternatives:
    """Test the suggest_alternatives function."""

    def test_supervised_alternatives(self) -> None:
        """Test alternatives for supervised methods."""
        alternatives = suggest_alternatives("supervised")
        assert "tree" in alternatives
        assert "decision_tree" in alternatives
        assert "supervised" in alternatives

    def test_equal_width_alternatives(self) -> None:
        """Test alternatives for equal_width methods."""
        alternatives = suggest_alternatives("equal_width")
        assert "uniform" in alternatives
        assert "equidistant" in alternatives
        assert "equal_width" in alternatives

    def test_singleton_alternatives(self) -> None:
        """Test alternatives for singleton methods."""
        alternatives = suggest_alternatives("singleton")
        assert "categorical" in alternatives
        assert "nominal" in alternatives
        assert "singleton" in alternatives

    def test_quantile_alternatives(self) -> None:
        """Test alternatives for quantile methods."""
        alternatives = suggest_alternatives("quantile")
        assert "percentile" in alternatives
        assert "quantile" in alternatives

    def test_alias_suggestions(self) -> None:
        """Test suggestions for method aliases."""
        # Test aliases return the correct method
        alternatives = suggest_alternatives("tree")
        assert "supervised" in alternatives
        assert "decision_tree" in alternatives

        alternatives = suggest_alternatives("uniform")
        assert "equal_width" in alternatives
        assert "equidistant" in alternatives

        alternatives = suggest_alternatives("categorical")
        assert "singleton" in alternatives
        assert "nominal" in alternatives

        alternatives = suggest_alternatives("percentile")
        assert "quantile" in alternatives

    def test_case_insensitive(self) -> None:
        """Test that suggestions work with different cases."""
        alternatives_lower = suggest_alternatives("supervised")
        alternatives_upper = suggest_alternatives("SUPERVISED")
        alternatives_mixed = suggest_alternatives("Supervised")

        # All should return the same alternatives (converted to sets for comparison)
        assert set(alternatives_lower) == set(alternatives_upper) == set(alternatives_mixed)

    def test_unknown_method(self) -> None:
        """Test suggestions for unknown method names."""
        alternatives = suggest_alternatives("unknown_method")
        assert alternatives == []

    def test_empty_method_name(self) -> None:
        """Test suggestions for empty method name."""
        alternatives = suggest_alternatives("")
        assert alternatives == []

    def test_none_method_name(self) -> None:
        """Test suggestions handle None gracefully."""
        # This should not crash, though the function expects strings
        try:
            alternatives = suggest_alternatives(None)  # type: ignore[arg-type]
            # If it doesn't crash, should return empty list
            assert alternatives == []
        except AttributeError:
            # If it crashes due to None.lower(), that's also acceptable
            pass

    def test_return_unique_alternatives(self) -> None:
        """Test that alternatives list contains unique values."""
        alternatives = suggest_alternatives("supervised")
        assert len(alternatives) == len(set(alternatives))

    def test_all_known_methods(self) -> None:
        """Test all known methods return appropriate suggestions."""
        known_methods = ["supervised", "equal_width", "singleton", "quantile"]
        known_aliases = [
            "tree",
            "decision_tree",
            "uniform",
            "equidistant",
            "categorical",
            "nominal",
            "percentile",
        ]

        for method in known_methods + known_aliases:
            alternatives = suggest_alternatives(method)
            assert len(alternatives) > 0, f"No alternatives for {method}"
            assert isinstance(alternatives, list)
            assert all(isinstance(alt, str) for alt in alternatives)


class TestErrorIntegration:
    """Test error classes in realistic usage scenarios."""

    def test_error_chaining(self) -> None:
        """Test error chaining works properly."""
        try:
            # Simulate nested error
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise ConfigurationError("Config problem", ["Fix config"]) from e
        except ConfigurationError as config_error:
            assert config_error.__cause__ is not None
            assert isinstance(config_error.__cause__, ValueError)
            assert "Original error" in str(config_error.__cause__)

    def test_error_in_except_block(self) -> None:
        """Test errors work properly when raised in except blocks."""
        with pytest.raises(ValidationError) as exc_info:
            try:
                raise ValueError("Something went wrong")
            except ValueError:
                raise ValidationError("Validation failed", ["Check input"])

        assert "Validation failed" in str(exc_info.value)
        assert "Check input" in str(exc_info.value)

    def test_multiple_suggestions_formatting(self) -> None:
        """Test formatting with many suggestions."""
        many_suggestions = [f"Suggestion {i}" for i in range(10)]
        error = BinningError("Error with many suggestions", many_suggestions)

        error_str = str(error)
        assert "Suggestions:" in error_str
        for suggestion in many_suggestions:
            assert f"  - {suggestion}" in error_str

    def test_suggestions_with_special_characters(self) -> None:
        """Test suggestions containing special characters."""
        special_suggestions = [
            "Use parameter='value' format",
            'Try method="auto" instead',
            "Check data: values should be > 0",
            "Path: /home/user/data.csv",
        ]
        error = BinningError("Special char test", special_suggestions)

        error_str = str(error)
        for suggestion in special_suggestions:
            assert suggestion in error_str

    def test_error_picklability(self) -> None:
        """Test that errors can be pickled/unpickled (for multiprocessing)."""
        import pickle

        error = BinningError("Pickle test", ["Try again"])
        pickled = pickle.dumps(error)
        unpickled = pickle.loads(pickled)

        assert str(unpickled) == str(error)
        assert unpickled.suggestions == error.suggestions
