"""
Test suite for parameter_conversion utilities.

This module tests the utility functions in parameter_conversion module that support
string parameter specifications for bin numbers across all binning methods.
"""

import pytest

from binlearn.utils._errors import ConfigurationError
from binlearn.utils._parameter_conversion import (
    resolve_n_bins_parameter,
    resolve_string_parameter,
    validate_bin_number_for_calculation,
    validate_bin_number_parameter,
    validate_numeric_parameter,
)


class TestParameterConversionUtilities:
    """Test parameter conversion utility functions."""

    def test_resolve_n_bins_string_specifications(self):
        """Test resolving string specifications for n_bins."""
        # Test sqrt
        assert resolve_n_bins_parameter("sqrt", data_shape=(16, 1)) == 4
        assert resolve_n_bins_parameter("sqrt", data_shape=(100, 1)) == 10

        # Test log specifications
        assert resolve_n_bins_parameter("log2", data_shape=(8, 1)) == 3
        assert resolve_n_bins_parameter("log10", data_shape=(100, 1)) == 2
        assert resolve_n_bins_parameter("sturges", data_shape=(8, 1)) == 4

        # Test case insensitive
        assert resolve_n_bins_parameter("SQRT", data_shape=(16, 1)) == 4

    def test_resolve_n_bins_integer_passthrough(self):
        """Test that integer values pass through unchanged."""
        assert resolve_n_bins_parameter(5) == 5
        assert resolve_n_bins_parameter(10) == 10

    def test_resolve_n_bins_invalid_cases(self):
        """Test error handling for invalid cases."""
        # Missing data_shape for string
        with pytest.raises(ConfigurationError, match="requires data to be fitted first"):
            resolve_n_bins_parameter("sqrt")

        # Invalid integer
        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            resolve_n_bins_parameter(0)

    def test_validate_bin_number_parameter(self):
        """Test bin number parameter validation."""
        # Valid integers should not raise
        validate_bin_number_parameter(5)  # Should not raise

        # Valid strings should not raise
        validate_bin_number_parameter("sqrt")  # Should not raise
        validate_bin_number_parameter("log")  # Should not raise

        # Invalid cases should raise
        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            validate_bin_number_parameter(0)

    def test_validate_bin_number_for_calculation(self):
        """Test validation for calculation context."""
        # Integers should be validated
        validate_bin_number_for_calculation(5)  # Should not raise

        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            validate_bin_number_for_calculation(0)

        # Strings should pass through
        validate_bin_number_for_calculation("sqrt")  # Should not raise

    def test_validate_numeric_parameter_edge_cases(self):
        """Test numeric parameter validation edge cases."""
        # None handling
        assert validate_numeric_parameter(None, "test", allow_none=True) is None

        with pytest.raises(ConfigurationError, match="test cannot be None"):
            validate_numeric_parameter(None, "test", allow_none=False)

        # Boolean exclusion
        with pytest.raises(ConfigurationError, match="test must be numeric"):
            validate_numeric_parameter(True, "test")

        # Integer only validation
        with pytest.raises(ConfigurationError, match="test must be an integer"):
            validate_numeric_parameter(True, "test", integer_only=True)

        # Bounds checking
        with pytest.raises(ConfigurationError, match="test must be >= 5"):
            validate_numeric_parameter(3, "test", min_value=5)

        with pytest.raises(ConfigurationError, match="test must be <= 10"):
            validate_numeric_parameter(15, "test", max_value=10)

    def test_resolve_string_parameter_function(self):
        """Test resolve_string_parameter function to cover lines 221-258."""
        # Test valid resolution
        options = {"small": 5, "medium": 10, "large": 20}
        assert resolve_string_parameter("small", options, "test_param") == 5
        assert resolve_string_parameter("medium", options, "test_param") == 10

        # Test non-string passthrough
        assert resolve_string_parameter(42, options, "test_param") == 42
        assert resolve_string_parameter(3.14, options, "test_param") == 3.14

        # Test invalid string
        with pytest.raises(ConfigurationError, match="Invalid test_param specification"):
            resolve_string_parameter("invalid", options, "test_param")

        # Test with allow_passthrough=False
        with pytest.raises(ConfigurationError, match="test_param must be one of"):
            resolve_string_parameter(42, options, "test_param", allow_passthrough=False)

    def test_validate_numeric_parameter_comprehensive(self):
        """Test all paths in validate_numeric_parameter to cover remaining lines."""
        # Test integer validation with booleans (should fail)
        with pytest.raises(ConfigurationError, match="test must be an integer"):
            validate_numeric_parameter(True, "test", integer_only=True)

        with pytest.raises(ConfigurationError, match="test must be an integer"):
            validate_numeric_parameter(False, "test", integer_only=True)

        # Test general numeric validation with various invalid types
        with pytest.raises(ConfigurationError, match="test must be numeric"):
            validate_numeric_parameter("string", "test")

        with pytest.raises(ConfigurationError, match="test must be numeric"):
            validate_numeric_parameter([], "test")

        with pytest.raises(ConfigurationError, match="test must be numeric"):
            validate_numeric_parameter({}, "test")

        # Test valid numeric values
        assert validate_numeric_parameter(10, "test") == 10
        assert validate_numeric_parameter(10.5, "test") == 10.5
        assert validate_numeric_parameter(-5, "test") == -5

    def test_parameter_conversion_integration(self):
        """Test full parameter conversion workflow."""
        # Integer workflow
        validate_bin_number_parameter(5)  # Should not raise
        validate_bin_number_for_calculation(5)  # Should not raise
        resolved = resolve_n_bins_parameter(5)  # Direct integer
        assert resolved == 5

        # String workflow
        validate_bin_number_parameter("sqrt")  # Should not raise
        validate_bin_number_for_calculation("sqrt")  # Should not raise
        resolved = resolve_n_bins_parameter("sqrt", data_shape=(16, 1))
        assert resolved == 4

    def test_validate_numeric_parameter_non_integer_only(self):
        """Test validate_numeric_parameter with integer_only=False to cover line 232->246."""
        # This test covers the branch where integer_only=False
        # causing the code to skip integer-only validation and go to general numeric validation

        # Test valid float with integer_only=False (this should trigger the 232->246 branch)
        result = validate_numeric_parameter(3.14, "test_param", integer_only=False)
        assert result == 3.14

        # Test valid int with integer_only=False
        result = validate_numeric_parameter(5, "test_param", integer_only=False)
        assert result == 5

        # Test that boolean is still rejected even with integer_only=False
        with pytest.raises(ConfigurationError, match="test_param must be numeric"):
            validate_numeric_parameter(True, "test_param", integer_only=False)

        # Test invalid type with integer_only=False (should still fail)
        with pytest.raises(ConfigurationError, match="test_param must be numeric"):
            validate_numeric_parameter("string", "test_param", integer_only=False)

    def test_validate_numeric_parameter_integer_only_true_branch(self):
        """Test validate_numeric_parameter with integer_only=True for completeness."""
        # This ensures we test the integer_only=True path as well

        # Valid integer
        result = validate_numeric_parameter(10, "test_param", integer_only=True)
        assert result == 10

        # Invalid: float when integer required
        with pytest.raises(ConfigurationError, match="test_param must be an integer"):
            validate_numeric_parameter(3.14, "test_param", integer_only=True)

        # Invalid: boolean when integer required
        with pytest.raises(ConfigurationError, match="test_param must be an integer"):
            validate_numeric_parameter(True, "test_param", integer_only=True)

    def test_validate_bin_number_parameter_non_integer_types(self):
        """Test validate_bin_number_parameter with non-integer types to cover line 377->380."""
        # This test covers the branch where the value is not an int
        # causing the code to jump from line 377 to line 387 (elif isinstance(value, str))

        # Test with string (valid path through elif) - covers line 389 (valid string path)
        validate_bin_number_parameter(
            "sqrt"
        )  # Should not raise - covers positive string validation
        validate_bin_number_parameter("log2")  # Should not raise
        validate_bin_number_parameter("LOG")  # Case insensitive test
        validate_bin_number_parameter("STURGES")  # Case insensitive test

        # Test with invalid string (covers negative string validation path)
        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            validate_bin_number_parameter("invalid_string")

        # Test with invalid non-int, non-string type (goes to else branch)
        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            validate_bin_number_parameter(3.14)  # type: ignore # float

        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            validate_bin_number_parameter([])  # type: ignore # list

        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            validate_bin_number_parameter({})  # type: ignore # dict

        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            validate_bin_number_parameter(None)  # type: ignore # None

    def test_coverage_specific_branches(self):
        """Specific tests to cover the exact missing branches 232->246 and 377->380."""

        # Test that explicitly triggers the else branch on line 238
        result = validate_numeric_parameter(10.5, "test_param", integer_only=False)
        assert result == 10.5

        # Test 377->380: non-int path in validate_bin_number_parameter
        # This should cause jump from line 380 (if isinstance(value, int):) to line 387
        # (elif isinstance(value, str):)

        # First test with float (should go to else branch)
        with pytest.raises(ConfigurationError):
            validate_bin_number_parameter(10.5)  # type: ignore

        # Then test with string (should go to elif branch)
        validate_bin_number_parameter("sqrt")  # This should not raise

    def test_validate_bin_number_exact_else_branch(self):
        """Test to specifically trigger the else branch (377->380) in
        validate_bin_number_parameter."""
        # This test should trigger the exact path: not int, not string -> else clause

        # Test with float (neither int nor string) - should go to else branch
        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            validate_bin_number_parameter(3.14)  # type: ignore

        # Test with complex type (neither int nor string) - should go to else branch
        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            validate_bin_number_parameter(complex(1, 2))  # type: ignore

        # Test with tuple (neither int nor string) - should go to else branch
        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            validate_bin_number_parameter((1, 2))  # type: ignore

    def test_validate_bin_number_with_custom_valid_strings(self):
        """Test validate_bin_number_parameter with custom valid_strings to cover different
        branches."""
        # Test with custom valid_strings set to exercise different validation paths
        custom_strings = {"small", "medium", "large"}

        # Test valid string with custom set
        validate_bin_number_parameter("small", "n_bins", custom_strings)  # Should not raise
        validate_bin_number_parameter("medium", "n_bins", custom_strings)  # Should not raise

        # Test invalid string with custom set (this should trigger different branch behavior)
        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            validate_bin_number_parameter(
                "sqrt", "n_bins", custom_strings
            )  # 'sqrt' not in custom set

        # Test the default valid_strings behavior vs custom behavior
        # With default set, "sqrt" is valid
        validate_bin_number_parameter("sqrt")  # Should not raise

        # With custom set, "sqrt" is invalid
        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            validate_bin_number_parameter("sqrt", "n_bins", custom_strings)

        # Test that integers still work with custom valid_strings
        validate_bin_number_parameter(10, "n_bins", custom_strings)  # Should not raise

        # Test edge case: empty custom valid_strings set
        empty_strings = set()
        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            validate_bin_number_parameter("any_string", "n_bins", empty_strings)
