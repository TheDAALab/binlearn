"""
Comprehensive tests for binlearn.utils._validation module.

Tests all validation and parameter conversion functions with 100% coverage.
"""

import numpy as np
import pytest

from binlearn.utils._errors import ConfigurationError
from binlearn.utils._validation import (
    resolve_n_bins_parameter,
    resolve_string_parameter,
    validate_bin_number_for_calculation,
    validate_bin_number_parameter,
    validate_numeric_parameter,
    validate_tree_params,
)


class TestResolveNBinsParameter:
    """Test resolve_n_bins_parameter function."""

    def test_resolve_integer_bins(self) -> None:
        """Test resolving integer n_bins values."""
        # Valid integers
        assert resolve_n_bins_parameter(1) == 1
        assert resolve_n_bins_parameter(10) == 10
        assert resolve_n_bins_parameter(100) == 100

    def test_resolve_integer_bins_invalid(self) -> None:
        """Test invalid integer n_bins values."""
        # Zero bins
        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            resolve_n_bins_parameter(0)

        # Negative bins
        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            resolve_n_bins_parameter(-5)

    def test_resolve_string_bins_sqrt(self) -> None:
        """Test resolving sqrt string specification."""
        # sqrt(100) = 10
        result = resolve_n_bins_parameter("sqrt", data_shape=(100, 3))
        assert result == 10

        # sqrt(49) = 7
        result = resolve_n_bins_parameter("sqrt", data_shape=(49, 2))
        assert result == 7

        # sqrt(50) = 7.07... -> ceil -> 8
        result = resolve_n_bins_parameter("sqrt", data_shape=(50, 2))
        assert result == 8

    def test_resolve_string_bins_log(self) -> None:
        """Test resolving log string specifications."""
        # Natural log
        result = resolve_n_bins_parameter("log", data_shape=(1000, 2))
        expected = int(np.ceil(np.log(1000)))  # ~7
        assert result == expected

        result = resolve_n_bins_parameter("ln", data_shape=(1000, 2))  # alias
        assert result == expected

        # Log base 2
        result = resolve_n_bins_parameter("log2", data_shape=(1024, 2))
        assert result == 10  # log2(1024) = 10

        # Log base 10
        result = resolve_n_bins_parameter("log10", data_shape=(1000, 2))
        assert result == 3  # log10(1000) = 3

    def test_resolve_string_bins_sturges(self) -> None:
        """Test resolving Sturges' rule."""
        # Sturges: 1 + log2(n)
        result = resolve_n_bins_parameter("sturges", data_shape=(100, 2))
        expected = int(np.ceil(1 + np.log2(100)))  # ~7.6 -> 8
        assert result == expected

    def test_resolve_string_case_insensitive(self) -> None:
        """Test that string specifications are case-insensitive."""
        data_shape = (100, 2)

        # Different cases should give same result
        lower = resolve_n_bins_parameter("sqrt", data_shape)
        upper = resolve_n_bins_parameter("SQRT", data_shape)
        mixed = resolve_n_bins_parameter("Sqrt", data_shape)

        assert lower == upper == mixed == 10

    def test_resolve_string_with_whitespace(self) -> None:
        """Test string specifications with whitespace."""
        result = resolve_n_bins_parameter("  sqrt  ", data_shape=(100, 2))
        assert result == 10

    def test_resolve_string_without_data_shape(self) -> None:
        """Test string specifications require data_shape."""
        with pytest.raises(ConfigurationError, match="String specification.*requires data"):
            resolve_n_bins_parameter("sqrt")

        with pytest.raises(ConfigurationError, match="String specification.*requires data"):
            resolve_n_bins_parameter("log", data_shape=None)

    def test_resolve_invalid_data_shape(self) -> None:
        """Test invalid data_shape formats."""
        # Not a tuple
        with pytest.raises(ConfigurationError, match="Invalid data_shape format"):
            resolve_n_bins_parameter("sqrt", data_shape=[100, 2])  # type: ignore[arg-type]

        # Too short tuple
        with pytest.raises(ConfigurationError, match="Invalid data_shape format"):
            resolve_n_bins_parameter("sqrt", data_shape=(100,))

        # Zero samples
        with pytest.raises(ConfigurationError, match="Invalid number of samples"):
            resolve_n_bins_parameter("sqrt", data_shape=(0, 2))

        # Negative samples
        with pytest.raises(ConfigurationError, match="Invalid number of samples"):
            resolve_n_bins_parameter("sqrt", data_shape=(-10, 2))

    def test_resolve_invalid_string_spec(self) -> None:
        """Test invalid string specifications."""
        with pytest.raises(ConfigurationError, match="Unrecognized.*specification"):
            resolve_n_bins_parameter("invalid", data_shape=(100, 2))

        with pytest.raises(ConfigurationError, match="Unrecognized.*specification"):
            resolve_n_bins_parameter("unknown", data_shape=(100, 2))

    def test_resolve_invalid_type(self) -> None:
        """Test invalid parameter types."""
        # Float
        with pytest.raises(ConfigurationError, match="must be an integer or string"):
            resolve_n_bins_parameter(3.14)  # type: ignore[arg-type]

        # Boolean
        with pytest.raises(ConfigurationError, match="must be an integer or string"):
            resolve_n_bins_parameter(True)  # type: ignore[arg-type]

        # None
        with pytest.raises(ConfigurationError, match="must be an integer or string"):
            resolve_n_bins_parameter(None)  # type: ignore[arg-type]

    def test_resolve_minimum_value_enforcement(self) -> None:
        """Test that minimum of 1 bin is enforced."""
        # Very small dataset might resolve to 0, should be clamped to 1
        result = resolve_n_bins_parameter("sqrt", data_shape=(1, 2))
        assert result >= 1

        result = resolve_n_bins_parameter("log", data_shape=(1, 2))
        assert result >= 1

    def test_resolve_custom_param_name(self) -> None:
        """Test custom parameter name in error messages."""
        with pytest.raises(ConfigurationError, match="n_components must be a positive integer"):
            resolve_n_bins_parameter(-1, param_name="n_components")

        with pytest.raises(ConfigurationError, match="max_bins.*requires data"):
            resolve_n_bins_parameter("sqrt", param_name="max_bins")

    def test_resolve_math_errors(self) -> None:
        """Test handling of math errors during computation."""
        # This would be difficult to trigger naturally, but we can test the error handling
        # by using edge cases that might cause overflow
        try:
            # Extremely large data might cause overflow in some implementations
            result = resolve_n_bins_parameter("log", data_shape=(10**100, 2))
            assert isinstance(result, int) and result >= 1
        except ConfigurationError:
            # If it fails due to overflow, that's also acceptable behavior
            pass


class TestResolveStringParameter:
    """Test resolve_string_parameter function."""

    def test_resolve_valid_strings(self) -> None:
        """Test resolving valid string parameters."""
        options = {"auto": None, "sqrt": "sqrt_method", "log": "log_method"}

        assert resolve_string_parameter("auto", options, "param") is None
        assert resolve_string_parameter("sqrt", options, "param") == "sqrt_method"
        assert resolve_string_parameter("log", options, "param") == "log_method"

    def test_resolve_invalid_strings(self) -> None:
        """Test invalid string specifications."""
        options = {"auto": None, "sqrt": "sqrt_method"}

        with pytest.raises(ConfigurationError, match='Invalid param specification: "invalid"'):
            resolve_string_parameter("invalid", options, "param")

    def test_resolve_passthrough_enabled(self) -> None:
        """Test passthrough behavior when enabled."""
        options = {"auto": None}

        # Non-string values should pass through unchanged
        assert resolve_string_parameter(10, options, "param", allow_passthrough=True) == 10
        assert resolve_string_parameter(3.14, options, "param", allow_passthrough=True) == 3.14
        assert resolve_string_parameter(None, options, "param", allow_passthrough=True) is None
        assert resolve_string_parameter([1, 2], options, "param", allow_passthrough=True) == [1, 2]

    def test_resolve_passthrough_disabled(self) -> None:
        """Test passthrough behavior when disabled."""
        options = {"auto": None}

        with pytest.raises(ConfigurationError, match="param must be one of"):
            resolve_string_parameter(10, options, "param", allow_passthrough=False)

        with pytest.raises(ConfigurationError, match="param must be one of"):
            resolve_string_parameter(None, options, "param", allow_passthrough=False)

    def test_resolve_empty_options(self) -> None:
        """Test behavior with empty options dict."""
        options = {}

        # Any string should be invalid
        with pytest.raises(ConfigurationError):
            resolve_string_parameter("any", options, "param")

        # Passthrough should still work
        assert resolve_string_parameter(42, options, "param", allow_passthrough=True) == 42

    def test_resolve_case_sensitive(self) -> None:
        """Test that string matching is case-sensitive."""
        options = {"auto": None, "AUTO": "upper"}

        assert resolve_string_parameter("auto", options, "param") is None
        assert resolve_string_parameter("AUTO", options, "param") == "upper"

        # Different case should fail
        with pytest.raises(ConfigurationError):
            resolve_string_parameter("Auto", options, "param")

    def test_resolve_error_suggestions(self) -> None:
        """Test that error messages include helpful suggestions."""
        options = {"opt1": "val1", "opt2": "val2"}

        try:
            resolve_string_parameter("invalid", options, "test_param")
        except ConfigurationError as e:
            error_str = str(e)
            assert "opt1" in error_str
            assert "opt2" in error_str
            assert "test_param" in error_str


class TestValidateNumericParameter:
    """Test validate_numeric_parameter function."""

    def test_validate_integers(self) -> None:
        """Test integer validation."""
        # Valid integers
        assert validate_numeric_parameter(1, "param", integer_only=True) == 1
        assert validate_numeric_parameter(0, "param", integer_only=True) == 0
        assert validate_numeric_parameter(-5, "param", integer_only=True) == -5

    def test_validate_floats(self) -> None:
        """Test float validation."""
        # Valid floats
        assert validate_numeric_parameter(1.5, "param") == 1.5
        assert validate_numeric_parameter(0.0, "param") == 0.0
        assert validate_numeric_parameter(-3.14, "param") == -3.14

    def test_validate_integers_allow_floats(self) -> None:
        """Test that integers are allowed when not requiring integer_only."""
        assert validate_numeric_parameter(42, "param") == 42

    def test_validate_invalid_types(self) -> None:
        """Test invalid types are rejected."""
        # String
        with pytest.raises(ConfigurationError, match="param must be numeric"):
            validate_numeric_parameter("42", "param")

        # Boolean (explicitly excluded)
        with pytest.raises(ConfigurationError, match="param must be numeric"):
            validate_numeric_parameter(True, "param")

        with pytest.raises(ConfigurationError, match="param must be an integer"):
            validate_numeric_parameter(True, "param", integer_only=True)

        # List
        with pytest.raises(ConfigurationError, match="param must be numeric"):
            validate_numeric_parameter([1, 2, 3], "param")

    def test_validate_none_handling(self) -> None:
        """Test None value handling."""
        # None allowed
        assert validate_numeric_parameter(None, "param", allow_none=True) is None

        # None not allowed
        with pytest.raises(ConfigurationError, match="param cannot be None"):
            validate_numeric_parameter(None, "param", allow_none=False)

        # Default is None not allowed
        with pytest.raises(ConfigurationError, match="param cannot be None"):
            validate_numeric_parameter(None, "param")

    def test_validate_integer_only_rejects_floats(self) -> None:
        """Test integer_only parameter rejects floats."""
        with pytest.raises(ConfigurationError, match="param must be an integer"):
            validate_numeric_parameter(3.14, "param", integer_only=True)

        with pytest.raises(ConfigurationError, match="param must be an integer"):
            validate_numeric_parameter(1.0, "param", integer_only=True)

    def test_validate_min_value_constraint(self) -> None:
        """Test minimum value constraint."""
        # Valid values
        assert validate_numeric_parameter(5, "param", min_value=0) == 5
        assert validate_numeric_parameter(0, "param", min_value=0) == 0
        assert validate_numeric_parameter(0.1, "param", min_value=0.0) == 0.1

        # Invalid values
        with pytest.raises(ConfigurationError, match="param must be >= 0"):
            validate_numeric_parameter(-1, "param", min_value=0)

        with pytest.raises(ConfigurationError, match="param must be >= 1.5"):
            validate_numeric_parameter(1.0, "param", min_value=1.5)

    def test_validate_max_value_constraint(self) -> None:
        """Test maximum value constraint."""
        # Valid values
        assert validate_numeric_parameter(5, "param", max_value=10) == 5
        assert validate_numeric_parameter(10, "param", max_value=10) == 10
        assert validate_numeric_parameter(0.5, "param", max_value=1.0) == 0.5

        # Invalid values
        with pytest.raises(ConfigurationError, match="param must be <= 10"):
            validate_numeric_parameter(11, "param", max_value=10)

        with pytest.raises(ConfigurationError, match="param must be <= 1.0"):
            validate_numeric_parameter(1.5, "param", max_value=1.0)

    def test_validate_both_constraints(self) -> None:
        """Test both min and max constraints together."""
        # Valid values
        assert validate_numeric_parameter(5, "param", min_value=0, max_value=10) == 5

        # Below minimum
        with pytest.raises(ConfigurationError, match="param must be >= 0"):
            validate_numeric_parameter(-1, "param", min_value=0, max_value=10)

        # Above maximum
        with pytest.raises(ConfigurationError, match="param must be <= 10"):
            validate_numeric_parameter(11, "param", min_value=0, max_value=10)

    def test_validate_no_constraints(self) -> None:
        """Test validation without constraints."""
        # Should accept any numeric value
        assert validate_numeric_parameter(-100, "param") == -100
        assert validate_numeric_parameter(0, "param") == 0
        assert validate_numeric_parameter(100.5, "param") == 100.5

    def test_validate_custom_param_name(self) -> None:
        """Test custom parameter names in error messages."""
        with pytest.raises(ConfigurationError, match="alpha must be numeric"):
            validate_numeric_parameter("invalid", "alpha")

        with pytest.raises(ConfigurationError, match="n_estimators must be an integer"):
            validate_numeric_parameter(3.14, "n_estimators", integer_only=True)

    def test_validate_edge_case_values(self) -> None:
        """Test edge case numeric values."""
        # Very large numbers
        large_val = 1e10
        assert validate_numeric_parameter(large_val, "param") == large_val

        # Very small numbers
        small_val = 1e-10
        assert validate_numeric_parameter(small_val, "param") == small_val

        # Zero
        assert validate_numeric_parameter(0, "param") == 0
        assert validate_numeric_parameter(0.0, "param") == 0.0


class TestValidateBinNumberParameter:
    """Test validate_bin_number_parameter function."""

    def test_validate_valid_integers(self) -> None:
        """Test validation of valid integer values."""
        # Should not raise exceptions
        validate_bin_number_parameter(1)
        validate_bin_number_parameter(10)
        validate_bin_number_parameter(100)

    def test_validate_invalid_integers(self) -> None:
        """Test validation of invalid integer values."""
        # Zero
        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            validate_bin_number_parameter(0)

        # Negative
        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            validate_bin_number_parameter(-5)

    def test_validate_valid_strings(self) -> None:
        """Test validation of valid string specifications."""
        # Default valid strings
        valid_specs = ["sqrt", "log", "ln", "log2", "log10", "sturges"]

        for spec in valid_specs:
            # Should not raise exceptions
            validate_bin_number_parameter(spec)

    def test_validate_string_case_insensitive(self) -> None:
        """Test string validation is case-insensitive."""
        # Different cases should all be valid
        validate_bin_number_parameter("sqrt")
        validate_bin_number_parameter("SQRT")
        validate_bin_number_parameter("Sqrt")
        validate_bin_number_parameter("SqRt")

    def test_validate_string_with_whitespace(self) -> None:
        """Test strings with whitespace are handled."""
        validate_bin_number_parameter("  sqrt  ")
        validate_bin_number_parameter("\tlog\n")

    def test_validate_invalid_strings(self) -> None:
        """Test validation of invalid string specifications."""
        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            validate_bin_number_parameter("invalid")

        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            validate_bin_number_parameter("unknown")

    def test_validate_invalid_types(self) -> None:
        """Test validation of invalid types."""
        # Float
        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            validate_bin_number_parameter(3.14)  # type: ignore[arg-type]

        # None
        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            validate_bin_number_parameter(None)  # type: ignore[arg-type]

        # Boolean
        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            validate_bin_number_parameter(True)  # type: ignore[arg-type]

    def test_validate_custom_param_name(self) -> None:
        """Test validation with custom parameter name."""
        with pytest.raises(ConfigurationError, match="n_components must be a positive integer"):
            validate_bin_number_parameter(0, param_name="n_components")

        with pytest.raises(ConfigurationError, match="max_bins must be a positive integer"):
            validate_bin_number_parameter("invalid", param_name="max_bins")

    def test_validate_custom_valid_strings(self) -> None:
        """Test validation with custom valid strings."""
        custom_strings = {"auto", "best", "optimal"}

        # Valid custom strings
        validate_bin_number_parameter("auto", valid_strings=custom_strings)
        validate_bin_number_parameter("best", valid_strings=custom_strings)

        # Default strings should now be invalid
        with pytest.raises(ConfigurationError):
            validate_bin_number_parameter("sqrt", valid_strings=custom_strings)

    def test_validate_empty_valid_strings(self) -> None:
        """Test validation with empty valid strings set."""
        empty_strings: set[str] = set()

        # Any string should be invalid
        with pytest.raises(ConfigurationError):
            validate_bin_number_parameter("sqrt", valid_strings=empty_strings)

        # Integers should still be valid
        validate_bin_number_parameter(10, valid_strings=empty_strings)

    def test_validate_error_suggestions(self) -> None:
        """Test that error messages include helpful suggestions."""
        try:
            validate_bin_number_parameter("invalid")
        except ConfigurationError as e:
            error_str = str(e)
            # Should include suggestions about valid strings and integers
            assert "sqrt" in error_str or "Valid string options" in error_str


class TestValidateBinNumberForCalculation:
    """Test validate_bin_number_for_calculation function."""

    def test_validate_valid_integers(self) -> None:
        """Test validation passes for valid integers."""
        # Should not raise exceptions
        validate_bin_number_for_calculation(1)
        validate_bin_number_for_calculation(10)
        validate_bin_number_for_calculation(100)

    def test_validate_invalid_integers(self) -> None:
        """Test validation fails for invalid integers with exact error format."""
        # Zero - should raise ValueError with exact format
        with pytest.raises(ValueError, match="n_bins must be >= 1, got 0"):
            validate_bin_number_for_calculation(0)

        # Negative - should raise ValueError with exact format
        with pytest.raises(ValueError, match="n_bins must be >= 1, got -5"):
            validate_bin_number_for_calculation(-5)

    def test_validate_strings_pass_through(self) -> None:
        """Test that strings pass through without validation."""
        # All strings should pass through without errors
        validate_bin_number_for_calculation("sqrt")
        validate_bin_number_for_calculation("log")
        validate_bin_number_for_calculation("invalid")  # Even invalid strings pass

    def test_validate_other_types_pass_through(self) -> None:
        """Test that non-integer types pass through without validation."""
        # Floats should pass through
        validate_bin_number_for_calculation(3.14)  # type: ignore[arg-type]

        # Other types should pass through
        validate_bin_number_for_calculation(None)  # type: ignore[arg-type]
        validate_bin_number_for_calculation([1, 2, 3])  # type: ignore[arg-type]

    def test_validate_custom_param_name(self) -> None:
        """Test custom parameter name in error messages."""
        with pytest.raises(ValueError, match="n_components must be >= 1, got 0"):
            validate_bin_number_for_calculation(0, param_name="n_components")

        with pytest.raises(ValueError, match="max_bins must be >= 1, got -1"):
            validate_bin_number_for_calculation(-1, param_name="max_bins")

    def test_validate_exact_error_message_format(self) -> None:
        """Test exact error message format for backward compatibility."""
        # This test ensures the exact format expected by existing tests
        try:
            validate_bin_number_for_calculation(-3)
        except ValueError as e:
            assert str(e) == "n_bins must be >= 1, got -3"

        try:
            validate_bin_number_for_calculation(0, param_name="test_param")
        except ValueError as e:
            assert str(e) == "test_param must be >= 1, got 0"


class TestValidateTreeParams:
    """Test validate_tree_params function."""

    def test_validate_empty_params(self) -> None:
        """Test validation of empty parameters."""
        # Empty dict
        assert validate_tree_params("classification", {}) == {}

        # None
        assert validate_tree_params("regression", None) == {}  # type: ignore[arg-type]

    def test_validate_valid_params(self) -> None:
        """Test validation of valid tree parameters."""
        valid_params = {
            "max_depth": 5,
            "min_samples_split": 10,
            "min_samples_leaf": 3,
            "max_features": "sqrt",
            "random_state": 42,
            "criterion": "gini",
        }

        result = validate_tree_params("classification", valid_params)
        assert result == valid_params

    def test_validate_invalid_params(self) -> None:
        """Test validation fails for invalid parameter names."""
        invalid_params = {
            "invalid_param": "value",
            "unknown_setting": 42,
        }

        with pytest.raises(ConfigurationError, match="Invalid tree parameters"):
            validate_tree_params("classification", invalid_params)

    def test_validate_max_depth_param(self) -> None:
        """Test validation of max_depth parameter."""
        # Valid values
        assert validate_tree_params("classification", {"max_depth": 5}) == {"max_depth": 5}
        assert validate_tree_params("classification", {"max_depth": None}) == {"max_depth": None}

        # Invalid values
        with pytest.raises(ConfigurationError, match="max_depth must be a positive integer"):
            validate_tree_params("classification", {"max_depth": 0})

        with pytest.raises(ConfigurationError, match="max_depth must be a positive integer"):
            validate_tree_params("classification", {"max_depth": -1})

        with pytest.raises(ConfigurationError, match="max_depth must be a positive integer"):
            validate_tree_params("classification", {"max_depth": 3.5})

    def test_validate_min_samples_split_param(self) -> None:
        """Test validation of min_samples_split parameter."""
        # Valid values
        assert validate_tree_params("classification", {"min_samples_split": 2}) == {
            "min_samples_split": 2
        }
        assert validate_tree_params("classification", {"min_samples_split": 10}) == {
            "min_samples_split": 10
        }

        # Invalid values
        with pytest.raises(ConfigurationError, match="min_samples_split must be an integer >= 2"):
            validate_tree_params("classification", {"min_samples_split": 1})

        with pytest.raises(ConfigurationError, match="min_samples_split must be an integer >= 2"):
            validate_tree_params("classification", {"min_samples_split": 0})

        with pytest.raises(ConfigurationError, match="min_samples_split must be an integer >= 2"):
            validate_tree_params("classification", {"min_samples_split": 2.5})

    def test_validate_min_samples_leaf_param(self) -> None:
        """Test validation of min_samples_leaf parameter."""
        # Valid values
        assert validate_tree_params("classification", {"min_samples_leaf": 1}) == {
            "min_samples_leaf": 1
        }
        assert validate_tree_params("classification", {"min_samples_leaf": 5}) == {
            "min_samples_leaf": 5
        }

        # Invalid values
        with pytest.raises(ConfigurationError, match="min_samples_leaf must be a positive integer"):
            validate_tree_params("classification", {"min_samples_leaf": 0})

        with pytest.raises(ConfigurationError, match="min_samples_leaf must be a positive integer"):
            validate_tree_params("classification", {"min_samples_leaf": -1})

        with pytest.raises(ConfigurationError, match="min_samples_leaf must be a positive integer"):
            validate_tree_params("classification", {"min_samples_leaf": 1.5})

    def test_validate_multiple_params(self) -> None:
        """Test validation of multiple parameters together."""
        params = {
            "max_depth": 3,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
        }

        result = validate_tree_params("classification", params)
        assert result == params

    def test_validate_mixed_valid_invalid_params(self) -> None:
        """Test validation with mix of valid and invalid parameters."""
        # Invalid parameter names should be caught first
        params = {
            "max_depth": 5,  # Valid
            "invalid_param": "value",  # Invalid name
            "min_samples_split": 1,  # Invalid value but valid name
        }

        with pytest.raises(ConfigurationError, match="Invalid tree parameters"):
            validate_tree_params("classification", params)

    def test_validate_all_valid_param_names(self) -> None:
        """Test that all documented valid parameter names are accepted."""
        valid_param_names = [
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "random_state",
            "max_leaf_nodes",
            "min_impurity_decrease",
            "class_weight",
            "ccp_alpha",
            "criterion",
        ]

        # Each parameter should be accepted individually
        for param_name in valid_param_names:
            # Use safe values that won't trigger value validation
            safe_values = {
                "max_depth": 5,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "random_state": 42,
                "max_leaf_nodes": 10,
                "min_impurity_decrease": 0.0,
                "class_weight": "balanced",
                "ccp_alpha": 0.0,
                "criterion": "gini",
            }

            params = {param_name: safe_values[param_name]}
            result = validate_tree_params("classification", params)
            assert result == params

    def test_validate_task_type_parameter(self) -> None:
        """Test that task_type parameter is accepted but ignored."""
        params = {"max_depth": 3}

        # Different task types should give same result
        result1 = validate_tree_params("classification", params)
        result2 = validate_tree_params("regression", params)
        result3 = validate_tree_params("unknown", params)

        assert result1 == result2 == result3 == params

    def test_validate_suggestions_in_error_messages(self) -> None:
        """Test that error messages contain helpful suggestions."""
        try:
            validate_tree_params("classification", {"invalid_param": "value"})
        except ConfigurationError as e:
            error_str = str(e)
            assert "max_depth" in error_str  # Should suggest valid params
            assert "min_samples_split" in error_str

        try:
            validate_tree_params("classification", {"max_depth": 0})
        except ConfigurationError as e:
            error_str = str(e)
            assert "positive integer" in error_str or "None" in error_str


class TestValidationIntegration:
    """Test integration scenarios and edge cases across validation functions."""

    def test_validation_consistency(self) -> None:
        """Test that validation functions are consistent with each other."""
        # Same parameter should be validated consistently
        test_values = [1, 5, 10, 100]

        for value in test_values:
            # All these should pass or fail consistently
            validate_bin_number_parameter(value)
            validate_bin_number_for_calculation(value)
            validate_numeric_parameter(value, "test", min_value=1, integer_only=True)

    def test_error_message_consistency(self) -> None:
        """Test that error messages are consistent and helpful."""
        # Test that similar errors have consistent format
        validation_funcs = [
            lambda: validate_numeric_parameter("string", "param"),
            lambda: resolve_string_parameter("invalid", {}, "param"),
            lambda: validate_bin_number_parameter(-1),
        ]

        for func in validation_funcs:
            try:
                func()
            except (ConfigurationError, ValueError) as e:
                # All should be proper exception types with messages
                assert len(str(e)) > 0
                assert isinstance(e, (ConfigurationError, ValueError))

    def test_parameter_name_propagation(self) -> None:
        """Test that parameter names are properly used in error messages."""
        custom_name = "custom_parameter_name"

        # All functions that accept param_name should use it
        error_funcs = [
            lambda: validate_numeric_parameter("invalid", custom_name),
            lambda: resolve_n_bins_parameter(-1, param_name=custom_name),
            lambda: validate_bin_number_parameter(0, param_name=custom_name),
            lambda: validate_bin_number_for_calculation(0, param_name=custom_name),
        ]

        for func in error_funcs:
            try:
                func()
            except (ConfigurationError, ValueError) as e:
                assert custom_name in str(e)

    def test_edge_case_handling(self) -> None:
        """Test edge cases that might occur in practice."""
        # Test very small positive numbers
        tiny_value = 1e-100
        validate_numeric_parameter(tiny_value, "param", min_value=0.0)

        # Test very large numbers
        huge_value = 1e100
        validate_numeric_parameter(huge_value, "param", max_value=1e101)

        # Test boundary values
        validate_numeric_parameter(0, "param", min_value=0, max_value=0)
        validate_bin_number_parameter(1)  # Minimum valid
        validate_bin_number_for_calculation(1)  # Minimum valid
