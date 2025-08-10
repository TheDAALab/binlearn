"""
Tests for parameter validation utilities.
"""

from unittest.mock import Mock

import pytest

from binlearn.utils._errors import ConfigurationError
from binlearn.utils._parameter_validation import (
    create_configuration_error,
    validate_common_parameters,
    validate_positive_integer,
    validate_positive_number,
    validate_random_state,
    validate_range_parameter,
)


class TestValidateCommonParameters:
    """Test the validate_common_parameters function."""

    def test_type_validation_success(self):
        """Test successful type validation."""
        obj = Mock()
        obj.test_param = 5

        param_specs = {"test_param": {"type": int, "example": 10}}

        # Should not raise any exception
        validate_common_parameters(obj, param_specs)

    def test_type_validation_failure(self):
        """Test type validation failure."""
        obj = Mock()
        obj.test_param = "string"

        param_specs = {"test_param": {"type": int, "example": 10}}

        with pytest.raises(ConfigurationError, match="must be of type int, got str"):
            validate_common_parameters(obj, param_specs)

    def test_range_validation_success(self):
        """Test successful range validation."""
        obj = Mock()
        obj.test_param = 5

        param_specs = {"test_param": {"min": 0, "max": 10, "example": 5}}

        # Should not raise any exception
        validate_common_parameters(obj, param_specs)

    def test_range_validation_below_min(self):
        """Test range validation failure - below minimum."""
        obj = Mock()
        obj.test_param = -1

        param_specs = {"test_param": {"min": 0, "max": 10, "example": 5}}

        with pytest.raises(ConfigurationError, match="must be in range \\[0, 10\\], got -1"):
            validate_common_parameters(obj, param_specs)

    def test_range_validation_above_max(self):
        """Test range validation failure - above maximum."""
        obj = Mock()
        obj.test_param = 15

        param_specs = {"test_param": {"min": 0, "max": 10, "example": 5}}

        with pytest.raises(ConfigurationError, match="must be in range \\[0, 10\\], got 15"):
            validate_common_parameters(obj, param_specs)

    def test_custom_validator_success(self):
        """Test successful custom validation."""
        obj = Mock()
        obj.test_param = "valid"

        def custom_validator(x):
            return x == "valid"

        param_specs = {
            "test_param": {
                "validator": custom_validator,
            }
        }

        # Should not raise any exception
        validate_common_parameters(obj, param_specs)

    def test_custom_validator_failure(self):
        """Test custom validation failure."""
        obj = Mock()
        obj.test_param = "invalid"

        def custom_validator(x):
            return x == "valid"

        param_specs = {
            "test_param": {
                "validator": custom_validator,
                "validator_message": "must be 'valid'",
                "suggestions": ["Use 'valid' instead"],
            }
        }

        with pytest.raises(ConfigurationError, match="must be 'valid'"):
            validate_common_parameters(obj, param_specs)

    def test_none_values_skipped(self):
        """Test that None values are skipped."""
        obj = Mock()
        obj.test_param = None

        param_specs = {
            "test_param": {
                "type": int,
                "min": 0,
                "validator": lambda x: False,  # Would fail if called
            }
        }

        # Should not raise any exception
        validate_common_parameters(obj, param_specs)

    def test_missing_attribute_skipped(self):
        """Test that missing attributes are skipped."""
        obj = Mock()
        # Mock doesn't have the specific attribute, but Mock returns Mock for any attribute access
        # So we need to explicitly test when getattr returns None
        obj.test_param = None  # Explicitly set to None

        param_specs = {
            "test_param": {"type": int, "validator": lambda x: False}  # Would fail if called
        }

        # Should not raise any exception since None values are skipped
        validate_common_parameters(obj, param_specs)

    def test_range_validation_non_numeric(self):
        """Test that range validation is skipped for non-numeric types."""
        obj = Mock()
        obj.test_param = "string"

        param_specs = {
            "test_param": {
                "min": 0,
                "max": 10,
            }
        }

        # Should not raise any exception (range validation skipped for strings)
        validate_common_parameters(obj, param_specs)

    def test_custom_validator_default_message(self):
        """Test custom validator with default error message."""
        obj = Mock()
        obj.test_param = "invalid"

        param_specs = {
            "test_param": {
                "validator": lambda x: False,
                # No validator_message provided
            }
        }

        with pytest.raises(ConfigurationError, match="failed custom validation"):
            validate_common_parameters(obj, param_specs)


class TestValidateRangeParameter:
    """Test the validate_range_parameter function."""

    def test_valid_tuple(self):
        """Test with valid tuple range."""
        value = (0, 10)
        # Should not raise any exception
        validate_range_parameter(value, "test_param")

    def test_valid_list(self):
        """Test with valid list range."""
        value = [0, 10]
        # Should not raise any exception
        validate_range_parameter(value, "test_param")

    def test_none_allowed(self):
        """Test None value when allowed."""
        # Should not raise any exception
        validate_range_parameter(None, "test_param", allow_none=True)

    def test_none_not_allowed(self):
        """Test None value when not allowed."""
        with pytest.raises(ConfigurationError, match="must be a tuple/list of two numbers"):
            validate_range_parameter(None, "test_param", allow_none=False)

    def test_wrong_length(self):
        """Test with wrong length tuple/list."""
        value = (0, 10, 20)

        with pytest.raises(ConfigurationError, match="must be a tuple/list of two numbers"):
            validate_range_parameter(value, "test_param")

    def test_single_value(self):
        """Test with single value."""
        value = (10,)

        with pytest.raises(ConfigurationError, match="must be a tuple/list of two numbers"):
            validate_range_parameter(value, "test_param")

    def test_non_numeric_values(self):
        """Test with non-numeric values."""
        value = ("a", "b")

        with pytest.raises(ConfigurationError, match="values must be numbers"):
            validate_range_parameter(value, "test_param")

    def test_mixed_numeric_non_numeric(self):
        """Test with mixed numeric and non-numeric values."""
        value = (0, "b")

        with pytest.raises(ConfigurationError, match="values must be numbers"):
            validate_range_parameter(value, "test_param")

    def test_min_equals_max(self):
        """Test when minimum equals maximum."""
        value = (5, 5)

        with pytest.raises(
            ConfigurationError, match="minimum \\(5\\) must be less than maximum \\(5\\)"
        ):
            validate_range_parameter(value, "test_param")

    def test_min_greater_than_max(self):
        """Test when minimum is greater than maximum."""
        value = (10, 5)

        with pytest.raises(
            ConfigurationError, match="minimum \\(10\\) must be less than maximum \\(5\\)"
        ):
            validate_range_parameter(value, "test_param")

    def test_wrong_type(self):
        """Test with wrong type (not tuple/list)."""
        value = "not a range"

        with pytest.raises(ConfigurationError, match="must be a tuple/list of two numbers"):
            validate_range_parameter(value, "test_param")

    def test_float_values(self):
        """Test with float values."""
        value = (0.5, 10.7)
        # Should not raise any exception
        validate_range_parameter(value, "test_param")


class TestValidatePositiveNumber:
    """Test the validate_positive_number function."""

    def test_valid_positive_int(self):
        """Test with valid positive integer."""
        # Should not raise any exception
        validate_positive_number(5, "test_param")

    def test_valid_positive_float(self):
        """Test with valid positive float."""
        # Should not raise any exception
        validate_positive_number(5.5, "test_param")

    def test_zero_allowed(self):
        """Test zero when allowed."""
        # Should not raise any exception
        validate_positive_number(0, "test_param", allow_zero=True)

    def test_zero_not_allowed(self):
        """Test zero when not allowed."""
        with pytest.raises(ConfigurationError, match="test_param must be > 0.0, got 0"):
            validate_positive_number(0, "test_param", allow_zero=False)

    def test_negative_number(self):
        """Test with negative number."""
        with pytest.raises(ConfigurationError, match="test_param must be > 0.0, got -5"):
            validate_positive_number(-5, "test_param")

    def test_negative_number_allow_zero(self):
        """Test negative number when zero is allowed."""
        with pytest.raises(ConfigurationError, match="test_param must be >= 0.0, got -3"):
            validate_positive_number(-3, "test_param", allow_zero=True)

    def test_non_numeric_type(self):
        """Test with non-numeric type."""
        with pytest.raises(ConfigurationError, match="must be a number, got str"):
            validate_positive_number("string", "test_param")

    def test_none_type(self):
        """Test with None."""
        with pytest.raises(ConfigurationError, match="must be a number, got NoneType"):
            validate_positive_number(None, "test_param")


class TestValidatePositiveInteger:
    """Test the validate_positive_integer function."""

    def test_valid_positive_integer(self):
        """Test with valid positive integer."""
        # Should not raise any exception
        validate_positive_integer(5, "test_param")

    def test_zero_value(self):
        """Test with zero value."""
        with pytest.raises(ConfigurationError, match="must be a positive integer"):
            validate_positive_integer(0, "test_param")

    def test_negative_integer(self):
        """Test with negative integer."""
        with pytest.raises(ConfigurationError, match="must be a positive integer"):
            validate_positive_integer(-5, "test_param")

    def test_float_type(self):
        """Test with float type."""
        with pytest.raises(ConfigurationError, match="must be a positive integer"):
            validate_positive_integer(5.5, "test_param")

    def test_string_type(self):
        """Test with string type."""
        with pytest.raises(ConfigurationError, match="must be a positive integer"):
            validate_positive_integer("5", "test_param")


class TestValidateRandomState:
    """Test the validate_random_state function."""

    def test_valid_positive_integer(self):
        """Test with valid positive integer."""
        # Should not raise any exception
        validate_random_state(42, "random_state")

    def test_zero_value(self):
        """Test with zero value."""
        # Should not raise any exception (zero is valid for random state)
        validate_random_state(0, "random_state")

    def test_none_value(self):
        """Test with None value."""
        # Should not raise any exception
        validate_random_state(None, "random_state")

    def test_negative_integer(self):
        """Test with negative integer."""
        with pytest.raises(ConfigurationError, match="must be non-negative, got -1"):
            validate_random_state(-1, "random_state")

    def test_float_type(self):
        """Test with float type."""
        with pytest.raises(ConfigurationError, match="must be an integer or None, got float"):
            validate_random_state(42.0, "random_state")

    def test_string_type(self):
        """Test with string type."""
        with pytest.raises(ConfigurationError, match="must be an integer or None, got str"):
            validate_random_state("42", "random_state")

    def test_default_param_name(self):
        """Test with default parameter name."""
        with pytest.raises(ConfigurationError, match="random_state must be"):
            validate_random_state("invalid")  # Uses default param_name="random_state"

    def test_custom_param_name(self):
        """Test with custom parameter name."""
        with pytest.raises(ConfigurationError, match="seed must be"):
            validate_random_state("invalid", "seed")


class TestCreateConfigurationError:
    """Test the create_configuration_error function."""

    def test_basic_functionality(self):
        """Test basic error creation."""
        param_name = "test_param"
        issue = "is invalid"
        suggestions = ["Try a different value", "Check documentation"]

        error = create_configuration_error(param_name, issue, suggestions)

        assert isinstance(error, ConfigurationError)
        # ConfigurationError includes suggestions in the string representation
        error_str = str(error)
        assert "test_param is invalid" in error_str
        assert "Try a different value" in error_str
        assert "Check documentation" in error_str
        assert error.suggestions == suggestions

    def test_empty_suggestions(self):
        """Test with empty suggestions list."""
        error = create_configuration_error("param", "is wrong", [])

        assert error.suggestions == []

    def test_single_suggestion(self):
        """Test with single suggestion."""
        suggestions = ["Fix it"]
        error = create_configuration_error("param", "is bad", suggestions)

        assert error.suggestions == suggestions
