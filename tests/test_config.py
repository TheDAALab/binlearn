"""
Comprehensive test suite for the binning configuration system.

This module provides extensive testing for the configuration management
system in the binning package, including the BinningConfig dataclass,
ConfigManager class, configuration loading/saving, environment variable
handling, context management, and all configuration utilities.

The test suite aims for 100% code coverage by systematically testing
every configuration feature, edge case, and error condition.

Test Classes:
    TestBinningConfig: Tests for the BinningConfig dataclass including
        initialization, validation, and dictionary conversion.
    TestConfigManager: Tests for the ConfigManager singleton including
        loading, saving, environment variable handling, and state management.
    TestGlobalFunctions: Tests for global configuration functions like
        get_config, set_config, load_config, and reset_config.
    TestConfigContext: Tests for the ConfigContext context manager.
    TestConfigEdgeCases: Tests for edge cases and error conditions.
    TestConfigIntegration: Integration tests with binning transformers.
"""

# pylint: disable=protected-access

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from binlearn.config import (
    BinningConfig,
    ConfigContext,
    ConfigManager,
    _get_parameter_description,
    apply_config_defaults,
    get_config,
    get_config_schema,
    load_config,
    reset_config,
    set_config,
    validate_config_parameter,
)


def test_basic_config_import():
    """Basic test to ensure pytest discovers this module correctly.

    Verifies that the main configuration classes can be imported
    successfully and are available for testing.
    """
    assert BinningConfig is not None
    assert ConfigManager is not None


class TestBinningConfig:
    """Comprehensive test suite for the BinningConfig dataclass.

    This test class verifies all aspects of the BinningConfig dataclass
    including default initialization, dictionary conversion, validation,
    and configuration parameter handling.
    """

    def test_init_defaults(self):
        """Test initialization with default parameter values.

        Verifies that the BinningConfig dataclass initializes correctly
        with the expected default values for all configuration parameters.
        """
        config = BinningConfig()
        assert config.preserve_dataframe is False
        assert config.fit_jointly is False
        assert config.float_tolerance == 1e-10
        assert config.default_clip is True
        assert config.equal_width_default_bins == 5

    def test_from_dict_valid_keys(self):
        """Test configuration creation from dictionary with valid keys.

        Verifies that the from_dict method correctly creates a BinningConfig
        instance from a dictionary with valid configuration parameters.
        """
        config_dict = {"preserve_dataframe": True, "fit_jointly": True, "float_tolerance": 1e-8}
        config = BinningConfig.from_dict(config_dict)
        assert config.preserve_dataframe is True
        assert config.fit_jointly is True
        assert config.float_tolerance == 1e-8

    def test_from_dict_filters_invalid_keys(self):
        """Test from_dict filters out invalid keys."""
        config_dict = {
            "preserve_dataframe": True,
            "invalid_key": "should_be_ignored",
            "another_invalid": 123,
        }
        config = BinningConfig.from_dict(config_dict)
        assert config.preserve_dataframe is True
        assert not hasattr(config, "invalid_key")

    def test_to_dict(self):
        """Test to_dict conversion."""
        config = BinningConfig(preserve_dataframe=True, fit_jointly=True)
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["preserve_dataframe"] is True
        assert config_dict["fit_jointly"] is True

    def test_load_from_file(self):
        """Test loading from JSON file."""
        config_data = {"preserve_dataframe": True, "equal_width_default_bins": 10}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as file_handle:
            json.dump(config_data, file_handle)
            temp_path = file_handle.name

        try:
            config = BinningConfig.load_from_file(temp_path)
            assert config.preserve_dataframe is True
            assert config.equal_width_default_bins == 10
        finally:
            os.unlink(temp_path)

    def test_save_to_file(self):
        """Test saving to JSON file."""
        config = BinningConfig(preserve_dataframe=True, fit_jointly=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as file_handle:
            temp_path = file_handle.name

        try:
            config.save_to_file(temp_path)

            # Verify file contents
            with open(temp_path, encoding="utf-8") as file_handle:
                saved_data = json.load(file_handle)

            assert saved_data["preserve_dataframe"] is True
            assert saved_data["fit_jointly"] is True
        finally:
            os.unlink(temp_path)

    def test_update(self):
        """Test update method."""
        config = BinningConfig()
        config.update(preserve_dataframe=True, fit_jointly=True)
        assert config.preserve_dataframe is True
        assert config.fit_jointly is True

    def test_update_invalid_parameter(self):
        """Test update with invalid parameter raises error."""
        config = BinningConfig()

        with pytest.raises(ValueError, match="Unknown configuration parameter"):
            config.update(invalid_parameter="value")

    def test_update_validation(self):
        """Test update method with validation."""
        config = BinningConfig()

        # Test positive depth validation - negative value should raise error
        with pytest.raises(ValueError, match="must be positive"):
            config.update(supervised_default_max_depth=-1)

        # Test positive depth validation - positive value should work
        config.update(supervised_default_max_depth=5)  # Should pass validation
        assert config.supervised_default_max_depth == 5

        # Test float tolerance validation - negative should raise error
        with pytest.raises(ValueError, match="float_tolerance must be positive"):
            config.update(float_tolerance=-1.0)

        # Test float tolerance validation - positive should work
        config.update(float_tolerance=0.001)  # Should pass validation
        assert config.float_tolerance == 0.001

        # Test NEGATION: depth parameter with non-integer value (should skip depth
        # validation branch)
        # This covers the case where key matches pattern but isinstance(value, int) is False
        config.update(supervised_default_max_depth="3")  # String instead of int, skips validation
        assert config.supervised_default_max_depth == "3"

        # Test NEGATION: float_tolerance with non-numeric value (should skip tolerance validation
        # branch)
        # This covers the case where key == "float_tolerance" but isinstance(value, (int, float))
        # is False
        config.update(float_tolerance="0.001")  # String instead of number, skips validation
        assert config.float_tolerance == "0.001"

    def test_validate_strategy_parameter(self):
        """Test _validate_strategy_parameter method."""
        config = BinningConfig()

        # Valid values
        config._validate_strategy_parameter("equal_width_default_range_strategy", "min_max")
        config._validate_strategy_parameter("equal_width_default_range_strategy", "percentile")
        config._validate_strategy_parameter("equal_width_default_range_strategy", "std")

        # Invalid value
        with pytest.raises(ValueError, match="must be one of"):
            config._validate_strategy_parameter("equal_width_default_range_strategy", "invalid")

    def test_get_method_defaults(self):
        """Test get_method_defaults for different methods."""
        config = BinningConfig()

        # Equal width
        ew_defaults = config.get_method_defaults("equal_width")
        assert "n_bins" in ew_defaults
        assert "clip" in ew_defaults
        assert ew_defaults["n_bins"] == config.equal_width_default_bins

        # Supervised
        sup_defaults = config.get_method_defaults("supervised")
        assert "max_depth" in sup_defaults
        assert "min_samples_leaf" in sup_defaults

        # Singleton
        singleton_defaults = config.get_method_defaults("singleton")
        assert "max_unique_values" in singleton_defaults

        # Unknown method - should return base defaults
        unknown_defaults = config.get_method_defaults("UnknownMethod")
        assert "preserve_dataframe" in unknown_defaults
        assert "fit_jointly" in unknown_defaults


class TestConfigManager:
    """Test ConfigManager singleton."""

    def test_singleton_behavior(self):
        """Test that ConfigManager is a singleton."""
        manager1 = ConfigManager()
        manager2 = ConfigManager()
        assert manager1 is manager2

    def test_config_property(self):
        """Test config property access."""
        manager = ConfigManager()
        config = manager.config
        assert isinstance(config, BinningConfig)

    def test_update_config(self):
        """Test update_config method."""
        manager = ConfigManager()
        manager.update_config(preserve_dataframe=True)
        assert manager.config.preserve_dataframe is True
        manager.reset_to_defaults()  # Clean up

    def test_load_config_from_file(self):
        """Test load_config method."""
        config_data = {"preserve_dataframe": True}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        manager = None
        try:
            manager = ConfigManager()
            manager.load_config(temp_path)
            assert manager.config.preserve_dataframe is True
        finally:
            os.unlink(temp_path)
            if manager is not None:
                manager.reset_to_defaults()  # Clean up

    def test_reset_to_defaults(self):
        """Test reset_to_defaults method."""
        manager = ConfigManager()
        manager.update_config(preserve_dataframe=True)

        manager.reset_to_defaults()
        assert manager.config.preserve_dataframe is False  # Default value

    def test_load_from_env(self):
        """Test _load_from_env with environment variables."""
        env_vars = {
            "BINNING_PRESERVE_DATAFRAME": "true",
            "BINNING_FIT_JOINTLY": "false",
            "BINNING_FLOAT_TOLERANCE": "1e-8",
            "BINNING_EQUAL_WIDTH_BINS": "10",
        }

        with patch.dict(os.environ, env_vars):
            manager = ConfigManager()
            manager._load_from_env()

            assert manager.config.preserve_dataframe is True
            assert manager.config.fit_jointly is False
            assert manager.config.float_tolerance == 1e-8
            assert manager.config.equal_width_default_bins == 10

        manager.reset_to_defaults()  # Clean up

    def test_load_from_env_invalid_values(self):
        """Test _load_from_env with invalid environment values."""
        env_vars = {
            "BINNING_PRESERVE_DATAFRAME": "invalid_bool",
            "BINNING_FLOAT_TOLERANCE": "not_a_number",
        }

        with patch.dict(os.environ, env_vars):
            manager = ConfigManager()
            # Test with raise_on_config_errors=False
            manager.config.raise_on_config_errors = False
            manager._load_from_env()  # Should not raise

            # Test with raise_on_config_errors=True
            manager.config.raise_on_config_errors = True
            with pytest.raises(ValueError, match="Invalid environment variable"):
                manager._load_from_env()

        # Clean up after exiting the patch context
        manager.reset_to_defaults()

    def test_load_config_file_error(self):
        """Test error handling in load_config."""
        manager = ConfigManager()

        with pytest.raises(FileNotFoundError):
            manager.load_config("/nonexistent/file.json")


class TestGlobalFunctions:
    """Test global configuration functions."""

    def test_get_config(self):
        """Test get_config function."""
        config = get_config()
        assert isinstance(config, BinningConfig)

    def test_set_config(self):
        """Test set_config function."""
        set_config(preserve_dataframe=True)
        assert get_config().preserve_dataframe is True
        reset_config()  # Clean up

    def test_load_config_function(self):
        """Test load_config function."""
        config_data = {"preserve_dataframe": True}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            load_config(temp_path)
            assert get_config().preserve_dataframe is True
        finally:
            os.unlink(temp_path)
            reset_config()  # Clean up

    def test_reset_config_function(self):
        """Test reset_config function."""
        set_config(preserve_dataframe=True)
        reset_config()
        assert get_config().preserve_dataframe is False  # Default

    def test_apply_config_defaults(self):
        """Test apply_config_defaults function."""
        set_config(preserve_dataframe=True, fit_jointly=True)

        # Test with method name
        params = apply_config_defaults("equal_width")
        assert params["preserve_dataframe"] is True
        assert params["fit_jointly"] is True
        assert "n_bins" in params  # Method-specific default

        # Test with user_params preservation
        params = apply_config_defaults("equal_width", user_params={"preserve_dataframe": False})
        assert params["preserve_dataframe"] is False

        # Test with override parameters
        params = apply_config_defaults(
            "equal_width", user_params={"n_bins": 5}, n_bins=10
        )  # Override should win
        assert params["n_bins"] == 10

        # Test with None and empty user_params
        params = apply_config_defaults("equal_width", user_params=None)
        assert "n_bins" in params

        params = apply_config_defaults("equal_width", user_params={})
        assert "n_bins" in params

        reset_config()  # Clean up

    def test_validate_config_parameter(self):
        """Test validate_config_parameter function."""
        # Valid parameters
        assert validate_config_parameter("preserve_dataframe", True) is True
        assert validate_config_parameter("fit_jointly", False) is True
        assert validate_config_parameter("float_tolerance", 1e-8) is True
        assert validate_config_parameter("equal_width_default_bins", 10) is True

        # Invalid parameter names
        assert validate_config_parameter("invalid_param", "value") is False
        assert validate_config_parameter("nonexistent_param", "value") is False

        # Invalid strategy value
        assert validate_config_parameter("equal_width_default_range_strategy", "invalid") is False

    def test_get_config_schema(self):
        """Test get_config_schema function."""
        schema = get_config_schema()

        assert isinstance(schema, dict)
        assert "preserve_dataframe" in schema
        assert "fit_jointly" in schema
        assert "float_tolerance" in schema

        # Check schema structure
        param_schema = schema["preserve_dataframe"]
        assert "type" in param_schema
        assert "default" in param_schema
        assert "description" in param_schema

    def test_get_parameter_description(self):
        """Test _get_parameter_description function."""
        desc = _get_parameter_description("preserve_dataframe")
        assert isinstance(desc, str)
        assert len(desc) > 0

        # Test fallback for unknown parameter
        desc = _get_parameter_description("unknown_param")
        assert desc == "No description available"


class TestConfigContext:
    """Test ConfigContext context manager for 100% coverage."""

    def test_context_manager_basic(self):
        """Test basic ConfigContext functionality."""
        # Set initial config
        reset_config()
        original_preserve = get_config().preserve_dataframe

        # Use context manager to temporarily change config
        with ConfigContext(preserve_dataframe=True, fit_jointly=True):
            config = get_config()
            assert config.preserve_dataframe is True
            assert config.fit_jointly is True

        # Config should be restored after context
        config = get_config()
        assert config.preserve_dataframe == original_preserve
        assert config.fit_jointly is False  # Default value

    def test_context_manager_exception_handling(self):
        """Test ConfigContext properly restores config even with exceptions."""
        reset_config()
        original_preserve = get_config().preserve_dataframe

        try:
            with ConfigContext(preserve_dataframe=True):
                assert get_config().preserve_dataframe is True
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Config should still be restored after exception
        config = get_config()
        assert config.preserve_dataframe == original_preserve

    def test_context_manager_nested(self):
        """Test nested ConfigContext usage."""
        reset_config()

        with ConfigContext(preserve_dataframe=True):
            assert get_config().preserve_dataframe is True

            with ConfigContext(fit_jointly=True):
                config = get_config()
                assert config.preserve_dataframe is True  # Still preserved
                assert config.fit_jointly is True

            # Inner context restored
            config = get_config()
            assert config.preserve_dataframe is True
            assert config.fit_jointly is False  # Restored

        # Outer context restored
        config = get_config()
        assert config.preserve_dataframe is False
        assert config.fit_jointly is False


class TestConfigEdgeCases:
    """Test edge cases and error conditions."""

    def test_config_file_not_found(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            BinningConfig.load_from_file("/non/existent/file.json")

    def test_config_file_invalid_json(self):
        """Test loading from file with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                BinningConfig.load_from_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_save_to_file_permission_error(self):
        """Test saving to file with permission error."""
        config = BinningConfig()

        # Try to save to a directory that doesn't exist
        with pytest.raises(FileNotFoundError):
            config.save_to_file("/non/existent/directory/config.json")


class TestConfigIntegration:
    """Integration tests for complete configuration workflow."""

    def test_full_workflow(self):
        """Test complete configuration workflow."""
        # Start with defaults
        reset_config()

        # Update config
        set_config(preserve_dataframe=True, equal_width_default_bins=8)
        config = get_config()
        assert config.preserve_dataframe is True
        assert config.equal_width_default_bins == 8

        # Save to file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            config.save_to_file(temp_path)

            # Reset and reload
            reset_config()
            assert get_config().preserve_dataframe is False  # Back to default

            load_config(temp_path)
            config = get_config()
            assert config.preserve_dataframe is True
            assert config.equal_width_default_bins == 8

        finally:
            os.unlink(temp_path)
            reset_config()

    def test_method_defaults_integration(self):
        """Test method defaults integration with configuration."""
        reset_config()

        # Modify config
        set_config(equal_width_default_bins=7, default_clip=False, preserve_dataframe=True)

        config = get_config()

        # Test method defaults
        ew_defaults = config.get_method_defaults("equal_width")
        assert ew_defaults["n_bins"] == 7
        assert ew_defaults["clip"] is False

        # Test apply_config_defaults
        params = apply_config_defaults("equal_width")

        assert params["n_bins"] == 7
        assert params["clip"] is False
        assert params["preserve_dataframe"] is True

        reset_config()
