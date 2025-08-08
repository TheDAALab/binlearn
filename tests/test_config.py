"""
Comprehensive test suite for binlearn configuration system.

This module tests all functionality of the config.py module including:
- BinningConfig dataclass functionality
- ConfigManager singleton pattern
- Environment variable loading
- File I/O operations
- Parameter validation
- Method-specific defaults
- Configuration context managers
- Global configuration functions
- Configuration utilities and schema
"""

import json
import os
import tempfile
from pathlib import Path
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


class TestBinningConfig:
    """Test BinningConfig dataclass functionality."""

    def test_default_initialization(self):
        """Test BinningConfig creates with expected defaults."""
        config = BinningConfig()

        # Test core framework settings
        assert config.float_tolerance == 1e-10
        assert config.preserve_dataframe is False
        assert config.fit_jointly is False

        # Test interval binning defaults
        assert config.default_clip is True
        assert config.default_n_bins == 10
        assert config.default_random_state is None

        # Test specific method defaults
        assert config.equal_width_default_bins == 5
        assert config.equal_width_default_range_strategy == "min_max"
        assert config.equal_frequency_default_bins == 10
        assert config.kmeans_default_bins == 10
        assert config.gaussian_mixture_default_components == 10

        # Test flexible binning defaults
        assert config.singleton_max_unique_values == 1000
        assert config.singleton_sort_values is True

        # Test supervised binning defaults
        assert config.supervised_default_max_depth == 3
        assert config.supervised_default_min_samples_leaf == 5
        assert config.supervised_default_min_samples_split == 10
        assert config.supervised_default_task_type == "classification"

        # Test validation settings
        assert config.strict_validation is True
        assert config.allow_empty_bins is False
        assert config.validate_input_types is True

        # Test performance settings
        assert config.parallel_processing is False
        assert config.enable_caching is False

        # Test data handling
        assert config.missing_value_strategy == "preserve"
        assert config.infinite_value_strategy == "clip"
        assert config.auto_detect_numeric_columns is True

    def test_custom_initialization(self):
        """Test BinningConfig with custom parameters."""
        config = BinningConfig(
            float_tolerance=1e-8,
            equal_width_default_bins=7,
            supervised_default_max_depth=5,
            strict_validation=False,
        )

        assert config.float_tolerance == 1e-8
        assert config.equal_width_default_bins == 7
        assert config.supervised_default_max_depth == 5
        assert config.strict_validation is False
        # Other values should remain defaults
        assert config.default_n_bins == 10

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "float_tolerance": 1e-9,
            "equal_width_default_bins": 8,
            "unknown_parameter": "ignored",  # Should be ignored
        }

        config = BinningConfig.from_dict(config_dict)
        assert config.float_tolerance == 1e-9
        assert config.equal_width_default_bins == 8
        assert not hasattr(config, "unknown_parameter")

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = BinningConfig(float_tolerance=1e-9)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["float_tolerance"] == 1e-9
        assert "equal_width_default_bins" in config_dict

    def test_load_from_file(self, tmp_path):
        """Test loading configuration from JSON file."""
        config_data = {
            "float_tolerance": 1e-9,
            "equal_width_default_bins": 7,
            "preserve_dataframe": True,
        }

        config_file = tmp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        config = BinningConfig.load_from_file(str(config_file))
        assert config.float_tolerance == 1e-9
        assert config.equal_width_default_bins == 7
        assert config.preserve_dataframe is True

    def test_load_from_file_nonexistent(self):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            BinningConfig.load_from_file("nonexistent.json")

    def test_save_to_file(self, tmp_path):
        """Test saving configuration to JSON file."""
        config = BinningConfig(float_tolerance=1e-9, equal_width_default_bins=7)

        config_file = tmp_path / "test_save.json"
        config.save_to_file(str(config_file))

        assert config_file.exists()
        with open(config_file) as f:
            saved_data = json.load(f)

        assert saved_data["float_tolerance"] == 1e-9
        assert saved_data["equal_width_default_bins"] == 7

    def test_update_valid_parameters(self):
        """Test updating configuration with valid parameters."""
        config = BinningConfig()
        config.update(float_tolerance=1e-8, equal_width_default_bins=6, preserve_dataframe=True)

        assert config.float_tolerance == 1e-8
        assert config.equal_width_default_bins == 6
        assert config.preserve_dataframe is True

    def test_update_invalid_parameter(self):
        """Test updating with invalid parameter raises error."""
        config = BinningConfig()
        with pytest.raises(ValueError, match="Unknown configuration parameter"):
            config.update(invalid_param="value")

    def test_update_strategy_parameter_valid(self):
        """Test updating strategy parameters with valid values."""
        config = BinningConfig()
        config.update(
            equal_width_default_range_strategy="percentile",
            missing_value_strategy="error",
            infinite_value_strategy="preserve",
        )

        assert config.equal_width_default_range_strategy == "percentile"
        assert config.missing_value_strategy == "error"
        assert config.infinite_value_strategy == "preserve"

    def test_update_strategy_parameter_invalid(self):
        """Test updating strategy parameters with invalid values."""
        config = BinningConfig()

        with pytest.raises(ValueError, match="must be one of"):
            config.update(equal_width_default_range_strategy="invalid")

        with pytest.raises(ValueError, match="must be one of"):
            config.update(missing_value_strategy="invalid")

    def test_update_depth_parameter_validation(self):
        """Test validation of depth parameters."""
        config = BinningConfig()

        # Valid depth
        config.update(supervised_default_max_depth=5)
        assert config.supervised_default_max_depth == 5

        # Invalid depth (negative)
        with pytest.raises(ValueError, match="must be positive"):
            config.update(supervised_default_max_depth=-1)

        # Invalid depth (zero)
        with pytest.raises(ValueError, match="must be positive"):
            config.update(supervised_default_max_depth=0)

    def test_update_float_tolerance_validation(self):
        """Test validation of float_tolerance parameter."""
        config = BinningConfig()

        # Valid tolerance
        config.update(float_tolerance=1e-8)
        assert config.float_tolerance == 1e-8

        # Invalid tolerance (negative)
        with pytest.raises(ValueError, match="must be positive"):
            config.update(float_tolerance=-1e-8)

        # Invalid tolerance (zero)
        with pytest.raises(ValueError, match="must be positive"):
            config.update(float_tolerance=0)

    def test_validate_strategy_parameter_coverage(self):
        """Test _validate_strategy_parameter method covers all strategies."""
        config = BinningConfig()

        # Test strategy parameters (only those ending with "_strategy")
        strategies_to_test = [
            ("equal_width_default_range_strategy", ["min_max", "percentile", "std"]),
            ("missing_value_strategy", ["preserve", "error", "ignore"]),
            ("infinite_value_strategy", ["clip", "error", "preserve"]),
        ]

        for strategy_key, valid_values in strategies_to_test:
            # Test valid values
            for valid_value in valid_values:
                config.update(**{strategy_key: valid_value})
                assert getattr(config, strategy_key) == valid_value

            # Test invalid value
            with pytest.raises(ValueError):
                config.update(**{strategy_key: "invalid_value"})

        # Test default_column_selection separately since it doesn't end with "_strategy"
        # but is still validated internally
        for valid_value in ["numeric", "all", "explicit"]:
            config.update(default_column_selection=valid_value)
            assert config.default_column_selection == valid_value

    def test_get_method_defaults_equal_width(self):
        """Test get_method_defaults for equal_width method."""
        config = BinningConfig()
        defaults = config.get_method_defaults("equal_width")

        expected_keys = {
            "preserve_dataframe",
            "fit_jointly",
            "strict_validation",
            "clip",
            "n_bins",
            "range_strategy",
        }
        assert set(defaults.keys()) == expected_keys
        assert defaults["n_bins"] == config.equal_width_default_bins
        assert defaults["range_strategy"] == config.equal_width_default_range_strategy

    def test_get_method_defaults_equal_frequency(self):
        """Test get_method_defaults for equal_frequency method."""
        config = BinningConfig()
        defaults = config.get_method_defaults("equal_frequency")

        assert "n_bins" in defaults
        assert "quantile_range" in defaults
        assert defaults["n_bins"] == config.equal_frequency_default_bins

    def test_get_method_defaults_kmeans(self):
        """Test get_method_defaults for kmeans method."""
        config = BinningConfig()
        defaults = config.get_method_defaults("kmeans")

        assert "n_bins" in defaults
        assert "random_state" in defaults
        assert defaults["n_bins"] == config.kmeans_default_bins

    def test_get_method_defaults_gaussian_mixture(self):
        """Test get_method_defaults for gaussian_mixture method."""
        config = BinningConfig()
        defaults = config.get_method_defaults("gaussian_mixture")

        assert "n_components" in defaults
        assert "random_state" in defaults
        assert defaults["n_components"] == config.gaussian_mixture_default_components

    def test_get_method_defaults_equal_width_minimum_weight(self):
        """Test get_method_defaults for equal_width_minimum_weight method."""
        config = BinningConfig()
        defaults = config.get_method_defaults("equal_width_minimum_weight")

        assert "n_bins" in defaults
        assert "minimum_weight" in defaults
        assert defaults["n_bins"] == config.equal_width_min_weight_default_bins

    def test_get_method_defaults_chi2(self):
        """Test get_method_defaults for chi2 method."""
        config = BinningConfig()
        defaults = config.get_method_defaults("chi2")

        assert "max_bins" in defaults
        assert "min_bins" in defaults
        assert "alpha" in defaults
        assert "initial_bins" in defaults

    def test_get_method_defaults_dbscan(self):
        """Test get_method_defaults for dbscan method."""
        config = BinningConfig()
        defaults = config.get_method_defaults("dbscan")

        assert "eps" in defaults
        assert "min_samples" in defaults
        assert "random_state" in defaults

    def test_get_method_defaults_isotonic(self):
        """Test get_method_defaults for isotonic method."""
        config = BinningConfig()
        defaults = config.get_method_defaults("isotonic")

        assert "n_bins" in defaults
        assert "increasing" in defaults

    def test_get_method_defaults_singleton(self):
        """Test get_method_defaults for singleton method."""
        config = BinningConfig()
        defaults = config.get_method_defaults("singleton")

        assert "max_unique_values" in defaults
        assert "sort_values" in defaults

    def test_get_method_defaults_supervised(self):
        """Test get_method_defaults for supervised method."""
        config = BinningConfig()
        defaults = config.get_method_defaults("supervised")

        expected_supervised_keys = {
            "max_depth",
            "min_samples_leaf",
            "min_samples_split",
            "task_type",
            "random_state",
        }
        for key in expected_supervised_keys:
            assert key in defaults

    def test_get_method_defaults_unknown_method(self):
        """Test get_method_defaults for unknown method returns generic defaults."""
        config = BinningConfig()
        defaults = config.get_method_defaults("unknown_method")

        assert "n_bins" in defaults
        assert "random_state" in defaults
        assert defaults["n_bins"] == config.default_n_bins


class TestConfigManager:
    """Test ConfigManager singleton functionality."""

    def test_singleton_pattern(self):
        """Test ConfigManager implements singleton pattern."""
        manager1 = ConfigManager()
        manager2 = ConfigManager()
        assert manager1 is manager2

    def test_config_property(self):
        """Test config property returns BinningConfig instance."""
        manager = ConfigManager()
        config = manager.config
        assert isinstance(config, BinningConfig)

    def test_update_config(self):
        """Test updating configuration through manager."""
        manager = ConfigManager()
        original_tolerance = manager.config.float_tolerance

        manager.update_config(float_tolerance=1e-8)
        assert manager.config.float_tolerance == 1e-8

        # Reset for other tests
        manager.update_config(float_tolerance=original_tolerance)

    def test_load_config_from_file(self, tmp_path):
        """Test loading config from file through manager."""
        config_data = {"float_tolerance": 1e-7}
        config_file = tmp_path / "manager_test.json"

        with open(config_file, "w") as f:
            json.dump(config_data, f)

        manager = ConfigManager()
        original_tolerance = manager.config.float_tolerance

        manager.load_config(str(config_file))
        assert manager.config.float_tolerance == 1e-7

        # Reset
        manager.reset_to_defaults()
        assert manager.config.float_tolerance != 1e-7

    def test_reset_to_defaults(self):
        """Test resetting configuration to defaults."""
        manager = ConfigManager()
        manager.update_config(float_tolerance=1e-8, equal_width_default_bins=20)

        manager.reset_to_defaults()

        # Should be back to defaults
        default_config = BinningConfig()
        assert manager.config.float_tolerance == default_config.float_tolerance
        assert manager.config.equal_width_default_bins == default_config.equal_width_default_bins

    @patch.dict(
        os.environ,
        {
            "BINNING_FLOAT_TOLERANCE": "1e-8",
            "BINNING_DEFAULT_CLIP": "false",
            "BINNING_PRESERVE_DATAFRAME": "true",
            "BINNING_STRICT_VALIDATION": "no",
            "BINNING_SHOW_WARNINGS": "1",
            "BINNING_EQUAL_WIDTH_BINS": "15",
            "BINNING_SUPERVISED_MAX_DEPTH": "7",
        },
        clear=False,
    )
    def test_load_from_env_success(self):
        """Test loading configuration from environment variables successfully."""
        # Create a new manager to trigger environment loading
        ConfigManager._instance = None
        manager = ConfigManager()

        assert manager.config.float_tolerance == 1e-8
        assert manager.config.default_clip is False
        assert manager.config.preserve_dataframe is True
        assert manager.config.strict_validation is False
        assert manager.config.show_warnings is True
        assert manager.config.equal_width_default_bins == 15
        assert manager.config.supervised_default_max_depth == 7

        # Reset singleton for other tests
        ConfigManager._instance = None

    @patch.dict(
        os.environ,
        {"BINNING_FLOAT_TOLERANCE": "invalid_float", "BINNING_EQUAL_WIDTH_BINS": "not_an_int"},
        clear=False,
    )
    def test_load_from_env_invalid_values_with_raise(self):
        """Test invalid environment values raise errors when configured to do so."""
        ConfigManager._instance = None

        with pytest.raises(ValueError, match="Invalid environment variable"):
            ConfigManager()

        # Reset singleton
        ConfigManager._instance = None

    def test_load_from_env_invalid_values_no_raise(self):
        """Test invalid environment values don't raise when configured not to."""
        ConfigManager._instance = None

        # Create a manager first (without invalid env vars)
        manager = ConfigManager()
        # Then set raise_on_config_errors to False
        manager._config.raise_on_config_errors = False

        # Now test with invalid environment variable
        with patch.dict(os.environ, {"BINNING_FLOAT_TOLERANCE": "invalid_float"}, clear=False):
            # Should not raise an error even with invalid values
            try:
                manager._load_from_env()
                # If we get here, the test passes - invalid values were ignored
            except Exception as e:
                pytest.fail(
                    f"_load_from_env should not raise when raise_on_config_errors=False, but got: {e}"
                )

        # Reset singleton
        ConfigManager._instance = None

    def test_load_from_env_boolean_variations(self):
        """Test different boolean value formats in environment variables."""
        bool_variations = [
            ("true", True),
            ("True", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
            ("off", False),
            ("other", False),
        ]

        for env_value, expected in bool_variations:
            with patch.dict(os.environ, {"BINNING_DEFAULT_CLIP": env_value}, clear=False):
                ConfigManager._instance = None
                manager = ConfigManager()
                assert manager.config.default_clip == expected
                ConfigManager._instance = None


class TestGlobalConfigFunctions:
    """Test global configuration functions."""

    def test_get_config(self):
        """Test get_config returns current configuration."""
        config = get_config()
        assert isinstance(config, BinningConfig)

    def test_set_config(self):
        """Test set_config updates global configuration."""
        original_tolerance = get_config().float_tolerance

        set_config(float_tolerance=1e-8)
        assert get_config().float_tolerance == 1e-8

        # Reset
        set_config(float_tolerance=original_tolerance)

    def test_load_config(self, tmp_path):
        """Test load_config function."""
        config_data = {"equal_width_default_bins": 12}
        config_file = tmp_path / "global_test.json"

        with open(config_file, "w") as f:
            json.dump(config_data, f)

        load_config(str(config_file))
        assert get_config().equal_width_default_bins == 12

        # Reset
        reset_config()

    def test_reset_config(self):
        """Test reset_config function."""
        set_config(float_tolerance=1e-8)
        reset_config()

        default_config = BinningConfig()
        assert get_config().float_tolerance == default_config.float_tolerance


class TestConfigurationUtilities:
    """Test configuration utility functions."""

    def test_apply_config_defaults_basic(self):
        """Test apply_config_defaults with basic usage."""
        result = apply_config_defaults("equal_width")

        assert isinstance(result, dict)
        assert "n_bins" in result
        assert "preserve_dataframe" in result

    def test_apply_config_defaults_with_user_params(self):
        """Test apply_config_defaults with user parameters."""
        user_params = {"n_bins": 8, "custom_param": "value"}
        result = apply_config_defaults("equal_width", user_params)

        assert result["n_bins"] == 8
        assert result["custom_param"] == "value"

    def test_apply_config_defaults_with_overrides(self):
        """Test apply_config_defaults with override parameters."""
        user_params = {"n_bins": 8}
        result = apply_config_defaults(
            "equal_width",
            user_params,
            n_bins=15,  # Should override user_params
            preserve_dataframe=True,
        )

        assert result["n_bins"] == 15  # Override wins
        assert result["preserve_dataframe"] is True

    def test_apply_config_defaults_precedence(self):
        """Test parameter precedence in apply_config_defaults."""
        # Test all precedence levels
        result = apply_config_defaults(
            "equal_width",
            user_params={"n_bins": 8, "clip": False},
            n_bins=20,  # Override should win
            preserve_dataframe=True,  # New override param
        )

        assert result["n_bins"] == 20  # Override beats user_params
        assert result["clip"] is False  # User param preserved
        assert result["preserve_dataframe"] is True  # Override added

    def test_validate_config_parameter_valid(self):
        """Test validate_config_parameter with valid parameters."""
        assert validate_config_parameter("float_tolerance", 1e-8) is True
        assert validate_config_parameter("equal_width_default_bins", 10) is True
        assert validate_config_parameter("preserve_dataframe", True) is True

    def test_validate_config_parameter_invalid(self):
        """Test validate_config_parameter with invalid parameters."""
        assert validate_config_parameter("nonexistent_param", "value") is False
        assert validate_config_parameter("float_tolerance", -1e-8) is False
        assert validate_config_parameter("supervised_default_max_depth", -1) is False

    def test_get_config_schema(self):
        """Test get_config_schema returns complete schema."""
        schema = get_config_schema()

        assert isinstance(schema, dict)
        assert "float_tolerance" in schema
        assert "equal_width_default_bins" in schema

        # Check structure of schema entries
        for param_name, param_info in schema.items():
            assert "type" in param_info
            assert "default" in param_info
            assert "description" in param_info

    def test_get_parameter_description_known(self):
        """Test _get_parameter_description with known parameters."""
        desc = _get_parameter_description("float_tolerance")
        assert "tolerance" in desc.lower()

        desc = _get_parameter_description("preserve_dataframe")
        assert "dataframe" in desc.lower()

    def test_get_parameter_description_unknown(self):
        """Test _get_parameter_description with unknown parameter."""
        desc = _get_parameter_description("unknown_param")
        assert desc == "No description available"


class TestConfigContext:
    """Test ConfigContext context manager."""

    def test_config_context_basic(self):
        """Test basic ConfigContext functionality."""
        original_tolerance = get_config().float_tolerance

        with ConfigContext(float_tolerance=1e-8):
            assert get_config().float_tolerance == 1e-8

        # Should be restored
        assert get_config().float_tolerance == original_tolerance

    def test_config_context_multiple_params(self):
        """Test ConfigContext with multiple parameters."""
        original_config = get_config()
        original_tolerance = original_config.float_tolerance
        original_bins = original_config.equal_width_default_bins

        with ConfigContext(float_tolerance=1e-8, equal_width_default_bins=15):
            assert get_config().float_tolerance == 1e-8
            assert get_config().equal_width_default_bins == 15

        # Should be restored
        assert get_config().float_tolerance == original_tolerance
        assert get_config().equal_width_default_bins == original_bins

    def test_config_context_with_exception(self):
        """Test ConfigContext restores config even when exception occurs."""
        original_tolerance = get_config().float_tolerance

        try:
            with ConfigContext(float_tolerance=1e-8):
                assert get_config().float_tolerance == 1e-8
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still be restored
        assert get_config().float_tolerance == original_tolerance

    def test_config_context_nested(self):
        """Test nested ConfigContext managers."""
        original_tolerance = get_config().float_tolerance

        with ConfigContext(float_tolerance=1e-8):
            assert get_config().float_tolerance == 1e-8

            with ConfigContext(float_tolerance=1e-9):
                assert get_config().float_tolerance == 1e-9

            # Inner context should be restored to outer
            assert get_config().float_tolerance == 1e-8

        # Outer context should be restored to original
        assert get_config().float_tolerance == original_tolerance

    def test_config_context_return_self(self):
        """Test ConfigContext __enter__ returns self."""
        with ConfigContext(float_tolerance=1e-8) as ctx:
            assert isinstance(ctx, ConfigContext)


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    def test_config_file_json_decode_error(self, tmp_path):
        """Test handling of invalid JSON in config file."""
        config_file = tmp_path / "invalid.json"
        with open(config_file, "w") as f:
            f.write("invalid json content")

        with pytest.raises(json.JSONDecodeError):
            BinningConfig.load_from_file(str(config_file))

    def test_config_file_permission_error(self, tmp_path):
        """Test handling of permission errors in file operations."""
        config_file = tmp_path / "no_permission.json"
        config_file.touch()
        config_file.chmod(0o000)

        try:
            with pytest.raises(PermissionError):
                BinningConfig.load_from_file(str(config_file))
        finally:
            # Restore permissions for cleanup
            config_file.chmod(0o644)

    def test_save_to_invalid_directory(self):
        """Test saving to invalid directory path."""
        config = BinningConfig()

        with pytest.raises(FileNotFoundError):
            config.save_to_file("/nonexistent/directory/config.json")

    def test_update_with_none_values(self):
        """Test updating config with None values."""
        config = BinningConfig()

        # Some parameters should accept None
        config.update(default_random_state=None)
        assert config.default_random_state is None

    def test_environment_variable_type_conversion_edge_cases(self):
        """Test edge cases in environment variable type conversion."""
        with patch.dict(os.environ, {"BINNING_FLOAT_TOLERANCE": "0.0001"}, clear=False):
            ConfigManager._instance = None
            manager = ConfigManager()
            assert manager.config.float_tolerance == 0.0001
            ConfigManager._instance = None


class TestAllMethodDefaults:
    """Test that all method defaults are properly covered."""

    def test_all_implemented_methods_have_defaults(self):
        """Test that get_method_defaults handles all known methods."""
        config = BinningConfig()

        # List of all methods that should have specific defaults
        methods_with_defaults = [
            "equal_width",
            "equal_frequency",
            "kmeans",
            "gaussian_mixture",
            "equal_width_minimum_weight",
            "chi2",
            "dbscan",
            "isotonic",
            "singleton",
            "supervised",
        ]

        for method in methods_with_defaults:
            defaults = config.get_method_defaults(method)
            assert isinstance(defaults, dict)
            assert len(defaults) > 0
            # All should have basic common parameters
            assert "preserve_dataframe" in defaults
            assert "fit_jointly" in defaults
            assert "strict_validation" in defaults
            assert "clip" in defaults

    def test_generic_method_defaults(self):
        """Test generic defaults for unknown methods."""
        config = BinningConfig()
        defaults = config.get_method_defaults("unknown_method")

        # Should get generic defaults
        assert "n_bins" in defaults
        assert "random_state" in defaults
        assert defaults["n_bins"] == config.default_n_bins


if __name__ == "__main__":
    pytest.main([__file__])
