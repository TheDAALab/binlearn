"""
Configuration management system for the binning framework.

This module provides a comprehensive configuration system that supports:
- Default parameter management for all binning methods
- Environment variable configuration
- File-based configuration loading/saving
- Type-safe configuration with validation
- Integration with the binning framework's type system
"""

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


# pylint: disable=too-many-instance-attributes
@dataclass
class BinningConfig:
    """
    Global configuration for binning framework.

    This dataclass contains all configurable parameters for the binning framework,
    organized by category for better maintainability and type safety.
    """

    # =============================================================================
    # CORE FRAMEWORK SETTINGS
    # =============================================================================

    # Numerical precision
    float_tolerance: float = 1e-10

    # Default behaviors for all binning methods
    preserve_dataframe: bool = False  # Whether to preserve DataFrame structure by default
    fit_jointly: bool = False  # Whether to fit bins jointly across columns by default

    # =============================================================================
    # INTERVAL BINNING DEFAULTS
    # =============================================================================

    # General interval binning settings
    default_clip: bool = True  # Whether to clip values outside bin ranges

    # EqualWidthBinning specific defaults
    equal_width_default_bins: int = 5
    equal_width_default_range_strategy: str = "min_max"  # "min_max", "percentile", "std"

    # =============================================================================
    # FLEXIBLE BINNING DEFAULTS
    # =============================================================================

    # SingletonBinning specific defaults
    singleton_max_unique_values: int = 1000  # Safety limit for unique values
    singleton_sort_values: bool = True  # Whether to sort unique values

    # =============================================================================
    # SUPERVISED BINNING DEFAULTS
    # =============================================================================

    # SupervisedBinning specific defaults
    supervised_default_max_depth: int = 3
    supervised_default_min_samples_leaf: int = 5
    supervised_default_min_samples_split: int = 10
    supervised_default_task_type: str = "classification"
    supervised_random_state: Optional[int] = None

    # =============================================================================
    # VALIDATION AND ERROR HANDLING
    # =============================================================================

    # Input validation settings
    strict_validation: bool = True
    allow_empty_bins: bool = False
    validate_input_types: bool = True
    validate_column_names: bool = True

    # Error handling and messaging
    show_warnings: bool = True
    detailed_error_messages: bool = True
    raise_on_config_errors: bool = True

    # =============================================================================
    # PERFORMANCE AND OPTIMIZATION
    # =============================================================================

    # Performance settings
    parallel_processing: bool = False
    max_workers: Optional[int] = None
    memory_efficient_mode: bool = False

    # Caching settings
    enable_caching: bool = False
    cache_size_limit: int = 1000

    # =============================================================================
    # DATA HANDLING PREFERENCES
    # =============================================================================

    # Default handling for special values
    missing_value_strategy: str = "preserve"  # "preserve", "error", "ignore"
    infinite_value_strategy: str = "clip"  # "clip", "error", "preserve"

    # Column handling
    auto_detect_numeric_columns: bool = True
    default_column_selection: str = "numeric"  # "numeric", "all", "explicit"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BinningConfig":
        """Create config from dictionary, ignoring unknown keys."""
        valid_keys = {
            field.name for field in cls.__dataclass_fields__.values()  # pylint: disable=no-member
        }
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def load_from_file(cls, filepath: str) -> "BinningConfig":
        """Load configuration from JSON file."""
        with open(filepath, encoding="utf-8") as file_handle:
            config_dict = json.load(file_handle)
        return cls.from_dict(config_dict)

    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, "w", encoding="utf-8") as file_handle:
            json.dump(self.to_dict(), file_handle, indent=2)

    def update(self, **kwargs: Any) -> None:
        """
        Update configuration parameters with validation.

        Args:
            **kwargs: Configuration parameters to update

        Raises:
            ValueError: If parameter name is unknown or value is invalid
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                # Validate specific parameter types
                if key.endswith("_strategy") and isinstance(value, str):
                    self._validate_strategy_parameter(key, value)
                elif (
                    key.startswith("supervised_")
                    and key.endswith("_depth")
                    and isinstance(value, int)
                ):
                    if value < 1:
                        raise ValueError(f"{key} must be positive, got {value}")
                elif key == "float_tolerance" and isinstance(value, (int, float)):
                    if value <= 0:
                        raise ValueError("float_tolerance must be positive")

                setattr(self, key, value)
            else:
                available_keys = list(self.__dataclass_fields__.keys())  # pylint: disable=no-member
                raise ValueError(
                    f"Unknown configuration parameter: {key}. Available: {available_keys}"
                )

    def _validate_strategy_parameter(self, key: str, value: str) -> None:
        """Validate strategy-type parameters."""
        valid_strategies = {
            "equal_width_default_range_strategy": ["min_max", "percentile", "std"],
            "missing_value_strategy": ["preserve", "error", "ignore"],
            "infinite_value_strategy": ["clip", "error", "preserve"],
            "default_column_selection": ["numeric", "all", "explicit"],
        }

        if key in valid_strategies and value not in valid_strategies[key]:
            raise ValueError(f"{key} must be one of {valid_strategies[key]}, got '{value}'")

    def get_method_defaults(self, method_name: str) -> Dict[str, Any]:
        """
        Get default configuration values for a specific binning method.

        Args:
            method_name: Name of the binning method ("equal_width", "singleton", "supervised")

        Returns:
            Dictionary of default parameters for the method
        """
        defaults: Dict[str, Any] = {
            "preserve_dataframe": self.preserve_dataframe,
            "fit_jointly": self.fit_jointly,
            "strict_validation": self.strict_validation,
        }

        if method_name == "equal_width":
            defaults.update(
                {
                    "n_bins": self.equal_width_default_bins,
                    "clip": self.default_clip,
                    "range_strategy": self.equal_width_default_range_strategy,
                }
            )
        elif method_name == "singleton":
            defaults.update(
                {
                    "max_unique_values": self.singleton_max_unique_values,
                    "sort_values": self.singleton_sort_values,
                }
            )
        elif method_name == "supervised":
            defaults.update(
                {
                    "max_depth": self.supervised_default_max_depth,
                    "min_samples_leaf": self.supervised_default_min_samples_leaf,
                    "min_samples_split": self.supervised_default_min_samples_split,
                    "task_type": self.supervised_default_task_type,
                    "random_state": self.supervised_random_state,
                }
            )

        return defaults


class ConfigManager:
    """Global configuration manager singleton with enhanced functionality."""

    _instance: Optional["ConfigManager"] = None
    _config: BinningConfig

    def __new__(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = BinningConfig()
            cls._instance._load_from_env()
        return cls._instance

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        env_mapping = {
            "BINNING_FLOAT_TOLERANCE": ("float_tolerance", float),
            "BINNING_DEFAULT_CLIP": ("default_clip", bool),
            "BINNING_PRESERVE_DATAFRAME": ("preserve_dataframe", bool),
            "BINNING_STRICT_VALIDATION": ("strict_validation", bool),
            "BINNING_SHOW_WARNINGS": ("show_warnings", bool),
            "BINNING_EQUAL_WIDTH_BINS": ("equal_width_default_bins", int),
            "BINNING_SUPERVISED_MAX_DEPTH": ("supervised_default_max_depth", int),
        }

        for env_var, (config_key, value_type) in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    if value_type is bool:
                        value = env_value.lower() in ("true", "1", "yes", "on")
                    else:
                        value = value_type(env_value)

                    setattr(self._config, config_key, value)
                except (ValueError, TypeError) as exc:
                    if self._config.raise_on_config_errors:
                        raise ValueError(
                            f"Invalid environment variable {env_var}={env_value}: {exc}"
                        ) from exc

    @property
    def config(self) -> BinningConfig:
        """Get current configuration."""
        return self._config

    def update_config(self, **kwargs: Any) -> None:
        """Update configuration parameters."""
        self._config.update(**kwargs)

    def load_config(self, filepath: str) -> None:
        """Load configuration from file."""
        self._config = BinningConfig.load_from_file(filepath)

    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self._config = BinningConfig()
        self._load_from_env()


# Global configuration instance
_config_manager = ConfigManager()  # pylint: disable=invalid-name


def get_config() -> BinningConfig:
    """Get the global configuration."""
    return _config_manager.config


def set_config(**kwargs: Any) -> None:
    """Set configuration parameters."""
    _config_manager.update_config(**kwargs)


def load_config(filepath: str) -> None:
    """Load configuration from file."""
    _config_manager.load_config(filepath)


def reset_config() -> None:
    """Reset configuration to defaults."""
    _config_manager.reset_to_defaults()


# =============================================================================
# CONFIGURATION INTEGRATION UTILITIES
# =============================================================================


def apply_config_defaults(
    method_name: str, user_params: Optional[Dict[str, Any]] = None, **override_params: Any
) -> Dict[str, Any]:
    """
    Apply configuration defaults to user parameters for a specific method.

    This function provides a clean way to merge user-provided parameters
    with configuration defaults, following a clear precedence order:
    1. override_params (highest priority)
    2. user_params
    3. configuration defaults (lowest priority)

    Args:
        method_name: Name of the binning method
        user_params: User-provided parameters (can be None)
        **override_params: Additional override parameters

    Returns:
        Dictionary with merged parameters

    Example:
        >>> config_params = apply_config_defaults("equal_width",
        ...     user_params={"n_bins": 10},
        ...     preserve_dataframe=True)
    """
    config = get_config()

    # Start with configuration defaults
    params = config.get_method_defaults(method_name)

    # Apply user parameters
    if user_params:
        params.update(user_params)

    # Apply override parameters (highest priority)
    params.update(override_params)

    return params


def validate_config_parameter(name: str, value: Any) -> bool:
    """
    Validate a configuration parameter name and value.

    Args:
        name: Parameter name
        value: Parameter value

    Returns:
        True if valid, False otherwise
    """
    try:
        # Create a temporary config to test validation
        temp_config = BinningConfig()
        temp_config.update(**{name: value})
        return True
    except (ValueError, TypeError):
        return False


def get_config_schema() -> Dict[str, Dict[str, Any]]:
    """
    Get the configuration schema with parameter descriptions and types.

    Returns:
        Dictionary describing all configuration parameters
    """
    config = BinningConfig()
    schema = {}

    for field_name, field in config.__dataclass_fields__.items():  # pylint: disable=no-member
        schema[field_name] = {
            "type": field.type,
            "default": getattr(config, field_name),
            "description": _get_parameter_description(field_name),
        }

    return schema


def _get_parameter_description(param_name: str) -> str:
    """Get human-readable description for a configuration parameter."""
    descriptions = {
        "float_tolerance": "Numerical precision tolerance for floating point comparisons",
        "preserve_dataframe": "Whether to preserve DataFrame structure in outputs by default",
        "fit_jointly": "Whether to fit bins jointly across columns by default",
        "default_clip": "Whether to clip values outside bin ranges by default",
        "equal_width_default_bins": "Default number of bins for equal-width binning",
        "equal_width_default_range_strategy": "Strategy for determining bin ranges",
        "singleton_max_unique_values": "Maximum unique values allowed for singleton binning",
        "singleton_sort_values": "Whether to sort unique values in singleton binning",
        "supervised_default_max_depth": "Default maximum depth for decision trees",
        "supervised_default_min_samples_leaf": "Default minimum samples per leaf",
        "supervised_default_min_samples_split": "Default minimum samples to split",
        "supervised_default_task_type": "Default task type for supervised binning",
        "supervised_random_state": "Default random state for supervised binning",
        "strict_validation": "Whether to perform strict input validation",
        "allow_empty_bins": "Whether to allow bins with no data points",
        "validate_input_types": "Whether to validate input data types",
        "validate_column_names": "Whether to validate column names",
        "show_warnings": "Whether to display warning messages",
        "detailed_error_messages": "Whether to provide detailed error messages",
        "raise_on_config_errors": "Whether to raise exceptions on configuration errors",
        "parallel_processing": "Whether to enable parallel processing",
        "max_workers": "Maximum number of worker processes",
        "memory_efficient_mode": "Whether to use memory-efficient algorithms",
        "enable_caching": "Whether to enable result caching",
        "cache_size_limit": "Maximum number of cached results",
        "missing_value_strategy": "How to handle missing values",
        "infinite_value_strategy": "How to handle infinite values",
        "auto_detect_numeric_columns": "Whether to automatically detect numeric columns",
        "default_column_selection": "Default strategy for column selection",
    }

    return descriptions.get(param_name, "No description available")


# =============================================================================
# CONTEXT MANAGERS FOR CONFIGURATION
# =============================================================================


class ConfigContext:
    """
    Context manager for temporary configuration changes.

    Example:
        >>> with ConfigContext(strict_validation=False):
        ...     # Code with relaxed validation
        ...     pass
        # Configuration is automatically restored
    """

    def __init__(self, **temp_config: Any) -> None:
        self.temp_config = temp_config
        self.original_config: Dict[str, Any] = {}

    def __enter__(self) -> "ConfigContext":
        # Save current configuration
        current_config = get_config()
        self.original_config = current_config.to_dict()

        # Apply temporary configuration
        set_config(**self.temp_config)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Restore original configuration
        _config_manager._config = BinningConfig.from_dict(self.original_config)
