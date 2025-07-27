"""
Configuration management system for the binning framework.
"""

import os
import json
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class BinningConfig:
    """Global configuration for binning framework."""

    # Numerical precision
    float_tolerance: float = 1e-10

    # Default parameters for binning methods
    default_clip: bool = True
    preserve_dataframe: bool = False  # For GeneralBinningBase
    fit_jointly: bool = False  # For GeneralBinningBase
    default_preserve_dataframe: bool = False
    default_fit_jointly: bool = False

    # Validation settings
    strict_validation: bool = True
    allow_empty_bins: bool = False
    validate_input_types: bool = True

    # Error handling
    show_warnings: bool = True
    detailed_error_messages: bool = True

    # SupervisedBinning defaults
    supervised_default_max_depth: int = 3
    supervised_default_min_samples_leaf: int = 5
    supervised_default_min_samples_split: int = 10

    # EqualWidthBinning defaults
    equal_width_default_bins: int = 5

    # Performance settings (for future use)
    parallel_processing: bool = False
    max_workers: Optional[int] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BinningConfig":
        """Create config from dictionary, ignoring unknown keys."""
        valid_keys = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def load_from_file(cls, filepath: str) -> "BinningConfig":
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def update(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")


class ConfigManager:
    """Global configuration manager singleton."""

    _instance: Optional["ConfigManager"] = None
    _config: BinningConfig

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = BinningConfig()
            cls._instance._load_from_env()
        return cls._instance

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        env_mapping = {
            "BINNING_FLOAT_TOLERANCE": "float_tolerance",
            "BINNING_DEFAULT_CLIP": "default_clip",
            "BINNING_PRESERVE_DATAFRAME": "default_preserve_dataframe",
            "BINNING_STRICT_VALIDATION": "strict_validation",
            "BINNING_SHOW_WARNINGS": "show_warnings",
        }

        for env_var, config_key in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string to appropriate type
                if config_key in [
                    "default_clip",
                    "default_preserve_dataframe",
                    "strict_validation",
                    "show_warnings",
                ]:
                    value = env_value.lower() in ("true", "1", "yes", "on")
                elif config_key == "float_tolerance":
                    value = float(env_value)
                else:
                    value = env_value

                setattr(self._config, config_key, value)

    @property
    def config(self) -> BinningConfig:
        """Get current configuration."""
        return self._config

    def update_config(self, **kwargs) -> None:
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
_config_manager = ConfigManager()


def get_config() -> BinningConfig:
    """Get the global configuration."""
    return _config_manager.config


def set_config(**kwargs) -> None:
    """Set configuration parameters."""
    _config_manager.update_config(**kwargs)


def load_config(filepath: str) -> None:
    """Load configuration from file."""
    _config_manager.load_config(filepath)


def reset_config() -> None:
    """Reset configuration to defaults."""
    _config_manager.reset_to_defaults()
