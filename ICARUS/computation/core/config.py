"""
Configuration management for the simulation framework.

This module provides centralized configuration management with validation,
environment variable support, and configuration file loading.
"""

import json
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import yaml

from .exceptions import ConfigurationError
from .types import ExecutionMode
from .types import Priority


class LogLevel(Enum):
    """Logging levels for the simulation framework."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class SimulationConfig:
    """
    Comprehensive configuration for the simulation framework.

    This class centralizes all configuration options with sensible defaults,
    validation, and support for loading from files or environment variables.
    """

    # Execution settings
    execution_mode: ExecutionMode = ExecutionMode.ASYNC
    max_workers: Optional[int] = None
    task_timeout_seconds: float = 300.0
    default_task_priority: Priority = Priority.NORMAL
    enable_checkpointing: bool = True
    checkpoint_interval: int = 100

    # Progress monitoring
    enable_progress_monitoring: bool = True
    progress_refresh_rate: float = 0.2
    progress_bar_width: int = 30
    show_task_details: bool = True

    # Resource management
    max_memory_usage_mb: Optional[int] = None
    resource_cleanup_timeout: float = 30.0
    enable_resource_monitoring: bool = True

    # Logging configuration
    log_level: LogLevel = LogLevel.INFO
    log_to_file: bool = False
    log_file_path: Optional[Path] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Performance tuning
    batch_size: int = 50
    retry_delay_seconds: float = 1.0
    max_retry_attempts: int = 3
    enable_adaptive_scaling: bool = True

    # Development settings
    debug_mode: bool = False
    profile_execution: bool = False
    validate_inputs: bool = True

    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_config()
        if isinstance(self.log_file_path, str):
            self.log_file_path = Path(self.log_file_path)

    def _validate_config(self) -> None:
        """Validate configuration values."""
        if self.max_workers is not None and self.max_workers <= 0:
            raise ConfigurationError("max_workers must be positive")

        if self.task_timeout_seconds <= 0:
            raise ConfigurationError("task_timeout_seconds must be positive")

        if self.progress_refresh_rate <= 0:
            raise ConfigurationError("progress_refresh_rate must be positive")

        if self.batch_size <= 0:
            raise ConfigurationError("batch_size must be positive")

        if self.max_retry_attempts < 0:
            raise ConfigurationError("max_retry_attempts must be non-negative")

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "SimulationConfig":
        """
        Load configuration from a file.

        Args:
            config_path: Path to JSON or YAML configuration file

        Returns:
            SimulationConfig instance

        Raises:
            ConfigurationError: If file cannot be loaded or parsed
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path) as f:
                if config_path.suffix.lower() in [".yml", ".yaml"]:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == ".json":
                    config_data = json.load(f)
                else:
                    raise ConfigurationError(
                        f"Unsupported config file format: {config_path.suffix}",
                    )

            # Convert enums from string values
            if "execution_mode" in config_data:
                config_data["execution_mode"] = ExecutionMode(
                    config_data["execution_mode"],
                )
            if "default_task_priority" in config_data:
                config_data["default_task_priority"] = Priority[
                    config_data["default_task_priority"]
                ]
            if "log_level" in config_data:
                config_data["log_level"] = LogLevel(config_data["log_level"])

            return cls(**config_data)

        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ConfigurationError(f"Failed to parse configuration file: {e}")
        except (TypeError, KeyError) as e:
            raise ConfigurationError(f"Invalid configuration parameters: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a serializable dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result

    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """
        Save configuration to a file.

        Args:
            config_path: Path where to save the configuration (e.g., 'config.yml')
        """
        config_path = Path(config_path)
        config_dict = self.to_dict()

        with open(config_path, "w") as f:
            if config_path.suffix.lower() in [".yml", ".yaml"]:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            else:
                json.dump(config_dict, f, indent=2)

    def merge(self, other: "SimulationConfig") -> "SimulationConfig":
        """
        Merge with another configuration, with 'other' taking precedence.

        Args:
            other: The configuration to merge with.

        Returns:
            A new, merged SimulationConfig instance.
        """
        merged_dict = self.to_dict()
        merged_dict.update(other.to_dict())
        return SimulationConfig(**merged_dict)


# Default configuration instance
DEFAULT_CONFIG = SimulationConfig()
