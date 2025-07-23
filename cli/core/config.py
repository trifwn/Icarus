"""Configuration Management

This module provides configuration management for the ICARUS CLI,
including default settings, user preferences, and configuration persistence.
"""

import json
import logging
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Dict

import yaml


@dataclass
class UIConfig:
    """UI-specific configuration settings."""

    theme: str = "default"
    show_session_info: bool = True
    show_progress_bar: bool = True
    auto_refresh: bool = True
    animation_speed: str = "normal"  # slow, normal, fast
    terminal_title: bool = True


@dataclass
class AnalysisConfig:
    """Analysis-specific configuration settings."""

    default_solver_2d: str = "xfoil"
    default_solver_3d: str = "avl"
    auto_save_results: bool = True
    parallel_processing: bool = True
    max_workers: int = 4
    timeout_seconds: int = 300


@dataclass
class DatabaseConfig:
    """Database configuration settings."""

    path: str = "./Data"
    auto_backup: bool = True
    backup_interval_hours: int = 24
    max_backups: int = 10


@dataclass
class WorkflowConfig:
    """Workflow configuration settings."""

    auto_save_workflows: bool = True
    default_template_dir: str = "~/.icarus/templates"
    max_concurrent_workflows: int = 2
    step_timeout_seconds: int = 600


class ConfigManager:
    """Manages application configuration and settings."""

    def __init__(self, config_dir: str = "~/.icarus/config"):
        self.config_dir = Path(config_dir).expanduser()
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.config_file = self.config_dir / "config.json"
        self.user_config_file = self.config_dir / "user_config.json"

        # Configuration sections
        self.ui_config = UIConfig()
        self.analysis_config = AnalysisConfig()
        self.database_config = DatabaseConfig()
        self.workflow_config = WorkflowConfig()

        # Additional settings
        self.custom_settings: Dict[str, Any] = {}

        self.logger = logging.getLogger(__name__)

    async def load_config(self) -> None:
        """Load configuration from files."""
        # Load default config
        await self._load_default_config()

        # Load user config (overrides defaults)
        await self._load_user_config()

        self.logger.info("Configuration loaded successfully")

    async def _load_default_config(self) -> None:
        """Load default configuration."""
        try:
            if self.config_file.exists():
                with open(self.config_file) as f:
                    data = json.load(f)

                # Update configuration sections
                if "ui" in data:
                    self.ui_config = UIConfig(**data["ui"])
                if "analysis" in data:
                    self.analysis_config = AnalysisConfig(**data["analysis"])
                if "database" in data:
                    self.database_config = DatabaseConfig(**data["database"])
                if "workflow" in data:
                    self.workflow_config = WorkflowConfig(**data["workflow"])
                if "custom" in data:
                    self.custom_settings = data["custom"]

        except Exception as e:
            self.logger.warning(f"Failed to load default config: {e}")
            await self._create_default_config()

    async def _load_user_config(self) -> None:
        """Load user-specific configuration."""
        try:
            if self.user_config_file.exists():
                with open(self.user_config_file) as f:
                    data = json.load(f)

                # Merge user settings with defaults
                self._merge_config(data)

        except Exception as e:
            self.logger.warning(f"Failed to load user config: {e}")

    def _merge_config(self, user_data: Dict[str, Any]) -> None:
        """Merge user configuration with defaults."""
        if "ui" in user_data:
            for key, value in user_data["ui"].items():
                if hasattr(self.ui_config, key):
                    setattr(self.ui_config, key, value)

        if "analysis" in user_data:
            for key, value in user_data["analysis"].items():
                if hasattr(self.analysis_config, key):
                    setattr(self.analysis_config, key, value)

        if "database" in user_data:
            for key, value in user_data["database"].items():
                if hasattr(self.database_config, key):
                    setattr(self.database_config, key, value)

        if "workflow" in user_data:
            for key, value in user_data["workflow"].items():
                if hasattr(self.workflow_config, key):
                    setattr(self.workflow_config, key, value)

        if "custom" in user_data:
            self.custom_settings.update(user_data["custom"])

    async def _create_default_config(self) -> None:
        """Create default configuration file."""
        default_config = {
            "ui": asdict(self.ui_config),
            "analysis": asdict(self.analysis_config),
            "database": asdict(self.database_config),
            "workflow": asdict(self.workflow_config),
            "custom": {},
        }

        try:
            with open(self.config_file, "w") as f:
                json.dump(default_config, f, indent=2)

            self.logger.info("Created default configuration file")

        except Exception as e:
            self.logger.error(f"Failed to create default config: {e}")

    async def save_config(self) -> None:
        """Save current configuration to user config file."""
        user_config = {
            "ui": asdict(self.ui_config),
            "analysis": asdict(self.analysis_config),
            "database": asdict(self.database_config),
            "workflow": asdict(self.workflow_config),
            "custom": self.custom_settings,
        }

        try:
            with open(self.user_config_file, "w") as f:
                json.dump(user_config, f, indent=2)

            self.logger.debug("Configuration saved")

        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        # Check custom settings first
        if key in self.custom_settings:
            return self.custom_settings[key]

        # Check configuration sections
        for config_obj in [
            self.ui_config,
            self.analysis_config,
            self.database_config,
            self.workflow_config,
        ]:
            if hasattr(config_obj, key):
                return getattr(config_obj, key)

        return default

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        # Try to set in appropriate configuration section
        for config_obj in [
            self.ui_config,
            self.analysis_config,
            self.database_config,
            self.workflow_config,
        ]:
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
                return

        # If not found in sections, add to custom settings
        self.custom_settings[key] = value

    def get_ui_config(self) -> UIConfig:
        """Get UI configuration."""
        return self.ui_config

    def get_analysis_config(self) -> AnalysisConfig:
        """Get analysis configuration."""
        return self.analysis_config

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        return self.database_config

    def get_workflow_config(self) -> WorkflowConfig:
        """Get workflow configuration."""
        return self.workflow_config

    def get_database_path(self) -> str:
        """Get database path with expansion."""
        path = Path(self.database_config.path).expanduser()
        return str(path)

    def get_template_dir(self) -> str:
        """Get template directory with expansion."""
        path = Path(self.workflow_config.default_template_dir).expanduser()
        return str(path)

    async def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self.ui_config = UIConfig()
        self.analysis_config = AnalysisConfig()
        self.database_config = DatabaseConfig()
        self.workflow_config = WorkflowConfig()
        self.custom_settings.clear()

        await self.save_config()
        self.logger.info("Configuration reset to defaults")

    def export_config(self, filepath: str, format: str = "json") -> bool:
        """Export configuration to file."""
        config_data = {
            "ui": asdict(self.ui_config),
            "analysis": asdict(self.analysis_config),
            "database": asdict(self.database_config),
            "workflow": asdict(self.workflow_config),
            "custom": self.custom_settings,
        }

        try:
            path = Path(filepath)

            if format.lower() == "json":
                with open(path, "w") as f:
                    json.dump(config_data, f, indent=2)
            elif format.lower() == "yaml":
                with open(path, "w") as f:
                    yaml.dump(config_data, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")

            self.logger.info(f"Configuration exported to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export config: {e}")
            return False

    def import_config(self, filepath: str) -> bool:
        """Import configuration from file."""
        try:
            path = Path(filepath)

            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {filepath}")

            if path.suffix.lower() == ".json":
                with open(path) as f:
                    data = json.load(f)
            elif path.suffix.lower() in [".yaml", ".yml"]:
                with open(path) as f:
                    data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

            # Merge imported configuration
            self._merge_config(data)

            self.logger.info(f"Configuration imported from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to import config: {e}")
            return False

    def load_config_file(self, filepath: str) -> bool:
        """Load configuration from a specific file path.

        This is a synchronous version of import_config that can be called
        during application initialization.
        """
        try:
            path = Path(filepath)

            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {filepath}")

            if path.suffix.lower() == ".json":
                with open(path) as f:
                    data = json.load(f)
            elif path.suffix.lower() in [".yaml", ".yml"]:
                with open(path) as f:
                    data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

            # Merge imported configuration
            self._merge_config(data)

            self.logger.info(f"Configuration loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load config file: {e}")
            return False
