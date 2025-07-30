"""Unified Configuration Management System

This module provides a streamlined, unified configuration system that combines
the functionality of the previous config.py and settings.py modules into a single,
focused implementation.
"""

import json
import logging
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import yaml


class ConfigScope(Enum):
    """Scope levels for configuration."""

    GLOBAL = "global"
    WORKSPACE = "workspace"
    PROJECT = "project"
    SESSION = "session"


@dataclass
class UIConfig:
    """UI-specific configuration settings."""

    theme: str = "default"
    color_scheme: str = "dark"
    layout_style: str = "modern"
    show_session_info: bool = True
    show_progress_bar: bool = True
    animation_speed: str = "normal"
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
class WorkspaceConfig:
    """Workspace configuration settings."""

    name: str = "Default Workspace"
    data_directory: str = "./Data"
    results_directory: str = "./Results"
    templates_directory: str = "./Templates"
    auto_save_workflows: bool = True
    max_concurrent_workflows: int = 2


@dataclass
class UserPreferences:
    """User-specific preferences."""

    startup_screen: str = "dashboard"
    show_welcome: bool = True
    auto_save_interval: int = 300
    enable_notifications: bool = True
    max_memory_usage: int = 512
    max_cpu_cores: int = 0  # 0 = auto-detect


class UnifiedConfigManager:
    """Unified configuration management system that combines config and settings functionality."""

    def __init__(self, config_dir: str = "~/.icarus/config"):
        self.config_dir = Path(config_dir).expanduser()
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Configuration files
        self.global_config_file = self.config_dir / "config.json"
        self.user_config_file = self.config_dir / "user_config.json"
        self.workspaces_dir = Path("~/.icarus/workspaces").expanduser()
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)

        # Configuration sections
        self.ui_config = UIConfig()
        self.analysis_config = AnalysisConfig()
        self.workspace_config = WorkspaceConfig()
        self.user_preferences = UserPreferences()

        # Additional settings
        self.custom_settings: Dict[str, Any] = {}

        # Current context
        self.current_workspace: Optional[str] = None

        # Change tracking
        self._change_callbacks: List[Callable] = []
        self._dirty_flags: Dict[str, bool] = {}

        self.logger = logging.getLogger(__name__)

    def register_change_callback(self, callback: Callable) -> None:
        """Register a callback for configuration changes."""
        self._change_callbacks.append(callback)

    def _notify_changes(self, scope: str, config_type: str) -> None:
        """Notify registered callbacks of configuration changes."""
        for callback in self._change_callbacks:
            try:
                callback(scope, config_type)
            except Exception as e:
                self.logger.error(f"Error in configuration change callback: {e}")

    def _mark_dirty(self, config_type: str) -> None:
        """Mark configuration as dirty (needing save)."""
        self._dirty_flags[config_type] = True

    async def load_config(self) -> None:
        """Load configuration from files."""
        # Load default config
        await self._load_default_config()

        # Load user config (overrides defaults)
        await self._load_user_config()

        # Load workspace config if available
        if self.current_workspace:
            await self._load_workspace_config()

        self.logger.info("Configuration loaded successfully")

    async def _load_default_config(self) -> None:
        """Load default configuration."""
        try:
            if self.global_config_file.exists():
                with open(self.global_config_file) as f:
                    data = json.load(f)

                # Update configuration sections
                if "ui" in data:
                    self.ui_config = UIConfig(**data["ui"])
                if "analysis" in data:
                    self.analysis_config = AnalysisConfig(**data["analysis"])
                if "user" in data:
                    self.user_preferences = UserPreferences(**data["user"])
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

    async def _load_workspace_config(self) -> None:
        """Load workspace configuration."""
        if not self.current_workspace:
            return

        workspace_dir = self.workspaces_dir / self.current_workspace
        settings_file = workspace_dir / "config.json"

        if settings_file.exists():
            try:
                with open(settings_file) as f:
                    data = json.load(f)
                    if "workspace" in data:
                        self.workspace_config = WorkspaceConfig(**data["workspace"])
            except Exception as e:
                self.logger.warning(f"Failed to load workspace config: {e}")

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

        if "user" in user_data:
            for key, value in user_data["user"].items():
                if hasattr(self.user_preferences, key):
                    setattr(self.user_preferences, key, value)

        if "custom" in user_data:
            self.custom_settings.update(user_data["custom"])

    async def _create_default_config(self) -> None:
        """Create default configuration file."""
        default_config = {
            "ui": asdict(self.ui_config),
            "analysis": asdict(self.analysis_config),
            "user": asdict(self.user_preferences),
            "custom": {},
        }

        try:
            with open(self.global_config_file, "w") as f:
                json.dump(default_config, f, indent=2)

            self.logger.info("Created default configuration file")

        except Exception as e:
            self.logger.error(f"Failed to create default config: {e}")

    async def save_config(self) -> None:
        """Save current configuration."""
        # Save user config
        await self._save_user_config()

        # Save workspace config if available
        if self.current_workspace:
            await self._save_workspace_config()

    async def _save_user_config(self) -> None:
        """Save user configuration."""
        user_config = {
            "ui": asdict(self.ui_config),
            "analysis": asdict(self.analysis_config),
            "user": asdict(self.user_preferences),
            "custom": self.custom_settings,
        }

        try:
            with open(self.user_config_file, "w") as f:
                json.dump(user_config, f, indent=2)

            self._dirty_flags["ui"] = False
            self._dirty_flags["analysis"] = False
            self._dirty_flags["user"] = False
            self._dirty_flags["custom"] = False

            self.logger.debug("User configuration saved")

        except Exception as e:
            self.logger.error(f"Failed to save user config: {e}")

    async def _save_workspace_config(self) -> None:
        """Save workspace configuration."""
        if not self.current_workspace:
            return

        workspace_dir = self.workspaces_dir / self.current_workspace
        workspace_dir.mkdir(parents=True, exist_ok=True)
        settings_file = workspace_dir / "config.json"

        workspace_config = {
            "workspace": asdict(self.workspace_config),
        }

        try:
            with open(settings_file, "w") as f:
                json.dump(workspace_config, f, indent=2)

            self._dirty_flags["workspace"] = False
            self.logger.debug("Workspace configuration saved")

        except Exception as e:
            self.logger.error(f"Failed to save workspace config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        # Check custom settings first
        if key in self.custom_settings:
            return self.custom_settings[key]

        # Check configuration sections
        for config_obj in [
            self.ui_config,
            self.analysis_config,
            self.workspace_config,
            self.user_preferences,
        ]:
            if hasattr(config_obj, key):
                return getattr(config_obj, key)

        return default

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        # Try to set in appropriate configuration section
        for config_obj, section_name in [
            (self.ui_config, "ui"),
            (self.analysis_config, "analysis"),
            (self.workspace_config, "workspace"),
            (self.user_preferences, "user"),
        ]:
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
                self._mark_dirty(section_name)
                self._notify_changes(ConfigScope.GLOBAL.value, section_name)
                return

        # If not found in sections, add to custom settings
        self.custom_settings[key] = value
        self._mark_dirty("custom")
        self._notify_changes(ConfigScope.GLOBAL.value, "custom")

    # Workspace management
    def create_workspace(self, name: str, **kwargs) -> bool:
        """Create a new workspace."""
        workspace_dir = self.workspaces_dir / name
        if workspace_dir.exists():
            return False

        workspace_dir.mkdir(parents=True)

        # Create workspace config
        workspace_config = WorkspaceConfig(name=name, **kwargs)
        self.workspace_config = workspace_config
        self.current_workspace = name

        # Save workspace config
        workspace_config_data = {
            "workspace": asdict(workspace_config),
        }

        try:
            settings_file = workspace_dir / "config.json"
            with open(settings_file, "w") as f:
                json.dump(workspace_config_data, f, indent=2)

            self.logger.info(f"Created workspace: {name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create workspace: {e}")
            return False

    def switch_workspace(self, name: str) -> bool:
        """Switch to a different workspace."""
        workspace_dir = self.workspaces_dir / name
        if not workspace_dir.exists():
            return False

        # Save current workspace if dirty
        if self._dirty_flags.get("workspace", False) and self.current_workspace:
            asyncio.create_task(self._save_workspace_config())

        # Set new workspace
        self.current_workspace = name

        # Load new workspace config
        asyncio.create_task(self._load_workspace_config())

        self._notify_changes(ConfigScope.WORKSPACE.value, "workspace")
        return True

    def list_workspaces(self) -> List[str]:
        """List available workspaces."""
        return [d.name for d in self.workspaces_dir.iterdir() if d.is_dir()]

    # Configuration export/import
    def export_config(self, filepath: str, format: str = "json") -> bool:
        """Export configuration to file."""
        config_data = {
            "ui": asdict(self.ui_config),
            "analysis": asdict(self.analysis_config),
            "user": asdict(self.user_preferences),
            "workspace": asdict(self.workspace_config),
            "custom": self.custom_settings,
            "_metadata": {
                "exported_at": datetime.now().isoformat(),
                "version": "2.0.0",
                "current_workspace": self.current_workspace,
            },
        }

        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)

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

    def import_config(self, filepath: str, merge: bool = True) -> bool:
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

            # Import configuration
            if merge:
                self._merge_config(data)
            else:
                # Replace configuration
                if "ui" in data:
                    self.ui_config = UIConfig(**data["ui"])
                if "analysis" in data:
                    self.analysis_config = AnalysisConfig(**data["analysis"])
                if "user" in data:
                    self.user_preferences = UserPreferences(**data["user"])
                if "workspace" in data and self.current_workspace:
                    self.workspace_config = WorkspaceConfig(**data["workspace"])
                if "custom" in data:
                    self.custom_settings = data["custom"]

            # Mark all as dirty
            self._mark_dirty("ui")
            self._mark_dirty("analysis")
            self._mark_dirty("user")
            self._mark_dirty("workspace")
            self._mark_dirty("custom")

            # Notify changes
            self._notify_changes(ConfigScope.GLOBAL.value, "all")

            self.logger.info(f"Configuration imported from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to import config: {e}")
            return False

    # Reset functionality
    async def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self.ui_config = UIConfig()
        self.analysis_config = AnalysisConfig()
        self.workspace_config = WorkspaceConfig()
        self.user_preferences = UserPreferences()
        self.custom_settings.clear()

        # Mark all as dirty
        self._mark_dirty("ui")
        self._mark_dirty("analysis")
        self._mark_dirty("user")
        self._mark_dirty("workspace")
        self._mark_dirty("custom")

        # Save changes
        await self.save_config()

        # Notify changes
        self._notify_changes(ConfigScope.GLOBAL.value, "all")

        self.logger.info("Configuration reset to defaults")

    # Validation
    def validate_config(self) -> Dict[str, List[str]]:
        """Validate configuration and return any issues."""
        issues = {
            "ui": [],
            "analysis": [],
            "workspace": [],
            "user": [],
        }

        # Validate UI config
        if not self.ui_config.theme:
            issues["ui"].append("Theme name cannot be empty")

        # Validate analysis config
        if self.analysis_config.max_workers < 1:
            issues["analysis"].append("Max workers must be at least 1")
        if self.analysis_config.timeout_seconds < 10:
            issues["analysis"].append("Timeout must be at least 10 seconds")

        # Validate workspace config
        if self.workspace_config.max_concurrent_workflows < 1:
            issues["workspace"].append("Max concurrent workflows must be at least 1")

        # Validate user preferences
        if self.user_preferences.auto_save_interval < 30:
            issues["user"].append("Auto-save interval must be at least 30 seconds")
        if self.user_preferences.max_memory_usage < 128:
            issues["user"].append("Maximum memory usage must be at least 128 MB")

        return issues

    # Convenience getters
    def get_ui_config(self) -> UIConfig:
        """Get UI configuration."""
        return self.ui_config

    def get_analysis_config(self) -> AnalysisConfig:
        """Get analysis configuration."""
        return self.analysis_config

    def get_workspace_config(self) -> WorkspaceConfig:
        """Get workspace configuration."""
        return self.workspace_config

    def get_user_preferences(self) -> UserPreferences:
        """Get user preferences."""
        return self.user_preferences

    def get_database_path(self) -> str:
        """Get database path with expansion."""
        path = Path(self.workspace_config.data_directory).expanduser()
        return str(path)

    def get_templates_dir(self) -> str:
        """Get template directory with expansion."""
        path = Path(self.workspace_config.templates_directory).expanduser()
        return str(path)


# Global configuration manager instance
config_manager: Optional[UnifiedConfigManager] = None


def get_config_manager() -> UnifiedConfigManager:
    """Get or create the global configuration manager instance."""
    global config_manager
    if config_manager is None:
        config_manager = UnifiedConfigManager()
    return config_manager
