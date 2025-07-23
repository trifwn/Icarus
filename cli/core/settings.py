"""Advanced Settings and Personalization System

This module provides comprehensive settings management, theme customization,
workspace configurations, and settings backup/restore functionality.
"""

import json
import logging
import shutil
from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import yaml


class SettingsScope(Enum):
    """Scope levels for settings."""

    GLOBAL = "global"
    WORKSPACE = "workspace"
    PROJECT = "project"
    SESSION = "session"


class SettingsFormat(Enum):
    """Supported settings file formats."""

    JSON = "json"
    YAML = "yaml"
    TOML = "toml"


@dataclass
class ThemeSettings:
    """Theme and visual customization settings."""

    # Base theme
    theme_name: str = "aerospace"
    color_scheme: str = "dark"

    # Layout preferences
    layout_style: str = "modern"  # modern, classic, compact
    sidebar_width: int = 30
    header_height: int = 3
    footer_height: int = 3

    # Visual effects
    animations_enabled: bool = True
    animation_speed: str = "normal"  # slow, normal, fast
    transitions_enabled: bool = True
    transparency_level: float = 0.95

    # Typography
    font_size: str = "normal"  # small, normal, large
    font_family: str = "default"
    line_spacing: float = 1.0

    # Colors (custom overrides)
    custom_colors: Dict[str, str] = field(default_factory=dict)

    # UI Elements
    show_icons: bool = True
    show_tooltips: bool = True
    show_status_bar: bool = True
    show_breadcrumbs: bool = True

    # Terminal specific
    terminal_title: bool = True
    bell_enabled: bool = False
    cursor_style: str = "block"  # block, underline, bar


@dataclass
class WorkspaceSettings:
    """Workspace-specific settings."""

    # Workspace info
    name: str = "Default Workspace"
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Paths
    data_directory: str = "./Data"
    results_directory: str = "./Results"
    templates_directory: str = "./Templates"
    cache_directory: str = "./Cache"

    # Default analysis settings
    default_solver_2d: str = "xfoil"
    default_solver_3d: str = "avl"
    default_reynolds: float = 1000000.0
    default_mach: float = 0.0
    default_angles: str = "0:15:16"

    # Workflow preferences
    auto_save_workflows: bool = True
    workflow_validation: bool = True
    max_concurrent_workflows: int = 2

    # Data management
    auto_backup: bool = True
    backup_interval_hours: int = 24
    max_backups: int = 10
    compress_backups: bool = True


@dataclass
class ProjectSettings:
    """Project-specific settings."""

    # Project info
    name: str = "Untitled Project"
    description: str = ""
    author: str = ""
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Analysis preferences
    preferred_solvers: List[str] = field(default_factory=list)
    analysis_precision: str = "normal"  # low, normal, high
    convergence_criteria: Dict[str, float] = field(default_factory=dict)

    # Visualization
    plot_style: str = "publication"  # draft, presentation, publication
    figure_format: str = "png"
    figure_dpi: int = 300

    # Collaboration
    sharing_enabled: bool = False
    collaborators: List[str] = field(default_factory=list)
    permissions: Dict[str, str] = field(default_factory=dict)


@dataclass
class UserPreferences:
    """User-specific preferences."""

    # Personal info
    name: str = ""
    email: str = ""
    organization: str = ""

    # Interface preferences
    startup_screen: str = "dashboard"
    show_welcome: bool = True
    auto_save_interval: int = 300  # seconds

    # Notifications
    enable_notifications: bool = True
    notification_sound: bool = False
    notification_duration: int = 5  # seconds

    # Performance
    max_memory_usage: int = 1024  # MB
    max_cpu_cores: int = 0  # 0 = auto-detect
    cache_size: int = 256  # MB

    # Privacy
    analytics_enabled: bool = True
    crash_reporting: bool = True
    usage_statistics: bool = True


class SettingsManager:
    """Comprehensive settings management system."""

    def __init__(self, base_dir: str = "~/.icarus"):
        self.base_dir = Path(base_dir).expanduser()
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Settings directories
        self.global_dir = self.base_dir / "settings"
        self.workspaces_dir = self.base_dir / "workspaces"
        self.backups_dir = self.base_dir / "backups"

        for dir_path in [self.global_dir, self.workspaces_dir, self.backups_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Current settings
        self.theme_settings = ThemeSettings()
        self.user_preferences = UserPreferences()
        self.workspace_settings = WorkspaceSettings()
        self.project_settings = ProjectSettings()

        # Current context
        self.current_workspace: Optional[str] = None
        self.current_project: Optional[str] = None

        # Change tracking
        self._change_callbacks: List[Callable] = []
        self._dirty_flags: Dict[str, bool] = {}

        self.logger = logging.getLogger(__name__)

    def register_change_callback(self, callback: Callable) -> None:
        """Register a callback for settings changes."""
        self._change_callbacks.append(callback)

    def _notify_changes(self, scope: str, settings_type: str) -> None:
        """Notify registered callbacks of settings changes."""
        for callback in self._change_callbacks:
            try:
                callback(scope, settings_type)
            except Exception as e:
                self.logger.error(f"Error in settings change callback: {e}")

    def _mark_dirty(self, settings_type: str) -> None:
        """Mark settings as dirty (needing save)."""
        self._dirty_flags[settings_type] = True

    # Theme Settings Management
    def get_theme_settings(self) -> ThemeSettings:
        """Get current theme settings."""
        return self.theme_settings

    def update_theme_settings(self, **kwargs) -> None:
        """Update theme settings."""
        for key, value in kwargs.items():
            if hasattr(self.theme_settings, key):
                setattr(self.theme_settings, key, value)

        self._mark_dirty("theme")
        self._notify_changes("global", "theme")

    def apply_theme_preset(self, preset_name: str) -> bool:
        """Apply a predefined theme preset."""
        presets = self._get_theme_presets()
        if preset_name in presets:
            preset = presets[preset_name]
            for key, value in preset.items():
                if hasattr(self.theme_settings, key):
                    setattr(self.theme_settings, key, value)

            self._mark_dirty("theme")
            self._notify_changes("global", "theme")
            return True
        return False

    def _get_theme_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get available theme presets."""
        return {
            "aerospace_dark": {
                "theme_name": "aerospace",
                "color_scheme": "dark",
                "layout_style": "modern",
                "animations_enabled": True,
                "show_icons": True,
            },
            "aerospace_light": {
                "theme_name": "aerospace",
                "color_scheme": "light",
                "layout_style": "modern",
                "animations_enabled": True,
                "show_icons": True,
            },
            "scientific": {
                "theme_name": "scientific",
                "color_scheme": "dark",
                "layout_style": "compact",
                "animations_enabled": False,
                "show_icons": False,
            },
            "classic": {
                "theme_name": "default",
                "color_scheme": "dark",
                "layout_style": "classic",
                "animations_enabled": False,
                "show_icons": True,
            },
        }

    # User Preferences Management
    def get_user_preferences(self) -> UserPreferences:
        """Get current user preferences."""
        return self.user_preferences

    def update_user_preferences(self, **kwargs) -> None:
        """Update user preferences."""
        for key, value in kwargs.items():
            if hasattr(self.user_preferences, key):
                setattr(self.user_preferences, key, value)

        self._mark_dirty("user")
        self._notify_changes("global", "user")

    # Workspace Settings Management
    def get_workspace_settings(self) -> WorkspaceSettings:
        """Get current workspace settings."""
        return self.workspace_settings

    def update_workspace_settings(self, **kwargs) -> None:
        """Update workspace settings."""
        for key, value in kwargs.items():
            if hasattr(self.workspace_settings, key):
                setattr(self.workspace_settings, key, value)

        self._mark_dirty("workspace")
        self._notify_changes("workspace", "workspace")

    def create_workspace(self, name: str, **kwargs) -> bool:
        """Create a new workspace."""
        workspace_dir = self.workspaces_dir / name
        if workspace_dir.exists():
            return False

        workspace_dir.mkdir(parents=True)

        # Create workspace settings
        workspace_settings = WorkspaceSettings(name=name, **kwargs)

        # Save workspace settings
        settings_file = workspace_dir / "settings.json"
        with open(settings_file, "w") as f:
            json.dump(asdict(workspace_settings), f, indent=2)

        self.logger.info(f"Created workspace: {name}")
        return True

    def switch_workspace(self, name: str) -> bool:
        """Switch to a different workspace."""
        workspace_dir = self.workspaces_dir / name
        if not workspace_dir.exists():
            return False

        # Save current workspace if dirty
        if self._dirty_flags.get("workspace", False):
            self.save_workspace_settings()

        # Load new workspace settings
        settings_file = workspace_dir / "settings.json"
        if settings_file.exists():
            with open(settings_file) as f:
                data = json.load(f)
                self.workspace_settings = WorkspaceSettings(**data)
        else:
            self.workspace_settings = WorkspaceSettings(name=name)

        self.current_workspace = name
        self._notify_changes("workspace", "workspace")
        return True

    def list_workspaces(self) -> List[str]:
        """List available workspaces."""
        return [d.name for d in self.workspaces_dir.iterdir() if d.is_dir()]

    def delete_workspace(self, name: str) -> bool:
        """Delete a workspace."""
        if name == self.current_workspace:
            return False  # Cannot delete current workspace

        workspace_dir = self.workspaces_dir / name
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
            self.logger.info(f"Deleted workspace: {name}")
            return True
        return False

    # Project Settings Management
    def get_project_settings(self) -> ProjectSettings:
        """Get current project settings."""
        return self.project_settings

    def update_project_settings(self, **kwargs) -> None:
        """Update project settings."""
        for key, value in kwargs.items():
            if hasattr(self.project_settings, key):
                setattr(self.project_settings, key, value)

        self._mark_dirty("project")
        self._notify_changes("project", "project")

    def create_project(self, name: str, workspace: str = None, **kwargs) -> bool:
        """Create a new project."""
        workspace = workspace or self.current_workspace
        if not workspace:
            return False

        workspace_dir = self.workspaces_dir / workspace
        projects_dir = workspace_dir / "projects"
        projects_dir.mkdir(exist_ok=True)

        project_dir = projects_dir / name
        if project_dir.exists():
            return False

        project_dir.mkdir(parents=True)

        # Create project settings
        project_settings = ProjectSettings(name=name, **kwargs)

        # Save project settings
        settings_file = project_dir / "settings.json"
        with open(settings_file, "w") as f:
            json.dump(asdict(project_settings), f, indent=2)

        self.logger.info(f"Created project: {name} in workspace: {workspace}")
        return True

    def switch_project(self, name: str, workspace: str = None) -> bool:
        """Switch to a different project."""
        workspace = workspace or self.current_workspace
        if not workspace:
            return False

        workspace_dir = self.workspaces_dir / workspace
        project_dir = workspace_dir / "projects" / name

        if not project_dir.exists():
            return False

        # Save current project if dirty
        if self._dirty_flags.get("project", False):
            self.save_project_settings()

        # Load new project settings
        settings_file = project_dir / "settings.json"
        if settings_file.exists():
            with open(settings_file) as f:
                data = json.load(f)
                self.project_settings = ProjectSettings(**data)
        else:
            self.project_settings = ProjectSettings(name=name)

        self.current_project = name
        self._notify_changes("project", "project")
        return True

    def list_projects(self, workspace: str = None) -> List[str]:
        """List projects in a workspace."""
        workspace = workspace or self.current_workspace
        if not workspace:
            return []

        workspace_dir = self.workspaces_dir / workspace
        projects_dir = workspace_dir / "projects"

        if not projects_dir.exists():
            return []

        return [d.name for d in projects_dir.iterdir() if d.is_dir()]

    # Settings Persistence
    def load_all_settings(self) -> None:
        """Load all settings from files."""
        self._load_global_settings()
        if self.current_workspace:
            self._load_workspace_settings()
        if self.current_project:
            self._load_project_settings()

    def save_all_settings(self) -> None:
        """Save all settings to files."""
        self._save_global_settings()
        if self.current_workspace:
            self.save_workspace_settings()
        if self.current_project:
            self.save_project_settings()

    def _load_global_settings(self) -> None:
        """Load global settings."""
        # Load theme settings
        theme_file = self.global_dir / "theme.json"
        if theme_file.exists():
            with open(theme_file) as f:
                data = json.load(f)
                self.theme_settings = ThemeSettings(**data)

        # Load user preferences
        user_file = self.global_dir / "user.json"
        if user_file.exists():
            with open(user_file) as f:
                data = json.load(f)
                self.user_preferences = UserPreferences(**data)

    def _save_global_settings(self) -> None:
        """Save global settings."""
        # Save theme settings
        if self._dirty_flags.get("theme", False):
            theme_file = self.global_dir / "theme.json"
            with open(theme_file, "w") as f:
                json.dump(asdict(self.theme_settings), f, indent=2)
            self._dirty_flags["theme"] = False

        # Save user preferences
        if self._dirty_flags.get("user", False):
            user_file = self.global_dir / "user.json"
            with open(user_file, "w") as f:
                json.dump(asdict(self.user_preferences), f, indent=2)
            self._dirty_flags["user"] = False

    def _load_workspace_settings(self) -> None:
        """Load workspace settings."""
        if not self.current_workspace:
            return

        workspace_dir = self.workspaces_dir / self.current_workspace
        settings_file = workspace_dir / "settings.json"

        if settings_file.exists():
            with open(settings_file) as f:
                data = json.load(f)
                self.workspace_settings = WorkspaceSettings(**data)

    def save_workspace_settings(self) -> None:
        """Save workspace settings."""
        if not self.current_workspace:
            return

        workspace_dir = self.workspaces_dir / self.current_workspace
        settings_file = workspace_dir / "settings.json"

        with open(settings_file, "w") as f:
            json.dump(asdict(self.workspace_settings), f, indent=2)

        self._dirty_flags["workspace"] = False

    def _load_project_settings(self) -> None:
        """Load project settings."""
        if not self.current_workspace or not self.current_project:
            return

        workspace_dir = self.workspaces_dir / self.current_workspace
        project_dir = workspace_dir / "projects" / self.current_project
        settings_file = project_dir / "settings.json"

        if settings_file.exists():
            with open(settings_file) as f:
                data = json.load(f)
                self.project_settings = ProjectSettings(**data)

    def save_project_settings(self) -> None:
        """Save project settings."""
        if not self.current_workspace or not self.current_project:
            return

        workspace_dir = self.workspaces_dir / self.current_workspace
        project_dir = workspace_dir / "projects" / self.current_project
        settings_file = project_dir / "settings.json"

        with open(settings_file, "w") as f:
            json.dump(asdict(self.project_settings), f, indent=2)

        self._dirty_flags["project"] = False

    # Import/Export Functionality
    def export_settings(
        self,
        filepath: str,
        scope: SettingsScope = SettingsScope.GLOBAL,
        format: SettingsFormat = SettingsFormat.JSON,
    ) -> bool:
        """Export settings to file."""
        try:
            export_data = {}

            if scope in [SettingsScope.GLOBAL, SettingsScope.SESSION]:
                export_data["theme"] = asdict(self.theme_settings)
                export_data["user"] = asdict(self.user_preferences)

            if scope in [SettingsScope.WORKSPACE, SettingsScope.SESSION]:
                if self.current_workspace:
                    export_data["workspace"] = asdict(self.workspace_settings)

            if scope in [SettingsScope.PROJECT, SettingsScope.SESSION]:
                if self.current_project:
                    export_data["project"] = asdict(self.project_settings)

            # Add metadata
            export_data["_metadata"] = {
                "exported_at": datetime.now().isoformat(),
                "scope": scope.value,
                "format": format.value,
                "version": "2.0.0",
                "current_workspace": self.current_workspace,
                "current_project": self.current_project,
            }

            # Write to file
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)

            if format == SettingsFormat.JSON:
                with open(path, "w") as f:
                    json.dump(export_data, f, indent=2)
            elif format == SettingsFormat.YAML:
                with open(path, "w") as f:
                    yaml.dump(export_data, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")

            self.logger.info(f"Settings exported to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export settings: {e}")
            return False

    def import_settings(self, filepath: str, merge: bool = True) -> bool:
        """Import settings from file."""
        try:
            path = Path(filepath)
            if not path.exists():
                raise FileNotFoundError(f"Settings file not found: {filepath}")

            # Determine format from extension
            if path.suffix.lower() == ".json":
                with open(path) as f:
                    data = json.load(f)
            elif path.suffix.lower() in [".yaml", ".yml"]:
                with open(path) as f:
                    data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

            # Validate metadata
            metadata = data.get("_metadata", {})
            if "version" in metadata:
                version = metadata["version"]
                if not self._is_compatible_version(version):
                    self.logger.warning(
                        f"Settings version {version} may not be compatible",
                    )

            # Import settings based on what's available
            if "theme" in data:
                if merge:
                    self._merge_theme_settings(data["theme"])
                else:
                    self.theme_settings = ThemeSettings(**data["theme"])
                self._mark_dirty("theme")

            if "user" in data:
                if merge:
                    self._merge_user_preferences(data["user"])
                else:
                    self.user_preferences = UserPreferences(**data["user"])
                self._mark_dirty("user")

            if "workspace" in data and self.current_workspace:
                if merge:
                    self._merge_workspace_settings(data["workspace"])
                else:
                    self.workspace_settings = WorkspaceSettings(**data["workspace"])
                self._mark_dirty("workspace")

            if "project" in data and self.current_project:
                if merge:
                    self._merge_project_settings(data["project"])
                else:
                    self.project_settings = ProjectSettings(**data["project"])
                self._mark_dirty("project")

            # Notify changes
            self._notify_changes("import", "all")

            self.logger.info(f"Settings imported from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to import settings: {e}")
            return False

    def _is_compatible_version(self, version: str) -> bool:
        """Check if settings version is compatible."""
        # Simple version compatibility check
        major_version = version.split(".")[0]
        return major_version == "2"

    def _merge_theme_settings(self, data: Dict[str, Any]) -> None:
        """Merge theme settings data."""
        for key, value in data.items():
            if hasattr(self.theme_settings, key):
                setattr(self.theme_settings, key, value)

    def _merge_user_preferences(self, data: Dict[str, Any]) -> None:
        """Merge user preferences data."""
        for key, value in data.items():
            if hasattr(self.user_preferences, key):
                setattr(self.user_preferences, key, value)

    def _merge_workspace_settings(self, data: Dict[str, Any]) -> None:
        """Merge workspace settings data."""
        for key, value in data.items():
            if hasattr(self.workspace_settings, key):
                setattr(self.workspace_settings, key, value)

    def _merge_project_settings(self, data: Dict[str, Any]) -> None:
        """Merge project settings data."""
        for key, value in data.items():
            if hasattr(self.project_settings, key):
                setattr(self.project_settings, key, value)

    # Backup and Restore Functionality
    def create_backup(self, name: str = None) -> str:
        """Create a backup of all settings."""
        if not name:
            name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        backup_dir = self.backups_dir / name
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Export all settings to backup
        backup_file = backup_dir / "settings.json"
        self.export_settings(str(backup_file), SettingsScope.SESSION)

        # Copy workspace and project directories if they exist
        if self.current_workspace:
            workspace_src = self.workspaces_dir / self.current_workspace
            workspace_dst = backup_dir / "workspace"
            if workspace_src.exists():
                shutil.copytree(workspace_src, workspace_dst)

        # Create backup metadata
        metadata = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "workspace": self.current_workspace,
            "project": self.current_project,
            "version": "2.0.0",
        }

        metadata_file = backup_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Backup created: {name}")
        return name

    def restore_backup(self, name: str) -> bool:
        """Restore settings from a backup."""
        try:
            backup_dir = self.backups_dir / name
            if not backup_dir.exists():
                raise FileNotFoundError(f"Backup not found: {name}")

            # Load backup metadata
            metadata_file = backup_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    self.logger.info(
                        f"Restoring backup from {metadata.get('created_at', 'unknown')}",
                    )

            # Restore settings
            settings_file = backup_dir / "settings.json"
            if settings_file.exists():
                self.import_settings(str(settings_file), merge=False)

            # Restore workspace if it exists in backup
            workspace_backup = backup_dir / "workspace"
            if workspace_backup.exists() and self.current_workspace:
                workspace_dir = self.workspaces_dir / self.current_workspace
                if workspace_dir.exists():
                    shutil.rmtree(workspace_dir)
                shutil.copytree(workspace_backup, workspace_dir)

            self.logger.info(f"Backup restored: {name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to restore backup: {e}")
            return False

    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups."""
        backups = []

        for backup_dir in self.backups_dir.iterdir():
            if backup_dir.is_dir():
                metadata_file = backup_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                            backups.append(metadata)
                    except Exception as e:
                        self.logger.warning(f"Failed to read backup metadata: {e}")
                        # Create basic metadata
                        backups.append(
                            {
                                "name": backup_dir.name,
                                "created_at": datetime.fromtimestamp(
                                    backup_dir.stat().st_mtime,
                                ).isoformat(),
                                "workspace": "unknown",
                                "project": "unknown",
                            },
                        )

        return sorted(backups, key=lambda x: x["created_at"], reverse=True)

    def delete_backup(self, name: str) -> bool:
        """Delete a backup."""
        try:
            backup_dir = self.backups_dir / name
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
                self.logger.info(f"Backup deleted: {name}")
                return True
            return False

        except Exception as e:
            self.logger.error(f"Failed to delete backup: {e}")
            return False

    def cleanup_old_backups(self, max_backups: int = 10) -> int:
        """Clean up old backups, keeping only the most recent ones."""
        backups = self.list_backups()

        if len(backups) <= max_backups:
            return 0

        # Delete oldest backups
        deleted_count = 0
        for backup in backups[max_backups:]:
            if self.delete_backup(backup["name"]):
                deleted_count += 1

        self.logger.info(f"Cleaned up {deleted_count} old backups")
        return deleted_count

    # Settings Validation and Reset
    def validate_settings(self) -> Dict[str, List[str]]:
        """Validate all settings and return any issues."""
        issues = {
            "theme": [],
            "user": [],
            "workspace": [],
            "project": [],
        }

        # Validate theme settings
        if (
            self.theme_settings.sidebar_width < 10
            or self.theme_settings.sidebar_width > 100
        ):
            issues["theme"].append("Sidebar width must be between 10 and 100")

        if (
            self.theme_settings.transparency_level < 0.0
            or self.theme_settings.transparency_level > 1.0
        ):
            issues["theme"].append("Transparency level must be between 0.0 and 1.0")

        # Validate user preferences
        if self.user_preferences.auto_save_interval < 30:
            issues["user"].append("Auto-save interval must be at least 30 seconds")

        if self.user_preferences.max_memory_usage < 128:
            issues["user"].append("Maximum memory usage must be at least 128 MB")

        # Validate workspace settings
        if self.workspace_settings.max_concurrent_workflows < 1:
            issues["workspace"].append(
                "Maximum concurrent workflows must be at least 1",
            )

        if self.workspace_settings.backup_interval_hours < 1:
            issues["workspace"].append("Backup interval must be at least 1 hour")

        # Remove empty issue lists
        return {k: v for k, v in issues.items() if v}

    def reset_to_defaults(self, scope: SettingsScope = SettingsScope.GLOBAL) -> None:
        """Reset settings to defaults."""
        if scope in [SettingsScope.GLOBAL, SettingsScope.SESSION]:
            self.theme_settings = ThemeSettings()
            self.user_preferences = UserPreferences()
            self._mark_dirty("theme")
            self._mark_dirty("user")

        if scope in [SettingsScope.WORKSPACE, SettingsScope.SESSION]:
            self.workspace_settings = WorkspaceSettings()
            self._mark_dirty("workspace")

        if scope in [SettingsScope.PROJECT, SettingsScope.SESSION]:
            self.project_settings = ProjectSettings()
            self._mark_dirty("project")

        self._notify_changes(scope.value, "reset")
        self.logger.info(f"Settings reset to defaults for scope: {scope.value}")

    # Context Managers
    @contextmanager
    def settings_transaction(self):
        """Context manager for atomic settings changes."""
        # Save current state
        original_theme = asdict(self.theme_settings)
        original_user = asdict(self.user_preferences)
        original_workspace = asdict(self.workspace_settings)
        original_project = asdict(self.project_settings)

        try:
            yield self
        except Exception:
            # Restore original state on error
            self.theme_settings = ThemeSettings(**original_theme)
            self.user_preferences = UserPreferences(**original_user)
            self.workspace_settings = WorkspaceSettings(**original_workspace)
            self.project_settings = ProjectSettings(**original_project)
            raise

    # Utility Methods
    def get_setting(self, key: str, scope: SettingsScope = None) -> Any:
        """Get a setting value by key."""
        if scope == SettingsScope.THEME or (
            scope is None and hasattr(self.theme_settings, key)
        ):
            return getattr(self.theme_settings, key, None)
        elif scope == SettingsScope.USER or (
            scope is None and hasattr(self.user_preferences, key)
        ):
            return getattr(self.user_preferences, key, None)
        elif scope == SettingsScope.WORKSPACE or (
            scope is None and hasattr(self.workspace_settings, key)
        ):
            return getattr(self.workspace_settings, key, None)
        elif scope == SettingsScope.PROJECT or (
            scope is None and hasattr(self.project_settings, key)
        ):
            return getattr(self.project_settings, key, None)

        return None

    def set_setting(self, key: str, value: Any, scope: SettingsScope = None) -> bool:
        """Set a setting value by key."""
        if scope == SettingsScope.THEME or (
            scope is None and hasattr(self.theme_settings, key)
        ):
            setattr(self.theme_settings, key, value)
            self._mark_dirty("theme")
            return True
        elif scope == SettingsScope.USER or (
            scope is None and hasattr(self.user_preferences, key)
        ):
            setattr(self.user_preferences, key, value)
            self._mark_dirty("user")
            return True
        elif scope == SettingsScope.WORKSPACE or (
            scope is None and hasattr(self.workspace_settings, key)
        ):
            setattr(self.workspace_settings, key, value)
            self._mark_dirty("workspace")
            return True
        elif scope == SettingsScope.PROJECT or (
            scope is None and hasattr(self.project_settings, key)
        ):
            setattr(self.project_settings, key, value)
            self._mark_dirty("project")
            return True

        return False

    def get_all_settings(self) -> Dict[str, Dict[str, Any]]:
        """Get all current settings."""
        return {
            "theme": asdict(self.theme_settings),
            "user": asdict(self.user_preferences),
            "workspace": asdict(self.workspace_settings),
            "project": asdict(self.project_settings),
            "context": {
                "current_workspace": self.current_workspace,
                "current_project": self.current_project,
            },
        }

    def get_settings_summary(self) -> Dict[str, Any]:
        """Get a summary of current settings."""
        return {
            "theme": {
                "name": self.theme_settings.theme_name,
                "color_scheme": self.theme_settings.color_scheme,
                "layout_style": self.theme_settings.layout_style,
            },
            "workspace": {
                "name": self.workspace_settings.name,
                "current": self.current_workspace,
            },
            "project": {
                "name": self.project_settings.name,
                "current": self.current_project,
            },
            "user": {
                "name": self.user_preferences.name,
                "startup_screen": self.user_preferences.startup_screen,
            },
        }


# Global settings manager instance
settings_manager = SettingsManager()
