"""Settings Integration Module

This module provides integration between the settings system and the main application,
including live theme updates, settings validation, and configuration management.
"""

import logging
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from .settings import SettingsManager
from .settings import SettingsScope
from .settings import ThemeSettings
from .ui import Theme
from .ui import ThemeManager


class SettingsIntegration:
    """Integrates settings system with the main application."""

    def __init__(
        self,
        app,
        settings_manager: SettingsManager,
        theme_manager: ThemeManager,
    ):
        self.app = app
        self.settings_manager = settings_manager
        self.theme_manager = theme_manager
        self.logger = logging.getLogger(__name__)

        # Callbacks for settings changes
        self._theme_callbacks: List[Callable] = []
        self._settings_callbacks: List[Callable] = []

        # Register for settings changes
        self.settings_manager.register_change_callback(self._on_settings_changed)

        # Initialize with current settings
        self._apply_current_settings()

    def register_theme_callback(self, callback: Callable) -> None:
        """Register a callback for theme changes."""
        self._theme_callbacks.append(callback)

    def register_settings_callback(self, callback: Callable) -> None:
        """Register a callback for general settings changes."""
        self._settings_callbacks.append(callback)

    def _on_settings_changed(self, scope: str, settings_type: str) -> None:
        """Handle settings changes."""
        try:
            if settings_type == "theme":
                self._apply_theme_settings()
                self._notify_theme_callbacks()

            self._notify_settings_callbacks(scope, settings_type)

        except Exception as e:
            self.logger.error(f"Error handling settings change: {e}")

    def _apply_current_settings(self) -> None:
        """Apply current settings to the application."""
        self._apply_theme_settings()
        self._apply_user_preferences()
        self._apply_workspace_settings()

    def _apply_theme_settings(self) -> None:
        """Apply theme settings to the theme manager."""
        theme_settings = self.settings_manager.get_theme_settings()

        # Map theme name to Theme enum
        theme_map = {
            "aerospace": Theme.AEROSPACE,
            "scientific": Theme.SCIENTIFIC,
            "default": Theme.DEFAULT,
            "classic": Theme.DEFAULT,
        }

        theme = theme_map.get(theme_settings.theme_name, Theme.DEFAULT)
        self.theme_manager.set_theme(theme)

        # Apply additional theme customizations
        if hasattr(self.app, "apply_theme_settings"):
            self.app.apply_theme_settings(theme_settings)

    def _apply_user_preferences(self) -> None:
        """Apply user preferences to the application."""
        prefs = self.settings_manager.get_user_preferences()

        # Apply startup screen preference
        if hasattr(self.app, "set_startup_screen"):
            self.app.set_startup_screen(prefs.startup_screen)

        # Apply notification settings
        if hasattr(self.app, "configure_notifications"):
            self.app.configure_notifications(
                enabled=prefs.enable_notifications,
                sound=prefs.notification_sound,
                duration=prefs.notification_duration,
            )

        # Apply performance settings
        if hasattr(self.app, "configure_performance"):
            self.app.configure_performance(
                max_memory=prefs.max_memory_usage,
                max_cpu_cores=prefs.max_cpu_cores,
                cache_size=prefs.cache_size,
            )

    def _apply_workspace_settings(self) -> None:
        """Apply workspace settings to the application."""
        workspace = self.settings_manager.get_workspace_settings()

        # Apply default solver settings
        if hasattr(self.app, "set_default_solvers"):
            self.app.set_default_solvers(
                solver_2d=workspace.default_solver_2d,
                solver_3d=workspace.default_solver_3d,
            )

        # Apply data directories
        if hasattr(self.app, "set_data_directories"):
            self.app.set_data_directories(
                data_dir=workspace.data_directory,
                results_dir=workspace.results_directory,
                templates_dir=workspace.templates_directory,
                cache_dir=workspace.cache_directory,
            )

    def _notify_theme_callbacks(self) -> None:
        """Notify theme change callbacks."""
        for callback in self._theme_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"Error in theme callback: {e}")

    def _notify_settings_callbacks(self, scope: str, settings_type: str) -> None:
        """Notify settings change callbacks."""
        for callback in self._settings_callbacks:
            try:
                callback(scope, settings_type)
            except Exception as e:
                self.logger.error(f"Error in settings callback: {e}")

    async def initialize_settings(self) -> bool:
        """Initialize settings system."""
        try:
            # Load all settings
            self.settings_manager.load_all_settings()

            # Apply settings to application
            self._apply_current_settings()

            # Validate settings
            issues = self.settings_manager.validate_settings()
            if issues:
                self.logger.warning(f"Settings validation issues: {issues}")
                # Could show a notification to user about issues

            self.logger.info("Settings system initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize settings: {e}")
            return False

    async def save_settings_on_exit(self) -> None:
        """Save settings when application exits."""
        try:
            self.settings_manager.save_all_settings()
            self.logger.info("Settings saved on exit")
        except Exception as e:
            self.logger.error(f"Failed to save settings on exit: {e}")

    def get_live_preview_settings(self) -> Dict[str, Any]:
        """Get settings for live preview."""
        return {
            "theme": self.settings_manager.get_theme_settings(),
            "user": self.settings_manager.get_user_preferences(),
            "workspace": self.settings_manager.get_workspace_settings(),
            "project": self.settings_manager.get_project_settings(),
        }

    def apply_preview_settings(self, preview_settings: Dict[str, Any]) -> None:
        """Apply preview settings temporarily."""
        # This would temporarily apply settings for preview
        # without saving them permanently
        if "theme" in preview_settings:
            theme_data = preview_settings["theme"]
            # Apply theme preview
            self._apply_theme_preview(theme_data)

    def _apply_theme_preview(self, theme_data: Dict[str, Any]) -> None:
        """Apply theme settings for preview."""
        # Create temporary theme settings
        temp_theme = ThemeSettings(**theme_data)

        # Apply to theme manager temporarily
        theme_map = {
            "aerospace": Theme.AEROSPACE,
            "scientific": Theme.SCIENTIFIC,
            "default": Theme.DEFAULT,
            "classic": Theme.DEFAULT,
        }

        theme = theme_map.get(temp_theme.theme_name, Theme.DEFAULT)
        self.theme_manager.set_theme(theme)

        # Notify callbacks for preview update
        self._notify_theme_callbacks()

    def revert_preview_settings(self) -> None:
        """Revert preview settings to saved settings."""
        self._apply_current_settings()

    def export_settings_profile(self, name: str, filepath: str) -> bool:
        """Export a settings profile."""
        try:
            # Add profile metadata
            profile_data = self.settings_manager.get_all_settings()
            profile_data["_profile"] = {
                "name": name,
                "description": f"Settings profile: {name}",
                "created_at": self.settings_manager.get_current_timestamp(),
            }

            return self.settings_manager.export_settings(
                filepath,
                SettingsScope.SESSION,
            )

        except Exception as e:
            self.logger.error(f"Failed to export settings profile: {e}")
            return False

    def import_settings_profile(
        self,
        filepath: str,
        apply_immediately: bool = True,
    ) -> bool:
        """Import a settings profile."""
        try:
            success = self.settings_manager.import_settings(filepath, merge=False)

            if success and apply_immediately:
                self._apply_current_settings()

            return success

        except Exception as e:
            self.logger.error(f"Failed to import settings profile: {e}")
            return False

    def create_settings_backup(self, name: str = None) -> Optional[str]:
        """Create a settings backup."""
        try:
            backup_name = self.settings_manager.create_backup(name)
            self.logger.info(f"Settings backup created: {backup_name}")
            return backup_name
        except Exception as e:
            self.logger.error(f"Failed to create settings backup: {e}")
            return None

    def restore_settings_backup(self, backup_name: str) -> bool:
        """Restore settings from backup."""
        try:
            success = self.settings_manager.restore_backup(backup_name)

            if success:
                self._apply_current_settings()
                self.logger.info(f"Settings restored from backup: {backup_name}")

            return success

        except Exception as e:
            self.logger.error(f"Failed to restore settings backup: {e}")
            return False

    def get_settings_summary(self) -> Dict[str, Any]:
        """Get a summary of current settings."""
        return self.settings_manager.get_settings_summary()

    def validate_and_fix_settings(self) -> Dict[str, List[str]]:
        """Validate settings and attempt to fix issues."""
        issues = self.settings_manager.validate_settings()

        # Attempt to fix common issues
        fixed_issues = []

        for category, category_issues in issues.items():
            for issue in category_issues:
                if "Sidebar width" in issue:
                    # Fix sidebar width
                    self.settings_manager.update_theme_settings(sidebar_width=30)
                    fixed_issues.append(f"Fixed sidebar width in {category}")

                elif "Transparency level" in issue:
                    # Fix transparency level
                    self.settings_manager.update_theme_settings(transparency_level=0.95)
                    fixed_issues.append(f"Fixed transparency level in {category}")

                elif "Auto-save interval" in issue:
                    # Fix auto-save interval
                    self.settings_manager.update_user_preferences(
                        auto_save_interval=300,
                    )
                    fixed_issues.append(f"Fixed auto-save interval in {category}")

                elif "Maximum memory usage" in issue:
                    # Fix memory usage
                    self.settings_manager.update_user_preferences(max_memory_usage=512)
                    fixed_issues.append(f"Fixed memory usage in {category}")

        if fixed_issues:
            self.logger.info(f"Fixed settings issues: {fixed_issues}")
            # Re-validate after fixes
            remaining_issues = self.settings_manager.validate_settings()
            return remaining_issues

        return issues

    def get_theme_presets(self) -> List[Dict[str, Any]]:
        """Get available theme presets."""
        return [
            {
                "name": "Aerospace Dark",
                "id": "aerospace_dark",
                "description": "Dark theme optimized for aerospace applications",
                "preview": {
                    "primary": "#00bfff",
                    "background": "#000000",
                    "text": "#ffffff",
                },
            },
            {
                "name": "Aerospace Light",
                "id": "aerospace_light",
                "description": "Light theme optimized for aerospace applications",
                "preview": {
                    "primary": "#0066cc",
                    "background": "#ffffff",
                    "text": "#000000",
                },
            },
            {
                "name": "Scientific",
                "id": "scientific",
                "description": "High-contrast theme for scientific applications",
                "preview": {
                    "primary": "#00ff00",
                    "background": "#000000",
                    "text": "#ffffff",
                },
            },
            {
                "name": "Classic",
                "id": "classic",
                "description": "Traditional terminal theme",
                "preview": {
                    "primary": "#ffffff",
                    "background": "#000000",
                    "text": "#ffffff",
                },
            },
        ]

    def apply_theme_preset(self, preset_id: str) -> bool:
        """Apply a theme preset."""
        try:
            success = self.settings_manager.apply_theme_preset(preset_id)
            if success:
                self._apply_theme_settings()
                self.logger.info(f"Applied theme preset: {preset_id}")
            return success
        except Exception as e:
            self.logger.error(f"Failed to apply theme preset: {e}")
            return False


class LiveSettingsPreview:
    """Provides live preview functionality for settings changes."""

    def __init__(self, settings_integration: SettingsIntegration):
        self.settings_integration = settings_integration
        self.original_settings: Optional[Dict[str, Any]] = None
        self.preview_active = False

    def start_preview(self) -> None:
        """Start live preview mode."""
        if not self.preview_active:
            # Save original settings
            self.original_settings = (
                self.settings_integration.get_live_preview_settings()
            )
            self.preview_active = True

    def update_preview(self, settings_changes: Dict[str, Any]) -> None:
        """Update preview with new settings."""
        if self.preview_active:
            # Apply preview settings
            self.settings_integration.apply_preview_settings(settings_changes)

    def commit_preview(self) -> None:
        """Commit preview changes as permanent settings."""
        if self.preview_active:
            # Settings are already applied, just mark as committed
            self.preview_active = False
            self.original_settings = None

    def cancel_preview(self) -> None:
        """Cancel preview and revert to original settings."""
        if self.preview_active and self.original_settings:
            # Revert to original settings
            self.settings_integration.apply_preview_settings(self.original_settings)
            self.preview_active = False
            self.original_settings = None

    def is_preview_active(self) -> bool:
        """Check if preview mode is active."""
        return self.preview_active


# Global settings integration instance
settings_integration: Optional[SettingsIntegration] = None


def initialize_settings_integration(
    app,
    settings_manager: SettingsManager,
    theme_manager: ThemeManager,
) -> SettingsIntegration:
    """Initialize the global settings integration."""
    global settings_integration
    settings_integration = SettingsIntegration(app, settings_manager, theme_manager)
    return settings_integration


def get_settings_integration() -> Optional[SettingsIntegration]:
    """Get the global settings integration instance."""
    return settings_integration
