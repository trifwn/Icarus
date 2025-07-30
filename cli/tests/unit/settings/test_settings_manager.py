
import pytest
from cli.core.settings import SettingsManager, ThemeSettings, UserPreferences, WorkspaceSettings
 
class MockApp:
    """Mock application for settings integration and manager tests."""
    def __init__(self):
        self.theme_settings_applied = None
        self.startup_screen = None
        self.notifications_config = None
        self.performance_config = None
        self.default_solvers = None
        self.data_directories = None

    def apply_theme_settings(self, theme_settings):
        self.theme_settings_applied = theme_settings

    def set_startup_screen(self, screen):
        self.startup_screen = screen

    def configure_notifications(self, enabled, sound, duration):
        self.notifications_config = {
            "enabled": enabled,
            "sound": sound,
            "duration": duration,
        }

    def configure_performance(self, max_memory, max_cpu_cores, cache_size):
        self.performance_config = {
            "max_memory": max_memory,
            "max_cpu_cores": max_cpu_cores,
            "cache_size": cache_size,
        }

    def set_default_solvers(self, solver_2d, solver_3d):
        self.default_solvers = {"solver_2d": solver_2d, "solver_3d": solver_3d}

    def set_data_directories(self, data_dir, results_dir, templates_dir, cache_dir):
        self.data_directories = {
            "data_dir": data_dir,
            "results_dir": results_dir,
            "templates_dir": templates_dir,
            "cache_dir": cache_dir,
        }

def test_update_theme_settings_and_getters():
    manager = SettingsManager()
    # Default theme settings should be ThemeSettings
    default_theme = manager.get_theme_settings()
    assert isinstance(default_theme, ThemeSettings)
    # Update theme settings
    manager.update_theme_settings(theme_name="scientific", color_scheme="light")
    theme = manager.get_theme_settings()
    assert theme.theme_name == "scientific"
    assert theme.color_scheme == "light"

import pytest
from cli.core.settings import SettingsManager, ThemeSettings, UserPreferences, WorkspaceSettings

class MockApp:
    """Mock application for settings integration tests."""
    def __init__(self):
        self.theme_settings_applied = None
        self.startup_screen = None
        self.notifications_config = None
        self.performance_config = None
        self.default_solvers = None
        self.data_directories = None

    def apply_theme_settings(self, theme_settings):
        self.theme_settings_applied = theme_settings

    def set_startup_screen(self, screen):
        self.startup_screen = screen

    def configure_notifications(self, enabled, sound, duration):
        self.notifications_config = {
            "enabled": enabled,
            "sound": sound,
            "duration": duration,
        }

    def configure_performance(self, max_memory, max_cpu_cores, cache_size):
        self.performance_config = {
            "max_memory": max_memory,
            "max_cpu_cores": max_cpu_cores,
            "cache_size": cache_size,
        }

    def set_default_solvers(self, solver_2d, solver_3d):
        self.default_solvers = {"solver_2d": solver_2d, "solver_3d": solver_3d}

    def set_data_directories(self, data_dir, results_dir, templates_dir, cache_dir):
        self.data_directories = {
            "data_dir": data_dir,
            "results_dir": results_dir,
            "templates_dir": templates_dir,
            "cache_dir": cache_dir,
        }

def test_update_user_preferences_and_getters():
    manager = SettingsManager()
    # Default user preferences should be UserPreferences
    default_prefs = manager.get_user_preferences()
    assert isinstance(default_prefs, UserPreferences)
    # Update user preferences
    manager.update_user_preferences(enable_notifications=False, notification_sound=True, notification_duration=10)
    prefs = manager.get_user_preferences()
    assert prefs.enable_notifications is False
    assert prefs.notification_sound is True
    assert prefs.notification_duration == 10

    manager = SettingsManager()
    # Default theme settings should be ThemeSettings
    default_theme = manager.get_theme_settings()
    assert isinstance(default_theme, ThemeSettings)
    # Update theme settings
    manager.update_theme_settings(theme_name="scientific", color_scheme="light")
    theme = manager.get_theme_settings()
    assert theme.theme_name == "scientific"
    assert theme.color_scheme == "light"

def test_update_user_preferences_and_getters():
    manager = SettingsManager()
    # Default user preferences should be UserPreferences
    default_prefs = manager.get_user_preferences()
    assert isinstance(default_prefs, UserPreferences)
    # Update user preferences
    manager.update_user_preferences(enable_notifications=False, notification_sound=True, notification_duration=10)
    prefs = manager.get_user_preferences()
    assert prefs.enable_notifications is False
    assert prefs.notification_sound is True
    assert prefs.notification_duration == 10

def test_update_workspace_settings_and_getters(tmp_path):
    # Use a temporary base directory to avoid filesystem side effects
    base_dir = tmp_path / "settings_base"
    manager = SettingsManager(base_dir=str(base_dir))
    # Default workspace settings should be WorkspaceSettings
    default_ws = manager.get_workspace_settings()
    assert isinstance(default_ws, WorkspaceSettings)
    # Update workspace settings
    manager.update_workspace_settings(name="TestWS", description="Desc", default_solver_2d="foils", max_concurrent_workflows=5)
    ws = manager.get_workspace_settings()
    assert ws.name == "TestWS"
    assert ws.description == "Desc"
    assert ws.default_solver_2d == "foils"
    assert ws.max_concurrent_workflows == 5
