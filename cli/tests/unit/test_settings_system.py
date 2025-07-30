"""Test Settings System

This module provides comprehensive tests for the settings and personalization system.
"""

import json
import tempfile
from pathlib import Path

from cli.core.settings import SettingsManager
from cli.core.settings import SettingsScope
from cli.core.settings_integration import SettingsIntegration
from cli.core.ui import ThemeManager


class MockApp:
    """Mock application for testing."""

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


async def test_settings_manager():
    """Test the settings manager functionality."""
    print("Testing Settings Manager...")

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        settings_manager = SettingsManager(temp_dir)

        # Test theme settings
        print("  Testing theme settings...")
        theme_settings = settings_manager.get_theme_settings()
        assert theme_settings.theme_name == "aerospace"

        settings_manager.update_theme_settings(
            theme_name="scientific",
            color_scheme="light",
        )
        updated_theme = settings_manager.get_theme_settings()
        assert updated_theme.theme_name == "scientific"
        assert updated_theme.color_scheme == "light"

        # Test user preferences
        print("  Testing user preferences...")
        user_prefs = settings_manager.get_user_preferences()
        assert user_prefs.startup_screen == "dashboard"

        settings_manager.update_user_preferences(
            name="Test User",
            startup_screen="analysis",
        )
        updated_prefs = settings_manager.get_user_preferences()
        assert updated_prefs.name == "Test User"
        assert updated_prefs.startup_screen == "analysis"

        # Test workspace management
        print("  Testing workspace management...")
        assert settings_manager.create_workspace(
            "test_workspace",
            description="Test workspace",
        )
        workspaces = settings_manager.list_workspaces()
        assert "test_workspace" in workspaces

        assert settings_manager.switch_workspace("test_workspace")
        assert settings_manager.current_workspace == "test_workspace"

        # Test project management
        print("  Testing project management...")
        assert settings_manager.create_project(
            "test_project",
            description="Test project",
        )
        projects = settings_manager.list_projects()
        assert "test_project" in projects

        assert settings_manager.switch_project("test_project")
        assert settings_manager.current_project == "test_project"

        # Test settings persistence
        print("  Testing settings persistence...")
        settings_manager.save_all_settings()

        # Create new manager and load settings
        new_manager = SettingsManager(temp_dir)
        new_manager.load_all_settings()

        # Verify settings were loaded correctly
        loaded_theme = new_manager.get_theme_settings()
        assert loaded_theme.theme_name == "scientific"
        assert loaded_theme.color_scheme == "light"

        loaded_prefs = new_manager.get_user_preferences()
        assert loaded_prefs.name == "Test User"
        assert loaded_prefs.startup_screen == "analysis"

        print("  ✓ Settings Manager tests passed")


async def test_import_export():
    """Test settings import/export functionality."""
    print("Testing Import/Export...")

    with tempfile.TemporaryDirectory() as temp_dir:
        settings_manager = SettingsManager(temp_dir)

        # Configure some settings
        settings_manager.update_theme_settings(
            theme_name="scientific",
            animations_enabled=False,
        )
        settings_manager.update_user_preferences(
            name="Export Test",
            max_memory_usage=2048,
        )

        # Test export
        export_file = Path(temp_dir) / "export_test.json"
        success = settings_manager.export_settings(
            str(export_file),
            SettingsScope.GLOBAL,
        )
        assert success
        assert export_file.exists()

        # Verify export content
        with open(export_file) as f:
            exported_data = json.load(f)

        assert "theme" in exported_data
        assert exported_data["theme"]["theme_name"] == "scientific"
        assert exported_data["theme"]["animations_enabled"] == False

        assert "user" in exported_data
        assert exported_data["user"]["name"] == "Export Test"
        assert exported_data["user"]["max_memory_usage"] == 2048

        # Test import
        new_manager = SettingsManager(temp_dir + "_import")
        success = new_manager.import_settings(str(export_file))
        assert success

        # Verify imported settings
        imported_theme = new_manager.get_theme_settings()
        assert imported_theme.theme_name == "scientific"
        assert imported_theme.animations_enabled == False

        imported_prefs = new_manager.get_user_preferences()
        assert imported_prefs.name == "Export Test"
        assert imported_prefs.max_memory_usage == 2048

        print("  ✓ Import/Export tests passed")


async def test_backup_restore():
    """Test backup and restore functionality."""
    print("Testing Backup/Restore...")

    with tempfile.TemporaryDirectory() as temp_dir:
        settings_manager = SettingsManager(temp_dir)

        # Configure settings
        settings_manager.update_theme_settings(theme_name="aerospace", sidebar_width=40)
        settings_manager.update_user_preferences(name="Backup Test", cache_size=512)

        # Create backup
        backup_name = settings_manager.create_backup("test_backup")
        assert backup_name == "test_backup"

        # Verify backup exists
        backups = settings_manager.list_backups()
        backup_names = [b["name"] for b in backups]
        assert "test_backup" in backup_names

        # Modify settings
        settings_manager.update_theme_settings(
            theme_name="scientific",
            sidebar_width=60,
        )
        settings_manager.update_user_preferences(name="Modified", cache_size=1024)

        # Restore backup
        success = settings_manager.restore_backup("test_backup")
        assert success

        # Verify settings were restored
        restored_theme = settings_manager.get_theme_settings()
        assert restored_theme.theme_name == "aerospace"
        assert restored_theme.sidebar_width == 40

        restored_prefs = settings_manager.get_user_preferences()
        assert restored_prefs.name == "Backup Test"
        assert restored_prefs.cache_size == 512

        # Test backup cleanup
        # Create multiple backups
        for i in range(15):
            settings_manager.create_backup(f"backup_{i}")

        # Cleanup old backups
        deleted_count = settings_manager.cleanup_old_backups(5)
        assert deleted_count > 0

        # Verify only 5 backups remain
        remaining_backups = settings_manager.list_backups()
        assert len(remaining_backups) <= 6  # 5 + original test_backup

        print("  ✓ Backup/Restore tests passed")


async def test_settings_validation():
    """Test settings validation."""
    print("Testing Settings Validation...")

    with tempfile.TemporaryDirectory() as temp_dir:
        settings_manager = SettingsManager(temp_dir)

        # Test valid settings
        issues = settings_manager.validate_settings()
        assert len(issues) == 0  # Should be no issues with defaults

        # Create invalid settings
        settings_manager.update_theme_settings(sidebar_width=5)  # Too small
        settings_manager.update_theme_settings(transparency_level=1.5)  # Too high
        settings_manager.update_user_preferences(auto_save_interval=10)  # Too small
        settings_manager.update_user_preferences(max_memory_usage=50)  # Too small

        # Validate and check for issues
        issues = settings_manager.validate_settings()
        assert len(issues) > 0

        # Check specific issues
        theme_issues = issues.get("theme", [])
        user_issues = issues.get("user", [])

        assert any("Sidebar width" in issue for issue in theme_issues)
        assert any("Transparency level" in issue for issue in theme_issues)
        assert any("Auto-save interval" in issue for issue in user_issues)
        assert any("Maximum memory usage" in issue for issue in user_issues)

        print("  ✓ Settings Validation tests passed")


async def test_settings_integration():
    """Test settings integration with application."""
    print("Testing Settings Integration...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock app and managers
        app = MockApp()
        settings_manager = SettingsManager(temp_dir)
        theme_manager = ThemeManager()

        # Create integration
        integration = SettingsIntegration(app, settings_manager, theme_manager)

        # Initialize settings
        success = await integration.initialize_settings()
        assert success

        # Test theme application
        settings_manager.update_theme_settings(theme_name="scientific")
        # Theme should be applied to theme manager
        assert theme_manager.current_theme.value == "scientific"

        # Test user preferences application
        settings_manager.update_user_preferences(
            startup_screen="workflow",
            enable_notifications=True,
            notification_sound=False,
            notification_duration=10,
            max_memory_usage=1024,
            max_cpu_cores=4,
            cache_size=256,
        )

        # Manually trigger settings application since the callback might not work in test
        integration._apply_user_preferences()

        # Check if app received the settings
        assert app.startup_screen == "workflow"
        assert app.notifications_config == {
            "enabled": True,
            "sound": False,
            "duration": 10,
        }
        assert app.performance_config == {
            "max_memory": 1024,
            "max_cpu_cores": 4,
            "cache_size": 256,
        }

        # Test workspace settings application
        settings_manager.update_workspace_settings(
            default_solver_2d="xfoil",
            default_solver_3d="avl",
            data_directory="./TestData",
            results_directory="./TestResults",
        )

        # Manually trigger workspace settings application
        integration._apply_workspace_settings()

        assert app.default_solvers == {"solver_2d": "xfoil", "solver_3d": "avl"}

        # Test theme presets
        presets = integration.get_theme_presets()
        assert len(presets) > 0
        assert any(preset["id"] == "aerospace_dark" for preset in presets)

        success = integration.apply_theme_preset("aerospace_dark")
        assert success

        print("  ✓ Settings Integration tests passed")


async def test_theme_presets():
    """Test theme preset functionality."""
    print("Testing Theme Presets...")

    with tempfile.TemporaryDirectory() as temp_dir:
        settings_manager = SettingsManager(temp_dir)

        # Test applying presets
        presets = ["aerospace_dark", "aerospace_light", "scientific", "classic"]

        for preset in presets:
            success = settings_manager.apply_theme_preset(preset)
            assert success

            theme_settings = settings_manager.get_theme_settings()
            # Verify theme was changed (specific values depend on preset)
            assert theme_settings.theme_name in ["aerospace", "scientific", "default"]

        print("  ✓ Theme Presets tests passed")


async def run_all_tests():
    """Run all settings system tests."""
    print("Running Settings System Tests...\n")

    try:
        await test_settings_manager()
        await test_import_export()
        await test_backup_restore()
        await test_settings_validation()
        await test_settings_integration()
        await test_theme_presets()

        print("\n✅ All Settings System tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
