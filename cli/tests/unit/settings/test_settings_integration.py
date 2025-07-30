import pytest
from cli.core.settings_integration import SettingsIntegration
from cli.tests.unit.settings.test_settings_manager import MockApp

def test_settings_integration_theme():
    app = MockApp()
    integration = SettingsIntegration(app)
    integration.apply_theme_settings({"theme": "aerospace"})
    assert app.theme_settings_applied == {"theme": "aerospace"}

def test_settings_integration_notifications():
    app = MockApp()
    integration = SettingsIntegration(app)
    integration.apply_notification_settings(True, "ding", 5)
    assert app.notifications_config == {"enabled": True, "sound": "ding", "duration": 5}
