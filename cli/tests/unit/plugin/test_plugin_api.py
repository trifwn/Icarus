import pytest
from unittest.mock import Mock

def test_plugin_api_creation():
    from cli.plugins.api import PluginAPI, PluginContext
    context = PluginContext(
        app_instance=Mock(),
        session_manager=Mock(),
        config_manager=Mock(),
        event_system=Mock(),
        ui_manager=Mock(),
        data_manager=Mock(),
        logger=Mock(),
    )
    api = PluginAPI(context)
    assert api is not None
    assert api.context == context

def test_plugin_base_class():
    from cli.plugins.api import IcarusPlugin
    from cli.plugins.models import PluginManifest, PluginAuthor, PluginType, SecurityLevel, PluginVersion
    class TestPlugin(IcarusPlugin):
        def get_manifest(self):
            return PluginManifest(
                name="test_plugin",
                version=PluginVersion(1, 0, 0),
                description="Test plugin",
                author=PluginAuthor("Test Author"),
                plugin_type=PluginType.UTILITY,
                security_level=SecurityLevel.SAFE,
                main_module="test_plugin",
                main_class="TestPlugin",
            )
        def on_activate(self):
            pass
    plugin = TestPlugin()
    assert plugin is not None
    manifest = plugin.get_manifest()
    assert manifest.name == "test_plugin"
    assert manifest.plugin_type == PluginType.UTILITY
