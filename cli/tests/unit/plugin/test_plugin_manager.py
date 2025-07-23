import pytest
from unittest.mock import Mock

def test_manager_creation():
    from cli.plugins.manager import PluginManager
    app_context = Mock()
    manager = PluginManager(app_context)
    assert manager is not None
    assert manager.discovery is not None
    assert manager.security is not None
    assert manager.registry is not None

def test_plugin_discovery():
    from cli.plugins.manager import PluginManager
    app_context = Mock()
    manager = PluginManager(app_context)
    plugins = manager.discover_plugins()
    assert isinstance(plugins, list)

def test_plugin_status_summary():
    from cli.plugins.manager import PluginManager
    app_context = Mock()
    manager = PluginManager(app_context)
    summary = manager.get_plugin_status_summary()
    assert isinstance(summary, dict)
    from cli.plugins.models import PluginStatus
    assert all(status in summary for status in [s.value for s in PluginStatus])
