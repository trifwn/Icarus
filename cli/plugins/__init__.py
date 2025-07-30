"""
ICARUS CLI Plugin System

This module provides a comprehensive plugin architecture for extending the ICARUS CLI
with custom functionality, integrations, and specialized tools.

The plugin system supports:
- Dynamic plugin discovery and loading
- Secure plugin execution with sandboxing
- Dependency management and version compatibility
- Plugin lifecycle management (install, update, uninstall)
- Comprehensive API for plugin development
"""

from .api import IcarusPlugin
from .api import PluginAPI
from .discovery import PluginDiscovery
from .exceptions import PluginError
from .exceptions import PluginLoadError
from .exceptions import PluginSecurityError
from .manager import PluginManager
from .models import PluginConfig
from .models import PluginInfo
from .models import PluginStatus
from .security import PluginSecurity
from .security import SecurityValidator

__all__ = [
    "PluginAPI",
    "IcarusPlugin",
    "PluginManager",
    "PluginDiscovery",
    "PluginSecurity",
    "SecurityValidator",
    "PluginInfo",
    "PluginConfig",
    "PluginStatus",
    "PluginError",
    "PluginLoadError",
    "PluginSecurityError",
]
