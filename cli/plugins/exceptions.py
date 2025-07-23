"""
Plugin system exceptions.
"""


class PluginError(Exception):
    """Base exception for plugin system errors."""

    def __init__(self, message: str, plugin_name: str = None, cause: Exception = None):
        super().__init__(message)
        self.plugin_name = plugin_name
        self.cause = cause


class PluginLoadError(PluginError):
    """Exception raised when plugin loading fails."""

    pass


class PluginSecurityError(PluginError):
    """Exception raised when plugin security validation fails."""

    pass


class PluginDependencyError(PluginError):
    """Exception raised when plugin dependencies cannot be resolved."""

    pass


class PluginVersionError(PluginError):
    """Exception raised when plugin version is incompatible."""

    pass


class PluginConfigError(PluginError):
    """Exception raised when plugin configuration is invalid."""

    pass


class PluginPermissionError(PluginError):
    """Exception raised when plugin lacks required permissions."""

    pass


class PluginInstallError(PluginError):
    """Exception raised when plugin installation fails."""

    pass


class PluginUpdateError(PluginError):
    """Exception raised when plugin update fails."""

    pass


class PluginUninstallError(PluginError):
    """Exception raised when plugin uninstallation fails."""

    pass
