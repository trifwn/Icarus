"""
Plugin manager for ICARUS CLI plugin system.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from .api import IcarusPlugin
from .api import PluginAPI
from .api import PluginContext
from .discovery import PluginDiscovery
from .exceptions import PluginInstallError
from .exceptions import PluginLoadError
from .exceptions import PluginSecurityError
from .models import PluginConfig
from .models import PluginInfo
from .models import PluginRegistry
from .models import PluginStatus
from .models import PluginType
from .security import PluginSecurity


class PluginManager:
    """
    Main plugin manager that orchestrates plugin discovery, loading,
    security validation, and lifecycle management.
    """

    def __init__(self, app_context: Any, logger: Optional[logging.Logger] = None):
        self.app_context = app_context
        self.logger = logger or logging.getLogger(__name__)

        # Core components
        self.discovery = PluginDiscovery(logger)
        self.security = PluginSecurity(logger)
        self.registry = PluginRegistry()

        # Plugin APIs and instances
        self.plugin_apis: Dict[str, PluginAPI] = {}
        self.active_plugins: Dict[str, IcarusPlugin] = {}

        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}

        # Configuration
        self.config_file = Path.home() / ".icarus" / "plugins" / "config.json"
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self._load_configuration()

    def _load_configuration(self) -> None:
        """Load plugin configuration from file."""
        try:
            if self.config_file.exists():
                with open(self.config_file) as f:
                    config_data = json.load(f)

                # Load plugin configs
                for plugin_name, config_dict in config_data.get("plugins", {}).items():
                    self.registry.configs[plugin_name] = PluginConfig.from_dict(
                        config_dict,
                    )

                # Load trusted/blocked plugins
                self.security.trusted_plugins.update(
                    config_data.get("trusted_plugins", []),
                )
                self.security.blocked_plugins.update(
                    config_data.get("blocked_plugins", []),
                )

                self.logger.info("Plugin configuration loaded")
        except Exception as e:
            self.logger.error(f"Error loading plugin configuration: {e}")

    def _save_configuration(self) -> None:
        """Save plugin configuration to file."""
        try:
            config_data = {
                "plugins": {
                    name: config.to_dict()
                    for name, config in self.registry.configs.items()
                },
                "trusted_plugins": list(self.security.trusted_plugins),
                "blocked_plugins": list(self.security.blocked_plugins),
            }

            with open(self.config_file, "w") as f:
                json.dump(config_data, f, indent=2)

            self.logger.debug("Plugin configuration saved")
        except Exception as e:
            self.logger.error(f"Error saving plugin configuration: {e}")

    def discover_plugins(self) -> List[PluginInfo]:
        """
        Discover all available plugins.

        Returns:
            List of discovered plugins
        """
        self.logger.info("Starting plugin discovery")

        try:
            discovered = self.discovery.discover_plugins()

            # Add to registry
            for plugin_info in discovered:
                self.registry.add_plugin(plugin_info)

            self.registry.last_scan = datetime.now()

            self.logger.info(f"Discovered {len(discovered)} plugins")
            return discovered

        except Exception as e:
            self.logger.error(f"Plugin discovery failed: {e}")
            return []

    def load_plugin(self, plugin_id: str) -> bool:
        """
        Load a specific plugin.

        Args:
            plugin_id: Plugin identifier

        Returns:
            True if loaded successfully, False otherwise
        """
        plugin_info = self.registry.get_plugin(plugin_id)
        if not plugin_info:
            self.logger.error(f"Plugin not found: {plugin_id}")
            return False

        if plugin_info.is_loaded:
            self.logger.info(f"Plugin already loaded: {plugin_id}")
            return True

        try:
            # Security validation
            if not self.security.validate_plugin_security(plugin_info):
                raise PluginSecurityError(
                    f"Plugin failed security validation: {plugin_id}",
                )

            # Load plugin instance
            plugin_instance = self.discovery.load_plugin_instance(plugin_info)
            if not plugin_instance:
                raise PluginLoadError(f"Failed to load plugin instance: {plugin_id}")

            # Create plugin API
            plugin_api = self._create_plugin_api(plugin_info)

            # Initialize plugin
            plugin_instance.initialize(plugin_api)

            # Store references
            self.plugin_apis[plugin_id] = plugin_api
            self.active_plugins[plugin_id] = plugin_instance

            # Update status
            plugin_info.status = PluginStatus.LOADED
            plugin_info.loaded_at = datetime.now()
            plugin_info.instance = plugin_instance

            self.logger.info(f"Plugin loaded successfully: {plugin_id}")

            # Emit event
            self._emit_event("plugin_loaded", {"plugin_id": plugin_id})

            return True

        except Exception as e:
            self.logger.error(f"Failed to load plugin {plugin_id}: {e}")
            plugin_info.status = PluginStatus.ERROR
            plugin_info.last_error = str(e)
            return False

    def activate_plugin(self, plugin_id: str) -> bool:
        """
        Activate a loaded plugin.

        Args:
            plugin_id: Plugin identifier

        Returns:
            True if activated successfully, False otherwise
        """
        plugin_info = self.registry.get_plugin(plugin_id)
        if not plugin_info:
            self.logger.error(f"Plugin not found: {plugin_id}")
            return False

        if not plugin_info.is_loaded:
            # Try to load first
            if not self.load_plugin(plugin_id):
                return False

        if plugin_info.is_active:
            self.logger.info(f"Plugin already active: {plugin_id}")
            return True

        try:
            plugin_instance = self.active_plugins[plugin_id]
            plugin_instance.activate()

            # Update status
            plugin_info.status = PluginStatus.ACTIVE

            self.logger.info(f"Plugin activated: {plugin_id}")

            # Emit event
            self._emit_event("plugin_activated", {"plugin_id": plugin_id})

            return True

        except Exception as e:
            self.logger.error(f"Failed to activate plugin {plugin_id}: {e}")
            plugin_info.status = PluginStatus.ERROR
            plugin_info.last_error = str(e)
            return False

    def deactivate_plugin(self, plugin_id: str) -> bool:
        """
        Deactivate an active plugin.

        Args:
            plugin_id: Plugin identifier

        Returns:
            True if deactivated successfully, False otherwise
        """
        plugin_info = self.registry.get_plugin(plugin_id)
        if not plugin_info:
            self.logger.error(f"Plugin not found: {plugin_id}")
            return False

        if not plugin_info.is_active:
            self.logger.info(f"Plugin not active: {plugin_id}")
            return True

        try:
            plugin_instance = self.active_plugins[plugin_id]
            plugin_instance.deactivate()

            # Update status
            plugin_info.status = PluginStatus.LOADED

            self.logger.info(f"Plugin deactivated: {plugin_id}")

            # Emit event
            self._emit_event("plugin_deactivated", {"plugin_id": plugin_id})

            return True

        except Exception as e:
            self.logger.error(f"Failed to deactivate plugin {plugin_id}: {e}")
            plugin_info.last_error = str(e)
            return False

    def unload_plugin(self, plugin_id: str) -> bool:
        """
        Unload a plugin.

        Args:
            plugin_id: Plugin identifier

        Returns:
            True if unloaded successfully, False otherwise
        """
        plugin_info = self.registry.get_plugin(plugin_id)
        if not plugin_info:
            self.logger.error(f"Plugin not found: {plugin_id}")
            return False

        try:
            # Deactivate if active
            if plugin_info.is_active:
                self.deactivate_plugin(plugin_id)

            # Clean up references
            if plugin_id in self.active_plugins:
                del self.active_plugins[plugin_id]

            if plugin_id in self.plugin_apis:
                del self.plugin_apis[plugin_id]

            # Update status
            plugin_info.status = PluginStatus.DISCOVERED
            plugin_info.instance = None

            self.logger.info(f"Plugin unloaded: {plugin_id}")

            # Emit event
            self._emit_event("plugin_unloaded", {"plugin_id": plugin_id})

            return True

        except Exception as e:
            self.logger.error(f"Failed to unload plugin {plugin_id}: {e}")
            return False

    def install_plugin(self, source: str, force: bool = False) -> bool:
        """
        Install a plugin from a source.

        Args:
            source: Plugin source (file path, URL, etc.)
            force: Force installation even if plugin exists

        Returns:
            True if installed successfully, False otherwise
        """
        try:
            self.logger.info(f"Installing plugin from: {source}")

            # Determine source type and handle accordingly
            source_path = Path(source)

            if source_path.exists():
                return self._install_from_local_path(source_path, force)
            elif source.startswith(("http://", "https://")):
                return self._install_from_url(source, force)
            else:
                raise PluginInstallError(f"Invalid plugin source: {source}")

        except Exception as e:
            self.logger.error(f"Plugin installation failed: {e}")
            return False

    def _install_from_local_path(self, source_path: Path, force: bool) -> bool:
        """Install plugin from local path."""
        # Create plugins directory if it doesn't exist
        plugins_dir = Path.home() / ".icarus" / "plugins"
        plugins_dir.mkdir(parents=True, exist_ok=True)

        if source_path.is_file():
            # Single file plugin
            dest_path = plugins_dir / source_path.name
            if dest_path.exists() and not force:
                raise PluginInstallError(f"Plugin file already exists: {dest_path}")

            shutil.copy2(source_path, dest_path)

        elif source_path.is_dir():
            # Directory plugin
            dest_path = plugins_dir / source_path.name
            if dest_path.exists() and not force:
                raise PluginInstallError(
                    f"Plugin directory already exists: {dest_path}",
                )

            if dest_path.exists():
                shutil.rmtree(dest_path)

            shutil.copytree(source_path, dest_path)

        else:
            raise PluginInstallError(f"Invalid source path: {source_path}")

        # Refresh discovery to pick up new plugin
        self.discover_plugins()

        self.logger.info(f"Plugin installed successfully: {source_path.name}")
        return True

    def _install_from_url(self, url: str, force: bool) -> bool:
        """Install plugin from URL."""
        # This would implement downloading and installing from URL
        # For now, just raise not implemented
        raise NotImplementedError("URL installation not yet implemented")

    def uninstall_plugin(self, plugin_id: str) -> bool:
        """
        Uninstall a plugin.

        Args:
            plugin_id: Plugin identifier

        Returns:
            True if uninstalled successfully, False otherwise
        """
        plugin_info = self.registry.get_plugin(plugin_id)
        if not plugin_info:
            self.logger.error(f"Plugin not found: {plugin_id}")
            return False

        try:
            # Unload plugin first
            self.unload_plugin(plugin_id)

            # Remove plugin files
            plugin_path = Path(plugin_info.path)
            if plugin_path.exists():
                if plugin_path.is_file():
                    plugin_path.unlink()
                else:
                    shutil.rmtree(plugin_path)

            # Remove from registry
            self.registry.remove_plugin(plugin_id)

            # Save configuration
            self._save_configuration()

            self.logger.info(f"Plugin uninstalled: {plugin_id}")

            # Emit event
            self._emit_event("plugin_uninstalled", {"plugin_id": plugin_id})

            return True

        except Exception as e:
            self.logger.error(f"Failed to uninstall plugin {plugin_id}: {e}")
            return False

    def get_plugin_info(self, plugin_id: str) -> Optional[PluginInfo]:
        """Get plugin information."""
        return self.registry.get_plugin(plugin_id)

    def get_all_plugins(self) -> List[PluginInfo]:
        """Get all plugins."""
        return list(self.registry.plugins.values())

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInfo]:
        """Get plugins by type."""
        return self.registry.get_plugins_by_type(plugin_type)

    def get_active_plugins(self) -> List[PluginInfo]:
        """Get active plugins."""
        return self.registry.get_active_plugins()

    def configure_plugin(self, plugin_id: str, config: Dict[str, Any]) -> bool:
        """
        Configure a plugin.

        Args:
            plugin_id: Plugin identifier
            config: Configuration dictionary

        Returns:
            True if configured successfully, False otherwise
        """
        plugin_info = self.registry.get_plugin(plugin_id)
        if not plugin_info:
            self.logger.error(f"Plugin not found: {plugin_id}")
            return False

        try:
            # Update plugin config
            plugin_config = self.registry.configs.get(plugin_info.manifest.name)
            if not plugin_config:
                plugin_config = PluginConfig(plugin_info.manifest.name)
                self.registry.configs[plugin_info.manifest.name] = plugin_config

            plugin_config.settings.update(config)

            # Configure plugin instance if loaded
            if plugin_info.is_loaded and plugin_id in self.active_plugins:
                plugin_instance = self.active_plugins[plugin_id]
                plugin_instance.configure(config)

            # Save configuration
            self._save_configuration()

            self.logger.info(f"Plugin configured: {plugin_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to configure plugin {plugin_id}: {e}")
            return False

    def _create_plugin_api(self, plugin_info: PluginInfo) -> PluginAPI:
        """Create plugin API instance."""
        context = PluginContext(
            app_instance=self.app_context,
            session_manager=getattr(self.app_context, "session_manager", None),
            config_manager=getattr(self.app_context, "config_manager", None),
            event_system=getattr(self.app_context, "event_system", None),
            ui_manager=getattr(self.app_context, "ui_manager", None),
            data_manager=getattr(self.app_context, "data_manager", None),
            logger=self.logger.getChild(plugin_info.manifest.name),
        )

        return PluginAPI(context)

    def _emit_event(self, event_name: str, data: Any = None) -> None:
        """Emit an event to registered handlers."""
        if hasattr(self.app_context, "event_system"):
            self.app_context.event_system.emit(event_name, data)

    def register_event_handler(self, event_name: str, handler: Callable) -> None:
        """Register an event handler."""
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        self.event_handlers[event_name].append(handler)

    def get_plugin_status_summary(self) -> Dict[str, int]:
        """Get summary of plugin statuses."""
        summary = {}
        for status in PluginStatus:
            summary[status.value] = 0

        for plugin_info in self.registry.plugins.values():
            summary[plugin_info.status.value] += 1

        return summary

    def auto_load_plugins(self) -> None:
        """Automatically load and activate enabled plugins."""
        self.logger.info("Auto-loading plugins")

        for plugin_info in self.registry.plugins.values():
            plugin_config = self.registry.configs.get(plugin_info.manifest.name)

            if plugin_config and plugin_config.enabled:
                if self.load_plugin(plugin_info.id):
                    self.activate_plugin(plugin_info.id)

    def shutdown(self) -> None:
        """Shutdown plugin manager and clean up."""
        self.logger.info("Shutting down plugin manager")

        # Deactivate all active plugins
        for plugin_id in list(self.active_plugins.keys()):
            self.deactivate_plugin(plugin_id)

        # Save configuration
        self._save_configuration()

        self.logger.info("Plugin manager shutdown complete")
