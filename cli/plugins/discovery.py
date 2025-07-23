"""
Plugin discovery system for finding and loading plugins.
"""

import importlib
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

from .api import IcarusPlugin
from .exceptions import PluginLoadError
from .models import PluginInfo
from .models import PluginManifest
from .models import PluginStatus


class PluginDiscovery:
    """
    Plugin discovery system that finds and loads plugins from various sources.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.search_paths: List[Path] = []
        self.discovered_plugins: Dict[str, PluginInfo] = {}

        # Default search paths
        self._setup_default_paths()

    def _setup_default_paths(self) -> None:
        """Setup default plugin search paths."""
        # User plugins directory
        user_home = Path.home()
        user_plugins = user_home / ".icarus" / "plugins"
        if user_plugins.exists():
            self.search_paths.append(user_plugins)

        # System plugins directory
        system_plugins = Path("/usr/local/share/icarus/plugins")
        if system_plugins.exists():
            self.search_paths.append(system_plugins)

        # Development plugins directory (relative to CLI)
        dev_plugins = Path(__file__).parent.parent / "dev_plugins"
        if dev_plugins.exists():
            self.search_paths.append(dev_plugins)

        # Current working directory plugins
        cwd_plugins = Path.cwd() / "plugins"
        if cwd_plugins.exists():
            self.search_paths.append(cwd_plugins)

    def add_search_path(self, path: Path) -> None:
        """
        Add a directory to the plugin search paths.

        Args:
            path: Directory path to search for plugins
        """
        if path.exists() and path.is_dir():
            self.search_paths.append(path)
            self.logger.info(f"Added plugin search path: {path}")
        else:
            self.logger.warning(f"Plugin search path does not exist: {path}")

    def discover_plugins(self) -> List[PluginInfo]:
        """
        Discover all plugins in the search paths.

        Returns:
            List of discovered plugin information
        """
        self.discovered_plugins.clear()

        for search_path in self.search_paths:
            self.logger.info(f"Searching for plugins in: {search_path}")
            self._discover_in_path(search_path)

        return list(self.discovered_plugins.values())

    def _discover_in_path(self, path: Path) -> None:
        """
        Discover plugins in a specific path.

        Args:
            path: Path to search for plugins
        """
        try:
            for item in path.iterdir():
                if item.is_dir():
                    self._discover_plugin_directory(item)
                elif item.suffix == ".py":
                    self._discover_plugin_file(item)
        except Exception as e:
            self.logger.error(f"Error discovering plugins in {path}: {e}")

    def _discover_plugin_directory(self, plugin_dir: Path) -> None:
        """
        Discover a plugin in a directory.

        Args:
            plugin_dir: Plugin directory path
        """
        # Look for plugin manifest
        manifest_file = plugin_dir / "plugin.json"
        if not manifest_file.exists():
            # Try alternative names
            for alt_name in ["manifest.json", "plugin.yaml", "plugin.yml"]:
                alt_file = plugin_dir / alt_name
                if alt_file.exists():
                    manifest_file = alt_file
                    break
            else:
                self.logger.debug(f"No manifest found in {plugin_dir}")
                return

        try:
            plugin_info = self._load_plugin_from_directory(plugin_dir, manifest_file)
            if plugin_info:
                self.discovered_plugins[plugin_info.id] = plugin_info
                self.logger.info(f"Discovered plugin: {plugin_info.manifest.name}")
        except Exception as e:
            self.logger.error(f"Error loading plugin from {plugin_dir}: {e}")

    def _discover_plugin_file(self, plugin_file: Path) -> None:
        """
        Discover a plugin in a single Python file.

        Args:
            plugin_file: Plugin file path
        """
        try:
            plugin_info = self._load_plugin_from_file(plugin_file)
            if plugin_info:
                self.discovered_plugins[plugin_info.id] = plugin_info
                self.logger.info(f"Discovered plugin: {plugin_info.manifest.name}")
        except Exception as e:
            self.logger.error(f"Error loading plugin from {plugin_file}: {e}")

    def _load_plugin_from_directory(
        self,
        plugin_dir: Path,
        manifest_file: Path,
    ) -> Optional[PluginInfo]:
        """
        Load a plugin from a directory with manifest.

        Args:
            plugin_dir: Plugin directory
            manifest_file: Manifest file path

        Returns:
            PluginInfo if successful, None otherwise
        """
        # Load manifest
        try:
            with open(manifest_file) as f:
                if manifest_file.suffix in [".yaml", ".yml"]:
                    import yaml

                    manifest_data = yaml.safe_load(f)
                else:
                    manifest_data = json.load(f)

            manifest = PluginManifest.from_dict(manifest_data)
        except Exception as e:
            raise PluginLoadError(f"Failed to load manifest: {e}")

        # Find main module file
        main_module_file = plugin_dir / f"{manifest.main_module}.py"
        if not main_module_file.exists():
            # Try __init__.py
            main_module_file = plugin_dir / "__init__.py"
            if not main_module_file.exists():
                raise PluginLoadError(f"Main module not found: {manifest.main_module}")

        return PluginInfo(
            manifest=manifest,
            status=PluginStatus.DISCOVERED,
            path=str(plugin_dir),
        )

    def _load_plugin_from_file(self, plugin_file: Path) -> Optional[PluginInfo]:
        """
        Load a plugin from a single Python file.

        Args:
            plugin_file: Plugin file path

        Returns:
            PluginInfo if successful, None otherwise
        """
        # Try to import the module to get manifest
        try:
            spec = importlib.util.spec_from_file_location(plugin_file.stem, plugin_file)
            if spec is None or spec.loader is None:
                raise PluginLoadError(f"Cannot load module spec from {plugin_file}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Look for plugin class
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, IcarusPlugin)
                    and attr != IcarusPlugin
                ):
                    plugin_class = attr
                    break

            if plugin_class is None:
                self.logger.debug(f"No plugin class found in {plugin_file}")
                return None

            # Get manifest from plugin class
            plugin_instance = plugin_class()
            manifest = plugin_instance.get_manifest()

            return PluginInfo(
                manifest=manifest,
                status=PluginStatus.DISCOVERED,
                path=str(plugin_file),
            )

        except Exception as e:
            self.logger.error(f"Error loading plugin from {plugin_file}: {e}")
            return None

    def load_plugin_instance(self, plugin_info: PluginInfo) -> Optional[IcarusPlugin]:
        """
        Load and instantiate a plugin.

        Args:
            plugin_info: Plugin information

        Returns:
            Plugin instance if successful, None otherwise
        """
        try:
            plugin_path = Path(plugin_info.path)

            if plugin_path.is_dir():
                return self._load_instance_from_directory(plugin_info)
            else:
                return self._load_instance_from_file(plugin_info)

        except Exception as e:
            self.logger.error(f"Error loading plugin instance {plugin_info.id}: {e}")
            plugin_info.status = PluginStatus.ERROR
            plugin_info.last_error = str(e)
            return None

    def _load_instance_from_directory(self, plugin_info: PluginInfo) -> IcarusPlugin:
        """Load plugin instance from directory."""
        plugin_dir = Path(plugin_info.path)
        manifest = plugin_info.manifest

        # Add plugin directory to Python path
        if str(plugin_dir) not in sys.path:
            sys.path.insert(0, str(plugin_dir))

        try:
            # Import the main module
            module = importlib.import_module(manifest.main_module)

            # Get the plugin class
            plugin_class = getattr(module, manifest.main_class)

            # Create instance
            plugin_instance = plugin_class()

            # Update plugin info
            plugin_info.instance = plugin_instance
            plugin_info.status = PluginStatus.LOADED

            return plugin_instance

        finally:
            # Remove from path to avoid conflicts
            if str(plugin_dir) in sys.path:
                sys.path.remove(str(plugin_dir))

    def _load_instance_from_file(self, plugin_info: PluginInfo) -> IcarusPlugin:
        """Load plugin instance from single file."""
        plugin_file = Path(plugin_info.path)
        manifest = plugin_info.manifest

        # Load module from file
        spec = importlib.util.spec_from_file_location(plugin_file.stem, plugin_file)
        if spec is None or spec.loader is None:
            raise PluginLoadError(f"Cannot load module spec from {plugin_file}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the plugin class
        plugin_class = getattr(module, manifest.main_class)

        # Create instance
        plugin_instance = plugin_class()

        # Update plugin info
        plugin_info.instance = plugin_instance
        plugin_info.status = PluginStatus.LOADED

        return plugin_instance

    def get_plugin_info(self, plugin_id: str) -> Optional[PluginInfo]:
        """
        Get plugin information by ID.

        Args:
            plugin_id: Plugin identifier

        Returns:
            PluginInfo if found, None otherwise
        """
        return self.discovered_plugins.get(plugin_id)

    def refresh_plugin_discovery(self) -> List[PluginInfo]:
        """
        Refresh plugin discovery by rescanning all paths.

        Returns:
            List of all discovered plugins
        """
        self.logger.info("Refreshing plugin discovery")
        return self.discover_plugins()
