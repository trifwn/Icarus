"""
Plugin API for ICARUS CLI plugin development.

This module provides the core API that plugin developers use to create
extensions for the ICARUS CLI system.
"""

import logging
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from .exceptions import PluginError
from .models import PluginManifest


@dataclass
class PluginContext:
    """Context information provided to plugins."""

    app_instance: Any  # Main application instance
    session_manager: Any  # Session manager
    config_manager: Any  # Configuration manager
    event_system: Any  # Event system
    ui_manager: Any  # UI manager
    data_manager: Any  # Data management system
    logger: logging.Logger  # Plugin-specific logger


class PluginAPI:
    """
    Main API interface for plugin development.

    This class provides all the methods and utilities that plugins
    can use to interact with the ICARUS CLI system.
    """

    def __init__(self, context: PluginContext):
        self.context = context
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._ui_components: Dict[str, Any] = {}
        self._menu_items: List[Dict[str, Any]] = []
        self._commands: Dict[str, Callable] = {}

    # Event System API
    def register_event_handler(self, event_name: str, handler: Callable) -> None:
        """
        Register an event handler for a specific event.

        Args:
            event_name: Name of the event to listen for
            handler: Function to call when event occurs
        """
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        self._event_handlers[event_name].append(handler)

        # Register with the main event system
        self.context.event_system.register_handler(event_name, handler)

    def emit_event(self, event_name: str, data: Any = None) -> None:
        """
        Emit an event to the system.

        Args:
            event_name: Name of the event to emit
            data: Optional data to send with the event
        """
        self.context.event_system.emit(event_name, data)

    # UI API
    def register_screen(self, screen_name: str, screen_class: Any) -> None:
        """
        Register a new screen with the UI system.

        Args:
            screen_name: Unique name for the screen
            screen_class: Screen class to register
        """
        self.context.ui_manager.register_screen(screen_name, screen_class)

    def add_menu_item(
        self,
        menu_path: str,
        label: str,
        action: Callable,
        icon: str = None,
        shortcut: str = None,
    ) -> None:
        """
        Add a menu item to the application.

        Args:
            menu_path: Path where to add the menu item (e.g., "Tools/My Plugin")
            label: Display label for the menu item
            action: Function to call when menu item is selected
            icon: Optional icon for the menu item
            shortcut: Optional keyboard shortcut
        """
        menu_item = {
            "path": menu_path,
            "label": label,
            "action": action,
            "icon": icon,
            "shortcut": shortcut,
        }
        self._menu_items.append(menu_item)
        self.context.ui_manager.add_menu_item(menu_item)

    def register_widget(self, widget_name: str, widget_class: Any) -> None:
        """
        Register a custom widget.

        Args:
            widget_name: Unique name for the widget
            widget_class: Widget class to register
        """
        self._ui_components[widget_name] = widget_class
        self.context.ui_manager.register_widget(widget_name, widget_class)

    def show_notification(
        self,
        message: str,
        level: str = "info",
        duration: int = 3000,
    ) -> None:
        """
        Show a notification to the user.

        Args:
            message: Notification message
            level: Notification level (info, warning, error, success)
            duration: Duration in milliseconds
        """
        self.context.ui_manager.show_notification(message, level, duration)

    def show_dialog(
        self,
        title: str,
        content: str,
        dialog_type: str = "info",
        buttons: List[str] = None,
    ) -> str:
        """
        Show a dialog to the user.

        Args:
            title: Dialog title
            content: Dialog content
            dialog_type: Type of dialog (info, warning, error, question)
            buttons: List of button labels

        Returns:
            Selected button label
        """
        return self.context.ui_manager.show_dialog(title, content, dialog_type, buttons)

    # Data API
    def get_analysis_data(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Get analysis data by ID.

        Args:
            analysis_id: Unique analysis identifier

        Returns:
            Analysis data or None if not found
        """
        return self.context.data_manager.get_analysis(analysis_id)

    def save_analysis_data(self, analysis_id: str, data: Dict[str, Any]) -> None:
        """
        Save analysis data.

        Args:
            analysis_id: Unique analysis identifier
            data: Analysis data to save
        """
        self.context.data_manager.save_analysis(analysis_id, data)

    def get_user_data(self, key: str, default: Any = None) -> Any:
        """
        Get user-specific data.

        Args:
            key: Data key
            default: Default value if key not found

        Returns:
            Stored value or default
        """
        return self.context.data_manager.get_user_data(key, default)

    def set_user_data(self, key: str, value: Any) -> None:
        """
        Set user-specific data.

        Args:
            key: Data key
            value: Value to store
        """
        self.context.data_manager.set_user_data(key, value)

    # Configuration API
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get plugin configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self.context.config_manager.get_plugin_config(
            self.context.plugin_name,
            key,
            default,
        )

    def set_config(self, key: str, value: Any) -> None:
        """
        Set plugin configuration value.

        Args:
            key: Configuration key
            value: Value to set
        """
        self.context.config_manager.set_plugin_config(
            self.context.plugin_name,
            key,
            value,
        )

    # Command API
    def register_command(
        self,
        command_name: str,
        handler: Callable,
        description: str = "",
        usage: str = "",
    ) -> None:
        """
        Register a command that can be executed from the command palette.

        Args:
            command_name: Unique command name
            handler: Function to execute the command
            description: Command description
            usage: Usage information
        """
        self._commands[command_name] = handler
        self.context.app_instance.register_command(
            command_name,
            handler,
            description,
            usage,
        )

    # Logging API
    def log_info(self, message: str) -> None:
        """Log an info message."""
        self.context.logger.info(message)

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.context.logger.warning(message)

    def log_error(self, message: str, exc_info: bool = False) -> None:
        """Log an error message."""
        self.context.logger.error(message, exc_info=exc_info)

    def log_debug(self, message: str) -> None:
        """Log a debug message."""
        self.context.logger.debug(message)

    # Analysis Integration API
    def register_analysis_type(self, analysis_name: str, analysis_class: Any) -> None:
        """
        Register a new analysis type.

        Args:
            analysis_name: Unique analysis name
            analysis_class: Analysis implementation class
        """
        self.context.app_instance.register_analysis_type(analysis_name, analysis_class)

    def register_solver(self, solver_name: str, solver_class: Any) -> None:
        """
        Register a new solver.

        Args:
            solver_name: Unique solver name
            solver_class: Solver implementation class
        """
        self.context.app_instance.register_solver(solver_name, solver_class)

    # Workflow API
    def register_workflow_step(self, step_name: str, step_class: Any) -> None:
        """
        Register a custom workflow step.

        Args:
            step_name: Unique step name
            step_class: Step implementation class
        """
        self.context.app_instance.register_workflow_step(step_name, step_class)

    # Export/Import API
    def register_exporter(self, format_name: str, exporter_class: Any) -> None:
        """
        Register a data exporter.

        Args:
            format_name: Export format name
            exporter_class: Exporter implementation class
        """
        self.context.app_instance.register_exporter(format_name, exporter_class)

    def register_importer(self, format_name: str, importer_class: Any) -> None:
        """
        Register a data importer.

        Args:
            format_name: Import format name
            importer_class: Importer implementation class
        """
        self.context.app_instance.register_importer(format_name, importer_class)


class IcarusPlugin(ABC):
    """
    Base class for all ICARUS CLI plugins.

    Plugin developers should inherit from this class and implement
    the required methods.
    """

    def __init__(self):
        self.api: Optional[PluginAPI] = None
        self.manifest: Optional[PluginManifest] = None
        self._initialized = False

    @abstractmethod
    def get_manifest(self) -> PluginManifest:
        """
        Return the plugin manifest.

        This method must be implemented by all plugins to provide
        metadata about the plugin.

        Returns:
            PluginManifest with plugin information
        """
        pass

    def initialize(self, api: PluginAPI) -> None:
        """
        Initialize the plugin with the API.

        This method is called when the plugin is loaded. Plugins
        should perform their setup here.

        Args:
            api: Plugin API instance
        """
        self.api = api
        self.manifest = self.get_manifest()
        self._initialized = True

        # Call the plugin-specific initialization
        self.on_initialize()

    def on_initialize(self) -> None:
        """
        Plugin-specific initialization.

        Override this method to perform plugin-specific setup.
        """
        pass

    def activate(self) -> None:
        """
        Activate the plugin.

        This method is called when the plugin is activated.
        Plugins should register their functionality here.
        """
        if not self._initialized:
            raise PluginError("Plugin not initialized")

        self.on_activate()

    def on_activate(self) -> None:
        """
        Plugin-specific activation.

        Override this method to register functionality when activated.
        """
        pass

    def deactivate(self) -> None:
        """
        Deactivate the plugin.

        This method is called when the plugin is deactivated.
        Plugins should clean up their functionality here.
        """
        self.on_deactivate()

    def on_deactivate(self) -> None:
        """
        Plugin-specific deactivation.

        Override this method to clean up when deactivated.
        """
        pass

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the plugin with settings.

        Args:
            config: Configuration dictionary
        """
        self.on_configure(config)

    def on_configure(self, config: Dict[str, Any]) -> None:
        """
        Plugin-specific configuration.

        Override this method to handle configuration changes.

        Args:
            config: Configuration dictionary
        """
        pass

    def get_status(self) -> Dict[str, Any]:
        """
        Get plugin status information.

        Returns:
            Dictionary with status information
        """
        return {
            "initialized": self._initialized,
            "manifest": self.manifest.to_dict() if self.manifest else None,
        }


# Convenience decorators for plugin development
def plugin_command(name: str, description: str = "", usage: str = ""):
    """
    Decorator to register a plugin command.

    Args:
        name: Command name
        description: Command description
        usage: Usage information
    """

    def decorator(func):
        func._plugin_command = {
            "name": name,
            "description": description,
            "usage": usage,
        }
        return func

    return decorator


def plugin_event_handler(event_name: str):
    """
    Decorator to register an event handler.

    Args:
        event_name: Name of the event to handle
    """

    def decorator(func):
        func._plugin_event_handler = event_name
        return func

    return decorator


def plugin_menu_item(path: str, label: str, icon: str = None, shortcut: str = None):
    """
    Decorator to register a menu item.

    Args:
        path: Menu path
        label: Menu label
        icon: Optional icon
        shortcut: Optional shortcut
    """

    def decorator(func):
        func._plugin_menu_item = {
            "path": path,
            "label": label,
            "icon": icon,
            "shortcut": shortcut,
        }
        return func

    return decorator
