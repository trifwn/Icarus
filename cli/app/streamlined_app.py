"""Streamlined Main Application

This module provides a streamlined version of the main application controller
that integrates the unified configuration and session management systems.
"""

import asyncio
import logging
from typing import Any
from typing import Dict

try:
    from textual.app import App
    from textual.binding import Binding
    from textual.widgets import Footer
    from textual.widgets import Header

    TEXTUAL_AVAILABLE = True
except ImportError:
    # Create mock classes for when Textual is not available
    class App:
        def __init__(self, **kwargs):
            pass

    class Binding:
        def __init__(self, *args, **kwargs):
            pass

    TEXTUAL_AVAILABLE = False

from ..core.session_manager import get_session_manager
from ..core.ui import ThemeManager
from ..core.unified_config import get_config_manager
from .event_system import EventSystem
from .screen_manager import ScreenManager


class IcarusApp(App):
    """Streamlined ICARUS CLI application controller."""

    CSS_PATH = "tui_styles.css"
    TITLE = "ICARUS Aerodynamics"
    SUB_TITLE = "Advanced Aircraft Design & Analysis"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+h", "show_help", "Help"),
        Binding("ctrl+s", "show_settings", "Settings"),
        Binding("f5", "refresh", "Refresh"),
        Binding("ctrl+a", "show_airfoil", "Airfoil Analysis"),
        Binding("ctrl+p", "show_airplane", "Airplane Analysis"),
        Binding("ctrl+e", "show_export", "Export Results"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Core managers - using the streamlined implementations
        self.config_manager = get_config_manager()
        self.session_manager = get_session_manager()
        self.theme_manager = ThemeManager()
        self.screen_manager = ScreenManager(self)
        self.event_system = EventSystem()

        # Application state
        self.initialized = False
        self.logger = logging.getLogger(__name__)

    async def on_mount(self) -> None:
        """Initialize the application when mounted."""
        await self._initialize_app()

    async def _initialize_app(self) -> None:
        """Initialize application components."""
        try:
            # Load configuration
            await self.config_manager.load_config()

            # Initialize session
            await self.session_manager.initialize()

            # Apply theme
            theme_name = self.config_manager.get("theme", "default")
            self.theme_manager.apply_theme(theme_name)

            # Initialize screen manager
            await self.screen_manager.initialize()

            # Set up event handlers
            self._setup_event_handlers()

            # Load initial screen
            current_session = self.session_manager.current_session
            initial_screen = (
                current_session.current_screen if current_session else "dashboard"
            )
            await self.screen_manager.switch_to(initial_screen)

            self.initialized = True
            self.logger.info("ICARUS CLI application initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize application: {e}")
            raise

    def _setup_event_handlers(self) -> None:
        """Set up application-level event handlers."""
        self.event_system.subscribe("screen_change", self._on_screen_change)
        self.event_system.subscribe("theme_change", self._on_theme_change)
        self.event_system.subscribe("config_change", self._on_config_change)

    async def _on_screen_change(self, event_data: Dict[str, Any]) -> None:
        """Handle screen change events."""
        new_screen = event_data.get("screen")
        if new_screen:
            # Update session with current screen
            await self.session_manager.update_current_screen(new_screen)
            self.logger.info(f"Screen changed to: {new_screen}")

    async def _on_theme_change(self, event_data: Dict[str, Any]) -> None:
        """Handle theme change events."""
        theme_name = event_data.get("theme")
        if theme_name:
            self.theme_manager.apply_theme(theme_name)
            self.config_manager.set("theme", theme_name)
            await self.config_manager.save_config()

    async def _on_config_change(self, event_data: Dict[str, Any]) -> None:
        """Handle configuration change events."""
        await self.config_manager.save_config()

    def action_show_help(self) -> None:
        """Show help screen."""
        asyncio.create_task(self.screen_manager.switch_to("help"))

    def action_show_settings(self) -> None:
        """Show settings screen."""
        asyncio.create_task(self.screen_manager.switch_to("settings"))

    def action_refresh(self) -> None:
        """Refresh current screen."""
        asyncio.create_task(self.screen_manager.refresh_current())

    def action_show_airfoil(self) -> None:
        """Show airfoil analysis screen."""
        asyncio.create_task(self.screen_manager.switch_to("airfoil"))

    def action_show_airplane(self) -> None:
        """Show airplane analysis screen."""
        asyncio.create_task(self.screen_manager.switch_to("airplane"))

    def action_show_export(self) -> None:
        """Show export screen."""
        asyncio.create_task(self.screen_manager.switch_to("export"))

    async def shutdown(self) -> None:
        """Clean shutdown of the application."""
        try:
            # Save current session
            await self.session_manager.save_session()

            # Save configuration
            await self.config_manager.save_config()

            self.logger.info("Application shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
