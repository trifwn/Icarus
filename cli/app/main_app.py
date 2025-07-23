"""Main Application Controller

This module provides the main application controller that manages the overall
CLI application lifecycle, screen transitions, and core functionality.
"""

import asyncio
from typing import Any
from typing import Dict
from typing import List

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

from ..core.config import ConfigManager
from ..core.ui import ThemeManager
from ..learning.learning_manager import LearningManager
from .event_system import EventSystem
from .screen_manager import ScreenManager
from .state_manager import StateManager


class IcarusApp(App):
    """Main ICARUS CLI application controller."""

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

        # Core managers
        self.config_manager = ConfigManager()
        self.theme_manager = ThemeManager()
        self.screen_manager = ScreenManager(self)
        self.event_system = EventSystem()
        self.state_manager = StateManager()
        self.learning_manager = LearningManager()

        # Application state
        self.current_screen = "dashboard"
        self.initialized = False

    async def on_mount(self) -> None:
        """Initialize the application when mounted."""
        await self._initialize_app()

    async def _initialize_app(self) -> None:
        """Initialize application components."""
        try:
            # Load configuration
            await self.config_manager.load_config()

            # Apply theme
            theme_name = self.config_manager.get("theme", "default")
            self.theme_manager.apply_theme(theme_name)

            # Initialize screen manager
            await self.screen_manager.initialize()

            # Set up event handlers
            self._setup_event_handlers()

            # Initialize learning system callbacks
            self._setup_learning_callbacks()

            # Check if this is a new user and show welcome
            await self._check_new_user_welcome()

            # Load initial screen
            await self.screen_manager.switch_to("dashboard")

            self.initialized = True
            self.log.info("ICARUS CLI application initialized successfully")

        except Exception as e:
            self.log.error(f"Failed to initialize application: {e}")
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
            self.current_screen = new_screen
            self.log.info(f"Screen changed to: {new_screen}")

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

    def _setup_learning_callbacks(self) -> None:
        """Set up learning system callbacks."""
        self.learning_manager.register_callbacks(
            on_tutorial_completed=self._on_tutorial_completed,
            on_achievement_earned=self._on_achievement_earned,
            on_help_requested=self._on_help_requested,
            on_error_explained=self._on_error_explained,
        )

    async def _check_new_user_welcome(self) -> None:
        """Check if this is a new user and show welcome."""
        welcome_info = self.learning_manager.show_welcome_for_new_user()

        if welcome_info["is_new_user"]:
            # Log new user
            self.log.info("New user detected - showing welcome information")

            # Could show welcome dialog or tutorial prompt here
            # For now, just emit an event that screens can listen to
            await self.event_system.emit("new_user_welcome", welcome_info)

    def _on_tutorial_completed(self, tutorial_id: str, achievements: List) -> None:
        """Handle tutorial completion."""
        self.log.info(f"Tutorial completed: {tutorial_id}")

        # Show achievement notifications
        for achievement in achievements:
            self.log.info(f"Achievement earned: {achievement.title}")

        # Emit event for UI updates
        asyncio.create_task(
            self.event_system.emit(
                "tutorial_completed",
                {"tutorial_id": tutorial_id, "achievements": achievements},
            ),
        )

    def _on_achievement_earned(self, achievement) -> None:
        """Handle achievement earned."""
        self.log.info(
            f"Achievement earned: {achievement.title} (+{achievement.points} points)",
        )

        # Emit event for UI notifications
        asyncio.create_task(
            self.event_system.emit("achievement_earned", {"achievement": achievement}),
        )

    def _on_help_requested(self, query: str, results: List) -> None:
        """Handle help request."""
        self.log.info(f"Help requested for: {query} ({len(results)} results)")

    def _on_error_explained(self, error_message: str, explanation) -> None:
        """Handle error explanation."""
        self.log.info(f"Error explained: {explanation.title}")

        # Emit event for UI updates
        asyncio.create_task(
            self.event_system.emit(
                "error_explained",
                {"error_message": error_message, "explanation": explanation},
            ),
        )

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
            # Save current state
            await self.state_manager.save_state()

            # Save configuration
            await self.config_manager.save_config()

            # Save learning data
            self.learning_manager.save_all_data()

            self.log.info("Application shutdown complete")

        except Exception as e:
            self.log.error(f"Error during shutdown: {e}")
