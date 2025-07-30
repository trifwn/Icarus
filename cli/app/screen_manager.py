"""Screen Manager

This module manages screen transitions, navigation history, and screen lifecycle
for the ICARUS CLI application.
"""

from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Optional

if TYPE_CHECKING:
    from .main_app import IcarusApp

try:
    from textual.screen import Screen
    from textual.widgets import Static

    TEXTUAL_AVAILABLE = True
except ImportError:
    # Create mock classes for when Textual is not available
    class Screen:
        def __init__(self, **kwargs):
            pass

        async def on_mount(self):
            pass

        def compose(self):
            return []

    class Static:
        def __init__(self, *args, **kwargs):
            pass

    TEXTUAL_AVAILABLE = False


class BaseScreen(Screen):
    """Base class for all ICARUS CLI screens."""

    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self.screen_name = name
        self.initialized = False

    async def on_mount(self) -> None:
        """Initialize screen when mounted."""
        if not self.initialized:
            await self.initialize()
            self.initialized = True

    async def initialize(self) -> None:
        """Initialize screen-specific components."""
        pass

    async def refresh_data(self) -> None:
        """Refresh screen data."""
        pass

    async def cleanup(self) -> None:
        """Cleanup screen resources."""
        pass


class DashboardScreen(BaseScreen):
    """Main dashboard screen."""

    def __init__(self, **kwargs):
        super().__init__("dashboard", **kwargs)

    def compose(self):
        try:
            from textual.containers import Container
            from textual.containers import Horizontal
            from textual.containers import Vertical
            from textual.widgets import Button
            from textual.widgets import Label
            from textual.widgets import Static

            yield Label("ICARUS Aerodynamics", classes="title")
            yield Label("Advanced Aircraft Design & Analysis", classes="subtitle")

            with Vertical():
                yield Label("Analysis Tools", classes="section-title")

                with Horizontal():
                    yield Button(
                        "Airfoil Analysis",
                        id="airfoil_btn",
                        variant="primary",
                    )
                    yield Button(
                        "Airplane Analysis",
                        id="airplane_btn",
                        variant="primary",
                    )
                    yield Button("Export Results", id="export_btn", variant="primary")

                yield Label("Quick Links", classes="section-title")

                with Horizontal():
                    yield Button("Settings", id="settings_btn", variant="default")
                    yield Button("Help", id="help_btn", variant="default")

                yield Label("Recent Analyses", classes="section-title")
                yield Static("No recent analyses", classes="placeholder")

        except ImportError:
            yield Static("Dashboard - Coming Soon", classes="placeholder")

    async def initialize(self) -> None:
        """Initialize dashboard components."""
        self.app.log.info("Initializing dashboard screen")

    async def on_button_pressed(self, event) -> None:
        """Handle button press events."""
        button_id = event.button.id

        if button_id == "airfoil_btn":
            await self.app.screen_manager.switch_to("airfoil")
        elif button_id == "airplane_btn":
            await self.app.screen_manager.switch_to("airplane")
        elif button_id == "export_btn":
            await self.app.screen_manager.switch_to("export")
        elif button_id == "settings_btn":
            await self.app.screen_manager.switch_to("settings")
        elif button_id == "help_btn":
            await self.app.screen_manager.switch_to("help")


class AnalysisScreen(BaseScreen):
    """Analysis configuration and execution screen."""

    def __init__(self, **kwargs):
        super().__init__("analysis", **kwargs)

    def compose(self):
        yield Static("Analysis - Coming Soon", classes="placeholder")

    async def initialize(self) -> None:
        """Initialize analysis components."""
        # TODO: Initialize analysis widgets
        pass


class ResultsScreen(BaseScreen):
    """Results viewing and visualization screen."""

    def __init__(self, **kwargs):
        super().__init__("results", **kwargs)

    def compose(self):
        yield Static("Results - Coming Soon", classes="placeholder")

    async def initialize(self) -> None:
        """Initialize results components."""
        # TODO: Initialize results widgets
        pass


class WorkflowScreen(BaseScreen):
    """Workflow management screen."""

    def __init__(self, **kwargs):
        super().__init__("workflow", **kwargs)

    def compose(self):
        yield Static("Workflow - Coming Soon", classes="placeholder")

    async def initialize(self) -> None:
        """Initialize workflow components."""
        # TODO: Initialize workflow widgets
        pass


class SettingsScreen(BaseScreen):
    """Settings and configuration screen."""

    def __init__(self, **kwargs):
        super().__init__("settings", **kwargs)

    def compose(self):
        yield Static("Settings - Coming Soon", classes="placeholder")

    async def initialize(self) -> None:
        """Initialize settings components."""
        # TODO: Initialize settings widgets
        pass


class HelpScreen(BaseScreen):
    """Help and documentation screen."""

    def __init__(self, **kwargs):
        super().__init__("help", **kwargs)

    def compose(self):
        # Import here to avoid circular imports
        from cli.learning.learning_screen import LearningScreen

        yield LearningScreen()

    async def initialize(self) -> None:
        """Initialize help components."""
        # Learning screen handles its own initialization
        pass


class ScreenManager:
    """Manages screen transitions and navigation."""

    def __init__(self, app: "IcarusApp"):
        self.app = app
        self.screens: Dict[str, BaseScreen] = {}
        self.screen_history: List[str] = []
        self.current_screen: Optional[str] = None

    async def initialize(self) -> None:
        """Initialize screen manager and register screens."""
        # Register built-in screens
        self._register_screen("dashboard", DashboardScreen)
        self._register_screen("analysis", AnalysisScreen)
        self._register_screen("results", ResultsScreen)
        self._register_screen("workflow", WorkflowScreen)
        self._register_screen("settings", SettingsScreen)
        self._register_screen("help", HelpScreen)

        # Register new analysis screens
        try:
            from .screens import AirfoilScreen
            from .screens import AirplaneScreen
            from .screens import ExportScreen

            self._register_screen("airfoil", AirfoilScreen)
            self._register_screen("airplane", AirplaneScreen)
            self._register_screen("export", ExportScreen)
            self.app.log.info("Registered analysis screens successfully")
        except ImportError as e:
            self.app.log.error(f"Failed to register analysis screens: {e}")

    def _register_screen(self, name: str, screen_class: type) -> None:
        """Register a screen class."""
        screen = screen_class()
        self.screens[name] = screen
        # Note: install_screen is only available when Textual is present
        if hasattr(self.app, "install_screen"):
            self.app.install_screen(screen, name=name)

    async def switch_to(self, screen_name: str, **kwargs) -> bool:
        """Switch to a specific screen."""
        if screen_name not in self.screens:
            if hasattr(self.app, "log"):
                self.app.log.error(f"Screen not found: {screen_name}")
            return False

        try:
            # Add to history if switching from another screen
            if self.current_screen and self.current_screen != screen_name:
                self.screen_history.append(self.current_screen)

            # Switch to new screen (only if Textual is available)
            if hasattr(self.app, "push_screen"):
                await self.app.push_screen(screen_name)

            self.current_screen = screen_name

            # Emit screen change event
            await self.app.event_system.emit(
                "screen_change",
                {
                    "screen": screen_name,
                    "previous": self.screen_history[-1]
                    if self.screen_history
                    else None,
                },
            )

            return True

        except Exception as e:
            if hasattr(self.app, "log"):
                self.app.log.error(f"Failed to switch to screen {screen_name}: {e}")
            return False

    async def go_back(self) -> bool:
        """Go back to the previous screen."""
        if not self.screen_history:
            return False

        previous_screen = self.screen_history.pop()
        return await self.switch_to(previous_screen)

    async def refresh_current(self) -> None:
        """Refresh the current screen."""
        if self.current_screen and self.current_screen in self.screens:
            screen = self.screens[self.current_screen]
            await screen.refresh_data()

    def get_current_screen(self) -> Optional[BaseScreen]:
        """Get the current screen instance."""
        if self.current_screen:
            return self.screens.get(self.current_screen)
        return None

    def get_screen_history(self) -> List[str]:
        """Get the screen navigation history."""
        return self.screen_history.copy()

    async def cleanup_screen(self, screen_name: str) -> None:
        """Cleanup a specific screen."""
        if screen_name in self.screens:
            await self.screens[screen_name].cleanup()
