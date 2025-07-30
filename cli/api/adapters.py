"""
UI Adapter abstraction layer for multiple frontend support

This module provides abstract interfaces that allow the same business logic
to work with different UI implementations (TUI, Web, etc.).
"""

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Optional
from typing import Protocol

from .models import InputEvent
from .models import ScreenData


class UIAdapter(ABC):
    """Abstract interface for different UI implementations"""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the UI adapter"""
        pass

    @abstractmethod
    async def render_screen(self, screen_data: ScreenData) -> None:
        """Render a screen with given data"""
        pass

    @abstractmethod
    async def handle_user_input(self, input_event: InputEvent) -> None:
        """Handle user input events"""
        pass

    @abstractmethod
    async def update_component(self, component_id: str, data: Any) -> None:
        """Update a specific UI component"""
        pass

    @abstractmethod
    async def show_notification(self, message: str, level: str = "info") -> None:
        """Show a notification to the user"""
        pass

    @abstractmethod
    async def show_progress(self, progress: float, message: str = "") -> None:
        """Show progress indicator"""
        pass

    @abstractmethod
    async def prompt_user(
        self,
        message: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Prompt user for input"""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources when shutting down"""
        pass


class TextualUIAdapter(UIAdapter):
    """Textual-specific UI implementation"""

    def __init__(self):
        self.app = None
        self.current_screen = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the Textual UI adapter"""
        if self._initialized:
            return

        # Import here to avoid circular dependencies
        try:
            from textual.app import App
            from textual.screen import Screen

            self._textual_available = True
        except ImportError:
            self._textual_available = False
            raise RuntimeError("Textual framework not available")

        self._initialized = True

    async def render_screen(self, screen_data: ScreenData) -> None:
        """Render a screen with given data in Textual"""
        if not self._textual_available:
            raise RuntimeError("Textual framework not available")

        # This would integrate with the actual Textual app
        # For now, we'll store the screen data for later use
        self.current_screen = screen_data

        # In a real implementation, this would:
        # 1. Create or update Textual widgets based on screen_data
        # 2. Apply the content to the current screen
        # 3. Handle any actions or interactive elements

    async def handle_user_input(self, input_event: InputEvent) -> None:
        """Handle user input events in Textual"""
        if not self._textual_available:
            return

        # This would translate the generic input event to Textual-specific handling
        # For example, key presses, mouse clicks, etc.
        pass

    async def update_component(self, component_id: str, data: Any) -> None:
        """Update a specific UI component in Textual"""
        if not self._textual_available:
            return

        # This would find the specific widget by ID and update its content
        pass

    async def show_notification(self, message: str, level: str = "info") -> None:
        """Show a notification using Textual's notification system"""
        if not self._textual_available:
            print(f"[{level.upper()}] {message}")
            return

        # This would use Textual's notification system
        pass

    async def show_progress(self, progress: float, message: str = "") -> None:
        """Show progress indicator using Textual widgets"""
        if not self._textual_available:
            print(f"Progress: {progress:.1%} - {message}")
            return

        # This would update a Textual ProgressBar widget
        pass

    async def prompt_user(
        self,
        message: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Prompt user for input using Textual input widgets"""
        if not self._textual_available:
            return input(f"{message}: ")

        # This would create a modal dialog or input screen in Textual
        return None

    async def cleanup(self) -> None:
        """Cleanup Textual resources"""
        if self.app:
            await self.app.shutdown()


class WebUIAdapter(UIAdapter):
    """Future web UI implementation"""

    def __init__(self):
        self.websocket_manager = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the Web UI adapter"""
        if self._initialized:
            return

        # This would initialize WebSocket connections and web-specific resources
        self._initialized = True

    async def render_screen(self, screen_data: ScreenData) -> None:
        """Render a screen by sending data via WebSocket"""
        if self.websocket_manager:
            await self.websocket_manager.broadcast(
                {"type": "screen_update", "data": screen_data.model_dump()},
            )

    async def handle_user_input(self, input_event: InputEvent) -> None:
        """Handle user input events from web interface"""
        # This would process input events received via WebSocket
        pass

    async def update_component(self, component_id: str, data: Any) -> None:
        """Update a specific UI component via WebSocket"""
        if self.websocket_manager:
            await self.websocket_manager.broadcast(
                {
                    "type": "component_update",
                    "component_id": component_id,
                    "data": data,
                },
            )

    async def show_notification(self, message: str, level: str = "info") -> None:
        """Show notification via WebSocket"""
        if self.websocket_manager:
            await self.websocket_manager.broadcast(
                {"type": "notification", "message": message, "level": level},
            )

    async def show_progress(self, progress: float, message: str = "") -> None:
        """Show progress via WebSocket"""
        if self.websocket_manager:
            await self.websocket_manager.broadcast(
                {"type": "progress_update", "progress": progress, "message": message},
            )

    async def prompt_user(
        self,
        message: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Prompt user via WebSocket and wait for response"""
        if self.websocket_manager:
            # This would send a prompt and wait for user response
            await self.websocket_manager.broadcast(
                {"type": "user_prompt", "message": message, "options": options},
            )
        return None

    async def cleanup(self) -> None:
        """Cleanup web resources"""
        if self.websocket_manager:
            await self.websocket_manager.shutdown()


class UIAdapterFactory:
    """Factory for creating UI adapters"""

    @staticmethod
    def create_adapter(adapter_type: str) -> UIAdapter:
        """Create a UI adapter of the specified type"""
        if adapter_type.lower() == "textual":
            return TextualUIAdapter()
        elif adapter_type.lower() == "web":
            return WebUIAdapter()
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")


# Protocol for UI event handlers
class UIEventHandler(Protocol):
    """Protocol for handling UI events"""

    async def handle_analysis_request(self, config: Dict[str, Any]) -> None:
        """Handle analysis request from UI"""
        ...

    async def handle_workflow_request(self, workflow: Dict[str, Any]) -> None:
        """Handle workflow execution request from UI"""
        ...

    async def handle_session_update(self, session_data: Dict[str, Any]) -> None:
        """Handle session state update from UI"""
        ...


class UIBridge:
    """Bridge between UI adapters and business logic"""

    def __init__(self, adapter: UIAdapter, event_handler: UIEventHandler):
        self.adapter = adapter
        self.event_handler = event_handler

    async def initialize(self) -> None:
        """Initialize the UI bridge"""
        await self.adapter.initialize()

    async def process_input_event(self, event: InputEvent) -> None:
        """Process input event and route to appropriate handler"""
        await self.adapter.handle_user_input(event)

        # Route to business logic based on event type
        if event.event_type == "analysis_request":
            await self.event_handler.handle_analysis_request(event.data)
        elif event.event_type == "workflow_request":
            await self.event_handler.handle_workflow_request(event.data)
        elif event.event_type == "session_update":
            await self.event_handler.handle_session_update(event.data)

    async def update_ui(self, screen_data: ScreenData) -> None:
        """Update the UI with new screen data"""
        await self.adapter.render_screen(screen_data)

    async def notify_user(self, message: str, level: str = "info") -> None:
        """Send notification to user"""
        await self.adapter.show_notification(message, level)

    async def show_progress(self, progress: float, message: str = "") -> None:
        """Show progress to user"""
        await self.adapter.show_progress(progress, message)

    async def cleanup(self) -> None:
        """Cleanup resources"""
        await self.adapter.cleanup()
