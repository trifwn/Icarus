"""Session Widget for ICARUS TUI

This widget displays and manages session information using the core state management system.
"""

from core.state import session_manager
from core.tui_integration import TUIEvent
from core.tui_integration import TUIEventType
from textual.reactive import reactive
from textual.widget import Widget


class SessionWidget(Widget):
    """Widget for displaying session information."""

    session_id = reactive("")
    duration = reactive("")
    workflow = reactive("")
    airfoils_count = reactive(0)
    airplanes_count = reactive(0)
    results_count = reactive(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_session_info()

    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        self.update_session_info()

    def update_session_info(self) -> None:
        """Update session information from the core state manager."""
        try:
            session_info = session_manager.get_session_info()
            self.session_id = session_info.get("session_id", "")
            self.duration = session_info.get("duration", "")
            self.workflow = session_info.get("workflow", "None")
            self.airfoils_count = session_info.get("airfoils", 0)
            self.airplanes_count = session_info.get("airplanes", 0)
            self.results_count = session_info.get("results", 0)
        except Exception:
            self.session_id = "Error"
            self.duration = "Error"
            self.workflow = "Error"

    def _on_session_updated(self, event: TUIEvent) -> None:
        """Handle session update events."""
        if event.type == TUIEventType.SESSION_UPDATED:
            self.update_session_info()

    def render(self) -> str:
        """Render the session information."""
        return f"""Session: {self.session_id}
Duration: {self.duration}
Workflow: {self.workflow}
Airfoils: {self.airfoils_count}
Airplanes: {self.airplanes_count}
Results: {self.results_count}"""
