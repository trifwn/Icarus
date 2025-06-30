"""Notification Widget for ICARUS TUI

This widget displays notifications using the core notification system.
"""

from textual.widgets import Log
from textual.reactive import reactive
from typing import Dict, Any, List

from core.ui import notification_system
from core.tui_integration import TUIEvent, TUIEventType


class NotificationWidget(Log):
    """Widget for displaying notifications."""

    notification_count = reactive(0)
    max_notifications = reactive(100)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.notification_history = []

    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        self.write("Notification system ready")

    def _on_notification(self, event: TUIEvent) -> None:
        """Handle notification events."""
        if event.type == TUIEventType.NOTIFICATION:
            self.add_notification(event.data.get("message", "Unknown notification"), event.data.get("level", "info"))

    def add_notification(self, message: str, level: str = "info") -> None:
        """Add a notification to the log."""
        timestamp = self._get_timestamp()
        formatted_message = f"[{timestamp}] [{level.upper()}] {message}"

        self.write(formatted_message)
        self.notification_count += 1

        # Keep notification history
        self.notification_history.append({"timestamp": timestamp, "level": level, "message": message})

        # Limit history size
        if len(self.notification_history) > self.max_notifications:
            self.notification_history = self.notification_history[-self.max_notifications :]

    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime

        return datetime.now().strftime("%H:%M:%S")

    def get_recent_notifications(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent notifications."""
        return self.notification_history[-count:] if self.notification_history else []

    def clear_notifications(self) -> None:
        """Clear all notifications."""
        self.clear()
        self.notification_history.clear()
        self.notification_count = 0
        self.write("Notifications cleared")

    def filter_notifications(self, level: str = None) -> List[Dict[str, Any]]:
        """Filter notifications by level."""
        if level is None:
            return self.notification_history

        return [n for n in self.notification_history if n["level"] == level]

    def get_notification_summary(self) -> Dict[str, Any]:
        """Get a summary of notifications."""
        if not self.notification_history:
            return {"total": 0, "by_level": {}}

        by_level = {}
        for notification in self.notification_history:
            level = notification["level"]
            by_level[level] = by_level.get(level, 0) + 1

        return {"total": len(self.notification_history), "by_level": by_level}
