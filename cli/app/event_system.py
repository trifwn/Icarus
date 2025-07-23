"""Event System

This module provides a centralized event system for inter-component communication
within the ICARUS CLI application.
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional


class EventType(Enum):
    """Standard event types for the ICARUS CLI."""

    # Application events
    APP_STARTED = "app_started"
    APP_SHUTDOWN = "app_shutdown"

    # Screen events
    SCREEN_CHANGE = "screen_change"
    SCREEN_REFRESH = "screen_refresh"

    # Analysis events
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_PROGRESS = "analysis_progress"
    ANALYSIS_COMPLETED = "analysis_completed"
    ANALYSIS_FAILED = "analysis_failed"

    # Workflow events
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_STEP_COMPLETED = "workflow_step_completed"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"

    # Configuration events
    CONFIG_CHANGED = "config_changed"
    THEME_CHANGED = "theme_changed"

    # Notification events
    NOTIFICATION = "notification"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Event:
    """Represents an event in the system."""

    type: str
    data: Dict[str, Any]
    source: Optional[str] = None
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            import time

            self.timestamp = time.time()


class EventSystem:
    """Centralized event system for the ICARUS CLI."""

    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_history: List[Event] = []
        self.max_history = 1000
        self.logger = logging.getLogger(__name__)

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe to an event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []

        if callback not in self.subscribers[event_type]:
            self.subscribers[event_type].append(callback)
            self.logger.debug(f"Subscribed to event: {event_type}")

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Unsubscribe from an event type."""
        if event_type in self.subscribers:
            if callback in self.subscribers[event_type]:
                self.subscribers[event_type].remove(callback)
                self.logger.debug(f"Unsubscribed from event: {event_type}")

    async def emit(
        self,
        event_type: str,
        data: Dict[str, Any],
        source: str = None,
    ) -> None:
        """Emit an event to all subscribers."""
        event = Event(type=event_type, data=data, source=source)

        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)

        # Notify subscribers
        if event_type in self.subscribers:
            tasks = []
            for callback in self.subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        tasks.append(callback(data))
                    else:
                        # Run sync callback in thread pool
                        tasks.append(
                            asyncio.get_event_loop().run_in_executor(
                                None,
                                callback,
                                data,
                            ),
                        )
                except Exception as e:
                    self.logger.error(f"Error in event callback for {event_type}: {e}")

            # Wait for all callbacks to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        self.logger.debug(f"Emitted event: {event_type}")

    def emit_sync(
        self,
        event_type: str,
        data: Dict[str, Any],
        source: str = None,
    ) -> None:
        """Emit an event synchronously (for non-async contexts)."""
        asyncio.create_task(self.emit(event_type, data, source))

    def get_event_history(
        self,
        event_type: str = None,
        limit: int = 100,
    ) -> List[Event]:
        """Get event history, optionally filtered by type."""
        events = self.event_history

        if event_type:
            events = [e for e in events if e.type == event_type]

        return events[-limit:] if limit else events

    def clear_history(self) -> None:
        """Clear event history."""
        self.event_history.clear()

    def get_subscribers(self, event_type: str = None) -> Dict[str, int]:
        """Get subscriber counts by event type."""
        if event_type:
            return {event_type: len(self.subscribers.get(event_type, []))}
        else:
            return {et: len(callbacks) for et, callbacks in self.subscribers.items()}

    def has_subscribers(self, event_type: str) -> bool:
        """Check if an event type has any subscribers."""
        return event_type in self.subscribers and len(self.subscribers[event_type]) > 0
