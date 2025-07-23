"""Streamlined Event System

This module provides a simplified event system for the ICARUS CLI,
focusing on essential functionality and performance.
"""

import asyncio
import logging
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Set


class EventSystem:
    """Manages application events and callbacks."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_history: List[Dict[str, Any]] = []
        self._max_history = 100
        self.logger = logging.getLogger(__name__)

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Unsubscribe from an event type."""
        if (
            event_type in self._subscribers
            and callback in self._subscribers[event_type]
        ):
            self._subscribers[event_type].remove(callback)

    async def emit(self, event_type: str, data: Dict[str, Any] = None) -> None:
        """Emit an event to all subscribers."""
        if data is None:
            data = {}

        # Add to history
        event_record = {
            "type": event_type,
            "data": data,
            "timestamp": asyncio.get_event_loop().time(),
        }

        self._event_history.append(event_record)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history :]

        # Notify subscribers
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    result = callback(data)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    self.logger.error(f"Error in event callback: {e}")

    def get_event_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get event history."""
        if limit:
            return self._event_history[-limit:]
        return self._event_history.copy()

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()

    def get_subscribed_events(self) -> Set[str]:
        """Get all event types with subscribers."""
        return set(self._subscribers.keys())


# Global event system instance
event_system = None


def get_event_system() -> EventSystem:
    """Get or create the global event system instance."""
    global event_system
    if event_system is None:
        event_system = EventSystem()
    return event_system
