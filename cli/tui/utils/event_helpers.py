"""Event Helper Utilities for ICARUS TUI

This module provides utilities for working with TUI events and the core integration system.
"""

from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass

from core.tui_integration import TUIEvent, TUIEventType, TUIEventManager


@dataclass
class EventSubscription:
    """Represents an event subscription."""

    event_type: TUIEventType
    callback: Callable
    source: str


class EventHelper:
    """Helper class for managing TUI events."""

    def __init__(self, event_manager: TUIEventManager):
        self.event_manager = event_manager
        self.subscriptions: Dict[str, EventSubscription] = {}

    def subscribe(self, event_type: TUIEventType, callback: Callable, source: str = "unknown") -> str:
        """Subscribe to an event type and return subscription ID."""
        subscription_id = f"{source}_{event_type.value}_{id(callback)}"

        subscription = EventSubscription(event_type=event_type, callback=callback, source=source)

        self.subscriptions[subscription_id] = subscription
        self.event_manager.subscribe(event_type, callback)

        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from an event using subscription ID."""
        if subscription_id in self.subscriptions:
            subscription = self.subscriptions[subscription_id]
            self.event_manager.unsubscribe(subscription.event_type, subscription.callback)
            del self.subscriptions[subscription_id]
            return True
        return False

    def emit_event(self, event_type: TUIEventType, data: Dict[str, Any], source: str) -> None:
        """Emit an event through the event manager."""
        event = TUIEvent(
            type=event_type,
            data=data,
            timestamp=0.0,  # Will be set by event manager
            source=source,
        )
        self.event_manager.emit(event)

    def get_recent_events(self, count: int = 10) -> list:
        """Get recent events from the event manager."""
        return self.event_manager.get_recent_events(count)

    def clear_subscriptions(self) -> None:
        """Clear all subscriptions."""
        for subscription_id, subscription in self.subscriptions.items():
            self.event_manager.unsubscribe(subscription.event_type, subscription.callback)
        self.subscriptions.clear()

    def get_subscription_info(self) -> Dict[str, Any]:
        """Get information about current subscriptions."""
        by_type = {}
        for subscription in self.subscriptions.values():
            event_type = subscription.event_type.value
            if event_type not in by_type:
                by_type[event_type] = 0
            by_type[event_type] += 1

        return {"total_subscriptions": len(self.subscriptions), "by_event_type": by_type}
