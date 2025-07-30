"""
WebSocket support for real-time features

This module provides WebSocket functionality for real-time collaboration,
progress updates, and live data synchronization.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from uuid import uuid4

from .models import CollaborationEvent
from .models import WebSocketMessage

logger = logging.getLogger(__name__)


class RealTimeEventType(str, Enum):
    """Types of real-time events"""

    CURSOR_MOVE = "cursor_move"
    TEXT_EDIT = "text_edit"
    SELECTION_CHANGE = "selection_change"
    SCREEN_CHANGE = "screen_change"
    ANALYSIS_START = "analysis_start"
    ANALYSIS_PROGRESS = "analysis_progress"
    ANALYSIS_COMPLETE = "analysis_complete"
    PARAMETER_CHANGE = "parameter_change"
    WORKFLOW_UPDATE = "workflow_update"
    USER_JOIN = "user_join"
    USER_LEAVE = "user_leave"
    NOTIFICATION = "notification"
    CONFLICT_DETECTED = "conflict_detected"
    CONFLICT_RESOLVED = "conflict_resolved"


@dataclass
class RealTimeUpdate:
    """Represents a real-time update event"""

    id: str
    event_type: RealTimeEventType
    user_id: str
    session_id: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RealTimeUpdate":
        return cls(
            id=data["id"],
            event_type=RealTimeEventType(data["event_type"]),
            user_id=data["user_id"],
            session_id=data["session_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data["data"],
            metadata=data.get("metadata", {}),
        )


class WebSocketConnection:
    """Represents a single WebSocket connection"""

    def __init__(self, websocket, session_id: Optional[str] = None):
        self.websocket = websocket
        self.session_id = session_id or str(uuid4())
        self.user_id: Optional[str] = None
        self.subscriptions: Set[str] = set()
        self.is_active = True

    async def send_message(self, message: WebSocketMessage) -> bool:
        """Send a message to this connection"""
        try:
            if not self.is_active:
                return False

            await self.websocket.send_text(message.model_dump_json())
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {self.session_id}: {e}")
            self.is_active = False
            return False

    async def close(self):
        """Close the WebSocket connection"""
        self.is_active = False
        try:
            await self.websocket.close()
        except Exception as e:
            logger.error(f"Error closing WebSocket {self.session_id}: {e}")


class WebSocketManager:
    """Manages WebSocket connections and message routing"""

    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> session_ids
        self.collaboration_rooms: Dict[str, Set[str]] = {}  # room_id -> session_ids
        self.real_time_updates: Dict[
            str,
            List[RealTimeUpdate],
        ] = {}  # session_id -> updates
        self.update_handlers: Dict[RealTimeEventType, List] = {}
        self._lock = asyncio.Lock()

        # Real-time collaboration settings
        self.max_updates_per_session = 1000
        self.update_batch_size = 10
        self.conflict_detection_window = 5  # seconds

    async def add_connection(
        self,
        websocket,
        session_id: Optional[str] = None,
    ) -> WebSocketConnection:
        """Add a new WebSocket connection"""
        async with self._lock:
            connection = WebSocketConnection(websocket, session_id)
            self.connections[connection.session_id] = connection
            logger.info(f"Added WebSocket connection: {connection.session_id}")
            return connection

    async def remove_connection(self, session_id: str):
        """Remove a WebSocket connection"""
        async with self._lock:
            if session_id in self.connections:
                connection = self.connections[session_id]

                # Remove from user sessions
                if connection.user_id and connection.user_id in self.user_sessions:
                    self.user_sessions[connection.user_id].discard(session_id)
                    if not self.user_sessions[connection.user_id]:
                        del self.user_sessions[connection.user_id]

                # Remove from collaboration rooms
                for room_id, sessions in self.collaboration_rooms.items():
                    sessions.discard(session_id)

                # Clean up empty rooms
                empty_rooms = [
                    room_id
                    for room_id, sessions in self.collaboration_rooms.items()
                    if not sessions
                ]
                for room_id in empty_rooms:
                    del self.collaboration_rooms[room_id]

                await connection.close()
                del self.connections[session_id]
                logger.info(f"Removed WebSocket connection: {session_id}")

    async def authenticate_connection(self, session_id: str, user_id: str):
        """Authenticate a WebSocket connection with a user ID"""
        async with self._lock:
            if session_id in self.connections:
                connection = self.connections[session_id]
                connection.user_id = user_id

                if user_id not in self.user_sessions:
                    self.user_sessions[user_id] = set()
                self.user_sessions[user_id].add(session_id)

                logger.info(f"Authenticated connection {session_id} for user {user_id}")

    async def join_collaboration_room(self, session_id: str, room_id: str):
        """Add a connection to a collaboration room"""
        async with self._lock:
            if session_id in self.connections:
                if room_id not in self.collaboration_rooms:
                    self.collaboration_rooms[room_id] = set()
                self.collaboration_rooms[room_id].add(session_id)

                connection = self.connections[session_id]
                connection.subscriptions.add(room_id)

                logger.info(f"Connection {session_id} joined room {room_id}")

    async def leave_collaboration_room(self, session_id: str, room_id: str):
        """Remove a connection from a collaboration room"""
        async with self._lock:
            if room_id in self.collaboration_rooms:
                self.collaboration_rooms[room_id].discard(session_id)
                if not self.collaboration_rooms[room_id]:
                    del self.collaboration_rooms[room_id]

            if session_id in self.connections:
                self.connections[session_id].subscriptions.discard(room_id)

                logger.info(f"Connection {session_id} left room {room_id}")

    async def send_to_session(self, session_id: str, message: WebSocketMessage) -> bool:
        """Send a message to a specific session"""
        if session_id in self.connections:
            return await self.connections[session_id].send_message(message)
        return False

    async def send_to_user(self, user_id: str, message: WebSocketMessage) -> int:
        """Send a message to all sessions of a user"""
        sent_count = 0
        if user_id in self.user_sessions:
            for session_id in self.user_sessions[user_id].copy():
                if await self.send_to_session(session_id, message):
                    sent_count += 1
                else:
                    # Connection is dead, remove it
                    await self.remove_connection(session_id)
        return sent_count

    async def broadcast_to_room(
        self,
        room_id: str,
        message: WebSocketMessage,
        exclude_session: Optional[str] = None,
    ) -> int:
        """Broadcast a message to all connections in a collaboration room"""
        sent_count = 0
        if room_id in self.collaboration_rooms:
            for session_id in self.collaboration_rooms[room_id].copy():
                if session_id != exclude_session:
                    if await self.send_to_session(session_id, message):
                        sent_count += 1
                    else:
                        # Connection is dead, remove it
                        await self.remove_connection(session_id)
        return sent_count

    async def broadcast(
        self,
        message: WebSocketMessage,
        exclude_session: Optional[str] = None,
    ) -> int:
        """Broadcast a message to all active connections"""
        sent_count = 0
        for session_id in list(self.connections.keys()):
            if session_id != exclude_session:
                if await self.send_to_session(session_id, message):
                    sent_count += 1
                else:
                    # Connection is dead, remove it
                    await self.remove_connection(session_id)
        return sent_count

    async def handle_collaboration_event(self, event: CollaborationEvent):
        """Handle a collaboration event and broadcast to relevant connections"""
        message = WebSocketMessage(
            type="collaboration_event",
            payload=event.model_dump(),
            session_id=event.session_id,
        )

        # Broadcast to the collaboration room
        await self.broadcast_to_room(
            event.session_id,
            message,
            exclude_session=event.session_id,
        )

    async def send_progress_update(
        self,
        session_id: str,
        progress: float,
        message: str = "",
    ):
        """Send a progress update to a specific session"""
        ws_message = WebSocketMessage(
            type="progress_update",
            payload={"progress": progress, "message": message},
        )
        await self.send_to_session(session_id, ws_message)

    async def send_analysis_result(self, session_id: str, result_data: Dict):
        """Send analysis result to a specific session"""
        ws_message = WebSocketMessage(type="analysis_result", payload=result_data)
        await self.send_to_session(session_id, ws_message)

    async def send_error_notification(
        self,
        session_id: str,
        error_message: str,
        details: Optional[Dict] = None,
    ):
        """Send error notification to a specific session"""
        ws_message = WebSocketMessage(
            type="error_notification",
            payload={"error": error_message, "details": details or {}},
        )
        await self.send_to_session(session_id, ws_message)

    # Real-time collaboration methods

    def register_update_handler(self, event_type: RealTimeEventType, handler):
        """Register a handler for real-time updates"""
        if event_type not in self.update_handlers:
            self.update_handlers[event_type] = []
        self.update_handlers[event_type].append(handler)

    async def send_real_time_update(self, update: RealTimeUpdate) -> bool:
        """Send a real-time update to collaboration room"""
        # Store the update
        if update.session_id not in self.real_time_updates:
            self.real_time_updates[update.session_id] = []

        self.real_time_updates[update.session_id].append(update)

        # Maintain update history limit
        if (
            len(self.real_time_updates[update.session_id])
            > self.max_updates_per_session
        ):
            self.real_time_updates[update.session_id] = self.real_time_updates[
                update.session_id
            ][-self.max_updates_per_session :]

        # Execute handlers
        await self._execute_update_handlers(update)

        # Broadcast to room participants
        message = WebSocketMessage(
            type="real_time_update",
            payload=update.to_dict(),
            session_id=update.session_id,
        )

        sent_count = await self.broadcast_to_room(
            update.session_id,
            message,
            exclude_session=update.user_id,  # Exclude the originator
        )

        return sent_count > 0

    async def _execute_update_handlers(self, update: RealTimeUpdate):
        """Execute registered handlers for the update"""
        handlers = self.update_handlers.get(update.event_type, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(update)
                else:
                    handler(update)
            except Exception as e:
                logger.error(f"Error in update handler: {e}")

    async def detect_conflicts(self, update: RealTimeUpdate) -> List[RealTimeUpdate]:
        """Detect conflicts with recent updates"""
        conflicts = []
        session_updates = self.real_time_updates.get(update.session_id, [])

        # Look for conflicting updates within the detection window
        cutoff_time = update.timestamp - timedelta(
            seconds=self.conflict_detection_window,
        )

        for existing_update in session_updates:
            if (
                existing_update.timestamp >= cutoff_time
                and existing_update.user_id != update.user_id
                and self._updates_conflict(existing_update, update)
            ):
                conflicts.append(existing_update)

        return conflicts

    def _updates_conflict(
        self,
        update1: RealTimeUpdate,
        update2: RealTimeUpdate,
    ) -> bool:
        """Check if two updates conflict with each other"""
        # Same path/target conflicts
        if update1.event_type == update2.event_type and update1.data.get(
            "path",
        ) == update2.data.get("path"):
            return True

        # Text editing conflicts
        if (
            update1.event_type == RealTimeEventType.TEXT_EDIT
            and update2.event_type == RealTimeEventType.TEXT_EDIT
        ):
            # Check if editing ranges overlap
            range1 = update1.data.get("range", {})
            range2 = update2.data.get("range", {})

            if range1 and range2:
                return self._ranges_overlap(range1, range2)

        # Parameter change conflicts
        if (
            update1.event_type == RealTimeEventType.PARAMETER_CHANGE
            and update2.event_type == RealTimeEventType.PARAMETER_CHANGE
            and update1.data.get("parameter") == update2.data.get("parameter")
        ):
            return True

        return False

    def _ranges_overlap(self, range1: Dict, range2: Dict) -> bool:
        """Check if two text ranges overlap"""
        start1, end1 = range1.get("start", 0), range1.get("end", 0)
        start2, end2 = range2.get("start", 0), range2.get("end", 0)

        return not (end1 <= start2 or end2 <= start1)

    async def send_conflict_notification(
        self,
        session_id: str,
        conflicts: List[RealTimeUpdate],
    ):
        """Send conflict notification to session participants"""
        message = WebSocketMessage(
            type="conflict_detected",
            payload={
                "conflicts": [conflict.to_dict() for conflict in conflicts],
                "timestamp": datetime.now().isoformat(),
            },
            session_id=session_id,
        )

        await self.broadcast_to_room(session_id, message)

    async def send_notification(
        self,
        session_id: str,
        notification_type: str,
        title: str,
        message: str,
        data: Dict = None,
    ):
        """Send a notification to session participants"""
        notification = WebSocketMessage(
            type="notification",
            payload={
                "notification_type": notification_type,
                "title": title,
                "message": message,
                "data": data or {},
                "timestamp": datetime.now().isoformat(),
            },
            session_id=session_id,
        )

        await self.broadcast_to_room(session_id, notification)

    def get_session_updates(
        self,
        session_id: str,
        limit: int = 100,
    ) -> List[RealTimeUpdate]:
        """Get recent updates for a session"""
        updates = self.real_time_updates.get(session_id, [])
        return updates[-limit:] if limit else updates

    def clear_session_updates(self, session_id: str):
        """Clear updates for a session"""
        if session_id in self.real_time_updates:
            del self.real_time_updates[session_id]

    async def get_connection_stats(self) -> Dict:
        """Get statistics about current connections"""
        async with self._lock:
            return {
                "total_connections": len(self.connections),
                "authenticated_users": len(self.user_sessions),
                "active_rooms": len(self.collaboration_rooms),
                "connections_per_user": {
                    user_id: len(sessions)
                    for user_id, sessions in self.user_sessions.items()
                },
                "sessions_per_room": {
                    room_id: len(sessions)
                    for room_id, sessions in self.collaboration_rooms.items()
                },
            }

    async def cleanup_inactive_connections(self):
        """Remove inactive connections"""
        inactive_sessions = []
        for session_id, connection in self.connections.items():
            if not connection.is_active:
                inactive_sessions.append(session_id)

        for session_id in inactive_sessions:
            await self.remove_connection(session_id)

        logger.info(f"Cleaned up {len(inactive_sessions)} inactive connections")

    async def shutdown(self):
        """Shutdown the WebSocket manager and close all connections"""
        logger.info("Shutting down WebSocket manager...")

        # Close all connections
        for session_id in list(self.connections.keys()):
            await self.remove_connection(session_id)

        # Clear all data structures
        self.connections.clear()
        self.user_sessions.clear()
        self.collaboration_rooms.clear()

        logger.info("WebSocket manager shutdown complete")


# Global WebSocket manager instance
websocket_manager = WebSocketManager()


async def handle_websocket_connection(websocket):
    """Handle a new WebSocket connection"""
    connection = await websocket_manager.add_connection(websocket)

    try:
        while True:
            # Wait for messages from the client
            data = await websocket.receive_text()

            try:
                message_data = json.loads(data)
                message = WebSocketMessage(**message_data)

                # Handle different message types
                await handle_websocket_message(connection, message)

            except json.JSONDecodeError:
                await connection.send_message(
                    WebSocketMessage(
                        type="error",
                        payload={"error": "Invalid JSON format"},
                    ),
                )
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await connection.send_message(
                    WebSocketMessage(
                        type="error",
                        payload={"error": "Message processing failed"},
                    ),
                )

    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")

    finally:
        await websocket_manager.remove_connection(connection.session_id)


async def handle_websocket_message(
    connection: WebSocketConnection,
    message: WebSocketMessage,
):
    """Handle incoming WebSocket messages"""
    if message.type == "authenticate":
        user_id = message.payload.get("user_id")
        if user_id:
            await websocket_manager.authenticate_connection(
                connection.session_id,
                user_id,
            )
            await connection.send_message(
                WebSocketMessage(
                    type="authenticated",
                    payload={"session_id": connection.session_id},
                ),
            )

    elif message.type == "join_room":
        room_id = message.payload.get("room_id")
        if room_id:
            await websocket_manager.join_collaboration_room(
                connection.session_id,
                room_id,
            )
            await connection.send_message(
                WebSocketMessage(type="joined_room", payload={"room_id": room_id}),
            )

    elif message.type == "leave_room":
        room_id = message.payload.get("room_id")
        if room_id:
            await websocket_manager.leave_collaboration_room(
                connection.session_id,
                room_id,
            )
            await connection.send_message(
                WebSocketMessage(type="left_room", payload={"room_id": room_id}),
            )

    elif message.type == "collaboration_event":
        # Handle collaboration events (cursor movement, text changes, etc.)
        event_data = message.payload
        if connection.user_id:
            event = CollaborationEvent(
                event_type=event_data.get("event_type", "unknown"),
                user_id=connection.user_id,
                session_id=connection.session_id,
                data=event_data.get("data", {}),
            )
            await websocket_manager.handle_collaboration_event(event)

    elif message.type == "real_time_update":
        # Handle real-time updates
        update_data = message.payload
        if connection.user_id:
            update = RealTimeUpdate(
                id=str(uuid4()),
                event_type=RealTimeEventType(update_data.get("event_type", "custom")),
                user_id=connection.user_id,
                session_id=connection.session_id,
                timestamp=datetime.now(),
                data=update_data.get("data", {}),
                metadata=update_data.get("metadata", {}),
            )

            # Check for conflicts
            conflicts = await websocket_manager.detect_conflicts(update)
            if conflicts:
                await websocket_manager.send_conflict_notification(
                    connection.session_id,
                    conflicts,
                )

            # Send the update
            await websocket_manager.send_real_time_update(update)

    elif message.type == "cursor_move":
        # Handle cursor movement
        if connection.user_id:
            update = RealTimeUpdate(
                id=str(uuid4()),
                event_type=RealTimeEventType.CURSOR_MOVE,
                user_id=connection.user_id,
                session_id=connection.session_id,
                timestamp=datetime.now(),
                data={
                    "position": message.payload.get("position", {}),
                    "screen": message.payload.get("screen", ""),
                },
            )
            await websocket_manager.send_real_time_update(update)

    elif message.type == "text_edit":
        # Handle text editing
        if connection.user_id:
            update = RealTimeUpdate(
                id=str(uuid4()),
                event_type=RealTimeEventType.TEXT_EDIT,
                user_id=connection.user_id,
                session_id=connection.session_id,
                timestamp=datetime.now(),
                data={
                    "path": message.payload.get("path", ""),
                    "range": message.payload.get("range", {}),
                    "text": message.payload.get("text", ""),
                    "operation": message.payload.get("operation", "insert"),
                },
            )

            # Check for conflicts
            conflicts = await websocket_manager.detect_conflicts(update)
            if conflicts:
                await websocket_manager.send_conflict_notification(
                    connection.session_id,
                    conflicts,
                )

            await websocket_manager.send_real_time_update(update)

    elif message.type == "parameter_change":
        # Handle parameter changes
        if connection.user_id:
            update = RealTimeUpdate(
                id=str(uuid4()),
                event_type=RealTimeEventType.PARAMETER_CHANGE,
                user_id=connection.user_id,
                session_id=connection.session_id,
                timestamp=datetime.now(),
                data={
                    "parameter": message.payload.get("parameter", ""),
                    "old_value": message.payload.get("old_value"),
                    "new_value": message.payload.get("new_value"),
                    "path": message.payload.get("path", ""),
                },
            )

            # Check for conflicts
            conflicts = await websocket_manager.detect_conflicts(update)
            if conflicts:
                await websocket_manager.send_conflict_notification(
                    connection.session_id,
                    conflicts,
                )

            await websocket_manager.send_real_time_update(update)

    elif message.type == "resolve_conflict":
        # Handle conflict resolution
        conflict_id = message.payload.get("conflict_id")
        resolution = message.payload.get("resolution")

        if conflict_id and resolution and connection.user_id:
            # This would integrate with the conflict resolution system
            # For now, just broadcast the resolution
            await websocket_manager.send_notification(
                connection.session_id,
                "conflict_resolution",
                "Conflict Resolved",
                f"Conflict {conflict_id} resolved with strategy: {resolution}",
                {"conflict_id": conflict_id, "resolution": resolution},
            )

    elif message.type == "ping":
        # Respond to ping with pong
        await connection.send_message(
            WebSocketMessage(type="pong", payload=message.payload),
        )

    else:
        logger.warning(f"Unknown WebSocket message type: {message.type}")
