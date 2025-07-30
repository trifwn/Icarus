"""
Real-time State Synchronization System

This module handles real-time synchronization of application state between
multiple users in a collaboration session.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from uuid import uuid4

try:
    from cli.api.websocket import WebSocketMessage
    from cli.api.websocket import websocket_manager
except ImportError:
    # Fallback for testing without full API setup
    websocket_manager = None
    WebSocketMessage = None
from .session_sharing import SessionManager


class StateChangeType(str, Enum):
    """Types of state changes that can be synchronized"""

    SCREEN_CHANGE = "screen_change"
    ANALYSIS_UPDATE = "analysis_update"
    WORKFLOW_UPDATE = "workflow_update"
    PARAMETER_CHANGE = "parameter_change"
    RESULT_UPDATE = "result_update"
    CURSOR_MOVE = "cursor_move"
    SELECTION_CHANGE = "selection_change"
    UI_INTERACTION = "ui_interaction"
    DATA_IMPORT = "data_import"
    DATA_EXPORT = "data_export"
    CUSTOM = "custom"


class ConflictResolution(str, Enum):
    """Strategies for resolving state conflicts"""

    LAST_WRITER_WINS = "last_writer_wins"
    FIRST_WRITER_WINS = "first_writer_wins"
    MERGE = "merge"
    MANUAL = "manual"
    OWNER_WINS = "owner_wins"


@dataclass
class StateChange:
    """Represents a state change event"""

    id: str
    session_id: str
    user_id: str
    change_type: StateChangeType
    path: str  # JSON path to the changed data
    old_value: Any
    new_value: Any
    timestamp: datetime
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "change_type": self.change_type.value,
            "path": self.path,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "StateChange":
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            user_id=data["user_id"],
            change_type=StateChangeType(data["change_type"]),
            path=data["path"],
            old_value=data["old_value"],
            new_value=data["new_value"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class StateConflict:
    """Represents a conflict between state changes"""

    id: str
    session_id: str
    path: str
    changes: List[StateChange]
    resolution_strategy: ConflictResolution
    resolved: bool = False
    resolution_value: Any = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "path": self.path,
            "changes": [change.to_dict() for change in self.changes],
            "resolution_strategy": self.resolution_strategy.value,
            "resolved": self.resolved,
            "resolution_value": self.resolution_value,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


class StateSynchronizer:
    """Manages real-time state synchronization between users"""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.logger = logging.getLogger(__name__)

        # State storage
        self.session_states: Dict[str, Dict] = {}  # session_id -> state
        self.state_history: Dict[str, List[StateChange]] = {}  # session_id -> changes
        self.pending_conflicts: Dict[
            str,
            List[StateConflict],
        ] = {}  # session_id -> conflicts

        # Event handlers
        self.change_handlers: Dict[StateChangeType, List[Callable]] = {}
        self.conflict_handlers: List[Callable] = []

        # Synchronization settings
        self.max_history_size = 1000
        self.conflict_timeout = 30  # seconds
        self.batch_size = 10

        # Initialize default handlers
        self._setup_default_handlers()

    def _setup_default_handlers(self):
        """Setup default state change handlers"""
        self.register_change_handler(
            StateChangeType.SCREEN_CHANGE,
            self._handle_screen_change,
        )
        self.register_change_handler(
            StateChangeType.CURSOR_MOVE,
            self._handle_cursor_move,
        )
        self.register_change_handler(
            StateChangeType.ANALYSIS_UPDATE,
            self._handle_analysis_update,
        )
        self.register_change_handler(
            StateChangeType.WORKFLOW_UPDATE,
            self._handle_workflow_update,
        )

    def register_change_handler(self, change_type: StateChangeType, handler: Callable):
        """Register a handler for a specific state change type"""
        if change_type not in self.change_handlers:
            self.change_handlers[change_type] = []
        self.change_handlers[change_type].append(handler)

    def register_conflict_handler(self, handler: Callable):
        """Register a conflict resolution handler"""
        self.conflict_handlers.append(handler)

    def initialize_session_state(
        self,
        session_id: str,
        initial_state: Dict = None,
    ) -> Dict:
        """Initialize state for a collaboration session"""
        if initial_state is None:
            initial_state = {
                "current_screen": "dashboard",
                "active_analyses": {},
                "workflows": {},
                "parameters": {},
                "results": {},
                "ui_state": {},
                "cursors": {},
                "selections": {},
            }

        self.session_states[session_id] = initial_state.copy()
        self.state_history[session_id] = []
        self.pending_conflicts[session_id] = []

        self.logger.info(f"Initialized state for session: {session_id}")
        return initial_state

    def get_session_state(self, session_id: str) -> Optional[Dict]:
        """Get current state for a session"""
        return self.session_states.get(session_id)

    async def apply_state_change(self, change: StateChange) -> bool:
        """Apply a state change and synchronize with other users"""
        session_id = change.session_id

        # Validate session exists
        session = self.session_manager.get_session(session_id)
        if not session or not session.is_participant(change.user_id):
            self.logger.warning(
                f"Invalid session or user for state change: {change.id}",
            )
            return False

        # Get current session state
        if session_id not in self.session_states:
            self.initialize_session_state(session_id)

        current_state = self.session_states[session_id]

        # Check for conflicts
        conflict = self._detect_conflict(change, current_state)
        if conflict:
            await self._handle_conflict(conflict)
            return False

        # Apply the change
        success = self._apply_change_to_state(change, current_state)
        if not success:
            return False

        # Add to history
        self._add_to_history(change)

        # Execute handlers
        await self._execute_change_handlers(change)

        # Broadcast to other participants
        await self._broadcast_state_change(change)

        self.logger.debug(
            f"Applied state change: {change.change_type} at {change.path}",
        )
        return True

    def _detect_conflict(
        self,
        change: StateChange,
        current_state: Dict,
    ) -> Optional[StateConflict]:
        """Detect if a state change conflicts with recent changes"""
        session_id = change.session_id
        path = change.path

        # Get recent changes to the same path
        recent_changes = [
            c
            for c in self.state_history.get(session_id, [])
            if c.path == path
            and (datetime.now() - c.timestamp).total_seconds() < self.conflict_timeout
        ]

        if len(recent_changes) > 0:
            # Check if the old_value matches current state
            current_value = self._get_value_at_path(current_state, path)
            if current_value != change.old_value:
                # Conflict detected
                conflict = StateConflict(
                    id=str(uuid4()),
                    session_id=session_id,
                    path=path,
                    changes=recent_changes + [change],
                    resolution_strategy=ConflictResolution.LAST_WRITER_WINS,
                )
                return conflict

        return None

    def _apply_change_to_state(self, change: StateChange, state: Dict) -> bool:
        """Apply a state change to the session state"""
        try:
            path_parts = change.path.split(".")
            current = state

            # Navigate to the parent of the target
            for part in path_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Apply the change
            final_key = path_parts[-1]
            current[final_key] = change.new_value

            return True
        except Exception as e:
            self.logger.error(f"Failed to apply state change: {e}")
            return False

    def _get_value_at_path(self, state: Dict, path: str) -> Any:
        """Get value at a specific path in the state"""
        try:
            path_parts = path.split(".")
            current = state

            for part in path_parts:
                current = current[part]

            return current
        except (KeyError, TypeError):
            return None

    def _add_to_history(self, change: StateChange):
        """Add change to history with size limit"""
        session_id = change.session_id

        if session_id not in self.state_history:
            self.state_history[session_id] = []

        history = self.state_history[session_id]
        history.append(change)

        # Maintain history size limit
        if len(history) > self.max_history_size:
            history[:] = history[-self.max_history_size :]

    async def _execute_change_handlers(self, change: StateChange):
        """Execute registered handlers for the state change"""
        handlers = self.change_handlers.get(change.change_type, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(change)
                else:
                    handler(change)
            except Exception as e:
                self.logger.error(f"Error in change handler: {e}")

    async def _broadcast_state_change(self, change: StateChange):
        """Broadcast state change to other session participants"""
        if websocket_manager is None or WebSocketMessage is None:
            # Skip broadcasting if websocket is not available
            return

        message = WebSocketMessage(
            type="state_change",
            payload=change.to_dict(),
            session_id=change.session_id,
        )

        # Broadcast to all participants except the originator
        await websocket_manager.broadcast_to_room(
            change.session_id,
            message,
            exclude_session=change.user_id,
        )

    async def _handle_conflict(self, conflict: StateConflict):
        """Handle a state conflict"""
        session_id = conflict.session_id

        if session_id not in self.pending_conflicts:
            self.pending_conflicts[session_id] = []

        self.pending_conflicts[session_id].append(conflict)

        # Try automatic resolution
        resolved = await self._resolve_conflict_automatically(conflict)

        if not resolved:
            # Notify participants about the conflict
            await self._notify_conflict(conflict)

            # Execute conflict handlers
            for handler in self.conflict_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(conflict)
                    else:
                        handler(conflict)
                except Exception as e:
                    self.logger.error(f"Error in conflict handler: {e}")

    async def _resolve_conflict_automatically(self, conflict: StateConflict) -> bool:
        """Attempt to resolve conflict automatically"""
        if conflict.resolution_strategy == ConflictResolution.LAST_WRITER_WINS:
            # Use the most recent change
            latest_change = max(conflict.changes, key=lambda c: c.timestamp)
            conflict.resolution_value = latest_change.new_value
            conflict.resolved = True
            conflict.resolved_at = datetime.now()

            # Apply the resolution
            session_state = self.session_states[conflict.session_id]
            self._set_value_at_path(
                session_state,
                conflict.path,
                conflict.resolution_value,
            )

            return True

        elif conflict.resolution_strategy == ConflictResolution.FIRST_WRITER_WINS:
            # Use the earliest change
            earliest_change = min(conflict.changes, key=lambda c: c.timestamp)
            conflict.resolution_value = earliest_change.new_value
            conflict.resolved = True
            conflict.resolved_at = datetime.now()

            # Apply the resolution
            session_state = self.session_states[conflict.session_id]
            self._set_value_at_path(
                session_state,
                conflict.path,
                conflict.resolution_value,
            )

            return True

        elif conflict.resolution_strategy == ConflictResolution.OWNER_WINS:
            # Find change from session owner
            session = self.session_manager.get_session(conflict.session_id)
            if session:
                owner_changes = [
                    c for c in conflict.changes if c.user_id == session.owner_id
                ]
                if owner_changes:
                    owner_change = owner_changes[0]
                    conflict.resolution_value = owner_change.new_value
                    conflict.resolved = True
                    conflict.resolved_at = datetime.now()

                    # Apply the resolution
                    session_state = self.session_states[conflict.session_id]
                    self._set_value_at_path(
                        session_state,
                        conflict.path,
                        conflict.resolution_value,
                    )

                    return True

        return False

    def _set_value_at_path(self, state: Dict, path: str, value: Any):
        """Set value at a specific path in the state"""
        try:
            path_parts = path.split(".")
            current = state

            # Navigate to the parent of the target
            for part in path_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set the value
            final_key = path_parts[-1]
            current[final_key] = value

        except Exception as e:
            self.logger.error(f"Failed to set value at path {path}: {e}")

    async def _notify_conflict(self, conflict: StateConflict):
        """Notify session participants about a conflict"""
        if websocket_manager is None or WebSocketMessage is None:
            # Skip notification if websocket is not available
            return

        message = WebSocketMessage(
            type="state_conflict",
            payload=conflict.to_dict(),
            session_id=conflict.session_id,
        )

        await websocket_manager.broadcast_to_room(conflict.session_id, message)

    # Default change handlers

    async def _handle_screen_change(self, change: StateChange):
        """Handle screen change events"""
        session = self.session_manager.get_session(change.session_id)
        if session:
            # Update participant's current screen
            participant = session.get_participant(change.user_id)
            if participant:
                participant.current_screen = change.new_value

    async def _handle_cursor_move(self, change: StateChange):
        """Handle cursor movement events"""
        session = self.session_manager.get_session(change.session_id)
        if session:
            # Update participant's cursor position
            participant = session.get_participant(change.user_id)
            if participant:
                participant.cursor_position = change.new_value

    async def _handle_analysis_update(self, change: StateChange):
        """Handle analysis update events"""
        # Update shared analysis state
        session_state = self.session_states.get(change.session_id)
        if session_state:
            if "active_analyses" not in session_state:
                session_state["active_analyses"] = {}

            # Extract analysis ID from path
            path_parts = change.path.split(".")
            if len(path_parts) >= 2 and path_parts[0] == "active_analyses":
                analysis_id = path_parts[1]
                session_state["active_analyses"][analysis_id] = change.new_value

    async def _handle_workflow_update(self, change: StateChange):
        """Handle workflow update events"""
        # Update shared workflow state
        session_state = self.session_states.get(change.session_id)
        if session_state:
            if "workflows" not in session_state:
                session_state["workflows"] = {}

            # Extract workflow ID from path
            path_parts = change.path.split(".")
            if len(path_parts) >= 2 and path_parts[0] == "workflows":
                workflow_id = path_parts[1]
                session_state["workflows"][workflow_id] = change.new_value

    # Public API methods

    async def create_state_change(
        self,
        session_id: str,
        user_id: str,
        change_type: StateChangeType,
        path: str,
        old_value: Any,
        new_value: Any,
        metadata: Dict = None,
    ) -> StateChange:
        """Create and apply a new state change"""
        change = StateChange(
            id=str(uuid4()),
            session_id=session_id,
            user_id=user_id,
            change_type=change_type,
            path=path,
            old_value=old_value,
            new_value=new_value,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )

        await self.apply_state_change(change)
        return change

    def get_state_history(self, session_id: str, limit: int = 100) -> List[StateChange]:
        """Get state change history for a session"""
        history = self.state_history.get(session_id, [])
        return history[-limit:] if limit else history

    def get_pending_conflicts(self, session_id: str) -> List[StateConflict]:
        """Get pending conflicts for a session"""
        return self.pending_conflicts.get(session_id, [])

    async def resolve_conflict_manually(
        self,
        conflict_id: str,
        resolution_value: Any,
        resolved_by: str,
    ) -> bool:
        """Manually resolve a conflict"""
        for session_id, conflicts in self.pending_conflicts.items():
            for conflict in conflicts:
                if conflict.id == conflict_id:
                    conflict.resolution_value = resolution_value
                    conflict.resolved = True
                    conflict.resolved_by = resolved_by
                    conflict.resolved_at = datetime.now()

                    # Apply the resolution
                    session_state = self.session_states[session_id]
                    self._set_value_at_path(
                        session_state,
                        conflict.path,
                        resolution_value,
                    )

                    # Notify participants
                    if websocket_manager is not None and WebSocketMessage is not None:
                        message = WebSocketMessage(
                            type="conflict_resolved",
                            payload=conflict.to_dict(),
                            session_id=session_id,
                        )
                        await websocket_manager.broadcast_to_room(session_id, message)

                    return True

        return False

    def cleanup_session_state(self, session_id: str):
        """Clean up state data for an ended session"""
        if session_id in self.session_states:
            del self.session_states[session_id]

        if session_id in self.state_history:
            del self.state_history[session_id]

        if session_id in self.pending_conflicts:
            del self.pending_conflicts[session_id]

        self.logger.info(f"Cleaned up state for session: {session_id}")

    def get_synchronization_stats(self) -> Dict:
        """Get synchronization statistics"""
        total_changes = sum(len(history) for history in self.state_history.values())
        total_conflicts = sum(
            len(conflicts) for conflicts in self.pending_conflicts.values()
        )
        resolved_conflicts = sum(
            len([c for c in conflicts if c.resolved])
            for conflicts in self.pending_conflicts.values()
        )

        return {
            "active_sessions": len(self.session_states),
            "total_state_changes": total_changes,
            "total_conflicts": total_conflicts,
            "resolved_conflicts": resolved_conflicts,
            "pending_conflicts": total_conflicts - resolved_conflicts,
            "average_changes_per_session": (
                total_changes / len(self.session_states) if self.session_states else 0
            ),
        }
