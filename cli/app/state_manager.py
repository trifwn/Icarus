"""State Manager

This module manages application state, session data, and persistent storage
for the ICARUS CLI application.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional


@dataclass
class SessionState:
    """Represents the current application session state."""

    session_id: str
    start_time: datetime
    current_screen: str
    user_preferences: Dict[str, Any]
    active_analyses: List[str]
    recent_results: List[str]
    workflow_state: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "current_screen": self.current_screen,
            "user_preferences": self.user_preferences,
            "active_analyses": self.active_analyses,
            "recent_results": self.recent_results,
            "workflow_state": self.workflow_state,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            current_screen=data["current_screen"],
            user_preferences=data.get("user_preferences", {}),
            active_analyses=data.get("active_analyses", []),
            recent_results=data.get("recent_results", []),
            workflow_state=data.get("workflow_state", {}),
        )


class StateManager:
    """Manages application state and session persistence."""

    def __init__(self, state_dir: str = "~/.icarus/state"):
        self.state_dir = Path(state_dir).expanduser()
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.session_file = self.state_dir / "session.json"
        self.history_file = self.state_dir / "history.json"

        self.current_session: Optional[SessionState] = None
        self.session_history: List[Dict[str, Any]] = []

        self.logger = logging.getLogger(__name__)

    async def initialize_session(self) -> SessionState:
        """Initialize a new session."""
        import uuid

        session_id = str(uuid.uuid4())[:8]

        self.current_session = SessionState(
            session_id=session_id,
            start_time=datetime.now(),
            current_screen="dashboard",
            user_preferences={},
            active_analyses=[],
            recent_results=[],
            workflow_state={},
        )

        await self.save_state()
        self.logger.info(f"Initialized new session: {session_id}")

        return self.current_session

    async def load_state(self) -> Optional[SessionState]:
        """Load the last session state."""
        try:
            if self.session_file.exists():
                with open(self.session_file) as f:
                    data = json.load(f)
                    self.current_session = SessionState.from_dict(data)
                    self.logger.info(
                        f"Loaded session: {self.current_session.session_id}",
                    )
                    return self.current_session
        except Exception as e:
            self.logger.error(f"Failed to load session state: {e}")

        # If loading fails, create new session
        return await self.initialize_session()

    async def save_state(self) -> None:
        """Save current session state."""
        if not self.current_session:
            return

        try:
            with open(self.session_file, "w") as f:
                json.dump(self.current_session.to_dict(), f, indent=2)

            self.logger.debug("Session state saved")

        except Exception as e:
            self.logger.error(f"Failed to save session state: {e}")

    async def update_session(self, **kwargs) -> None:
        """Update current session with new data."""
        if not self.current_session:
            await self.initialize_session()

        for key, value in kwargs.items():
            if hasattr(self.current_session, key):
                setattr(self.current_session, key, value)

        await self.save_state()

    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information."""
        if not self.current_session:
            return {}

        duration = datetime.now() - self.current_session.start_time

        return {
            "session_id": self.current_session.session_id,
            "start_time": self.current_session.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": str(duration).split(".")[0],  # Remove microseconds
            "current_screen": self.current_session.current_screen,
            "active_analyses": len(self.current_session.active_analyses),
            "recent_results": len(self.current_session.recent_results),
        }

    async def add_analysis(self, analysis_id: str) -> None:
        """Add an analysis to the active list."""
        if self.current_session:
            if analysis_id not in self.current_session.active_analyses:
                self.current_session.active_analyses.append(analysis_id)
                await self.save_state()

    async def remove_analysis(self, analysis_id: str) -> None:
        """Remove an analysis from the active list."""
        if self.current_session:
            if analysis_id in self.current_session.active_analyses:
                self.current_session.active_analyses.remove(analysis_id)
                await self.save_state()

    async def add_result(self, result_id: str) -> None:
        """Add a result to the recent list."""
        if self.current_session:
            if result_id not in self.current_session.recent_results:
                self.current_session.recent_results.insert(0, result_id)
                # Keep only last 20 results
                self.current_session.recent_results = (
                    self.current_session.recent_results[:20]
                )
                await self.save_state()

    async def set_preference(self, key: str, value: Any) -> None:
        """Set a user preference."""
        if self.current_session:
            self.current_session.user_preferences[key] = value
            await self.save_state()

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference."""
        if self.current_session:
            return self.current_session.user_preferences.get(key, default)
        return default

    async def set_workflow_state(self, workflow_id: str, state: Dict[str, Any]) -> None:
        """Set workflow state."""
        if self.current_session:
            self.current_session.workflow_state[workflow_id] = state
            await self.save_state()

    def get_workflow_state(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow state."""
        if self.current_session:
            return self.current_session.workflow_state.get(workflow_id, {})
        return {}

    async def archive_session(self) -> None:
        """Archive current session to history."""
        if not self.current_session:
            return

        try:
            # Load existing history
            history = []
            if self.history_file.exists():
                with open(self.history_file) as f:
                    history = json.load(f)

            # Add current session to history
            session_data = self.current_session.to_dict()
            session_data["end_time"] = datetime.now().isoformat()
            history.append(session_data)

            # Keep only last 50 sessions
            history = history[-50:]

            # Save history
            with open(self.history_file, "w") as f:
                json.dump(history, f, indent=2)

            self.logger.info(f"Archived session: {self.current_session.session_id}")

        except Exception as e:
            self.logger.error(f"Failed to archive session: {e}")

    async def get_session_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get session history."""
        try:
            if self.history_file.exists():
                with open(self.history_file) as f:
                    history = json.load(f)
                    return history[-limit:] if limit else history
        except Exception as e:
            self.logger.error(f"Failed to load session history: {e}")

        return []

    async def clear_state(self) -> None:
        """Clear all state data."""
        try:
            if self.session_file.exists():
                self.session_file.unlink()
            if self.history_file.exists():
                self.history_file.unlink()

            self.current_session = None
            self.session_history.clear()

            self.logger.info("State data cleared")

        except Exception as e:
            self.logger.error(f"Failed to clear state: {e}")
