"""Session Manager

This module provides a streamlined session management system for the ICARUS CLI,
handling session state, persistence, and history.
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
class Session:
    """Represents an application session."""

    id: str
    start_time: datetime
    current_screen: str
    active_analyses: List[str]
    recent_results: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "start_time": self.start_time.isoformat(),
            "current_screen": self.current_screen,
            "active_analyses": self.active_analyses,
            "recent_results": self.recent_results,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            current_screen=data.get("current_screen", "dashboard"),
            active_analyses=data.get("active_analyses", []),
            recent_results=data.get("recent_results", []),
        )


class SessionManager:
    """Manages application sessions and persistence."""

    def __init__(self, session_dir: str = "~/.icarus/sessions"):
        self.session_dir = Path(session_dir).expanduser()
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.current_session_file = self.session_dir / "current.json"
        self.history_file = self.session_dir / "history.json"

        self.current_session: Optional[Session] = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> Session:
        """Initialize session management."""
        # Try to load existing session
        session = await self.load_session()

        # If no session exists, create a new one
        if not session:
            session = await self.create_session()

        return session

    async def create_session(self) -> Session:
        """Create a new session."""
        import uuid

        session_id = str(uuid.uuid4())[:8]

        self.current_session = Session(
            id=session_id,
            start_time=datetime.now(),
            current_screen="dashboard",
            active_analyses=[],
            recent_results=[],
        )

        await self.save_session()
        self.logger.info(f"Created new session: {session_id}")

        return self.current_session

    async def load_session(self) -> Optional[Session]:
        """Load the current session."""
        try:
            if self.current_session_file.exists():
                with open(self.current_session_file) as f:
                    data = json.load(f)
                    self.current_session = Session.from_dict(data)
                    self.logger.info(f"Loaded session: {self.current_session.id}")
                    return self.current_session
        except Exception as e:
            self.logger.error(f"Failed to load session: {e}")

        return None

    async def save_session(self) -> None:
        """Save the current session."""
        if not self.current_session:
            return

        try:
            with open(self.current_session_file, "w") as f:
                json.dump(self.current_session.to_dict(), f, indent=2)

            self.logger.debug("Session saved")
        except Exception as e:
            self.logger.error(f"Failed to save session: {e}")

    async def end_session(self) -> None:
        """End the current session and archive it."""
        if not self.current_session:
            return

        try:
            # Load existing history
            history = []
            if self.history_file.exists():
                with open(self.history_file) as f:
                    history = json.load(f)

            # Add current session to history with end time
            session_data = self.current_session.to_dict()
            session_data["end_time"] = datetime.now().isoformat()
            history.append(session_data)

            # Keep only last 50 sessions
            history = history[-50:]

            # Save history
            with open(self.history_file, "w") as f:
                json.dump(history, f, indent=2)

            self.logger.info(f"Ended session: {self.current_session.id}")

            # Clear current session
            self.current_session = None
            if self.current_session_file.exists():
                self.current_session_file.unlink()

        except Exception as e:
            self.logger.error(f"Failed to end session: {e}")

    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session."""
        if not self.current_session:
            return {}

        duration = datetime.now() - self.current_session.start_time

        return {
            "session_id": self.current_session.id,
            "start_time": self.current_session.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": str(duration).split(".")[0],  # Remove microseconds
            "current_screen": self.current_session.current_screen,
            "active_analyses": len(self.current_session.active_analyses),
            "recent_results": len(self.current_session.recent_results),
        }

    async def update_current_screen(self, screen: str) -> None:
        """Update the current screen."""
        if self.current_session:
            self.current_session.current_screen = screen
            await self.save_session()

    async def add_analysis(self, analysis_id: str) -> None:
        """Add an analysis to the active list."""
        if self.current_session:
            if analysis_id not in self.current_session.active_analyses:
                self.current_session.active_analyses.append(analysis_id)
                await self.save_session()

    async def remove_analysis(self, analysis_id: str) -> None:
        """Remove an analysis from the active list."""
        if self.current_session:
            if analysis_id in self.current_session.active_analyses:
                self.current_session.active_analyses.remove(analysis_id)
                await self.save_session()

    async def add_result(self, result_id: str) -> None:
        """Add a result to the recent list."""
        if self.current_session:
            if result_id not in self.current_session.recent_results:
                self.current_session.recent_results.insert(0, result_id)
                # Keep only last 20 results
                self.current_session.recent_results = (
                    self.current_session.recent_results[:20]
                )
                await self.save_session()

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

    async def clear_history(self) -> None:
        """Clear session history."""
        try:
            if self.history_file.exists():
                self.history_file.unlink()

            self.logger.info("Session history cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear session history: {e}")


# Global session manager instance
session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create the global session manager instance."""
    global session_manager
    if session_manager is None:
        session_manager = SessionManager()
    return session_manager
