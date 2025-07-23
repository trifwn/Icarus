"""State Management for ICARUS CLI

This module provides state management, session handling, and history tracking
for the ICARUS CLI application.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional


@dataclass
class SessionInfo:
    """Information about the current session."""

    session_id: str
    start_time: datetime
    duration: timedelta
    current_screen: str
    active_analyses: int
    completed_analyses: int
    workflow: str


class SessionManager:
    """Manages user sessions and session state."""

    def __init__(self, session_dir: str = "~/.icarus/sessions"):
        self.session_dir = Path(session_dir).expanduser()
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.current_session_file = self.session_dir / "current.json"
        self.session_history_file = self.session_dir / "history.json"

        self.session_id: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.session_data: Dict[str, Any] = {}

        self.logger = logging.getLogger(__name__)

        # Initialize session
        self._initialize_session()

    def _initialize_session(self) -> None:
        """Initialize a new session."""
        import uuid

        self.session_id = str(uuid.uuid4())[:8]
        self.start_time = datetime.now()

        self.session_data = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "current_screen": "dashboard",
            "active_analyses": [],
            "completed_analyses": [],
            "current_workflow": None,
            "user_preferences": {},
            "recent_files": [],
        }

        self._save_session()
        self.logger.info(f"Initialized session: {self.session_id}")

    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information."""
        if not self.start_time:
            return {}

        duration = datetime.now() - self.start_time

        return {
            "session_id": self.session_id or "unknown",
            "start_time": self.start_time.strftime("%H:%M:%S"),
            "duration": str(duration).split(".")[0],  # Remove microseconds
            "current_screen": self.session_data.get("current_screen", "unknown"),
            "active_analyses": len(self.session_data.get("active_analyses", [])),
            "completed_analyses": len(self.session_data.get("completed_analyses", [])),
            "workflow": self.session_data.get("current_workflow", "None"),
        }

    def update_current_screen(self, screen: str) -> None:
        """Update the current screen."""
        self.session_data["current_screen"] = screen
        self._save_session()

    def add_active_analysis(self, analysis_id: str) -> None:
        """Add an analysis to the active list."""
        if "active_analyses" not in self.session_data:
            self.session_data["active_analyses"] = []

        if analysis_id not in self.session_data["active_analyses"]:
            self.session_data["active_analyses"].append(analysis_id)
            self._save_session()

    def complete_analysis(self, analysis_id: str) -> None:
        """Move an analysis from active to completed."""
        if "active_analyses" not in self.session_data:
            self.session_data["active_analyses"] = []
        if "completed_analyses" not in self.session_data:
            self.session_data["completed_analyses"] = []

        if analysis_id in self.session_data["active_analyses"]:
            self.session_data["active_analyses"].remove(analysis_id)

        if analysis_id not in self.session_data["completed_analyses"]:
            self.session_data["completed_analyses"].append(analysis_id)

        self._save_session()

    def set_current_workflow(self, workflow_name: str) -> None:
        """Set the current workflow."""
        self.session_data["current_workflow"] = workflow_name
        self._save_session()

    def _save_session(self) -> None:
        """Save current session to file."""
        try:
            with open(self.current_session_file, "w") as f:
                json.dump(self.session_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save session: {e}")

    def _load_session(self) -> bool:
        """Load session from file."""
        try:
            if self.current_session_file.exists():
                with open(self.current_session_file) as f:
                    self.session_data = json.load(f)

                self.session_id = self.session_data.get("session_id")
                start_time_str = self.session_data.get("start_time")
                if start_time_str:
                    self.start_time = datetime.fromisoformat(start_time_str)

                return True
        except Exception as e:
            self.logger.error(f"Failed to load session: {e}")

        return False


class ConfigManager:
    """Manages application configuration."""

    def __init__(self, config_dir: str = "~/.icarus/config"):
        self.config_dir = Path(config_dir).expanduser()
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.config_file = self.config_dir / "config.json"
        self.config: Dict[str, Any] = {}

        self.logger = logging.getLogger(__name__)

        # Load configuration
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file."""
        try:
            if self.config_file.exists():
                with open(self.config_file) as f:
                    self.config = json.load(f)
            else:
                self._create_default_config()
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            self._create_default_config()

    def _create_default_config(self) -> None:
        """Create default configuration."""
        self.config = {
            "theme": "default",
            "database_path": "./Data",
            "auto_save": True,
            "confirm_exit": True,
            "ui": {
                "show_session_info": True,
                "show_progress": True,
                "animation_speed": "normal",
            },
            "analysis": {
                "default_solver_2d": "xfoil",
                "default_solver_3d": "avl",
                "parallel_processing": True,
                "max_workers": 4,
            },
        }
        self._save_config()

    def _save_config(self) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value
        self._save_config()

    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration."""
        return self.config.get("ui", {})

    def get_database_path(self) -> str:
        """Get database path."""
        return self.config.get("database_path", "./Data")

    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self._create_default_config()


class HistoryManager:
    """Manages command and action history."""

    def __init__(self, history_dir: str = "~/.icarus/history"):
        self.history_dir = Path(history_dir).expanduser()
        self.history_dir.mkdir(parents=True, exist_ok=True)

        self.command_history_file = self.history_dir / "commands.json"
        self.analysis_history_file = self.history_dir / "analyses.json"

        self.command_history: List[Dict[str, Any]] = []
        self.analysis_history: List[Dict[str, Any]] = []

        self.max_history_items = 1000

        self.logger = logging.getLogger(__name__)

        # Load history
        self._load_history()

    def _load_history(self) -> None:
        """Load history from files."""
        try:
            if self.command_history_file.exists():
                with open(self.command_history_file) as f:
                    self.command_history = json.load(f)

            if self.analysis_history_file.exists():
                with open(self.analysis_history_file) as f:
                    self.analysis_history = json.load(f)

        except Exception as e:
            self.logger.error(f"Failed to load history: {e}")

    def _save_history(self) -> None:
        """Save history to files."""
        try:
            with open(self.command_history_file, "w") as f:
                json.dump(self.command_history[-self.max_history_items :], f, indent=2)

            with open(self.analysis_history_file, "w") as f:
                json.dump(self.analysis_history[-self.max_history_items :], f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save history: {e}")

    def add_command(self, command: str, context: Dict[str, Any] = None) -> None:
        """Add a command to history."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "context": context or {},
        }

        self.command_history.append(entry)
        self._save_history()

    def add_analysis(
        self,
        analysis_type: str,
        parameters: Dict[str, Any],
        result: Dict[str, Any] = None,
    ) -> None:
        """Add an analysis to history."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": analysis_type,
            "parameters": parameters,
            "result": result or {},
        }

        self.analysis_history.append(entry)
        self._save_history()

    def get_recent_commands(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent commands."""
        return self.command_history[-count:] if self.command_history else []

    def get_recent_analyses(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent analyses."""
        return self.analysis_history[-count:] if self.analysis_history else []

    def clear_history(self) -> None:
        """Clear all history."""
        self.command_history.clear()
        self.analysis_history.clear()
        self._save_history()


# Global instances
session_manager = SessionManager()
config_manager = ConfigManager()
history_manager = HistoryManager()
