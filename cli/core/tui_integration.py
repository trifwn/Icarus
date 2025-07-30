"""TUI Integration Module for ICARUS CLI

This module provides integration between our core features and the Textual TUI application,
including event handling, data binding, and real-time updates.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from textual import work
from textual.app import App

from .services import export_service
from .services import validation_service
from .state import config_manager
from .state import session_manager
from .workflow import workflow_engine


class TUIEventType(Enum):
    """Types of TUI events."""

    SESSION_UPDATED = "session_updated"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_COMPLETED = "analysis_completed"
    VALIDATION_RESULT = "validation_result"
    EXPORT_COMPLETED = "export_completed"
    NOTIFICATION = "notification"
    SETTINGS_CHANGED = "settings_changed"


@dataclass
class TUIEvent:
    """Represents a TUI event."""

    type: TUIEventType
    data: Dict[str, Any]
    timestamp: float
    source: str


class TUIEventManager:
    """Manages TUI events and callbacks."""

    def __init__(self):
        self.callbacks: Dict[TUIEventType, List[Callable]] = {}
        self.event_history: List[TUIEvent] = []
        self.max_history = 100

    def subscribe(self, event_type: TUIEventType, callback: Callable) -> None:
        """Subscribe to an event type."""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)

    def unsubscribe(self, event_type: TUIEventType, callback: Callable) -> None:
        """Unsubscribe from an event type."""
        if event_type in self.callbacks:
            try:
                self.callbacks[event_type].remove(callback)
            except ValueError:
                pass

    def emit(self, event: TUIEvent) -> None:
        """Emit an event to all subscribers."""
        self.event_history.append(event)

        # Keep history size manageable
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history :]

        # Notify subscribers
        if event.type in self.callbacks:
            for callback in self.callbacks[event.type]:
                try:
                    callback(event)
                except Exception as e:
                    print(f"Error in event callback: {e}")

    def get_recent_events(self, count: int = 10) -> List[TUIEvent]:
        """Get recent events."""
        return self.event_history[-count:] if self.event_history else []


class TUISessionManager:
    """TUI-specific session management."""

    def __init__(self, event_manager: TUIEventManager):
        self.event_manager = event_manager
        self._setup_callbacks()

    def _setup_callbacks(self) -> None:
        """Setup callbacks for session changes."""
        # Monitor session changes
        self.event_manager.subscribe(
            TUIEventType.SESSION_UPDATED,
            self._on_session_updated,
        )

    def _on_session_updated(self, event: TUIEvent) -> None:
        """Handle session update events."""
        # This can be used to update UI components
        pass

    def update_session_info(self) -> Dict[str, Any]:
        """Get current session information for TUI display."""
        session_info = session_manager.get_session_info()
        return {
            "session_id": session_info["session_id"],
            "duration": session_info["duration"],
            "workflow": session_info["workflow"],
            "airfoils": session_info["airfoils"],
            "airplanes": session_info["airplanes"],
            "results": session_info["results"],
        }

    def add_airfoil_to_session(self, airfoil_name: str) -> None:
        """Add airfoil to session and emit event."""
        session_manager.add_airfoil(airfoil_name)

        event = TUIEvent(
            type=TUIEventType.SESSION_UPDATED,
            data={"action": "add_airfoil", "airfoil": airfoil_name},
            timestamp=asyncio.get_event_loop().time(),
            source="session_manager",
        )
        self.event_manager.emit(event)

    def set_session_result(self, key: str, value: Any) -> None:
        """Set session result and emit event."""
        session_manager.set_result(key, value)

        event = TUIEvent(
            type=TUIEventType.SESSION_UPDATED,
            data={"action": "set_result", "key": key, "value": value},
            timestamp=asyncio.get_event_loop().time(),
            source="session_manager",
        )
        self.event_manager.emit(event)


class TUIWorkflowManager:
    """TUI-specific workflow management."""

    def __init__(self, event_manager: TUIEventManager):
        self.event_manager = event_manager
        self.current_workflow: Optional[str] = None
        self.workflow_progress: Dict[str, float] = {}

    def start_workflow(self, workflow_name: str) -> bool:
        """Start a workflow and emit event."""
        if workflow_engine.start_workflow(workflow_name):
            self.current_workflow = workflow_name
            self.workflow_progress[workflow_name] = 0.0

            event = TUIEvent(
                type=TUIEventType.WORKFLOW_STARTED,
                data={"workflow": workflow_name},
                timestamp=asyncio.get_event_loop().time(),
                source="workflow_manager",
            )
            self.event_manager.emit(event)
            return True
        return False

    def execute_workflow_step(self, step_name: str) -> bool:
        """Execute a workflow step and update progress."""
        if not self.current_workflow:
            return False

        workflow = workflow_engine.get_workflow(self.current_workflow)
        if not workflow:
            return False

        # Find the step
        step = None
        for s in workflow.steps:
            if s.name == step_name:
                step = s
                break

        if step and workflow_engine.execute_step(step):
            # Update progress
            step_index = workflow.steps.index(step)
            progress = (step_index + 1) / len(workflow.steps) * 100
            self.workflow_progress[self.current_workflow] = progress

            return True

        return False

    def complete_workflow(self) -> None:
        """Complete the current workflow and emit event."""
        if self.current_workflow:
            event = TUIEvent(
                type=TUIEventType.WORKFLOW_COMPLETED,
                data={"workflow": self.current_workflow},
                timestamp=asyncio.get_event_loop().time(),
                source="workflow_manager",
            )
            self.event_manager.emit(event)

            self.workflow_progress[self.current_workflow] = 100.0
            self.current_workflow = None

    def get_workflow_progress(self) -> float:
        """Get current workflow progress."""
        if self.current_workflow:
            return self.workflow_progress.get(self.current_workflow, 0.0)
        return 0.0

    def get_available_workflows(self) -> List[Dict[str, Any]]:
        """Get available workflows for TUI display."""
        workflows = workflow_engine.get_workflows()
        return [
            {
                "name": w.name,
                "type": w.type.value,
                "description": w.description,
                "steps": len(w.steps),
            }
            for w in workflows
        ]


class TUIAnalysisManager:
    """TUI-specific analysis management."""

    def __init__(self, event_manager: TUIEventManager):
        self.event_manager = event_manager
        self.current_analysis: Optional[Dict[str, Any]] = None
        self.analysis_progress: float = 0.0

    @work
    async def run_analysis(self, analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis with progress tracking."""
        self.current_analysis = analysis_config
        self.analysis_progress = 0.0

        # Emit analysis started event
        event = TUIEvent(
            type=TUIEventType.ANALYSIS_STARTED,
            data={"config": analysis_config},
            timestamp=asyncio.get_event_loop().time(),
            source="analysis_manager",
        )
        self.event_manager.emit(event)

        # Validate configuration
        errors = validation_service.validate_data(analysis_config, "airfoil")
        if errors:
            validation_event = TUIEvent(
                type=TUIEventType.VALIDATION_RESULT,
                data={"errors": errors, "valid": False},
                timestamp=asyncio.get_event_loop().time(),
                source="analysis_manager",
            )
            self.event_manager.emit(validation_event)
            return {"error": "Validation failed", "details": errors}

        # Simulate analysis steps
        steps = [
            ("Loading airfoil data", 20),
            ("Configuring solver", 40),
            ("Running analysis", 70),
            ("Processing results", 90),
            ("Saving results", 100),
        ]

        results = {
            "target": analysis_config.get("name", "unknown"),
            "solver": analysis_config.get("solver", "xfoil"),
            "reynolds": analysis_config.get("reynolds", 1e6),
            "angles": analysis_config.get("angles", "0:15:16"),
            "results": {
                "cl_max": 1.2,
                "cd_min": 0.005,
                "alpha_stall": 15.5,
                "efficiency": 0.85,
            },
        }

        for step_name, progress in steps:
            self.analysis_progress = progress
            await asyncio.sleep(0.5)  # Simulate work

        # Emit analysis completed event
        completed_event = TUIEvent(
            type=TUIEventType.ANALYSIS_COMPLETED,
            data={"results": results},
            timestamp=asyncio.get_event_loop().time(),
            source="analysis_manager",
        )
        self.event_manager.emit(completed_event)

        return results

    def get_analysis_progress(self) -> float:
        """Get current analysis progress."""
        return self.analysis_progress

    def validate_analysis_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate analysis configuration."""
        errors = validation_service.validate_data(config, "airfoil")

        event = TUIEvent(
            type=TUIEventType.VALIDATION_RESULT,
            data={"errors": errors, "valid": len(errors) == 0},
            timestamp=asyncio.get_event_loop().time(),
            source="analysis_manager",
        )
        self.event_manager.emit(event)

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "summary": validation_service.get_validation_summary(errors)
            if errors
            else "Validation passed",
        }


class TUIExportManager:
    """TUI-specific export management."""

    def __init__(self, event_manager: TUIEventManager):
        self.event_manager = event_manager

    def export_data(self, data: Any, filename: str, format_type: str) -> bool:
        """Export data and emit event."""
        success = export_service.export_data(data, filename, format_type)

        event = TUIEvent(
            type=TUIEventType.EXPORT_COMPLETED,
            data={
                "success": success,
                "filename": filename,
                "format": format_type,
                "error": None if success else "Export failed",
            },
            timestamp=asyncio.get_event_loop().time(),
            source="export_manager",
        )
        self.event_manager.emit(event)

        return success

    def generate_report(
        self,
        data: Dict[str, Any],
        report_type: str = "summary",
    ) -> str:
        """Generate report and emit event."""
        try:
            report = export_service.create_report(data, report_type)

            event = TUIEvent(
                type=TUIEventType.EXPORT_COMPLETED,
                data={
                    "success": True,
                    "report_type": report_type,
                    "report_length": len(report),
                },
                timestamp=asyncio.get_event_loop().time(),
                source="export_manager",
            )
            self.event_manager.emit(event)

            return report
        except Exception as e:
            event = TUIEvent(
                type=TUIEventType.EXPORT_COMPLETED,
                data={"success": False, "error": str(e)},
                timestamp=asyncio.get_event_loop().time(),
                source="export_manager",
            )
            self.event_manager.emit(event)

            return f"Report generation failed: {e}"


class TUISettingsManager:
    """TUI-specific settings management."""

    def __init__(self, event_manager: TUIEventManager):
        self.event_manager = event_manager

    def update_setting(self, key: str, value: Any) -> None:
        """Update a setting and emit event."""
        config_manager.set(key, value)

        event = TUIEvent(
            type=TUIEventType.SETTINGS_CHANGED,
            data={"key": key, "value": value},
            timestamp=asyncio.get_event_loop().time(),
            source="settings_manager",
        )
        self.event_manager.emit(event)

    def get_settings(self) -> Dict[str, Any]:
        """Get current settings for TUI display."""
        return {
            "theme": config_manager.get("theme", "default"),
            "database_path": config_manager.get_database_path(),
            "auto_save": config_manager.get("auto_save", True),
            "show_progress": config_manager.get("show_progress", True),
            "confirm_exit": config_manager.get("confirm_exit", True),
            "max_history": config_manager.get("max_history", 100),
            "default_solver": config_manager.get("default_solver", "xfoil"),
            "default_reynolds": config_manager.get("default_reynolds", 1e6),
            "default_angles": config_manager.get("default_angles", "0:15:16"),
        }

    def apply_theme(self, theme_name: str) -> None:
        """Apply theme and emit event."""
        try:
            # Update theme in config
            config_manager.set("theme", theme_name)

            event = TUIEvent(
                type=TUIEventType.SETTINGS_CHANGED,
                data={"key": "theme", "value": theme_name},
                timestamp=asyncio.get_event_loop().time(),
                source="settings_manager",
            )
            self.event_manager.emit(event)

        except Exception as e:
            # Emit error event
            error_event = TUIEvent(
                type=TUIEventType.NOTIFICATION,
                data={"message": f"Theme application failed: {e}", "level": "error"},
                timestamp=asyncio.get_event_loop().time(),
                source="settings_manager",
            )
            self.event_manager.emit(error_event)


class TUIIntegration:
    """Main TUI integration class that coordinates all managers."""

    def __init__(self):
        self.event_manager = TUIEventManager()
        self.session_manager = TUISessionManager(self.event_manager)
        self.workflow_manager = TUIWorkflowManager(self.event_manager)
        self.analysis_manager = TUIAnalysisManager(self.event_manager)
        self.export_manager = TUIExportManager(self.event_manager)
        self.settings_manager = TUISettingsManager(self.event_manager)

    def setup_notification_handler(self, app: App) -> None:
        """Setup notification handling for the TUI app."""

        def notification_handler(event: TUIEvent) -> None:
            if event.type == TUIEventType.NOTIFICATION:
                message = event.data.get("message", "")
                level = event.data.get("level", "info")
                app.notify(message, severity=level)

        self.event_manager.subscribe(TUIEventType.NOTIFICATION, notification_handler)

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status for TUI display."""
        return {
            "session": self.session_manager.update_session_info(),
            "workflow": {
                "current": self.workflow_manager.current_workflow,
                "progress": self.workflow_manager.get_workflow_progress(),
            },
            "analysis": {
                "current": self.analysis_manager.current_analysis is not None,
                "progress": self.analysis_manager.get_analysis_progress(),
            },
            "settings": self.settings_manager.get_settings(),
            "recent_events": len(self.event_manager.get_recent_events()),
        }


# Global TUI integration instance
tui_integration = TUIIntegration()
