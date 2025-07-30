#!/usr/bin/env python3
"""ICARUS CLI v2.0 - Textual TUI Application

This module provides a modern, interactive Terminal User Interface (TUI) for ICARUS
using Textual framework. It integrates all core features including session management,
workflow automation, validation, and export services.

This is the streamlined version of the TUI application, optimized for performance
and maintainability with enhanced aerospace-inspired UI design and advanced visualization
capabilities.
"""

import asyncio
import sys
import traceback
from io import StringIO
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

from textual.app import App
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.containers import Horizontal
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Button
from textual.widgets import DataTable
from textual.widgets import Footer
from textual.widgets import Input
from textual.widgets import Label
from textual.widgets import Log
from textual.widgets import ProgressBar
from textual.widgets import Select
from textual.widgets import Static
from textual.widgets import TabbedContent
from textual.widgets import TabPane
from textual.widgets import TextArea
from textual.widgets import Tree

from cli.app.streamlined_event import get_event_system
from cli.core.session_manager import get_session_manager
from cli.core.ui_theme import get_theme_manager

# Import streamlined core modules
from cli.core.unified_config import get_config_manager

# Import visualization components
from cli.visualization.dashboard import ActionWidget
from cli.visualization.dashboard import ChartWidget
from cli.visualization.dashboard import Dashboard
from cli.visualization.dashboard import ProgressWidget
from cli.visualization.dashboard import StatWidget
from cli.visualization.dashboard import TableWidget
from cli.visualization.dashboard import Visualization3DWidget
from cli.visualization.interactive_dashboard import DashboardScreen
from cli.visualization.real_time_visualization import RealTimeVisualizer
from cli.visualization.real_time_visualization import SimulatedDataSource
from cli.visualization.renderer_3d import Renderer3D

# Import ICARUS modules
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ICARUS import __version__
    from ICARUS.database import Database

    ICARUS_AVAILABLE = True
except ImportError:
    __version__ = "2.0.0"
    Database = None
    ICARUS_AVAILABLE = False

# Get core managers
config_manager = get_config_manager()
session_manager = get_session_manager()
theme_manager = get_theme_manager()
event_system = get_event_system()


# Event types
class EventType:
    """Event types for the TUI application."""

    SESSION_UPDATED = "session_updated"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_COMPLETED = "analysis_completed"
    NOTIFICATION = "notification"


# Create workflow engine
class WorkflowEngine:
    """Simple workflow engine."""

    def get_available_workflows(self):
        return [
            {"name": "Airfoil Analysis", "steps": 4},
            {"name": "Wing Design", "steps": 6},
            {"name": "Aircraft Performance", "steps": 8},
        ]


workflow_engine = WorkflowEngine()


class NamespaceManager:
    """Manages objects created in the REPL namespace."""

    def __init__(self):
        self.namespace: Dict[str, Any] = {}
        self.object_history: List[Dict[str, Any]] = []

    def add_object(self, name: str, obj: Any, obj_type: str = "unknown") -> None:
        """Add an object to the namespace."""
        self.namespace[name] = obj
        self.object_history.append(
            {
                "name": name,
                "type": obj_type,
                "timestamp": asyncio.get_event_loop().time(),
                "repr": str(obj)[:100] + "..." if len(str(obj)) > 100 else str(obj),
            },
        )

    def get_object(self, name: str) -> Any:
        """Get an object from the namespace."""
        return self.namespace.get(name)

    def list_objects(self) -> List[Dict[str, Any]]:
        """List all objects in the namespace."""
        return [
            {
                "name": name,
                "type": type(obj).__name__,
                "repr": str(obj)[:50] + "..." if len(str(obj)) > 50 else str(obj),
            }
            for name, obj in self.namespace.items()
        ]

    def clear_namespace(self) -> None:
        """Clear all objects from the namespace."""
        self.namespace.clear()
        self.object_history.clear()


class REPLCodeEditor(TextArea):
    """Enhanced code editor for the REPL with better navigation and copy/paste."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.placeholder = """# ICARUS REPL - Interactive Code Execution
# Create airfoils, airplanes, and other objects
#
# Keyboard shortcuts:
# Ctrl+Enter: Execute code (use button or binding)
# Ctrl+C: Copy selection
# Ctrl+V: Paste
# Ctrl+Z: Undo
# Ctrl+Y: Redo
# Ctrl+A: Select all
#
# Examples:
#
# from ICARUS.airfoils import Airfoil
# naca2412 = Airfoil.naca("2412")
# namespace.add_object("naca2412", naca2412, "airfoil")
#
# from ICARUS.vehicle import Airplane
# boeing737 = Airplane.from_file("boeing737.json")
# namespace.add_object("boeing737", boeing737, "airplane")
#
# Run your code and objects will be available in the app!"""
        self.code_history = []

    def add_to_history(self, code):
        """Add executed code to history."""
        if code.strip() and code not in self.code_history:
            self.code_history.append(code)
            # Keep only last 50 entries
            if len(self.code_history) > 50:
                self.code_history.pop(0)


class REPLOutput(Log):
    """Enhanced output area for REPL execution results."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_history = []
        self.write("REPL ready. Enter code and press Ctrl+Enter to execute.")

    def write(self, text, **kwargs):
        """Enhanced write with history and better formatting."""
        super().write(text, **kwargs)
        self.output_history.append(text)
        # Keep only last 1000 lines
        if len(self.output_history) > 1000:
            self.output_history = self.output_history[-1000:]

    def clear(self):
        """Clear output and reset history."""
        super().clear()
        self.output_history.clear()
        self.write("Output cleared.")


class ObjectBrowser(DataTable):
    """Enhanced browser for viewing objects in the namespace."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_columns("Name", "Type", "Preview")

    def update_objects(self, objects: List[Dict[str, Any]]) -> None:
        """Update the object browser with current namespace objects."""
        self.clear()
        self.add_columns("Name", "Type", "Preview")

        for obj in objects:
            self.add_row(obj["name"], obj["type"], obj["repr"])


class REPLPanel(Container):
    """Main REPL panel with code editor, output, and object browser."""

    def __init__(self, namespace_manager: NamespaceManager, **kwargs):
        super().__init__(**kwargs)
        self.namespace_manager = namespace_manager

    def compose(self) -> ComposeResult:
        with Horizontal():
            # Left side: Code editor and output
            with Vertical():
                yield Label("Code Editor", classes="section-title")
                yield REPLCodeEditor(id="repl_editor")
                with Horizontal():
                    yield Button(
                        "Execute (Ctrl+Enter)",
                        id="execute_code_btn",
                        variant="primary",
                    )
                    yield Button(
                        "Clear Output",
                        id="clear_output_btn",
                        variant="default",
                    )
                yield Label("Output", classes="section-title")
                yield REPLOutput(id="repl_output")

            # Right side: Object browser
            with Vertical():
                yield Label("Namespace Objects", classes="section-title")
                yield ObjectBrowser(id="object_browser")
                with Horizontal():
                    yield Button("Refresh", id="refresh_objects_btn", variant="default")
                    yield Button(
                        "Clear Namespace",
                        id="clear_namespace_btn",
                        variant="warning",
                    )


class ICARUSHeader(Static):
    """Custom header for ICARUS TUI."""

    def compose(self) -> ComposeResult:
        yield Label(f"ICARUS Aerodynamics v{__version__}", classes="title")
        yield Label("Advanced Aircraft Design & Analysis", classes="subtitle")


class SessionInfo(Static):
    """Display session information."""

    session_id = reactive("")
    duration = reactive("")
    workflow = reactive("")

    def on_mount(self) -> None:
        """Update session info when mounted."""
        self.update_session_info()
        # Subscribe to session updates
        event_system.subscribe(EventType.SESSION_UPDATED, self._on_session_updated)

    def update_session_info(self) -> None:
        """Update session information display."""
        try:
            session_info = session_manager.get_session_info()
            self.session_id = session_info.get("session_id", "Unknown")
            self.duration = session_info.get("duration", "0:00:00")

            # Get current workflow from config
            self.workflow = config_manager.get("current_workflow", "None")

            self.watch_session()
        except Exception as e:
            self.session_id = "Error"
            self.duration = "0:00:00"
            self.workflow = "None"
            self.watch_session()
            self.app.log.error(f"Error updating session info: {e}")

    async def _on_session_updated(self, data: Dict[str, Any]) -> None:
        """Handle session update events."""
        self.update_session_info()

    def watch_session(self) -> None:
        """Watch for session changes."""
        self.update(
            f"Session: {self.session_id[:8] if len(self.session_id) > 8 else self.session_id}... | Duration: {self.duration} | Workflow: {self.workflow}",
        )


class WorkflowTree(Tree):
    """Tree widget for displaying workflows."""

    def __init__(self, **kwargs):
        super().__init__("Workflows", **kwargs)
        self.workflows = {}

    def on_mount(self) -> None:
        """Load workflows when mounted."""
        self.load_workflows()
        # Subscribe to workflow events
        event_system.subscribe(EventType.WORKFLOW_STARTED, self._on_workflow_started)
        event_system.subscribe(
            EventType.WORKFLOW_COMPLETED,
            self._on_workflow_completed,
        )

    def load_workflows(self) -> None:
        """Load available workflows into the tree."""
        self.clear()
        try:
            workflows = workflow_engine.get_available_workflows()
        except Exception as e:
            self.app.log.error(f"Error loading workflows: {e}")
            workflows = [{"name": "Sample Workflow", "steps": 3}]

        for workflow in workflows:
            workflow_node = self.root.add(workflow["name"], data=workflow)
            # Add placeholder for steps (would be populated from actual workflow)
            workflow_node.add(
                f"• {workflow['steps']} steps",
                data={"type": "step_count"},
            )

        self.workflows = {w["name"]: w for w in workflows}

    async def _on_workflow_started(self, data: Dict[str, Any]) -> None:
        """Handle workflow started events."""
        workflow_name = data.get("workflow")
        if workflow_name:
            # Update config with current workflow
            config_manager.set("current_workflow", workflow_name)

            # Update UI
            for node in self.root.children:
                if node.label == workflow_name:
                    node.expand()
                    self.select_node(node)
                    break

    async def _on_workflow_completed(self, data: Dict[str, Any]) -> None:
        """Handle workflow completed events."""
        workflow_name = data.get("workflow")
        if workflow_name:
            # Update workflow status in tree
            for node in self.root.children:
                if node.label == workflow_name:
                    # Add completion status
                    node.add(
                        "✓ Completed",
                        data={"type": "status", "status": "completed"},
                    )
                    break


class AirfoilAnalysisForm(Container):
    """Form for airfoil analysis configuration."""

    def compose(self) -> ComposeResult:
        yield Label("Airfoil Analysis", classes="form-title")
        with Vertical():
            yield Label("Airfoil:")
            yield Input(
                placeholder="Enter airfoil name or select from namespace",
                id="airfoil_name",
            )
            yield Label("Solver:")
            yield Select(
                (
                    ("XFoil", "xfoil"),
                    ("Foil2Wake", "foil2wake"),
                    ("OpenFoam", "openfoam"),
                ),
                value="xfoil",
                id="airfoil_solver_select",
            )
            yield Label("Reynolds Number:")
            yield Input(placeholder="1e6", value="1000000", id="airfoil_reynolds_input")
            yield Label("Angle Range:")
            yield Input(
                placeholder="0:15:16",
                value="0:15:16",
                id="airfoil_angles_input",
            )
            with Horizontal():
                yield Button("Validate", id="validate_airfoil_btn", variant="primary")
                yield Button(
                    "Run Analysis",
                    id="run_airfoil_analysis_btn",
                    variant="success",
                )


class AirplaneAnalysisForm(Container):
    """Form for airplane analysis configuration."""

    def compose(self) -> ComposeResult:
        yield Label("Airplane Analysis", classes="form-title")
        with Vertical():
            yield Label("Airplane:")
            yield Input(
                placeholder="Enter airplane name or select from namespace",
                id="airplane_name",
            )
            yield Label("Solver:")
            yield Select(
                (("AVL", "avl"), ("GenuVP", "gnvp3"), ("OpenFoam", "openfoam")),
                value="avl",
                id="airplane_solver_select",
            )
            yield Label("Flight State:")
            yield Input(placeholder="e.g. ISA, 10000 ft", id="flight_state_input")
            with Horizontal():
                yield Button("Validate", id="validate_airplane_btn", variant="primary")
                yield Button(
                    "Run Analysis",
                    id="run_airplane_analysis_btn",
                    variant="success",
                )


class ResultsViewer(DataTable):
    """Data table for displaying analysis results."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_columns("Parameter", "Value", "Unit")

    def add_result(self, parameter: str, value: str, unit: str = "") -> None:
        """Add a result row to the table."""
        self.add_row(parameter, value, unit)

    def clear_results(self) -> None:
        """Clear all results."""
        self.clear()
        self.add_columns("Parameter", "Value", "Unit")

    def on_mount(self) -> None:
        """Subscribe to analysis events when mounted."""
        event_system.subscribe(
            EventType.ANALYSIS_COMPLETED,
            self._on_analysis_completed,
        )

    async def _on_analysis_completed(self, data: Dict[str, Any]) -> None:
        """Handle analysis completion events."""
        results = data.get("results", {})
        self.update_with_results(results)

        # Add result to session
        if "id" in results:
            await session_manager.add_result(results["id"])

    def update_with_results(self, results: Dict[str, Any]) -> None:
        """Update the results viewer with new results."""
        self.clear_results()

        if not results:
            return

        # Add basic info
        self.add_result("Target", results.get("target", "Unknown"))
        self.add_result("Solver", results.get("solver", "Unknown"))
        self.add_result("Reynolds", f"{results.get('reynolds', 0):.0f}")
        self.add_result("Angle Range", results.get("angles", "Unknown"))

        # Add timestamp
        from datetime import datetime

        timestamp = results.get("timestamp", datetime.now().isoformat())
        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp)
                timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
        self.add_result("Timestamp", str(timestamp))

        # Add results
        result_data = results.get("results", {})
        for key, value in result_data.items():
            if isinstance(value, float):
                self.add_result(key.replace("_", " ").title(), f"{value:.3f}")
            else:
                self.add_result(key.replace("_", " ").title(), str(value))


class ProgressViewer(Container):
    """Container for progress tracking."""

    def compose(self) -> ComposeResult:
        yield Label("Progress", classes="section-title")
        yield ProgressBar(id="main_progress")
        yield Label("Ready", id="progress_status")

    def on_mount(self) -> None:
        """Subscribe to progress events when mounted."""
        event_system.subscribe(EventType.ANALYSIS_STARTED, self._on_analysis_started)
        event_system.subscribe(EventType.WORKFLOW_STARTED, self._on_workflow_started)

        # Add custom event for progress updates
        event_system.subscribe("progress_update", self._on_progress_update)

    async def _on_analysis_started(self, data: Dict[str, Any]) -> None:
        """Handle analysis start events."""
        try:
            progress_bar = self.query_one("#main_progress", ProgressBar)
            progress_status = self.query_one("#progress_status", Label)

            analysis_name = data.get("analysis", "Unknown")
            progress_bar.update(progress=0)
            progress_status.update(f"Analysis started: {analysis_name}")

            # Add analysis to session
            if "id" in data:
                await session_manager.add_analysis(data["id"])
        except Exception as e:
            self.app.log.error(f"Error updating progress for analysis start: {e}")

    async def _on_workflow_started(self, data: Dict[str, Any]) -> None:
        """Handle workflow start events."""
        try:
            progress_bar = self.query_one("#main_progress", ProgressBar)
            progress_status = self.query_one("#progress_status", Label)

            workflow_name = data.get("workflow", "Unknown")
            progress_bar.update(progress=0)
            progress_status.update(f"Workflow started: {workflow_name}")

            # Update config with current workflow
            config_manager.set("current_workflow", workflow_name)
        except Exception as e:
            self.app.log.error(f"Error updating progress for workflow start: {e}")

    async def _on_progress_update(self, data: Dict[str, Any]) -> None:
        """Handle progress update events."""
        try:
            progress = data.get("progress", 0.0)
            status = data.get("status", "Processing...")
            self.update_progress(progress, status)
        except Exception as e:
            self.app.log.error(f"Error handling progress update: {e}")

    def update_progress(self, progress: float, status: str) -> None:
        """Update progress display."""
        try:
            progress_bar = self.query_one("#main_progress", ProgressBar)
            progress_status = self.query_one("#progress_status", Label)

            progress_bar.update(progress=progress)
            progress_status.update(status)
        except Exception as e:
            self.app.log.error(f"Error updating progress: {e}")


class NotificationLog(Log):
    """Log widget for notifications."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.write("ICARUS TUI started. Welcome!")

    def on_mount(self) -> None:
        """Subscribe to notification events when mounted."""
        event_system.subscribe(EventType.NOTIFICATION, self._on_notification)

    async def _on_notification(self, data: Dict[str, Any]) -> None:
        """Handle notification events."""
        message = data.get("message", "Unknown notification")
        level = data.get("level", "info")
        self.add_notification(message, level)

    def add_notification(self, message: str, level: str = "info") -> None:
        """Add a notification to the log."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.write(f"[{timestamp}] {level.upper()}: {message}")


class ICARUSTUI(App):
    """Main ICARUS TUI application."""

    CSS_PATH = "tui_styles_aerospace.css"
    TITLE = "ICARUS Aerodynamics"
    SUB_TITLE = "Advanced Aircraft Design & Analysis"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+h", "show_help", "Help"),
        Binding("ctrl+s", "show_settings", "Settings"),
        Binding("ctrl+w", "show_workflows", "Workflows"),
        Binding("ctrl+a", "show_analysis", "Analysis"),
        Binding("ctrl+r", "show_results", "Results"),
        Binding("ctrl+n", "show_notifications", "Notifications"),
        Binding("ctrl+d", "show_dashboard", "Dashboard"),
        Binding("ctrl+shift+s", "save_session", "Save Session"),
        Binding("f5", "refresh", "Refresh"),
        Binding("ctrl+enter", "execute_repl", "Execute REPL"),
        Binding("f1", "launch_analysis_config", "New Analysis"),
        Binding("f3", "toggle_3d_view", "Toggle 3D View"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.database = None
        self.current_tab = "analysis"
        self.namespace_manager = NamespaceManager()

        # Initialize visualization components
        self.renderer_3d = Renderer3D()
        self.real_time_visualizer = RealTimeVisualizer()

        # Create 3D models
        self.renderer_3d.create_airfoil("naca0012", "0012")
        self.renderer_3d.create_airplane("boeing737", 35.8, 39.5)

        # Create data sources for real-time visualization
        self.data_sources = {
            "performance": SimulatedDataSource("performance", "scalar", 1.0),
            "system_metrics": SimulatedDataSource("system_metrics", "vector", 2.0),
        }

        # Performance tracking
        self.startup_time = asyncio.get_event_loop().time()

    async def on_mount(self) -> None:
        """Initialize the application when mounted."""
        await self.initialize_database()
        await self.setup_theme()
        await self.initialize_session()
        await self.initialize_visualizations()

        # Register screens
        self.install_screen(DashboardScreen(), name="dashboard")

        # Log startup time
        elapsed = asyncio.get_event_loop().time() - self.startup_time
        self.log.info(f"Application startup completed in {elapsed:.2f} seconds")

    async def initialize_visualizations(self) -> None:
        """Initialize visualization components."""
        try:
            # Add data sources to visualizer
            for name, source in self.data_sources.items():
                self.real_time_visualizer.add_data_source(source)

            # Start real-time visualizer
            await self.real_time_visualizer.start()

            self.log.info("Visualization components initialized")
        except Exception as e:
            self.log.error(f"Failed to initialize visualizations: {e}")

    async def initialize_database(self) -> None:
        """Initialize the ICARUS database."""
        try:
            if Database:
                # Get database path from config
                db_path = config_manager.get("data_directory", "./Data")

                # Initialize database
                self.database = Database(db_path)
                self.log.info("Database initialized successfully")
            else:
                self.log.warning("Database not available")
        except Exception as e:
            self.log.error(f"Failed to initialize database: {e}")

    async def setup_theme(self) -> None:
        """Setup the application theme."""
        try:
            # Get theme from config
            theme_name = config_manager.get("theme", "default")

            # Apply theme
            theme_manager.apply_theme(self, theme_name)
            self.log.info(f"Theme applied: {theme_name}")
        except Exception as e:
            self.log.warning(f"Could not apply theme: {e}")

    async def initialize_session(self) -> None:
        """Initialize session."""
        try:
            # Initialize session
            await session_manager.initialize()

            # Get session info
            session_info = session_manager.get_session_info()
            self.log.info(
                f"Session initialized: {session_info.get('session_id', 'unknown')}",
            )

            # Subscribe to session events
            event_system.subscribe(EventType.SESSION_UPDATED, self._on_session_updated)
        except Exception as e:
            self.log.error(f"Failed to initialize session: {e}")

    async def _on_session_updated(self, data: Dict[str, Any]) -> None:
        """Handle session update events."""
        try:
            # Update session info display
            session_info = session_manager.get_session_info()
            self.log.info(
                f"Session updated: {session_info.get('session_id', 'unknown')}",
            )

            # Update UI components
            session_display = self.query_one(SessionInfo, default=None)
            if session_display:
                session_display.update_session_info()
        except Exception as e:
            self.log.error(f"Error handling session update: {e}")

    def compose(self) -> ComposeResult:
        """Compose the main application layout."""
        yield ICARUSHeader()

        with TabbedContent():
            with TabPane("Dashboard", id="dashboard_tab"):
                yield self.create_dashboard()

            with TabPane("Analysis", id="analysis"):
                with TabbedContent():
                    with TabPane("Airfoil", id="airfoil_analysis_tab"):
                        with Vertical():
                            yield AirfoilAnalysisForm()
                            yield ProgressViewer()
                    with TabPane("Airplane", id="airplane_analysis_tab"):
                        with Vertical():
                            yield AirplaneAnalysisForm()
                            yield ProgressViewer()
                    with TabPane("3D View", id="3d_view_tab"):
                        yield self.create_3d_view()

            with TabPane("REPL", id="repl"):
                yield REPLPanel(self.namespace_manager)

            with TabPane("Workflows", id="workflows"):
                with Vertical():
                    yield WorkflowTree()
                    yield Button(
                        "Execute Selected",
                        id="execute_workflow_btn",
                        variant="primary",
                    )

            with TabPane("Results", id="results"):
                with TabbedContent():
                    with TabPane("Data", id="results_data_tab"):
                        yield ResultsViewer()
                    with TabPane("Visualization", id="results_viz_tab"):
                        yield self.create_visualization_panel()

            with TabPane("Settings", id="settings"):
                yield self.create_settings_form()

            with TabPane("Notifications", id="notifications"):
                yield NotificationLog()

        yield Footer()

    def create_settings_form(self) -> Container:
        """Create the settings form."""
        return Container(
            Label("Settings", classes="form-title"),
            Vertical(
                Label("Theme:"),
                Select(
                    (("Default", "default"), ("Dark", "dark"), ("Light", "light")),
                    value="default",
                    id="theme_select",
                ),
                Label("Database Path:"),
                Input(placeholder="Path to database", id="db_path_input"),
                Label("Auto Save:"),
                Select(
                    (("Enabled", "true"), ("Disabled", "false")),
                    value="true",
                    id="auto_save_select",
                ),
                Horizontal(
                    Button("Save Settings", id="save_settings_btn", variant="primary"),
                    Button(
                        "Reset to Defaults",
                        id="reset_settings_btn",
                        variant="default",
                    ),
                ),
            ),
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        button_id = event.button.id

        # REPL buttons
        if button_id == "execute_code_btn":
            self.execute_repl_code()
        elif button_id == "clear_output_btn":
            self.clear_repl_output()
        elif button_id == "refresh_objects_btn":
            self.refresh_object_browser()
        elif button_id == "clear_namespace_btn":
            self.clear_namespace()

        # Airfoil analysis
        elif button_id == "validate_airfoil_btn":
            self.validate_airfoil_form()
        elif button_id == "run_airfoil_analysis_btn":
            self.launch_analysis_config()
        # Airplane analysis
        elif button_id == "validate_airplane_btn":
            self.validate_airplane_form()
        elif button_id == "run_airplane_analysis_btn":
            self.launch_analysis_config()
        # New analysis button
        elif button_id == "new_analysis_btn":
            self.launch_analysis_config()
        # General
        elif button_id == "execute_workflow_btn":
            self.execute_workflow()
        elif button_id == "save_settings_btn":
            self.save_settings()
        elif button_id == "reset_settings_btn":
            self.reset_settings()

    def execute_repl_code(self) -> None:
        """Execute code from the REPL editor."""
        try:
            code_editor = self.query_one("#repl_editor", REPLCodeEditor)
            output = self.query_one("#repl_output", REPLOutput)

            code = code_editor.text
            if not code.strip():
                output.write("No code to execute.")
                return

            # Add to history
            code_editor.add_to_history(code)

            # Capture stdout/stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            stdout_capture = StringIO()
            stderr_capture = StringIO()

            try:
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture

                # Create execution namespace with access to namespace manager
                exec_namespace = {
                    "namespace": self.namespace_manager,
                    "add_object": self.namespace_manager.add_object,
                    "get_object": self.namespace_manager.get_object,
                    "list_objects": self.namespace_manager.list_objects,
                }

                # Execute the code
                exec(code, exec_namespace)

                # Get captured output
                stdout_output = stdout_capture.getvalue()
                stderr_output = stderr_capture.getvalue()

                # Display results
                if stdout_output:
                    output.write(f"Output:\n{stdout_output}")
                if stderr_output:
                    output.write(f"Errors:\n{stderr_output}")
                else:
                    output.write("Code executed successfully.")

                # Update object browser
                self.refresh_object_browser()

            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        except Exception:
            output = self.query_one("#repl_output", REPLOutput)
            output.write(f"Error executing code:\n{traceback.format_exc()}")

    def clear_repl_output(self) -> None:
        """Clear the REPL output."""
        output = self.query_one("#repl_output", REPLOutput)
        output.clear()
        output.write("Output cleared.")

    def refresh_object_browser(self) -> None:
        """Refresh the object browser with current namespace objects."""
        try:
            browser = self.query_one("#object_browser", ObjectBrowser)
            objects = self.namespace_manager.list_objects()
            browser.update_objects(objects)
        except Exception as e:
            self.log.error(f"Failed to refresh object browser: {e}")

    def clear_namespace(self) -> None:
        """Clear all objects from the namespace."""
        self.namespace_manager.clear_namespace()
        self.refresh_object_browser()
        output = self.query_one("#repl_output", REPLOutput)
        output.write("Namespace cleared.")

    def action_execute_repl(self) -> None:
        """Action to execute REPL code (bound to Ctrl+Enter)."""
        self.execute_repl_code()

    def validate_airfoil_form(self) -> None:
        try:
            name = self.query_one("#airfoil_name", Input).value
            reynolds = self.query_one("#airfoil_reynolds_input", Input).value
            angles = self.query_one("#airfoil_angles_input", Input).value
            if not name:
                self.log.error("Please enter an airfoil name")
                return
            float(reynolds)
            self.log.info("Airfoil form validation passed")
        except Exception as e:
            self.log.error(f"Airfoil validation error: {e}")

    def run_airfoil_analysis(self) -> None:
        self.log.info("Running airfoil analysis (demo)...")
        # Simulate analysis and update results
        results_viewer = self.query_one(ResultsViewer)
        results_viewer.update_with_results(
            {
                "target": "NACA 2412",
                "solver": "XFoil",
                "reynolds": 1000000,
                "angles": "0:15:16",
                "results": {"cl_max": 1.234, "cd_min": 0.012, "alpha_stall": 12.5},
            },
        )
        self.log.info("Airfoil analysis completed.")

    def validate_airplane_form(self) -> None:
        try:
            name = self.query_one("#airplane_name", Input).value
            flight_state = self.query_one("#flight_state_input", Input).value
            if not name:
                self.log.error("Please enter an airplane name")
                return
            if not flight_state:
                self.log.error("Please enter a flight state")
                return
            self.log.info("Airplane form validation passed")
        except Exception as e:
            self.log.error(f"Airplane validation error: {e}")

    def run_airplane_analysis(self) -> None:
        self.log.info("Running airplane analysis (demo)...")
        # Simulate analysis and update results
        results_viewer = self.query_one(ResultsViewer)
        results_viewer.update_with_results(
            {
                "target": "Boeing 737",
                "solver": "AVL",
                "flight_state": "ISA, 10000 ft",
                "results": {"cl": 0.95, "cd": 0.032, "cm": -0.12},
            },
        )
        self.log.info("Airplane analysis completed.")

    def execute_workflow(self) -> None:
        """Execute the selected workflow."""
        try:
            workflow_tree = self.query_one(WorkflowTree)
            selected_node = workflow_tree.cursor_node

            if selected_node and selected_node.data and "name" in selected_node.data:
                workflow_name = selected_node.data["name"]
                self.log.info(f"Executing workflow: {workflow_name}")

                # Simulate workflow execution
                import time

                time.sleep(2)

                self.log.info(f"Workflow {workflow_name} completed")
            else:
                self.log.warning("Please select a workflow to execute")

        except Exception as e:
            self.log.error(f"Workflow execution failed: {e}")

    def launch_analysis_config(self) -> None:
        """Launch the new analysis configuration screen."""
        try:
            from tui.screens.analysis_screen import AnalysisScreen

            self.push_screen(AnalysisScreen())
        except ImportError as e:
            self.notify(f"Could not load analysis screen: {e}", severity="error")
        except Exception as e:
            self.notify(
                f"Error launching analysis configuration: {e}",
                severity="error",
            )

    def action_launch_analysis_config(self) -> None:
        """Action to launch analysis configuration (bound to F1)."""
        self.launch_analysis_config()

    def create_workflow_template(self) -> None:
        """Create a new workflow template."""
        self.log.info("Creating workflow template...")

    def save_settings(self) -> None:
        """Save current settings."""
        try:
            theme_select = self.query_one("#theme_select", Select)
            db_path_input = self.query_one("#db_path_input", Input)
            auto_save_select = self.query_one("#auto_save_select", Select)

            settings = {
                "theme": theme_select.value,
                "database_path": db_path_input.value,
                "auto_save": auto_save_select.value == "true",
            }

            if config_manager:
                for key, value in settings.items():
                    config_manager.set(key, value)

            self.log.info("Settings saved successfully")

        except Exception as e:
            self.log.error(f"Failed to save settings: {e}")

    def reset_settings(self) -> None:
        """Reset settings to defaults."""
        try:
            if config_manager and hasattr(config_manager, "reset_to_defaults"):
                config_manager.reset_to_defaults()
                self.log.info("Settings reset to defaults")
            else:
                self.log.warning("Settings reset not available")
        except Exception as e:
            self.log.error(f"Failed to reset settings: {e}")

    def action_show_help(self) -> None:
        """Show help information."""
        self.log.info("Help: Use Ctrl+Q to quit, Ctrl+H for help")

    def action_show_settings(self) -> None:
        """Show settings tab."""
        self.query_one(TabbedContent).active = "settings"

    def action_show_workflows(self) -> None:
        """Show workflows tab."""
        self.query_one(TabbedContent).active = "workflows"

    def action_show_analysis(self) -> None:
        """Show analysis tab."""
        self.query_one(TabbedContent).active = "analysis"

    def action_show_results(self) -> None:
        """Show results tab."""
        self.query_one(TabbedContent).active = "results"

    def action_show_notifications(self) -> None:
        """Show notifications tab."""
        self.query_one(TabbedContent).active = "notifications"

    def action_save_session(self) -> None:
        """Save current session."""
        try:
            if session_manager and hasattr(session_manager, "_save_session"):
                session_manager._save_session()
                self.log.info("Session saved")
            else:
                self.log.warning("Session save not available")
        except Exception as e:
            self.log.error(f"Failed to save session: {e}")

    def action_refresh(self) -> None:
        """Refresh the application."""
        self.log.info("Refreshing application...")
        self.update_session_info()


def main():
    """Main entry point for the TUI application."""
    app = ICARUSTUI()
    app.run()


if __name__ == "__main__":
    main()

    def create_dashboard(self) -> Dashboard:
        """Create the main dashboard."""
        return Dashboard(
            widgets=[
                StatWidget(
                    title="Active Session",
                    value="1",
                    label="Current Sessions",
                    id="session_stat",
                ),
                StatWidget(
                    title="Analysis Jobs",
                    value="3",
                    label="Running Jobs",
                    id="jobs_stat",
                ),
                ChartWidget(
                    title="Performance Metrics",
                    chart_type="line",
                    id="performance_chart",
                    data={
                        "x": [0, 1, 2, 3, 4, 5],
                        "y": [0, 2, 1, 3, 2, 4],
                        "xlabel": "Time (min)",
                        "ylabel": "CPU Usage (%)",
                    },
                ),
                TableWidget(
                    title="Recent Results",
                    columns=["ID", "Type", "Status", "Time"],
                    rows=[
                        ["A001", "Airfoil", "Complete", "10:30"],
                        ["A002", "Wing", "Running", "10:45"],
                        ["A003", "Airplane", "Queued", "11:00"],
                    ],
                    id="results_table",
                ),
                ProgressWidget(
                    title="Current Analysis",
                    progress=0.65,
                    status="Processing wing analysis...",
                    id="analysis_progress",
                ),
                ActionWidget(
                    title="Quick Actions",
                    actions=[
                        {
                            "id": "new_analysis",
                            "label": "New Analysis",
                            "variant": "primary",
                        },
                        {
                            "id": "view_results",
                            "label": "View Results",
                            "variant": "default",
                        },
                        {
                            "id": "export_data",
                            "label": "Export Data",
                            "variant": "default",
                        },
                    ],
                    id="quick_actions",
                ),
                Visualization3DWidget(
                    title="3D Model Viewer",
                    model_data={"type": "airplane", "name": "Boeing 737"},
                    id="model_viewer",
                ),
            ],
            layout="grid",
            id="main_dashboard",
        )

    def create_3d_view(self) -> Container:
        """Create the 3D view panel."""
        return Container(
            Label("3D Visualization", classes="section-title"),
            Static(id="3d_view_content", classes="visualization-3d"),
            Horizontal(
                Button("Rotate X", id="rotate_x_btn", variant="default"),
                Button("Rotate Y", id="rotate_y_btn", variant="default"),
                Button("Rotate Z", id="rotate_z_btn", variant="default"),
                Button("Reset", id="reset_view_btn", variant="default"),
            ),
            Horizontal(
                Label("Model:"),
                Select(
                    (("NACA 0012", "naca0012"), ("Boeing 737", "boeing737")),
                    value="boeing737",
                    id="model_select",
                ),
                Button("Load Model", id="load_model_btn", variant="primary"),
            ),
            id="3d_view_panel",
        )

    def create_visualization_panel(self) -> Container:
        """Create the visualization panel."""
        return Container(
            Label("Advanced Visualization", classes="section-title"),
            Horizontal(
                Button("Line Chart", id="line_chart_btn", variant="default"),
                Button("Bar Chart", id="bar_chart_btn", variant="default"),
                Button("Gauge", id="gauge_btn", variant="default"),
                Button("Real-time", id="real_time_btn", variant="primary"),
            ),
            Static(id="visualization_content", classes="chart-container"),
            id="visualization_panel",
        )

    def action_show_dashboard(self) -> None:
        """Show the dashboard screen."""
        self.push_screen("dashboard")

    def action_toggle_3d_view(self) -> None:
        """Toggle the 3D view."""
        # Update the 3D view content
        view_content = self.query_one("#3d_view_content", Static)
        if view_content:
            # Render the active model
            rendered = self.renderer_3d.render_to_text(width=60, height=15)
            view_content.update(rendered)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select change events."""
        select_id = event.select.id

        if select_id == "model_select":
            model_name = event.value
            self.renderer_3d.set_active_mesh(model_name)
            self.action_toggle_3d_view()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        button_id = event.button.id

        # REPL buttons
        if button_id == "execute_code_btn":
            self.execute_repl_code()
        elif button_id == "clear_output_btn":
            self.clear_repl_output()
        elif button_id == "refresh_objects_btn":
            self.refresh_object_browser()
        elif button_id == "clear_namespace_btn":
            self.clear_namespace()

        # 3D view buttons
        elif button_id == "rotate_x_btn":
            self.renderer_3d.rotate_mesh(self.renderer_3d.active_mesh, 30, 0, 0)
            self.action_toggle_3d_view()
        elif button_id == "rotate_y_btn":
            self.renderer_3d.rotate_mesh(self.renderer_3d.active_mesh, 0, 30, 0)
            self.action_toggle_3d_view()
        elif button_id == "rotate_z_btn":
            self.renderer_3d.rotate_mesh(self.renderer_3d.active_mesh, 0, 0, 30)
            self.action_toggle_3d_view()
        elif button_id == "reset_view_btn":
            # Reset the active mesh
            model_name = self.renderer_3d.active_mesh
            if model_name == "naca0012":
                self.renderer_3d.create_airfoil("naca0012", "0012")
            elif model_name == "boeing737":
                self.renderer_3d.create_airplane("boeing737", 35.8, 39.5)
            self.action_toggle_3d_view()
        elif button_id == "load_model_btn":
            model_select = self.query_one("#model_select", Select)
            model_name = model_select.value
            self.renderer_3d.set_active_mesh(model_name)
            self.action_toggle_3d_view()

        # Visualization buttons
        elif button_id == "line_chart_btn":
            viz_content = self.query_one("#visualization_content", Static)
            viz_content.update("""
            ╭──────────────────────────────────────────────────────────╮
            │                                                          │
            │                                                          │
            │                      ╭─╮                                 │
            │                     ╭╯ │                                 │
            │                    ╭╯  │                                 │
            │                   ╭╯   │                 ╭───╮           │
            │                  ╭╯    │                ╭╯   ╰╮          │
            │                 ╭╯     ╰╮              ╭╯     ╰╮         │
            │                ╭╯       ╰╮            ╭╯       ╰╮        │
            │               ╭╯         ╰────────────╯         ╰╮       │
            │              ╭╯                                   ╰╮     │
            │             ╭╯                                     ╰╮    │
            │            ╭╯                                       ╰╮   │
            │           ╭╯                                         ╰╮  │
            │          ╭╯                                           ╰╮ │
            │         ╭╯                                             ╰╮│
            │        ╭╯                                               ╰│
            │                                                          │
            ╰──────────────────────────────────────────────────────────╯
            """)
        elif button_id == "bar_chart_btn":
            viz_content = self.query_one("#visualization_content", Static)
            viz_content.update("""
            ╭──────────────────────────────────────────────────────────╮
            │                                                          │
            │  ██                                                      │
            │  ██                                                      │
            │  ██            ██                                        │
            │  ██            ██                                        │
            │  ██            ██            ██                          │
            │  ██            ██            ██                          │
            │  ██            ██            ██            ██            │
            │  ██            ██            ██            ██            │
            │  ██            ██            ██            ██            │
            │  ██            ██            ██            ██            │
            │  ██            ██            ██            ██            │
            │  ██            ██            ██            ██            │
            │  ██            ██            ██            ██            │
            │  ██            ██            ██            ██            │
            │ ─────────────────────────────────────────────────────── │
            │  CL             CD            CM            L/D          │
            │                                                          │
            ╰──────────────────────────────────────────────────────────╯
            """)
        elif button_id == "gauge_btn":
            viz_content = self.query_one("#visualization_content", Static)
            viz_content.update("""
            ╭──────────────────────────────────────────────────────────╮
            │                                                          │
            │                    CPU Usage: 65%                        │
            │                                                          │
            │  0% ████████████████████████████░░░░░░░░░░░░░░░░░░░ 100% │
            │                                                          │
            │                                                          │
            │                   Memory Usage: 42%                      │
            │                                                          │
            │  0% ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░ 100% │
            │                                                          │
            │                                                          │
            │                    Disk I/O: 23%                         │
            │                                                          │
            │  0% ███████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 100% │
            │                                                          │
            │                                                          │
            ╰──────────────────────────────────────────────────────────╯
            """)
        elif button_id == "real_time_btn":
            viz_content = self.query_one("#visualization_content", Static)
            viz_content.update("""
            ╭──────────────────────────────────────────────────────────╮
            │                                                          │
            │              Real-time Visualization Active              │
            │                                                          │
            │  ╭───────────────────────────────────────────────────╮   │
            │  │                                                   │   │
            │  │  Performance data streaming...                    │   │
            │  │                                                   │   │
            │  │  CPU: ████████████████░░░░░░░░░░ 65%              │   │
            │  │  MEM: ████████████░░░░░░░░░░░░░░ 48%              │   │
            │  │  I/O: ██████░░░░░░░░░░░░░░░░░░░░ 23%              │   │
            │  │                                                   │   │
            │  │  Last update: 10:45:32                           │   │
            │  │                                                   │   │
            │  ╰───────────────────────────────────────────────────╯   │
            │                                                          │
            │  [Open Dashboard for full real-time visualization]       │
            │                                                          │
            ╰──────────────────────────────────────────────────────────╯
            """)

        # Airfoil analysis
        elif button_id == "validate_airfoil_btn":
            self.validate_airfoil_form()
        elif button_id == "run_airfoil_analysis_btn":
            self.launch_analysis_config()
        # Airplane analysis
        elif button_id == "validate_airplane_btn":
            self.validate_airplane_form()
        elif button_id == "run_airplane_analysis_btn":
            self.launch_analysis_config()
        # New analysis button
        elif button_id == "new_analysis_btn":
            self.launch_analysis_config()
        # General
        elif button_id == "execute_workflow_btn":
            self.execute_workflow()
        elif button_id == "save_settings_btn":
            self.save_settings()
        elif button_id == "reset_settings_btn":
            self.reset_settings()
