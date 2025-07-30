"""Analysis Execution Screen

This screen provides analysis execution with real-time progress tracking,
cancellation support, and live status updates.
"""

import asyncio
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.containers import Horizontal
from textual.containers import ScrollableContainer
from textual.containers import Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Button
from textual.widgets import Collapsible
from textual.widgets import DataTable
from textual.widgets import Label
from textual.widgets import Log
from textual.widgets import ProgressBar
from textual.widgets import Select
from textual.widgets import Switch

# Import integration modules
try:
    from ...integration.analysis_service import AnalysisService
    from ...integration.models import AnalysisConfig
    from ...integration.models import AnalysisProgress
    from ...integration.models import AnalysisResult
    from ...integration.models import AnalysisType
    from ...integration.models import ProcessedResult
    from ...integration.models import SolverType
except ImportError:
    AnalysisService = None
    AnalysisConfig = None
    AnalysisType = None
    SolverType = None
    AnalysisResult = None
    AnalysisProgress = None
    ProcessedResult = None


class ProgressTracker(Container):
    """Real-time progress tracking widget."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_progress = reactive(0.0)
        self.current_step = reactive("Ready")
        self.total_steps = reactive(5)
        self.completed_steps = reactive(0)
        self.start_time = None
        self.estimated_completion = reactive("")

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Analysis Progress", classes="section-title")

            # Main progress bar
            yield ProgressBar(id="main_progress_bar")

            # Progress details
            with Horizontal():
                yield Label("Status:", classes="field-label")
                yield Label("Ready", id="progress_status", classes="status-text")

            with Horizontal():
                yield Label("Step:", classes="field-label")
                yield Label("0 / 5", id="step_counter", classes="status-text")

            with Horizontal():
                yield Label("Elapsed:", classes="field-label")
                yield Label("00:00:00", id="elapsed_time", classes="status-text")

            with Horizontal():
                yield Label("Estimated:", classes="field-label")
                yield Label("--:--:--", id="estimated_time", classes="status-text")

            # Detailed step progress
            yield Label("Step Details", classes="subsection-title")
            yield Container(id="step_details_container")

    def start_tracking(self, total_steps: int = 5) -> None:
        """Start progress tracking."""
        self.start_time = datetime.now()
        self.total_steps = total_steps
        self.completed_steps = 0
        self.current_progress = 0.0
        self.current_step = "Starting analysis..."

        # Update UI
        self._update_display()

        # Start elapsed time updater
        self._start_time_updater()

    def update_progress(self, progress: AnalysisProgress) -> None:
        """Update progress from AnalysisProgress object."""
        self.current_progress = progress.progress_percent
        self.current_step = progress.current_step
        self.completed_steps = progress.completed_steps
        self.total_steps = progress.total_steps

        self._update_display()
        self._update_step_details(progress)

    def _update_display(self) -> None:
        """Update the progress display."""
        try:
            # Update progress bar
            progress_bar = self.query_one("#main_progress_bar", ProgressBar)
            progress_bar.update(progress=self.current_progress)

            # Update status
            status_label = self.query_one("#progress_status", Label)
            status_label.update(self.current_step)

            # Update step counter
            step_counter = self.query_one("#step_counter", Label)
            step_counter.update(f"{self.completed_steps} / {self.total_steps}")

            # Update estimated completion
            if self.start_time and self.current_progress > 0:
                elapsed = datetime.now() - self.start_time
                if self.current_progress > 5:  # Avoid division by very small numbers
                    total_estimated = elapsed * (100 / self.current_progress)
                    remaining = total_estimated - elapsed

                    estimated_label = self.query_one("#estimated_time", Label)
                    estimated_label.update(self._format_timedelta(remaining))

        except Exception:
            pass

    def _update_step_details(self, progress: AnalysisProgress) -> None:
        """Update detailed step information."""
        try:
            details_container = self.query_one("#step_details_container", Container)
            details_container.remove_children()

            # Create step indicators
            for i in range(self.total_steps):
                if i < self.completed_steps:
                    icon = "✅"
                    status = "Completed"
                elif i == self.completed_steps:
                    icon = "🔄"
                    status = "In Progress"
                else:
                    icon = "⏳"
                    status = "Pending"

                step_name = self._get_step_name(i)
                details_container.mount(
                    Label(
                        f"{icon} Step {i+1}: {step_name} - {status}",
                        classes="step-detail",
                    ),
                )

        except Exception:
            pass

    def _get_step_name(self, step_index: int) -> str:
        """Get human-readable step name."""
        step_names = [
            "Initialize Analysis",
            "Load Target",
            "Configure Solver",
            "Execute Analysis",
            "Process Results",
        ]

        if step_index < len(step_names):
            return step_names[step_index]
        return f"Step {step_index + 1}"

    @work(exclusive=True)
    async def _start_time_updater(self) -> None:
        """Update elapsed time periodically."""
        while self.start_time and self.current_progress < 100:
            try:
                elapsed = datetime.now() - self.start_time
                elapsed_label = self.query_one("#elapsed_time", Label)
                elapsed_label.update(self._format_timedelta(elapsed))

                await asyncio.sleep(1)
            except Exception:
                break

    def _format_timedelta(self, td: timedelta) -> str:
        """Format timedelta as HH:MM:SS."""
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def complete_tracking(self) -> None:
        """Complete progress tracking."""
        self.current_progress = 100.0
        self.current_step = "Analysis completed"
        self.completed_steps = self.total_steps
        self._update_display()


class ExecutionLog(Log):
    """Enhanced log widget for execution details."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_entries = []

    def log_info(self, message: str) -> None:
        """Log an info message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] INFO: {message}"
        self.write(entry)
        self.log_entries.append(("INFO", timestamp, message))

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] WARN: {message}"
        self.write(entry)
        self.log_entries.append(("WARN", timestamp, message))

    def log_error(self, message: str) -> None:
        """Log an error message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] ERROR: {message}"
        self.write(entry)
        self.log_entries.append(("ERROR", timestamp, message))

    def log_success(self, message: str) -> None:
        """Log a success message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] SUCCESS: {message}"
        self.write(entry)
        self.log_entries.append(("SUCCESS", timestamp, message))

    def export_log(self) -> List[tuple]:
        """Export log entries."""
        return self.log_entries.copy()


class ExecutionControls(Container):
    """Controls for analysis execution."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_running = reactive(False)
        self.can_cancel = reactive(False)
        self.can_pause = reactive(False)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Execution Controls", classes="section-title")

            with Horizontal():
                yield Button("Start Analysis", id="start_btn", variant="success")
                yield Button("Cancel", id="cancel_btn", variant="error", disabled=True)
                yield Button("Pause", id="pause_btn", variant="warning", disabled=True)

            # Execution options
            with Collapsible(title="Execution Options", collapsed=True):
                with Vertical():
                    with Horizontal():
                        yield Label("Priority:", classes="field-label")
                        yield Select(
                            options=[
                                ("Normal", "normal"),
                                ("High", "high"),
                                ("Low", "low"),
                            ],
                            value="normal",
                            id="priority_select",
                        )

                    with Horizontal():
                        yield Label("Save Intermediate Results:", classes="field-label")
                        yield Switch(value=True, id="save_intermediate_switch")

                    with Horizontal():
                        yield Label("Auto Export Results:", classes="field-label")
                        yield Switch(value=False, id="auto_export_switch")

    def set_running_state(self, is_running: bool) -> None:
        """Update control states based on execution status."""
        self.is_running = is_running

        try:
            start_btn = self.query_one("#start_btn", Button)
            cancel_btn = self.query_one("#cancel_btn", Button)
            pause_btn = self.query_one("#pause_btn", Button)

            if is_running:
                start_btn.disabled = True
                cancel_btn.disabled = False
                pause_btn.disabled = False
                start_btn.label = "Running..."
            else:
                start_btn.disabled = False
                cancel_btn.disabled = True
                pause_btn.disabled = True
                start_btn.label = "Start Analysis"

        except Exception:
            pass

    def get_execution_options(self) -> Dict[str, Any]:
        """Get current execution options."""
        try:
            return {
                "priority": self.query_one("#priority_select", Select).value,
                "save_intermediate": self.query_one(
                    "#save_intermediate_switch",
                    Switch,
                ).value,
                "auto_export": self.query_one("#auto_export_switch", Switch).value,
            }
        except Exception:
            return {}


class ConfigurationSummary(Container):
    """Display summary of analysis configuration."""

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Configuration Summary", classes="section-title")
            yield DataTable(id="config_summary_table")

    def on_mount(self) -> None:
        """Initialize the configuration summary."""
        self._populate_summary()

    def _populate_summary(self) -> None:
        """Populate the configuration summary table."""
        table = self.query_one("#config_summary_table", DataTable)
        table.add_columns("Parameter", "Value")

        # Basic configuration
        table.add_row("Analysis Type", self.config.get("analysis_type", "Unknown"))
        table.add_row("Solver", self.config.get("solver", "Unknown"))
        table.add_row("Target", self.config.get("target", "Unknown"))

        # Parameters
        parameters = self.config.get("parameters", {})
        for param, value in parameters.items():
            if isinstance(value, list):
                value_str = ", ".join(map(str, value))
            else:
                value_str = str(value)
            table.add_row(param.replace("_", " ").title(), value_str)

        # Solver parameters
        solver_params = self.config.get("solver_parameters", {})
        if solver_params:
            table.add_row("--- Solver Options ---", "")
            for param, value in solver_params.items():
                table.add_row(f"  {param.replace('_', ' ').title()}", str(value))


class ExecutionEngine(Container):
    """Main execution engine that coordinates the analysis."""

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.analysis_service = AnalysisService() if AnalysisService else None
        self.current_analysis_id = None
        self.analysis_result = None
        self.is_running = reactive(False)
        self.progress_callback = None

    def compose(self) -> ComposeResult:
        with ScrollableContainer():
            with Vertical():
                # Configuration summary
                yield ConfigurationSummary(self.config)

                # Execution controls
                yield ExecutionControls(id="execution_controls")

                # Progress tracking
                yield ProgressTracker(id="progress_tracker")

                # Execution log
                yield Label("Execution Log", classes="section-title")
                yield ExecutionLog(id="execution_log")

    def on_mount(self) -> None:
        """Initialize the execution engine."""
        self.progress_callback = self._on_progress_update

        # Initialize log
        log = self.query_one("#execution_log", ExecutionLog)
        log.log_info("Execution engine initialized")
        log.log_info(f"Analysis type: {self.config.get('analysis_type')}")
        log.log_info(f"Solver: {self.config.get('solver')}")
        log.log_info(f"Target: {self.config.get('target')}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "start_btn":
            self._start_analysis()
        elif event.button.id == "cancel_btn":
            self._cancel_analysis()
        elif event.button.id == "pause_btn":
            self._pause_analysis()

    @work(exclusive=True)
    async def _start_analysis(self) -> None:
        """Start the analysis execution."""
        if not self.analysis_service:
            self.notify("Analysis service not available", severity="error")
            return

        if self.is_running:
            self.notify("Analysis is already running", severity="warning")
            return

        try:
            self.is_running = True

            # Update controls
            controls = self.query_one("#execution_controls", ExecutionControls)
            controls.set_running_state(True)

            # Start progress tracking
            progress_tracker = self.query_one("#progress_tracker", ProgressTracker)
            progress_tracker.start_tracking()

            # Initialize log
            log = self.query_one("#execution_log", ExecutionLog)
            log.log_info("Starting analysis execution...")

            # Create analysis configuration
            analysis_config = self._create_analysis_config()
            if not analysis_config:
                raise ValueError("Failed to create analysis configuration")

            log.log_info("Analysis configuration created successfully")

            # Execute analysis
            log.log_info("Submitting analysis to execution service...")

            self.analysis_result = await self.analysis_service.run_analysis(
                analysis_config,
                progress_callback=self.progress_callback,
            )

            # Handle results
            if self.analysis_result.is_successful:
                log.log_success("Analysis completed successfully!")
                progress_tracker.complete_tracking()

                # Process results
                processed_result = self.analysis_service.process_result(
                    self.analysis_result,
                )

                # Auto-export if enabled
                execution_options = controls.get_execution_options()
                if execution_options.get("auto_export", False):
                    self._auto_export_results(processed_result)

                self.notify("Analysis completed successfully!", severity="success")
            else:
                log.log_error(f"Analysis failed: {self.analysis_result.error_message}")
                self.notify(
                    "Analysis failed. Check the log for details.",
                    severity="error",
                )

        except Exception as e:
            log = self.query_one("#execution_log", ExecutionLog)
            log.log_error(f"Execution error: {str(e)}")
            self.notify(f"Execution error: {str(e)}", severity="error")

        finally:
            self.is_running = False
            controls = self.query_one("#execution_controls", ExecutionControls)
            controls.set_running_state(False)

    def _create_analysis_config(self) -> Optional[Any]:
        """Create AnalysisConfig from the configuration dictionary."""
        if not AnalysisConfig or not AnalysisType or not SolverType:
            return None

        try:
            analysis_type = getattr(AnalysisType, self.config["analysis_type"])
            solver_type = getattr(SolverType, self.config["solver"].upper())

            return AnalysisConfig(
                analysis_type=analysis_type,
                solver_type=solver_type,
                target=self.config["target"],
                parameters=self.config.get("parameters", {}),
                solver_parameters=self.config.get("solver_parameters", {}),
                output_format="json",
            )
        except Exception as e:
            log = self.query_one("#execution_log", ExecutionLog)
            log.log_error(f"Error creating analysis config: {e}")
            return None

    def _on_progress_update(self, progress: Any) -> None:
        """Handle progress updates from the analysis service."""
        try:
            progress_tracker = self.query_one("#progress_tracker", ProgressTracker)
            progress_tracker.update_progress(progress)

            log = self.query_one("#execution_log", ExecutionLog)
            log.log_info(
                f"Progress: {progress.current_step} ({progress.progress_percent:.1f}%)",
            )

        except Exception:
            pass

    def _cancel_analysis(self) -> None:
        """Cancel the running analysis."""
        if not self.is_running:
            self.notify("No analysis is currently running", severity="info")
            return

        try:
            if self.analysis_service and self.current_analysis_id:
                success = self.analysis_service.cancel_analysis(
                    self.current_analysis_id,
                )
                if success:
                    log = self.query_one("#execution_log", ExecutionLog)
                    log.log_warning("Analysis cancelled by user")
                    self.notify("Analysis cancelled", severity="info")
                else:
                    self.notify("Failed to cancel analysis", severity="error")

            self.is_running = False
            controls = self.query_one("#execution_controls", ExecutionControls)
            controls.set_running_state(False)

        except Exception as e:
            self.notify(f"Error cancelling analysis: {e}", severity="error")

    def _pause_analysis(self) -> None:
        """Pause the running analysis."""
        # TODO: Implement analysis pausing
        self.notify("Analysis pausing not yet implemented", severity="info")

    def _auto_export_results(self, processed_result: Any) -> None:
        """Auto-export results if enabled."""
        try:
            if not processed_result:
                return

            # Export as JSON by default
            export_path = self.analysis_service.export_result(processed_result, "json")

            log = self.query_one("#execution_log", ExecutionLog)
            log.log_success(f"Results auto-exported to: {export_path}")

        except Exception as e:
            log = self.query_one("#execution_log", ExecutionLog)
            log.log_error(f"Auto-export failed: {e}")

    def get_analysis_result(self) -> Optional[Any]:
        """Get the analysis result if available."""
        return self.analysis_result


class ExecutionScreen(Screen):
    """Main execution screen."""

    BINDINGS = [
        Binding("ctrl+s", "start", "Start"),
        Binding("ctrl+c", "cancel", "Cancel"),
        Binding("ctrl+e", "export", "Export"),
        Binding("escape", "back", "Back"),
    ]

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.execution_engine = None

    def compose(self) -> ComposeResult:
        with Container():
            yield Label("Analysis Execution", classes="screen-title")
            yield ExecutionEngine(self.config, id="execution_engine")

            with Horizontal(classes="screen-actions"):
                yield Button(
                    "View Results",
                    id="view_results_btn",
                    variant="primary",
                    disabled=True,
                )
                yield Button(
                    "Export Results",
                    id="export_results_btn",
                    variant="default",
                    disabled=True,
                )
                yield Button("Back", id="back_btn", variant="default")

    def on_mount(self) -> None:
        """Initialize screen when mounted."""
        self.execution_engine = self.query_one("#execution_engine", ExecutionEngine)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "view_results_btn":
            self._view_results()
        elif event.button.id == "export_results_btn":
            self._export_results()
        elif event.button.id == "back_btn":
            self.app.pop_screen()

    def _view_results(self) -> None:
        """View analysis results."""
        if not self.execution_engine:
            self.notify("Execution engine not available", severity="error")
            return

        result = self.execution_engine.get_analysis_result()
        if result and result.is_successful:
            # Navigate to results screen
            from .results_screen import ResultsScreen

            self.app.push_screen(ResultsScreen(result))
        else:
            self.notify("No successful results available", severity="warning")

    def _export_results(self) -> None:
        """Export analysis results."""
        if not self.execution_engine:
            self.notify("Execution engine not available", severity="error")
            return

        result = self.execution_engine.get_analysis_result()
        if result and result.is_successful:
            # TODO: Show export options dialog
            self.notify("Export functionality not yet implemented", severity="info")
        else:
            self.notify("No results to export", severity="warning")

    def action_start(self) -> None:
        """Start analysis action."""
        if self.execution_engine:
            self.execution_engine._start_analysis()

    def action_cancel(self) -> None:
        """Cancel analysis action."""
        if self.execution_engine:
            self.execution_engine._cancel_analysis()

    def action_export(self) -> None:
        """Export results action."""
        self._export_results()

    def action_back(self) -> None:
        """Go back action."""
        self.app.pop_screen()
