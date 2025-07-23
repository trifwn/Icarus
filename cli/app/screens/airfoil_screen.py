"""
Airfoil Analysis Screen

This module provides the airfoil analysis screen for the ICARUS CLI application.
"""

try:
    from textual.app import ComposeResult
    from textual.containers import Container
    from textual.containers import Horizontal
    from textual.containers import Vertical
    from textual.widgets import Button
    from textual.widgets import DataTable
    from textual.widgets import Header
    from textual.widgets import Input
    from textual.widgets import Label
    from textual.widgets import Log
    from textual.widgets import ProgressBar
    from textual.widgets import Select
    from textual.widgets import Static

    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False
    ComposeResult = None

from ...app.screen_manager import BaseScreen
from ...integration.analysis_service import AnalysisService
from ...integration.models import AnalysisConfig
from ...integration.models import AnalysisType
from ...integration.models import SolverType


class AirfoilAnalysisForm(Container):
    """Form for airfoil analysis configuration."""

    def compose(self) -> ComposeResult:
        yield Label("Airfoil Analysis", classes="form-title")
        with Vertical():
            yield Label("Airfoil:")
            yield Input(
                placeholder="Enter airfoil name or NACA code",
                value="NACA2412",
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
                placeholder="-10:15:0.5",
                value="-10:15:0.5",
                id="airfoil_angles_input",
            )
            with Horizontal():
                yield Button("Validate", id="validate_airfoil_btn", variant="primary")
                yield Button(
                    "Run Analysis",
                    id="run_airfoil_analysis_btn",
                    variant="success",
                )


class ProgressDisplay(Container):
    """Container for progress tracking."""

    def compose(self) -> ComposeResult:
        yield Label("Progress", classes="section-title")
        yield ProgressBar(id="progress_bar")
        yield Label("Ready", id="progress_status")

    def update_progress(self, progress_percent: float, status: str):
        """Update progress display."""
        progress_bar = self.query_one("#progress_bar", ProgressBar)
        status_label = self.query_one("#progress_status", Label)

        progress_bar.update(progress=progress_percent)
        status_label.update(status)


class ResultsDisplay(Container):
    """Display analysis results."""

    def compose(self) -> ComposeResult:
        yield Label("Analysis Results", classes="section-title")
        yield DataTable(id="results_table")
        yield Log(id="results_log")

    def update_results(self, results: dict):
        """Update the results display."""
        table = self.query_one("#results_table", DataTable)
        log = self.query_one("#results_log", Log)

        # Store results for export
        self._last_results = results

        # Clear previous results
        table.clear()

        # Add columns
        table.add_columns("Parameter", "Value", "Unit")

        # Add basic info
        if "config" in results:
            config = results["config"]
            table.add_row("Analysis Type", config.get("analysis_type", "Unknown"))
            table.add_row("Target", config.get("target", "Unknown"))
            table.add_row("Solver", config.get("solver_type", "Unknown"))

        # Add analysis-specific results
        if "raw_data" in results and results["raw_data"]:
            raw_data = results["raw_data"]

            if "polars" in raw_data:
                # Airfoil results
                polars = raw_data["polars"]
                if "cl" in polars and len(polars["cl"]) > 0:
                    table.add_row("Max CL", f"{max(polars['cl']):.3f}", "-")
                    table.add_row("Min CD", f"{min(polars['cd']):.4f}", "-")

            elif "CL" in raw_data:
                # Airplane results
                table.add_row("Max CL", f"{max(raw_data['CL']):.3f}", "-")
                table.add_row("Min CD", f"{min(raw_data['CD']):.4f}", "-")

        # Enhanced results display using processed results if available
        if "processed_results" in results:
            processed = results.get("processed_results", {})
            summary = processed.get("performance_summary", {})

            # Add performance metrics
            for metric_name, metric_data in summary.items():
                if isinstance(metric_data, dict) and "value" in metric_data:
                    display_name = metric_name.replace("_", " ").title()
                    value = metric_data["value"]
                    alpha = metric_data.get("alpha", "")

                    if "cl" in metric_name.lower() or "CL" in metric_name:
                        table.add_row(
                            display_name,
                            f"{value:.3f}",
                            f"@ {alpha:.1f}°" if alpha else "-",
                        )
                    elif "cd" in metric_name.lower() or "CD" in metric_name:
                        table.add_row(
                            display_name,
                            f"{value:.4f}",
                            f"@ {alpha:.1f}°" if alpha else "-",
                        )
                    elif "ld" in metric_name.lower() or "LD" in metric_name:
                        table.add_row(
                            display_name,
                            f"{value:.1f}",
                            f"@ {alpha:.1f}°" if alpha else "-",
                        )
                    elif "stall" in metric_name.lower():
                        if "speed" in metric_name.lower():
                            table.add_row(display_name, f"{value:.1f}", "m/s")
                        else:
                            table.add_row(display_name, f"{value:.1f}", "°")

        # Log the full results
        log.write(f"Analysis completed: {results.get('status', 'Unknown')}")
        if results.get("error_message"):
            log.write(f"Error: {results['error_message']}")
        elif results.get("success"):
            log.write("Analysis completed successfully")

            # Try to display rich table if available
            try:
                from cli.visualization.demo_visualization import DemoVisualizer

                visualizer = DemoVisualizer()
                # Note: This would need to be adapted for TUI display
                # For now, just log that visualization is available
                log.write("Visualization available - use export to save plots")
            except ImportError:
                pass


class AirfoilScreen(BaseScreen):
    """Airfoil analysis screen."""

    def __init__(self, **kwargs):
        super().__init__("airfoil", **kwargs)
        self.analysis_service = None
        if TEXTUAL_AVAILABLE:
            try:
                self.analysis_service = AnalysisService()
            except Exception as e:
                if hasattr(self, "log"):
                    self.log.error(f"Failed to initialize analysis service: {e}")

    def compose(self) -> ComposeResult:
        """Compose the airfoil analysis screen."""
        with Horizontal():
            with Vertical():
                yield AirfoilAnalysisForm()
                yield ProgressDisplay()
            yield ResultsDisplay()

    async def initialize(self) -> None:
        """Initialize airfoil analysis screen."""
        self.app.log.info("Initializing airfoil analysis screen")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        button_id = event.button.id

        if button_id == "validate_airfoil_btn":
            await self.validate_airfoil_form()
        elif button_id == "run_airfoil_analysis_btn":
            await self.run_airfoil_analysis()

    async def validate_airfoil_form(self) -> None:
        """Validate airfoil form inputs."""
        try:
            name = self.query_one("#airfoil_name", Input).value
            reynolds = self.query_one("#airfoil_reynolds_input", Input).value
            angles = self.query_one("#airfoil_angles_input", Input).value

            log = self.query_one(ResultsDisplay).query_one("#results_log", Log)

            if not name:
                log.write("Error: Please enter an airfoil name")
                return

            try:
                float(reynolds)
            except ValueError:
                log.write("Error: Reynolds number must be a valid number")
                return

            try:
                angle_parts = angles.split(":")
                if len(angle_parts) < 2 or len(angle_parts) > 3:
                    raise ValueError("Invalid angle range format")
                float(angle_parts[0])
                float(angle_parts[1])
                if len(angle_parts) == 3:
                    float(angle_parts[2])
            except ValueError:
                log.write("Error: Invalid angle range format. Use min:max:step")
                return

            log.write("Validation passed. Ready to run analysis.")

        except Exception as e:
            self.app.log.error(f"Airfoil validation error: {e}")

    async def run_airfoil_analysis(self) -> None:
        """Run airfoil analysis."""
        if not self.analysis_service:
            self.app.log.error("Analysis service not available")
            return

        try:
            # Get form inputs
            airfoil_input = self.query_one("#airfoil_name", Input)
            reynolds_input = self.query_one("#airfoil_reynolds_input", Input)
            angle_range_input = self.query_one("#airfoil_angles_input", Input)
            solver_select = self.query_one("#airfoil_solver_select", Select)

            # Parse angle range
            angle_range = angle_range_input.value.split(":")
            min_aoa = float(angle_range[0])
            max_aoa = float(angle_range[1])
            aoa_step = float(angle_range[2]) if len(angle_range) > 2 else 0.5

            # Create analysis config
            config = AnalysisConfig(
                analysis_type=AnalysisType.AIRFOIL_POLAR,
                solver_type=SolverType(solver_select.value),
                target=airfoil_input.value,
                parameters={
                    "reynolds": float(reynolds_input.value),
                    "min_aoa": min_aoa,
                    "max_aoa": max_aoa,
                    "aoa_step": aoa_step,
                    "mach": 0.0,  # Default mach number
                },
            )

            # Update progress
            progress_display = self.query_one(ProgressDisplay)
            progress_display.update_progress(0, "Starting airfoil analysis...")

            # Run analysis asynchronously
            await self._run_analysis_async(config)

        except Exception as e:
            self.app.log.error(f"Error starting airfoil analysis: {e}")
            progress_display = self.query_one(ProgressDisplay)
            progress_display.update_progress(0, f"Error: {e}")

    async def _run_analysis_async(self, config: AnalysisConfig):
        """Run analysis asynchronously with progress updates."""
        try:

            def progress_callback(progress_percent, status):
                """Update progress display."""
                progress_display = self.query_one(ProgressDisplay)
                progress_display.update_progress(progress_percent, status)

            # Use appropriate workflow based on analysis type
            if config.analysis_type == AnalysisType.AIRFOIL_POLAR:
                from cli.workflows.airfoil_workflow import AirfoilWorkflow

                workflow = AirfoilWorkflow()

                # Convert config to workflow parameters
                parameters = config.parameters.copy()

                result = await workflow.run_airfoil_analysis(
                    config.target,
                    parameters,
                    progress_callback=progress_callback,
                )
            else:
                # Fallback to original analysis service
                def service_progress_callback(progress):
                    progress_callback(progress.progress_percent, progress.current_step)

                service_result = await self.analysis_service.run_analysis(
                    config,
                    progress_callback=service_progress_callback,
                )

                result = {
                    "success": service_result.status == "success",
                    "config": config.to_dict(),
                    "status": service_result.status,
                    "raw_data": service_result.raw_data,
                    "error_message": service_result.error_message,
                }

            # Update results display
            results_display = self.query_one(ResultsDisplay)
            results_display.update_results(result)

            # Update progress to complete
            progress_display = self.query_one(ProgressDisplay)
            if result.get("success"):
                progress_display.update_progress(100, "Analysis completed successfully")
            else:
                error_msg = result.get(
                    "error",
                    result.get("error_message", "Unknown error"),
                )
                progress_display.update_progress(0, f"Analysis failed: {error_msg}")

        except Exception as e:
            self.app.log.error(f"Analysis error: {e}")
            progress_display = self.query_one(ProgressDisplay)
            progress_display.update_progress(0, f"Error: {e}")
