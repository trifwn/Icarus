"""
Airplane Analysis Screen

This module provides the airplane analysis screen for the ICARUS CLI application.
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


class AirplaneAnalysisForm(Container):
    """Form for airplane analysis configuration."""

    def compose(self) -> ComposeResult:
        yield Label("Airplane Analysis", classes="form-title")
        with Vertical():
            yield Label("Airplane:")
            yield Input(
                placeholder="Enter airplane name or configuration",
                value="demo_airplane",
                id="airplane_name",
            )
            yield Label("Solver:")
            yield Select(
                (("AVL", "avl"), ("GenuVP", "genuvp"), ("OpenFoam", "openfoam")),
                value="avl",
                id="airplane_solver_select",
            )
            yield Label("Velocity (m/s):")
            yield Input(placeholder="50", value="50", id="velocity_input")
            yield Label("Altitude (m):")
            yield Input(placeholder="1000", value="1000", id="altitude_input")
            yield Label("Angle Range:")
            yield Input(
                placeholder="-5:15:1",
                value="-5:15:1",
                id="airplane_angles_input",
            )
            with Horizontal():
                yield Button("Validate", id="validate_airplane_btn", variant="primary")
                yield Button(
                    "Run Analysis",
                    id="run_airplane_analysis_btn",
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


class AirplaneScreen(BaseScreen):
    """Airplane analysis screen."""

    def __init__(self, **kwargs):
        super().__init__("airplane", **kwargs)
        self.analysis_service = None
        if TEXTUAL_AVAILABLE:
            try:
                self.analysis_service = AnalysisService()
            except Exception as e:
                if hasattr(self, "log"):
                    self.log.error(f"Failed to initialize analysis service: {e}")

    def compose(self) -> ComposeResult:
        """Compose the airplane analysis screen."""
        with Horizontal():
            with Vertical():
                yield AirplaneAnalysisForm()
                yield ProgressDisplay()
            yield ResultsDisplay()

    async def initialize(self) -> None:
        """Initialize airplane analysis screen."""
        self.app.log.info("Initializing airplane analysis screen")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        button_id = event.button.id

        if button_id == "validate_airplane_btn":
            await self.validate_airplane_form()
        elif button_id == "run_airplane_analysis_btn":
            await self.run_airplane_analysis()

    async def validate_airplane_form(self) -> None:
        """Validate airplane form inputs."""
        try:
            name = self.query_one("#airplane_name", Input).value
            velocity = self.query_one("#velocity_input", Input).value
            altitude = self.query_one("#altitude_input", Input).value
            angles = self.query_one("#airplane_angles_input", Input).value

            log = self.query_one(ResultsDisplay).query_one("#results_log", Log)

            if not name:
                log.write("Error: Please enter an airplane name or configuration")
                return

            try:
                float(velocity)
            except ValueError:
                log.write("Error: Velocity must be a valid number")
                return

            try:
                float(altitude)
            except ValueError:
                log.write("Error: Altitude must be a valid number")
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
            self.app.log.error(f"Airplane validation error: {e}")

    async def run_airplane_analysis(self) -> None:
        """Run airplane analysis."""
        if not self.analysis_service:
            self.app.log.error("Analysis service not available")
            return

        try:
            # Get form inputs
            airplane_input = self.query_one("#airplane_name", Input)
            velocity_input = self.query_one("#velocity_input", Input)
            altitude_input = self.query_one("#altitude_input", Input)
            angle_range_input = self.query_one("#airplane_angles_input", Input)
            solver_select = self.query_one("#airplane_solver_select", Select)

            # Parse angle range
            angle_range = angle_range_input.value.split(":")
            min_aoa = float(angle_range[0])
            max_aoa = float(angle_range[1])
            aoa_step = float(angle_range[2]) if len(angle_range) > 2 else 1.0

            # Update progress
            progress_display = self.query_one(ProgressDisplay)
            progress_display.update_progress(0, "Starting airplane analysis...")

            # For airplane analysis, we'll use the workflow directly since it has mock data
            from cli.workflows.airplane_workflow import AirplaneWorkflow

            workflow = AirplaneWorkflow()

            # Extract flight conditions and analysis parameters
            flight_conditions = {
                "velocity": float(velocity_input.value),
                "altitude": float(altitude_input.value),
                "density": 1.112,  # kg/m³ at 1000m ISA
            }

            analysis_parameters = {
                "min_aoa": min_aoa,
                "max_aoa": max_aoa,
                "aoa_step": aoa_step,
            }

            def progress_callback(progress_percent, status):
                progress_display.update_progress(progress_percent, status)

            result = await workflow.run_airplane_analysis(
                flight_conditions=flight_conditions,
                analysis_parameters=analysis_parameters,
                progress_callback=progress_callback,
            )

            # Update results display
            results_display = self.query_one(ResultsDisplay)
            results_display.update_results(result)

            # Update progress to complete
            if result.get("success"):
                progress_display.update_progress(100, "Analysis completed successfully")
            else:
                error_msg = result.get("error", "Unknown error")
                progress_display.update_progress(0, f"Analysis failed: {error_msg}")

        except Exception as e:
            self.app.log.error(f"Error running airplane analysis: {e}")
            progress_display = self.query_one(ProgressDisplay)
            progress_display.update_progress(0, f"Error: {e}")
