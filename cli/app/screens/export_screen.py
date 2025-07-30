"""
Export Screen

This module provides the export functionality screen for the ICARUS CLI application.
"""

try:
    from textual.app import ComposeResult
    from textual.containers import Container
    from textual.containers import Horizontal
    from textual.containers import Vertical
    from textual.widgets import Button
    from textual.widgets import Input
    from textual.widgets import Label
    from textual.widgets import Log
    from textual.widgets import Select
    from textual.widgets import Static

    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False
    ComposeResult = None

from ...app.screen_manager import BaseScreen


class ExportScreen(BaseScreen):
    """Export functionality screen."""

    def __init__(self, **kwargs):
        super().__init__("export", **kwargs)
        self._last_results = None

    def compose(self) -> ComposeResult:
        """Compose the export screen."""
        yield Label("Export Results", classes="form-title")
        with Vertical():
            yield Label("Export Format:")
            yield Select(
                [
                    ("JSON", "json"),
                    ("CSV", "csv"),
                    ("MATLAB", "mat"),
                    ("Text Report", "txt"),
                ],
                value="json",
                id="export_format",
            )
            yield Label("Output File:")
            yield Input(placeholder="results.json", id="export_filename")
            yield Label("Plot Options:")
            yield Select(
                [
                    ("None", "none"),
                    ("Polar Plots", "polar"),
                    ("Performance Summary", "summary"),
                    ("All Plots", "all"),
                ],
                value="none",
                id="plot_options",
            )
            with Horizontal():
                yield Button("Export Results", id="export_btn", variant="primary")
                yield Button(
                    "Export with Plot",
                    id="export_plot_btn",
                    variant="success",
                )
            yield Log(id="export_log")

    async def initialize(self) -> None:
        """Initialize export screen."""
        self.app.log.info("Initializing export screen")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        button_id = event.button.id

        if button_id == "export_btn":
            await self.export_results(with_plot=False)
        elif button_id == "export_plot_btn":
            await self.export_results(with_plot=True)

    async def export_results(self, with_plot: bool = False) -> None:
        """Export analysis results."""
        try:
            export_format = self.query_one("#export_format", Select).value
            export_filename = self.query_one("#export_filename", Input).value
            plot_option = (
                self.query_one("#plot_options", Select).value if with_plot else "none"
            )
            export_log = self.query_one("#export_log", Log)

            if not export_filename:
                export_log.write("Error: Please specify output filename")
                return

            # Get the last analysis results
            if not hasattr(self, "_last_results") or not self._last_results:
                # Try to get results from other screens
                try:
                    # Check if we can find results in airfoil or airplane screens
                    screen_manager = self.app.screen_manager
                    for screen_name in ["airfoil", "airplane"]:
                        if screen_name in screen_manager.screens:
                            screen = screen_manager.screens[screen_name]
                            results_display = screen.query_one(
                                ResultsDisplay,
                                default=None,
                            )
                            if results_display and hasattr(
                                results_display,
                                "_last_results",
                            ):
                                self._last_results = results_display._last_results
                                break
                except Exception:
                    pass

            if not self._last_results:
                export_log.write("Error: No analysis results to export")
                return

            results = self._last_results

            # Import visualization module
            try:
                from cli.visualization.demo_visualization import DemoExporter
                from cli.visualization.demo_visualization import DemoVisualizer

                exporter = DemoExporter()

                # Export results
                success = False
                if export_format == "json":
                    success = exporter.export_to_json(results, export_filename)
                elif export_format == "csv":
                    success = exporter.export_to_csv(results, export_filename)
                elif export_format == "mat":
                    success = exporter.export_to_matlab(results, export_filename)
                elif export_format == "txt":
                    success = exporter.create_summary_report(results, export_filename)

                # Export plots if requested
                if with_plot and plot_option != "none" and success:
                    visualizer = DemoVisualizer()
                    plot_filename = export_filename.rsplit(".", 1)[0] + ".png"

                    if "airfoil_info" in results:
                        if plot_option in ["polar", "all"]:
                            visualizer.plot_airfoil_polar(
                                results,
                                plot_filename,
                                show_plot=False,
                            )
                            export_log.write(
                                f"Airfoil polar plot saved to: {plot_filename}",
                            )
                    elif "airplane_config" in results:
                        if plot_option in ["polar", "all"]:
                            visualizer.plot_airplane_polar(
                                results,
                                plot_filename,
                                show_plot=False,
                            )
                            export_log.write(
                                f"Airplane polar plot saved to: {plot_filename}",
                            )

                if success:
                    export_log.write(
                        f"Results exported successfully to: {export_filename}",
                    )
                else:
                    export_log.write(f"Failed to export results to: {export_filename}")

            except ImportError:
                export_log.write("Error: Export functionality not available")
            except Exception as e:
                export_log.write(f"Export error: {str(e)}")

        except Exception as e:
            self.app.log.error(f"Export error: {e}")
            export_log = self.query_one("#export_log", Log)
            export_log.write(f"Export error: {str(e)}")

    def set_results(self, results: dict) -> None:
        """Set results for export."""
        self._last_results = results
