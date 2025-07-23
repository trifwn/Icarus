"""TUI Visualization Screen - Textual-based interface for visualization

This module provides a Textual-based screen for the visualization system,
allowing users to create, customize, and export plots through the TUI.
"""

from typing import Any
from typing import Dict
from typing import Optional

from rich.console import Console
from textual import on
from textual.app import ComposeResult
from textual.containers import Container
from textual.containers import Horizontal
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button
from textual.widgets import DataTable
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Input
from textual.widgets import Label
from textual.widgets import Select
from textual.widgets import TabbedContent
from textual.widgets import TabPane

from .visualization_manager import VisualizationManager


class VisualizationScreen(Screen):
    """Main visualization screen for the TUI."""

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("c", "create_plot", "Create Plot"),
        ("e", "export_plot", "Export Plot"),
        ("r", "refresh", "Refresh"),
    ]

    def __init__(self, console: Optional[Console] = None):
        """Initialize the visualization screen.

        Args:
            console: Rich console for output (optional)
        """
        super().__init__()
        self.console = console or Console()
        self.viz_manager = VisualizationManager(console=self.console)
        self.selected_plot_id: Optional[str] = None

    def compose(self) -> ComposeResult:
        """Compose the screen layout."""
        yield Header()

        with TabbedContent(initial="plots"):
            with TabPane("Active Plots", id="plots"):
                yield Container(
                    Label("Active Plots", classes="section-title"),
                    DataTable(id="plots_table"),
                    Horizontal(
                        Button("Create Plot", id="create_plot", variant="primary"),
                        Button("Customize", id="customize_plot"),
                        Button("Export", id="export_plot"),
                        Button("Close Plot", id="close_plot", variant="error"),
                        classes="button-row",
                    ),
                    classes="tab-content",
                )

            with TabPane("Create Plot", id="create"):
                yield Container(
                    Label("Create New Plot", classes="section-title"),
                    Vertical(
                        Label("Plot Type:"),
                        Select(
                            [
                                ("Line Plot", "line"),
                                ("Scatter Plot", "scatter"),
                                ("Bar Plot", "bar"),
                                ("Airfoil Polar", "airfoil_polar"),
                                ("Pressure Distribution", "pressure_distribution"),
                            ],
                            id="plot_type_select",
                        ),
                        Label("Title:"),
                        Input(placeholder="Enter plot title", id="plot_title"),
                        Label("Data Source:"),
                        Select(
                            [
                                ("Sample Data", "sample"),
                                ("Analysis Results", "analysis"),
                                ("Custom Data", "custom"),
                            ],
                            id="data_source_select",
                        ),
                        Button("Create Plot", id="create_plot_btn", variant="primary"),
                        classes="form-container",
                    ),
                    classes="tab-content",
                )

            with TabPane("Customize", id="customize"):
                yield Container(
                    Label("Plot Customization", classes="section-title"),
                    Vertical(
                        Label("Selected Plot:"),
                        Select([], id="plot_select"),
                        Label("Color Palette:"),
                        Select(
                            [
                                ("Default", "default"),
                                ("Aerospace", "aerospace"),
                                ("Publication", "publication"),
                                ("Colorblind Friendly", "colorblind"),
                                ("Vibrant", "vibrant"),
                            ],
                            id="color_palette_select",
                        ),
                        Label("Line Width:"),
                        Input(placeholder="2", id="line_width"),
                        Label("Grid:"),
                        Select(
                            [("Show Grid", "show"), ("Hide Grid", "hide")],
                            id="grid_select",
                        ),
                        Button(
                            "Apply Customization",
                            id="apply_custom_btn",
                            variant="primary",
                        ),
                        classes="form-container",
                    ),
                    classes="tab-content",
                )

            with TabPane("Export", id="export"):
                yield Container(
                    Label("Export Plot", classes="section-title"),
                    Vertical(
                        Label("Selected Plot:"),
                        Select([], id="export_plot_select"),
                        Label("Export Format:"),
                        Select(
                            [
                                ("PNG Image", "png"),
                                ("PDF Document", "pdf"),
                                ("SVG Vector", "svg"),
                                ("JSON Data", "json"),
                            ],
                            id="export_format_select",
                        ),
                        Label("Output Path:"),
                        Input(placeholder="/path/to/output", id="export_path"),
                        Label("Quality Preset:"),
                        Select(
                            [
                                ("Draft", "draft"),
                                ("Standard", "standard"),
                                ("High Quality", "high"),
                                ("Publication", "publication"),
                            ],
                            id="quality_preset_select",
                        ),
                        Button("Export Plot", id="export_btn", variant="primary"),
                        classes="form-container",
                    ),
                    classes="tab-content",
                )

            with TabPane("Real-time", id="realtime"):
                yield Container(
                    Label("Real-time Updates", classes="section-title"),
                    Vertical(
                        Label("Selected Plot:"),
                        Select([], id="realtime_plot_select"),
                        Label("Update Interval (seconds):"),
                        Input(placeholder="1.0", id="update_interval"),
                        Horizontal(
                            Button(
                                "Start Updates",
                                id="start_updates",
                                variant="primary",
                            ),
                            Button("Stop Updates", id="stop_updates", variant="error"),
                            Button("Pause Updates", id="pause_updates"),
                            classes="button-row",
                        ),
                        Label("Active Updates:", classes="section-subtitle"),
                        DataTable(id="updates_table"),
                        classes="form-container",
                    ),
                    classes="tab-content",
                )

        yield Footer()

    def on_mount(self) -> None:
        """Initialize the screen when mounted."""
        self.refresh_plots_table()
        self.refresh_plot_selects()

    def refresh_plots_table(self) -> None:
        """Refresh the active plots table."""
        table = self.query_one("#plots_table", DataTable)
        table.clear(columns=True)

        # Add columns
        table.add_columns("ID", "Type", "Title", "Real-time", "Customized")

        # Add plot data
        active_plots = self.viz_manager.list_active_plots()
        for plot_info in active_plots:
            table.add_row(
                plot_info["id"],
                plot_info["type"],
                plot_info["title"],
                "Yes" if plot_info["real_time"] else "No",
                "Yes" if plot_info["has_customizations"] else "No",
            )

    def refresh_plot_selects(self) -> None:
        """Refresh all plot selection dropdowns."""
        active_plots = self.viz_manager.list_active_plots()
        plot_options = [(plot["title"], plot["id"]) for plot in active_plots]

        # Update all plot select widgets
        select_ids = ["plot_select", "export_plot_select", "realtime_plot_select"]
        for select_id in select_ids:
            try:
                select_widget = self.query_one(f"#{select_id}", Select)
                select_widget.set_options(plot_options)
            except:
                pass  # Widget might not be visible

    @on(Button.Pressed, "#create_plot")
    @on(Button.Pressed, "#create_plot_btn")
    async def create_plot(self, event: Button.Pressed) -> None:
        """Handle create plot button press."""
        try:
            # Get form values
            plot_type = self.query_one("#plot_type_select", Select).value or "line"
            title = self.query_one("#plot_title", Input).value or "Untitled Plot"
            data_source = (
                self.query_one("#data_source_select", Select).value or "sample"
            )

            # Generate sample data based on type
            data = self._generate_sample_data(plot_type, data_source)

            # Create the plot
            if plot_type in ["line", "scatter", "bar"]:
                plot_id = self.viz_manager.create_interactive_plot(
                    data=data,
                    plot_type=plot_type,
                    title=title,
                )
            else:
                plot_id = self.viz_manager.generate_chart(
                    analysis_results=data,
                    chart_type=plot_type,
                )

            # Refresh displays
            self.refresh_plots_table()
            self.refresh_plot_selects()

            # Show success message
            self.notify(f"Created plot: {plot_id}", severity="information")

        except Exception as e:
            self.notify(f"Failed to create plot: {e}", severity="error")

    @on(Button.Pressed, "#customize_plot")
    @on(Button.Pressed, "#apply_custom_btn")
    async def customize_plot(self, event: Button.Pressed) -> None:
        """Handle customize plot button press."""
        try:
            # Get selected plot
            plot_id = self.query_one("#plot_select", Select).value
            if not plot_id:
                self.notify("Please select a plot to customize", severity="warning")
                return

            # Get customization options
            color_palette = (
                self.query_one("#color_palette_select", Select).value or "default"
            )
            line_width_str = self.query_one("#line_width", Input).value
            grid_option = self.query_one("#grid_select", Select).value or "show"

            # Build customizations dictionary
            customizations = {
                "color_palette": color_palette,
                "grid": {"visible": grid_option == "show"},
            }

            if line_width_str:
                try:
                    customizations["line_widths"] = float(line_width_str)
                except ValueError:
                    pass

            # Apply customizations
            success = self.viz_manager.customize_plot(plot_id, customizations)

            if success:
                self.refresh_plots_table()
                self.notify("Plot customized successfully", severity="information")
            else:
                self.notify("Failed to customize plot", severity="error")

        except Exception as e:
            self.notify(f"Customization error: {e}", severity="error")

    @on(Button.Pressed, "#export_plot")
    @on(Button.Pressed, "#export_btn")
    async def export_plot(self, event: Button.Pressed) -> None:
        """Handle export plot button press."""
        try:
            # Get export parameters
            plot_id = self.query_one("#export_plot_select", Select).value
            if not plot_id:
                self.notify("Please select a plot to export", severity="warning")
                return

            format = self.query_one("#export_format_select", Select).value or "png"
            output_path = self.query_one("#export_path", Input).value
            quality_preset = self.query_one("#quality_preset_select", Select).value

            if not output_path:
                output_path = f"plot_export.{format}"

            # Export options
            export_options = {}
            if quality_preset:
                export_options["quality_preset"] = quality_preset

            # Perform export
            success = self.viz_manager.export_plot(
                plot_id=plot_id,
                output_path=output_path,
                format=format,
                **export_options,
            )

            if success:
                self.notify(f"Plot exported to {output_path}", severity="information")
            else:
                self.notify("Export failed", severity="error")

        except Exception as e:
            self.notify(f"Export error: {e}", severity="error")

    @on(Button.Pressed, "#close_plot")
    async def close_plot(self, event: Button.Pressed) -> None:
        """Handle close plot button press."""
        try:
            # Get selected plot from table
            table = self.query_one("#plots_table", DataTable)
            if table.cursor_row is not None:
                row_data = table.get_row_at(table.cursor_row)
                plot_id = row_data[0]  # First column is ID

                success = self.viz_manager.close_plot(plot_id)

                if success:
                    self.refresh_plots_table()
                    self.refresh_plot_selects()
                    self.notify(f"Closed plot: {plot_id}", severity="information")
                else:
                    self.notify("Failed to close plot", severity="error")
            else:
                self.notify("Please select a plot to close", severity="warning")

        except Exception as e:
            self.notify(f"Close error: {e}", severity="error")

    @on(Button.Pressed, "#start_updates")
    async def start_real_time_updates(self, event: Button.Pressed) -> None:
        """Handle start real-time updates button press."""
        try:
            plot_id = self.query_one("#realtime_plot_select", Select).value
            if not plot_id:
                self.notify(
                    "Please select a plot for real-time updates",
                    severity="warning",
                )
                return

            interval_str = self.query_one("#update_interval", Input).value or "1.0"
            try:
                interval = float(interval_str)
            except ValueError:
                interval = 1.0

            # Create a simple data source for demo
            def demo_data_source():
                import random

                return {"x": random.random(), "y": random.random()}

            # Start updates
            success = self.viz_manager.start_real_time_updates(
                plot_id=plot_id,
                data_source=demo_data_source,
                update_interval=interval,
            )

            if success:
                self.refresh_plots_table()
                self.notify("Started real-time updates", severity="information")
            else:
                self.notify("Failed to start updates", severity="error")

        except Exception as e:
            self.notify(f"Real-time update error: {e}", severity="error")

    @on(Button.Pressed, "#stop_updates")
    async def stop_real_time_updates(self, event: Button.Pressed) -> None:
        """Handle stop real-time updates button press."""
        try:
            plot_id = self.query_one("#realtime_plot_select", Select).value
            if not plot_id:
                self.notify("Please select a plot", severity="warning")
                return

            success = self.viz_manager.stop_real_time_updates(plot_id)

            if success:
                self.refresh_plots_table()
                self.notify("Stopped real-time updates", severity="information")
            else:
                self.notify("Failed to stop updates", severity="error")

        except Exception as e:
            self.notify(f"Stop updates error: {e}", severity="error")

    def _generate_sample_data(self, plot_type: str, data_source: str) -> Dict[str, Any]:
        """Generate sample data for plot creation."""
        import numpy as np

        if data_source == "sample":
            if plot_type in ["line", "scatter"]:
                x = np.linspace(0, 10, 50)
                return {
                    "x": x.tolist(),
                    "y": np.sin(x).tolist(),
                    "xlabel": "X Values",
                    "ylabel": "Y Values",
                }
            elif plot_type == "bar":
                categories = ["A", "B", "C", "D", "E"]
                values = [23, 45, 56, 78, 32]
                return {
                    "x": categories,
                    "y": values,
                    "xlabel": "Categories",
                    "ylabel": "Values",
                }
            elif plot_type == "airfoil_polar":
                alpha = np.linspace(-5, 15, 21)
                return {
                    "alpha": alpha.tolist(),
                    "cl": (0.1 * alpha).tolist(),
                    "cd": (0.006 + 0.0001 * alpha**2).tolist(),
                    "cm": (-0.002 * alpha).tolist(),
                }
            elif plot_type == "pressure_distribution":
                x = np.linspace(0, 1, 50)
                return {
                    "x": x.tolist(),
                    "cp_upper": (-np.sin(np.pi * x)).tolist(),
                    "cp_lower": (np.sin(np.pi * x) * 0.5).tolist(),
                }

        # Default fallback
        return {
            "x": list(range(10)),
            "y": [i**2 for i in range(10)],
            "xlabel": "X",
            "ylabel": "Y",
        }

    def action_quit(self) -> None:
        """Quit the visualization screen."""
        # Clean up all plots
        self.viz_manager.close_all_plots()
        self.app.pop_screen()

    def action_refresh(self) -> None:
        """Refresh all displays."""
        self.refresh_plots_table()
        self.refresh_plot_selects()
        self.notify("Refreshed", severity="information")


# CSS styles for the visualization screen
VISUALIZATION_CSS = """
.section-title {
    text-style: bold;
    color: $primary;
    margin: 1 0;
}

.section-subtitle {
    text-style: bold;
    margin: 1 0;
}

.tab-content {
    padding: 1;
}

.form-container {
    padding: 1;
    border: solid $primary;
    margin: 1 0;
}

.button-row {
    margin: 1 0;
}

DataTable {
    height: 10;
}

Input {
    margin: 0 0 1 0;
}

Select {
    margin: 0 0 1 0;
}

Button {
    margin: 0 1 0 0;
}
"""
