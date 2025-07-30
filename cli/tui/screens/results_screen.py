"""Results Display Screen

This screen provides comprehensive result display with interactive visualization,
data tables, export options, and analysis summary.
"""

import json
from typing import Any
from typing import Dict
from typing import List

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
from textual.widgets import Input
from textual.widgets import Label
from textual.widgets import Log
from textual.widgets import Select
from textual.widgets import Static
from textual.widgets import Switch
from textual.widgets import TabbedContent
from textual.widgets import TabPane

# Import integration modules
try:
    from ...integration.analysis_service import AnalysisService
    from ...integration.models import AnalysisResult
    from ...integration.models import ProcessedResult
    from ...integration.result_processor import ResultProcessor
except ImportError:
    AnalysisService = None
    ResultProcessor = None
    AnalysisResult = None
    ProcessedResult = None


class PlotViewer(Container):
    """Interactive plot viewer for analysis results."""

    def __init__(self, plots: List[Dict[str, Any]], **kwargs):
        super().__init__(**kwargs)
        self.plots = plots
        self.current_plot_index = reactive(0)
        self.plot_settings = {
            "show_grid": True,
            "show_legend": True,
            "line_style": "solid",
            "marker_style": "circle",
        }

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Plot Viewer", classes="section-title")

            if not self.plots:
                yield Label("No plots available", classes="info-message")
                return

            # Plot selection
            with Horizontal():
                yield Label("Plot:", classes="field-label")
                yield Select(
                    options=[
                        (plot.get("title", f"Plot {i+1}"), i)
                        for i, plot in enumerate(self.plots)
                    ],
                    value=0,
                    id="plot_select",
                )
                yield Button("Export Plot", id="export_plot_btn", variant="default")

            # Plot display area (ASCII art representation)
            yield Container(id="plot_display_area", classes="plot-area")

            # Plot controls
            with Collapsible(title="Plot Settings", collapsed=True):
                with Vertical():
                    with Horizontal():
                        yield Label("Show Grid:", classes="field-label")
                        yield Switch(value=True, id="grid_switch")

                    with Horizontal():
                        yield Label("Show Legend:", classes="field-label")
                        yield Switch(value=True, id="legend_switch")

                    with Horizontal():
                        yield Label("Line Style:", classes="field-label")
                        yield Select(
                            options=[
                                ("Solid", "solid"),
                                ("Dashed", "dashed"),
                                ("Dotted", "dotted"),
                            ],
                            value="solid",
                            id="line_style_select",
                        )

            # Plot data table
            yield Label("Plot Data", classes="subsection-title")
            yield DataTable(id="plot_data_table")

    def on_mount(self) -> None:
        """Initialize plot viewer."""
        if self.plots:
            self._update_plot_display(0)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle plot selection changes."""
        if event.select.id == "plot_select":
            self.current_plot_index = event.value
            self._update_plot_display(event.value)

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle plot setting changes."""
        if event.switch.id == "grid_switch":
            self.plot_settings["show_grid"] = event.value
        elif event.switch.id == "legend_switch":
            self.plot_settings["show_legend"] = event.value

        self._update_plot_display(self.current_plot_index)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "export_plot_btn":
            self._export_current_plot()

    def _update_plot_display(self, plot_index: int) -> None:
        """Update the plot display."""
        if plot_index >= len(self.plots):
            return

        plot = self.plots[plot_index]

        # Update plot display area with ASCII representation
        display_area = self.query_one("#plot_display_area", Container)
        display_area.remove_children()

        # Create ASCII plot representation
        ascii_plot = self._create_ascii_plot(plot)
        display_area.mount(Static(ascii_plot, classes="ascii-plot"))

        # Update plot data table
        self._update_plot_data_table(plot)

    def _create_ascii_plot(self, plot: Dict[str, Any]) -> str:
        """Create ASCII representation of the plot."""
        plot_type = plot.get("type", "unknown")
        title = plot.get("title", "Plot")
        x_label = plot.get("x_label", "X")
        y_label = plot.get("y_label", "Y")

        # Simple ASCII plot representation
        ascii_art = f"""
┌─ {title} ─────────────────────────────────────────────┐
│                                                      │
│  {y_label}                                                │
│   ↑                                                  │
│   │                                                  │
│   │     ●                                            │
│   │       ●●                                         │
│   │         ●●●                                      │
│   │            ●●●                                   │
│   │               ●●●                                │
│   │                  ●●●                             │
│   │                     ●●●                          │
│   │                        ●●●                       │
│   │                           ●●●                    │
│   │                              ●●●                 │
│   └─────────────────────────────────────→            │
│                                        {x_label}        │
│                                                      │
└──────────────────────────────────────────────────────┘

Data Points: {len(plot.get('x_data', []))}
Plot Type: {plot_type.replace('_', ' ').title()}
"""

        if self.plot_settings["show_grid"]:
            ascii_art += "\n[Grid: ON]"

        if self.plot_settings["show_legend"]:
            ascii_art += f"\n[Legend: {plot.get('line_style', 'solid')} line]"

        return ascii_art

    def _update_plot_data_table(self, plot: Dict[str, Any]) -> None:
        """Update the plot data table."""
        table = self.query_one("#plot_data_table", DataTable)
        table.clear(columns=True)

        x_data = plot.get("x_data", [])
        y_data = plot.get("y_data", [])

        if not x_data or not y_data:
            table.add_column("Message")
            table.add_row("No data available")
            return

        # Add columns
        x_label = plot.get("x_label", "X")
        y_label = plot.get("y_label", "Y")
        table.add_columns(x_label, y_label)

        # Add data rows (limit to first 50 points for display)
        max_rows = min(50, len(x_data), len(y_data))
        for i in range(max_rows):
            x_val = (
                f"{x_data[i]:.4f}"
                if isinstance(x_data[i], (int, float))
                else str(x_data[i])
            )
            y_val = (
                f"{y_data[i]:.4f}"
                if isinstance(y_data[i], (int, float))
                else str(y_data[i])
            )
            table.add_row(x_val, y_val)

        if len(x_data) > 50:
            table.add_row("...", "...")
            table.add_row(f"Total: {len(x_data)} points", "")

    def _export_current_plot(self) -> None:
        """Export the current plot."""
        if not self.plots or self.current_plot_index >= len(self.plots):
            return

        plot = self.plots[self.current_plot_index]

        # TODO: Implement actual plot export
        self.notify(
            f"Plot export not yet implemented: {plot.get('title', 'Plot')}",
            severity="info",
        )


class DataTableViewer(Container):
    """Interactive data table viewer."""

    def __init__(self, tables: List[Dict[str, Any]], **kwargs):
        super().__init__(**kwargs)
        self.tables = tables
        self.current_table_index = reactive(0)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Data Tables", classes="section-title")

            if not self.tables:
                yield Label("No tables available", classes="info-message")
                return

            # Table selection
            with Horizontal():
                yield Label("Table:", classes="field-label")
                yield Select(
                    options=[
                        (table.get("title", f"Table {i+1}"), i)
                        for i, table in enumerate(self.tables)
                    ],
                    value=0,
                    id="table_select",
                )
                yield Button("Export Table", id="export_table_btn", variant="default")

            # Table display
            yield DataTable(id="main_data_table")

            # Table statistics
            yield Label("Table Statistics", classes="subsection-title")
            yield Container(id="table_stats_container")

    def on_mount(self) -> None:
        """Initialize table viewer."""
        if self.tables:
            self._update_table_display(0)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle table selection changes."""
        if event.select.id == "table_select":
            self.current_table_index = event.value
            self._update_table_display(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "export_table_btn":
            self._export_current_table()

    def _update_table_display(self, table_index: int) -> None:
        """Update the table display."""
        if table_index >= len(self.tables):
            return

        table_data = self.tables[table_index]
        table_widget = self.query_one("#main_data_table", DataTable)

        # Clear existing data
        table_widget.clear(columns=True)

        # Get table information
        columns = table_data.get("columns", [])
        column_labels = table_data.get("column_labels", columns)
        data = table_data.get("data", [])

        if not columns or not data:
            table_widget.add_column("Message")
            table_widget.add_row("No data available")
            return

        # Add columns
        table_widget.add_columns(*column_labels)

        # Add data rows
        for row in data:
            if isinstance(row, dict):
                row_values = [str(row.get(col, "")) for col in columns]
            elif isinstance(row, (list, tuple)):
                row_values = [str(val) for val in row]
            else:
                row_values = [str(row)]

            table_widget.add_row(*row_values)

        # Update statistics
        self._update_table_statistics(table_data)

    def _update_table_statistics(self, table_data: Dict[str, Any]) -> None:
        """Update table statistics."""
        stats_container = self.query_one("#table_stats_container", Container)
        stats_container.remove_children()

        data = table_data.get("data", [])
        columns = table_data.get("columns", [])

        stats_container.mount(
            Vertical(
                Label(f"Rows: {len(data)}", classes="stat-item"),
                Label(f"Columns: {len(columns)}", classes="stat-item"),
                Label(
                    f"Table: {table_data.get('title', 'Unknown')}",
                    classes="stat-item",
                ),
            ),
        )

        # Calculate numeric statistics if applicable
        if data and isinstance(data[0], dict):
            numeric_columns = []
            for col in columns:
                try:
                    values = [
                        float(row.get(col, 0))
                        for row in data
                        if row.get(col) is not None
                    ]
                    if values:
                        numeric_columns.append((col, values))
                except (ValueError, TypeError):
                    continue

            if numeric_columns:
                stats_container.mount(
                    Label("Numeric Statistics:", classes="subsection-title"),
                )
                for col_name, values in numeric_columns[
                    :3
                ]:  # Show first 3 numeric columns
                    mean_val = sum(values) / len(values)
                    min_val = min(values)
                    max_val = max(values)
                    stats_container.mount(
                        Label(
                            f"{col_name}: μ={mean_val:.3f}, min={min_val:.3f}, max={max_val:.3f}",
                            classes="stat-detail",
                        ),
                    )

    def _export_current_table(self) -> None:
        """Export the current table."""
        if not self.tables or self.current_table_index >= len(self.tables):
            return

        table = self.tables[self.current_table_index]

        # TODO: Implement actual table export
        self.notify(
            f"Table export not yet implemented: {table.get('title', 'Table')}",
            severity="info",
        )


class SummaryPanel(Container):
    """Analysis summary panel."""

    def __init__(self, summary: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.summary = summary

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Analysis Summary", classes="section-title")
            yield Container(id="summary_content")

    def on_mount(self) -> None:
        """Initialize summary panel."""
        self._populate_summary()

    def _populate_summary(self) -> None:
        """Populate the summary content."""
        content = self.query_one("#summary_content", Container)

        if not self.summary:
            content.mount(Label("No summary available", classes="info-message"))
            return

        # Basic information
        content.mount(Label("Basic Information", classes="subsection-title"))

        basic_info = [
            ("Analysis Type", self.summary.get("analysis_type", "Unknown")),
            ("Target", self.summary.get("target", "Unknown")),
            ("Solver", self.summary.get("solver", "Unknown")),
            ("Data Points", self.summary.get("data_points", "Unknown")),
        ]

        for label, value in basic_info:
            content.mount(Label(f"{label}: {value}", classes="summary-item"))

        # Conditions
        conditions = self.summary.get("conditions", {})
        if conditions:
            content.mount(Label("Analysis Conditions", classes="subsection-title"))
            for condition, value in conditions.items():
                content.mount(Label(f"{condition}: {value}", classes="summary-item"))

        # Key results
        key_results = self.summary.get("key_results", {})
        if key_results:
            content.mount(Label("Key Results", classes="subsection-title"))
            self._add_key_results(content, key_results)

    def _add_key_results(self, container: Container, results: Dict[str, Any]) -> None:
        """Add key results to the summary."""
        for key, value in results.items():
            if isinstance(value, dict):
                # Nested results (e.g., max_cl with value and alpha)
                container.mount(
                    Label(
                        f"{key.replace('_', ' ').title()}:",
                        classes="result-category",
                    ),
                )
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        formatted_value = f"{sub_value:.4f}"
                    else:
                        formatted_value = str(sub_value)
                    container.mount(
                        Label(
                            f"  {sub_key}: {formatted_value}",
                            classes="result-detail",
                        ),
                    )
            elif isinstance(value, (int, float)):
                formatted_value = f"{value:.4f}"
                container.mount(
                    Label(
                        f"{key.replace('_', ' ').title()}: {formatted_value}",
                        classes="result-item",
                    ),
                )
            elif isinstance(value, list):
                # Don't display large lists in summary
                if len(value) <= 5:
                    formatted_value = ", ".join(map(str, value))
                    container.mount(
                        Label(
                            f"{key.replace('_', ' ').title()}: {formatted_value}",
                            classes="result-item",
                        ),
                    )
                else:
                    container.mount(
                        Label(
                            f"{key.replace('_', ' ').title()}: {len(value)} values",
                            classes="result-item",
                        ),
                    )
            else:
                container.mount(
                    Label(
                        f"{key.replace('_', ' ').title()}: {value}",
                        classes="result-item",
                    ),
                )


class ExportPanel(Container):
    """Results export panel."""

    def __init__(self, processed_result: Any, **kwargs):
        super().__init__(**kwargs)
        self.processed_result = processed_result
        self.result_processor = ResultProcessor() if ResultProcessor else None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Export Results", classes="section-title")

            # Export format selection
            with Horizontal():
                yield Label("Format:", classes="field-label")
                yield Select(
                    options=[
                        ("JSON", "json"),
                        ("CSV", "csv"),
                        ("Excel", "xlsx"),
                        ("HTML", "html"),
                        ("PDF", "pdf"),
                    ],
                    value="json",
                    id="export_format_select",
                )

            # Export options
            with Vertical():
                yield Label("Export Options:", classes="field-label")

                with Horizontal():
                    yield Label("Include Plots:", classes="option-label")
                    yield Switch(value=True, id="include_plots_switch")

                with Horizontal():
                    yield Label("Include Raw Data:", classes="option-label")
                    yield Switch(value=False, id="include_raw_switch")

                with Horizontal():
                    yield Label("Include Summary:", classes="option-label")
                    yield Switch(value=True, id="include_summary_switch")

            # Output path
            with Horizontal():
                yield Label("Output Path:", classes="field-label")
                yield Input(
                    placeholder="Leave empty for auto-generated filename",
                    id="output_path_input",
                )
                yield Button("Browse", id="browse_output_btn", variant="default")

            # Export actions
            with Horizontal():
                yield Button("Export", id="export_btn", variant="success")
                yield Button("Preview", id="preview_btn", variant="default")

            # Export log
            yield Label("Export Log", classes="subsection-title")
            yield Log(id="export_log")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "export_btn":
            self._export_results()
        elif event.button.id == "preview_btn":
            self._preview_export()
        elif event.button.id == "browse_output_btn":
            self._browse_output_path()

    @work(exclusive=True)
    async def _export_results(self) -> None:
        """Export the results."""
        if not self.result_processor or not self.processed_result:
            self.notify("Export service not available", severity="error")
            return

        try:
            export_log = self.query_one("#export_log", Log)
            export_log.write("Starting export...")

            # Get export settings
            format_type = self.query_one("#export_format_select", Select).value
            output_path = self.query_one("#output_path_input", Input).value

            # Export options
            include_plots = self.query_one("#include_plots_switch", Switch).value
            include_raw = self.query_one("#include_raw_switch", Switch).value
            include_summary = self.query_one("#include_summary_switch", Switch).value

            export_log.write(f"Export format: {format_type}")
            export_log.write(f"Include plots: {include_plots}")
            export_log.write(f"Include raw data: {include_raw}")
            export_log.write(f"Include summary: {include_summary}")

            # Perform export
            output_file = self.result_processor.export_result(
                self.processed_result,
                format_type,
                output_path if output_path else None,
            )

            export_log.write(f"Export completed: {output_file}")
            self.notify(f"Results exported to: {output_file}", severity="success")

        except Exception as e:
            export_log = self.query_one("#export_log", Log)
            export_log.write(f"Export failed: {str(e)}")
            self.notify(f"Export failed: {str(e)}", severity="error")

    def _preview_export(self) -> None:
        """Preview the export."""
        format_type = self.query_one("#export_format_select", Select).value

        export_log = self.query_one("#export_log", Log)
        export_log.write(f"Preview for {format_type} format:")

        if format_type == "json":
            # Show JSON preview
            if self.processed_result and hasattr(
                self.processed_result,
                "formatted_data",
            ):
                preview_data = str(self.processed_result.formatted_data)[:500] + "..."
                export_log.write(f"JSON Preview: {preview_data}")
        else:
            export_log.write(f"Preview not available for {format_type} format")

    def _browse_output_path(self) -> None:
        """Browse for output path."""
        # TODO: Implement file browser
        self.notify("File browser not yet implemented", severity="info")


class ResultsScreen(Screen):
    """Main results display screen."""

    BINDINGS = [
        Binding("ctrl+e", "export", "Export"),
        Binding("ctrl+s", "save", "Save"),
        Binding("ctrl+p", "print", "Print"),
        Binding("escape", "back", "Back"),
    ]

    def __init__(self, analysis_result: Any, **kwargs):
        super().__init__(**kwargs)
        self.analysis_result = analysis_result
        self.processed_result = None
        self.result_processor = ResultProcessor() if ResultProcessor else None

    def compose(self) -> ComposeResult:
        with Container():
            yield Label("Analysis Results", classes="screen-title")

            with TabbedContent():
                with TabPane("Summary", id="summary_tab"):
                    yield Container(id="summary_container")

                with TabPane("Plots", id="plots_tab"):
                    yield Container(id="plots_container")

                with TabPane("Data", id="data_tab"):
                    yield Container(id="data_container")

                with TabPane("Export", id="export_tab"):
                    yield Container(id="export_container")

                with TabPane("Raw Data", id="raw_tab"):
                    yield ScrollableContainer(
                        Static("", id="raw_data_display", classes="raw-data"),
                        classes="raw-data-container",
                    )

            with Horizontal(classes="screen-actions"):
                yield Button(
                    "Export Results",
                    id="export_results_btn",
                    variant="primary",
                )
                yield Button("Save Session", id="save_session_btn", variant="default")
                yield Button("New Analysis", id="new_analysis_btn", variant="default")
                yield Button("Back", id="back_btn", variant="default")

    def on_mount(self) -> None:
        """Initialize results screen."""
        self._process_results()

    @work(exclusive=True)
    async def _process_results(self) -> None:
        """Process the analysis results."""
        if not self.result_processor or not self.analysis_result:
            self.notify("Result processor not available", severity="error")
            return

        try:
            # Process the results
            self.processed_result = self.result_processor.process_result(
                self.analysis_result,
            )

            # Populate all tabs
            self._populate_summary_tab()
            self._populate_plots_tab()
            self._populate_data_tab()
            self._populate_export_tab()
            self._populate_raw_data_tab()

        except Exception as e:
            self.notify(f"Error processing results: {e}", severity="error")

    def _populate_summary_tab(self) -> None:
        """Populate the summary tab."""
        container = self.query_one("#summary_container", Container)

        if self.processed_result and hasattr(self.processed_result, "summary"):
            summary_panel = SummaryPanel(self.processed_result.summary)
            container.mount(summary_panel)
        else:
            container.mount(Label("No summary available", classes="info-message"))

    def _populate_plots_tab(self) -> None:
        """Populate the plots tab."""
        container = self.query_one("#plots_container", Container)

        if self.processed_result and hasattr(self.processed_result, "plots"):
            plot_viewer = PlotViewer(self.processed_result.plots)
            container.mount(plot_viewer)
        else:
            container.mount(Label("No plots available", classes="info-message"))

    def _populate_data_tab(self) -> None:
        """Populate the data tab."""
        container = self.query_one("#data_container", Container)

        if self.processed_result and hasattr(self.processed_result, "tables"):
            table_viewer = DataTableViewer(self.processed_result.tables)
            container.mount(table_viewer)
        else:
            container.mount(Label("No data tables available", classes="info-message"))

    def _populate_export_tab(self) -> None:
        """Populate the export tab."""
        container = self.query_one("#export_container", Container)

        if self.processed_result:
            export_panel = ExportPanel(self.processed_result)
            container.mount(export_panel)
        else:
            container.mount(Label("No results to export", classes="info-message"))

    def _populate_raw_data_tab(self) -> None:
        """Populate the raw data tab."""
        raw_display = self.query_one("#raw_data_display", Static)

        if self.analysis_result and hasattr(self.analysis_result, "raw_data"):
            try:
                # Format raw data as JSON for display
                raw_data_str = json.dumps(
                    self.analysis_result.raw_data,
                    indent=2,
                    default=str,
                )
                raw_display.update(raw_data_str)
            except Exception:
                raw_display.update(str(self.analysis_result.raw_data))
        else:
            raw_display.update("No raw data available")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "export_results_btn":
            self._export_results()
        elif event.button.id == "save_session_btn":
            self._save_session()
        elif event.button.id == "new_analysis_btn":
            self._new_analysis()
        elif event.button.id == "back_btn":
            self.app.pop_screen()

    def _export_results(self) -> None:
        """Export results action."""
        # Switch to export tab
        tabbed_content = self.query_one(TabbedContent)
        tabbed_content.active = "export_tab"

    def _save_session(self) -> None:
        """Save current session."""
        # TODO: Implement session saving
        self.notify("Session saving not yet implemented", severity="info")

    def _new_analysis(self) -> None:
        """Start a new analysis."""
        # Return to analysis configuration screen
        from .analysis_screen import AnalysisScreen

        self.app.push_screen(AnalysisScreen())

    def action_export(self) -> None:
        """Export action."""
        self._export_results()

    def action_save(self) -> None:
        """Save action."""
        self._save_session()

    def action_print(self) -> None:
        """Print action."""
        # TODO: Implement printing
        self.notify("Print functionality not yet implemented", severity="info")

    def action_back(self) -> None:
        """Go back action."""
        self.app.pop_screen()
