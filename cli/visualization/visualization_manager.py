"""Visualization Manager - Central coordinator for all visualization operations

This module provides the main interface for managing visualization operations in the ICARUS CLI.
It coordinates between different visualization components and provides a unified API.
"""

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from rich.console import Console

from .chart_generator import ChartGenerator
from .export_manager import ExportManager
from .interactive_plotter import InteractivePlotter
from .plot_customizer import PlotCustomizer
from .real_time_updater import RealTimeUpdater


class VisualizationManager:
    """Central manager for all visualization operations."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize the visualization manager.

        Args:
            console: Rich console for output (optional)
        """
        self.console = console or Console()

        # Initialize components
        self.interactive_plotter = InteractivePlotter(console=self.console)
        self.chart_generator = ChartGenerator(console=self.console)
        self.plot_customizer = PlotCustomizer(console=self.console)
        self.export_manager = ExportManager(console=self.console)
        self.real_time_updater = RealTimeUpdater(console=self.console)

        # Track active visualizations
        self.active_plots: Dict[str, Any] = {}
        self.plot_counter = 0

    def create_interactive_plot(
        self,
        data: Dict[str, Any],
        plot_type: str = "line",
        title: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Create an interactive plot with customization options.

        Args:
            data: Data to plot
            plot_type: Type of plot (line, scatter, bar, etc.)
            title: Plot title
            **kwargs: Additional plot parameters

        Returns:
            Plot ID for tracking
        """
        plot_id = f"plot_{self.plot_counter}"
        self.plot_counter += 1

        try:
            plot_obj = self.interactive_plotter.create_plot(
                data=data,
                plot_type=plot_type,
                title=title,
                **kwargs,
            )

            self.active_plots[plot_id] = {
                "plot": plot_obj,
                "type": plot_type,
                "data": data,
                "title": title,
                "kwargs": kwargs,
            }

            self.console.print(f"[green]✓[/green] Created interactive plot: {plot_id}")
            return plot_id

        except Exception as e:
            self.console.print(f"[red]✗[/red] Failed to create plot: {e}")
            raise

    def generate_chart(
        self,
        analysis_results: Dict[str, Any],
        chart_type: str = "polar",
        **options,
    ) -> str:
        """Generate a chart from analysis results.

        Args:
            analysis_results: Results from ICARUS analysis
            chart_type: Type of chart to generate
            **options: Chart generation options

        Returns:
            Chart ID for tracking
        """
        chart_id = f"chart_{self.plot_counter}"
        self.plot_counter += 1

        try:
            chart_obj = self.chart_generator.generate_chart(
                results=analysis_results,
                chart_type=chart_type,
                **options,
            )

            self.active_plots[chart_id] = {
                "plot": chart_obj,
                "type": f"chart_{chart_type}",
                "data": analysis_results,
                "options": options,
            }

            self.console.print(f"[green]✓[/green] Generated chart: {chart_id}")
            return chart_id

        except Exception as e:
            self.console.print(f"[red]✗[/red] Failed to generate chart: {e}")
            raise

    def customize_plot(self, plot_id: str, customizations: Dict[str, Any]) -> bool:
        """Apply customizations to an existing plot.

        Args:
            plot_id: ID of the plot to customize
            customizations: Dictionary of customization options

        Returns:
            True if successful, False otherwise
        """
        if plot_id not in self.active_plots:
            self.console.print(f"[red]✗[/red] Plot {plot_id} not found")
            return False

        try:
            plot_info = self.active_plots[plot_id]
            customized_plot = self.plot_customizer.apply_customizations(
                plot=plot_info["plot"],
                customizations=customizations,
            )

            # Update the stored plot
            self.active_plots[plot_id]["plot"] = customized_plot
            self.active_plots[plot_id]["customizations"] = customizations

            self.console.print(f"[green]✓[/green] Customized plot: {plot_id}")
            return True

        except Exception as e:
            self.console.print(f"[red]✗[/red] Failed to customize plot: {e}")
            return False

    def export_plot(
        self,
        plot_id: str,
        output_path: Union[str, Path],
        format: str = "png",
        **export_options,
    ) -> bool:
        """Export a plot to file.

        Args:
            plot_id: ID of the plot to export
            output_path: Path where to save the plot
            format: Export format (png, pdf, svg, etc.)
            **export_options: Additional export options

        Returns:
            True if successful, False otherwise
        """
        if plot_id not in self.active_plots:
            self.console.print(f"[red]✗[/red] Plot {plot_id} not found")
            return False

        try:
            plot_info = self.active_plots[plot_id]
            success = self.export_manager.export_plot(
                plot=plot_info["plot"],
                output_path=output_path,
                format=format,
                **export_options,
            )

            if success:
                self.console.print(
                    f"[green]✓[/green] Exported plot {plot_id} to {output_path}",
                )
            else:
                self.console.print(f"[red]✗[/red] Failed to export plot {plot_id}")

            return success

        except Exception as e:
            self.console.print(f"[red]✗[/red] Export failed: {e}")
            return False

    def start_real_time_updates(
        self,
        plot_id: str,
        data_source: Any,
        update_interval: float = 1.0,
    ) -> bool:
        """Start real-time updates for a plot.

        Args:
            plot_id: ID of the plot to update
            data_source: Source of real-time data
            update_interval: Update interval in seconds

        Returns:
            True if successful, False otherwise
        """
        if plot_id not in self.active_plots:
            self.console.print(f"[red]✗[/red] Plot {plot_id} not found")
            return False

        try:
            plot_info = self.active_plots[plot_id]
            success = self.real_time_updater.start_updates(
                plot=plot_info["plot"],
                data_source=data_source,
                update_interval=update_interval,
            )

            if success:
                self.console.print(
                    f"[green]✓[/green] Started real-time updates for {plot_id}",
                )
                plot_info["real_time"] = True
            else:
                self.console.print(
                    f"[red]✗[/red] Failed to start real-time updates for {plot_id}",
                )

            return success

        except Exception as e:
            self.console.print(f"[red]✗[/red] Real-time update setup failed: {e}")
            return False

    def stop_real_time_updates(self, plot_id: str) -> bool:
        """Stop real-time updates for a plot.

        Args:
            plot_id: ID of the plot

        Returns:
            True if successful, False otherwise
        """
        if plot_id not in self.active_plots:
            self.console.print(f"[red]✗[/red] Plot {plot_id} not found")
            return False

        try:
            plot_info = self.active_plots[plot_id]
            if plot_info.get("real_time", False):
                success = self.real_time_updater.stop_updates(plot_info["plot"])
                if success:
                    plot_info["real_time"] = False
                    self.console.print(
                        f"[green]✓[/green] Stopped real-time updates for {plot_id}",
                    )
                return success
            else:
                self.console.print(
                    f"[yellow]![/yellow] Plot {plot_id} is not in real-time mode",
                )
                return True

        except Exception as e:
            self.console.print(f"[red]✗[/red] Failed to stop real-time updates: {e}")
            return False

    def list_active_plots(self) -> List[Dict[str, Any]]:
        """Get list of all active plots.

        Returns:
            List of plot information dictionaries
        """
        plots_info = []
        for plot_id, plot_info in self.active_plots.items():
            plots_info.append(
                {
                    "id": plot_id,
                    "type": plot_info["type"],
                    "title": plot_info.get("title", "Untitled"),
                    "real_time": plot_info.get("real_time", False),
                    "has_customizations": "customizations" in plot_info,
                },
            )
        return plots_info

    def close_plot(self, plot_id: str) -> bool:
        """Close and remove a plot.

        Args:
            plot_id: ID of the plot to close

        Returns:
            True if successful, False otherwise
        """
        if plot_id not in self.active_plots:
            self.console.print(f"[red]✗[/red] Plot {plot_id} not found")
            return False

        try:
            plot_info = self.active_plots[plot_id]

            # Stop real-time updates if active
            if plot_info.get("real_time", False):
                self.stop_real_time_updates(plot_id)

            # Close the plot
            if hasattr(plot_info["plot"], "close"):
                plot_info["plot"].close()

            # Remove from active plots
            del self.active_plots[plot_id]

            self.console.print(f"[green]✓[/green] Closed plot: {plot_id}")
            return True

        except Exception as e:
            self.console.print(f"[red]✗[/red] Failed to close plot: {e}")
            return False

    def close_all_plots(self) -> int:
        """Close all active plots.

        Returns:
            Number of plots closed
        """
        plot_ids = list(self.active_plots.keys())
        closed_count = 0

        for plot_id in plot_ids:
            if self.close_plot(plot_id):
                closed_count += 1

        self.console.print(f"[green]✓[/green] Closed {closed_count} plots")
        return closed_count

    async def batch_export(
        self,
        plot_ids: List[str],
        output_directory: Union[str, Path],
        format: str = "png",
        **export_options,
    ) -> Dict[str, bool]:
        """Export multiple plots in batch.

        Args:
            plot_ids: List of plot IDs to export
            output_directory: Directory to save plots
            format: Export format
            **export_options: Additional export options

        Returns:
            Dictionary mapping plot IDs to success status
        """
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        for plot_id in plot_ids:
            if plot_id in self.active_plots:
                plot_info = self.active_plots[plot_id]
                title = plot_info.get("title", plot_id)
                filename = f"{title.replace(' ', '_')}.{format}"
                output_path = output_dir / filename

                results[plot_id] = self.export_plot(
                    plot_id=plot_id,
                    output_path=output_path,
                    format=format,
                    **export_options,
                )
            else:
                results[plot_id] = False
                self.console.print(f"[red]✗[/red] Plot {plot_id} not found")

        successful = sum(results.values())
        total = len(plot_ids)
        self.console.print(
            f"[green]✓[/green] Batch export completed: {successful}/{total} successful",
        )

        return results
