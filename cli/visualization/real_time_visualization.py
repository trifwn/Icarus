"""Real-Time Visualization - Advanced visualization with real-time updates

This module provides real-time visualization capabilities for the ICARUS CLI,
including animated plots, streaming data visualization, and interactive controls.
"""

import asyncio
import math
import random
import time
from datetime import datetime
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


class DataSource:
    """Abstract base class for data sources."""

    def __init__(self, name: str):
        """Initialize the data source.

        Args:
            name: Data source name
        """
        self.name = name
        self.listeners: List[Callable[[Dict[str, Any]], None]] = []

    def add_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        """Add a data listener.

        Args:
            listener: Callback function that receives data updates
        """
        if listener not in self.listeners:
            self.listeners.append(listener)

    def remove_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        """Remove a data listener.

        Args:
            listener: Listener to remove
        """
        if listener in self.listeners:
            self.listeners.remove(listener)

    def notify_listeners(self, data: Dict[str, Any]) -> None:
        """Notify all listeners with new data.

        Args:
            data: Data update
        """
        for listener in self.listeners:
            try:
                listener(data)
            except Exception as e:
                print(f"Error in listener: {e}")

    async def start(self) -> None:
        """Start the data source."""
        pass

    async def stop(self) -> None:
        """Stop the data source."""
        pass


class SimulatedDataSource(DataSource):
    """Simulated data source for testing."""

    def __init__(
        self,
        name: str,
        data_type: str = "scalar",
        update_interval: float = 1.0,
    ):
        """Initialize the simulated data source.

        Args:
            name: Data source name
            data_type: Type of data to generate (scalar, vector, matrix)
            update_interval: Update interval in seconds
        """
        super().__init__(name)
        self.data_type = data_type
        self.update_interval = update_interval
        self.running = False
        self.task = None

        # Initial values
        self.value = 0.0
        self.trend = 0.1
        self.noise = 0.05
        self.time = 0.0

    async def start(self) -> None:
        """Start the data source."""
        if self.running:
            return

        self.running = True
        self.task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the data source."""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

    async def _run(self) -> None:
        """Run the data generation loop."""
        try:
            while self.running:
                data = self._generate_data()
                self.notify_listeners(data)
                await asyncio.sleep(self.update_interval)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error in data source: {e}")

    def _generate_data(self) -> Dict[str, Any]:
        """Generate simulated data.

        Returns:
            Generated data
        """
        self.time += self.update_interval

        # Add random walk with noise
        self.value += self.trend + random.uniform(-self.noise, self.noise)

        # Occasionally change trend
        if random.random() < 0.1:
            self.trend = random.uniform(-0.2, 0.2)

        # Keep value in reasonable range
        if abs(self.value) > 10:
            self.trend = -0.1 * self.value

        if self.data_type == "scalar":
            return {
                "timestamp": self.time,
                "value": self.value,
                "name": self.name,
            }
        elif self.data_type == "vector":
            return {
                "timestamp": self.time,
                "values": [
                    self.value,
                    self.value * math.sin(self.time * 0.1),
                    self.value * math.cos(self.time * 0.1),
                ],
                "name": self.name,
            }
        elif self.data_type == "matrix":
            size = 3
            matrix = []
            for i in range(size):
                row = []
                for j in range(size):
                    row.append(
                        self.value * math.sin(self.time * 0.1 + i * 0.2 + j * 0.3),
                    )
                matrix.append(row)

            return {
                "timestamp": self.time,
                "matrix": matrix,
                "name": self.name,
            }
        else:
            return {
                "timestamp": self.time,
                "value": self.value,
                "name": self.name,
            }


class RealTimeVisualizer:
    """Real-time data visualizer."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize the real-time visualizer.

        Args:
            console: Rich console for output (optional)
        """
        self.console = console or Console()
        self.data_sources: Dict[str, DataSource] = {}
        self.visualizations: Dict[str, Any] = {}
        self.running = False

    def add_data_source(self, source: DataSource) -> None:
        """Add a data source.

        Args:
            source: Data source to add
        """
        self.data_sources[source.name] = source

    def remove_data_source(self, name: str) -> bool:
        """Remove a data source.

        Args:
            name: Data source name

        Returns:
            True if successful, False if not found
        """
        if name in self.data_sources:
            del self.data_sources[name]
            return True
        return False

    def create_visualization(
        self,
        name: str,
        viz_type: str,
        data_source: str,
        **options,
    ) -> bool:
        """Create a visualization.

        Args:
            name: Visualization name
            viz_type: Visualization type (line, bar, gauge, etc.)
            data_source: Data source name
            **options: Visualization options

        Returns:
            True if successful
        """
        if data_source not in self.data_sources:
            self.console.print(f"[red]✗[/red] Data source not found: {data_source}")
            return False

        if name in self.visualizations:
            self.console.print(f"[red]✗[/red] Visualization already exists: {name}")
            return False

        try:
            if viz_type == "line":
                viz = self._create_line_visualization(
                    name,
                    self.data_sources[data_source],
                    **options,
                )
            elif viz_type == "bar":
                viz = self._create_bar_visualization(
                    name,
                    self.data_sources[data_source],
                    **options,
                )
            elif viz_type == "gauge":
                viz = self._create_gauge_visualization(
                    name,
                    self.data_sources[data_source],
                    **options,
                )
            elif viz_type == "table":
                viz = self._create_table_visualization(
                    name,
                    self.data_sources[data_source],
                    **options,
                )
            else:
                self.console.print(
                    f"[red]✗[/red] Unsupported visualization type: {viz_type}",
                )
                return False

            self.visualizations[name] = {
                "type": viz_type,
                "data_source": data_source,
                "options": options,
                "visualization": viz,
            }

            self.console.print(f"[green]✓[/green] Created visualization: {name}")
            return True

        except Exception as e:
            self.console.print(f"[red]✗[/red] Failed to create visualization: {e}")
            return False

    def _create_line_visualization(
        self,
        name: str,
        data_source: DataSource,
        **options,
    ) -> Dict[str, Any]:
        """Create a line visualization.

        Args:
            name: Visualization name
            data_source: Data source
            **options: Visualization options

        Returns:
            Visualization object
        """
        max_points = options.get("max_points", 100)
        title = options.get("title", name)
        width = options.get("width", 80)
        height = options.get("height", 20)

        viz = {
            "title": title,
            "width": width,
            "height": height,
            "x_data": [],
            "y_data": [],
            "max_points": max_points,
        }

        # Add data listener
        def update_line_data(data: Dict[str, Any]) -> None:
            viz["x_data"].append(data.get("timestamp", len(viz["x_data"])))

            if "value" in data:
                viz["y_data"].append(data["value"])
            elif "values" in data and len(data["values"]) > 0:
                viz["y_data"].append(data["values"][0])

            # Trim to max points
            if len(viz["x_data"]) > max_points:
                viz["x_data"] = viz["x_data"][-max_points:]
                viz["y_data"] = viz["y_data"][-max_points:]

        data_source.add_listener(update_line_data)
        viz["listener"] = update_line_data

        return viz

    def _create_bar_visualization(
        self,
        name: str,
        data_source: DataSource,
        **options,
    ) -> Dict[str, Any]:
        """Create a bar visualization.

        Args:
            name: Visualization name
            data_source: Data source
            **options: Visualization options

        Returns:
            Visualization object
        """
        title = options.get("title", name)
        width = options.get("width", 80)
        height = options.get("height", 20)

        viz = {
            "title": title,
            "width": width,
            "height": height,
            "categories": options.get("categories", ["A", "B", "C"]),
            "values": [0] * len(options.get("categories", ["A", "B", "C"])),
        }

        # Add data listener
        def update_bar_data(data: Dict[str, Any]) -> None:
            if "values" in data and len(data["values"]) == len(viz["categories"]):
                viz["values"] = data["values"]
            elif "value" in data:
                # Update a random bar
                import random

                index = random.randint(0, len(viz["values"]) - 1)
                viz["values"][index] = data["value"]

        data_source.add_listener(update_bar_data)
        viz["listener"] = update_bar_data

        return viz

    def _create_gauge_visualization(
        self,
        name: str,
        data_source: DataSource,
        **options,
    ) -> Dict[str, Any]:
        """Create a gauge visualization.

        Args:
            name: Visualization name
            data_source: Data source
            **options: Visualization options

        Returns:
            Visualization object
        """
        title = options.get("title", name)
        min_value = options.get("min_value", 0)
        max_value = options.get("max_value", 100)

        viz = {
            "title": title,
            "min_value": min_value,
            "max_value": max_value,
            "value": min_value,
        }

        # Add data listener
        def update_gauge_data(data: Dict[str, Any]) -> None:
            if "value" in data:
                viz["value"] = max(min_value, min(max_value, data["value"]))

        data_source.add_listener(update_gauge_data)
        viz["listener"] = update_gauge_data

        return viz

    def _create_table_visualization(
        self,
        name: str,
        data_source: DataSource,
        **options,
    ) -> Dict[str, Any]:
        """Create a table visualization.

        Args:
            name: Visualization name
            data_source: Data source
            **options: Visualization options

        Returns:
            Visualization object
        """
        title = options.get("title", name)
        columns = options.get("columns", ["Time", "Value"])
        max_rows = options.get("max_rows", 10)

        viz = {
            "title": title,
            "columns": columns,
            "rows": [],
            "max_rows": max_rows,
        }

        # Add data listener
        def update_table_data(data: Dict[str, Any]) -> None:
            timestamp = datetime.fromtimestamp(
                data.get("timestamp", time.time()),
            ).strftime("%H:%M:%S")

            if "value" in data:
                row = [timestamp, f"{data['value']:.2f}"]
            elif "values" in data:
                row = [timestamp] + [f"{v:.2f}" for v in data["values"]]
            else:
                row = [timestamp, "N/A"]

            viz["rows"].append(row)

            # Trim to max rows
            if len(viz["rows"]) > max_rows:
                viz["rows"] = viz["rows"][-max_rows:]

        data_source.add_listener(update_table_data)
        viz["listener"] = update_table_data

        return viz

    def remove_visualization(self, name: str) -> bool:
        """Remove a visualization.

        Args:
            name: Visualization name

        Returns:
            True if successful, False if not found
        """
        if name not in self.visualizations:
            return False

        viz_info = self.visualizations[name]
        data_source_name = viz_info["data_source"]

        if data_source_name in self.data_sources and "listener" in viz_info:
            self.data_sources[data_source_name].remove_listener(viz_info["listener"])

        del self.visualizations[name]
        return True

    async def start(self) -> None:
        """Start all data sources and visualizations."""
        if self.running:
            return

        self.running = True

        # Start all data sources
        for source in self.data_sources.values():
            await source.start()

    async def stop(self) -> None:
        """Stop all data sources and visualizations."""
        self.running = False

        # Stop all data sources
        for source in self.data_sources.values():
            await source.stop()

    def render_visualization(self, name: str) -> Union[str, Panel]:
        """Render a visualization to a string or Rich panel.

        Args:
            name: Visualization name

        Returns:
            Rendered visualization
        """
        if name not in self.visualizations:
            return f"Visualization not found: {name}"

        viz_info = self.visualizations[name]
        viz_type = viz_info["type"]
        viz = viz_info["visualization"]

        if viz_type == "line":
            return self._render_line_visualization(viz)
        elif viz_type == "bar":
            return self._render_bar_visualization(viz)
        elif viz_type == "gauge":
            return self._render_gauge_visualization(viz)
        elif viz_type == "table":
            return self._render_table_visualization(viz)
        else:
            return f"Unsupported visualization type: {viz_type}"

    def _render_line_visualization(self, viz: Dict[str, Any]) -> Panel:
        """Render a line visualization.

        Args:
            viz: Visualization data

        Returns:
            Rich panel with the visualization
        """
        title = viz["title"]
        width = viz["width"]
        height = viz["height"]
        x_data = viz["x_data"]
        y_data = viz["y_data"]

        if not x_data or not y_data:
            return Panel("No data available", title=title)

        # Determine y-axis range
        if y_data:
            y_min = min(y_data)
            y_max = max(y_data)
            y_range = max(0.1, y_max - y_min)
        else:
            y_min = 0
            y_max = 1
            y_range = 1

        # Create canvas
        canvas = [[" " for _ in range(width)] for _ in range(height)]

        # Plot points
        for i in range(1, len(y_data)):
            x1 = int((i - 1) / (len(y_data) - 1) * (width - 1))
            y1 = int((height - 1) - (y_data[i - 1] - y_min) / y_range * (height - 1))
            x2 = int(i / (len(y_data) - 1) * (width - 1))
            y2 = int((height - 1) - (y_data[i] - y_min) / y_range * (height - 1))

            # Ensure points are within bounds
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))

            # Draw line using Bresenham's algorithm
            self._draw_line(canvas, x1, y1, x2, y2)

        # Convert canvas to string
        result = ""
        for row in canvas:
            result += "".join(row) + "\n"

        # Add y-axis labels
        result += f"Min: {y_min:.2f} Max: {y_max:.2f}"

        return Panel(result, title=title)

    def _render_bar_visualization(self, viz: Dict[str, Any]) -> Panel:
        """Render a bar visualization.

        Args:
            viz: Visualization data

        Returns:
            Rich panel with the visualization
        """
        title = viz["title"]
        categories = viz["categories"]
        values = viz["values"]

        table = Table(title=title)
        table.add_column("Category")
        table.add_column("Value")
        table.add_column("Bar")

        max_value = max(values) if values else 1

        for category, value in zip(categories, values):
            bar_width = int(20 * value / max_value) if max_value > 0 else 0
            bar = "█" * bar_width
            table.add_row(category, f"{value:.2f}", bar)

        return table

    def _render_gauge_visualization(self, viz: Dict[str, Any]) -> Panel:
        """Render a gauge visualization.

        Args:
            viz: Visualization data

        Returns:
            Rich panel with the visualization
        """
        title = viz["title"]
        min_value = viz["min_value"]
        max_value = viz["max_value"]
        value = viz["value"]

        # Calculate percentage
        percentage = (
            (value - min_value) / (max_value - min_value)
            if max_value > min_value
            else 0
        )
        percentage = max(0, min(1, percentage))

        # Create gauge
        gauge_width = 40
        filled = int(gauge_width * percentage)
        gauge = "█" * filled + "░" * (gauge_width - filled)

        return Panel(
            f"{gauge}\n\n{value:.2f} / {max_value:.2f} ({percentage:.1%})",
            title=title,
        )

    def _render_table_visualization(self, viz: Dict[str, Any]) -> Table:
        """Render a table visualization.

        Args:
            viz: Visualization data

        Returns:
            Rich table
        """
        title = viz["title"]
        columns = viz["columns"]
        rows = viz["rows"]

        table = Table(title=title)

        for column in columns:
            table.add_column(column)

        for row in rows:
            table.add_row(*row)

        return table

    def _draw_line(
        self,
        canvas: List[List[str]],
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> None:
        """Draw a line on the canvas using Bresenham's algorithm.

        Args:
            canvas: Canvas to draw on
            x1, y1: Start point
            x2, y2: End point
        """
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            if 0 <= x1 < len(canvas[0]) and 0 <= y1 < len(canvas):
                canvas[y1][x1] = "█"

            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

    async def run_demo(self) -> None:
        """Run a demonstration of real-time visualization."""
        # Create data sources
        scalar_source = SimulatedDataSource("scalar_data", "scalar", 0.5)
        vector_source = SimulatedDataSource("vector_data", "vector", 0.5)

        self.add_data_source(scalar_source)
        self.add_data_source(vector_source)

        # Create visualizations
        self.create_visualization(
            "temperature",
            "line",
            "scalar_data",
            title="Temperature",
            max_points=50,
        )

        self.create_visualization(
            "system_metrics",
            "bar",
            "vector_data",
            title="System Metrics",
            categories=["CPU", "Memory", "Disk"],
        )

        self.create_visualization(
            "cpu_gauge",
            "gauge",
            "scalar_data",
            title="CPU Usage",
            min_value=0,
            max_value=10,
        )

        self.create_visualization(
            "data_log",
            "table",
            "scalar_data",
            title="Data Log",
            columns=["Time", "Value"],
            max_rows=5,
        )

        # Start visualization
        await self.start()

        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="top"),
            Layout(name="bottom"),
        )

        layout["top"].split_row(
            Layout(name="top_left"),
            Layout(name="top_right"),
        )

        layout["bottom"].split_row(
            Layout(name="bottom_left"),
            Layout(name="bottom_right"),
        )

        # Run live display
        with Live(layout, refresh_per_second=4) as live:
            try:
                for _ in range(100):  # Run for 100 updates
                    # Update layout with visualizations
                    layout["top_left"].update(self.render_visualization("temperature"))
                    layout["top_right"].update(
                        self.render_visualization("system_metrics"),
                    )
                    layout["bottom_left"].update(self.render_visualization("cpu_gauge"))
                    layout["bottom_right"].update(self.render_visualization("data_log"))

                    await asyncio.sleep(0.25)
            finally:
                await self.stop()


# Run demo if executed directly
if __name__ == "__main__":

    async def main():
        visualizer = RealTimeVisualizer()
        await visualizer.run_demo()

    asyncio.run(main())
