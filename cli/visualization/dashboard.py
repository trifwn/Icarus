"""Dashboard - Interactive dashboard with customizable widgets

This module provides a customizable dashboard for the ICARUS CLI with real-time
updates and interactive widgets.
"""

import asyncio
from datetime import datetime
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container
from textual.containers import Grid
from textual.containers import Horizontal
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button
from textual.widgets import DataTable
from textual.widgets import Label
from textual.widgets import ProgressBar
from textual.widgets import Static


class DashboardWidget(Static):
    """Base class for dashboard widgets."""

    def __init__(
        self,
        title: str,
        id: Optional[str] = None,
        classes: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the widget.

        Args:
            title: Widget title
            id: Widget ID
            classes: CSS classes
            **kwargs: Additional widget parameters
        """
        super().__init__(id=id, classes=f"dashboard-widget {classes or ''}", **kwargs)
        self.title = title
        self._last_update = datetime.now()
        self._update_interval = kwargs.get("update_interval", 5.0)  # seconds
        self._auto_update = kwargs.get("auto_update", False)
        self._update_task = None

    def compose(self) -> ComposeResult:
        """Compose the widget."""
        yield Label(self.title, classes="dashboard-widget-header")
        with Container(classes="dashboard-widget-content"):
            yield from self.compose_content()

    def compose_content(self) -> ComposeResult:
        """Compose the widget content. Override in subclasses."""
        yield Label("Widget content")

    def on_mount(self) -> None:
        """Handle mount event."""
        if self._auto_update:
            self.start_auto_update()

    def on_unmount(self) -> None:
        """Handle unmount event."""
        self.stop_auto_update()

    def start_auto_update(self) -> None:
        """Start automatic updates."""
        if self._update_task is None:
            self._update_task = asyncio.create_task(self._auto_update_task())

    def stop_auto_update(self) -> None:
        """Stop automatic updates."""
        if self._update_task is not None:
            self._update_task.cancel()
            self._update_task = None

    async def _auto_update_task(self) -> None:
        """Background task for automatic updates."""
        try:
            while True:
                await asyncio.sleep(self._update_interval)
                await self.update_content()
                self._last_update = datetime.now()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.app.log.error(f"Error in widget auto-update: {e}")

    async def update_content(self) -> None:
        """Update widget content. Override in subclasses."""
        pass


class StatWidget(DashboardWidget):
    """Widget for displaying a single statistic."""

    value = reactive("0")
    label = reactive("")

    def __init__(
        self,
        title: str,
        value: str = "0",
        label: str = "",
        id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the stat widget.

        Args:
            title: Widget title
            value: Statistic value
            label: Statistic label
            id: Widget ID
            **kwargs: Additional widget parameters
        """
        super().__init__(title=title, id=id, **kwargs)
        self.value = value
        self.label = label

    def compose_content(self) -> ComposeResult:
        """Compose the widget content."""
        with Container(classes="stat-display"):
            yield Label(self.value, classes="stat-value", id=f"{self.id}_value")
            yield Label(self.label, classes="stat-label", id=f"{self.id}_label")

    def update_stat(self, value: str, label: Optional[str] = None) -> None:
        """Update the statistic value and label.

        Args:
            value: New statistic value
            label: New statistic label (optional)
        """
        self.value = value
        if label is not None:
            self.label = label

        # Update the display
        value_label = self.query_one(f"#{self.id}_value", Label)
        label_label = self.query_one(f"#{self.id}_label", Label)

        value_label.update(self.value)
        label_label.update(self.label)


class ChartWidget(DashboardWidget):
    """Widget for displaying charts."""

    def __init__(
        self,
        title: str,
        chart_type: str = "line",
        id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the chart widget.

        Args:
            title: Widget title
            chart_type: Type of chart (line, bar, etc.)
            id: Widget ID
            **kwargs: Additional widget parameters
        """
        super().__init__(title=title, id=id, **kwargs)
        self.chart_type = chart_type
        self.chart_data = kwargs.get("data", {})

    def compose_content(self) -> ComposeResult:
        """Compose the widget content."""
        yield Container(classes="chart-container", id=f"{self.id}_chart")

    def update_chart(self, data: Dict[str, Any]) -> None:
        """Update the chart data.

        Args:
            data: New chart data
        """
        self.chart_data = data
        chart_container = self.query_one(f"#{self.id}_chart", Container)

        # In a real implementation, this would update the chart
        # For now, we'll just update a text representation
        chart_text = f"[Chart: {self.chart_type}]\n"
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 5:
                chart_text += f"{key}: [{value[0]}, {value[1]}, ... {value[-1]}]\n"
            else:
                chart_text += f"{key}: {value}\n"

        chart_container.update(chart_text)


class TableWidget(DashboardWidget):
    """Widget for displaying tabular data."""

    def __init__(
        self,
        title: str,
        columns: List[str],
        rows: List[List[Any]] = None,
        id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the table widget.

        Args:
            title: Widget title
            columns: Table column names
            rows: Table rows (optional)
            id: Widget ID
            **kwargs: Additional widget parameters
        """
        super().__init__(title=title, id=id, **kwargs)
        self.columns = columns
        self.rows = rows or []

    def compose_content(self) -> ComposeResult:
        """Compose the widget content."""
        table = DataTable(id=f"{self.id}_table")
        yield table

    def on_mount(self) -> None:
        """Handle mount event."""
        super().on_mount()

        # Initialize the table
        table = self.query_one(f"#{self.id}_table", DataTable)
        table.add_columns(*self.columns)

        for row in self.rows:
            table.add_row(*row)

    def update_table(self, rows: List[List[Any]]) -> None:
        """Update the table rows.

        Args:
            rows: New table rows
        """
        self.rows = rows
        table = self.query_one(f"#{self.id}_table", DataTable)

        # Clear and repopulate the table
        table.clear()
        table.add_columns(*self.columns)

        for row in self.rows:
            table.add_row(*row)


class ProgressWidget(DashboardWidget):
    """Widget for displaying progress."""

    progress = reactive(0.0)
    status = reactive("Ready")

    def __init__(
        self,
        title: str,
        progress: float = 0.0,
        status: str = "Ready",
        id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the progress widget.

        Args:
            title: Widget title
            progress: Initial progress value (0.0-1.0)
            status: Status text
            id: Widget ID
            **kwargs: Additional widget parameters
        """
        super().__init__(title=title, id=id, **kwargs)
        self.progress = progress
        self.status = status

    def compose_content(self) -> ComposeResult:
        """Compose the widget content."""
        yield ProgressBar(value=self.progress, id=f"{self.id}_bar")
        yield Label(self.status, id=f"{self.id}_status")

    def update_progress(self, progress: float, status: Optional[str] = None) -> None:
        """Update the progress value and status.

        Args:
            progress: New progress value (0.0-1.0)
            status: New status text (optional)
        """
        self.progress = progress
        if status is not None:
            self.status = status

        # Update the display
        progress_bar = self.query_one(f"#{self.id}_bar", ProgressBar)
        status_label = self.query_one(f"#{self.id}_status", Label)

        progress_bar.update(progress=self.progress)
        status_label.update(self.status)


class ActionWidget(DashboardWidget):
    """Widget for displaying action buttons."""

    def __init__(
        self,
        title: str,
        actions: List[Dict[str, Any]],
        id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the action widget.

        Args:
            title: Widget title
            actions: List of action definitions
            id: Widget ID
            **kwargs: Additional widget parameters
        """
        super().__init__(title=title, id=id, **kwargs)
        self.actions = actions
        self.action_handlers = {}

    def compose_content(self) -> ComposeResult:
        """Compose the widget content."""
        with Grid(id=f"{self.id}_actions", classes="action-grid"):
            for action in self.actions:
                yield Button(
                    action["label"],
                    id=f"{self.id}_{action['id']}",
                    variant=action.get("variant", "default"),
                )

    def register_action_handler(self, action_id: str, handler: Callable) -> None:
        """Register a handler for an action button.

        Args:
            action_id: Action ID
            handler: Handler function
        """
        self.action_handlers[action_id] = handler

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        button_id = event.button.id

        # Extract action ID from button ID
        if button_id.startswith(f"{self.id}_"):
            action_id = button_id[len(f"{self.id}_") :]

            # Call the handler if registered
            if action_id in self.action_handlers:
                self.action_handlers[action_id]()


class Visualization3DWidget(DashboardWidget):
    """Widget for 3D visualization."""

    def __init__(
        self,
        title: str,
        model_data: Dict[str, Any] = None,
        id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the 3D visualization widget.

        Args:
            title: Widget title
            model_data: 3D model data
            id: Widget ID
            **kwargs: Additional widget parameters
        """
        super().__init__(title=title, id=id, **kwargs)
        self.model_data = model_data or {}
        self.rotation = 0

    def compose_content(self) -> ComposeResult:
        """Compose the widget content."""
        yield Container(classes="visualization-3d", id=f"{self.id}_container")
        with Horizontal():
            yield Button("Rotate", id=f"{self.id}_rotate")
            yield Button("Reset", id=f"{self.id}_reset")

    def on_mount(self) -> None:
        """Handle mount event."""
        super().on_mount()
        self.update_visualization()

    def update_visualization(self) -> None:
        """Update the visualization."""
        container = self.query_one(f"#{self.id}_container", Container)

        # In a real implementation, this would render a 3D model
        # For now, we'll just display a text representation
        model_type = self.model_data.get("type", "airfoil")
        model_name = self.model_data.get("name", "Unknown")

        if model_type == "airfoil":
            visualization = f"""
            ╭──────────────────────────────────────╮
            │                                      │
            │                                      │
            │      ╭─────────────────────╮         │
            │      │                     │         │
            │      │  NACA {model_name.ljust(10)}  │         │
            │      │                     │         │
            │      ╰─────────────────────╯         │
            │                                      │
            │      Rotation: {self.rotation}°                │
            │                                      │
            ╰──────────────────────────────────────╯
            """
        elif model_type == "airplane":
            visualization = f"""
            ╭──────────────────────────────────────╮
            │                                      │
            │              ╱|╲                    │
            │             ╱ | ╲                   │
            │      ╭─────────────────╮            │
            │      │  {model_name.ljust(10)}  │            │
            │      ╰─────────────────╯            │
            │        ╱|╲     ╱|╲                  │
            │                                      │
            │      Rotation: {self.rotation}°                │
            │                                      │
            ╰──────────────────────────────────────╯
            """
        else:
            visualization = f"""
            ╭──────────────────────────────────────╮
            │                                      │
            │      3D Model: {model_type}                  │
            │      Name: {model_name}                      │
            │                                      │
            │      Rotation: {self.rotation}°                │
            │                                      │
            │      [No preview available]          │
            │                                      │
            ╰──────────────────────────────────────╯
            """

        container.update(visualization)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        button_id = event.button.id

        if button_id == f"{self.id}_rotate":
            self.rotation = (self.rotation + 45) % 360
            self.update_visualization()
        elif button_id == f"{self.id}_reset":
            self.rotation = 0
            self.update_visualization()

    def update_model(self, model_data: Dict[str, Any]) -> None:
        """Update the 3D model data.

        Args:
            model_data: New model data
        """
        self.model_data = model_data
        self.update_visualization()


class Dashboard(Container):
    """Main dashboard container with customizable layout."""

    def __init__(
        self,
        widgets: List[Widget] = None,
        layout: str = "grid",
        id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the dashboard.

        Args:
            widgets: List of dashboard widgets
            layout: Layout type (grid, horizontal, vertical)
            id: Dashboard ID
            **kwargs: Additional dashboard parameters
        """
        super().__init__(id=id or "dashboard", **kwargs)
        self.widgets = widgets or []
        self.layout = layout

    def compose(self) -> ComposeResult:
        """Compose the dashboard."""
        if self.layout == "grid":
            with Grid(id="dashboard_grid", classes="dashboard-grid"):
                yield from self.widgets
        elif self.layout == "horizontal":
            with Horizontal(id="dashboard_horizontal", classes="dashboard-horizontal"):
                yield from self.widgets
        else:  # vertical
            with Vertical(id="dashboard_vertical", classes="dashboard-vertical"):
                yield from self.widgets

    def add_widget(self, widget: Widget) -> None:
        """Add a widget to the dashboard.

        Args:
            widget: Widget to add
        """
        self.widgets.append(widget)
        self.mount(widget)

    def remove_widget(self, widget_id: str) -> None:
        """Remove a widget from the dashboard.

        Args:
            widget_id: ID of the widget to remove
        """
        for i, widget in enumerate(self.widgets):
            if widget.id == widget_id:
                self.widgets.pop(i)
                widget.remove()
                break

    def get_widget(self, widget_id: str) -> Optional[Widget]:
        """Get a widget by ID.

        Args:
            widget_id: Widget ID

        Returns:
            Widget if found, None otherwise
        """
        for widget in self.widgets:
            if widget.id == widget_id:
                return widget
        return None
