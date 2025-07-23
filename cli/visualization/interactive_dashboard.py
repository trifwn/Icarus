"""Interactive Dashboard - Customizable dashboard with real-time updates

This module provides an interactive dashboard for the ICARUS CLI with real-time
updates, customizable widgets, and advanced visualization capabilities.
"""

import asyncio

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Button
from textual.widgets import Label

from .dashboard import ActionWidget
from .dashboard import ChartWidget
from .dashboard import Dashboard
from .dashboard import ProgressWidget
from .dashboard import StatWidget
from .dashboard import TableWidget
from .dashboard import Visualization3DWidget
from .renderer_3d import Renderer3D


class DashboardScreen(Screen):
    """Interactive dashboard screen with customizable widgets."""

    BINDINGS = [
        ("d", "toggle_dark", "Toggle Dark Mode"),
        ("a", "add_widget", "Add Widget"),
        ("r", "refresh_dashboard", "Refresh"),
        ("c", "customize_layout", "Customize"),
    ]

    def __init__(self, **kwargs):
        """Initialize the dashboard screen."""
        super().__init__(**kwargs)
        self.dashboard = None
        self.renderer_3d = Renderer3D()

        # Initialize 3D models
        self.renderer_3d.create_airfoil("naca0012", "0012")
        self.renderer_3d.create_airfoil("naca2412", "2412")
        self.renderer_3d.create_airplane("boeing737", 35.8, 39.5)

    def compose(self) -> ComposeResult:
        """Compose the dashboard screen."""
        yield Label("ICARUS Interactive Dashboard", id="dashboard_title")

        # Create dashboard with widgets
        self.dashboard = Dashboard(
            widgets=[
                StatWidget(
                    title="Active Session",
                    value="1",
                    label="Current Sessions",
                    id="session_stat",
                    auto_update=True,
                    update_interval=5.0,
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
        yield self.dashboard

        with Horizontal(id="dashboard_controls"):
            yield Button("Add Widget", id="add_widget_btn", variant="primary")
            yield Button("Change Layout", id="change_layout_btn")
            yield Button("Refresh Data", id="refresh_data_btn")
            yield Button("Export Dashboard", id="export_dashboard_btn")

    def on_mount(self) -> None:
        """Handle mount event."""
        # Register action handlers
        actions_widget = self.dashboard.get_widget("quick_actions")
        if actions_widget:
            actions_widget.register_action_handler(
                "new_analysis",
                self.action_new_analysis,
            )
            actions_widget.register_action_handler(
                "view_results",
                self.action_view_results,
            )
            actions_widget.register_action_handler(
                "export_data",
                self.action_export_data,
            )

        # Start auto-updates for widgets
        asyncio.create_task(self.update_dashboard_data())

    async def update_dashboard_data(self) -> None:
        """Update dashboard data periodically."""
        try:
            while True:
                await asyncio.sleep(3.0)

                # Update performance chart
                chart_widget = self.dashboard.get_widget("performance_chart")
                if chart_widget:
                    import random

                    new_data = {
                        "x": list(range(6)),
                        "y": [random.randint(0, 5) for _ in range(6)],
                        "xlabel": "Time (min)",
                        "ylabel": "CPU Usage (%)",
                    }
                    chart_widget.update_chart(new_data)

                # Update progress
                progress_widget = self.dashboard.get_widget("analysis_progress")
                if progress_widget:
                    import random

                    new_progress = min(
                        1.0,
                        progress_widget.progress + random.uniform(-0.1, 0.1),
                    )
                    if new_progress < 0:
                        new_progress = 0.0
                    progress_widget.update_progress(new_progress)

                # Rotate 3D model
                model_widget = self.dashboard.get_widget("model_viewer")
                if model_widget:
                    model_widget.rotation = (model_widget.rotation + 15) % 360
                    model_widget.update_visualization()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.log.error(f"Error updating dashboard: {e}")

    def action_new_analysis(self) -> None:
        """Handle new analysis action."""
        self.notify("Starting new analysis...", title="Action")

    def action_view_results(self) -> None:
        """Handle view results action."""
        self.notify("Opening results viewer...", title="Action")

    def action_export_data(self) -> None:
        """Handle export data action."""
        self.notify("Exporting data...", title="Action")

    def action_toggle_dark(self) -> None:
        """Toggle dark mode."""
        # This would be implemented with a theme switch
        self.notify("Toggling dark mode", title="Theme")

    def action_add_widget(self) -> None:
        """Add a new widget to the dashboard."""
        # In a real implementation, this would show a widget selection dialog
        import random

        widget_id = f"new_widget_{random.randint(1000, 9999)}"

        new_widget = StatWidget(
            title="New Widget",
            value=str(random.randint(10, 100)),
            label="Random Value",
            id=widget_id,
        )

        self.dashboard.add_widget(new_widget)
        self.notify(f"Added new widget: {widget_id}", title="Dashboard")

    def action_refresh_dashboard(self) -> None:
        """Refresh all dashboard widgets."""
        self.notify("Refreshing dashboard data...", title="Dashboard")

        # Update stats
        import random

        session_stat = self.dashboard.get_widget("session_stat")
        if session_stat:
            session_stat.update_stat(str(random.randint(1, 5)))

        jobs_stat = self.dashboard.get_widget("jobs_stat")
        if jobs_stat:
            jobs_stat.update_stat(str(random.randint(0, 10)))

    def action_customize_layout(self) -> None:
        """Customize dashboard layout."""
        # In a real implementation, this would show a layout customization dialog
        layouts = ["grid", "horizontal", "vertical"]
        import random

        new_layout = random.choice(layouts)

        # Store current widgets
        widgets = self.dashboard.widgets

        # Remove current dashboard
        self.dashboard.remove()

        # Create new dashboard with different layout
        self.dashboard = Dashboard(
            widgets=widgets,
            layout=new_layout,
            id="main_dashboard",
        )

        # Mount new dashboard
        self.mount(self.dashboard)

        self.notify(f"Changed layout to: {new_layout}", title="Dashboard")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        button_id = event.button.id

        if button_id == "add_widget_btn":
            self.action_add_widget()
        elif button_id == "change_layout_btn":
            self.action_customize_layout()
        elif button_id == "refresh_data_btn":
            self.action_refresh_dashboard()
        elif button_id == "export_dashboard_btn":
            self.notify("Exporting dashboard configuration...", title="Dashboard")


# Create a standalone app for testing
if __name__ == "__main__":
    from textual.app import App

    class DashboardApp(App):
        """Standalone dashboard app for testing."""

        CSS_PATH = "../tui_styles_aerospace.css"

        def on_mount(self) -> None:
            """Handle mount event."""
            self.push_screen(DashboardScreen())

    app = DashboardApp()
    app.run()
