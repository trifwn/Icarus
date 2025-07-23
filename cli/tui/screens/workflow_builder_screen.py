"""Workflow Builder Screen for ICARUS CLI

This screen provides the main interface for the visual workflow builder,
allowing users to create, edit, and manage workflows through a drag-and-drop interface.
"""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Button
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Label

from ..widgets.base_widgets import AerospaceButton
from ..widgets.base_widgets import ButtonVariant
from ..widgets.base_widgets import NotificationPanel
from ..widgets.visual_workflow_builder import VisualWorkflowBuilder


class WorkflowBuilderScreen(Screen):
    """Screen for visual workflow building."""

    TITLE = "ICARUS - Visual Workflow Builder"

    BINDINGS = [
        Binding("ctrl+n", "new_workflow", "New Workflow"),
        Binding("ctrl+s", "save_workflow", "Save Workflow"),
        Binding("ctrl+o", "open_workflow", "Open Workflow"),
        Binding("ctrl+t", "test_workflow", "Test Workflow"),
        Binding("f1", "help", "Help"),
        Binding("escape", "back", "Back to Main Menu"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.workflow_builder = None

    def compose(self) -> ComposeResult:
        """Compose the workflow builder screen."""
        yield Header(show_clock=True)

        with Container(id="main_container", classes="workflow-builder-screen"):
            # Toolbar
            with Horizontal(id="toolbar", classes="toolbar"):
                yield AerospaceButton(
                    "New Workflow",
                    variant=ButtonVariant.SUCCESS,
                    id="new_workflow_btn",
                )
                yield AerospaceButton(
                    "Save Workflow",
                    variant=ButtonVariant.PRIMARY,
                    id="save_workflow_btn",
                )
                yield AerospaceButton(
                    "Load Template",
                    variant=ButtonVariant.INFO,
                    id="load_template_btn",
                )
                yield AerospaceButton(
                    "Validate",
                    variant=ButtonVariant.WARNING,
                    id="validate_workflow_btn",
                )
                yield AerospaceButton(
                    "Test Run",
                    variant=ButtonVariant.SECONDARY,
                    id="test_workflow_btn",
                )

            # Main workflow builder
            yield VisualWorkflowBuilder(id="workflow_builder", classes="main-builder")

            # Status bar
            with Horizontal(id="status_bar", classes="status-bar"):
                yield Label("Ready", id="status_text", classes="status-text")
                yield Label("Nodes: 0", id="node_count", classes="status-info")
                yield Label(
                    "Connections: 0",
                    id="connection_count",
                    classes="status-info",
                )

        yield Footer()

    def on_mount(self) -> None:
        """Initialize the screen when mounted."""
        self.workflow_builder = self.query_one(
            "#workflow_builder",
            VisualWorkflowBuilder,
        )
        self._update_status()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle toolbar button presses."""
        if event.button.id == "new_workflow_btn":
            self.action_new_workflow()
        elif event.button.id == "save_workflow_btn":
            self.action_save_workflow()
        elif event.button.id == "load_template_btn":
            self.action_open_workflow()
        elif event.button.id == "validate_workflow_btn":
            self.action_validate_workflow()
        elif event.button.id == "test_workflow_btn":
            self.action_test_workflow()

    def action_new_workflow(self) -> None:
        """Create a new workflow."""
        if self.workflow_builder:
            canvas = self.workflow_builder.query_one("WorkflowCanvas")
            canvas.clear_canvas()
            self._update_status("New workflow created")

    def action_save_workflow(self) -> None:
        """Save the current workflow."""
        if self.workflow_builder:
            # Trigger save in the workflow builder
            save_btn = self.workflow_builder.query_one("#save_template", Button)
            if save_btn:
                save_btn.press()
            self._update_status("Workflow saved")

    def action_open_workflow(self) -> None:
        """Open/load a workflow template."""
        if self.workflow_builder:
            # Trigger load in the workflow builder
            load_btn = self.workflow_builder.query_one("#load_template", Button)
            if load_btn:
                load_btn.press()
            self._update_status("Loading workflow template")

    def action_validate_workflow(self) -> None:
        """Validate the current workflow."""
        if self.workflow_builder:
            try:
                canvas = self.workflow_builder.query_one("WorkflowCanvas")
                errors = canvas.validate_workflow()

                if not errors:
                    self._update_status("Workflow validation passed", "success")
                else:
                    self._update_status(
                        f"Validation failed: {len(errors)} errors",
                        "error",
                    )

                    # Show errors in notification panel
                    notification_panel = self.workflow_builder.query_one(
                        NotificationPanel,
                    )
                    for error in errors:
                        notification_panel.add_notification(
                            error,
                            NotificationPanel.NotificationType.ERROR,
                        )
            except Exception as e:
                self._update_status(f"Validation error: {str(e)}", "error")

    def action_test_workflow(self) -> None:
        """Test the current workflow."""
        if self.workflow_builder:
            try:
                canvas = self.workflow_builder.query_one("WorkflowCanvas")

                # First validate
                errors = canvas.validate_workflow()
                if errors:
                    self._update_status(
                        "Cannot test: workflow has validation errors",
                        "error",
                    )
                    return

                # Simulate test execution
                self._update_status("Test execution completed", "success")

                # Show test result in notification panel
                notification_panel = self.workflow_builder.query_one(NotificationPanel)
                notification_panel.add_notification(
                    "Workflow test execution simulation completed successfully",
                    NotificationPanel.NotificationType.SUCCESS,
                )
            except Exception as e:
                self._update_status(f"Test error: {str(e)}", "error")

    def action_help(self) -> None:
        """Show help information."""
        if self.workflow_builder:
            notification_panel = self.workflow_builder.query_one(NotificationPanel)
            help_messages = [
                "Visual Workflow Builder Help:",
                "• Drag nodes from the palette to create workflow steps",
                "• Click nodes to select and edit their properties",
                "• Connect nodes by dragging between ports",
                "• Use Ctrl+N for new workflow, Ctrl+S to save",
                "• Validate workflows before testing or execution",
                "• Templates can be saved and shared with team members",
            ]

            for message in help_messages:
                notification_panel.add_notification(
                    message,
                    NotificationPanel.NotificationType.INFO,
                )

    def action_back(self) -> None:
        """Return to the main menu."""
        self.app.pop_screen()

    def _update_status(self, message: str = "Ready", status_type: str = "info") -> None:
        """Update the status bar."""
        status_text = self.query_one("#status_text", Label)
        status_text.update(message)

        # Update status styling
        status_text.remove_class("status-success")
        status_text.remove_class("status-error")
        status_text.remove_class("status-warning")

        if status_type == "success":
            status_text.add_class("status-success")
        elif status_type == "error":
            status_text.add_class("status-error")
        elif status_type == "warning":
            status_text.add_class("status-warning")

        # Update node and connection counts
        if self.workflow_builder:
            try:
                canvas = self.workflow_builder.query_one("WorkflowCanvas")

                node_count = self.query_one("#node_count", Label)
                connection_count = self.query_one("#connection_count", Label)

                node_count.update(f"Nodes: {len(canvas.nodes)}")
                connection_count.update(f"Connections: {len(canvas.connections)}")
            except:
                pass

    def on_visual_workflow_builder_workflow_canvas_node_selected(self, event) -> None:
        """Handle node selection events."""
        self._update_status(f"Selected node: {event.node.name}")

    def on_visual_workflow_builder_workflow_canvas_connection_created(
        self,
        event,
    ) -> None:
        """Handle connection creation events."""
        self._update_status("Connection created")
        self._update_node_counts()

    def _update_node_counts(self) -> None:
        """Update node and connection counts in status bar."""
        if self.workflow_builder:
            try:
                canvas = self.workflow_builder.query_one("WorkflowCanvas")

                node_count = self.query_one("#node_count", Label)
                connection_count = self.query_one("#connection_count", Label)

                node_count.update(f"Nodes: {len(canvas.nodes)}")
                connection_count.update(f"Connections: {len(canvas.connections)}")
            except:
                pass
