"""Progress Widget for ICARUS TUI

This widget displays progress information using the core progress management system.
"""

from core.tui_integration import TUIEvent
from core.tui_integration import TUIEventType
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Label
from textual.widgets import ProgressBar


class ProgressWidget(Container):
    """Widget for displaying progress information."""

    current_progress = reactive(0.0)
    current_status = reactive("Ready")
    is_active = reactive(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.active_tasks = {}

    def compose(self):
        """Compose the progress display."""
        yield Label("Progress", classes="title")
        yield ProgressBar(id="main_progress")
        yield Label("Ready", id="status_label")

    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        self.update_display()

    def _on_analysis_started(self, event: TUIEvent) -> None:
        """Handle analysis started events."""
        if event.type == TUIEventType.ANALYSIS_STARTED:
            self.current_status = "Analysis started"
            self.current_progress = 0.0
            self.is_active = True
            self.update_display()

    def _on_workflow_started(self, event: TUIEvent) -> None:
        """Handle workflow started events."""
        if event.type == TUIEventType.WORKFLOW_STARTED:
            self.current_status = (
                f"Workflow started: {event.data.get('workflow', 'Unknown')}"
            )
            self.current_progress = 0.0
            self.is_active = True
            self.update_display()

    def update_progress(self, progress: float, status: str) -> None:
        """Update progress and status."""
        self.current_progress = max(0.0, min(100.0, progress))
        self.current_status = status
        self.is_active = progress > 0.0 and progress < 100.0
        self.update_display()

    def update_display(self) -> None:
        """Update the progress display."""
        progress_bar = self.query_one("#main_progress", ProgressBar)
        status_label = self.query_one("#status_label", Label)

        progress_bar.progress = self.current_progress
        status_label.update(self.current_status)

        if self.is_active:
            progress_bar.styles.background = "blue"
        else:
            progress_bar.styles.background = (
                "green" if self.current_progress >= 100.0 else "gray"
            )

    def start_task(self, task_id: str, description: str) -> None:
        """Start a new task."""
        self.active_tasks[task_id] = {"description": description, "progress": 0.0}
        self.current_status = f"Started: {description}"
        self.is_active = True
        self.update_display()

    def update_task(
        self,
        task_id: str,
        progress: float,
        description: str = None,
    ) -> None:
        """Update a task's progress."""
        if task_id in self.active_tasks:
            self.active_tasks[task_id]["progress"] = progress
            if description:
                self.active_tasks[task_id]["description"] = description

        # Calculate overall progress
        if self.active_tasks:
            total_progress = sum(
                task["progress"] for task in self.active_tasks.values()
            )
            self.current_progress = total_progress / len(self.active_tasks)

            # Use the most recent task description
            current_task = list(self.active_tasks.values())[-1]
            self.current_status = current_task["description"]
        else:
            self.current_progress = 0.0
            self.current_status = "Ready"

        self.update_display()

    def complete_task(self, task_id: str) -> None:
        """Complete a task."""
        if task_id in self.active_tasks:
            self.active_tasks[task_id]["progress"] = 100.0
            self.update_task(task_id, 100.0)

        # Check if all tasks are complete
        if all(task["progress"] >= 100.0 for task in self.active_tasks.values()):
            self.is_active = False
            self.current_status = "All tasks completed"
            self.update_display()

    def reset(self) -> None:
        """Reset the progress widget."""
        self.current_progress = 0.0
        self.current_status = "Ready"
        self.is_active = False
        self.active_tasks = {}
        self.update_display()
