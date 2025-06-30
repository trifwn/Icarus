"""Workflow Widget for ICARUS TUI

This widget displays and manages workflows using the core workflow engine.
"""

from textual.widget import Tree
from textual.reactive import reactive
from typing import Dict, Any, List

from core.workflow import workflow_engine
from core.tui_integration import TUIEvent, TUIEventType


class WorkflowWidget(Tree):
    """Widget for displaying and managing workflows."""

    current_workflow = reactive("")
    workflow_progress = reactive(0.0)

    def __init__(self, **kwargs):
        super().__init__("Workflows", **kwargs)
        self.load_workflows()

    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        self.load_workflows()

    def load_workflows(self) -> None:
        """Load available workflows from the core workflow engine."""
        try:
            workflows = workflow_engine.get_workflows()

            # Clear existing tree
            self.clear()

            # Add workflows to tree
            for workflow in workflows:
                workflow_node = self.root.add(workflow.name)
                workflow_node.add(f"Type: {workflow.type.value}")
                workflow_node.add(f"Steps: {len(workflow.steps)}")
                workflow_node.add(f"Description: {workflow.description}")

                # Add steps
                steps_node = workflow_node.add("Steps")
                for step in workflow.steps:
                    steps_node.add(f"{step.name}: {step.description}")

        except Exception as e:
            self.root.add(f"Error loading workflows: {e}")

    def _on_workflow_event(self, event: TUIEvent) -> None:
        """Handle workflow events."""
        if event.type == TUIEventType.WORKFLOW_STARTED:
            self.current_workflow = event.data.get("workflow", "")
            self.workflow_progress = 0.0
        elif event.type == TUIEventType.WORKFLOW_COMPLETED:
            self.current_workflow = ""
            self.workflow_progress = 100.0

    def get_current_workflow_info(self) -> Dict[str, Any]:
        """Get information about the current workflow."""
        if not self.current_workflow:
            return {}

        try:
            workflow = workflow_engine.get_workflow(self.current_workflow)
            if workflow:
                return {
                    "name": workflow.name,
                    "type": workflow.type.value,
                    "steps": len(workflow.steps),
                    "progress": self.workflow_progress,
                }
        except Exception:
            pass

        return {}

    def execute_workflow(self, workflow_name: str) -> bool:
        """Execute a workflow."""
        try:
            return workflow_engine.start_workflow(workflow_name)
        except Exception:
            return False
