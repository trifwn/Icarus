"""Workflow Engine and Template Manager for ICARUS CLI

This module provides workflow automation, template management, and batch processing
capabilities for streamlined analysis workflows.
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .ui import theme_manager, notification_system, ui_components

console = Console()


class WorkflowType(Enum):
    """Types of workflows."""

    AIRFOIL_ANALYSIS = "airfoil_analysis"
    AIRPLANE_ANALYSIS = "airplane_analysis"
    BATCH_PROCESSING = "batch_processing"
    VISUALIZATION = "visualization"
    CUSTOM = "custom"


@dataclass
class WorkflowStep:
    """Represents a step in a workflow."""

    name: str
    description: str
    action: str
    parameters: Dict[str, Any]
    required: bool = True
    skip_condition: Optional[str] = None
    validation: Optional[Dict[str, Any]] = None


@dataclass
class WorkflowTemplate:
    """Represents a workflow template."""

    name: str
    description: str
    type: WorkflowType
    steps: List[WorkflowStep]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]


class WorkflowEngine:
    """Engine for executing workflows and managing workflow state."""

    def __init__(self):
        self.workflows: Dict[str, WorkflowTemplate] = {}
        self.current_workflow: Optional[str] = None
        self.workflow_state: Dict[str, Any] = {}
        self.step_results: Dict[str, Any] = {}
        self._load_builtin_workflows()

    def _load_builtin_workflows(self):
        """Load built-in workflow templates."""
        self._add_airfoil_analysis_workflow()
        self._add_airplane_analysis_workflow()
        self._add_batch_processing_workflow()

    def _add_airfoil_analysis_workflow(self):
        """Add built-in airfoil analysis workflow."""
        workflow = WorkflowTemplate(
            name="Standard Airfoil Analysis",
            description="Complete airfoil analysis with multiple solvers",
            type=WorkflowType.AIRFOIL_ANALYSIS,
            steps=[
                WorkflowStep(
                    name="airfoil_selection",
                    description="Select airfoil(s) for analysis",
                    action="select_airfoils",
                    parameters={"source": "database", "count": 1},
                ),
                WorkflowStep(
                    name="solver_selection",
                    description="Choose analysis solvers",
                    action="select_solvers",
                    parameters={"solvers": ["xfoil", "foil2wake"]},
                ),
                WorkflowStep(
                    name="parameter_configuration",
                    description="Configure analysis parameters",
                    action="configure_parameters",
                    parameters={"angles": "0:15:16", "reynolds": 1e6},
                ),
                WorkflowStep(
                    name="analysis_execution", description="Execute analysis", action="run_analysis", parameters={}
                ),
                WorkflowStep(
                    name="results_saving",
                    description="Save results to database",
                    action="save_results",
                    parameters={"auto_save": True},
                ),
            ],
            parameters={"default_solvers": ["xfoil"], "default_angles": "0:15:16", "default_reynolds": 1e6},
            metadata={"version": "1.0", "author": "ICARUS"},
        )
        self.workflows[workflow.name] = workflow

    def _add_airplane_analysis_workflow(self):
        """Add built-in airplane analysis workflow."""
        workflow = WorkflowTemplate(
            name="Standard Airplane Analysis",
            description="Complete airplane analysis with 3D solvers",
            type=WorkflowType.AIRPLANE_ANALYSIS,
            steps=[
                WorkflowStep(
                    name="airplane_selection",
                    description="Select airplane for analysis",
                    action="select_airplane",
                    parameters={"source": "database"},
                ),
                WorkflowStep(
                    name="solver_selection",
                    description="Choose 3D analysis solvers",
                    action="select_3d_solvers",
                    parameters={"solvers": ["avl", "gnvp3"]},
                ),
                WorkflowStep(
                    name="flight_state_configuration",
                    description="Configure flight state",
                    action="configure_flight_state",
                    parameters={"environment": "isa", "altitude": 10000},
                ),
                WorkflowStep(
                    name="analysis_execution",
                    description="Execute 3D analysis",
                    action="run_3d_analysis",
                    parameters={},
                ),
                WorkflowStep(
                    name="results_saving",
                    description="Save results to database",
                    action="save_results",
                    parameters={"auto_save": True},
                ),
            ],
            parameters={"default_solvers": ["avl"], "default_environment": "isa", "default_altitude": 10000},
            metadata={"version": "1.0", "author": "ICARUS"},
        )
        self.workflows[workflow.name] = workflow

    def _add_batch_processing_workflow(self):
        """Add built-in batch processing workflow."""
        workflow = WorkflowTemplate(
            name="Batch Airfoil Analysis",
            description="Analyze multiple airfoils with the same parameters",
            type=WorkflowType.BATCH_PROCESSING,
            steps=[
                WorkflowStep(
                    name="batch_selection",
                    description="Select airfoils for batch processing",
                    action="select_batch_airfoils",
                    parameters={"min_count": 1, "max_count": 10},
                ),
                WorkflowStep(
                    name="parameter_configuration",
                    description="Configure batch parameters",
                    action="configure_batch_parameters",
                    parameters={"angles": "0:15:16", "reynolds": 1e6},
                ),
                WorkflowStep(
                    name="batch_execution",
                    description="Execute batch analysis",
                    action="run_batch_analysis",
                    parameters={"parallel": True, "max_workers": 4},
                ),
                WorkflowStep(
                    name="results_aggregation",
                    description="Aggregate and save batch results",
                    action="aggregate_results",
                    parameters={"export_format": "json"},
                ),
            ],
            parameters={
                "default_angles": "0:15:16",
                "default_reynolds": 1e6,
                "parallel_processing": True,
                "max_workers": 4,
            },
            metadata={"version": "1.0", "author": "ICARUS"},
        )
        self.workflows[workflow.name] = workflow

    def get_workflows(self, workflow_type: Optional[WorkflowType] = None) -> List[WorkflowTemplate]:
        """Get available workflows, optionally filtered by type."""
        if workflow_type:
            return [w for w in self.workflows.values() if w.type == workflow_type]
        return list(self.workflows.values())

    def get_available_workflows(self) -> List[Dict[str, Any]]:
        """Get available workflows as dictionaries for TUI display."""
        workflows = []
        for workflow in self.workflows.values():
            workflows.append(
                {
                    "name": workflow.name,
                    "description": workflow.description,
                    "type": workflow.type.value,
                    "steps": len(workflow.steps),
                    "metadata": workflow.metadata,
                }
            )
        return workflows

    def get_workflow(self, name: str) -> Optional[WorkflowTemplate]:
        """Get a specific workflow by name."""
        return self.workflows.get(name)

    def start_workflow(self, workflow_name: str, parameters: Dict[str, Any] = None):
        """Start a workflow execution."""
        workflow = self.get_workflow(workflow_name)
        if not workflow:
            notification_system.error(f"Workflow '{workflow_name}' not found")
            return False

        self.current_workflow = workflow_name
        self.workflow_state = parameters or {}
        self.step_results = {}

        notification_system.info(f"Started workflow: {workflow.name}")
        return True

    def execute_step(self, step: WorkflowStep) -> bool:
        """Execute a single workflow step."""
        try:
            notification_system.info(f"Executing step: {step.name}")

            # Check skip condition
            if step.skip_condition and self._evaluate_condition(step.skip_condition):
                notification_system.info(f"Skipping step: {step.name}")
                return True

            # Execute step action
            result = self._execute_action(step.action, step.parameters)

            # Store result
            self.step_results[step.name] = result

            # Validate result if validation rules exist
            if step.validation and not self._validate_result(result, step.validation):
                notification_system.error(f"Step validation failed: {step.name}")
                return False

            notification_system.success(f"Completed step: {step.name}")
            return True

        except Exception as e:
            notification_system.error(f"Step execution failed: {step.name} - {e}")
            return False

    def _execute_action(self, action: str, parameters: Dict[str, Any]) -> Any:
        """Execute a workflow action."""
        # This would be connected to actual CLI functions
        action_handlers = {
            "select_airfoils": self._handle_select_airfoils,
            "select_solvers": self._handle_select_solvers,
            "configure_parameters": self._handle_configure_parameters,
            "run_analysis": self._handle_run_analysis,
            "save_results": self._handle_save_results,
            # Add more handlers as needed
        }

        handler = action_handlers.get(action)
        if handler:
            return handler(parameters)
        else:
            raise ValueError(f"Unknown action: {action}")

    def _handle_select_airfoils(self, parameters: Dict[str, Any]) -> List[str]:
        """Handle airfoil selection action."""
        # This would integrate with the actual airfoil selection logic
        return ["naca2412"]  # Placeholder

    def _handle_select_solvers(self, parameters: Dict[str, Any]) -> List[str]:
        """Handle solver selection action."""
        return parameters.get("solvers", ["xfoil"])

    def _handle_configure_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle parameter configuration action."""
        return parameters

    def _handle_run_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle analysis execution action."""
        # This would integrate with actual analysis execution
        return {"status": "completed", "results": {}}

    def _handle_save_results(self, parameters: Dict[str, Any]) -> bool:
        """Handle results saving action."""
        return True

    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a skip condition."""
        # Simple condition evaluation - could be enhanced
        return False

    def _validate_result(self, result: Any, validation: Dict[str, Any]) -> bool:
        """Validate a step result."""
        # Simple validation - could be enhanced
        return True

    def get_workflow_progress(self) -> Dict[str, Any]:
        """Get current workflow progress."""
        if not self.current_workflow:
            return {}

        workflow = self.get_workflow(self.current_workflow)
        if not workflow:
            return {}

        completed_steps = len(self.step_results)
        total_steps = len(workflow.steps)

        return {
            "workflow": self.current_workflow,
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "progress": (completed_steps / total_steps) * 100 if total_steps > 0 else 0,
            "current_step": workflow.steps[completed_steps].name if completed_steps < total_steps else None,
        }


class TemplateManager:
    """Manages workflow templates and custom template creation."""

    def __init__(self, templates_dir: str = "~/.icarus/templates"):
        self.templates_dir = Path(templates_dir).expanduser()
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.custom_templates: Dict[str, WorkflowTemplate] = {}
        self._load_custom_templates()

    def _load_custom_templates(self):
        """Load custom templates from disk."""
        for template_file in self.templates_dir.glob("*.json"):
            try:
                with open(template_file, "r") as f:
                    data = json.load(f)
                    template = self._dict_to_template(data)
                    self.custom_templates[template.name] = template
            except Exception as e:
                console.print(f"[yellow]Failed to load template {template_file}: {e}[/yellow]")

    def _dict_to_template(self, data: Dict[str, Any]) -> WorkflowTemplate:
        """Convert dictionary to WorkflowTemplate."""
        steps = [WorkflowStep(**step_data) for step_data in data.get("steps", [])]

        return WorkflowTemplate(
            name=data["name"],
            description=data["description"],
            type=WorkflowType(data["type"]),
            steps=steps,
            parameters=data.get("parameters", {}),
            metadata=data.get("metadata", {}),
        )

    def _template_to_dict(self, template: WorkflowTemplate) -> Dict[str, Any]:
        """Convert WorkflowTemplate to dictionary."""
        return {
            "name": template.name,
            "description": template.description,
            "type": template.type.value,
            "steps": [asdict(step) for step in template.steps],
            "parameters": template.parameters,
            "metadata": template.metadata,
        }

    def save_template(self, template: WorkflowTemplate):
        """Save a custom template to disk."""
        try:
            template_file = self.templates_dir / f"{template.name.lower().replace(' ', '_')}.json"
            with open(template_file, "w") as f:
                json.dump(self._template_to_dict(template), f, indent=2)

            self.custom_templates[template.name] = template
            notification_system.success(f"Template saved: {template.name}")
        except Exception as e:
            notification_system.error(f"Failed to save template: {e}")

    def create_template_from_workflow(
        self, workflow_name: str, template_name: str, description: str = None
    ) -> WorkflowTemplate:
        """Create a template from an existing workflow."""
        # This would capture the current workflow state and create a template
        # Implementation depends on how workflows are structured
        pass

    def get_all_templates(self) -> Dict[str, WorkflowTemplate]:
        """Get all available templates (built-in + custom)."""
        all_templates = {}
        # Add built-in templates from workflow engine
        # Add custom templates
        all_templates.update(self.custom_templates)
        return all_templates

    def delete_template(self, template_name: str):
        """Delete a custom template."""
        if template_name in self.custom_templates:
            template_file = self.templates_dir / f"{template_name.lower().replace(' ', '_')}.json"
            if template_file.exists():
                template_file.unlink()

            del self.custom_templates[template_name]
            notification_system.success(f"Template deleted: {template_name}")
        else:
            notification_system.error(f"Template not found: {template_name}")


# Global instances
workflow_engine = WorkflowEngine()
template_manager = TemplateManager()
