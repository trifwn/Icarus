"""Workflow Engine and Template Manager for ICARUS CLI

This module provides workflow automation, template management, and batch processing
capabilities for streamlined analysis workflows.
"""

import asyncio
import json
import logging
import traceback
import uuid
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from rich.console import Console

from .ui import notification_system

console = Console()
logger = logging.getLogger(__name__)


class WorkflowType(Enum):
    """Types of workflows."""

    AIRFOIL_ANALYSIS = "airfoil_analysis"
    AIRPLANE_ANALYSIS = "airplane_analysis"
    BATCH_PROCESSING = "batch_processing"
    VISUALIZATION = "visualization"
    CUSTOM = "custom"


class WorkflowStatus(Enum):
    """Status of workflow execution."""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(Enum):
    """Status of individual workflow steps."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class WorkflowStep:
    """Represents a step in a workflow."""

    id: str
    name: str
    description: str
    action: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    required: bool = True
    skip_condition: Optional[str] = None
    validation: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[int] = None
    status: StepStatus = StepStatus.PENDING
    error_message: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Any] = None


@dataclass
class WorkflowExecution:
    """Represents a workflow execution instance."""

    id: str
    workflow_name: str
    status: WorkflowStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    current_step: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    step_results: Dict[str, Any] = field(default_factory=dict)
    error_log: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowTemplate:
    """Represents a workflow template."""

    id: str
    name: str
    description: str
    type: WorkflowType
    steps: List[WorkflowStep]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class WorkflowDefinition:
    """Complete workflow definition with storage and versioning."""

    template: WorkflowTemplate
    executions: List[WorkflowExecution] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    is_active: bool = True


class WorkflowStorage:
    """Handles workflow definition storage and versioning."""

    def __init__(self, storage_dir: str = "~/.icarus/workflows"):
        self.storage_dir = Path(storage_dir).expanduser()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.definitions_file = self.storage_dir / "definitions.json"
        self.executions_dir = self.storage_dir / "executions"
        self.executions_dir.mkdir(exist_ok=True)

        self.definitions: Dict[str, WorkflowDefinition] = {}
        self._load_definitions()

    def _load_definitions(self):
        """Load workflow definitions from storage."""
        try:
            if self.definitions_file.exists():
                with open(self.definitions_file) as f:
                    data = json.load(f)
                    for def_data in data.values():
                        definition = self._dict_to_definition(def_data)
                        self.definitions[definition.template.id] = definition
        except Exception as e:
            logger.error(f"Failed to load workflow definitions: {e}")

    def _save_definitions(self):
        """Save workflow definitions to storage."""
        try:
            data = {}
            for def_id, definition in self.definitions.items():
                data[def_id] = self._definition_to_dict(definition)

            with open(self.definitions_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save workflow definitions: {e}")

    def _dict_to_definition(self, data: Dict[str, Any]) -> WorkflowDefinition:
        """Convert dictionary to WorkflowDefinition."""
        template_data = data["template"]
        steps = []
        for step_data in template_data.get("steps", []):
            # Create step without status first
            step_dict = step_data.copy()
            if "status" in step_dict:
                del step_dict["status"]
            step = WorkflowStep(**step_dict)
            # Set status separately if it exists
            if "status" in step_data:
                try:
                    step.status = StepStatus(step_data["status"])
                except ValueError:
                    step.status = StepStatus.PENDING
            steps.append(step)

        template = WorkflowTemplate(
            id=template_data["id"],
            name=template_data["name"],
            description=template_data["description"],
            type=WorkflowType(template_data["type"]),
            steps=steps,
            parameters=template_data.get("parameters", {}),
            metadata=template_data.get("metadata", {}),
            version=template_data.get("version", "1.0"),
            created_at=datetime.fromisoformat(
                template_data.get("created_at", datetime.now().isoformat()),
            ),
            updated_at=datetime.fromisoformat(
                template_data.get("updated_at", datetime.now().isoformat()),
            ),
        )

        executions = []
        for exec_data in data.get("executions", []):
            execution = WorkflowExecution(
                id=exec_data["id"],
                workflow_name=exec_data["workflow_name"],
                status=WorkflowStatus(exec_data["status"]),
                start_time=datetime.fromisoformat(exec_data["start_time"]),
                end_time=datetime.fromisoformat(exec_data["end_time"])
                if exec_data.get("end_time")
                else None,
                current_step=exec_data.get("current_step"),
                completed_steps=exec_data.get("completed_steps", []),
                failed_steps=exec_data.get("failed_steps", []),
                step_results=exec_data.get("step_results", {}),
                error_log=exec_data.get("error_log", []),
                parameters=exec_data.get("parameters", {}),
                metadata=exec_data.get("metadata", {}),
            )
            executions.append(execution)

        return WorkflowDefinition(
            template=template,
            executions=executions,
            tags=data.get("tags", []),
            is_active=data.get("is_active", True),
        )

    def _definition_to_dict(self, definition: WorkflowDefinition) -> Dict[str, Any]:
        """Convert WorkflowDefinition to dictionary."""
        return {
            "template": {
                "id": definition.template.id,
                "name": definition.template.name,
                "description": definition.template.description,
                "type": definition.template.type.value,
                "steps": [asdict(step) for step in definition.template.steps],
                "parameters": definition.template.parameters,
                "metadata": definition.template.metadata,
                "version": definition.template.version,
                "created_at": definition.template.created_at.isoformat(),
                "updated_at": definition.template.updated_at.isoformat(),
            },
            "executions": [asdict(execution) for execution in definition.executions],
            "tags": definition.tags,
            "is_active": definition.is_active,
        }

    def save_definition(self, definition: WorkflowDefinition):
        """Save a workflow definition."""
        self.definitions[definition.template.id] = definition
        self._save_definitions()

    def get_definition(self, template_id: str) -> Optional[WorkflowDefinition]:
        """Get a workflow definition by template ID."""
        return self.definitions.get(template_id)

    def get_all_definitions(self) -> List[WorkflowDefinition]:
        """Get all workflow definitions."""
        return list(self.definitions.values())

    def delete_definition(self, template_id: str):
        """Delete a workflow definition."""
        if template_id in self.definitions:
            del self.definitions[template_id]
            self._save_definitions()


class WorkflowEngine:
    """Enhanced workflow engine with step-by-step processing, progress tracking, and error recovery."""

    def __init__(self, storage_dir: str = "~/.icarus/workflows"):
        self.storage = WorkflowStorage(storage_dir)
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.action_handlers: Dict[str, Callable] = {}
        self.progress_callbacks: List[Callable] = []

        # Initialize built-in workflows
        self._initialize_builtin_workflows()
        self._register_default_handlers()

    def _initialize_builtin_workflows(self):
        """Initialize built-in workflow templates."""
        builtin_workflows = [
            self._create_airfoil_analysis_workflow(),
            self._create_airplane_analysis_workflow(),
            self._create_batch_processing_workflow(),
        ]

        for template in builtin_workflows:
            definition = WorkflowDefinition(template=template)
            self.storage.save_definition(definition)

    def _create_airfoil_analysis_workflow(self) -> WorkflowTemplate:
        """Create built-in airfoil analysis workflow."""
        steps = [
            WorkflowStep(
                id="step_1",
                name="Airfoil Selection",
                description="Select airfoil(s) for analysis",
                action="select_airfoils",
                parameters={"source": "database", "count": 1},
            ),
            WorkflowStep(
                id="step_2",
                name="Solver Selection",
                description="Choose analysis solvers",
                action="select_solvers",
                parameters={"solvers": ["xfoil", "foil2wake"]},
                dependencies=["step_1"],
            ),
            WorkflowStep(
                id="step_3",
                name="Parameter Configuration",
                description="Configure analysis parameters",
                action="configure_parameters",
                parameters={"angles": "0:15:16", "reynolds": 1e6},
                dependencies=["step_2"],
            ),
            WorkflowStep(
                id="step_4",
                name="Analysis Execution",
                description="Execute analysis",
                action="run_analysis",
                parameters={},
                dependencies=["step_3"],
                timeout=300,
            ),
            WorkflowStep(
                id="step_5",
                name="Results Saving",
                description="Save results to database",
                action="save_results",
                parameters={"auto_save": True},
                dependencies=["step_4"],
            ),
        ]

        return WorkflowTemplate(
            id=str(uuid.uuid4()),
            name="Standard Airfoil Analysis",
            description="Complete airfoil analysis with multiple solvers",
            type=WorkflowType.AIRFOIL_ANALYSIS,
            steps=steps,
            parameters={
                "default_solvers": ["xfoil"],
                "default_angles": "0:15:16",
                "default_reynolds": 1e6,
            },
            metadata={"version": "1.0", "author": "ICARUS", "builtin": True},
        )

    def _create_airplane_analysis_workflow(self) -> WorkflowTemplate:
        """Create built-in airplane analysis workflow."""
        steps = [
            WorkflowStep(
                id="step_1",
                name="Airplane Selection",
                description="Select airplane for analysis",
                action="select_airplane",
                parameters={"source": "database"},
            ),
            WorkflowStep(
                id="step_2",
                name="Solver Selection",
                description="Choose 3D analysis solvers",
                action="select_3d_solvers",
                parameters={"solvers": ["avl", "gnvp3"]},
                dependencies=["step_1"],
            ),
            WorkflowStep(
                id="step_3",
                name="Flight State Configuration",
                description="Configure flight state",
                action="configure_flight_state",
                parameters={"environment": "isa", "altitude": 10000},
                dependencies=["step_2"],
            ),
            WorkflowStep(
                id="step_4",
                name="Analysis Execution",
                description="Execute 3D analysis",
                action="run_3d_analysis",
                parameters={},
                dependencies=["step_3"],
                timeout=600,
            ),
            WorkflowStep(
                id="step_5",
                name="Results Saving",
                description="Save results to database",
                action="save_results",
                parameters={"auto_save": True},
                dependencies=["step_4"],
            ),
        ]

        return WorkflowTemplate(
            id=str(uuid.uuid4()),
            name="Standard Airplane Analysis",
            description="Complete airplane analysis with 3D solvers",
            type=WorkflowType.AIRPLANE_ANALYSIS,
            steps=steps,
            parameters={
                "default_solvers": ["avl"],
                "default_environment": "isa",
                "default_altitude": 10000,
            },
            metadata={"version": "1.0", "author": "ICARUS", "builtin": True},
        )

    def _create_batch_processing_workflow(self) -> WorkflowTemplate:
        """Create built-in batch processing workflow."""
        steps = [
            WorkflowStep(
                id="step_1",
                name="Batch Selection",
                description="Select airfoils for batch processing",
                action="select_batch_airfoils",
                parameters={"min_count": 1, "max_count": 10},
            ),
            WorkflowStep(
                id="step_2",
                name="Parameter Configuration",
                description="Configure batch parameters",
                action="configure_batch_parameters",
                parameters={"angles": "0:15:16", "reynolds": 1e6},
                dependencies=["step_1"],
            ),
            WorkflowStep(
                id="step_3",
                name="Batch Execution",
                description="Execute batch analysis",
                action="run_batch_analysis",
                parameters={"parallel": True, "max_workers": 4},
                dependencies=["step_2"],
                timeout=1800,
            ),
            WorkflowStep(
                id="step_4",
                name="Results Aggregation",
                description="Aggregate and save batch results",
                action="aggregate_results",
                parameters={"export_format": "json"},
                dependencies=["step_3"],
            ),
        ]

        return WorkflowTemplate(
            id=str(uuid.uuid4()),
            name="Batch Airfoil Analysis",
            description="Analyze multiple airfoils with the same parameters",
            type=WorkflowType.BATCH_PROCESSING,
            steps=steps,
            parameters={
                "default_angles": "0:15:16",
                "default_reynolds": 1e6,
                "parallel_processing": True,
                "max_workers": 4,
            },
            metadata={"version": "1.0", "author": "ICARUS", "builtin": True},
        )

    def _register_default_handlers(self):
        """Register default action handlers."""
        self.action_handlers.update(
            {
                "select_airfoils": self._handle_select_airfoils,
                "select_airplane": self._handle_select_airplane,
                "select_solvers": self._handle_select_solvers,
                "select_3d_solvers": self._handle_select_3d_solvers,
                "select_batch_airfoils": self._handle_select_batch_airfoils,
                "configure_parameters": self._handle_configure_parameters,
                "configure_flight_state": self._handle_configure_flight_state,
                "configure_batch_parameters": self._handle_configure_batch_parameters,
                "run_analysis": self._handle_run_analysis,
                "run_3d_analysis": self._handle_run_3d_analysis,
                "run_batch_analysis": self._handle_run_batch_analysis,
                "save_results": self._handle_save_results,
                "aggregate_results": self._handle_aggregate_results,
            },
        )

    def register_action_handler(self, action: str, handler: Callable):
        """Register a custom action handler."""
        self.action_handlers[action] = handler

    def register_progress_callback(self, callback: Callable):
        """Register a progress callback function."""
        self.progress_callbacks.append(callback)

    async def create_execution(
        self,
        template_id: str,
        parameters: Dict[str, Any] = None,
    ) -> Optional[str]:
        """Create a new workflow execution."""
        definition = self.storage.get_definition(template_id)
        if not definition:
            logger.error(f"Workflow template not found: {template_id}")
            return None

        execution_id = str(uuid.uuid4())
        execution = WorkflowExecution(
            id=execution_id,
            workflow_name=definition.template.name,
            status=WorkflowStatus.CREATED,
            start_time=datetime.now(),
            parameters=parameters or {},
            metadata={"template_id": template_id},
        )

        # Initialize step statuses
        for step in definition.template.steps:
            step.status = StepStatus.PENDING

        self.active_executions[execution_id] = execution
        definition.executions.append(execution)
        self.storage.save_definition(definition)

        logger.info(f"Created workflow execution: {execution_id}")
        return execution_id

    async def execute_workflow(self, execution_id: str) -> bool:
        """Execute a workflow with step-by-step processing and progress tracking."""
        execution = self.active_executions.get(execution_id)
        if not execution:
            logger.error(f"Workflow execution not found: {execution_id}")
            return False

        definition = self.storage.get_definition(execution.metadata.get("template_id"))
        if not definition:
            logger.error(f"Workflow template not found for execution: {execution_id}")
            return False

        try:
            execution.status = WorkflowStatus.RUNNING
            self._notify_progress(execution_id, "started", 0)

            # Execute steps in dependency order
            steps_to_execute = self._get_execution_order(definition.template.steps)
            total_steps = len(steps_to_execute)

            for i, step in enumerate(steps_to_execute):
                if execution.status == WorkflowStatus.CANCELLED:
                    break

                # Check dependencies
                if not self._check_dependencies(step, execution.completed_steps):
                    self._log_error(
                        execution,
                        f"Dependencies not met for step: {step.name}",
                    )
                    continue

                # Execute step
                success = await self._execute_step(execution, step)

                if success:
                    execution.completed_steps.append(step.id)
                    step.status = StepStatus.COMPLETED
                    progress = ((i + 1) / total_steps) * 100
                    self._notify_progress(
                        execution_id,
                        f"completed_step_{step.id}",
                        progress,
                    )
                else:
                    execution.failed_steps.append(step.id)
                    step.status = StepStatus.FAILED

                    if step.required:
                        execution.status = WorkflowStatus.FAILED
                        self._notify_progress(
                            execution_id,
                            "failed",
                            ((i + 1) / total_steps) * 100,
                        )
                        break

            # Finalize execution
            if execution.status == WorkflowStatus.RUNNING:
                execution.status = WorkflowStatus.COMPLETED
                self._notify_progress(execution_id, "completed", 100)

            execution.end_time = datetime.now()
            self.storage.save_definition(definition)

            logger.info(
                f"Workflow execution completed: {execution_id} - Status: {execution.status}",
            )
            return execution.status == WorkflowStatus.COMPLETED

        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.end_time = datetime.now()
            self._log_error(execution, f"Workflow execution failed: {str(e)}")
            self._notify_progress(execution_id, "failed", 0)
            logger.error(f"Workflow execution failed: {execution_id} - {e}")
            return False

    def _get_execution_order(self, steps: List[WorkflowStep]) -> List[WorkflowStep]:
        """Get steps in dependency execution order using topological sort."""
        # Simple topological sort implementation
        in_degree = {step.id: 0 for step in steps}
        step_map = {step.id: step for step in steps}

        # Calculate in-degrees
        for step in steps:
            for dep in step.dependencies:
                if dep in in_degree:
                    in_degree[step.id] += 1

        # Find steps with no dependencies
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            current_id = queue.pop(0)
            result.append(step_map[current_id])

            # Update in-degrees for dependent steps
            for step in steps:
                if current_id in step.dependencies:
                    in_degree[step.id] -= 1
                    if in_degree[step.id] == 0:
                        queue.append(step.id)

        return result

    def _check_dependencies(
        self,
        step: WorkflowStep,
        completed_steps: List[str],
    ) -> bool:
        """Check if step dependencies are satisfied."""
        return all(dep in completed_steps for dep in step.dependencies)

    async def _execute_step(
        self,
        execution: WorkflowExecution,
        step: WorkflowStep,
    ) -> bool:
        """Execute a single workflow step with error handling and retries."""
        step.status = StepStatus.RUNNING
        step.start_time = datetime.now()

        for attempt in range(step.max_retries + 1):
            try:
                # Check skip condition
                if step.skip_condition and self._evaluate_condition(
                    step.skip_condition,
                    execution,
                ):
                    step.status = StepStatus.SKIPPED
                    step.end_time = datetime.now()
                    logger.info(f"Skipped step: {step.name}")
                    return True

                # Execute with timeout
                if step.timeout:
                    result = await asyncio.wait_for(
                        self._execute_action(step.action, step.parameters, execution),
                        timeout=step.timeout,
                    )
                else:
                    result = await self._execute_action(
                        step.action,
                        step.parameters,
                        execution,
                    )

                # Store result
                step.result = result
                execution.step_results[step.id] = result

                # Validate result
                if step.validation and not self._validate_result(
                    result,
                    step.validation,
                ):
                    raise ValueError(f"Step validation failed: {step.name}")

                step.status = StepStatus.COMPLETED
                step.end_time = datetime.now()
                logger.info(f"Completed step: {step.name}")
                return True

            except asyncio.TimeoutError:
                error_msg = f"Step timeout after {step.timeout} seconds: {step.name}"
                step.error_message = error_msg
                self._log_error(execution, error_msg)

            except Exception as e:
                error_msg = f"Step execution error: {step.name} - {str(e)}"
                step.error_message = error_msg
                self._log_error(execution, error_msg)

            # Retry logic
            if attempt < step.max_retries:
                step.status = StepStatus.RETRYING
                step.retry_count += 1
                logger.warning(
                    f"Retrying step {step.name} (attempt {attempt + 2}/{step.max_retries + 1})",
                )
                await asyncio.sleep(2**attempt)  # Exponential backoff
            else:
                step.status = StepStatus.FAILED
                step.end_time = datetime.now()
                logger.error(
                    f"Step failed after {step.max_retries + 1} attempts: {step.name}",
                )
                return False

        return False

    async def _execute_action(
        self,
        action: str,
        parameters: Dict[str, Any],
        execution: WorkflowExecution,
    ) -> Any:
        """Execute a workflow action."""
        handler = self.action_handlers.get(action)
        if not handler:
            raise ValueError(f"Unknown action: {action}")

        # Pass execution context to handler
        if asyncio.iscoroutinefunction(handler):
            return await handler(parameters, execution)
        else:
            return handler(parameters, execution)

    def _evaluate_condition(self, condition: str, execution: WorkflowExecution) -> bool:
        """Evaluate a skip condition."""
        # Simple condition evaluation - can be enhanced with expression parser
        try:
            # Example conditions: "step_1_result.status == 'skip'"
            context = {
                "execution": execution,
                "results": execution.step_results,
                "parameters": execution.parameters,
            }
            return eval(condition, {"__builtins__": {}}, context)
        except Exception as e:
            logger.warning(f"Failed to evaluate condition '{condition}': {e}")
            return False

    def _validate_result(self, result: Any, validation: Dict[str, Any]) -> bool:
        """Validate a step result."""
        try:
            # Simple validation rules
            if "required_keys" in validation:
                if not isinstance(result, dict):
                    return False
                for key in validation["required_keys"]:
                    if key not in result:
                        return False

            if "type" in validation:
                expected_type = validation["type"]
                if expected_type == "dict" and not isinstance(result, dict):
                    return False
                elif expected_type == "list" and not isinstance(result, list):
                    return False
                elif expected_type == "str" and not isinstance(result, str):
                    return False

            return True
        except Exception as e:
            logger.warning(f"Validation error: {e}")
            return False

    def _log_error(self, execution: WorkflowExecution, message: str):
        """Log an error to the execution error log."""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "traceback": traceback.format_exc(),
        }
        execution.error_log.append(error_entry)

    def _notify_progress(self, execution_id: str, event: str, progress: float):
        """Notify progress callbacks."""
        for callback in self.progress_callbacks:
            try:
                callback(execution_id, event, progress)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    # Default action handlers (placeholders - would be connected to actual CLI functions)
    def _handle_select_airfoils(
        self,
        parameters: Dict[str, Any],
        execution: WorkflowExecution,
    ) -> List[str]:
        """Handle airfoil selection action."""
        return ["naca2412"]  # Placeholder

    def _handle_select_airplane(
        self,
        parameters: Dict[str, Any],
        execution: WorkflowExecution,
    ) -> str:
        """Handle airplane selection action."""
        return "example_airplane"  # Placeholder

    def _handle_select_solvers(
        self,
        parameters: Dict[str, Any],
        execution: WorkflowExecution,
    ) -> List[str]:
        """Handle solver selection action."""
        return parameters.get("solvers", ["xfoil"])

    def _handle_select_3d_solvers(
        self,
        parameters: Dict[str, Any],
        execution: WorkflowExecution,
    ) -> List[str]:
        """Handle 3D solver selection action."""
        return parameters.get("solvers", ["avl"])

    def _handle_select_batch_airfoils(
        self,
        parameters: Dict[str, Any],
        execution: WorkflowExecution,
    ) -> List[str]:
        """Handle batch airfoil selection action."""
        return ["naca2412", "naca4412"]  # Placeholder

    def _handle_configure_parameters(
        self,
        parameters: Dict[str, Any],
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Handle parameter configuration action."""
        return parameters

    def _handle_configure_flight_state(
        self,
        parameters: Dict[str, Any],
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Handle flight state configuration action."""
        return parameters

    def _handle_configure_batch_parameters(
        self,
        parameters: Dict[str, Any],
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Handle batch parameter configuration action."""
        return parameters

    async def _handle_run_analysis(
        self,
        parameters: Dict[str, Any],
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Handle analysis execution action."""
        # Simulate analysis execution
        await asyncio.sleep(1)
        return {"status": "completed", "results": {"cl": 1.2, "cd": 0.01}}

    async def _handle_run_3d_analysis(
        self,
        parameters: Dict[str, Any],
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Handle 3D analysis execution action."""
        # Simulate 3D analysis execution
        await asyncio.sleep(2)
        return {"status": "completed", "results": {"cl": 1.1, "cd": 0.015, "cm": -0.1}}

    async def _handle_run_batch_analysis(
        self,
        parameters: Dict[str, Any],
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Handle batch analysis execution action."""
        # Simulate batch analysis execution
        await asyncio.sleep(3)
        return {
            "status": "completed",
            "results": [
                {"airfoil": "naca2412", "cl": 1.2},
                {"airfoil": "naca4412", "cl": 1.3},
            ],
        }

    def _handle_save_results(
        self,
        parameters: Dict[str, Any],
        execution: WorkflowExecution,
    ) -> bool:
        """Handle results saving action."""
        return True

    def _handle_aggregate_results(
        self,
        parameters: Dict[str, Any],
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Handle results aggregation action."""
        return {"aggregated": True, "format": parameters.get("export_format", "json")}

    # Public API methods
    def get_all_templates(self) -> List[WorkflowTemplate]:
        """Get all available workflow templates."""
        definitions = self.storage.get_all_definitions()
        return [
            definition.template for definition in definitions if definition.is_active
        ]

    def get_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get a specific workflow template."""
        definition = self.storage.get_definition(template_id)
        return definition.template if definition else None

    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get a workflow execution."""
        return self.active_executions.get(execution_id)

    def get_execution_progress(self, execution_id: str) -> Dict[str, Any]:
        """Get workflow execution progress."""
        execution = self.get_execution(execution_id)
        if not execution:
            return {}

        definition = self.storage.get_definition(execution.metadata.get("template_id"))
        if not definition:
            return {}

        total_steps = len(definition.template.steps)
        completed_steps = len(execution.completed_steps)

        return {
            "execution_id": execution_id,
            "workflow_name": execution.workflow_name,
            "status": execution.status.value,
            "progress": (completed_steps / total_steps) * 100 if total_steps > 0 else 0,
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "current_step": execution.current_step,
            "start_time": execution.start_time.isoformat(),
            "duration": str(datetime.now() - execution.start_time)
            if execution.status == WorkflowStatus.RUNNING
            else None,
        }

    async def pause_execution(self, execution_id: str) -> bool:
        """Pause a workflow execution."""
        execution = self.active_executions.get(execution_id)
        if execution and execution.status == WorkflowStatus.RUNNING:
            execution.status = WorkflowStatus.PAUSED
            return True
        return False

    async def resume_execution(self, execution_id: str) -> bool:
        """Resume a paused workflow execution."""
        execution = self.active_executions.get(execution_id)
        if execution and execution.status == WorkflowStatus.PAUSED:
            return await self.execute_workflow(execution_id)
        return False

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a workflow execution."""
        execution = self.active_executions.get(execution_id)
        if execution and execution.status in [
            WorkflowStatus.CREATED,
            WorkflowStatus.RUNNING,
            WorkflowStatus.PAUSED,
        ]:
            execution.status = WorkflowStatus.CANCELLED
            execution.end_time = datetime.now()
            return True
        return False

    def create_template(
        self,
        name: str,
        description: str,
        workflow_type: WorkflowType,
        steps: List[WorkflowStep],
        parameters: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Create a new workflow template."""
        template_id = str(uuid.uuid4())
        template = WorkflowTemplate(
            id=template_id,
            name=name,
            description=description,
            type=workflow_type,
            steps=steps,
            parameters=parameters or {},
            metadata=metadata or {},
        )

        definition = WorkflowDefinition(template=template)
        self.storage.save_definition(definition)

        return template_id

    def delete_template(self, template_id: str) -> bool:
        """Delete a workflow template."""
        definition = self.storage.get_definition(template_id)
        if definition and not definition.template.metadata.get("builtin", False):
            self.storage.delete_definition(template_id)
            return True
        return False


class TemplateManager:
    """Enhanced template manager with pre-built templates and custom template creation."""

    def __init__(
        self,
        templates_dir: str = "~/.icarus/templates",
        workflow_engine: WorkflowEngine = None,
    ):
        self.templates_dir = Path(templates_dir).expanduser()
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.workflow_engine = workflow_engine
        self.custom_templates: Dict[str, WorkflowTemplate] = {}
        self.prebuilt_templates: Dict[str, WorkflowTemplate] = {}

        self._initialize_prebuilt_templates()
        self._load_custom_templates()

    def _initialize_prebuilt_templates(self):
        """Initialize pre-built workflow templates."""
        # Comprehensive airfoil analysis template
        self.prebuilt_templates["comprehensive_airfoil"] = (
            self._create_comprehensive_airfoil_template()
        )

        # Quick airfoil comparison template
        self.prebuilt_templates["airfoil_comparison"] = (
            self._create_airfoil_comparison_template()
        )

        # Airplane design workflow template
        self.prebuilt_templates["airplane_design"] = (
            self._create_airplane_design_template()
        )

        # Optimization workflow template
        self.prebuilt_templates["optimization_workflow"] = (
            self._create_optimization_template()
        )

        # Data export template
        self.prebuilt_templates["data_export"] = self._create_data_export_template()

    def _create_comprehensive_airfoil_template(self) -> WorkflowTemplate:
        """Create comprehensive airfoil analysis template."""
        steps = [
            WorkflowStep(
                id="airfoil_input",
                name="Airfoil Input",
                description="Load or select airfoil geometry",
                action="load_airfoil",
                parameters={"source_type": "file_or_database"},
                validation={"required_keys": ["geometry"], "type": "dict"},
            ),
            WorkflowStep(
                id="reynolds_sweep",
                name="Reynolds Number Sweep",
                description="Configure Reynolds number range",
                action="configure_reynolds_sweep",
                parameters={"min_re": 1e5, "max_re": 1e7, "steps": 5},
                dependencies=["airfoil_input"],
            ),
            WorkflowStep(
                id="angle_sweep",
                name="Angle of Attack Sweep",
                description="Configure angle of attack range",
                action="configure_angle_sweep",
                parameters={"min_alpha": -5, "max_alpha": 20, "steps": 26},
                dependencies=["reynolds_sweep"],
            ),
            WorkflowStep(
                id="multi_solver_analysis",
                name="Multi-Solver Analysis",
                description="Run analysis with multiple solvers",
                action="run_multi_solver_analysis",
                parameters={"solvers": ["xfoil", "foil2wake"], "compare_results": True},
                dependencies=["angle_sweep"],
                timeout=600,
                max_retries=2,
            ),
            WorkflowStep(
                id="results_validation",
                name="Results Validation",
                description="Validate and cross-check results",
                action="validate_results",
                parameters={"tolerance": 0.1, "check_convergence": True},
                dependencies=["multi_solver_analysis"],
            ),
            WorkflowStep(
                id="report_generation",
                name="Report Generation",
                description="Generate comprehensive analysis report",
                action="generate_report",
                parameters={"format": "pdf", "include_plots": True},
                dependencies=["results_validation"],
            ),
        ]

        return WorkflowTemplate(
            id=str(uuid.uuid4()),
            name="Comprehensive Airfoil Analysis",
            description="Complete airfoil analysis with multiple Reynolds numbers and solvers",
            type=WorkflowType.AIRFOIL_ANALYSIS,
            steps=steps,
            parameters={
                "default_solvers": ["xfoil", "foil2wake"],
                "output_formats": ["json", "csv", "pdf"],
                "validation_enabled": True,
            },
            metadata={
                "category": "analysis",
                "complexity": "advanced",
                "estimated_time": "10-30 minutes",
            },
        )

    def _create_airfoil_comparison_template(self) -> WorkflowTemplate:
        """Create airfoil comparison template."""
        steps = [
            WorkflowStep(
                id="select_airfoils",
                name="Select Airfoils",
                description="Select multiple airfoils for comparison",
                action="select_multiple_airfoils",
                parameters={"min_count": 2, "max_count": 5},
                validation={"type": "list", "min_length": 2},
            ),
            WorkflowStep(
                id="common_conditions",
                name="Common Test Conditions",
                description="Set common analysis conditions",
                action="set_common_conditions",
                parameters={"reynolds": 1e6, "angles": "-2:15:18"},
                dependencies=["select_airfoils"],
            ),
            WorkflowStep(
                id="parallel_analysis",
                name="Parallel Analysis",
                description="Run analysis for all airfoils in parallel",
                action="run_parallel_analysis",
                parameters={"solver": "xfoil", "parallel": True},
                dependencies=["common_conditions"],
                timeout=300,
            ),
            WorkflowStep(
                id="comparison_plots",
                name="Comparison Plots",
                description="Generate comparison plots",
                action="create_comparison_plots",
                parameters={"plot_types": ["cl_cd", "cl_alpha", "polar"]},
                dependencies=["parallel_analysis"],
            ),
            WorkflowStep(
                id="performance_ranking",
                name="Performance Ranking",
                description="Rank airfoils by performance metrics",
                action="rank_airfoils",
                parameters={"criteria": ["max_cl", "min_cd", "cl_cd_ratio"]},
                dependencies=["comparison_plots"],
            ),
        ]

        return WorkflowTemplate(
            id=str(uuid.uuid4()),
            name="Airfoil Comparison Study",
            description="Compare multiple airfoils under identical conditions",
            type=WorkflowType.BATCH_PROCESSING,
            steps=steps,
            parameters={
                "comparison_metrics": ["cl", "cd", "cm", "cl_cd"],
                "plot_formats": ["png", "svg"],
                "ranking_weights": {"cl": 0.4, "cd": 0.4, "stability": 0.2},
            },
            metadata={
                "category": "comparison",
                "complexity": "intermediate",
                "estimated_time": "5-15 minutes",
            },
        )

    def _create_airplane_design_template(self) -> WorkflowTemplate:
        """Create airplane design workflow template."""
        steps = [
            WorkflowStep(
                id="design_requirements",
                name="Design Requirements",
                description="Define airplane design requirements",
                action="define_requirements",
                parameters={
                    "mission_type": "general_aviation",
                    "payload": 500,
                    "range": 1000,
                },
            ),
            WorkflowStep(
                id="initial_sizing",
                name="Initial Sizing",
                description="Perform initial airplane sizing",
                action="initial_sizing",
                parameters={"method": "statistical", "safety_factor": 1.2},
                dependencies=["design_requirements"],
            ),
            WorkflowStep(
                id="wing_design",
                name="Wing Design",
                description="Design wing geometry and airfoil selection",
                action="design_wing",
                parameters={"aspect_ratio": 8, "taper_ratio": 0.6},
                dependencies=["initial_sizing"],
            ),
            WorkflowStep(
                id="stability_analysis",
                name="Stability Analysis",
                description="Analyze longitudinal and lateral stability",
                action="stability_analysis",
                parameters={"cg_range": 0.05, "check_modes": True},
                dependencies=["wing_design"],
                timeout=900,
            ),
            WorkflowStep(
                id="performance_analysis",
                name="Performance Analysis",
                description="Calculate performance characteristics",
                action="performance_analysis",
                parameters={"flight_envelope": True, "mission_analysis": True},
                dependencies=["stability_analysis"],
                timeout=600,
            ),
            WorkflowStep(
                id="optimization",
                name="Design Optimization",
                description="Optimize design parameters",
                action="optimize_design",
                parameters={
                    "objectives": ["minimize_weight", "maximize_range"],
                    "constraints": ["stability"],
                },
                dependencies=["performance_analysis"],
                timeout=1800,
                required=False,
            ),
            WorkflowStep(
                id="final_report",
                name="Final Design Report",
                description="Generate comprehensive design report",
                action="generate_design_report",
                parameters={"include_drawings": True, "format": "pdf"},
                dependencies=["performance_analysis", "optimization"],
            ),
        ]

        return WorkflowTemplate(
            id=str(uuid.uuid4()),
            name="Airplane Design Workflow",
            description="Complete airplane design from requirements to final report",
            type=WorkflowType.AIRPLANE_ANALYSIS,
            steps=steps,
            parameters={
                "design_categories": ["general_aviation", "transport", "fighter"],
                "analysis_fidelity": "medium",
                "optimization_enabled": True,
            },
            metadata={
                "category": "design",
                "complexity": "expert",
                "estimated_time": "1-3 hours",
            },
        )

    def _create_optimization_template(self) -> WorkflowTemplate:
        """Create optimization workflow template."""
        steps = [
            WorkflowStep(
                id="define_objectives",
                name="Define Objectives",
                description="Define optimization objectives and constraints",
                action="define_optimization_objectives",
                parameters={
                    "objectives": ["maximize_cl_cd"],
                    "constraints": ["min_cl", "max_cm"],
                },
            ),
            WorkflowStep(
                id="design_variables",
                name="Design Variables",
                description="Select design variables and bounds",
                action="select_design_variables",
                parameters={
                    "variables": ["camber", "thickness"],
                    "bounds": {"camber": [0, 0.1], "thickness": [0.08, 0.18]},
                },
                dependencies=["define_objectives"],
            ),
            WorkflowStep(
                id="optimization_setup",
                name="Optimization Setup",
                description="Configure optimization algorithm",
                action="setup_optimization",
                parameters={
                    "algorithm": "genetic_algorithm",
                    "population": 50,
                    "generations": 100,
                },
                dependencies=["design_variables"],
            ),
            WorkflowStep(
                id="run_optimization",
                name="Run Optimization",
                description="Execute optimization process",
                action="run_optimization",
                parameters={"parallel": True, "max_evaluations": 5000},
                dependencies=["optimization_setup"],
                timeout=3600,
                max_retries=1,
            ),
            WorkflowStep(
                id="results_analysis",
                name="Results Analysis",
                description="Analyze optimization results",
                action="analyze_optimization_results",
                parameters={"plot_convergence": True, "pareto_front": True},
                dependencies=["run_optimization"],
            ),
            WorkflowStep(
                id="optimal_design",
                name="Optimal Design Validation",
                description="Validate optimal design with detailed analysis",
                action="validate_optimal_design",
                parameters={"detailed_analysis": True, "sensitivity_study": True},
                dependencies=["results_analysis"],
            ),
        ]

        return WorkflowTemplate(
            id=str(uuid.uuid4()),
            name="Design Optimization Workflow",
            description="Automated design optimization with multi-objective capabilities",
            type=WorkflowType.CUSTOM,
            steps=steps,
            parameters={
                "optimization_algorithms": [
                    "genetic_algorithm",
                    "particle_swarm",
                    "gradient_based",
                ],
                "convergence_criteria": {"tolerance": 1e-6, "max_iterations": 1000},
                "parallel_processing": True,
            },
            metadata={
                "category": "optimization",
                "complexity": "expert",
                "estimated_time": "30 minutes - 2 hours",
            },
        )

    def _create_data_export_template(self) -> WorkflowTemplate:
        """Create data export workflow template."""
        steps = [
            WorkflowStep(
                id="select_data",
                name="Select Data",
                description="Select analysis results for export",
                action="select_export_data",
                parameters={"data_types": ["polars", "geometry", "performance"]},
            ),
            WorkflowStep(
                id="format_selection",
                name="Format Selection",
                description="Choose export formats",
                action="select_export_formats",
                parameters={"formats": ["csv", "json", "matlab", "excel"]},
                dependencies=["select_data"],
            ),
            WorkflowStep(
                id="data_processing",
                name="Data Processing",
                description="Process and format data for export",
                action="process_export_data",
                parameters={"normalize": True, "add_metadata": True},
                dependencies=["format_selection"],
            ),
            WorkflowStep(
                id="quality_check",
                name="Quality Check",
                description="Verify data integrity and completeness",
                action="verify_export_data",
                parameters={"check_completeness": True, "validate_format": True},
                dependencies=["data_processing"],
            ),
            WorkflowStep(
                id="export_data",
                name="Export Data",
                description="Export data to specified formats and locations",
                action="export_data_files",
                parameters={"output_dir": "./exports", "compress": True},
                dependencies=["quality_check"],
            ),
            WorkflowStep(
                id="generate_manifest",
                name="Generate Manifest",
                description="Create export manifest and documentation",
                action="create_export_manifest",
                parameters={"include_checksums": True, "documentation": True},
                dependencies=["export_data"],
            ),
        ]

        return WorkflowTemplate(
            id=str(uuid.uuid4()),
            name="Data Export Workflow",
            description="Comprehensive data export with multiple formats and validation",
            type=WorkflowType.CUSTOM,
            steps=steps,
            parameters={
                "supported_formats": ["csv", "json", "hdf5", "matlab", "excel", "xml"],
                "compression_options": ["zip", "gzip", "none"],
                "validation_enabled": True,
            },
            metadata={
                "category": "data_management",
                "complexity": "basic",
                "estimated_time": "2-5 minutes",
            },
        )

    def _load_custom_templates(self):
        """Load custom templates from disk."""
        for template_file in self.templates_dir.glob("*.json"):
            try:
                with open(template_file) as f:
                    data = json.load(f)
                    template = self._dict_to_template(data)
                    self.custom_templates[template.id] = template
            except Exception as e:
                logger.warning(f"Failed to load template {template_file}: {e}")

    def _dict_to_template(self, data: Dict[str, Any]) -> WorkflowTemplate:
        """Convert dictionary to WorkflowTemplate."""
        steps = []
        for step_data in data.get("steps", []):
            step = WorkflowStep(
                id=step_data["id"],
                name=step_data["name"],
                description=step_data["description"],
                action=step_data["action"],
                parameters=step_data.get("parameters", {}),
                dependencies=step_data.get("dependencies", []),
                required=step_data.get("required", True),
                skip_condition=step_data.get("skip_condition"),
                validation=step_data.get("validation"),
                max_retries=step_data.get("max_retries", 3),
                timeout=step_data.get("timeout"),
            )
            steps.append(step)

        return WorkflowTemplate(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            type=WorkflowType(data["type"]),
            steps=steps,
            parameters=data.get("parameters", {}),
            metadata=data.get("metadata", {}),
            version=data.get("version", "1.0"),
            created_at=datetime.fromisoformat(
                data.get("created_at", datetime.now().isoformat()),
            ),
            updated_at=datetime.fromisoformat(
                data.get("updated_at", datetime.now().isoformat()),
            ),
        )

    def _template_to_dict(self, template: WorkflowTemplate) -> Dict[str, Any]:
        """Convert WorkflowTemplate to dictionary."""
        # Convert steps to dict with proper enum serialization
        steps_data = []
        for step in template.steps:
            step_dict = asdict(step)
            # Convert enum values to strings
            if "status" in step_dict and hasattr(step_dict["status"], "value"):
                step_dict["status"] = step_dict["status"].value
            steps_data.append(step_dict)

        return {
            "id": template.id,
            "name": template.name,
            "description": template.description,
            "type": template.type.value,
            "steps": steps_data,
            "parameters": template.parameters,
            "metadata": template.metadata,
            "version": template.version,
            "created_at": template.created_at.isoformat(),
            "updated_at": template.updated_at.isoformat(),
        }

    def save_template(self, template: WorkflowTemplate):
        """Save a custom template to disk."""
        try:
            template_file = self.templates_dir / f"{template.id}.json"
            with open(template_file, "w") as f:
                json.dump(self._template_to_dict(template), f, indent=2)

            self.custom_templates[template.id] = template
            notification_system.success(f"Template saved: {template.name}")
            logger.info(f"Saved custom template: {template.name} ({template.id})")
        except Exception as e:
            notification_system.error(f"Failed to save template: {e}")
            logger.error(f"Failed to save template {template.name}: {e}")

    def create_template_from_execution(
        self,
        execution_id: str,
        template_name: str,
        description: str = None,
    ) -> Optional[str]:
        """Create a template from a successful workflow execution."""
        if not self.workflow_engine:
            logger.error("Workflow engine not available for template creation")
            return None

        execution = self.workflow_engine.get_execution(execution_id)
        if not execution or execution.status != WorkflowStatus.COMPLETED:
            logger.error(
                f"Cannot create template from incomplete execution: {execution_id}",
            )
            return None

        # Get the original template
        definition = self.workflow_engine.storage.get_definition(
            execution.metadata.get("template_id"),
        )
        if not definition:
            logger.error(f"Original template not found for execution: {execution_id}")
            return None

        # Create new template based on execution
        template_id = str(uuid.uuid4())
        new_template = WorkflowTemplate(
            id=template_id,
            name=template_name,
            description=description
            or f"Template created from execution {execution_id}",
            type=definition.template.type,
            steps=definition.template.steps.copy(),  # Copy steps from original
            parameters=execution.parameters,  # Use execution parameters
            metadata={
                "created_from_execution": execution_id,
                "original_template": definition.template.id,
                "category": "custom",
                "complexity": "user_defined",
            },
        )

        self.save_template(new_template)
        return template_id

    def get_all_templates(self) -> Dict[str, WorkflowTemplate]:
        """Get all available templates (pre-built + custom)."""
        all_templates = {}
        all_templates.update(self.prebuilt_templates)
        all_templates.update(self.custom_templates)
        return all_templates

    def get_templates_by_category(self, category: str) -> List[WorkflowTemplate]:
        """Get templates filtered by category."""
        templates = []
        for template in self.get_all_templates().values():
            if template.metadata.get("category") == category:
                templates.append(template)
        return templates

    def get_templates_by_type(
        self,
        workflow_type: WorkflowType,
    ) -> List[WorkflowTemplate]:
        """Get templates filtered by workflow type."""
        templates = []
        for template in self.get_all_templates().values():
            if template.type == workflow_type:
                templates.append(template)
        return templates

    def search_templates(self, query: str) -> List[WorkflowTemplate]:
        """Search templates by name, description, or metadata."""
        query_lower = query.lower()
        matching_templates = []

        for template in self.get_all_templates().values():
            if (
                query_lower in template.name.lower()
                or query_lower in template.description.lower()
                or any(
                    query_lower in str(v).lower() for v in template.metadata.values()
                )
            ):
                matching_templates.append(template)

        return matching_templates

    def delete_template(self, template_id: str) -> bool:
        """Delete a custom template."""
        if template_id in self.prebuilt_templates:
            logger.warning(f"Cannot delete pre-built template: {template_id}")
            return False

        if template_id in self.custom_templates:
            template_file = self.templates_dir / f"{template_id}.json"
            if template_file.exists():
                template_file.unlink()

            template_name = self.custom_templates[template_id].name
            del self.custom_templates[template_id]
            notification_system.success(f"Template deleted: {template_name}")
            logger.info(f"Deleted custom template: {template_name} ({template_id})")
            return True
        else:
            notification_system.error(f"Template not found: {template_id}")
            return False

    def export_template(self, template_id: str, export_path: str) -> bool:
        """Export a template to a file."""
        # First check our own templates
        all_templates = self.get_all_templates()
        template = all_templates.get(template_id)

        # If not found, check workflow engine storage
        if not template and self.workflow_engine:
            template = self.workflow_engine.get_template(template_id)

        if not template:
            logger.error(f"Template not found for export: {template_id}")
            return False

        try:
            export_data = self._template_to_dict(template)

            with open(export_path, "w") as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Exported template {template.name} to {export_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export template: {e}")
            return False

    def import_template(self, import_path: str) -> Optional[str]:
        """Import a template from a file."""
        try:
            with open(import_path) as f:
                data = json.load(f)

            template = self._dict_to_template(data)
            # Generate new ID to avoid conflicts
            template.id = str(uuid.uuid4())
            template.updated_at = datetime.now()

            self.save_template(template)
            logger.info(f"Imported template: {template.name}")
            return template.id
        except Exception as e:
            logger.error(f"Failed to import template from {import_path}: {e}")
            return None

    def get_template_statistics(self) -> Dict[str, Any]:
        """Get statistics about available templates."""
        all_templates = self.get_all_templates()

        stats = {
            "total_templates": len(all_templates),
            "prebuilt_templates": len(self.prebuilt_templates),
            "custom_templates": len(self.custom_templates),
            "by_type": {},
            "by_category": {},
            "by_complexity": {},
        }

        for template in all_templates.values():
            # Count by type
            type_name = template.type.value
            stats["by_type"][type_name] = stats["by_type"].get(type_name, 0) + 1

            # Count by category
            category = template.metadata.get("category", "uncategorized")
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1

            # Count by complexity
            complexity = template.metadata.get("complexity", "unknown")
            stats["by_complexity"][complexity] = (
                stats["by_complexity"].get(complexity, 0) + 1
            )

        return stats


# Global instances
workflow_engine = WorkflowEngine()
template_manager = TemplateManager(workflow_engine=workflow_engine)
