from __future__ import annotations

import inspect
import logging
from dataclasses import asdict
from io import StringIO
from typing import Any
from typing import Callable
from typing import Generic
from typing import TypeVar

import jsonpickle
from pandas import DataFrame
from rich.console import Console
from rich.table import Table

from ICARUS.computation import SimulationRunner
from ICARUS.computation import SolverParameters
from ICARUS.computation.core import ExecutionContext
from ICARUS.computation.core import Priority
from ICARUS.computation.core import Task
from ICARUS.computation.core import TaskConfiguration
from ICARUS.computation.core import TaskExecutorProtocol
from ICARUS.computation.core import TaskId
from ICARUS.computation.core import TaskResult
from ICARUS.computation.core import TaskState

from . import AnalysisInput
from . import Input

AnalysisInputType = TypeVar("AnalysisInputType", bound=AnalysisInput)


class AnalysisExecutor(TaskExecutorProtocol):
    """
    A simple task executor for running analysis functions.
    """

    def __init__(self, execute_fun: Callable[..., Any]):
        self.execute_fun = execute_fun

    async def execute(self, task_input: dict[str, Any], context: ExecutionContext) -> Any:
        """
        Executes the analysis function.
        Note: This executor runs a synchronous function. The execution engine
        is responsible for running this in a non-blocking manner (e.g., thread pool).
        """
        context.logger.info(f"Executing analysis function: {self.execute_fun.__name__}")
        try:
            result = self.execute_fun(**task_input)
            context.logger.info("Analysis function executed successfully.")
            return result
        except Exception as e:
            context.logger.error(
                "An error occurred during analysis execution.",
            )
            raise e

    async def validate_input(self, task_input: dict[str, Any]) -> bool:
        # For now, we assume input is validated before task creation.
        return True

    async def cancel(self) -> None:
        # Cancellation is not supported in this simple executor.
        pass

    async def cleanup(self) -> None:
        pass


class Analysis(Generic[AnalysisInputType]):
    """Analysis Class. Used to define an analysis and store all the necessary information for it.
    The analysis can be run by calling the object or by creating a task for execution engines.
    Results can be obtained by calling the get_results function.

    Args:
        solver_name (str): Name of the associated solver
        analysis_name (str): Name of the analysis
        inputs (list[Input]): Analysis options
        execute_fun (Callable[..., Any]): Function to run the analysis
        unhook (Callable[...,Any] | None, optional): Function to run after the analysis for post processing. Defaults to None.

    """

    def __init__(
        self,
        solver_name: str,
        analysis_name: str,
        inputs: list[Input],
        execute_fun: Callable[..., Any],
        unhook: Callable[..., Any] | None = None,
        input_type: type[AnalysisInputType] | None = None,
    ) -> None:
        """Initializes an Analysis object

        Args:
            solver_name (str): Name of the associated solver
            analysis_name (str): Name of the analysis
            inputs (list[Input]): Analysis options
            execute_fun (Callable[..., Any]): Function to run the analysis
            unhook (Callable[...,Any] | None, optional): Function to run after the analysis Mainly for post processing. Defaults to None.
            input_type (type[AnalysisInput] | None, optional): The dataclass type for analysis inputs. Defaults to None.
        """

        self.solver_name: str = solver_name
        self.name: str = analysis_name
        self.inputs: dict[str, Input] = {option.name: option for option in inputs}
        self.execute: Callable[..., Any] = execute_fun
        self.logger = logging.getLogger(self.__class__.__name__)
        self.input_type: type[AnalysisInputType] | None = input_type

        if callable(unhook):
            self.unhook: Callable[..., DataFrame | int] = unhook
        else:
            self.unhook = lambda: 0

    def __str__(self) -> str:
        """String representation of the analysis

        Returns:
            str: Name and Options of the analysis

        """
        str_io = StringIO()
        console = Console(file=str_io, force_terminal=False, width=120)

        table = Table(
            title=f"Available Options of {self.solver_name} for {self.name}",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("VarName", style="cyan", no_wrap=True, width=20)
        table.add_column("Type", style="yellow")
        table.add_column("Description", style="green")

        for opt in self.inputs.values():
            table.add_row(
                opt.name,
                str(opt.value_type.__name__),
                opt.description,
            )

        console.print(table)
        return str_io.getvalue()

    def create_task(
        self,
        analysis_input: AnalysisInputType,
        solver_parameters: SolverParameters,
        task_id: TaskId | None = None,
        priority: Priority = Priority.NORMAL,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        """Create a task definition for this analysis.

        Args:
            analysis_input: Dataclass with the inputs for the analysis.
            solver_parameters: Dataclass with the solver parameters.
            task_id: Optional task ID, generates UUID if None
            priority: Task priority
            metadata: Additional metadata for the task

        Returns:
            Task: Task definition compatible with execution engines

        Raises:
            ValueError: If analysis options are not properly defined
        """

        # Prepare execution arguments
        task_input = asdict(analysis_input)
        task_input["solver_parameters"] = solver_parameters

        # Prepare metadata
        task_metadata = {
            "solver_name": self.solver_name,
            "analysis_name": self.name,
            "created_by": "Analysis.create_task",
            **(metadata or {}),
        }
        # Create Task Configuration
        config = TaskConfiguration(
            priority=priority,
            tags=[self.solver_name, self.name],
            resources=task_metadata,
        )

        # Instantiate Executor
        executor = AnalysisExecutor(self.execute)

        # Return task definition
        task = Task(
            name=f"{self.solver_name} - {self.name}",
            executor=executor,
            task_input=task_input,
            config=config,
            task_id=task_id,
        )
        return task

    async def run_analysis(
        self: Analysis[AnalysisInputType],
        runner: SimulationRunner,
        solver_parameters: SolverParameters,
    ) -> list[TaskResult]:
        """
        Runs a given analysis for a series of inputs using a simulation runner.

        Args:
            analysis (Analysis): The analysis to run.

        Returns:
            list[TaskResult]: A list of task results.
        """
        # This is a simplification; a more robust implementation would be needed
        # to map the old Struct-based options to the new AnalysisInput dataclasses.
        # For now, we assume the analysis has a corresponding input type that can be instantiated.

        if self.input_type is None:
            # Attempt to find a suitable input type from the available analyses inputs
            # This part is complex and might require a more explicit mapping.
            # For this refactoring, we'll assume a naming convention or direct mapping exists.
            self.logger.error(f"Cannot execute analysis '{self.name}': missing 'input_type'.")
            raise ValueError

        input_data = {opt.name: opt.value for opt in self.inputs.values() if opt.value is not None}

        try:
            analysis_input = self.input_type(**input_data)
        except TypeError as e:
            self.logger.error(f"Failed to create analysis input for {self.name}: {e}")
            raise ValueError

        tasks = [self.create_task(analysis_input=analysis_input, solver_parameters=solver_parameters)]
        results = await runner.run_tasks(tasks)

        for result in results:
            if isinstance(result, Exception):
                self.logger.error(
                    f"Task execution failed with exception: {result}",
                )
                import traceback

                traceback.print_exc()

            if isinstance(result, TaskResult) and result.state == TaskState.FAILED:
                self.logger.error(f"Task {result.task_id} failed with error: {result.error}")
                if result.error:
                    self.logger.error(result.error)
                    raise (result.error)

        return results

    def estimate_resource_requirements(self) -> dict[str, Any]:
        """Estimate resource requirements for this analysis.

        Returns:
            dict: Resource requirements estimate
        """
        # Base requirements - can be overridden by subclasses
        base_requirements = {
            "cpu_cores": 1,
            "memory_mb": 512,
            "disk_mb": 100,
            "estimated_time_seconds": 60,
            "gpu_memory_mb": 0,
            "network_bandwidth_mbps": 0,
        }

        # Analyze analysis name for hints
        analysis_lower = self.name.lower()

        # CPU-intensive indicators
        if any(keyword in analysis_lower for keyword in ["cfd", "fem", "optimization", "simulation", "polar"]):
            base_requirements["cpu_cores"] = 4
            base_requirements["memory_mb"] = 2048
            base_requirements["estimated_time_seconds"] = 300

        # I/O intensive indicators
        if any(keyword in analysis_lower for keyword in ["file", "read", "write", "load", "save", "export"]):
            base_requirements["disk_mb"] = 1024
            base_requirements["network_bandwidth_mbps"] = 10

        # Memory intensive indicators
        if any(keyword in analysis_lower for keyword in ["mesh", "large", "detailed", "high_res"]):
            base_requirements["memory_mb"] = 4096

        return base_requirements

    def validate_configuration(self) -> dict[str, Any]:
        """Comprehensive validation of analysis configuration.

        Returns:
            dict: Validation result with status, errors, and warnings
        """
        errors = []
        warnings_list = []

        # Validate option values
        for option in self.inputs.values():
            # Add option-specific validation here
            # This can be enhanced based on Input class capabilities
            pass

        # Check if execute function is callable
        if not callable(self.execute):
            errors.append("Execute function is not callable")

        # Check if unhook function is callable (if provided)
        if hasattr(self, "unhook") and self.unhook is not None:
            if not callable(self.unhook):
                warnings_list.append("Unhook function is not callable")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings_list,
            "timestamp": "2025-06-30",  # Could use datetime.now()
        }

    def supports_async_execution(self) -> bool:
        """Check if this analysis supports asynchronous execution.

        Returns:
            bool: True if async execution is supported
        """
        # For now, assume all analyses support async through task framework
        return True

    def get_execution_hints(self) -> dict[str, Any]:
        """Provide hints to execution engines about optimal execution strategy.

        Returns:
            dict: Execution strategy hints
        """
        return {
            "prefers_multiprocessing": False,  # Can be overridden by subclasses
            "is_io_bound": False,
            "is_cpu_bound": True,
            "is_memory_bound": False,
            "supports_checkpointing": False,
            "can_be_interrupted": True,
            "estimated_duration": "medium",  # short, medium, long
        }

    def get_task_dependencies(self) -> list[str]:
        """Get list of task IDs this analysis depends on.

        Returns:
            list: List of dependency task IDs (empty for now)
        """
        # This can be enhanced to analyze options for dependencies
        return []

    def get_results(self, analysis_input: AnalysisInputType) -> DataFrame | int:
        """Function to get the results. Calls the unhooks function.

        Returns:
            DataFrame | int: Results of the analysis or error code

        """
        args_needed = list(inspect.signature(self.unhook).parameters.keys())
        kwargs: dict[str, Any] = {key: value for key, value in asdict(analysis_input).items() if key in args_needed}
        return self.unhook(**kwargs)

    def encode_json(self) -> str:
        encoded: str = str(jsonpickle.encode(self))
        return encoded

    def __getstate__(
        self,
    ) -> tuple[
        str,
        str,
        list[Input],
        Callable[..., Any],
        Callable[..., DataFrame | int],
        type[AnalysisInputType] | None,
    ]:
        return (
            self.solver_name,
            self.name,
            [option for option in self.inputs.values()],
            self.execute,
            self.unhook if hasattr(self, "unhook") else lambda: 0,
            self.input_type,
        )

    def __setstate__(
        self,
        state: tuple[
            str,
            str,
            list[Input],
            Callable[..., Any],
            Callable[..., DataFrame | int],
            type[AnalysisInputType] | None,
        ],
    ) -> None:
        self.solver_name = state[0]
        self.name = state[1]
        self.inputs = {option.name: option for option in state[2]}
        self.execute = state[3]
        self.unhook = state[4]
        self.input_type = state[5]
