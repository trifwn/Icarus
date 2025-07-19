from __future__ import annotations

import inspect
import logging
from dataclasses import asdict
from functools import partial
from io import StringIO
from typing import Any
from typing import Callable
from typing import Generic
from typing import TypeVar

import jsonpickle
from rich.console import Console
from rich.table import Table

from ICARUS.computation import SimulationRunner
from ICARUS.computation import SolverParameters
from ICARUS.computation.core import ExecutionContext
from ICARUS.computation.core import Priority
from ICARUS.computation.core import Task
from ICARUS.computation.core import TaskConfiguration
from ICARUS.computation.core import TaskExecutorProtocol
from ICARUS.computation.core import TaskResult
from ICARUS.computation.core import TaskState

from . import BaseAnalysisInput

AnalysisInput = TypeVar("AnalysisInput", bound=BaseAnalysisInput)
AnalysisOutput = TypeVar("AnalysisOutput", bound=BaseAnalysisInput)


class AnalysisExecutor(TaskExecutorProtocol[AnalysisInput, AnalysisOutput]):
    """
    A simple task executor for running analysis functions.
    """

    def __init__(self, execute_fun: Callable[..., AnalysisOutput]) -> None:
        self.execute_fun = execute_fun

    async def execute(
        self,
        task_input: AnalysisInput,
        context: ExecutionContext,
    ) -> AnalysisOutput:
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

    async def validate_input(self, task_input: AnalysisInput) -> bool:
        # For now, we assume input is validated before task creation.
        return True

    async def cancel(self) -> None:
        # Cancellation is not supported in this simple executor.
        pass

    async def cleanup(self) -> None:
        pass


class Analysis(Generic[AnalysisInput]):
    """Analysis Class. Used to define an analysis and store all the necessary information for it.
    The analysis can be run by calling the object or by creating a task for execution engines.
    Results can be obtained by calling the get_results function.

    Args:
        solver_name (str): Name of the associated solver
        analysis_name (str): Name of the analysis
        inputs (list[Input]): Analysis options
        execute_fun (Callable[..., Any]): Function to run the analysis
        post_execute_fun (Callable[...,Any] | None, optional): Function to run after the analysis for post processing. Defaults to None.

    """

    def __init__(
        self,
        analysis_name: str,
        solver_name: str,
        execute_fun: Callable[..., Any],
        input_type: AnalysisInput,
        post_execute_fun: Callable[..., Any] | None = None,
        monitor_progress_fun: Callable[..., Any] | None = None,
    ) -> None:
        """Initializes an Analysis object

        Args:
            solver_name (str): Name of the associated solver
            analysis_name (str): Name of the analysis
            execute_fun (Callable[..., Any]): Function to run the analysis
            input_type (type[AnalysisInput]): The dataclass type for analysis inputs.
            post_execute_fun (Callable[...,Any] | None, optional): Function to run after the analysis Mainly for post processing. Defaults to None.
        """

        self.solver_name: str = solver_name
        self.name: str = analysis_name

        if not callable(execute_fun):
            raise TypeError(f"execute_fun must be callable, got {type(execute_fun)}")
        self._execute_fun = execute_fun

        self.logger = logging.getLogger(self.__class__.__name__)

        self.input_type: AnalysisInput = input_type
        self.inputs: list[AnalysisInput] = []

        self.post_execute_fun: Callable[..., Any] | None = post_execute_fun
        self.monitor_progress_fun: Callable[..., Any] | None = monitor_progress_fun

    def __str__(self) -> str:
        """String representation of the analysis

        Returns:
            str: Name and Options of the analysis

        """
        str_io = StringIO()
        console = Console(file=str_io, force_terminal=False, width=120)

        table = Table(
            title=f"Analysis {self.name} for {self.solver_name}",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("VarName", style="cyan", no_wrap=True, width=20)
        table.add_column("Type", style="yellow")
        table.add_column("Description", style="green")

        for name, f in self.input_type.__dataclass_fields__.items():
            description = f.metadata.get("description", "")
            # Get type name
            if hasattr(f.type, "__name__"):
                type_name = f.type.__name__  # noqa
            else:
                type_name = str(f.type).replace("typing.", "")

            table.add_row(
                name,
                type_name,
                description,
            )

        console.print(table)
        return str_io.getvalue()

    def get_analysis_input(self, verbose: bool = False) -> AnalysisInput:
        """Get the options of the selected analysis.

        Args:
            verbose (bool, optional): Displays the option if True. Defaults to False.

        Raises:
            Exception: If the analysis has not been selected.

        Returns:
            AnalysisInput: The options of the selected analysis.
        """
        if verbose:
            print(self)
        return self.input_type

    def set_analysis_input(self, inputs: AnalysisInput | dict[str, Any]) -> None:
        """Set the inputs of the selected analysis.

        Args:
            inputs (AnalysisInput | dict[str, Any]): Inputs to set. Can be a Struct or a dictionary.
        """
        if isinstance(inputs, BaseAnalysisInput):
            # If inputs is already an AnalysisInput, convert it to a dict
            input_dict = asdict(inputs)
        elif isinstance(inputs, dict):
            input_dict = inputs
        else:
            raise TypeError(
                f"Inputs must be an AnalysisInput or a dictionary, got {type(inputs)}",
            )

        # Update the analysis input with the provided inputs
        self.inputs = [self.input_type.get_input(input_dict)]

    def set_analysis_multiple_inputs(
        self,
        inputs: list[AnalysisInput | dict[str, Any]],
    ) -> None:
        """Set multiple inputs for the selected analysis.

        Args:
            inputs (list[AnalysisInput | dict[str, Any]]): List of inputs to set.
        """
        self.inputs = []

        for input_data in inputs:
            if isinstance(input_data, BaseAnalysisInput):
                input_dict = asdict(input_data)
            elif isinstance(input_data, dict):
                input_dict = input_data
            else:
                raise TypeError("Inputs must be an AnalysisInput or a dictionary")
            self.inputs.append(self.input_type.get_input(input_dict))

    def create_tasks(
        self,
        inputs: list[AnalysisInput],
        solver_parameters: SolverParameters,
        priority: Priority = Priority.NORMAL,
        metadata: dict[str, Any] | None = None,
    ) -> list[Task]:
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
        tasks = []
        for input in inputs:
            for input_name, sub_input in input.expand_dataclass().items():
                # Validate analysis input
                sub_input.validate()

                # Prepare execution arguments
                task_input = asdict(sub_input)
                task_input["solver_parameters"] = solver_parameters

                # Create Task Configuration
                config = TaskConfiguration(
                    priority=priority,
                    tags=[self.solver_name, self.name],
                )

                # Instantiate Executor
                executor = AnalysisExecutor(self._execute_fun)

                # Prepare metadata
                task_metadata = {
                    "solver_name": self.solver_name,
                    "analysis_name": self.name,
                    "created_by": "Analysis.create_task",
                    "input_name": input_name,
                    **(metadata or {}),
                }

                progress_probe = None
                if self.monitor_progress_fun:
                    args_needed = list(
                        inspect.signature(self.monitor_progress_fun).parameters.keys(),
                    )
                    kwargs: dict[str, Any] = {
                        key: value
                        for key, value in task_input.items()
                        if key in args_needed
                    }

                    progress_probe = partial(
                        self.monitor_progress_fun,
                        **kwargs,
                    )

                # Return task definition
                task = Task(
                    name=f"{input_name}" if input_name else self.name,
                    executor=executor,
                    task_input=task_input,
                    config=config,
                    metadata=task_metadata,
                    progress_probe=progress_probe,
                )
                tasks.append(task)

        return tasks

    async def run_analysis(
        self: Analysis[AnalysisInput],
        runner: SimulationRunner,
        solver_parameters: SolverParameters,
    ) -> tuple[list[Task], list[TaskResult]]:
        """
        Runs a given analysis for a series of inputs using a simulation runner.

        Args:
            analysis (Analysis): The analysis to run.

        Returns:
            list[TaskResult]: A list of task results.
        """
        if not self.inputs:
            raise ValueError(
                "No inputs provided for the analysis. Please set the inputs before running.",
            )

        if not isinstance(self.inputs, list):
            raise ValueError("Inputs must be a list of AnalysisInput instances.")

        if not all(
            isinstance(input_item, type(self.input_type)) for input_item in self.inputs
        ):
            raise ValueError(
                f"All inputs must be of type {self.input_type.__name__}. "
                f"Received types: {[type(input_item).__name__ for input_item in self.inputs]}",
            )

        # Validate inputs
        tasks = self.create_tasks(
            inputs=self.inputs,
            solver_parameters=solver_parameters,
        )
        results = await runner.run_tasks(tasks)

        for result in results:
            if isinstance(result, Exception):
                self.logger.error(
                    f"Task execution failed with exception: {result}",
                )
                import traceback

                traceback.print_exc()

            if (
                isinstance(result, TaskResult)
                and result.state == TaskState.FAILED
                and result.error
            ):
                self.logger.error(
                    f"Task {result.task_id} failed with error: {result.error}",
                )
        return tasks, results

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
        if any(
            keyword in analysis_lower
            for keyword in ["cfd", "fem", "optimization", "simulation", "polar"]
        ):
            base_requirements["cpu_cores"] = 4
            base_requirements["memory_mb"] = 2048
            base_requirements["estimated_time_seconds"] = 300

        # I/O intensive indicators
        if any(
            keyword in analysis_lower
            for keyword in ["file", "read", "write", "load", "save", "export"]
        ):
            base_requirements["disk_mb"] = 1024
            base_requirements["network_bandwidth_mbps"] = 10

        # Memory intensive indicators
        if any(
            keyword in analysis_lower
            for keyword in ["mesh", "large", "detailed", "high_res"]
        ):
            base_requirements["memory_mb"] = 4096

        return base_requirements

    def post_run_analysis(
        self,
        analysis_input: AnalysisInput,
        analysis_results: Any,
    ) -> Any:
        """Function to get the results. Calls the unhooks function.

        Returns:
            DataFrame | int: Results of the analysis or error code

        """
        if self.post_execute_fun is not None:
            args_needed = list(
                inspect.signature(self.post_execute_fun).parameters.keys(),
            )
            kwargs: dict[str, Any] = {
                key: value
                for key, value in asdict(analysis_input).items()
                if key in args_needed
            }
            if "results" in args_needed:
                kwargs["results"] = analysis_results
            return self.post_execute_fun(
                **kwargs,
            )
        else:
            return analysis_results

    def encode_json(self) -> str:
        encoded: str = str(jsonpickle.encode(self))
        return encoded

    def __getstate__(self) -> dict[str, Any]:
        return {
            "solver_name": self.solver_name,
            "name": self.name,
            "execute": self._execute_fun,
            "unhook": getattr(self, "unhook", lambda: 0),
            "input_type": self.input_type,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.solver_name = state["solver_name"]
        self.name = state["name"]
        self._execute_fun = state["execute"]
        self.post_execute_fun = state["unhook"]
        self.input_type = state["input_type"]
