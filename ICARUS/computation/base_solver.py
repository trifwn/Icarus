from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from dataclasses import is_dataclass
from io import StringIO
from typing import Any
from typing import Generic
from typing import TypeVar

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ICARUS import ICARUS_CONSOLE
from ICARUS.computation.core import ExecutionMode
from ICARUS.computation.core.data_structures import TaskResult
from ICARUS.computation.core.protocols import ProgressMonitor

from . import NoSolverParameters
from . import RichProgressMonitor
from . import SimulationRunner
from . import SolverParameters
from .analyses import Analysis
from .analyses import BaseAnalysisInput

SolverParametersType = TypeVar("SolverParametersType", bound=SolverParameters)
AnalysisInputType = TypeVar("AnalysisInputType", bound=BaseAnalysisInput)


class Solver(Generic[SolverParametersType]):
    """Abstract class to represent a solver. It is used to run analyses."""

    analyses: list[Analysis[Any]]

    def __init__(
        self,
        name: str,
        solver_type: str,
        fidelity: int,
        solver_parameters: SolverParametersType,
    ) -> None:
        """Initialize the Solver class.

        Args:
            name (str): Solver Name.
            solver_type (str): Solver Type.
            fidelity (int): Fidelity of the solver.
            solver_parameters (SolverParameters): Dataclass with solver parameters.
            available_analyses (list[Analysis], optional): DEPRECATED. List of available Analyses.
                If not provided, analyses will be auto-generated from class descriptors.

        """
        self.name: str = name
        self.logger = logging.getLogger(f"{name}: {solver_type}")

        self.type: str = solver_type
        try:
            assert isinstance(fidelity, int), "Fidelity must be an integer"
        except AssertionError:
            logging.error("Fidelity must be an integer")
        self.fidelity: int = fidelity

        # # Build analyses from descriptors
        # self.analyses: dict[str, AnalysisType] = {}
        # available_analyses = self._build_analyses_from_descriptors()
        # for analysis in available_analyses:
        #     self.analyses[analysis.name] = analysis
        self.analyses_names: list[str] = []
        for analysis in self.analyses:
            self.analyses_names.append(analysis.name)

        self.solver_parameters: SolverParametersType = solver_parameters
        self.task_results: list[TaskResult] = []

    def get_analyses(self, verbose: bool = False) -> list[Analysis]:
        if verbose:
            print(self)
        return list(self.analyses)

    def get_solver_parameters(self, verbose: bool = False) -> SolverParametersType:
        """Get the solver parameters of the selected analysis."""
        if verbose:
            self.print_solver_parameters()
        return self.solver_parameters

    def set_solver_parameters(
        self,
        parameters: dict[str, Any] | SolverParametersType,
    ) -> None:
        """Set the solver parameters of the selected analysis from a dictionary."""
        if not is_dataclass(self.solver_parameters):
            return

        if isinstance(parameters, self.solver_parameters.__class__):
            # If parameters is already a SolverParametersType, convert it to a dict
            parameters_dict = asdict(parameters)
        elif isinstance(parameters, dict):
            parameters_dict = parameters
        else:
            raise TypeError("Parameters must be a dataclass or a dictionary")

        current_params = asdict(self.solver_parameters)
        current_params.update(parameters_dict)

        try:
            self.solver_parameters = self.solver_parameters.__class__(**current_params)
        except TypeError as e:
            self.logger.error(f"Failed to set solver parameters: {e}")

    def execute(
        self,
        analysis: Analysis,
        inputs: BaseAnalysisInput
        | dict[str, Any]
        | list[BaseAnalysisInput | dict[str, Any]],
        solver_parameters: dict[str, Any] | SolverParametersType | None = None,
        execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        progress_monitor: ProgressMonitor | None = RichProgressMonitor(),
    ) -> Any:
        """
        Run the selected analysis asynchronously.
        This method constructs the AnalysisInput and uses the SimulationRunner.
        """
        if not isinstance(analysis, Analysis):
            raise TypeError("Analysis must be an instance of Analysis class")

        if not isinstance(inputs, (BaseAnalysisInput, dict, list)):
            raise TypeError(
                "Inputs must be a BaseAnalysisInput, dict, or list of BaseAnalysisInput or dict",
            )

        if not isinstance(solver_parameters, (SolverParameters, dict, type(None))):
            raise TypeError(
                "Solver parameters must be a SolverParametersType, dict, or None",
            )

        if not isinstance(execution_mode, ExecutionMode):
            raise TypeError("Execution mode must be an instance of ExecutionMode")

        if analysis.name not in self.analyses_names:
            raise ValueError(
                f"Analysis '{analysis.name}' is not available in this solver.",
            )

        if isinstance(inputs, list):
            analysis.set_analysis_multiple_inputs(inputs)
        else:
            analysis.set_analysis_input(inputs)

        if solver_parameters:
            self.set_solver_parameters(solver_parameters)

        runner = SimulationRunner(
            execution_mode=execution_mode,
            progress_monitor=progress_monitor,
            simulation_name=f"{self.name}: {analysis.name}",
        )

        # Check if there is an event loop running
        try:
            asyncio.get_running_loop()  # Triggers RuntimeError if no running event loop
            # Create a separate thread so we can block before returning
            with ThreadPoolExecutor(1) as pool:
                tasks, task_results = pool.submit(
                    lambda: asyncio.run(
                        analysis.run_analysis(
                            runner,
                            solver_parameters=self.solver_parameters,
                        ),
                    ),
                ).result()
        except RuntimeError:
            tasks, task_results = asyncio.run(
                analysis.run_analysis(runner, solver_parameters=self.solver_parameters),
            )

        results = []
        for input in analysis.inputs:
            input_task_ids = []
            for input_name, sub_input in input.expand_dataclass().items():
                task = next(
                    (
                        t
                        for t in tasks
                        if t.metadata.get("input_name", None) == input_name
                    ),
                    None,
                )
                if task is None:
                    self.logger.error(f"Input: {input_name} not found in task list.")
                    continue
                else:
                    input_task_ids.append(task.id)

            input_results = [
                next((tr.output for tr in task_results if tr.task_id == task.id), None)
                for task in tasks
                if task.id in input_task_ids
            ]
            # fold_results
            input_results = input.fold_results(input_results)

            results.append(
                analysis.post_run_analysis(
                    input,
                    input_results,
                ),
            )
        return results

    def print_solver_parameters(self) -> None:
        """Prints the solver parameters in a rich table format."""
        if (
            not self.solver_parameters
            or not is_dataclass(self.solver_parameters)
            or isinstance(self.solver_parameters, NoSolverParameters)
        ):
            print("No Solver Parameters Needed")
            return

        table = Table(
            title=f"Available Solver Parameters for {self.name}",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("VarName", style="cyan", no_wrap=True)
        table.add_column("Type", style="yellow")
        table.add_column("Value")
        table.add_column("Description", style="green")

        for name, f in self.solver_parameters.__class__.__dataclass_fields__.items():
            value = getattr(self.solver_parameters, name)
            if value is None:
                value_repr = "[red]None[/red]"
            elif isinstance(value, list) and len(value) > 2:
                value_repr = f"[yellow]List with {len(value)} items[/yellow]"
            else:
                value_repr = str(value)

            description = f.metadata.get("description", "")

            # Get type name
            try:
                type_name = f.type.__name__
            except AttributeError:
                # Handle complex types like tuple[float, float]
                type_name = str(f.type).replace("typing.", "")

            table.add_row(
                name,
                type_name,
                value_repr,
                description,
            )

        ICARUS_CONSOLE.print(table)
        ICARUS_CONSOLE.print(
            "\n[italic]If there are multiple values, inspect them separately by calling the option name.[/italic]\n",
        )

    def __str__(self) -> str:
        """String representation of the Solver."""
        str_io = StringIO()
        console = Console(file=str_io, force_terminal=False, width=120)

        analysis_list = "\n".join(
            [f"{i}) {key}" for i, key in enumerate(self.analyses_names, start=1)],
        )

        panel = Panel(
            analysis_list,
            title=f"[bold cyan]{self.type} Solver: {self.name}[/bold cyan]",
            subtitle="[default]Available Analyses[/default]",
            border_style="green",
        )
        console.print(panel)
        return str_io.getvalue()

    def __repr__(self) -> str:
        return f"Solver: {self.name}"
