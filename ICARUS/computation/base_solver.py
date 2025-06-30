from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from dataclasses import is_dataclass
from io import StringIO
from typing import Any
from typing import Generic
from typing import TypeVar

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ICARUS.computation import NoSolverParameters
from ICARUS.computation import RichProgressMonitor
from ICARUS.computation import SimulationRunner
from ICARUS.computation import SolverParameters
from ICARUS.computation.analyses import Analysis
from ICARUS.computation.core import ExecutionMode
from ICARUS.core.base_types import Struct
from ICARUS.settings import ICARUS_CONSOLE

SolverParametersType = TypeVar("SolverParametersType", bound=SolverParameters)


class Solver(Generic[SolverParametersType]):
    """Abstract class to represent a solver. It is used to run analyses."""

    def __init__(
        self,
        name: str,
        solver_type: str,
        fidelity: int,
        available_analyses: list[Analysis],
        solver_parameters: SolverParametersType,
    ) -> None:
        """Initialize the Solver class.

        Args:
            name (str): Solver Name.
            solver_type (str): Solver Type.
            fidelity (int): Fidelity of the solver.
            solver_parameters (SolverParameters, optional): Dataclass with solver parameters. Defaults to None.
            available_analyses (list[Analysis]): List of available Analyses.

        """
        self.name: str = name
        self.logger = logging.getLogger(f"{name}: {solver_type}")

        self.type: str = solver_type
        try:
            assert isinstance(fidelity, int), "Fidelity must be an integer"
        except AssertionError:
            logging.error("Fidelity must be an integer")
        self.fidelity: int = fidelity
        self.analyses: dict[str, Analysis] = {}
        for analysis in available_analyses:
            self.analyses[analysis.name] = analysis
        self.mode: str = "None"
        self.solver_parameters: SolverParametersType = solver_parameters
        self.latest_results: list[Any] = []

    def get_analyses_names(self, verbose: bool = False) -> list[str]:
        if verbose:
            print(self)
        return list(self.analyses.keys())

    def select_analysis(self, identifier: str | int) -> None:
        """Set the analysis to be used.

        Args:
            analysis (str): Analysis Name.

        """
        if isinstance(identifier, str):
            self.mode = identifier
        elif isinstance(identifier, int):
            self.mode = list(self.analyses.keys())[identifier]
        else:
            raise ValueError("Invalid Analysis Identifier")

    def get_analysis_options(self, verbose: bool = False) -> Struct:
        """Get the options of the selected analysis.

        Args:
            verbose (bool, optional): Displays the option if True. Defaults to False.

        Raises:
            Exception: If the analysis has not been selected.

        Returns:
            Struct: Struct Object containing the analysis options.

        """
        # Convert Option Object to struct
        ret: Struct = Struct()
        print(self.mode)
        for option in self.analyses[self.mode].inputs.values():
            ret[option.name] = option.value

        if verbose:
            print(self.analyses[self.mode])
        return ret

    def set_analysis_options(self, options: Struct | dict[str, Any]) -> None:
        """Set"""
        for key in options.keys():
            self.analyses[self.mode].inputs[key].value = options[key]

    def get_solver_parameters(self, verbose: bool = False) -> SolverParametersType:
        """Get the solver parameters of the selected analysis."""
        if verbose:
            self.print_solver_parameters()
        return self.solver_parameters

    def set_solver_parameters(self, parameters: dict[str, Any] | SolverParametersType) -> None:
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

    def define_analysis(
        self,
        options: Struct | dict[str, Any],
        solver_parameters: dict[str, Any] | SolverParametersType | None = None,
    ) -> None:
        """Set the options of the selected analysis."""
        if self.mode is not None:
            self.set_analysis_options(options)
            if solver_parameters:
                self.set_solver_parameters(solver_parameters)

    def execute(self, execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL) -> None:
        """
        Run the selected analysis asynchronously.
        This method constructs the AnalysisInput and uses the SimulationRunner.
        """
        analysis = self.analyses[self.mode]

        progress_monitor = RichProgressMonitor()
        runner = SimulationRunner(execution_mode=execution_mode, progress_monitor=progress_monitor)
        results = asyncio.run(analysis.run_analysis(runner, solver_parameters=self.solver_parameters))
        self.latest_results = results

    def get_results(self, analysis_name: str | None = None) -> Any:
        """
        Get the results of the last execution or run the unhook function.
        """
        analysis = self.analyses[analysis_name or self.mode]
        # In the new paradigm, results are returned by execute. This is for unhooking.

        if self.latest_results:
            # Assuming we want to process the first result
            task_result = self.latest_results[0]
            if task_result.success:
                # To call unhook, we need the original input
                # This part of the flow needs careful redesign.
                # For now, returning the raw result.
                return task_result.result

        # Fallback for old unhook logic, though it might not be applicable
        if hasattr(analysis, "unhook") and callable(analysis.unhook):
            # This would need an AnalysisInput, which we don't have here.
            # This indicates a bigger design change is needed for result processing.
            logging.warning(
                "Cannot get results via unhook without a corresponding input. Returning latest raw results if available.",
            )

        return self.latest_results

    def print_analysis_options(self) -> None:
        """Print the options of the selected analysis."""
        if self.mode is not None:
            print(self.analyses[self.mode])
        else:
            print("Analysis hase not been Selected")

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

        analysis_list = "\n".join([f"{i}) {key}" for i, key in enumerate(self.analyses.keys())])

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
