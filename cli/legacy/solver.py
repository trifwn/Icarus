"""Solver parameter configuration using Rich interface."""

from typing import Any

from rich.console import Console
from rich.prompt import Confirm, Prompt

from ICARUS.computation.solvers import Solver
from ICARUS.core.base_types import Struct

# Console for rich output
console = Console()


def set_solver_parameters(solver: Solver) -> None:
    """Configure solver parameters using Rich interface."""
    parameters: Struct = solver.get_solver_parameters(verbose=True)

    if not Confirm.ask("Do you want to change any of the solver parameters?"):
        return

    console.print("\n[bold]Solver Parameters:[/bold]")

    answers = {}
    for parameter in parameters.keys():
        value = Prompt.ask(f"Enter value for {parameter}")
        if value:
            try:
                # Try to convert to appropriate type
                if "." in value:
                    answers[parameter] = float(value)
                else:
                    answers[parameter] = int(value)
            except ValueError:
                answers[parameter] = value

    if answers:
        try:
            solver.set_solver_parameters(answers)
            console.print("[green]✓[/green] Solver parameters configured successfully")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to set parameters: {e}")
            if Confirm.ask("Try again?"):
                set_solver_parameters(solver)
