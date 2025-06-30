"""Analysis configuration using Rich interface."""

from typing import Any

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from cli.options import get_option
from cli.options import input_options
from ICARUS.airfoils import Airfoil
from ICARUS.computation.solvers import Solver
from ICARUS.vehicle import Airplane

# Console for rich output
console = Console()


def set_analysis(solver: Solver) -> None:
    """Set analysis type using Rich interface."""
    analyses: list[str] = solver.get_analyses_names(verbose=True)

    if not analyses:
        console.print("[yellow]No analyses available[/yellow]")
        return

    # Create selection table
    table = Table(title="Available Analyses", show_header=True, header_style="bold magenta")
    table.add_column("Index", style="cyan", no_wrap=True)
    table.add_column("Analysis", style="white")

    for i, analysis in enumerate(analyses, 1):
        table.add_row(str(i), analysis)

    console.print(table)

    choice = Prompt.ask("Select analysis", choices=[str(i) for i in range(1, len(analyses) + 1)], default="1")

    try:
        index = int(choice) - 1
        selected_analysis = analyses[index]
        solver.select_analysis(selected_analysis)
        console.print(f"[green]✓[/green] Selected analysis: [cyan]{selected_analysis}[/cyan]")
    except (ValueError, IndexError) as e:
        console.print(f"[red]✗[/red] Failed to select analysis: {e}")


def set_analysis_options(solver: Solver, obj: Airplane | Airfoil) -> None:
    """Set analysis options using Rich interface."""
    _ = solver.get_analysis_options(verbose=True)
    options = solver.analyses[solver.mode].options

    if not options:
        console.print("[yellow]No analysis options to configure[/yellow]")
        return

    console.print("\n[bold]Analysis Options:[/bold]")

    answers: dict[str, Any] = {}
    for option_name, option in options.items():
        if option.name == "airfoil":
            answers[option.name] = obj
            continue
        if option.name == "state" and isinstance(obj, Airplane):
            from cli.airplane_cli import get_flight_state

            state = get_flight_state(obj)
            if state:
                answers[option.name] = state
            continue
        if option.name == "solver2D":
            from cli.airplane_cli import get_2d_polars_solver

            solver_2d = get_2d_polars_solver()
            if solver_2d:
                answers[option.name] = solver_2d
            continue

        try:
            question_type = input_options[option.value_type]
        except KeyError:
            console.print(f"[red]Option {option} has an invalid type[/red]")
            continue

        answer = get_option(option.name, question_type)
        if answer:
            answers[option.name] = answer[option.name]

    if answers:
        try:
            solver.set_analysis_options(answers)
            console.print("[green]✓[/green] Analysis options configured successfully")
            _ = solver.get_analysis_options(verbose=True)
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to set options: {e}")
            if console.input("Try again? (y/n): ").lower().startswith("y"):
                set_analysis_options(solver, obj)
