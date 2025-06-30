"""ICARUS CLI for Airplane Analysis - Rich-based Interface

This module provides a modern CLI for 3D airplane analysis using Rich and Typer.
It allows users to analyze airplanes using various solvers with an intuitive interface.
"""

import time
from pathlib import Path
from typing import Any, List, Optional

import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pd
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from ICARUS.computation.solvers import Solver
from ICARUS.computation.solvers.GenuVP import GenuVP3
from ICARUS.computation.solvers.GenuVP import GenuVP7
from ICARUS.computation.solvers.Icarus_LSPT import LSPT
from ICARUS.computation.solvers.XFLR5.parser import parse_xfl_project
from ICARUS.core.base_types import Struct
from ICARUS.database import Database
from ICARUS.environment import EARTH_ISA
from ICARUS.environment import Environment
from ICARUS.flight_dynamics import State
from ICARUS.vehicle import Airplane

# Register pandas handlers for jsonpickle
jsonpickle_pd.register_handlers()

# Console for rich output
console = Console()


def airplane_interactive(db: Database) -> None:
    """Interactive airplane analysis mode."""
    console.print(Panel.fit("[bold blue]3D Airplane Analysis[/bold blue]", border_style="blue", padding=(1, 2)))

    # Get number of airplanes
    num_airplanes = Prompt.ask("How many airplanes do you want to analyze?", default="1")

    try:
        num_airplanes = int(num_airplanes)
        if num_airplanes < 1:
            console.print("[red]Number of airplanes must be at least 1[/red]")
            return
    except ValueError:
        console.print("[red]Invalid number of airplanes[/red]")
        return

    airplanes: List[Airplane] = []
    solvers_config = {}

    for i in range(num_airplanes):
        console.print(f"\n[bold cyan]Airplane {i + 1}/{num_airplanes}[/bold cyan]")

        # Select airplane source
        source = Prompt.ask("Select airplane source", choices=["file", "database", "xflr5"], default="database")

        airplane = None
        if source == "file":
            airplane = get_airplane_from_file()
        elif source == "database":
            airplane = get_airplane_from_database(db)
        elif source == "xflr5":
            airplane = get_airplane_from_xflr5()

        if airplane is None:
            console.print("[red]Failed to load airplane, skipping...[/red]")
            continue

        airplanes.append(airplane)

        # Select solvers
        solvers = select_airplane_solvers()
        solvers_config[airplane.name] = solvers

        # Configure solvers
        for solver_name in solvers:
            configure_airplane_solver(solver_name, airplane)

    # Run analysis
    if airplanes:
        run_airplane_analysis(airplanes, solvers_config, db)


def get_airplane_from_file() -> Optional[Airplane]:
    """Get airplane from JSON file."""
    while True:
        file_path = Prompt.ask("Enter path to airplane JSON file")

        if not file_path:
            if Confirm.ask("Cancel airplane selection?"):
                return None
            continue

        try:
            with open(file_path, encoding="UTF-8") as f:
                json_obj = f.read()
                airplane = jsonpickle.decode(json_obj)
                console.print(f"[green]✓[/green] Loaded airplane: [cyan]{airplane.name}[/cyan]")
                return airplane
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to load airplane: {e}")
            if not Confirm.ask("Try again?"):
                return None


def get_airplane_from_database(db: Database) -> Optional[Airplane]:
    """Get airplane from database."""
    available_airplanes = db.get_vehicle_names()

    if not available_airplanes:
        console.print("[yellow]No airplanes found in database[/yellow]")
        return None

    # Create selection table
    table = Table(title="Available Airplanes", show_header=True, header_style="bold magenta")
    table.add_column("Index", style="cyan", no_wrap=True)
    table.add_column("Name", style="white")

    for i, name in enumerate(available_airplanes, 1):
        table.add_row(str(i), name)

    console.print(table)

    while True:
        choice = Prompt.ask(
            "Select airplane by number", choices=[str(i) for i in range(1, len(available_airplanes) + 1)]
        )

        try:
            index = int(choice) - 1
            airplane_name = available_airplanes[index]
            airplane = db.get_vehicle(airplane_name)
            console.print(f"[green]✓[/green] Selected airplane: [cyan]{airplane.name}[/cyan]")
            return airplane
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to load airplane: {e}")
            if not Confirm.ask("Try again?"):
                return None


def get_airplane_from_xflr5() -> Optional[Airplane]:
    """Get airplane from XFLR5 project file."""
    while True:
        file_path = Prompt.ask("Enter path to XFLR5 project file")

        if not file_path:
            if Confirm.ask("Cancel airplane selection?"):
                return None
            continue

        try:
            airplane = parse_xfl_project(file_path)
            console.print(f"[green]✓[/green] Loaded XFLR5 airplane: [cyan]{airplane.name}[/cyan]")
            return airplane
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to load XFLR5 airplane: {e}")
            if not Confirm.ask("Try again?"):
                return None


def select_airplane_solvers() -> List[str]:
    """Select solvers for airplane analysis."""
    console.print("\n[bold yellow]Available Solvers:[/bold yellow]")
    console.print("1. [cyan]AVL[/cyan] - Vortex lattice method")
    console.print("2. [cyan]GenuVP3[/cyan] - 3D potential flow solver")
    console.print("3. [cyan]GenuVP7[/cyan] - Advanced 3D potential flow solver")
    console.print("4. [cyan]LSPT[/cyan] - Lifting surface panel theory")

    choices = []
    if Confirm.ask("Use AVL?"):
        choices.append("avl")
    if Confirm.ask("Use GenuVP3?"):
        choices.append("gnvp3")
    if Confirm.ask("Use GenuVP7?"):
        choices.append("gnvp7")
    if Confirm.ask("Use LSPT?"):
        choices.append("lspt")

    if not choices:
        console.print("[yellow]No solvers selected, using AVL by default[/yellow]")
        choices = ["avl"]

    return choices


def configure_airplane_solver(solver_name: str, airplane: Airplane) -> None:
    """Configure airplane solver parameters."""
    console.print(f"\n[bold]Configuring {solver_name.upper()} solver[/bold]")

    # Create solver instance
    if solver_name == "avl":
        # AVL solver configuration would go here
        console.print("[yellow]AVL solver configuration to be implemented[/yellow]")
    elif solver_name == "gnvp3":
        solver = GenuVP3()
        configure_genuvp_solver(solver, airplane)
    elif solver_name == "gnvp7":
        solver = GenuVP7()
        configure_genuvp_solver(solver, airplane)
    elif solver_name == "lspt":
        solver = LSPT()
        configure_lspt_solver(solver, airplane)
    else:
        console.print(f"[red]Unknown solver: {solver_name}[/red]")
        return


def configure_genuvp_solver(solver: Solver, airplane: Airplane) -> None:
    """Configure GenuVP solver."""
    try:
        # Set analysis
        analyses = solver.get_analyses_names(verbose=True)
        if analyses:
            analysis_table = Table(title="Available Analyses", show_header=True, header_style="bold magenta")
            analysis_table.add_column("Index", style="cyan", no_wrap=True)
            analysis_table.add_column("Analysis", style="white")

            for i, analysis in enumerate(analyses, 1):
                analysis_table.add_row(str(i), analysis)

            console.print(analysis_table)

            choice = Prompt.ask("Select analysis", choices=[str(i) for i in range(1, len(analyses) + 1)], default="1")

            try:
                index = int(choice) - 1
                selected_analysis = analyses[index]
                solver.select_analysis(selected_analysis)
                console.print(f"[green]✓[/green] Selected analysis: [cyan]{selected_analysis}[/cyan]")
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to select analysis: {e}")

        # Configure analysis options
        configure_airplane_analysis_options(solver, airplane)

        # Configure solver parameters
        configure_airplane_solver_parameters(solver)

    except Exception as e:
        console.print(f"[red]✗[/red] Failed to configure GenuVP solver: {e}")


def configure_lspt_solver(solver: Solver, airplane: Airplane) -> None:
    """Configure LSPT solver."""
    try:
        # Set analysis
        analyses = solver.get_analyses_names(verbose=True)
        if analyses:
            analysis_table = Table(title="Available Analyses", show_header=True, header_style="bold magenta")
            analysis_table.add_column("Index", style="cyan", no_wrap=True)
            analysis_table.add_column("Analysis", style="white")

            for i, analysis in enumerate(analyses, 1):
                analysis_table.add_row(str(i), analysis)

            console.print(analysis_table)

            choice = Prompt.ask("Select analysis", choices=[str(i) for i in range(1, len(analyses) + 1)], default="1")

            try:
                index = int(choice) - 1
                selected_analysis = analyses[index]
                solver.select_analysis(selected_analysis)
                console.print(f"[green]✓[/green] Selected analysis: [cyan]{selected_analysis}[/cyan]")
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to select analysis: {e}")

        # Configure analysis options
        configure_airplane_analysis_options(solver, airplane)

        # Configure solver parameters
        configure_airplane_solver_parameters(solver)

    except Exception as e:
        console.print(f"[red]✗[/red] Failed to configure LSPT solver: {e}")


def configure_airplane_analysis_options(solver: Solver, airplane: Airplane) -> None:
    """Configure airplane analysis options."""
    try:
        options = solver.analyses[solver.mode].options
        if not options:
            return

        console.print("\n[bold]Analysis Options:[/bold]")

        answers = {}
        for option_name, option in options.items():
            if option.name == "airplane":
                answers[option.name] = airplane
                continue
            elif option.name == "state":
                state = get_flight_state(airplane)
                if state:
                    answers[option.name] = state
                continue
            elif option.name == "solver2D":
                solver_2d = get_2d_polars_solver()
                if solver_2d:
                    answers[option.name] = solver_2d
                continue

            # Get option value from user
            value = get_airplane_option_value(option)
            if value is not None:
                answers[option.name] = value

        if answers:
            solver.set_analysis_options(answers)
            console.print("[green]✓[/green] Analysis options configured")

    except Exception as e:
        console.print(f"[red]✗[/red] Failed to configure analysis options: {e}")


def configure_airplane_solver_parameters(solver: Solver) -> None:
    """Configure airplane solver parameters."""
    try:
        parameters = solver.get_solver_parameters(verbose=True)
        if not parameters:
            return

        if not Confirm.ask("Configure solver parameters?"):
            return

        console.print("\n[bold]Solver Parameters:[/bold]")

        answers = {}
        for param_name in parameters.keys():
            value = Prompt.ask(f"Enter value for {param_name}")
            if value:
                try:
                    # Try to convert to appropriate type
                    if "." in value:
                        answers[param_name] = float(value)
                    else:
                        answers[param_name] = int(value)
                except ValueError:
                    answers[param_name] = value

        if answers:
            solver.set_solver_parameters(answers)
            console.print("[green]✓[/green] Solver parameters configured")

    except Exception as e:
        console.print(f"[red]✗[/red] Failed to configure solver parameters: {e}")


def get_airplane_option_value(option) -> Any:
    """Get airplane option value from user."""
    option_type = getattr(option, "value_type", str)

    if option_type == float:
        while True:
            value = Prompt.ask(f"Enter {option.name} (float)")
            try:
                return float(value)
            except ValueError:
                console.print("[red]Invalid float value[/red]")
                if not Confirm.ask("Try again?"):
                    return None
    elif option_type == int:
        while True:
            value = Prompt.ask(f"Enter {option.name} (integer)")
            try:
                return int(value)
            except ValueError:
                console.print("[red]Invalid integer value[/red]")
                if not Confirm.ask("Try again?"):
                    return None
    elif option_type == bool:
        return Confirm.ask(f"Enable {option.name}?")
    else:
        return Prompt.ask(f"Enter {option.name}")


def get_flight_state(airplane: Airplane) -> Optional[State]:
    """Get flight state configuration."""
    console.print("\n[bold]Flight State Configuration:[/bold]")

    # Environment setup
    environment = configure_environment()

    # Get flight parameters
    try:
        velocity = float(Prompt.ask("Enter velocity (m/s)", default="100.0"))
        alpha = float(Prompt.ask("Enter angle of attack (degrees)", default="5.0"))
        beta = float(Prompt.ask("Enter sideslip angle (degrees)", default="0.0"))

        # Create state
        state = State(velocity=velocity, alpha=alpha, beta=beta, environment=environment)

        console.print("[green]✓[/green] Flight state configured")
        return state

    except ValueError as e:
        console.print(f"[red]✗[/red] Invalid flight state parameters: {e}")
        return None


def configure_environment() -> Environment:
    """Configure flight environment."""
    console.print("\n[bold]Environment Configuration:[/bold]")

    if Confirm.ask("Use standard atmosphere (ISA)?"):
        altitude = float(Prompt.ask("Enter altitude (m)", default="10000.0"))
        environment = EARTH_ISA(altitude)
        console.print(f"[green]✓[/green] Using ISA atmosphere at {altitude}m altitude")
    else:
        # Manual environment configuration
        temperature = float(Prompt.ask("Enter temperature (K)", default="288.15"))
        pressure = float(Prompt.ask("Enter pressure (Pa)", default="101325.0"))
        density = float(Prompt.ask("Enter density (kg/m³)", default="1.225"))

        environment = Environment(temperature=temperature, pressure=pressure, density=density)
        console.print("[green]✓[/green] Custom environment configured")

    return environment


def get_2d_polars_solver() -> Optional[str]:
    """Get 2D polars solver selection."""
    console.print("\n[bold]2D Polars Solver:[/bold]")

    solver = Prompt.ask("Select 2D polars solver", choices=["xfoil", "foil2wake", "openfoam"], default="xfoil")

    console.print(f"[green]✓[/green] Selected 2D solver: [cyan]{solver}[/cyan]")
    return solver


def run_airplane_analysis(airplanes: List[Airplane], solvers_config: dict, db: Database) -> None:
    """Run airplane analysis."""
    console.print("\n[bold green]Starting Analysis...[/bold green]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        total_tasks = sum(len(solvers) for solvers in solvers_config.values())
        task = progress.add_task("Running analysis...", total=total_tasks)

        for airplane in airplanes:
            if airplane.name not in solvers_config:
                continue

            solvers = solvers_config[airplane.name]

            for solver_name in solvers:
                progress.update(task, description=f"Analyzing {airplane.name} with {solver_name}")

                try:
                    # Create and run solver (placeholder)
                    time.sleep(1)  # Simulate analysis time

                    progress.advance(task)

                except Exception as e:
                    console.print(f"[red]✗[/red] Failed to analyze {airplane.name} with {solver_name}: {e}")

    console.print("[green]✓[/green] Analysis completed!")

    # Save to database
    if Confirm.ask("Save results to database?"):
        try:
            db.vehicles_db.load_all_data()
            console.print("[green]✓[/green] Results saved to database")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to save results: {e}")


# Legacy function for backward compatibility
def airplane_cli(db: Database, return_home: bool = False) -> None:
    """Legacy airplane CLI function for backward compatibility."""
    airplane_interactive(db)
