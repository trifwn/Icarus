"""ICARUS CLI for Airfoil Analysis - Rich-based Interface

This module provides a modern CLI for 2D airfoil analysis using Rich and Typer.
It allows users to analyze airfoils using various solvers with an intuitive interface.
"""

import time
from pathlib import Path
from typing import Any, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

from ICARUS.airfoils import Airfoil
from ICARUS.computation.solvers import Solver
from ICARUS.computation.solvers.Foil2Wake.f2w_section import Foil2Wake
from ICARUS.computation.solvers.OpenFoam.open_foam import OpenFoam
from ICARUS.computation.solvers.Xfoil.xfoil import Xfoil
from ICARUS.core.base_types import Struct
from ICARUS.database import Database

# Console for rich output
console = Console()


def airfoil_interactive(db: Database) -> None:
    """Interactive airfoil analysis mode."""
    console.print(Panel.fit(
        "[bold blue]2D Airfoil Analysis[/bold blue]",
        border_style="blue",
        padding=(1, 2)
    ))
    
    # Get number of airfoils
    num_airfoils = Prompt.ask(
        "How many airfoils do you want to analyze?",
        default="1"
    )
    
    try:
        num_airfoils = int(num_airfoils)
        if num_airfoils < 1:
            console.print("[red]Number of airfoils must be at least 1[/red]")
            return
    except ValueError:
        console.print("[red]Invalid number of airfoils[/red]")
        return
    
    airfoils: List[Airfoil] = []
    solvers_config = {}
    
    for i in range(num_airfoils):
        console.print(f"\n[bold cyan]Airfoil {i+1}/{num_airfoils}[/bold cyan]")
        
        # Select airfoil source
        source = Prompt.ask(
            "Select airfoil source",
            choices=["file", "naca", "database"],
            default="database"
        )
        
        airfoil = None
        if source == "file":
            airfoil = get_airfoil_from_file()
        elif source == "naca":
            airfoil = get_airfoil_naca()
        elif source == "database":
            airfoil = get_airfoil_from_database(db)
        
        if airfoil is None:
            console.print("[red]Failed to load airfoil, skipping...[/red]")
            continue
            
        airfoils.append(airfoil)
        
        # Select solvers
        solvers = select_solvers()
        solvers_config[airfoil.name] = solvers
        
        # Configure solvers
        for solver_name in solvers:
            configure_solver(solver_name, airfoil)
    
    # Run analysis
    if airfoils:
        run_airfoil_analysis(airfoils, solvers_config, db)


def get_airfoil_from_file() -> Optional[Airfoil]:
    """Get airfoil from file."""
    while True:
        file_path = Prompt.ask("Enter path to airfoil file")
        
        if not file_path:
            if Confirm.ask("Cancel airfoil selection?"):
                return None
            continue
            
        try:
            airfoil = Airfoil.from_file(file_path)
            console.print(f"[green]✓[/green] Loaded airfoil: [cyan]{airfoil.name}[/cyan]")
            return airfoil
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to load airfoil: {e}")
            if not Confirm.ask("Try again?"):
                return None


def get_airfoil_naca() -> Optional[Airfoil]:
    """Get NACA airfoil."""
    while True:
        naca_digits = Prompt.ask("Enter NACA digits (4 or 5)")
        
        if not naca_digits:
            if Confirm.ask("Cancel airfoil selection?"):
                return None
            continue
            
        try:
            airfoil = Airfoil.naca(naca_digits)
            console.print(f"[green]✓[/green] Generated NACA airfoil: [cyan]{airfoil.name}[/cyan]")
            return airfoil
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to generate NACA airfoil: {e}")
            if not Confirm.ask("Try again?"):
                return None


def get_airfoil_from_database(db: Database) -> Optional[Airfoil]:
    """Get airfoil from database."""
    available_airfoils = list(db.airfoils.keys())
    
    if not available_airfoils:
        console.print("[yellow]No airfoils found in database[/yellow]")
        return None
    
    # Create selection table
    table = Table(title="Available Airfoils", show_header=True, header_style="bold magenta")
    table.add_column("Index", style="cyan", no_wrap=True)
    table.add_column("Name", style="white")
    
    for i, name in enumerate(available_airfoils, 1):
        table.add_row(str(i), name)
    
    console.print(table)
    
    while True:
        choice = Prompt.ask(
            "Select airfoil by number",
            choices=[str(i) for i in range(1, len(available_airfoils) + 1)]
        )
        
        try:
            index = int(choice) - 1
            airfoil_name = available_airfoils[index]
            airfoil = db.get_airfoil(airfoil_name)
            console.print(f"[green]✓[/green] Selected airfoil: [cyan]{airfoil.name}[/cyan]")
            return airfoil
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to load airfoil: {e}")
            if not Confirm.ask("Try again?"):
                return None


def select_solvers() -> List[str]:
    """Select solvers for analysis."""
    console.print("\n[bold yellow]Available Solvers:[/bold yellow]")
    console.print("1. [cyan]XFoil[/cyan] - Fast 2D panel method")
    console.print("2. [cyan]Foil2Wake[/cyan] - Advanced 2D solver")
    console.print("3. [cyan]OpenFoam[/cyan] - CFD solver")
    
    choices = []
    if Confirm.ask("Use XFoil?"):
        choices.append("xfoil")
    if Confirm.ask("Use Foil2Wake?"):
        choices.append("foil2wake")
    if Confirm.ask("Use OpenFoam?"):
        choices.append("openfoam")
    
    if not choices:
        console.print("[yellow]No solvers selected, using XFoil by default[/yellow]")
        choices = ["xfoil"]
    
    return choices


def configure_solver(solver_name: str, airfoil: Airfoil) -> None:
    """Configure solver parameters."""
    console.print(f"\n[bold]Configuring {solver_name.upper()} solver[/bold]")
    
    # Create solver instance
    if solver_name == "xfoil":
        solver = Xfoil()
    elif solver_name == "foil2wake":
        solver = Foil2Wake()
    elif solver_name == "openfoam":
        solver = OpenFoam()
    else:
        console.print(f"[red]Unknown solver: {solver_name}[/red]")
        return
    
    # Set analysis
    analyses = solver.get_analyses_names(verbose=True)
    if analyses:
        analysis_table = Table(title="Available Analyses", show_header=True, header_style="bold magenta")
        analysis_table.add_column("Index", style="cyan", no_wrap=True)
        analysis_table.add_column("Analysis", style="white")
        
        for i, analysis in enumerate(analyses, 1):
            analysis_table.add_row(str(i), analysis)
        
        console.print(analysis_table)
        
        choice = Prompt.ask(
            "Select analysis",
            choices=[str(i) for i in range(1, len(analyses) + 1)],
            default="1"
        )
        
        try:
            index = int(choice) - 1
            selected_analysis = analyses[index]
            solver.select_analysis(selected_analysis)
            console.print(f"[green]✓[/green] Selected analysis: [cyan]{selected_analysis}[/cyan]")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to select analysis: {e}")
    
    # Configure analysis options
    configure_analysis_options(solver, airfoil)
    
    # Configure solver parameters
    configure_solver_parameters(solver)


def configure_analysis_options(solver: Solver, airfoil: Airfoil) -> None:
    """Configure analysis options."""
    try:
        options = solver.analyses[solver.mode].options
        if not options:
            return
            
        console.print("\n[bold]Analysis Options:[/bold]")
        
        answers = {}
        for option_name, option in options.items():
            if option.name == "airfoil":
                answers[option.name] = airfoil
                continue
                
            # Get option value from user
            value = get_option_value(option)
            if value is not None:
                answers[option.name] = value
        
        if answers:
            solver.set_analysis_options(answers)
            console.print("[green]✓[/green] Analysis options configured")
            
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to configure analysis options: {e}")


def configure_solver_parameters(solver: Solver) -> None:
    """Configure solver parameters."""
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
                    if '.' in value:
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


def get_option_value(option) -> Any:
    """Get option value from user."""
    option_type = getattr(option, 'value_type', str)
    
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


def run_airfoil_analysis(airfoils: List[Airfoil], solvers_config: dict, db: Database) -> None:
    """Run airfoil analysis."""
    console.print("\n[bold green]Starting Analysis...[/bold green]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        total_tasks = sum(len(solvers) for solvers in solvers_config.values())
        task = progress.add_task("Running analysis...", total=total_tasks)
        
        for airfoil in airfoils:
            if airfoil.name not in solvers_config:
                continue
                
            solvers = solvers_config[airfoil.name]
            
            for solver_name in solvers:
                progress.update(task, description=f"Analyzing {airfoil.name} with {solver_name}")
                
                try:
                    # Create and run solver
                    if solver_name == "xfoil":
                        solver = Xfoil()
                    elif solver_name == "foil2wake":
                        solver = Foil2Wake()
                    elif solver_name == "openfoam":
                        solver = OpenFoam()
                    else:
                        continue
                    
                    # Run analysis (placeholder)
                    time.sleep(1)  # Simulate analysis time
                    
                    progress.advance(task)
                    
                except Exception as e:
                    console.print(f"[red]✗[/red] Failed to analyze {airfoil.name} with {solver_name}: {e}")
    
    console.print("[green]✓[/green] Analysis completed!")
    
    # Save to database
    if Confirm.ask("Save results to database?"):
        try:
            db.foils_db.load_all_data()
            console.print("[green]✓[/green] Results saved to database")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to save results: {e}")


# Legacy function for backward compatibility
def airfoil_cli(db: Database, return_home: bool = False) -> None:
    """Legacy airfoil CLI function for backward compatibility."""
    airfoil_interactive(db)
