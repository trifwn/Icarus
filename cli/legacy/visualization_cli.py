"""ICARUS CLI Visualization module - Rich-based Interface

This module provides a modern CLI for visualizing analysis results using Rich and Typer.
It allows users to visualize airfoil and airplane analysis results with various plotting options.
"""

from typing import Any, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from ICARUS.database import Database
from ICARUS.visualization.airfoil import __functions__ as airfoil_functions

# Console for rich output
console = Console()


def visualization_interactive(db: Database) -> None:
    """Interactive visualization mode."""
    console.print(Panel.fit("[bold blue]Results Visualization[/bold blue]", border_style="blue", padding=(1, 2)))

    # Ask what to visualize
    vis_category = Prompt.ask("What do you want to visualize?", choices=["airfoil", "airplane"], default="airfoil")

    if vis_category == "airfoil":
        airfoil_visualization_interactive()
    elif vis_category == "airplane":
        airplane_visualization_interactive()
    else:
        console.print(f"[red]Unknown visualization category: {vis_category}[/red]")


def airfoil_visualization_interactive() -> None:
    """Interactive airfoil visualization."""
    console.print("\n[bold cyan]Airfoil Visualization[/bold cyan]")

    # Get available visualization functions
    functions = airfoil_functions

    if not functions:
        console.print("[yellow]No airfoil visualization functions available[/yellow]")
        return

    # Create selection table
    table = Table(title="Available Visualization Functions", show_header=True, header_style="bold magenta")
    table.add_column("Index", style="cyan", no_wrap=True)
    table.add_column("Function", style="white")
    table.add_column("Description", style="dim")

    for i, func in enumerate(functions, 1):
        # Get function description from docstring
        description = getattr(func, "__doc__", "No description available")
        if description:
            description = description.strip().split("\n")[0]  # First line only
        else:
            description = "No description available"

        table.add_row(str(i), func.__name__, description)

    console.print(table)

    # Select functions
    selected_functions = []
    while True:
        choice = Prompt.ask(
            "Select function by number (or 'done' to finish)",
            choices=[str(i) for i in range(1, len(functions) + 1)] + ["done"],
        )

        if choice == "done":
            break

        try:
            index = int(choice) - 1
            selected_func = functions[index]
            if selected_func not in selected_functions:
                selected_functions.append(selected_func)
                console.print(f"[green]✓[/green] Added: [cyan]{selected_func.__name__}[/cyan]")
            else:
                console.print(f"[yellow]Function already selected[/yellow]")
        except (ValueError, IndexError):
            console.print("[red]Invalid selection[/red]")

    if not selected_functions:
        console.print("[yellow]No functions selected[/yellow]")
        return

    # Execute selected functions
    console.print(f"\n[bold]Executing {len(selected_functions)} visualization function(s)...[/bold]")

    for func in selected_functions:
        console.print(f"\n[bold]Running:[/bold] {func.__name__}")

        try:
            # Get function arguments
            import inspect

            sig = inspect.signature(func)
            args = []
            kwargs = {}

            for param_name, param in sig.parameters.items():
                if param.default == inspect.Parameter.empty:
                    # Required parameter
                    if param.annotation == str:
                        value = Prompt.ask(f"Enter {param_name}")
                    elif param.annotation == float:
                        value = float(Prompt.ask(f"Enter {param_name} (float)"))
                    elif param.annotation == int:
                        value = int(Prompt.ask(f"Enter {param_name} (integer)"))
                    elif param.annotation == bool:
                        value = Confirm.ask(f"Enable {param_name}?")
                    else:
                        value = Prompt.ask(f"Enter {param_name}")
                    args.append(value)
                else:
                    # Optional parameter
                    if Confirm.ask(f"Set {param_name}?"):
                        if param.annotation == str:
                            value = Prompt.ask(f"Enter {param_name}")
                        elif param.annotation == float:
                            value = float(Prompt.ask(f"Enter {param_name} (float)"))
                        elif param.annotation == int:
                            value = int(Prompt.ask(f"Enter {param_name} (integer)"))
                        elif param.annotation == bool:
                            value = Confirm.ask(f"Enable {param_name}?")
                        else:
                            value = Prompt.ask(f"Enter {param_name}")
                        kwargs[param_name] = value

            # Execute function
            result = func(*args, **kwargs)
            console.print(f"[green]✓[/green] {func.__name__} completed successfully")

            # Save plot if requested
            if Confirm.ask(f"Save plot from {func.__name__}?"):
                save_path = Prompt.ask("Enter save path (e.g., plot.png)")
                if save_path:
                    # This would need to be implemented based on the actual plotting library
                    console.print(f"[yellow]Save functionality to be implemented for {save_path}[/yellow]")

        except Exception as e:
            console.print(f"[red]✗[/red] Failed to execute {func.__name__}: {e}")


def airplane_visualization_interactive() -> None:
    """Interactive airplane visualization."""
    console.print("\n[bold cyan]Airplane Visualization[/bold cyan]")

    # This would be implemented similar to airfoil visualization
    # but with airplane-specific visualization functions

    console.print("[yellow]Airplane visualization to be implemented[/yellow]")

    # Placeholder for airplane visualization functions
    plot_types = ["polar", "geometry", "pressure", "forces"]

    table = Table(title="Available Plot Types", show_header=True, header_style="bold magenta")
    table.add_column("Index", style="cyan", no_wrap=True)
    table.add_column("Plot Type", style="white")
    table.add_column("Description", style="dim")

    descriptions = {
        "polar": "Aerodynamic coefficients vs angle of attack",
        "geometry": "3D airplane geometry visualization",
        "pressure": "Surface pressure distribution",
        "forces": "Force and moment analysis",
    }

    for i, plot_type in enumerate(plot_types, 1):
        description = descriptions.get(plot_type, "No description available")
        table.add_row(str(i), plot_type, description)

    console.print(table)

    # Select plot type
    choice = Prompt.ask(
        "Select plot type by number", choices=[str(i) for i in range(1, len(plot_types) + 1)], default="1"
    )

    try:
        index = int(choice) - 1
        selected_plot = plot_types[index]
        console.print(f"[green]✓[/green] Selected: [cyan]{selected_plot}[/cyan]")

        # Get airplane selection
        airplane_name = Prompt.ask("Enter airplane name")
        if airplane_name:
            console.print(f"[yellow]Plotting {selected_plot} for {airplane_name}...[/yellow]")
            console.print("[yellow]Plot generation to be implemented[/yellow]")

            # Save plot if requested
            if Confirm.ask("Save plot?"):
                save_path = Prompt.ask("Enter save path (e.g., airplane_plot.png)")
                if save_path:
                    console.print(f"[yellow]Save functionality to be implemented for {save_path}[/yellow]")

    except (ValueError, IndexError):
        console.print("[red]Invalid selection[/red]")


# Legacy function for backward compatibility
def visualization_cli(DB: Database, return_home: bool = True) -> None:
    """Legacy visualization CLI function for backward compatibility."""
    visualization_interactive(DB)


# Legacy function for backward compatibility
def aifoil_visualization_cli() -> None:
    """Legacy airfoil visualization CLI function for backward compatibility."""
    airfoil_visualization_interactive()


if __name__ == "__main__":
    visualization_cli(DB=Database("./Data"))
