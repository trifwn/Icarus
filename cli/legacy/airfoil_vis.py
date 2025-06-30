"""Airfoil visualization using Rich interface."""

from typing import Any

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from ICARUS.visualization.airfoil import __functions__ as airfoil_functions

# Console for rich output
console = Console()


def ask_visualization_function(functions: list[Any]) -> list[Any]:
    """Ask user to select visualization functions using Rich interface."""
    if not functions:
        console.print("[yellow]No visualization functions available[/yellow]")
        return []

    # Create selection table
    table = Table(title="Available Visualization Functions", show_header=True, header_style="bold magenta")
    table.add_column("Index", style="cyan", no_wrap=True)
    table.add_column("Function", style="white")
    table.add_column("Description", style="dim")

    for i, func in enumerate(functions, 1):
        description = getattr(func, "__doc__", "No description available")
        if description:
            description = description.strip().split("\n")[0]
        else:
            description = "No description available"

        table.add_row(str(i), func.__name__, description)

    console.print(table)

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

    return selected_functions


def aifoil_visualization_cli() -> None:
    """Airfoil visualization CLI using Rich interface."""
    functions = airfoil_functions

    if not functions:
        console.print("[yellow]No airfoil visualization functions available[/yellow]")
        return

    # Get selected functions
    vis_functions = ask_visualization_function(functions=functions)

    if not vis_functions:
        console.print("[yellow]No functions selected[/yellow]")
        return

    # Execute selected functions
    console.print(f"\n[bold]Executing {len(vis_functions)} visualization function(s)...[/bold]")

    for func in vis_functions:
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
                        value = Prompt.ask(f"Enable {param_name}?", choices=["y", "n"]).lower() == "y"
                    else:
                        value = Prompt.ask(f"Enter {param_name}")
                    args.append(value)
                else:
                    # Optional parameter
                    if Prompt.ask(f"Set {param_name}?", choices=["y", "n"]).lower() == "y":
                        if param.annotation == str:
                            value = Prompt.ask(f"Enter {param_name}")
                        elif param.annotation == float:
                            value = float(Prompt.ask(f"Enter {param_name} (float)"))
                        elif param.annotation == int:
                            value = int(Prompt.ask(f"Enter {param_name} (integer)"))
                        elif param.annotation == bool:
                            value = Prompt.ask(f"Enable {param_name}?", choices=["y", "n"]).lower() == "y"
                        else:
                            value = Prompt.ask(f"Enter {param_name}")
                        kwargs[param_name] = value

            # Execute function
            result = func(*args, **kwargs)
            console.print(f"[green]✓[/green] {func.__name__} completed successfully")

        except Exception as e:
            console.print(f"[red]✗[/red] Failed to execute {func.__name__}: {e}")
