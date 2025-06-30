#!/usr/bin/env python3
"""ICARUS CLI - Unified Entry Point

This is the main entry point for the ICARUS CLI that provides:
- Enhanced CLI mode with rich interface
- TUI mode with interactive interface
- Command-line interface for scripting
- Seamless switching between modes
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.align import Align

# Add cli directory to path for imports
cli_dir = Path(__file__).parent
sys.path.insert(0, str(cli_dir))

# Import core modules
try:
    from core.state import session_manager, config_manager, history_manager
    from core.ui import theme_manager, notification_system, ui_components
    from core.workflow import workflow_engine, template_manager
    from core.services import validation_service, export_service
except ImportError as e:
    print(f"Error importing core modules: {e}")
    print("Please ensure all dependencies are installed:")
    print("  pip install rich typer")
    sys.exit(1)

# Import ICARUS modules
try:
    from ICARUS import __version__
    from ICARUS.database import Database
except ImportError:
    __version__ = "2.0.0"
    Database = None

# Create the main app
app = typer.Typer(
    name="icarus",
    help="ICARUS Aerodynamics v2.0 - Advanced Aircraft Design and Analysis",
    add_completion=False,
    rich_markup_mode="rich",
)

# Create sub-apps
airfoil_app = typer.Typer(name="airfoil", help="2D Airfoil Analysis")
airplane_app = typer.Typer(name="airplane", help="3D Airplane Analysis")
visualization_app = typer.Typer(name="visualization", help="Results Visualization")
workflow_app = typer.Typer(name="workflow", help="Workflow Management")

# Add sub-apps to main app
app.add_typer(airfoil_app, help="2D Airfoil Analysis")
app.add_typer(airplane_app, help="3D Airplane Analysis")
app.add_typer(visualization_app, help="Results Visualization")
app.add_typer(workflow_app, help="Workflow Management")

# Console for rich output
console = Console()


class ICARUSCLI:
    """Main CLI controller with enhanced functionality."""

    def __init__(self):
        self.db: Optional[Database] = None
        self.current_screen = "main"
        self.screen_history = []
        self._initialize_cli()

    def _initialize_cli(self):
        """Initialize the CLI with configuration and database."""
        # Load configuration
        database_path = config_manager.get_database_path()

        # Initialize database
        try:
            if Database:
                self.db = Database(database_path)
                notification_system.success(f"Database initialized: {database_path}")
            else:
                notification_system.warning("ICARUS database not available")
        except Exception as e:
            notification_system.error(f"Database initialization failed: {e}")
            # Don't exit, continue without database

        # Set theme from config
        theme_name = config_manager.get("theme", "default")
        try:
            theme_manager.apply_theme(theme_name)
        except Exception:
            notification_system.warning(f"Unknown theme: {theme_name}, using default")

    def show_banner(self):
        """Display the enhanced ICARUS banner."""
        banner_text = Text("ICARUS AERODYNAMICS", style=f"bold {theme_manager.get_color('primary')}")
        subtitle = Text("Advanced Aircraft Design & Analysis Platform", style=theme_manager.get_color("secondary"))
        version_text = Text(f"v{__version__}", style=theme_manager.get_color("muted"))

        banner = Panel(
            Align.center(f"{banner_text}\n{subtitle}\n{version_text}"),
            border_style=theme_manager.get_color("primary"),
            padding=(2, 4),
            title="[bold white]Next Generation Aircraft Design[/bold white]",
            subtitle=f"[dim]Powered by Advanced Computational Methods[/dim]",
        )
        console.print(banner)

    def show_session_info(self):
        """Display current session information."""
        try:
            session_info = session_manager.get_session_info()

            info_table = Table(show_header=False, box=None, padding=(0, 1))
            info_table.add_column("Property", style=theme_manager.get_color("text"))
            info_table.add_column("Value", style=theme_manager.get_color("accent"))

            for key, value in session_info.items():
                info_table.add_row(key.replace("_", " ").title(), str(value))

            info_panel = Panel(
                str(info_table), title="Session Information", border_style=theme_manager.get_color("secondary")
            )
            console.print(info_panel)
        except Exception as e:
            notification_system.error(f"Failed to show session info: {e}")

    def show_main_menu(self):
        """Display the enhanced main menu."""
        menu_options = [
            {"label": "🚁 2D Airfoil Analysis", "description": "Analyze airfoils with multiple solvers"},
            {"label": "✈️ 3D Airplane Analysis", "description": "Perform 3D aerodynamic analysis"},
            {"label": "📊 Visualization", "description": "Visualize results and create plots"},
            {"label": "⚙️ Workflow Management", "description": "Manage and execute workflows"},
            {"label": "🔧 Settings", "description": "Configure CLI preferences"},
            {"label": "📚 Help & Documentation", "description": "Access help and examples"},
            {"label": "🖥️ Launch TUI Mode", "description": "Switch to interactive Textual UI"},
            {"label": "🐍 IPython Shell", "description": "Drop into interactive Python shell"},
            {"label": "🚪 Exit", "description": "Exit the application"},
        ]

        # Create menu table
        menu_table = Table(
            title="Main Menu", show_header=True, header_style=f"bold {theme_manager.get_color('primary')}"
        )
        menu_table.add_column("Option", style=theme_manager.get_color("text"), no_wrap=True)
        menu_table.add_column("Description", style=theme_manager.get_color("muted"))

        for i, option in enumerate(menu_options, 1):
            menu_table.add_row(f"{i}. {option['label']}", option["description"])

        console.print(menu_table)

        # Get user choice
        choice = Prompt.ask("Select an option", choices=[str(i) for i in range(1, len(menu_options) + 1)], default="9")

        return int(choice)

    def handle_main_menu_choice(self, choice: int):
        """Handle main menu selection."""
        if choice == 1:
            self.show_airfoil_menu()
        elif choice == 2:
            self.show_airplane_menu()
        elif choice == 3:
            self.show_visualization_menu()
        elif choice == 4:
            self.show_workflow_menu()
        elif choice == 5:
            self.show_settings_menu()
        elif choice == 6:
            self.show_help_menu()
        elif choice == 7:
            self.launch_tui_mode()
        elif choice == 8:
            self.launch_ipython_shell()
        elif choice == 9:
            self.exit_cli()

    def launch_tui_mode(self):
        """Launch the Textual TUI application."""
        console.print(
            f"\n[{theme_manager.get_color('info')}]Launching ICARUS TUI Mode...[/{theme_manager.get_color('info')}]"
        )
        console.print(
            f"[{theme_manager.get_color('muted')}]Press Ctrl+C to return to CLI mode[/{theme_manager.get_color('muted')}]\n"
        )

        try:
            # Import and run the TUI app
            from tui_app import ICARUSTUI

            # Create and run the TUI app
            app = ICARUSTUI()
            app.run()

            # When TUI exits, return to CLI
            console.print(
                f"\n[{theme_manager.get_color('success')}]Returned to CLI mode[/{theme_manager.get_color('success')}]"
            )

        except ImportError as e:
            notification_system.error(f"Failed to launch TUI: {e}")
            console.print(
                f"[{theme_manager.get_color('error')}]TUI mode not available. Install textual: pip install textual[/{theme_manager.get_color('error')}]"
            )
        except KeyboardInterrupt:
            console.print(
                f"\n[{theme_manager.get_color('info')}]TUI mode interrupted, returning to CLI[/{theme_manager.get_color('info')}]"
            )
        except Exception as e:
            notification_system.error(f"TUI mode failed: {e}")
            console.print(
                f"[{theme_manager.get_color('error')}]TUI mode encountered an error: {e}[/{theme_manager.get_color('error')}]"
            )

    def launch_ipython_shell(self):
        """Launch IPython shell with session context."""
        try:
            from IPython import embed

            # Prepare namespace with current session data
            namespace = {
                "db": self.db,
                "session": session_manager.current_session,
                "config": config_manager.config,
                "workflow_engine": workflow_engine,
                "template_manager": template_manager,
                "validation_service": validation_service,
                "export_service": export_service,
            }

            notification_system.info("Launching IPython shell. Type 'exit' or Ctrl-D to return.")
            embed(user_ns=namespace)

        except ImportError:
            notification_system.error("IPython is not installed. Install with 'pip install ipython'")

    def exit_cli(self):
        """Exit the CLI with confirmation."""
        if config_manager.get("confirm_exit", True):
            if not Confirm.ask("Are you sure you want to exit?"):
                return

        notification_system.info("Saving session state...")
        try:
            session_manager._save_session()
        except Exception as e:
            notification_system.warning(f"Failed to save session: {e}")

        notification_system.success("Thank you for using ICARUS Aerodynamics!")
        raise typer.Exit()

    def run_interactive(self):
        """Run the interactive CLI mode."""
        while True:
            try:
                console.clear()
                self.show_banner()

                if config_manager.get_ui_config().get("show_session_info", True):
                    self.show_session_info()

                choice = self.show_main_menu()
                self.handle_main_menu_choice(choice)

            except KeyboardInterrupt:
                if Confirm.ask("\nExit ICARUS CLI?"):
                    self.exit_cli()
            except Exception as e:
                notification_system.error(f"Unexpected error: {e}")
                if not Confirm.ask("Continue?"):
                    self.exit_cli()

    # Menu implementations (simplified for brevity)
    def show_airfoil_menu(self):
        """Display the enhanced airfoil analysis menu."""
        notification_system.info("Airfoil analysis menu - use 'icarus airfoil' commands")

    def show_airplane_menu(self):
        """Display the enhanced airplane analysis menu."""
        notification_system.info("Airplane analysis menu - use 'icarus airplane' commands")

    def show_visualization_menu(self):
        """Display the enhanced visualization menu."""
        notification_system.info("Visualization menu - use 'icarus visualization' commands")

    def show_workflow_menu(self):
        """Display the workflow management menu."""
        notification_system.info("Workflow management menu - use 'icarus workflow' commands")

    def show_settings_menu(self):
        """Display the settings menu."""
        notification_system.info("Settings menu - configuration options")

    def show_help_menu(self):
        """Display the help and documentation menu."""
        notification_system.info("Help menu - documentation and examples")


# Global CLI instance
cli = ICARUSCLI()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(None, "--version", "-v", help="Show version and exit"),
    database_path: str = typer.Option(None, "--database", "-d", help="Path to ICARUS database"),
    theme: str = typer.Option(None, "--theme", "-t", help="CLI theme (default, dark, light, aerospace, scientific)"),
):
    """ICARUS Aerodynamics v2.0 - Unified CLI with state management and workflow automation."""
    if version:
        console.print(
            f"[bold {theme_manager.get_color('primary')}]ICARUS[/bold {theme_manager.get_color('primary')}] version [green]{__version__}[/green]"
        )
        raise typer.Exit()

    if database_path:
        config_manager.set("database_path", database_path)

    if theme:
        theme_manager.apply_theme(theme)
        config_manager.set("theme", theme)


@app.command()
def interactive():
    """Launch interactive CLI mode."""
    cli.run_interactive()


@app.command()
def tui():
    """Launch TUI mode directly."""
    cli.launch_tui_mode()


# Airfoil commands
@airfoil_app.command()
def analyze(
    airfoil: str = typer.Argument(..., help="Airfoil name or file path"),
    solver: str = typer.Option("xfoil", "--solver", "-s", help="Solver to use"),
    angles: str = typer.Option("0:15:16", "--angles", "-a", help="Angle of attack range"),
    reynolds: float = typer.Option(1e6, "--reynolds", "-r", help="Reynolds number"),
):
    """Analyze an airfoil using the specified solver."""
    notification_system.info(f"Analyzing airfoil {airfoil} with {solver}")
    # TODO: Implement actual analysis
    notification_system.success("Analysis completed")


# Airplane commands
@airplane_app.command()
def analyze(
    airplane: str = typer.Argument(..., help="Airplane name or file path"),
    solver: str = typer.Option("avl", "--solver", "-s", help="Solver to use"),
    state: str = typer.Option("cruise", "--state", help="Flight state"),
):
    """Analyze an airplane using the specified solver."""
    notification_system.info(f"Analyzing airplane {airplane} with {solver}")
    # TODO: Implement actual analysis
    notification_system.success("Analysis completed")


# Visualization commands
@visualization_app.command()
def polar(
    airfoil: str = typer.Argument(..., help="Airfoil name"),
    reynolds: float = typer.Option(None, "--reynolds", "-r", help="Reynolds number"),
):
    """Plot airfoil polar."""
    notification_system.info(f"Plotting polar for {airfoil}")
    # TODO: Implement visualization
    notification_system.success("Plot generated")


# Workflow commands
@workflow_app.command()
def list():
    """List available workflows."""
    try:
        workflows = workflow_engine.get_workflows()

        if not workflows:
            notification_system.info("No workflows available")
            return

        workflow_table = Table(
            title="Available Workflows", show_header=True, header_style=f"bold {theme_manager.get_color('primary')}"
        )
        workflow_table.add_column("Name", style=theme_manager.get_color("text"))
        workflow_table.add_column("Type", style=theme_manager.get_color("secondary"))
        workflow_table.add_column("Description", style=theme_manager.get_color("muted"))

        for workflow in workflows:
            workflow_table.add_row(
                workflow.name,
                workflow.type.value.replace("_", " ").title(),
                workflow.description,
            )

        console.print(workflow_table)
    except Exception as e:
        notification_system.error(f"Failed to list workflows: {e}")


if __name__ == "__main__":
    app()
