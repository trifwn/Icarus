"""UI Framework for ICARUS CLI

This module provides a comprehensive UI framework with theming, layout management,
progress tracking, and notification systems for a polished user experience.
"""

import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.live import Live
from rich.align import Align
from rich.columns import Columns
from rich.rule import Rule
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.syntax import Syntax

console = Console()


class Theme(Enum):
    """Available themes for the CLI."""

    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    AEROSPACE = "aerospace"
    SCIENTIFIC = "scientific"


@dataclass
class ThemeColors:
    """Color scheme for a theme."""

    primary: str
    secondary: str
    accent: str
    success: str
    warning: str
    error: str
    info: str
    muted: str
    background: str
    text: str


class ThemeManager:
    """Manages CLI theming and visual styling."""

    def __init__(self):
        self.themes = {
            Theme.DEFAULT: ThemeColors(
                primary="blue",
                secondary="cyan",
                accent="magenta",
                success="green",
                warning="yellow",
                error="red",
                info="blue",
                muted="dim",
                background="black",
                text="white",
            ),
            Theme.DARK: ThemeColors(
                primary="bright_blue",
                secondary="bright_cyan",
                accent="bright_magenta",
                success="bright_green",
                warning="bright_yellow",
                error="bright_red",
                info="bright_blue",
                muted="bright_black",
                background="black",
                text="bright_white",
            ),
            Theme.LIGHT: ThemeColors(
                primary="blue",
                secondary="cyan",
                accent="magenta",
                success="green",
                warning="yellow",
                error="red",
                info="blue",
                muted="black",
                background="white",
                text="black",
            ),
            Theme.AEROSPACE: ThemeColors(
                primary="bright_blue",
                secondary="cyan",
                accent="bright_white",
                success="bright_green",
                warning="bright_yellow",
                error="bright_red",
                info="bright_cyan",
                muted="bright_black",
                background="black",
                text="bright_white",
            ),
            Theme.SCIENTIFIC: ThemeColors(
                primary="bright_green",
                secondary="cyan",
                accent="bright_yellow",
                success="green",
                warning="yellow",
                error="red",
                info="blue",
                muted="dim",
                background="black",
                text="white",
            ),
        }
        self.current_theme = Theme.DEFAULT

    def set_theme(self, theme: Theme):
        """Set the current theme."""
        self.current_theme = theme

    def apply_theme(self, theme_name: str):
        """Apply a theme by name."""
        try:
            theme = Theme(theme_name)
            self.set_theme(theme)
            return True
        except ValueError:
            # If theme name is not found, use default
            self.set_theme(Theme.DEFAULT)
            return False

    def get_color(self, color_type: str) -> str:
        """Get color for the current theme."""
        theme_colors = self.themes[self.current_theme]
        return getattr(theme_colors, color_type, theme_colors.text)

    def style_text(self, text: str, style: str) -> str:
        """Apply theme styling to text."""
        color = self.get_color(style)
        return f"[{color}]{text}[/{color}]"

    def create_panel(self, content: str, title: str = None, border_style: str = None) -> Panel:
        """Create a themed panel."""
        if border_style is None:
            border_style = self.get_color("primary")

        return Panel(content, title=title, border_style=border_style, padding=(1, 2))


class LayoutManager:
    """Manages CLI layout and screen organization."""

    def __init__(self, theme_manager: ThemeManager):
        self.theme = theme_manager
        self.layout = Layout()
        self.layout.split_column(Layout(name="header", size=3), Layout(name="main"), Layout(name="footer", size=3))
        self.layout["main"].split_row(Layout(name="sidebar", size=30), Layout(name="content"))

    def create_header(self, title: str, subtitle: str = None) -> Panel:
        """Create a themed header."""
        header_content = f"[bold {self.theme.get_color('primary')}]{title}[/bold {self.theme.get_color('primary')}]"
        if subtitle:
            header_content += f"\n[{self.theme.get_color('muted')}]{subtitle}[/{self.theme.get_color('muted')}]"

        return self.theme.create_panel(header_content, border_style=self.theme.get_color("primary"))

    def create_sidebar(self, items: List[Dict[str, str]]) -> Panel:
        """Create a navigation sidebar."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Item", style=self.theme.get_color("text"))

        for item in items:
            icon = item.get("icon", "•")
            label = item.get("label", "")
            table.add_row(f"{icon} {label}")

        return self.theme.create_panel(table, title="Navigation", border_style=self.theme.get_color("secondary"))

    def create_content_area(self, content: str) -> Panel:
        """Create the main content area."""
        return self.theme.create_panel(content, title="Content", border_style=self.theme.get_color("accent"))

    def create_footer(self, status: str = None) -> Panel:
        """Create a status footer."""
        footer_content = f"[{self.theme.get_color('muted')}]Ready[/{self.theme.get_color('muted')}]"
        if status:
            footer_content = status

        return self.theme.create_panel(footer_content, border_style=self.theme.get_color("muted"))


class ProgressManager:
    """Manages progress tracking and display."""

    def __init__(self, theme_manager: ThemeManager):
        self.theme = theme_manager
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        )
        self.active_tasks = {}

    def start_task(self, task_id: str, description: str, total: int = 100) -> str:
        """Start a new progress task."""
        task = self.progress.add_task(description, total=total)
        self.active_tasks[task_id] = task
        return task_id

    def update_task(self, task_id: str, advance: int = 1, description: str = None):
        """Update a progress task."""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            self.progress.advance(task, advance)
            if description:
                self.progress.update(task, description=description)

    def complete_task(self, task_id: str):
        """Complete a progress task."""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            self.progress.update(task, completed=True)
            del self.active_tasks[task_id]

    def __enter__(self):
        """Context manager entry."""
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.progress.stop()


class NotificationSystem:
    """Manages notifications and user feedback."""

    def __init__(self, theme_manager: ThemeManager):
        self.theme = theme_manager
        self.notifications = []

    def success(self, message: str, title: str = "Success"):
        """Show a success notification."""
        self._add_notification(message, title, "success")
        console.print(f"[{self.theme.get_color('success')}]✓ {message}[/{self.theme.get_color('success')}]")

    def warning(self, message: str, title: str = "Warning"):
        """Show a warning notification."""
        self._add_notification(message, title, "warning")
        console.print(f"[{self.theme.get_color('warning')}]⚠ {message}[/{self.theme.get_color('warning')}]")

    def error(self, message: str, title: str = "Error"):
        """Show an error notification."""
        self._add_notification(message, title, "error")
        console.print(f"[{self.theme.get_color('error')}]✗ {message}[/{self.theme.get_color('error')}]")

    def info(self, message: str, title: str = "Info"):
        """Show an info notification."""
        self._add_notification(message, title, "info")
        console.print(f"[{self.theme.get_color('info')}]ℹ {message}[/{self.theme.get_color('info')}]")

    def _add_notification(self, message: str, title: str, type_: str):
        """Add notification to history."""
        self.notifications.append({"timestamp": time.time(), "title": title, "message": message, "type": type_})

    def get_recent_notifications(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent notifications."""
        return self.notifications[-count:] if self.notifications else []

    def clear_notifications(self):
        """Clear all notifications."""
        self.notifications.clear()


class UIComponents:
    """Collection of reusable UI components."""

    def __init__(self, theme_manager: ThemeManager):
        self.theme = theme_manager

    def create_menu(self, title: str, options: List[Dict[str, str]], default: str = None) -> str:
        """Create an interactive menu."""
        table = Table(title=title, show_header=True, header_style=f"bold {self.theme.get_color('primary')}")
        table.add_column("Option", style=self.theme.get_color("text"))
        table.add_column("Description", style=self.theme.get_color("muted"))

        choices = []
        for i, option in enumerate(options, 1):
            label = option.get("label", f"Option {i}")
            description = option.get("description", "")
            table.add_row(f"{i}. {label}", description)
            choices.append(str(i))

        console.print(table)

        return Prompt.ask("Select an option", choices=choices, default=default or choices[0] if choices else None)

    def create_form(self, fields: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create an interactive form."""
        results = {}

        for field in fields:
            field_name = field["name"]
            field_type = field.get("type", "text")
            field_label = field.get("label", field_name)
            field_default = field.get("default")
            field_required = field.get("required", True)

            while True:
                try:
                    if field_type == "int":
                        value = IntPrompt.ask(field_label, default=field_default)
                    elif field_type == "float":
                        value = FloatPrompt.ask(field_label, default=field_default)
                    elif field_type == "bool":
                        value = Confirm.ask(field_label, default=field_default)
                    else:
                        value = Prompt.ask(field_label, default=field_default)

                    if value is not None or not field_required:
                        results[field_name] = value
                        break
                    else:
                        console.print(
                            f"[{self.theme.get_color('warning')}]This field is required[/{self.theme.get_color('warning')}]"
                        )

                except ValueError as e:
                    console.print(
                        f"[{self.theme.get_color('error')}]Invalid input: {e}[/{self.theme.get_color('error')}]"
                    )

        return results

    def create_status_display(self, status_data: Dict[str, Any]) -> Panel:
        """Create a status display panel."""
        table = Table(show_header=False, box=None)
        table.add_column("Property", style=self.theme.get_color("text"))
        table.add_column("Value", style=self.theme.get_color("accent"))

        for key, value in status_data.items():
            table.add_row(key.replace("_", " ").title(), str(value))

        return self.theme.create_panel(table, title="Status")


# Global UI manager
theme_manager = ThemeManager()
layout_manager = LayoutManager(theme_manager)
progress_manager = ProgressManager(theme_manager)
notification_system = NotificationSystem(theme_manager)
ui_components = UIComponents(theme_manager)
