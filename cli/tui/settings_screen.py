"""Settings Management Screen

This module provides a comprehensive settings management interface with
theme customization, workspace management, and configuration options.
"""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button
from textual.widgets import Container
from textual.widgets import DataTable
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Horizontal
from textual.widgets import Input
from textual.widgets import Label
from textual.widgets import Select
from textual.widgets import Slider
from textual.widgets import Static
from textual.widgets import Switch
from textual.widgets import TabPane
from textual.widgets import Tabs

from ..core.settings import SettingsManager
from ..core.settings import SettingsScope


class SettingsChanged(Message):
    """Message sent when settings are changed."""

    def __init__(self, scope: str, settings_type: str) -> None:
        self.scope = scope
        self.settings_type = settings_type
        super().__init__()


class ThemePreview(Static):
    """Widget for previewing theme changes."""

    def __init__(self, settings_manager: SettingsManager) -> None:
        super().__init__()
        self.settings_manager = settings_manager
        self.preview_content = "Theme Preview"

    def compose(self) -> ComposeResult:
        """Compose the theme preview."""
        yield Static(self.preview_content, id="theme-preview-content")

    def update_preview(self) -> None:
        """Update the theme preview."""
        theme = self.settings_manager.get_theme_settings()

        # Create preview content based on current theme
        preview_text = f"""
[bold]Theme: {theme.theme_name.title()}[/bold]
[dim]Color Scheme: {theme.color_scheme}[/dim]
[dim]Layout: {theme.layout_style}[/dim]

Sample text with different styles:
• [green]Success message[/green]
• [yellow]Warning message[/yellow]
• [red]Error message[/red]
• [blue]Information[/blue]
• [dim]Muted text[/dim]

Animations: {'Enabled' if theme.animations_enabled else 'Disabled'}
Icons: {'Shown' if theme.show_icons else 'Hidden'}
"""

        preview_widget = self.query_one("#theme-preview-content", Static)
        preview_widget.update(preview_text)


class ThemeSettingsPanel(Container):
    """Panel for theme settings."""

    def __init__(self, settings_manager: SettingsManager) -> None:
        super().__init__()
        self.settings_manager = settings_manager

    def compose(self) -> ComposeResult:
        """Compose the theme settings panel."""
        theme = self.settings_manager.get_theme_settings()

        with ScrollableContainer():
            yield Label("Theme Settings", classes="settings-section-title")

            # Theme selection
            with Horizontal(classes="settings-row"):
                yield Label("Theme:", classes="settings-label")
                yield Select(
                    [
                        ("Aerospace", "aerospace"),
                        ("Scientific", "scientific"),
                        ("Default", "default"),
                        ("Classic", "classic"),
                    ],
                    value=theme.theme_name,
                    id="theme-name-select",
                )

            # Color scheme
            with Horizontal(classes="settings-row"):
                yield Label("Color Scheme:", classes="settings-label")
                yield Select(
                    [("Dark", "dark"), ("Light", "light")],
                    value=theme.color_scheme,
                    id="color-scheme-select",
                )

            # Layout style
            with Horizontal(classes="settings-row"):
                yield Label("Layout Style:", classes="settings-label")
                yield Select(
                    [
                        ("Modern", "modern"),
                        ("Classic", "classic"),
                        ("Compact", "compact"),
                    ],
                    value=theme.layout_style,
                    id="layout-style-select",
                )

            # Sidebar width
            with Horizontal(classes="settings-row"):
                yield Label("Sidebar Width:", classes="settings-label")
                yield Slider(
                    min=20,
                    max=60,
                    step=5,
                    value=theme.sidebar_width,
                    id="sidebar-width-slider",
                )
                yield Label(f"{theme.sidebar_width}", id="sidebar-width-value")

            # Visual effects
            yield Label("Visual Effects", classes="settings-subsection-title")

            with Horizontal(classes="settings-row"):
                yield Label("Animations:", classes="settings-label")
                yield Switch(theme.animations_enabled, id="animations-switch")

            with Horizontal(classes="settings-row"):
                yield Label("Transitions:", classes="settings-label")
                yield Switch(theme.transitions_enabled, id="transitions-switch")

            with Horizontal(classes="settings-row"):
                yield Label("Show Icons:", classes="settings-label")
                yield Switch(theme.show_icons, id="show-icons-switch")

            with Horizontal(classes="settings-row"):
                yield Label("Show Tooltips:", classes="settings-label")
                yield Switch(theme.show_tooltips, id="show-tooltips-switch")

            # Animation speed
            with Horizontal(classes="settings-row"):
                yield Label("Animation Speed:", classes="settings-label")
                yield Select(
                    [("Slow", "slow"), ("Normal", "normal"), ("Fast", "fast")],
                    value=theme.animation_speed,
                    id="animation-speed-select",
                )

            # Transparency
            with Horizontal(classes="settings-row"):
                yield Label("Transparency:", classes="settings-label")
                yield Slider(
                    min=0.5,
                    max=1.0,
                    step=0.05,
                    value=theme.transparency_level,
                    id="transparency-slider",
                )
                yield Label(f"{theme.transparency_level:.2f}", id="transparency-value")

            # Theme preview
            yield Label("Preview", classes="settings-subsection-title")
            yield ThemePreview(self.settings_manager)

            # Theme presets
            yield Label("Quick Presets", classes="settings-subsection-title")
            with Horizontal(classes="settings-row"):
                yield Button("Aerospace Dark", id="preset-aerospace-dark")
                yield Button("Aerospace Light", id="preset-aerospace-light")
                yield Button("Scientific", id="preset-scientific")
                yield Button("Classic", id="preset-classic")

    async def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes."""
        if event.select.id == "theme-name-select":
            self.settings_manager.update_theme_settings(theme_name=event.value)
        elif event.select.id == "color-scheme-select":
            self.settings_manager.update_theme_settings(color_scheme=event.value)
        elif event.select.id == "layout-style-select":
            self.settings_manager.update_theme_settings(layout_style=event.value)
        elif event.select.id == "animation-speed-select":
            self.settings_manager.update_theme_settings(animation_speed=event.value)

        await self._update_preview()

    async def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch changes."""
        if event.switch.id == "animations-switch":
            self.settings_manager.update_theme_settings(animations_enabled=event.value)
        elif event.switch.id == "transitions-switch":
            self.settings_manager.update_theme_settings(transitions_enabled=event.value)
        elif event.switch.id == "show-icons-switch":
            self.settings_manager.update_theme_settings(show_icons=event.value)
        elif event.switch.id == "show-tooltips-switch":
            self.settings_manager.update_theme_settings(show_tooltips=event.value)

        await self._update_preview()

    async def on_slider_changed(self, event: Slider.Changed) -> None:
        """Handle slider changes."""
        if event.slider.id == "sidebar-width-slider":
            self.settings_manager.update_theme_settings(sidebar_width=int(event.value))
            self.query_one("#sidebar-width-value", Label).update(str(int(event.value)))
        elif event.slider.id == "transparency-slider":
            self.settings_manager.update_theme_settings(transparency_level=event.value)
            self.query_one("#transparency-value", Label).update(f"{event.value:.2f}")

        await self._update_preview()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle preset button presses."""
        preset_map = {
            "preset-aerospace-dark": "aerospace_dark",
            "preset-aerospace-light": "aerospace_light",
            "preset-scientific": "scientific",
            "preset-classic": "classic",
        }

        if event.button.id in preset_map:
            preset_name = preset_map[event.button.id]
            self.settings_manager.apply_theme_preset(preset_name)
            await self._refresh_controls()
            await self._update_preview()

    async def _update_preview(self) -> None:
        """Update the theme preview."""
        preview = self.query_one(ThemePreview)
        preview.update_preview()

    async def _refresh_controls(self) -> None:
        """Refresh all controls with current settings."""
        theme = self.settings_manager.get_theme_settings()

        # Update selects
        self.query_one("#theme-name-select", Select).value = theme.theme_name
        self.query_one("#color-scheme-select", Select).value = theme.color_scheme
        self.query_one("#layout-style-select", Select).value = theme.layout_style
        self.query_one("#animation-speed-select", Select).value = theme.animation_speed

        # Update switches
        self.query_one("#animations-switch", Switch).value = theme.animations_enabled
        self.query_one("#transitions-switch", Switch).value = theme.transitions_enabled
        self.query_one("#show-icons-switch", Switch).value = theme.show_icons
        self.query_one("#show-tooltips-switch", Switch).value = theme.show_tooltips

        # Update sliders
        self.query_one("#sidebar-width-slider", Slider).value = theme.sidebar_width
        self.query_one("#transparency-slider", Slider).value = theme.transparency_level

        # Update labels
        self.query_one("#sidebar-width-value", Label).update(str(theme.sidebar_width))
        self.query_one("#transparency-value", Label).update(
            f"{theme.transparency_level:.2f}",
        )


class UserPreferencesPanel(Container):
    """Panel for user preferences."""

    def __init__(self, settings_manager: SettingsManager) -> None:
        super().__init__()
        self.settings_manager = settings_manager

    def compose(self) -> ComposeResult:
        """Compose the user preferences panel."""
        prefs = self.settings_manager.get_user_preferences()

        with ScrollableContainer():
            yield Label("User Preferences", classes="settings-section-title")

            # Personal information
            yield Label("Personal Information", classes="settings-subsection-title")

            with Horizontal(classes="settings-row"):
                yield Label("Name:", classes="settings-label")
                yield Input(prefs.name, placeholder="Your name", id="user-name-input")

            with Horizontal(classes="settings-row"):
                yield Label("Email:", classes="settings-label")
                yield Input(
                    prefs.email,
                    placeholder="your.email@example.com",
                    id="user-email-input",
                )

            with Horizontal(classes="settings-row"):
                yield Label("Organization:", classes="settings-label")
                yield Input(
                    prefs.organization,
                    placeholder="Your organization",
                    id="user-org-input",
                )

            # Interface preferences
            yield Label("Interface Preferences", classes="settings-subsection-title")

            with Horizontal(classes="settings-row"):
                yield Label("Startup Screen:", classes="settings-label")
                yield Select(
                    [
                        ("Dashboard", "dashboard"),
                        ("Analysis", "analysis"),
                        ("Workflow", "workflow"),
                        ("Data", "data"),
                    ],
                    value=prefs.startup_screen,
                    id="startup-screen-select",
                )

            with Horizontal(classes="settings-row"):
                yield Label("Show Welcome:", classes="settings-label")
                yield Switch(prefs.show_welcome, id="show-welcome-switch")

            with Horizontal(classes="settings-row"):
                yield Label("Auto-save Interval (seconds):", classes="settings-label")
                yield Slider(
                    min=60,
                    max=1800,
                    step=60,
                    value=prefs.auto_save_interval,
                    id="auto-save-slider",
                )
                yield Label(f"{prefs.auto_save_interval}", id="auto-save-value")

            # Notifications
            yield Label("Notifications", classes="settings-subsection-title")

            with Horizontal(classes="settings-row"):
                yield Label("Enable Notifications:", classes="settings-label")
                yield Switch(prefs.enable_notifications, id="notifications-switch")

            with Horizontal(classes="settings-row"):
                yield Label("Notification Sound:", classes="settings-label")
                yield Switch(prefs.notification_sound, id="notification-sound-switch")

            with Horizontal(classes="settings-row"):
                yield Label(
                    "Notification Duration (seconds):",
                    classes="settings-label",
                )
                yield Slider(
                    min=1,
                    max=30,
                    step=1,
                    value=prefs.notification_duration,
                    id="notification-duration-slider",
                )
                yield Label(
                    f"{prefs.notification_duration}",
                    id="notification-duration-value",
                )

            # Performance
            yield Label("Performance", classes="settings-subsection-title")

            with Horizontal(classes="settings-row"):
                yield Label("Max Memory Usage (MB):", classes="settings-label")
                yield Slider(
                    min=256,
                    max=8192,
                    step=256,
                    value=prefs.max_memory_usage,
                    id="max-memory-slider",
                )
                yield Label(f"{prefs.max_memory_usage}", id="max-memory-value")

            with Horizontal(classes="settings-row"):
                yield Label("Max CPU Cores (0=auto):", classes="settings-label")
                yield Slider(
                    min=0,
                    max=16,
                    step=1,
                    value=prefs.max_cpu_cores,
                    id="max-cpu-slider",
                )
                yield Label(f"{prefs.max_cpu_cores}", id="max-cpu-value")

            with Horizontal(classes="settings-row"):
                yield Label("Cache Size (MB):", classes="settings-label")
                yield Slider(
                    min=64,
                    max=2048,
                    step=64,
                    value=prefs.cache_size,
                    id="cache-size-slider",
                )
                yield Label(f"{prefs.cache_size}", id="cache-size-value")

            # Privacy
            yield Label("Privacy", classes="settings-subsection-title")

            with Horizontal(classes="settings-row"):
                yield Label("Analytics:", classes="settings-label")
                yield Switch(prefs.analytics_enabled, id="analytics-switch")

            with Horizontal(classes="settings-row"):
                yield Label("Crash Reporting:", classes="settings-label")
                yield Switch(prefs.crash_reporting, id="crash-reporting-switch")

            with Horizontal(classes="settings-row"):
                yield Label("Usage Statistics:", classes="settings-label")
                yield Switch(prefs.usage_statistics, id="usage-stats-switch")

    async def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes."""
        if event.input.id == "user-name-input":
            self.settings_manager.update_user_preferences(name=event.value)
        elif event.input.id == "user-email-input":
            self.settings_manager.update_user_preferences(email=event.value)
        elif event.input.id == "user-org-input":
            self.settings_manager.update_user_preferences(organization=event.value)

    async def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes."""
        if event.select.id == "startup-screen-select":
            self.settings_manager.update_user_preferences(startup_screen=event.value)

    async def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch changes."""
        switch_map = {
            "show-welcome-switch": "show_welcome",
            "notifications-switch": "enable_notifications",
            "notification-sound-switch": "notification_sound",
            "analytics-switch": "analytics_enabled",
            "crash-reporting-switch": "crash_reporting",
            "usage-stats-switch": "usage_statistics",
        }

        if event.switch.id in switch_map:
            setting_name = switch_map[event.switch.id]
            self.settings_manager.update_user_preferences(**{setting_name: event.value})

    async def on_slider_changed(self, event: Slider.Changed) -> None:
        """Handle slider changes."""
        slider_map = {
            "auto-save-slider": ("auto_save_interval", "auto-save-value"),
            "notification-duration-slider": (
                "notification_duration",
                "notification-duration-value",
            ),
            "max-memory-slider": ("max_memory_usage", "max-memory-value"),
            "max-cpu-slider": ("max_cpu_cores", "max-cpu-value"),
            "cache-size-slider": ("cache_size", "cache-size-value"),
        }

        if event.slider.id in slider_map:
            setting_name, label_id = slider_map[event.slider.id]
            value = int(event.value)
            self.settings_manager.update_user_preferences(**{setting_name: value})
            self.query_one(f"#{label_id}", Label).update(str(value))


class WorkspaceManagementPanel(Container):
    """Panel for workspace management."""

    def __init__(self, settings_manager: SettingsManager) -> None:
        super().__init__()
        self.settings_manager = settings_manager

    def compose(self) -> ComposeResult:
        """Compose the workspace management panel."""
        with ScrollableContainer():
            yield Label("Workspace Management", classes="settings-section-title")

            # Current workspace info
            current_workspace = self.settings_manager.current_workspace or "None"
            yield Label(
                f"Current Workspace: {current_workspace}",
                classes="settings-info",
            )

            # Workspace list
            yield Label("Available Workspaces", classes="settings-subsection-title")
            yield DataTable(id="workspaces-table")

            # Workspace actions
            with Horizontal(classes="settings-row"):
                yield Button("New Workspace", id="new-workspace-btn")
                yield Button("Switch Workspace", id="switch-workspace-btn")
                yield Button("Delete Workspace", id="delete-workspace-btn")

            # Current workspace settings
            if self.settings_manager.current_workspace:
                yield Label(
                    "Current Workspace Settings",
                    classes="settings-subsection-title",
                )
                workspace = self.settings_manager.get_workspace_settings()

                with Horizontal(classes="settings-row"):
                    yield Label("Name:", classes="settings-label")
                    yield Input(workspace.name, id="workspace-name-input")

                with Horizontal(classes="settings-row"):
                    yield Label("Description:", classes="settings-label")
                    yield Input(workspace.description, id="workspace-desc-input")

                with Horizontal(classes="settings-row"):
                    yield Label("Data Directory:", classes="settings-label")
                    yield Input(workspace.data_directory, id="workspace-data-dir-input")

                with Horizontal(classes="settings-row"):
                    yield Label("Default 2D Solver:", classes="settings-label")
                    yield Select(
                        [("XFoil", "xfoil"), ("JavaFoil", "javafoil")],
                        value=workspace.default_solver_2d,
                        id="workspace-solver-2d-select",
                    )

                with Horizontal(classes="settings-row"):
                    yield Label("Default 3D Solver:", classes="settings-label")
                    yield Select(
                        [
                            ("AVL", "avl"),
                            ("GenuVP", "genuvp"),
                            ("OpenFOAM", "openfoam"),
                        ],
                        value=workspace.default_solver_3d,
                        id="workspace-solver-3d-select",
                    )

                with Horizontal(classes="settings-row"):
                    yield Label("Auto Backup:", classes="settings-label")
                    yield Switch(workspace.auto_backup, id="workspace-backup-switch")

                with Horizontal(classes="settings-row"):
                    yield Label("Backup Interval (hours):", classes="settings-label")
                    yield Slider(
                        min=1,
                        max=168,
                        step=1,
                        value=workspace.backup_interval_hours,
                        id="workspace-backup-interval-slider",
                    )
                    yield Label(
                        f"{workspace.backup_interval_hours}",
                        id="workspace-backup-interval-value",
                    )

    async def on_mount(self) -> None:
        """Initialize the workspace table."""
        await self._refresh_workspace_table()

    async def _refresh_workspace_table(self) -> None:
        """Refresh the workspace table."""
        table = self.query_one("#workspaces-table", DataTable)
        table.clear(columns=True)

        table.add_columns("Name", "Current", "Projects", "Created")

        workspaces = self.settings_manager.list_workspaces()
        for workspace_name in workspaces:
            is_current = (
                "✓" if workspace_name == self.settings_manager.current_workspace else ""
            )
            projects = len(self.settings_manager.list_projects(workspace_name))
            table.add_row(workspace_name, is_current, str(projects), "N/A")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "new-workspace-btn":
            await self._create_new_workspace()
        elif event.button.id == "switch-workspace-btn":
            await self._switch_workspace()
        elif event.button.id == "delete-workspace-btn":
            await self._delete_workspace()

    async def _create_new_workspace(self) -> None:
        """Create a new workspace."""
        # This would typically open a dialog
        # For now, create a default workspace
        import time

        workspace_name = f"workspace_{int(time.time())}"

        if self.settings_manager.create_workspace(workspace_name):
            await self._refresh_workspace_table()

    async def _switch_workspace(self) -> None:
        """Switch to a different workspace."""
        # This would typically open a selection dialog
        workspaces = self.settings_manager.list_workspaces()
        if workspaces:
            # Switch to first available workspace for demo
            self.settings_manager.switch_workspace(workspaces[0])
            await self._refresh_workspace_table()

    async def _delete_workspace(self) -> None:
        """Delete a workspace."""
        # This would typically show a confirmation dialog
        workspaces = self.settings_manager.list_workspaces()
        if len(workspaces) > 1:  # Don't delete if it's the only workspace
            for workspace in workspaces:
                if workspace != self.settings_manager.current_workspace:
                    self.settings_manager.delete_workspace(workspace)
                    await self._refresh_workspace_table()
                    break


class BackupManagementPanel(Container):
    """Panel for backup management."""

    def __init__(self, settings_manager: SettingsManager) -> None:
        super().__init__()
        self.settings_manager = settings_manager

    def compose(self) -> ComposeResult:
        """Compose the backup management panel."""
        with ScrollableContainer():
            yield Label("Backup Management", classes="settings-section-title")

            # Backup actions
            with Horizontal(classes="settings-row"):
                yield Button("Create Backup", id="create-backup-btn")
                yield Button("Restore Backup", id="restore-backup-btn")
                yield Button("Delete Backup", id="delete-backup-btn")
                yield Button("Cleanup Old", id="cleanup-backups-btn")

            # Backup list
            yield Label("Available Backups", classes="settings-subsection-title")
            yield DataTable(id="backups-table")

            # Import/Export
            yield Label("Import/Export Settings", classes="settings-subsection-title")

            with Horizontal(classes="settings-row"):
                yield Label("Export Scope:", classes="settings-label")
                yield Select(
                    [
                        ("Global Only", "global"),
                        ("Workspace Only", "workspace"),
                        ("Project Only", "project"),
                        ("All Settings", "session"),
                    ],
                    value="global",
                    id="export-scope-select",
                )

            with Horizontal(classes="settings-row"):
                yield Label("Export Format:", classes="settings-label")
                yield Select(
                    [("JSON", "json"), ("YAML", "yaml")],
                    value="json",
                    id="export-format-select",
                )

            with Horizontal(classes="settings-row"):
                yield Button("Export Settings", id="export-settings-btn")
                yield Button("Import Settings", id="import-settings-btn")

    async def on_mount(self) -> None:
        """Initialize the backup table."""
        await self._refresh_backup_table()

    async def _refresh_backup_table(self) -> None:
        """Refresh the backup table."""
        table = self.query_one("#backups-table", DataTable)
        table.clear(columns=True)

        table.add_columns("Name", "Created", "Workspace", "Project")

        backups = self.settings_manager.list_backups()
        for backup in backups:
            table.add_row(
                backup["name"],
                backup["created_at"][:19],  # Truncate timestamp
                backup.get("workspace", "N/A"),
                backup.get("project", "N/A"),
            )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "create-backup-btn":
            backup_name = self.settings_manager.create_backup()
            await self._refresh_backup_table()
        elif event.button.id == "cleanup-backups-btn":
            self.settings_manager.cleanup_old_backups(5)
            await self._refresh_backup_table()


class SettingsScreen(Screen):
    """Main settings management screen."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("ctrl+s", "save", "Save"),
        Binding("ctrl+r", "reset", "Reset"),
        Binding("f1", "help", "Help"),
    ]

    def __init__(self, settings_manager: SettingsManager) -> None:
        super().__init__()
        self.settings_manager = settings_manager

        # Register for settings changes
        self.settings_manager.register_change_callback(self._on_settings_changed)

    def compose(self) -> ComposeResult:
        """Compose the settings screen."""
        yield Header()

        with Container(id="settings-container"):
            with Tabs(id="settings-tabs"):
                with TabPane("Theme", id="theme-tab"):
                    yield ThemeSettingsPanel(self.settings_manager)

                with TabPane("User", id="user-tab"):
                    yield UserPreferencesPanel(self.settings_manager)

                with TabPane("Workspace", id="workspace-tab"):
                    yield WorkspaceManagementPanel(self.settings_manager)

                with TabPane("Backup", id="backup-tab"):
                    yield BackupManagementPanel(self.settings_manager)

        yield Footer()

    def _on_settings_changed(self, scope: str, settings_type: str) -> None:
        """Handle settings changes."""
        # Post a message about the change
        self.post_message(SettingsChanged(scope, settings_type))

    async def action_save(self) -> None:
        """Save all settings."""
        self.settings_manager.save_all_settings()
        self.notify("Settings saved successfully", severity="information")

    async def action_reset(self) -> None:
        """Reset settings to defaults."""
        # This would typically show a confirmation dialog
        self.settings_manager.reset_to_defaults(SettingsScope.GLOBAL)
        self.notify("Settings reset to defaults", severity="warning")

    async def action_back(self) -> None:
        """Go back to previous screen."""
        await self.action_save()  # Auto-save before leaving
        self.app.pop_screen()

    async def action_help(self) -> None:
        """Show help information."""
        self.notify(
            "Settings Help: Use tabs to navigate different setting categories",
            severity="information",
        )
