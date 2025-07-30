"""
Plugin management UI components for the ICARUS CLI.
"""

from typing import Dict
from typing import Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Container
from textual.containers import Horizontal
from textual.containers import Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button
from textual.widgets import Checkbox
from textual.widgets import DataTable
from textual.widgets import Input
from textual.widgets import Label
from textual.widgets import ProgressBar
from textual.widgets import Static
from textual.widgets import TabbedContent
from textual.widgets import TabPane

from .manager import PluginManager
from .models import PluginInfo
from .models import PluginStatus
from .models import SecurityLevel


class PluginListWidget(DataTable):
    """Widget for displaying a list of plugins."""

    def __init__(self, plugin_manager: PluginManager, **kwargs):
        super().__init__(**kwargs)
        self.plugin_manager = plugin_manager
        self.add_columns("Name", "Version", "Type", "Status", "Security")
        self.refresh_plugins()

    def refresh_plugins(self):
        """Refresh the plugin list."""
        self.clear()
        plugins = self.plugin_manager.get_all_plugins()

        for plugin in plugins:
            status_icon = self._get_status_icon(plugin.status)
            security_icon = self._get_security_icon(plugin.manifest.security_level)

            self.add_row(
                plugin.manifest.name,
                str(plugin.manifest.version),
                plugin.manifest.plugin_type.value.title(),
                f"{status_icon} {plugin.status.value.title()}",
                f"{security_icon} {plugin.manifest.security_level.value.title()}",
                key=plugin.id,
            )

    def _get_status_icon(self, status: PluginStatus) -> str:
        """Get icon for plugin status."""
        icons = {
            PluginStatus.UNKNOWN: "❓",
            PluginStatus.DISCOVERED: "🔍",
            PluginStatus.LOADED: "📦",
            PluginStatus.ACTIVE: "✅",
            PluginStatus.DISABLED: "❌",
            PluginStatus.ERROR: "⚠️",
            PluginStatus.UPDATING: "🔄",
            PluginStatus.INSTALLING: "⬇️",
            PluginStatus.UNINSTALLING: "🗑️",
        }
        return icons.get(status, "❓")

    def _get_security_icon(self, level: SecurityLevel) -> str:
        """Get icon for security level."""
        icons = {
            SecurityLevel.SAFE: "🟢",
            SecurityLevel.RESTRICTED: "🟡",
            SecurityLevel.ELEVATED: "🟠",
            SecurityLevel.DANGEROUS: "🔴",
        }
        return icons.get(level, "❓")


class PluginDetailsWidget(Container):
    """Widget for displaying plugin details."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_plugin: Optional[PluginInfo] = None

    def compose(self) -> ComposeResult:
        """Compose the plugin details widget."""
        with Vertical():
            yield Label("Plugin Details", classes="section-title")
            yield Static("", id="plugin-name", classes="plugin-name")
            yield Static("", id="plugin-description", classes="plugin-description")
            yield Static("", id="plugin-author", classes="plugin-author")
            yield Static("", id="plugin-version", classes="plugin-version")
            yield Static("", id="plugin-type", classes="plugin-type")
            yield Static("", id="plugin-security", classes="plugin-security")
            yield Static("", id="plugin-status", classes="plugin-status")
            yield Static("", id="plugin-path", classes="plugin-path")

            with Horizontal(classes="plugin-actions"):
                yield Button("Activate", id="btn-activate", variant="success")
                yield Button("Deactivate", id="btn-deactivate", variant="warning")
                yield Button("Configure", id="btn-configure", variant="primary")
                yield Button("Uninstall", id="btn-uninstall", variant="error")

    def update_plugin(self, plugin: Optional[PluginInfo]):
        """Update the displayed plugin information."""
        self.current_plugin = plugin

        if plugin is None:
            self._clear_details()
            return

        # Update details
        self.query_one("#plugin-name", Static).update(f"Name: {plugin.manifest.name}")
        self.query_one("#plugin-description", Static).update(
            f"Description: {plugin.manifest.description}",
        )
        self.query_one("#plugin-author", Static).update(
            f"Author: {plugin.manifest.author.name}",
        )
        self.query_one("#plugin-version", Static).update(
            f"Version: {plugin.manifest.version}",
        )
        self.query_one("#plugin-type", Static).update(
            f"Type: {plugin.manifest.plugin_type.value.title()}",
        )
        self.query_one("#plugin-security", Static).update(
            f"Security: {plugin.manifest.security_level.value.title()}",
        )
        self.query_one("#plugin-status", Static).update(
            f"Status: {plugin.status.value.title()}",
        )
        self.query_one("#plugin-path", Static).update(f"Path: {plugin.path}")

        # Update button states
        self._update_button_states(plugin)

    def _clear_details(self):
        """Clear plugin details."""
        for field_id in [
            "plugin-name",
            "plugin-description",
            "plugin-author",
            "plugin-version",
            "plugin-type",
            "plugin-security",
            "plugin-status",
            "plugin-path",
        ]:
            self.query_one(f"#{field_id}", Static).update("")

        # Disable all buttons
        for button in self.query(Button):
            button.disabled = True

    def _update_button_states(self, plugin: PluginInfo):
        """Update button states based on plugin status."""
        activate_btn = self.query_one("#btn-activate", Button)
        deactivate_btn = self.query_one("#btn-deactivate", Button)
        configure_btn = self.query_one("#btn-configure", Button)
        uninstall_btn = self.query_one("#btn-uninstall", Button)

        # Enable/disable buttons based on plugin status
        activate_btn.disabled = plugin.status in [
            PluginStatus.ACTIVE,
            PluginStatus.ERROR,
        ]
        deactivate_btn.disabled = plugin.status != PluginStatus.ACTIVE
        configure_btn.disabled = not plugin.is_loaded
        uninstall_btn.disabled = plugin.status in [
            PluginStatus.INSTALLING,
            PluginStatus.UNINSTALLING,
        ]


class PluginInstallWidget(Container):
    """Widget for installing new plugins."""

    def compose(self) -> ComposeResult:
        """Compose the plugin install widget."""
        with Vertical():
            yield Label("Install Plugin", classes="section-title")

            with Horizontal():
                yield Label("Source:")
                yield Input(placeholder="File path or URL", id="install-source")
                yield Button("Browse", id="btn-browse")

            with Horizontal():
                yield Checkbox("Force install", id="force-install")
                yield Button("Install", id="btn-install", variant="success")

            yield ProgressBar(id="install-progress", show_eta=False)
            yield Static("", id="install-status")


class PluginConfigWidget(Container):
    """Widget for configuring plugins."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_plugin: Optional[PluginInfo] = None
        self.config_inputs: Dict[str, Input] = {}

    def compose(self) -> ComposeResult:
        """Compose the plugin config widget."""
        with Vertical():
            yield Label("Plugin Configuration", classes="section-title")
            yield Container(id="config-fields")

            with Horizontal():
                yield Button("Save", id="btn-save-config", variant="success")
                yield Button("Reset", id="btn-reset-config", variant="warning")

    def update_plugin(
        self,
        plugin: Optional[PluginInfo],
        plugin_manager: PluginManager,
    ):
        """Update the configuration for a plugin."""
        self.current_plugin = plugin
        config_container = self.query_one("#config-fields", Container)
        config_container.remove_children()
        self.config_inputs.clear()

        if plugin is None:
            return

        # Get plugin configuration
        plugin_config = plugin_manager.registry.configs.get(plugin.manifest.name)
        if not plugin_config:
            config_container.mount(
                Static("No configuration available for this plugin."),
            )
            return

        # Create input fields for configuration
        for key, value in plugin_config.settings.items():
            with config_container:
                with Horizontal():
                    config_container.mount(Label(f"{key}:"))
                    input_widget = Input(value=str(value), id=f"config-{key}")
                    config_container.mount(input_widget)
                    self.config_inputs[key] = input_widget


class PluginManagerScreen(Screen):
    """Main plugin manager screen."""

    TITLE = "Plugin Manager"

    class PluginSelected(Message):
        """Message sent when a plugin is selected."""

        def __init__(self, plugin_id: str) -> None:
            self.plugin_id = plugin_id
            super().__init__()

    def __init__(self, plugin_manager: PluginManager, **kwargs):
        super().__init__(**kwargs)
        self.plugin_manager = plugin_manager

    def compose(self) -> ComposeResult:
        """Compose the plugin manager screen."""
        with TabbedContent():
            with TabPane("Plugins", id="tab-plugins"):
                with Horizontal():
                    with Vertical(classes="plugin-list-panel"):
                        yield Label("Installed Plugins", classes="panel-title")
                        yield PluginListWidget(self.plugin_manager, id="plugin-list")

                        with Horizontal():
                            yield Button("Refresh", id="btn-refresh")
                            yield Button("Discover", id="btn-discover")

                    with Vertical(classes="plugin-details-panel"):
                        yield PluginDetailsWidget(id="plugin-details")

            with TabPane("Install", id="tab-install"):
                yield PluginInstallWidget(id="plugin-install")

            with TabPane("Configure", id="tab-configure"):
                yield PluginConfigWidget(id="plugin-config")

    @on(DataTable.RowSelected, "#plugin-list")
    def on_plugin_selected(self, event: DataTable.RowSelected) -> None:
        """Handle plugin selection."""
        plugin_id = event.row_key.value
        plugin = self.plugin_manager.get_plugin_info(plugin_id)

        # Update details panel
        details_widget = self.query_one("#plugin-details", PluginDetailsWidget)
        details_widget.update_plugin(plugin)

        # Update config panel
        config_widget = self.query_one("#plugin-config", PluginConfigWidget)
        config_widget.update_plugin(plugin, self.plugin_manager)

        # Send message
        self.post_message(self.PluginSelected(plugin_id))

    @on(Button.Pressed, "#btn-refresh")
    def on_refresh_pressed(self) -> None:
        """Handle refresh button press."""
        plugin_list = self.query_one("#plugin-list", PluginListWidget)
        plugin_list.refresh_plugins()

    @on(Button.Pressed, "#btn-discover")
    def on_discover_pressed(self) -> None:
        """Handle discover button press."""
        self.plugin_manager.discover_plugins()
        plugin_list = self.query_one("#plugin-list", PluginListWidget)
        plugin_list.refresh_plugins()

    @on(Button.Pressed, "#btn-activate")
    def on_activate_pressed(self) -> None:
        """Handle activate button press."""
        details_widget = self.query_one("#plugin-details", PluginDetailsWidget)
        if details_widget.current_plugin:
            success = self.plugin_manager.activate_plugin(
                details_widget.current_plugin.id,
            )
            if success:
                self.notify("Plugin activated successfully", severity="information")
                self.on_refresh_pressed()
            else:
                self.notify("Failed to activate plugin", severity="error")

    @on(Button.Pressed, "#btn-deactivate")
    def on_deactivate_pressed(self) -> None:
        """Handle deactivate button press."""
        details_widget = self.query_one("#plugin-details", PluginDetailsWidget)
        if details_widget.current_plugin:
            success = self.plugin_manager.deactivate_plugin(
                details_widget.current_plugin.id,
            )
            if success:
                self.notify("Plugin deactivated successfully", severity="information")
                self.on_refresh_pressed()
            else:
                self.notify("Failed to deactivate plugin", severity="error")

    @on(Button.Pressed, "#btn-uninstall")
    def on_uninstall_pressed(self) -> None:
        """Handle uninstall button press."""
        details_widget = self.query_one("#plugin-details", PluginDetailsWidget)
        if details_widget.current_plugin:
            # Show confirmation dialog (would need to implement)
            success = self.plugin_manager.uninstall_plugin(
                details_widget.current_plugin.id,
            )
            if success:
                self.notify("Plugin uninstalled successfully", severity="information")
                self.on_refresh_pressed()
            else:
                self.notify("Failed to uninstall plugin", severity="error")

    @on(Button.Pressed, "#btn-install")
    def on_install_pressed(self) -> None:
        """Handle install button press."""
        source_input = self.query_one("#install-source", Input)
        force_checkbox = self.query_one("#force-install", Checkbox)

        if not source_input.value:
            self.notify("Please enter a plugin source", severity="warning")
            return

        success = self.plugin_manager.install_plugin(
            source_input.value,
            force=force_checkbox.value,
        )

        if success:
            self.notify("Plugin installed successfully", severity="information")
            source_input.value = ""
            self.on_refresh_pressed()
        else:
            self.notify("Failed to install plugin", severity="error")

    @on(Button.Pressed, "#btn-save-config")
    def on_save_config_pressed(self) -> None:
        """Handle save config button press."""
        config_widget = self.query_one("#plugin-config", PluginConfigWidget)
        if not config_widget.current_plugin:
            return

        # Collect configuration values
        config = {}
        for key, input_widget in config_widget.config_inputs.items():
            config[key] = input_widget.value

        success = self.plugin_manager.configure_plugin(
            config_widget.current_plugin.id,
            config,
        )

        if success:
            self.notify("Plugin configuration saved", severity="information")
        else:
            self.notify("Failed to save plugin configuration", severity="error")
