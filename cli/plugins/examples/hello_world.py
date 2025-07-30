"""
Hello World example plugin for ICARUS CLI.

This is a simple example plugin that demonstrates basic plugin functionality.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import IcarusPlugin
from api import PluginAuthor
from api import PluginManifest
from api import PluginType
from api import PluginVersion
from api import SecurityLevel
from api import plugin_command
from api import plugin_menu_item


class HelloWorldPlugin(IcarusPlugin):
    """
    Simple hello world plugin that demonstrates basic plugin functionality.
    """

    def get_manifest(self) -> PluginManifest:
        """Return plugin manifest."""
        return PluginManifest(
            name="hello_world",
            version=PluginVersion(1, 0, 0),
            description="A simple hello world plugin for demonstration",
            author=PluginAuthor(
                name="ICARUS Team",
                email="team@icarus.example.com",
                url="https://icarus.example.com",
            ),
            plugin_type=PluginType.UTILITY,
            security_level=SecurityLevel.SAFE,
            main_module="hello_world",
            main_class="HelloWorldPlugin",
            keywords=["example", "demo", "hello", "world"],
            homepage="https://icarus.example.com/plugins/hello-world",
            license="MIT",
        )

    def on_activate(self):
        """Called when plugin is activated."""
        self.api.log_info("Hello World plugin activated")

        # Add menu item
        self.api.add_menu_item(
            "Tools/Examples/Hello World",
            "Say Hello",
            self.say_hello,
            icon="👋",
            shortcut="Ctrl+H",
        )

        # Register commands
        self.api.register_command(
            "hello_world.hello",
            self.say_hello,
            "Display a hello world message",
            "hello_world.hello",
        )

        self.api.register_command(
            "hello_world.greet",
            self.greet_user,
            "Greet the current user",
            "hello_world.greet [name]",
        )

        # Register event handler
        self.api.register_event_handler("app_started", self.on_app_started)

    def on_deactivate(self):
        """Called when plugin is deactivated."""
        self.api.log_info("Hello World plugin deactivated")

    def on_configure(self, config):
        """Called when plugin configuration changes."""
        self.api.log_info(f"Hello World plugin configured: {config}")

    @plugin_menu_item("Tools/Examples/Hello World", "Say Hello", "👋")
    def say_hello(self):
        """Display a hello world message."""
        message = self.api.get_config("message", "Hello, World!")
        self.api.show_notification(message, "info", 3000)
        self.api.log_info(f"Displayed message: {message}")

    @plugin_command("hello_world.greet", "Greet a user")
    def greet_user(self, name: str = None):
        """Greet a specific user."""
        if not name:
            name = self.api.get_user_data("username", "User")

        greeting = f"Hello, {name}! Welcome to ICARUS CLI."
        self.api.show_notification(greeting, "success", 4000)
        self.api.log_info(f"Greeted user: {name}")

    def on_app_started(self, data):
        """Handle app started event."""
        if self.api.get_config("show_startup_greeting", True):
            self.api.show_notification("Hello World plugin is ready!", "info", 2000)


# Plugin manifest for directory-based loading
PLUGIN_MANIFEST = {
    "name": "hello_world",
    "version": "1.0.0",
    "description": "A simple hello world plugin for demonstration",
    "author": {
        "name": "ICARUS Team",
        "email": "team@icarus.example.com",
        "url": "https://icarus.example.com",
    },
    "type": "utility",
    "security_level": "safe",
    "main_module": "hello_world",
    "main_class": "HelloWorldPlugin",
    "keywords": ["example", "demo", "hello", "world"],
    "homepage": "https://icarus.example.com/plugins/hello-world",
    "license": "MIT",
    "default_config": {"message": "Hello, World!", "show_startup_greeting": True},
}
