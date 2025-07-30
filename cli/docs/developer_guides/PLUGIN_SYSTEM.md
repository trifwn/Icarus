# ICARUS CLI Plugin System

This document describes the plugin system for the ICARUS CLI project.

## Plugin System Overview

The ICARUS CLI plugin system allows extending the CLI with custom functionality. Plugins can add new commands, analysis capabilities, visualization options, and more.

## Plugin Structure

A plugin is a Python package with a specific structure:

```
plugin_name/
├── __init__.py           # Plugin initialization
├── plugin.json           # Plugin metadata
├── commands/             # Custom commands
├── analysis/             # Custom analysis modules
├── visualization/        # Custom visualization components
└── resources/            # Additional resources
```

### Plugin Metadata

The `plugin.json` file contains metadata about the plugin:

```json
{
  "name": "example-plugin",
  "version": "1.0.0",
  "description": "Example plugin for ICARUS CLI",
  "author": "Your Name",
  "email": "your.email@example.com",
  "website": "https://example.com",
  "license": "MIT",
  "requires": {
    "icarus-cli": ">=2.0.0",
    "numpy": ">=1.20.0"
  },
  "entry_point": "example_plugin:initialize",
  "hooks": {
    "analysis": ["example_plugin.analysis:register_analysis"],
    "visualization": ["example_plugin.visualization:register_visualization"],
    "commands": ["example_plugin.commands:register_commands"]
  }
}
```

### Plugin Initialization

The `__init__.py` file contains the plugin initialization code:

```python
from icarus.cli.plugins import Plugin

class ExamplePlugin(Plugin):
    def initialize(self):
        """Initialize the plugin"""
        self.logger.info("Initializing example plugin")
        # Register components
        self.register_analysis()
        self.register_visualization()
        self.register_commands()

    def register_analysis(self):
        """Register analysis modules"""
        from .analysis import ExampleAnalysis
        self.app.analysis_service.register_module(ExampleAnalysis())

    def register_visualization(self):
        """Register visualization components"""
        from .visualization import ExampleVisualization
        self.app.visualization_service.register_component(ExampleVisualization())

    def register_commands(self):
        """Register commands"""
        from .commands import ExampleCommand
        self.app.command_registry.register(ExampleCommand())
```

## Plugin API

The plugin API provides interfaces for extending different aspects of the CLI:

### Analysis API

```python
from icarus.cli.analysis import AnalysisModule, AnalysisResult

class ExampleAnalysis(AnalysisModule):
    """Example analysis module"""

    @property
    def name(self):
        return "example"

    @property
    def description(self):
        return "Example analysis module"

    def get_parameters(self):
        """Get parameter definitions"""
        return {
            "param1": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 10.0,
                "description": "Example parameter"
            }
        }

    async def run(self, parameters):
        """Run the analysis"""
        # Perform analysis
        result = {"value": parameters["param1"] * 2}
        return AnalysisResult(status="success", data=result)
```

### Visualization API

```python
from icarus.cli.visualization import VisualizationComponent

class ExampleVisualization(VisualizationComponent):
    """Example visualization component"""

    @property
    def name(self):
        return "example"

    @property
    def description(self):
        return "Example visualization component"

    def supports_data(self, data):
        """Check if this component supports the given data"""
        return "value" in data

    def generate(self, data, options=None):
        """Generate visualization"""
        options = options or {}
        # Generate visualization
        return {
            "type": "bar",
            "data": {
                "labels": ["Value"],
                "datasets": [{
                    "label": "Example",
                    "data": [data["value"]]
                }]
            }
        }
```

### Command API

```python
from icarus.cli.commands import Command

class ExampleCommand(Command):
    """Example command"""

    @property
    def name(self):
        return "example"

    @property
    def description(self):
        return "Example command"

    def configure_parser(self, parser):
        """Configure command parser"""
        parser.add_argument("--value", type=float, default=1.0, help="Example value")

    async def execute(self, args):
        """Execute the command"""
        self.logger.info(f"Executing example command with value {args.value}")
        # Perform command action
        return {"result": args.value * 2}
```

## Plugin Discovery and Loading

The plugin system discovers and loads plugins from the following locations:

- System plugin directory: `/usr/local/share/icarus-cli/plugins`
- User plugin directory: `~/.icarus/plugins`
- Project plugin directory: `./.icarus/plugins`
- Custom directories specified in configuration

Plugins are discovered by searching for `plugin.json` files in these directories.

## Plugin Lifecycle

1. **Discovery**: The plugin system searches for plugins in the plugin directories
2. **Validation**: Plugins are validated for compatibility and security
3. **Loading**: Plugin packages are loaded into the Python environment
4. **Initialization**: Plugin initialization code is executed
5. **Registration**: Plugin components are registered with the application
6. **Execution**: Plugin functionality is available for use
7. **Unloading**: Plugins are unloaded when the application exits

## Plugin Security

The plugin system includes security features to prevent malicious plugins:

- **Sandboxing**: Plugins run in a restricted environment
- **Permission System**: Plugins must request permissions for sensitive operations
- **Code Signing**: Plugin packages can be signed for verification
- **Dependency Validation**: Plugin dependencies are validated for security issues

## Plugin Development

### Creating a New Plugin

1. Create a new directory for your plugin
2. Create a `plugin.json` file with plugin metadata
3. Create an `__init__.py` file with plugin initialization code
4. Implement plugin functionality in additional modules
5. Test your plugin with the CLI
6. Package your plugin for distribution

### Plugin Testing

The CLI includes tools for testing plugins:

```bash
icarus plugin test my-plugin
```

This command runs tests on the plugin to ensure it works correctly and follows best practices.

### Plugin Packaging

Plugins can be packaged for distribution using standard Python packaging tools:

```bash
# Create a source distribution
python setup.py sdist

# Create a wheel distribution
python setup.py bdist_wheel
```

### Plugin Publishing

Plugins can be published to the ICARUS plugin repository:

```bash
icarus plugin publish my-plugin
```

## Plugin Management

The CLI includes commands for managing plugins:

```bash
# List installed plugins
icarus plugin list

# Install a plugin
icarus plugin install my-plugin

# Update a plugin
icarus plugin update my-plugin

# Remove a plugin
icarus plugin remove my-plugin

# Enable a plugin
icarus plugin enable my-plugin

# Disable a plugin
icarus plugin disable my-plugin
```

## Plugin Configuration

Plugins can define configuration options in their `plugin.json` file:

```json
{
  "name": "example-plugin",
  "version": "1.0.0",
  "config": {
    "option1": {
      "type": "string",
      "default": "default value",
      "description": "Example option"
    },
    "option2": {
      "type": "number",
      "default": 42,
      "description": "Another example option"
    }
  }
}
```

Users can configure plugins using the CLI:

```bash
icarus plugin config example-plugin option1 "new value"
```

Plugins can access their configuration using the plugin API:

```python
class ExamplePlugin(Plugin):
    def initialize(self):
        option1 = self.config.get("option1")
        self.logger.info(f"Option 1: {option1}")
```

## Plugin Examples

The CLI includes example plugins that demonstrate different plugin capabilities:

- `example-analysis`: Demonstrates adding a new analysis module
- `example-visualization`: Demonstrates adding new visualization components
- `example-command`: Demonstrates adding new commands
- `example-integration`: Demonstrates integrating with external tools
