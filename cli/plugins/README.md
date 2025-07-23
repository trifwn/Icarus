# ICARUS CLI Plugin System

The ICARUS CLI Plugin System provides a comprehensive framework for extending the ICARUS CLI with custom functionality, integrations, and specialized tools.

## Overview

The plugin system supports:

- **Dynamic Discovery**: Automatic discovery and loading of plugins
- **Security Validation**: Comprehensive security analysis and sandboxing
- **Lifecycle Management**: Install, update, activate, deactivate, and uninstall plugins
- **Rich API**: Comprehensive API for plugin development
- **Multiple Types**: Support for various plugin types (analysis, visualization, workflow, etc.)
- **Configuration**: Flexible configuration system for plugins
- **Event System**: Event-driven architecture for plugin communication

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Plugin API    │    │ Plugin Manager  │    │ Plugin Security │
│                 │    │                 │    │                 │
│ - Event System  │    │ - Discovery     │    │ - Validation    │
│ - UI Integration│    │ - Lifecycle     │    │ - Sandboxing    │
│ - Data Access   │    │ - Configuration │    │ - Permissions   │
│ - Commands      │    │ - Registry      │    │ - Risk Analysis │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Plugin Discovery│
                    │                 │
                    │ - File System   │
                    │ - Manifest      │
                    │ - Loading       │
                    │ - Validation    │
                    └─────────────────┘
```

## Plugin Types

The system supports several plugin types:

- **Analysis**: Custom analysis modules and solvers
- **Visualization**: Custom plotting and visualization tools
- **Export/Import**: Data format converters and processors
- **Workflow**: Custom workflow steps and templates
- **Integration**: External tool integrations
- **Utility**: General utility functions and tools
- **Theme**: Custom UI themes and styling

## Security Levels

Plugins are classified into security levels:

- **SAFE**: UI-only plugins with no system access
- **RESTRICTED**: Limited system access with validation
- **ELEVATED**: Full system access with user approval
- **DANGEROUS**: Potentially harmful operations (requires explicit trust)

## Quick Start

### Creating a Simple Plugin

1. Create a plugin directory:
```bash
mkdir my_plugin
cd my_plugin
```

2. Create a plugin manifest (`plugin.json`):
```json
{
  "name": "my_plugin",
  "version": "1.0.0",
  "description": "My first ICARUS plugin",
  "author": {
    "name": "Your Name",
    "email": "your.email@example.com"
  },
  "type": "utility",
  "security_level": "safe",
  "main_module": "main",
  "main_class": "MyPlugin",
  "permissions": []
}
```

3. Create the main plugin file (`main.py`):
```python
from cli.plugins.api import IcarusPlugin, PluginManifest, PluginType, SecurityLevel, PluginAuthor, PluginVersion

class MyPlugin(IcarusPlugin):
    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            name="my_plugin",
            version=PluginVersion(1, 0, 0),
            description="My first ICARUS plugin",
            author=PluginAuthor("Your Name", "your.email@example.com"),
            plugin_type=PluginType.UTILITY,
            security_level=SecurityLevel.SAFE,
            main_module="main",
            main_class="MyPlugin"
        )

    def on_activate(self):
        # Add a menu item
        self.api.add_menu_item(
            "Tools/My Plugin",
            "Hello World",
            self.hello_world,
            icon="🌍"
        )

        # Register a command
        self.api.register_command(
            "my_plugin.hello",
            self.hello_world,
            "Say hello from my plugin"
        )

    def hello_world(self):
        self.api.show_notification("Hello from My Plugin!", "info")
```

4. Install the plugin:
```bash
# Copy to plugins directory
cp -r my_plugin ~/.icarus/plugins/
```

### Plugin Development API

#### Core API Methods

```python
# Event System
api.register_event_handler('analysis_complete', my_handler)
api.emit_event('custom_event', data)

# UI Integration
api.register_screen('my_screen', MyScreenClass)
api.add_menu_item('Tools/My Tool', 'Action', my_action)
api.show_notification('Message', 'info')
api.show_dialog('Title', 'Content', 'question', ['Yes', 'No'])

# Data Access
data = api.get_analysis_data('analysis_id')
api.save_analysis_data('analysis_id', data)
value = api.get_user_data('key', default)
api.set_user_data('key', value)

# Configuration
config_value = api.get_config('setting', default)
api.set_config('setting', value)

# Commands
api.register_command('my_command', handler, 'Description')

# Analysis Integration
api.register_analysis_type('my_analysis', MyAnalysisClass)
api.register_solver('my_solver', MySolverClass)

# Workflow Integration
api.register_workflow_step('my_step', MyStepClass)

# Export/Import
api.register_exporter('my_format', MyExporterClass)
api.register_importer('my_format', MyImporterClass)
```

#### Decorators

```python
from cli.plugins.api import plugin_command, plugin_event_handler, plugin_menu_item

class MyPlugin(IcarusPlugin):
    @plugin_command('my_plugin.test', 'Test command')
    def test_command(self):
        pass

    @plugin_event_handler('analysis_complete')
    def on_analysis_complete(self, data):
        pass

    @plugin_menu_item('Tools/My Plugin', 'Test Action')
    def test_action(self):
        pass
```

## Plugin Examples

### Analysis Plugin

```python
from cli.plugins.api import IcarusPlugin, PluginManifest, PluginType, SecurityLevel

class CustomAnalysisPlugin(IcarusPlugin):
    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            name="custom_analysis",
            version=PluginVersion(1, 0, 0),
            description="Custom analysis plugin",
            author=PluginAuthor("Developer"),
            plugin_type=PluginType.ANALYSIS,
            security_level=SecurityLevel.RESTRICTED,
            main_module="analysis",
            main_class="CustomAnalysisPlugin"
        )

    def on_activate(self):
        # Register custom analysis type
        self.api.register_analysis_type('custom', CustomAnalysis)

        # Add menu item
        self.api.add_menu_item(
            'Analysis/Custom Analysis',
            'Run Custom Analysis',
            self.run_analysis
        )

    def run_analysis(self):
        # Implementation here
        pass

class CustomAnalysis:
    def __init__(self, config):
        self.config = config

    def run(self):
        # Analysis implementation
        return {"result": "success"}
```

### Visualization Plugin

```python
class VisualizationPlugin(IcarusPlugin):
    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            name="custom_viz",
            version=PluginVersion(1, 0, 0),
            description="Custom visualization plugin",
            author=PluginAuthor("Developer"),
            plugin_type=PluginType.VISUALIZATION,
            security_level=SecurityLevel.SAFE,
            main_module="visualization",
            main_class="VisualizationPlugin"
        )

    def on_activate(self):
        # Register event handler for analysis completion
        self.api.register_event_handler('analysis_complete', self.create_visualization)

    def create_visualization(self, data):
        # Create custom visualization
        pass
```

## Plugin Installation

### From Local Directory
```bash
icarus plugin install /path/to/plugin
```

### From File
```bash
icarus plugin install plugin.zip
```

### From URL (future)
```bash
icarus plugin install https://example.com/plugin.zip
```

## Plugin Management

### List Plugins
```bash
icarus plugin list
```

### Activate Plugin
```bash
icarus plugin activate my_plugin
```

### Deactivate Plugin
```bash
icarus plugin deactivate my_plugin
```

### Configure Plugin
```bash
icarus plugin configure my_plugin --setting value
```

### Uninstall Plugin
```bash
icarus plugin uninstall my_plugin
```

## Security Considerations

### Plugin Validation

All plugins undergo security validation:

1. **Code Analysis**: AST analysis for dangerous patterns
2. **Import Validation**: Check for dangerous imports
3. **Permission Validation**: Verify requested permissions
4. **Dependency Validation**: Check for risky dependencies

### Sandboxing

Plugins run in sandboxed environments with:

- Restricted file system access
- Limited network access
- Controlled import system
- Safe built-in functions

### Trust System

- **Trusted Plugins**: Bypass some security checks
- **Blocked Plugins**: Explicitly blocked from loading
- **Risk Assessment**: Automatic risk level calculation

## Best Practices

### Plugin Development

1. **Follow Security Guidelines**: Use minimal permissions
2. **Handle Errors Gracefully**: Implement proper error handling
3. **Document Your Plugin**: Provide clear documentation
4. **Test Thoroughly**: Test in different environments
5. **Version Properly**: Use semantic versioning

### Security

1. **Request Minimal Permissions**: Only request necessary permissions
2. **Validate Inputs**: Always validate user inputs
3. **Use Safe APIs**: Prefer plugin API over direct system access
4. **Handle Sensitive Data Carefully**: Encrypt sensitive information

### Performance

1. **Lazy Loading**: Load resources only when needed
2. **Async Operations**: Use async for long-running operations
3. **Memory Management**: Clean up resources properly
4. **Efficient Algorithms**: Use efficient data structures and algorithms

## Troubleshooting

### Common Issues

1. **Plugin Not Loading**: Check manifest syntax and file paths
2. **Security Validation Failed**: Review security requirements
3. **Import Errors**: Verify dependencies are installed
4. **Permission Denied**: Check requested permissions

### Debug Mode

Enable debug logging:
```python
import logging
logging.getLogger('cli.plugins').setLevel(logging.DEBUG)
```

### Plugin Development Tools

Use the plugin development tools:
```bash
icarus plugin validate /path/to/plugin
icarus plugin test /path/to/plugin
icarus plugin package /path/to/plugin
```

## API Reference

See the complete API documentation in the source code:

- `cli/plugins/api.py` - Main plugin API
- `cli/plugins/models.py` - Data models and structures
- `cli/plugins/manager.py` - Plugin manager
- `cli/plugins/security.py` - Security system
- `cli/plugins/discovery.py` - Plugin discovery

## Contributing

To contribute to the plugin system:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## License

The plugin system is part of the ICARUS CLI and follows the same license terms.
