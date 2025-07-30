# ICARUS CLI Plugin Development SDK

The ICARUS CLI Plugin Development SDK provides comprehensive tools for creating, testing, packaging, and distributing plugins for the ICARUS CLI system.

## Features

- **Plugin Generator**: Create new plugins from templates
- **Plugin Validator**: Comprehensive validation of plugin structure and code
- **Plugin Tester**: Automated testing framework for plugins
- **Plugin Packager**: Package plugins for distribution
- **Marketplace Integration**: Publish and discover plugins
- **Documentation Generator**: Automatic documentation generation

## Installation

The SDK is included with the ICARUS CLI system. To use it programmatically:

```python
from icarus_cli.plugins.sdk import (
    PluginGenerator,
    PluginValidator,
    PluginTester,
    PluginPackager,
    PluginMarketplace,
    PluginDocGenerator
)
```

## Command Line Interface

The SDK includes a command-line tool for plugin development:

```bash
# Generate a new plugin
icarus-plugin-sdk generate --name my-plugin --template basic --author "Your Name"

# Validate a plugin
icarus-plugin-sdk validate /path/to/plugin

# Test a plugin
icarus-plugin-sdk test /path/to/plugin

# Package a plugin
icarus-plugin-sdk package /path/to/plugin --output my-plugin.zip

# Search for plugins
icarus-plugin-sdk marketplace search --query "analysis"

# Generate documentation
icarus-plugin-sdk docs /path/to/plugin --formats markdown,html
```

## Plugin Templates

The SDK includes several built-in templates:

### Basic Plugin
A minimal plugin with basic functionality:
- Simple menu integration
- Command registration
- Configuration support

### Analysis Plugin
For plugins that add custom analysis capabilities:
- Analysis class structure
- Result processing
- Data integration

### Visualization Plugin
For plugins that add custom visualization:
- Plotting utilities
- Chart generation
- Export capabilities

### Integration Plugin
For plugins that integrate with external tools:
- API connectors
- Data synchronization
- Authentication handling

### Workflow Plugin
For plugins that add custom workflow steps:
- Workflow step implementation
- Template management
- Progress tracking

## Development Workflow

### 1. Generate Plugin

```bash
icarus-plugin-sdk generate \
  --name my-analysis-plugin \
  --template analysis \
  --author "Your Name" \
  --email "your.email@example.com" \
  --description "Custom analysis plugin"
```

### 2. Develop Plugin

Edit the generated files:
- `plugin.py`: Main plugin implementation
- `manifest.json`: Plugin metadata
- `analysis.py`: Custom analysis logic (for analysis plugins)

### 3. Validate Plugin

```bash
icarus-plugin-sdk validate my-analysis-plugin/
```

### 4. Test Plugin

```bash
# Generate test suite
icarus-plugin-sdk test my-analysis-plugin/ --generate-tests

# Run tests
icarus-plugin-sdk test my-analysis-plugin/ --performance --security
```

### 5. Generate Documentation

```bash
icarus-plugin-sdk docs my-analysis-plugin/ --formats markdown,html
```

### 6. Package Plugin

```bash
icarus-plugin-sdk package my-analysis-plugin/ \
  --output my-analysis-plugin-1.0.0.zip \
  --include-tests \
  --include-docs
```

### 7. Publish Plugin

```bash
icarus-plugin-sdk marketplace publish \
  --plugin-path my-analysis-plugin-1.0.0.zip \
  --marketplace official
```

## API Reference

### PluginGenerator

```python
from icarus_cli.plugins.sdk import PluginGenerator

generator = PluginGenerator()

# List available templates
templates = generator.list_templates()

# Generate plugin
success = generator.generate_plugin(
    plugin_name="my-plugin",
    template_name="basic",
    output_dir="./plugins",
    author_name="Your Name",
    author_email="your.email@example.com"
)
```

### PluginValidator

```python
from icarus_cli.plugins.sdk import PluginValidator

validator = PluginValidator()

# Validate plugin
result = validator.validate_plugin("/path/to/plugin")

if result.is_valid:
    print("Plugin is valid!")
else:
    for error in result.errors:
        print(f"Error: {error}")
```

### PluginTester

```python
from icarus_cli.plugins.sdk import PluginTester

tester = PluginTester()

# Run comprehensive tests
results = tester.run_plugin_tests(
    "/path/to/plugin",
    test_config={
        'performance_tests': True,
        'security_tests': True
    }
)

print(f"Overall status: {results['overall_status']}")
```

### PluginPackager

```python
from icarus_cli.plugins.sdk import PluginPackager

packager = PluginPackager()

# Package plugin
success = packager.package_plugin(
    plugin_path="/path/to/plugin",
    output_path="plugin.zip",
    format="zip",
    include_tests=True,
    validate=True
)
```

### PluginMarketplace

```python
from icarus_cli.plugins.sdk import PluginMarketplace

marketplace = PluginMarketplace()

# Search plugins
results = marketplace.search_plugins("analysis", limit=10)

# Download plugin
success = marketplace.download_plugin(
    plugin_id="analysis-plugin",
    marketplace="official",
    output_path="downloaded-plugin.zip"
)

# Publish plugin
success = marketplace.publish_plugin(
    plugin_package_path="my-plugin.zip",
    marketplace="official"
)
```

### PluginDocGenerator

```python
from icarus_cli.plugins.sdk import PluginDocGenerator

doc_generator = PluginDocGenerator()

# Generate documentation
docs = doc_generator.generate_documentation(
    plugin_path="/path/to/plugin",
    output_dir="./docs",
    formats=["markdown", "html"],
    include_api=True,
    include_examples=True
)
```

## Plugin Structure

A typical plugin structure:

```
my-plugin/
├── __init__.py          # Plugin package initialization
├── plugin.py            # Main plugin class
├── manifest.json        # Plugin metadata
├── README.md           # Plugin documentation
├── requirements.txt    # Python dependencies
├── tests/              # Test files
│   └── test_plugin.py
├── examples/           # Usage examples
│   └── example.py
└── docs/              # Generated documentation
    ├── README.md
    └── api.md
```

## Plugin Manifest

The `manifest.json` file contains plugin metadata:

```json
{
  "name": "my_plugin",
  "version": "1.0.0",
  "description": "A sample plugin",
  "author": {
    "name": "Your Name",
    "email": "your.email@example.com"
  },
  "type": "utility",
  "security_level": "safe",
  "main_module": "my_plugin",
  "main_class": "MyPlugin",
  "keywords": ["utility", "example"],
  "license": "MIT",
  "python_version": ">=3.8",
  "icarus_version": ">=1.0.0",
  "install_requires": [],
  "permissions": [],
  "default_config": {
    "enabled": true,
    "setting1": "value1"
  }
}
```

## Testing

The SDK provides comprehensive testing capabilities:

### Automated Tests

- **Basic Functionality**: Plugin loading, initialization, activation
- **API Usage**: Proper use of plugin API methods
- **Event Handling**: Event registration and handling
- **Configuration**: Configuration loading and validation
- **Error Handling**: Graceful error handling
- **Performance**: Initialization and execution timing
- **Security**: Code analysis for security issues

### Custom Tests

Generate custom test suites:

```bash
icarus-plugin-sdk test my-plugin/ --generate-tests --output tests/
```

This creates a complete test suite with:
- Unit tests for plugin methods
- Integration tests with ICARUS CLI
- Mock objects for testing
- Test fixtures and utilities

## Validation

The validator checks:

### Structure Validation
- Required files present
- Python package structure
- Manifest format and content

### Code Validation
- Python syntax correctness
- Plugin class inheritance
- Required method implementation
- Code quality metrics

### Security Validation
- Dangerous code patterns
- Unsafe imports
- Permission requirements

### Compatibility Validation
- Python version compatibility
- ICARUS CLI version compatibility
- Dependency requirements

## Packaging

Create distributable plugin packages:

### Package Formats
- ZIP archives (default)
- TAR.GZ archives
- TAR.BZ2 archives

### Package Contents
- Plugin source code
- Manifest and metadata
- Documentation (optional)
- Test files (optional)
- File integrity checksums

### Package Verification
- File integrity checking
- Manifest validation
- Security scanning

## Marketplace Integration

### Official Marketplace
- Curated plugins
- Security reviewed
- Quality assured

### Community Marketplace
- Community contributions
- Broader selection
- User ratings and reviews

### Private Marketplaces
- Enterprise deployments
- Custom plugin repositories
- Access control

## Documentation Generation

Automatically generate comprehensive documentation:

### Generated Documents
- README.md: Overview and usage
- User Guide: Detailed usage instructions
- Developer Guide: Architecture and development
- API Documentation: Complete API reference

### Supported Formats
- Markdown (default)
- HTML with styling
- reStructuredText

### Content Sources
- Plugin manifest metadata
- Code docstrings and comments
- Example files
- Test files

## Best Practices

### Plugin Development
1. Follow the plugin template structure
2. Implement comprehensive error handling
3. Use descriptive docstrings
4. Include usage examples
5. Write unit tests
6. Validate before packaging

### Security
1. Use appropriate security levels
2. Request minimal permissions
3. Validate all inputs
4. Avoid dangerous operations
5. Follow secure coding practices

### Performance
1. Minimize initialization time
2. Use asynchronous operations
3. Implement proper cleanup
4. Cache expensive operations
5. Monitor resource usage

### Documentation
1. Write clear descriptions
2. Include usage examples
3. Document configuration options
4. Provide troubleshooting guides
5. Keep documentation updated

## Contributing

To contribute to the SDK:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

## License

The ICARUS CLI Plugin Development SDK is licensed under the MIT License.

## Support

For support and questions:
- Documentation: [ICARUS CLI Docs](https://docs.icarus.example.com)
- Issues: [GitHub Issues](https://github.com/icarus/cli/issues)
- Community: [ICARUS Forum](https://forum.icarus.example.com)
