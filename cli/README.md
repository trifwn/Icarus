# ICARUS CLI v2.0

A modern, interactive Command Line Interface for ICARUS Aerodynamics, featuring a Textual-based Terminal User Interface (TUI) and comprehensive workflow automation.

## Features

### 🎯 **Modern TUI Interface**
- **Interactive REPL**: Run Python code, create objects, and manage namespace
- **Separated Analysis**: Dedicated forms for Airfoil and Airplane analysis
- **Real-time Results**: Live progress tracking and result visualization
- **Object Browser**: View and manage created objects
- **Workflow Management**: Execute and manage analysis workflows

### 🔧 **Core Functionality**
- **State Management**: Persistent sessions and configuration
- **Workflow Automation**: Template-based analysis workflows
- **Validation & Export**: Built-in validation and export services
- **Theme Support**: Customizable UI themes

### 📝 **Enhanced REPL**
- **Code History**: Navigate through executed code
- **Object Persistence**: Objects created in REPL available throughout app
- **Copy/Paste Support**: Full clipboard integration
- **Error Handling**: Comprehensive error reporting and debugging

## Quick Start

### Installation

```bash
# Install ICARUS CLI dependencies
pip install -r requirements.txt

# Run the CLI
python -m cli
```

### Basic Usage

1. **Launch the TUI**:
   ```bash
   python -m cli
   ```

2. **Use the REPL**:
   - Go to the "REPL" tab
   - Enter Python code to create objects
   - Use `namespace.add_object()` to save objects
   - Objects are available in analysis forms

3. **Run Analysis**:
   - Go to "Analysis" tab
   - Choose "Airfoil" or "Airplane" analysis
   - Fill in parameters or select objects from namespace
   - Click "Run Analysis"

## Architecture

### Directory Structure

```
cli/
├── __main__.py             # Main entry point
├── icarus.py               # CLI entry point with Typer
├── tui_app.py              # Textual TUI application
├── tui_styles.css          # TUI styling
├── app/                    # Main application components
│   ├── main_app.py         # Main TUI app
│   ├── event_system.py     # Event handling system
│   ├── screen_manager.py   # Screen management
│   └── state_manager.py    # State management
├── core/                   # Core functionality
│   ├── config.py           # Configuration management
│   ├── services/           # Core services
│   ├── state/              # State management
│   ├── ui/                 # UI components
│   └── workflow/           # Workflow engine
├── api/                    # API layer for web migration
├── commands/               # CLI commands
├── integration/            # ICARUS integration
├── workflows/              # Workflow definitions
└── visualization/          # Visualization components
```

### Key Components

#### **TUI Application (`tui/app.py`)**
- Main Textual application
- Tabbed interface with Analysis, REPL, Workflows, Results, Settings
- Integrated REPL with namespace management
- Real-time progress tracking

#### **Core Modules (`core/`)**
- **State Management**: Session persistence, configuration
- **Workflow Engine**: Template-based analysis workflows
- **Services**: Validation, export, notification systems
- **TUI Integration**: Event handling and data binding

#### **REPL System**
- **Code Editor**: Syntax highlighting, history navigation
- **Object Browser**: View and manage namespace objects
- **Execution Engine**: Safe code execution with namespace access
- **History Management**: Code and output history

## Usage Examples

### Creating Objects in REPL

```python
# Create an airfoil
from ICARUS.airfoils import Airfoil
naca2412 = Airfoil.naca("2412")
namespace.add_object("naca2412", naca2412, "airfoil")

# Create an airplane
from ICARUS.vehicle import Airplane
boeing737 = Airplane.from_file("boeing737.json")
namespace.add_object("boeing737", boeing737, "airplane")
```

### Running Analysis

1. **Airfoil Analysis**:
   - Enter airfoil name or select from namespace
   - Choose solver (XFoil, Foil2Wake, OpenFoam)
   - Set Reynolds number and angle range
   - Click "Run Analysis"

2. **Airplane Analysis**:
   - Enter airplane name or select from namespace
   - Choose 3D solver (AVL, GenuVP, OpenFoam)
   - Set flight state parameters
   - Click "Run Analysis"

### Workflow Management

- Browse available workflows in the "Workflows" tab
- Select and execute workflows
- Monitor progress in real-time
- View results in the "Results" tab

## Configuration

### Settings

Access settings in the "Settings" tab:
- **Theme**: Choose UI theme (Default, Dark, Light)
- **Database Path**: Set ICARUS database location
- **Auto Save**: Enable/disable automatic saving

### Session Management

- Sessions are automatically saved
- Objects persist between sessions
- Configuration is stored in `~/.icarus/`

## Keyboard Shortcuts

### General
- `Ctrl+Q`: Quit application
- `Ctrl+H`: Show help
- `F5`: Refresh

### Navigation
- `Ctrl+A`: Analysis tab
- `Ctrl+R`: Results tab
- `Ctrl+S`: Settings tab
- `Ctrl+N`: Notifications tab

### REPL
- `Ctrl+Enter`: Execute code
- `Ctrl+C`: Copy selection
- `Ctrl+V`: Paste
- `Ctrl+Z`: Undo
- `Ctrl+Y`: Redo

## Development

### Adding New Features

1. **TUI Widgets**: Add to `tui/widgets/`
2. **Core Services**: Add to `core/`
3. **Workflows**: Extend `core/workflow.py`
4. **Examples**: Add to `examples/`

### Testing

```bash
# Run tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_tui.py
```

## Web Migration Readiness

The CLI architecture is designed to support future migration to a web application:

### Key Features for Web Migration

- **API-First Design**: Core functionality exposed through REST APIs
- **UI Abstraction**: Separation of business logic from presentation
- **Data Serialization**: Standard JSON-based data exchange
- **Authentication**: Support for both local and web authentication
- **WebSocket Support**: Real-time features implemented with WebSocket

### Migration Path

1. **API Development**: REST API layer alongside TUI
2. **Frontend Framework Selection**: Evaluate modern web frameworks
3. **Feature Parity**: Migrate TUI features to web interface
4. **Deployment Options**: Support both TUI and web interfaces

## Troubleshooting

### Common Issues

1. **TUI not launching**: Check Textual installation
2. **REPL errors**: Verify ICARUS installation
3. **Object not found**: Check namespace browser
4. **Analysis fails**: Verify solver installation

### Debug Mode

```bash
# Run with debug output
python -m cli --verbose
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This CLI is part of the ICARUS project and follows the same license terms.

## Support

For issues and questions:
- Check the troubleshooting section
- Review the examples in `examples/`
- Open an issue on the ICARUS repository
