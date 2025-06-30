# ICARUS CLI v2.0 - Enhanced Command Line Interface

## 🚀 Overview

ICARUS CLI v2.0 is a complete redesign of the command-line interface for ICARUS Aerodynamics, featuring:

- **Advanced State Management**: Persistent sessions, configuration management, and history tracking
- **Modern UI Framework**: Rich theming, layout management, and interactive components
- **Workflow Automation**: Template-based workflows and batch processing
- **Smart Validation**: Comprehensive input validation and error handling
- **Enhanced UX**: Intuitive navigation, progress tracking, and user feedback

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    ICARUS CLI v2.0                          │
├─────────────────────────────────────────────────────────────┤
│  🎨 UI Layer (Rich Components)                              │
│  ├── Theme Manager                                          │
│  ├── Layout Manager                                         │
│  ├── Progress Manager                                       │
│  └── Notification System                                    │
├─────────────────────────────────────────────────────────────┤
│  🧠 State Management                                        │
│  ├── Session Manager                                        │
│  ├── Configuration Manager                                  │
│  ├── Cache Manager                                          │
│  └── History Manager                                        │
├─────────────────────────────────────────────────────────────┤
│  🔧 Core Services                                           │
│  ├── Workflow Engine                                        │
│  ├── Template Manager                                       │
│  ├── Validation Service                                     │
│  └── Export/Import Service                                  │
├─────────────────────────────────────────────────────────────┤
│  📊 Analysis Modules                                        │
│  ├── Airfoil Analysis                                       │
│  ├── Airplane Analysis                                      │
│  ├── Visualization                                          │
│  └── Batch Processing                                       │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install rich rich-cli typer

# Run the enhanced CLI
python cli/enhanced_main.py interactive
```

### Basic Usage

```bash
# Interactive mode (recommended)
icarus interactive

# Direct commands
icarus airfoil analyze naca2412 --solver xfoil
icarus airplane analyze my_airplane --solver avl
icarus workflow list
```

## 🎨 Features

### 1. State Management

- **Session Persistence**: Your analysis state is automatically saved and restored
- **Configuration Management**: User preferences and settings are stored persistently
- **History Tracking**: All operations are logged for reference and debugging

```python
# Session information is automatically managed
session_info = session_manager.get_session_info()
print(f"Session ID: {session_info['session_id']}")
print(f"Duration: {session_info['duration']}")
```

### 2. Modern UI Framework

- **Theme System**: Multiple themes (Default, Dark, Light, Aerospace, Scientific)
- **Layout Management**: Responsive layouts with sidebars and content areas
- **Progress Tracking**: Real-time progress bars and status updates
- **Notification System**: Success, warning, error, and info notifications

```python
# Change theme
theme_manager.set_theme(Theme.AEROSPACE)

# Send notifications
notification_system.success("Analysis completed!")
notification_system.warning("High angle of attack detected")
```

### 3. Workflow Automation

- **Template System**: Pre-defined workflows for common analysis tasks
- **Batch Processing**: Analyze multiple airfoils or airplanes efficiently
- **Custom Workflows**: Create and save your own workflow templates

```python
# Execute a workflow
workflow_engine.start_workflow("Standard Airfoil Analysis")
workflow_engine.execute_step(workflow.steps[0])

# Create custom template
template = WorkflowTemplate(
    name="My Custom Analysis",
    description="Custom analysis workflow",
    type=WorkflowType.AIRFOIL_ANALYSIS,
    steps=[...]
)
template_manager.save_template(template)
```

### 4. Smart Validation

- **Input Validation**: Comprehensive validation for all user inputs
- **Error Recovery**: Graceful error handling with helpful messages
- **Data Integrity**: Ensures data consistency across operations

```python
# Validate airfoil data
airfoil_data = {
    "name": "naca2412",
    "reynolds": 1e6,
    "angles": "0:15:16"
}
errors = validation_service.validate_data(airfoil_data, "airfoil")
```

### 5. Export/Import Services

- **Multiple Formats**: JSON, CSV, YAML, and plain text export
- **Report Generation**: Automated report creation with various formats
- **Data Import**: Import data from external sources

```python
# Export results
export_service.export_data(results, "output.json", "json")
export_service.export_data(results, "output.csv", "csv")

# Create reports
report = export_service.create_report(data, "detailed")
```

## 📋 Available Commands

### Main Commands

- `icarus interactive` - Launch interactive mode
- `icarus --version` - Show version information
- `icarus --help` - Show help

### Airfoil Analysis

- `icarus airfoil analyze <airfoil>` - Analyze single airfoil
- `icarus airfoil batch` - Batch airfoil analysis
- `icarus airfoil visualize` - Visualize airfoil results

### Airplane Analysis

- `icarus airplane analyze <airplane>` - Analyze single airplane
- `icarus airplane batch` - Batch airplane analysis
- `icarus airplane visualize` - Visualize airplane results

### Visualization

- `icarus visualization polar <airfoil>` - Plot airfoil polar
- `icarus visualization airplane <airplane>` - Plot airplane results
- `icarus visualization compare` - Compare multiple results

### Workflow Management

- `icarus workflow list` - List available workflows
- `icarus workflow execute <name>` - Execute workflow
- `icarus workflow create` - Create new workflow template
- `icarus workflow manage` - Manage workflow templates

## 🎨 Themes

### Available Themes

1. **Default** - Clean, professional appearance
2. **Dark** - Dark mode for low-light environments
3. **Light** - Light mode for bright environments
4. **Aerospace** - Aviation-themed colors
5. **Scientific** - Research-focused appearance

### Changing Themes

```bash
# Via command line
icarus interactive --theme aerospace

# Via settings menu
# Navigate to Settings → Theme Settings
```

## ⚙️ Configuration

### Configuration File

Configuration is stored in `~/.icarus/config.json`:

```json
{
  "theme": "aerospace",
  "database_path": "./Data",
  "auto_save": true,
  "show_progress": true,
  "confirm_exit": true,
  "max_history": 100,
  "default_solver": "xfoil",
  "default_reynolds": 1000000,
  "default_angles": "0:15:16",
  "ui": {
    "show_banner": true,
    "show_session_info": true,
    "compact_mode": false,
    "color_scheme": "auto"
  }
}
```

### Environment Variables

- `ICARUS_DATABASE_PATH` - Database path
- `ICARUS_THEME` - Default theme
- `ICARUS_CONFIG_DIR` - Configuration directory

## 🔧 Development

### Project Structure

```
cli/
├── core/                    # Core framework
│   ├── state.py            # State management
│   ├── ui.py               # UI framework
│   ├── workflow.py         # Workflow engine
│   └── services.py         # Core services
├── enhanced_main.py        # Main CLI application
├── test_enhanced_cli.py    # Test suite
└── README_ENHANCED.md      # This file
```

### Adding New Features

1. **New Commands**: Add to the appropriate Typer app in `enhanced_main.py`
2. **New Workflows**: Create workflow templates in `core/workflow.py`
3. **New Themes**: Add theme definitions in `core/ui.py`
4. **New Validations**: Add validation rules in `core/services.py`

### Testing

```bash
# Run test suite
python cli/test_enhanced_cli.py

# Test specific component
python -c "from cli.core.state import SessionManager; print('State management works!')"
```

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check database path in configuration
   - Ensure database directory exists
   - Verify file permissions

2. **Theme Not Applied**
   - Check theme name spelling
   - Verify theme exists in theme manager
   - Restart CLI after theme change

3. **Workflow Execution Failed**
   - Check workflow template validity
   - Verify all required parameters
   - Check solver availability

### Debug Mode

```bash
# Enable debug output
export ICARUS_DEBUG=1
icarus interactive
```

### Log Files

Logs are stored in `~/.icarus/logs/`:
- `session.log` - Session information
- `workflow.log` - Workflow execution logs
- `error.log` - Error logs

## 🤝 Contributing

### Guidelines

1. **Code Style**: Follow PEP 8 and use type hints
2. **Documentation**: Add docstrings and update README
3. **Testing**: Add tests for new features
4. **Backward Compatibility**: Maintain compatibility with existing CLI

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd icarus

# Install development dependencies
pip install -e .

# Run tests
python cli/test_enhanced_cli.py
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Rich**: For the beautiful terminal UI framework
- **Typer**: For the modern CLI framework
- **ICARUS Team**: For the underlying aerodynamics library

## 📞 Support

- **Documentation**: See the main ICARUS documentation
- **Issues**: Report bugs on GitHub
- **Discussions**: Join community discussions
- **Email**: Contact the development team

---

**ICARUS CLI v2.0** - Making aircraft design analysis more accessible and efficient! 🚁✈️ 