# ICARUS CLI v2.0 - Restructured Architecture

## Overview

The ICARUS CLI has been completely restructured to provide a modern, stateful, and extensible command-line interface for aircraft design and analysis. This new architecture replaces the old inquirer-based CLI with a rich, typer-based system that includes advanced state management, workflow automation, and seamless TUI integration.

## Architecture

### Core Framework (`cli/core/`)

The core framework provides the foundation for all CLI functionality:

#### State Management (`state.py`)
- **SessionManager**: Persistent session state with airfoils, airplanes, and results
- **ConfigManager**: User preferences and configuration with smart defaults
- **HistoryManager**: Operation history and audit trail

#### UI Framework (`ui.py`)
- **ThemeManager**: Multiple themes (default, dark, light, aerospace, scientific)
- **LayoutManager**: Screen organization and layout management
- **ProgressManager**: Progress tracking with rich progress bars
- **NotificationSystem**: Success, warning, error, and info notifications
- **UIComponents**: Reusable menu, form, and status components

#### Workflow Engine (`workflow.py`)
- **WorkflowEngine**: Automated analysis workflows with step-by-step execution
- **TemplateManager**: Custom workflow templates and batch processing
- **Built-in Workflows**: Standard airfoil analysis, airplane analysis, batch processing

#### Services (`services.py`)
- **ValidationService**: Input validation with configurable rules
- **ExportService**: Data export/import in multiple formats (JSON, CSV, YAML, TXT)
- **Report Generation**: Summary, detailed, and table reports

#### TUI Integration (`tui_integration.py`)
- **Event System**: Real-time event handling between CLI and TUI
- **Manager Classes**: TUI-specific managers for session, workflow, analysis, export, settings
- **Seamless Switching**: Switch between CLI and TUI modes

### TUI Framework (`cli/tui/`)

#### Widgets (`tui/widgets/`)
- **SessionWidget**: Display and manage session information
- **WorkflowWidget**: Browse and execute workflows
- **AnalysisWidget**: Configure and run analyses with validation
- **ResultsWidget**: View and export analysis results
- **ProgressWidget**: Real-time progress tracking
- **NotificationWidget**: Notification history and filtering

#### Utilities (`tui/utils/`)
- **EventHelper**: Event subscription and management
- **ThemeHelper**: Theme application and CSS generation
- **DataHelper**: Session data and export utilities
- **ValidationHelper**: Form validation and error handling

## Features

### Enhanced CLI Mode
- **Rich Interface**: Beautiful tables, panels, and progress bars
- **Stateful Sessions**: Persistent state across CLI sessions
- **Smart Defaults**: Intelligent parameter suggestions
- **Workflow Automation**: Pre-configured analysis workflows
- **Export/Import**: Multiple format support for results
- **Theme Support**: Multiple visual themes
- **IPython Integration**: Drop into interactive shell with context

### TUI Mode
- **Interactive Forms**: Dynamic form generation with validation
- **Real-time Updates**: Live progress and status updates
- **Event-driven**: Reactive UI with event system
- **Multi-tab Interface**: Organized workspace with tabs
- **REPL Integration**: Interactive code execution
- **Object Browser**: Browse created objects and results

### Workflow System
- **Standard Workflows**: Pre-built analysis workflows
- **Custom Templates**: Create and save custom workflows
- **Batch Processing**: Analyze multiple airfoils/airplanes
- **Step-by-step Execution**: Detailed progress tracking
- **Error Handling**: Graceful error recovery

### Validation & Export
- **Smart Validation**: Context-aware input validation
- **Multiple Formats**: JSON, CSV, YAML, TXT export
- **Report Generation**: Automated report creation
- **Data Import**: Import previous results and configurations

## Usage

### Basic Usage

```bash
# Start interactive CLI
python cli/main.py

# Or use the enhanced CLI directly
python cli/enhanced_main.py

# Command-line usage
python cli/main.py airfoil analyze naca2412 --solver xfoil
python cli/main.py airplane analyze boeing737 --solver avl
python cli/main.py visualization polar naca2412
```

### Interactive Mode

1. **Main Menu**: Choose from airfoil analysis, airplane analysis, visualization, workflows, settings, help, TUI mode, or IPython shell
2. **Analysis Menus**: Configure parameters, run analyses, view results
3. **Workflow Management**: Browse, execute, and create workflows
4. **Settings**: Configure themes, database paths, preferences
5. **TUI Mode**: Switch to interactive Textual UI

### TUI Mode

1. **Launch TUI**: Select "Launch TUI Mode" from main menu
2. **Multi-tab Interface**: 
   - Analysis: Configure and run analyses
   - Workflows: Browse and execute workflows
   - Results: View and export results
   - REPL: Interactive code execution
   - Settings: Configure preferences
3. **Real-time Updates**: Live progress and notifications
4. **Event System**: Reactive UI updates

### Workflow System

```python
# Standard workflows
- Standard Airfoil Analysis: Complete airfoil analysis with multiple solvers
- Standard Airplane Analysis: 3D airplane analysis with 3D solvers
- Batch Airfoil Analysis: Analyze multiple airfoils

# Custom workflows
- Create templates from existing workflows
- Modify parameters and steps
- Save and reuse custom workflows
```

## Configuration

### User Configuration (`~/.icarus/config.json`)
```json
{
  "theme": "default",
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

### Session State (`~/.icarus/session.json`)
```json
{
  "session_id": "uuid",
  "start_time": "2024-01-01T00:00:00",
  "current_workflow": "Standard Airfoil Analysis",
  "current_step": "analysis_execution",
  "airfoils": ["naca2412", "naca0012"],
  "airplanes": ["boeing737"],
  "last_results": {
    "naca2412_analysis": {...}
  },
  "user_preferences": {...}
}
```

## Development

### Adding New Features

1. **Core Framework**: Add new functionality to appropriate core module
2. **TUI Integration**: Create widgets and utilities for TUI integration
3. **Event System**: Use events for real-time updates
4. **Validation**: Add validation rules for new data types
5. **Export**: Add export support for new data formats

### Testing

```bash
# Run core framework tests
python -m pytest cli/tests/test_core_framework.py

# Run TUI integration tests
python -m pytest cli/tests/test_tui_integration.py

# Run all tests
python -m pytest cli/tests/
```

### Extending Workflows

```python
# Add new workflow type
class WorkflowType(Enum):
    CUSTOM_ANALYSIS = "custom_analysis"

# Create workflow template
workflow = WorkflowTemplate(
    name="Custom Analysis",
    description="Custom analysis workflow",
    type=WorkflowType.CUSTOM_ANALYSIS,
    steps=[...],
    parameters={...},
    metadata={...}
)

# Register with workflow engine
workflow_engine.workflows[workflow.name] = workflow
```

## Migration from v1.0

### Breaking Changes
- Removed inquirer dependency
- New command structure with typer
- Stateful sessions replace stateless operation
- New workflow system replaces manual analysis

### Migration Guide
1. **Install Dependencies**: `pip install rich typer textual`
2. **Update Scripts**: Use new command structure
3. **Migrate Config**: Convert old config to new format
4. **Test Workflows**: Verify workflows work with new system

### Backward Compatibility
- Legacy commands still available but deprecated
- Old config files automatically migrated
- Session data preserved across versions

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies installed
   ```bash
   pip install rich typer textual ipython
   ```

2. **TUI Not Available**: Install textual
   ```bash
   pip install textual
   ```

3. **Database Issues**: Check database path in config
   ```bash
   python cli/main.py --database /path/to/database
   ```

4. **Theme Issues**: Reset to default theme
   ```bash
   # In settings menu, select "Reset to Defaults"
   ```

### Debug Mode

```bash
# Enable debug logging
export ICARUS_DEBUG=1
python cli/main.py
```

## Performance

### Optimizations
- Lazy loading of heavy modules
- Caching of frequently accessed data
- Asynchronous operations where possible
- Efficient event system

### Memory Management
- Automatic cleanup of old sessions
- Limited history retention
- Efficient data structures

## Contributing

1. **Fork Repository**: Create feature branch
2. **Add Tests**: Include tests for new features
3. **Update Documentation**: Document new functionality
4. **Submit PR**: Pull request with description

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Rich**: Beautiful terminal output
- **Typer**: Modern CLI framework
- **Textual**: TUI framework
- **ICARUS**: Aerodynamics library 