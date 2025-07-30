# ICARUS CLI User Guide

This guide provides instructions for using the ICARUS Command Line Interface (CLI).

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation Steps

1. Create and activate a virtual environment (recommended):

```bash
# Create virtual environment
python -m venv icarus-env

# Activate on Windows
icarus-env\Scripts\activate

# Activate on macOS/Linux
source icarus-env/bin/activate
```

2. Install the ICARUS CLI:

```bash
pip install icarus-cli
```

3. Verify installation:

```bash
icarus --version
```

## Getting Started

### Launch the CLI

To launch the CLI with the full terminal user interface:

```bash
icarus
```

This will open the Textual-based terminal user interface.

### Command Line Usage

The CLI can also be used with direct commands:

```bash
# Get help
icarus --help

# Run an airfoil analysis
icarus analyze airfoil NACA0012 --alpha 5.0

# Run an airplane analysis
icarus analyze airplane my_airplane.avl
```

## Terminal User Interface

### Navigation

- Use arrow keys to navigate between UI elements
- Use Tab to cycle through focusable elements
- Use Enter to select or activate elements
- Use Escape to go back or cancel

### Main Screens

- **Dashboard**: Overview of recent analyses and quick actions
- **Analysis**: Configure and run analyses
- **Results**: View and manage analysis results
- **Workflows**: Create and manage workflows
- **Settings**: Configure application settings

### Keyboard Shortcuts

- `Ctrl+Q`: Quit the application
- `Ctrl+H`: Show help
- `Ctrl+S`: Save current work
- `F1`: Show keyboard shortcuts
- `F5`: Refresh current screen

## Running Analyses

### Airfoil Analysis

1. Navigate to the Analysis screen
2. Select "Airfoil Analysis"
3. Choose an airfoil (e.g., NACA0012) or upload a custom airfoil
4. Configure analysis parameters:
   - Angle of attack (alpha)
   - Reynolds number
   - Mach number
5. Select a solver (e.g., XFoil)
6. Click "Run Analysis"
7. View results in the Results screen

### Airplane Analysis

1. Navigate to the Analysis screen
2. Select "Airplane Analysis"
3. Choose an airplane configuration or upload a custom configuration
4. Configure analysis parameters:
   - Flight conditions
   - Control surface deflections
5. Select a solver (e.g., AVL)
6. Click "Run Analysis"
7. View results in the Results screen

## Working with Workflows

### Creating a Workflow

1. Navigate to the Workflows screen
2. Click "New Workflow"
3. Add analysis steps using the drag-and-drop interface
4. Configure each step's parameters
5. Connect steps to define dependencies
6. Save the workflow with a name and description

### Running a Workflow

1. Navigate to the Workflows screen
2. Select a workflow from the list
3. Click "Run Workflow"
4. Monitor progress in the workflow execution screen
5. View results when the workflow completes

### Managing Workflows

- **Save**: Save a workflow for future use
- **Export**: Export a workflow to share with others
- **Import**: Import a workflow from a file
- **Delete**: Remove a workflow from the system

## Data Management

### Importing Data

1. Navigate to the Data screen
2. Click "Import Data"
3. Select the data type (airfoil, airplane, etc.)
4. Choose a file to import
5. Configure import options
6. Click "Import"

### Exporting Results

1. Navigate to the Results screen
2. Select a result to export
3. Click "Export"
4. Choose an export format (CSV, JSON, etc.)
5. Configure export options
6. Click "Export"

### Managing Data

- **Browse**: Browse available data
- **Search**: Search for specific data
- **Delete**: Remove data from the system
- **Edit**: Edit data properties and metadata

## Collaboration

### Creating a Collaboration Session

1. Navigate to the Collaboration screen
2. Click "New Session"
3. Configure session settings:
   - Session name
   - Access permissions
4. Click "Create Session"
5. Share the session ID with collaborators

### Joining a Session

1. Navigate to the Collaboration screen
2. Click "Join Session"
3. Enter the session ID
4. Click "Join"

### Collaborating

- **Chat**: Communicate with other session participants
- **Annotations**: Add annotations to analyses and results
- **Shared Control**: Take turns controlling the interface
- **File Sharing**: Share files with other participants

## Customization

### Themes

1. Navigate to the Settings screen
2. Select "Themes"
3. Choose a theme from the list
4. Customize theme colors and styles
5. Save changes

### Layout

1. Navigate to the Settings screen
2. Select "Layout"
3. Configure layout options:
   - Screen layout
   - Widget visibility
   - Panel sizes
4. Save changes

### Keyboard Shortcuts

1. Navigate to the Settings screen
2. Select "Keyboard Shortcuts"
3. View and customize keyboard shortcuts
4. Save changes

## Advanced Features

### Plugin Management

1. Navigate to the Settings screen
2. Select "Plugins"
3. View installed plugins
4. Install, update, or remove plugins
5. Configure plugin settings

### Scripting

The CLI supports scripting for automation:

```bash
# Run a script
icarus script run my_script.py

# Create a new script
icarus script create my_script.py
```

### API Access

The CLI provides a REST API for programmatic access:

```bash
# Start API server
icarus api start

# Get API documentation
icarus api docs
```

## Troubleshooting

### Common Issues

- **Solver not found**: Ensure the required solver is installed and in your PATH
- **Analysis fails**: Check input parameters and file formats
- **UI rendering issues**: Try a different terminal emulator or adjust terminal settings
- **Performance problems**: Check system resources and reduce analysis complexity

### Logging

The CLI logs information to help diagnose issues:

```bash
# Enable debug logging
icarus --debug

# View logs
icarus logs show

# Export logs
icarus logs export my_logs.txt
```

### Getting Help

- **In-app Help**: Press F1 or Ctrl+H for in-app help
- **Documentation**: Visit the [ICARUS documentation website](https://icarus-docs.example.com)
- **Community Forum**: Join the [ICARUS community forum](https://icarus-forum.example.com)
- **Issue Tracker**: Report issues on the [GitHub issue tracker](https://github.com/example/icarus/issues)
