# ICARUS CLI Settings and Personalization System

## Overview

The ICARUS CLI Settings and Personalization System provides comprehensive configuration management with theme customization, workspace management, and settings backup/restore functionality. This system supports multiple configuration scopes and provides both GUI and CLI interfaces for settings management.

## Features

### ✅ Comprehensive Settings Management Interface
- **Multi-scope Settings**: Global, workspace, project, and session-level configurations
- **Real-time Updates**: Live preview and immediate application of settings changes
- **Settings Validation**: Automatic validation with error reporting and suggestions
- **Hierarchical Configuration**: Settings inheritance from global to project level

### ✅ Theme Customization with Live Preview
- **Multiple Theme Presets**: Aerospace Dark/Light, Scientific, Classic themes
- **Custom Theme Creation**: Full customization of colors, layout, and visual effects
- **Live Preview**: Real-time preview of theme changes before applying
- **Animation Controls**: Configurable animations, transitions, and visual effects
- **Layout Customization**: Adjustable sidebar width, transparency, and UI elements

### ✅ Workspace and Project-Specific Configurations
- **Workspace Management**: Create, switch, and manage multiple workspaces
- **Project Organization**: Project-specific settings within workspaces
- **Default Solver Configuration**: Per-workspace default solver settings
- **Data Directory Management**: Configurable paths for data, results, and templates
- **Backup Scheduling**: Automatic workspace backup with configurable intervals

### ✅ Settings Import/Export and Backup System
- **Multiple Export Formats**: JSON and YAML export formats
- **Selective Export**: Export specific scopes (global, workspace, project)
- **Settings Profiles**: Create and share settings profiles
- **Automatic Backups**: Scheduled backups with configurable retention
- **One-Click Restore**: Easy restoration from any backup point
- **Backup Cleanup**: Automatic cleanup of old backups

## Architecture

### Core Components

```
cli/core/
├── settings.py              # Core settings management
├── settings_integration.py  # Application integration
└── ui.py                   # Theme management

cli/tui/
├── settings_screen.py      # Settings UI interface
└── settings_styles.css     # UI styling

cli/commands/
└── settings_cli.py         # Command-line interface
```

### Settings Hierarchy

```
Global Settings (User-wide)
├── Theme Settings
├── User Preferences
└── Performance Settings

Workspace Settings
├── Workspace Configuration
├── Default Solvers
├── Data Directories
└── Backup Settings

Project Settings
├── Project Metadata
├── Analysis Preferences
├── Visualization Settings
└── Collaboration Settings
```

## Usage Examples

### Basic Settings Management

```python
from core.settings import SettingsManager

# Initialize settings manager
settings = SettingsManager()
settings.load_all_settings()

# Update theme settings
settings.update_theme_settings(
    theme_name="aerospace",
    color_scheme="dark",
    animations_enabled=True
)

# Update user preferences
settings.update_user_preferences(
    name="John Doe",
    startup_screen="analysis",
    max_memory_usage=2048
)

# Save settings
settings.save_all_settings()
```

### Workspace Management

```python
# Create a new workspace
settings.create_workspace(
    "aircraft_design",
    description="Aircraft design projects",
    default_solver_2d="xfoil",
    default_solver_3d="avl"
)

# Switch to workspace
settings.switch_workspace("aircraft_design")

# Create project in workspace
settings.create_project(
    "wing_optimization",
    description="Wing shape optimization study"
)
```

### Theme Customization

```python
# Apply theme preset
settings.apply_theme_preset("aerospace_dark")

# Custom theme configuration
settings.update_theme_settings(
    theme_name="aerospace",
    color_scheme="dark",
    layout_style="modern",
    sidebar_width=35,
    animations_enabled=True,
    animation_speed="fast",
    transparency_level=0.9,
    custom_colors={
        "primary": "#00bfff",
        "accent": "#ff6b35"
    }
)
```

### Import/Export Settings

```python
# Export settings
settings.export_settings(
    "my_settings.json",
    scope=SettingsScope.GLOBAL,
    format=SettingsFormat.JSON
)

# Import settings
settings.import_settings("my_settings.json", merge=True)
```

### Backup and Restore

```python
# Create backup
backup_name = settings.create_backup("before_major_changes")

# List backups
backups = settings.list_backups()

# Restore backup
settings.restore_backup(backup_name)

# Cleanup old backups
settings.cleanup_old_backups(max_backups=10)
```

## Command Line Interface

The settings system includes a comprehensive CLI for managing settings from the command line:

```bash
# Show current settings
icarus-settings show --scope=all --format=table

# Set a setting value
icarus-settings set theme.theme_name aerospace

# Export settings
icarus-settings export my_settings.json --scope=global --format=json

# Import settings
icarus-settings import my_settings.json --merge

# Create backup
icarus-settings backup create --name=pre_update_backup

# List workspaces
icarus-settings workspace list

# Apply theme preset
icarus-settings theme apply aerospace_dark
```

## Settings Categories

### Theme Settings
- **Theme Name**: Base theme (aerospace, scientific, default, classic)
- **Color Scheme**: Dark or light color scheme
- **Layout Style**: Modern, classic, or compact layout
- **Visual Effects**: Animations, transitions, transparency
- **UI Elements**: Icons, tooltips, status bar visibility
- **Typography**: Font size, family, line spacing
- **Custom Colors**: Override default color palette

### User Preferences
- **Personal Info**: Name, email, organization
- **Interface**: Startup screen, welcome screen, auto-save interval
- **Notifications**: Enable/disable, sound, duration
- **Performance**: Memory usage, CPU cores, cache size
- **Privacy**: Analytics, crash reporting, usage statistics

### Workspace Settings
- **Workspace Info**: Name, description, creation date
- **Paths**: Data, results, templates, cache directories
- **Default Analysis**: Default solvers for 2D/3D analysis
- **Workflow**: Auto-save workflows, validation, concurrency
- **Data Management**: Auto-backup, interval, compression

### Project Settings
- **Project Info**: Name, description, author, version
- **Analysis**: Preferred solvers, precision, convergence criteria
- **Visualization**: Plot style, figure format, DPI
- **Collaboration**: Sharing, collaborators, permissions

## Integration with Main Application

The settings system integrates seamlessly with the main ICARUS CLI application:

```python
from core.settings_integration import initialize_settings_integration

# Initialize integration
integration = initialize_settings_integration(app, settings_manager, theme_manager)

# Settings changes are automatically applied to the application
# Theme changes update the UI immediately
# Performance settings adjust resource usage
# Workspace settings configure default behaviors
```

## Settings Validation

The system includes comprehensive validation:

```python
# Validate all settings
issues = settings.validate_settings()

# Example validation rules:
# - Sidebar width: 10-100 pixels
# - Transparency: 0.0-1.0
# - Auto-save interval: minimum 30 seconds
# - Memory usage: minimum 128 MB
# - Backup interval: minimum 1 hour
```

## File Structure

Settings are stored in a hierarchical directory structure:

```
~/.icarus/
├── settings/
│   ├── theme.json          # Global theme settings
│   └── user.json           # User preferences
├── workspaces/
│   ├── workspace1/
│   │   ├── settings.json   # Workspace settings
│   │   └── projects/
│   │       └── project1/
│   │           └── settings.json  # Project settings
│   └── workspace2/
└── backups/
    ├── backup_20250722_144432/
    │   ├── settings.json
    │   ├── metadata.json
    │   └── workspace/
    └── backup_20250721_120000/
```

## Testing

The settings system includes comprehensive tests:

```bash
# Run all settings tests
python cli/test_settings_system.py

# Run settings demo
python cli/demo_settings_system.py
```

## Requirements Fulfilled

This implementation fulfills all requirements from the specification:

### ✅ Requirement 9.1: Comprehensive Settings Management
- Multi-scope settings (global, workspace, project)
- Real-time settings application
- Settings validation and error handling
- Hierarchical configuration system

### ✅ Requirement 9.2: Theme Customization with Live Preview
- Multiple theme presets
- Custom theme creation
- Live preview functionality
- Animation and visual effect controls

### ✅ Requirement 9.3: Workspace and Project-Specific Configurations
- Workspace management system
- Project organization within workspaces
- Per-workspace default configurations
- Data directory management

### ✅ Requirement 9.4: Settings Import/Export
- Multiple export formats (JSON, YAML)
- Selective scope export
- Settings profile management
- Merge and replace import options

### ✅ Requirement 9.5: Backup System
- Automatic backup creation
- Backup scheduling and retention
- One-click restore functionality
- Backup cleanup and management

## Future Enhancements

Potential future improvements:

1. **Cloud Sync**: Synchronize settings across devices
2. **Team Settings**: Shared team configuration templates
3. **Advanced Themes**: Theme marketplace and sharing
4. **Settings Migration**: Automatic migration between versions
5. **Performance Monitoring**: Settings impact on performance
6. **Accessibility**: Enhanced accessibility options
7. **Mobile Support**: Settings management for mobile interfaces

## Conclusion

The ICARUS CLI Settings and Personalization System provides a comprehensive, user-friendly configuration management solution that enhances the user experience through extensive customization options, robust backup/restore functionality, and seamless integration with the main application. The system is designed to be extensible and maintainable, supporting future enhancements and web migration requirements.
