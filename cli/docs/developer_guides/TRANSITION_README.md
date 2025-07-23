# ICARUS CLI Transition Guide

This guide helps users transition from the previous ICARUS CLI implementation to the new streamlined implementation.

## Overview

The ICARUS CLI has been streamlined to improve performance, reduce complexity, and enhance maintainability. This guide explains the key changes and how to adapt your code and workflows to the new implementation.

## Key Changes

### Configuration Management

**Previous Implementation:**
- Separate `config.py` and `settings.py` modules
- Multiple configuration classes and APIs
- Complex settings hierarchy

**New Implementation:**
- Unified `unified_config.py` module
- Single configuration API
- Simplified settings structure

### Session Management

**Previous Implementation:**
- Complex state management in `state_manager.py`
- Multiple state classes and APIs

**New Implementation:**
- Streamlined `session_manager.py` module
- Simplified session API
- Focused on essential functionality

### Application Structure

**Previous Implementation:**
- Complex application structure with many dependencies
- Slow startup due to excessive imports
- Experimental features mixed with core functionality

**New Implementation:**
- Streamlined application structure
- Optimized imports for faster startup
- Focus on essential functionality

## Migration Guide

### For Users

1. **Command Line Usage:**
   - The command line interface remains largely the same
   - New options are available for workspace and theme selection

   ```bash
   # Previous usage
   python -m cli --config config.json

   # New usage (with additional options)
   python -m cli --config config.json --workspace my_workspace --theme aerospace
   ```

2. **Configuration Files:**
   - Existing configuration files will be automatically migrated
   - The new format is simpler and more consistent

3. **Workspaces:**
   - Workspaces are now managed through the unified configuration system
   - Use the `--workspace` option to select a workspace

### For Developers

1. **Import Changes:**

   ```python
   # Previous imports
   from cli.core.config import ConfigManager
   from cli.core.settings import SettingsManager
   from cli.app.state_manager import StateManager

   # New imports
   from cli.core.unified_config import get_config_manager
   from cli.core.session_manager import get_session_manager
   ```

2. **Configuration Access:**

   ```python
   # Previous code
   config_manager = ConfigManager()
   settings_manager = SettingsManager()

   theme = config_manager.get("theme")
   workspace = settings_manager.get_workspace_settings()

   # New code
   config_manager = get_config_manager()

   theme = config_manager.get("theme")
   workspace = config_manager.get_workspace_config()
   ```

3. **Session Management:**

   ```python
   # Previous code
   state_manager = StateManager()
   await state_manager.initialize_session()
   session_info = state_manager.get_session_info()

   # New code
   session_manager = get_session_manager()
   await session_manager.initialize()
   session_info = session_manager.get_session_info()
   ```

4. **Event System:**

   ```python
   # Previous code
   from cli.app.event_system import EventSystem
   event_system = EventSystem()
   event_system.subscribe("screen_change", on_screen_change)

   # New code
   from cli.app.streamlined_event import get_event_system
   event_system = get_event_system()
   event_system.subscribe("screen_change", on_screen_change)
   ```

5. **Application Integration:**

   ```python
   # Previous code
   from cli.app.main_app import IcarusApp
   app = IcarusApp()

   # New code
   from cli.app.streamlined_app import IcarusApp
   app = IcarusApp()
   ```

## Feature Comparison

| Feature | Previous Implementation | New Implementation |
|---------|------------------------|-------------------|
| Configuration | Multiple systems | Unified system |
| Session Management | Complex state manager | Streamlined session manager |
| Startup Time | Slower | Faster |
| Memory Usage | Higher | Lower |
| API Complexity | Complex | Simplified |
| Error Handling | Inconsistent | Improved |
| Extensibility | Complex plugin system | Focused core with extension points |

## Removed Features

The following experimental features have been removed from the core implementation:

1. **Complex Plugin System** - Replaced with a simpler extension mechanism
2. **Advanced Workflow Builder** - Will be reimplemented in a future release
3. **Experimental Collaboration Features** - Will be reimplemented with a more stable design
4. **Complex Theme System** - Replaced with a streamlined theme manager

## Getting Help

If you encounter issues during the transition, please:

1. Check the documentation in `cli/STREAMLINED_README.md`
2. Look for examples in the `cli/examples` directory
3. Report issues through the standard channels
