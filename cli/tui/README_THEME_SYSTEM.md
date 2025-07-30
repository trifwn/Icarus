# ICARUS CLI Theme System

A comprehensive theme system for the ICARUS CLI that provides aerospace-focused themes, responsive layouts, and advanced UI components for terminal interfaces.

## Overview

The theme system consists of several key components:

- **Theme Management**: Centralized theme switching and configuration
- **Aerospace Themes**: 7 professionally designed themes for aerospace applications
- **Responsive Layout**: Adaptive layouts for different terminal sizes
- **Base Widgets**: Enhanced UI components with aerospace-specific features
- **Screen Transitions**: Smooth transitions between different views
- **Animation System**: Rich animations for enhanced user experience

## Architecture

```
tui/
├── themes/
│   ├── __init__.py              # Theme system exports
│   ├── theme_config.py          # Theme configuration classes
│   ├── theme_manager.py         # Theme management logic
│   ├── aerospace_themes.py      # Predefined aerospace themes
│   ├── responsive_layout.py     # Responsive layout engine
│   └── aerospace_dark.css       # Sample CSS output
├── widgets/
│   ├── __init__.py              # Widget exports
│   ├── base_widgets.py          # Enhanced base widgets
│   └── [existing widgets...]    # Legacy widgets
├── utils/
│   ├── __init__.py              # Utility exports
│   ├── screen_transitions.py    # Screen transition system
│   ├── animations.py            # Animation framework
│   └── layout_helpers.py        # Layout utility functions
└── demo_theme_system.py         # Complete demo application
```

## Features

### 1. Aerospace-Focused Themes

Seven professionally designed themes optimized for aerospace engineering applications:

- **Aerospace Dark**: Modern dark theme with blue accents (default)
- **Aerospace Light**: Clean light theme for bright environments
- **Aviation Blue**: Deep blue theme inspired by aviation colors
- **Space Dark**: Cosmic theme with purple and cyan accents
- **Cockpit Green**: Military-inspired green theme
- **High Contrast**: Accessibility-focused high contrast theme
- **Classic Terminal**: Retro green-on-black terminal theme

### 2. Responsive Layout System

Automatically adapts to different terminal sizes:

- **Minimal** (< 50 cols): Single column, essential components only
- **Compact** (50-79 cols): Reduced sidebar, compact layout
- **Standard** (80-119 cols): Default layout with full features
- **Expanded** (120-159 cols): Wider layout with more space
- **Wide** (160+ cols): Full layout with additional panels

### 3. Enhanced Base Widgets

Aerospace-specific UI components:

- **AerospaceButton**: Enhanced buttons with variants and icons
- **ValidatedInput**: Input fields with real-time validation
- **AerospaceProgressBar**: Progress bars with status and ETA
- **StatusIndicator**: Color-coded status indicators
- **AerospaceDataTable**: Data tables with aerospace formatting
- **FormContainer**: Complete forms with validation
- **AerospaceTree**: Tree widgets with aerospace icons
- **NotificationPanel**: Notification system with categorization

### 4. Screen Transitions

Smooth transitions between screens:

- Fade in/out
- Slide transitions (left, right, up, down)
- Zoom effects
- Configurable duration and easing
- Callback support

### 5. Animation System

Rich animation capabilities:

- Fade animations
- Pulse effects
- Shake animations
- Typing effects
- Progress animations
- Custom easing functions

## Usage

### Basic Theme Management

```python
from tui.themes import ThemeManager

# Initialize theme manager
theme_manager = ThemeManager()

# Get available themes
themes = theme_manager.get_available_themes()
print(themes)  # ['aerospace_dark', 'aerospace_light', ...]

# Switch themes
theme_manager.set_theme('aviation_blue')

# Get current CSS
css = theme_manager.get_current_css()
```

### Responsive Layout

```python
from tui.themes.responsive_layout import ResponsiveLayout

# Initialize responsive layout
layout = ResponsiveLayout()

# Update dimensions (typically from terminal resize)
layout.update_dimensions(120, 30)

# Get current layout mode
mode = layout.get_current_mode()  # LayoutMode.EXPANDED

# Check component visibility
show_sidebar = layout.should_show_component('sidebar')

# Get layout CSS
css = layout.get_layout_css()
```

### Using Enhanced Widgets

```python
from tui.widgets.base_widgets import (
    AerospaceButton, ValidatedInput, ButtonVariant, ValidationRule
)

# Create aerospace button
button = AerospaceButton(
    "Analyze Aircraft",
    variant=ButtonVariant.PRIMARY,
    icon="✈"
)

# Create validated input
altitude_input = ValidatedInput(
    "Altitude (ft)",
    placeholder="35000",
    validation_rules=[
        ValidationRule(
            "range",
            lambda x: 1000 <= float(x) <= 60000 if x.isdigit() else False,
            "Altitude must be between 1,000 and 60,000 ft"
        )
    ]
)
```

### Screen Transitions

```python
from tui.utils.screen_transitions import (
    ScreenTransitionManager, TransitionConfig, TransitionType
)

# Initialize transition manager
transition_manager = ScreenTransitionManager(app)

# Create transition config
config = TransitionConfig(
    type=TransitionType.FADE,
    duration=0.3,
    easing="ease_in_out"
)

# Transition to new screen
await transition_manager.transition_to_screen(new_screen, config)
```

### Animations

```python
from tui.utils.animations import AnimationManager, AnimationType

# Initialize animation manager
animation_manager = AnimationManager()

# Animate widget
await animation_manager.fade_in(widget, duration=1.0)
await animation_manager.pulse(widget, repeat=3)
await animation_manager.shake(widget, amplitude=2.0)
```

## Theme Configuration

### Creating Custom Themes

```python
from tui.themes.theme_config import ThemeConfig, ColorPalette, ThemeType

# Define color palette
colors = ColorPalette(
    primary="#your_primary_color",
    secondary="#your_secondary_color",
    background="#your_background_color",
    # ... other colors
)

# Create theme configuration
custom_theme = ThemeConfig(
    name="My Custom Theme",
    type=ThemeType.DARK,
    description="A custom theme for my application",
    colors=colors
)

# Load into theme manager
theme_manager.load_custom_theme(custom_theme, "my_custom_theme")
```

### Color Palette Structure

Each theme includes a comprehensive color palette:

```python
@dataclass
class ColorPalette:
    # Primary colors
    primary: str              # Main brand color
    primary_dark: str         # Darker variant
    primary_light: str        # Lighter variant

    # Secondary colors
    secondary: str            # Secondary brand color
    secondary_dark: str       # Darker variant
    secondary_light: str      # Lighter variant

    # Background colors
    background: str           # Main background
    background_dark: str      # Darker background
    background_light: str     # Lighter background
    surface: str              # Surface color

    # Text colors
    text_primary: str         # Primary text
    text_secondary: str       # Secondary text
    text_disabled: str        # Disabled text
    text_inverse: str         # Inverse text

    # Status colors
    success: str              # Success state
    warning: str              # Warning state
    error: str                # Error state
    info: str                 # Information state

    # UI colors
    border: str               # Default border
    border_focus: str         # Focused border
    accent: str               # Accent color
    highlight: str            # Highlight color

    # Aerospace-specific
    aerospace_blue: str       # Aviation blue
    aerospace_cyan: str       # Sky cyan
    aerospace_orange: str     # Warning orange
    aerospace_green: str      # Success green
```

## Responsive Breakpoints

The system uses the following breakpoints:

| Breakpoint | Width Range | Layout Mode | Description |
|------------|-------------|-------------|-------------|
| xs         | < 50 cols   | Minimal     | Essential components only |
| sm         | 50-79 cols  | Compact     | Reduced sidebar, compact layout |
| md         | 80-119 cols | Standard    | Default full-featured layout |
| lg         | 120-159 cols| Expanded    | Wider layout with more space |
| xl         | 160+ cols   | Wide        | Full layout with additional panels |

## CSS Classes

The theme system generates comprehensive CSS with the following classes:

### Component Classes
- `.btn-primary`, `.btn-secondary`, `.btn-success`, etc.
- `.form-title`, `.section-title`, `.panel-title`
- `.input-label`, `.error-message`, `.progress-label`
- `.status-success`, `.status-warning`, `.status-error`, `.status-info`

### Layout Classes
- `.main-container`, `.sidebar`, `.content`
- `.form-fields`, `.form-actions`
- `.hide-on-minimal`, `.hide-on-compact`
- `.show-on-expanded`, `.show-on-wide`

### Animation Classes
- `.glow-low`, `.glow-medium`, `.glow-high`
- `.loading`, `.animated-element`

## Testing

The theme system includes comprehensive tests:

```bash
# Run core framework verification
python cli/test_core_framework_verification.py

# Run full test suite (requires Textual)
python cli/test_theme_system.py
```

### Test Coverage

- ✅ Theme configuration and creation
- ✅ CSS generation and validation
- ✅ Responsive layout calculations
- ✅ Theme switching and management
- ✅ Color palette validation
- ✅ Breakpoint detection
- ✅ Component visibility rules

## Demo Application

A complete demo application showcases all features:

```bash
# Run the demo (requires Textual)
python cli/tui/demo_theme_system.py
```

The demo includes:
- Live theme switching (keys 1-7)
- Responsive layout demonstration
- All widget types in action
- Animation examples
- Form validation
- Data table with aerospace formatting

## Integration with ICARUS CLI

The theme system integrates seamlessly with the existing ICARUS CLI:

1. **Backward Compatibility**: Existing widgets continue to work
2. **Progressive Enhancement**: New features can be adopted gradually
3. **Configuration**: Themes can be saved in user preferences
4. **Extensibility**: Custom themes can be loaded from files

## Performance Considerations

- **Lazy Loading**: Themes are loaded on demand
- **CSS Caching**: Generated CSS is cached until theme changes
- **Responsive Updates**: Layout recalculations only on size changes
- **Animation Optimization**: Animations use efficient frame rates

## Future Enhancements

Planned improvements include:

- **Theme Editor**: Visual theme customization interface
- **Plugin Themes**: Support for third-party theme packages
- **Advanced Animations**: More sophisticated animation effects
- **Accessibility**: Enhanced accessibility features
- **Mobile Support**: Responsive design for mobile terminals

## Requirements Met

This implementation fulfills all requirements from task 4:

✅ **Design aerospace-focused theme system with multiple color schemes**
- 7 professionally designed aerospace themes
- Comprehensive color palettes
- Aviation, space, and military-inspired designs

✅ **Implement responsive layout engine for different terminal sizes**
- 5 responsive breakpoints (xs, sm, md, lg, xl)
- Automatic layout adaptation
- Component visibility rules
- CSS generation for each breakpoint

✅ **Create base widget library for common UI components**
- 8+ enhanced base widgets
- Aerospace-specific features
- Validation system
- Status indicators and progress bars

✅ **Build screen transition system with smooth animations**
- Multiple transition types (fade, slide, zoom)
- Configurable duration and easing
- Animation framework with 10+ animation types
- Callback support for transition events

The theme system provides a solid foundation for the ICARUS CLI's modern, professional interface while maintaining the flexibility to adapt to different user needs and terminal environments.
