"""Theme Configuration System

Defines the structure and configuration for ICARUS CLI themes.
"""

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Dict
from typing import List


class ThemeType(Enum):
    """Available theme types."""

    DARK = "dark"
    LIGHT = "light"
    HIGH_CONTRAST = "high_contrast"
    AEROSPACE = "aerospace"
    CLASSIC = "classic"


@dataclass
class ColorPalette:
    """Color palette for a theme."""

    # Primary colors
    primary: str
    primary_dark: str
    primary_light: str

    # Secondary colors
    secondary: str
    secondary_dark: str
    secondary_light: str

    # Background colors
    background: str
    background_dark: str
    background_light: str
    surface: str

    # Text colors
    text_primary: str
    text_secondary: str
    text_disabled: str
    text_inverse: str

    # Status colors
    success: str
    warning: str
    error: str
    info: str

    # Border and accent colors
    border: str
    border_focus: str
    accent: str
    highlight: str

    # Aerospace-specific colors
    aerospace_blue: str = "#1976d2"
    aerospace_cyan: str = "#4fc3f7"
    aerospace_orange: str = "#ff9800"
    aerospace_green: str = "#43a047"


@dataclass
class Typography:
    """Typography settings for a theme."""

    # Font families (terminal-compatible)
    primary_font: str = "monospace"
    secondary_font: str = "monospace"

    # Text styles
    title_style: str = "bold"
    subtitle_style: str = "italic"
    body_style: str = "none"
    caption_style: str = "dim"

    # Text sizes (relative to terminal)
    title_size: str = "large"
    subtitle_size: str = "medium"
    body_size: str = "medium"
    caption_size: str = "small"


@dataclass
class Spacing:
    """Spacing configuration for layouts."""

    # Base spacing units
    xs: int = 1
    sm: int = 2
    md: int = 3
    lg: int = 4
    xl: int = 6

    # Component-specific spacing
    widget_padding: int = 1
    container_margin: int = 1
    section_spacing: int = 2
    form_spacing: int = 1


@dataclass
class BorderStyles:
    """Border style configuration."""

    # Border types
    solid: str = "solid"
    dashed: str = "dashed"
    dotted: str = "dotted"
    double: str = "double"

    # Default border styles for components
    default_border: str = "solid"
    focus_border: str = "solid"
    error_border: str = "solid"

    # Border widths
    thin: str = "thin"
    medium: str = "medium"
    thick: str = "thick"


@dataclass
class AnimationConfig:
    """Animation and transition configuration."""

    # Transition durations (in CSS-compatible units)
    fast: str = "0.1s"
    medium: str = "0.2s"
    slow: str = "0.3s"

    # Easing functions
    ease_in: str = "ease-in"
    ease_out: str = "ease-out"
    ease_in_out: str = "ease-in-out"

    # Animation types
    fade: bool = True
    slide: bool = True
    scale: bool = False  # Limited in terminal

    # Screen transition settings
    screen_transition_duration: str = "0.2s"
    screen_transition_easing: str = "ease-in-out"


@dataclass
class ComponentStyles:
    """Styling configuration for specific components."""

    # Button styles
    button_padding: int = 1
    button_margin: int = 1
    button_border_radius: int = 0  # Not applicable in terminal

    # Input styles
    input_padding: int = 1
    input_margin: int = 1
    input_border_width: str = "thin"

    # Container styles
    container_padding: int = 1
    container_margin: int = 1

    # Table styles
    table_header_style: str = "bold"
    table_row_hover: bool = True
    table_border_style: str = "solid"

    # Progress bar styles
    progress_height: int = 1
    progress_border: bool = True

    # Modal styles
    modal_padding: int = 2
    modal_margin: int = 2
    modal_border: str = "solid"


@dataclass
class ThemeConfig:
    """Complete theme configuration."""

    name: str
    type: ThemeType
    description: str

    # Core theme components
    colors: ColorPalette
    typography: Typography = field(default_factory=Typography)
    spacing: Spacing = field(default_factory=Spacing)
    borders: BorderStyles = field(default_factory=BorderStyles)
    animations: AnimationConfig = field(default_factory=AnimationConfig)
    components: ComponentStyles = field(default_factory=ComponentStyles)

    # Theme metadata
    author: str = "ICARUS Team"
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)

    # Responsive breakpoints
    breakpoints: Dict[str, int] = field(
        default_factory=lambda: {
            "xs": 40,  # Very small terminals
            "sm": 60,  # Small terminals
            "md": 80,  # Medium terminals (default)
            "lg": 120,  # Large terminals
            "xl": 160,  # Extra large terminals
        },
    )

    # Custom CSS overrides
    custom_css: Dict[str, str] = field(default_factory=dict)

    def to_css(self) -> str:
        """Convert theme configuration to CSS."""
        css_rules = []

        # App-wide styles
        css_rules.append(f"""
App {{
    background: {self.colors.background};
    color: {self.colors.text_primary};
}}""")

        # Component styles
        css_rules.extend(self._generate_component_css())

        # Custom CSS overrides
        for selector, styles in self.custom_css.items():
            css_rules.append(f"{selector} {{ {styles} }}")

        return "\n\n".join(css_rules)

    def _generate_component_css(self) -> List[str]:
        """Generate CSS for standard components."""
        rules = []

        # Button styles
        rules.append(f"""
Button {{
    background: {self.colors.surface};
    color: {self.colors.text_primary};
    border: {self.borders.default_border} {self.colors.border};
    margin: {self.components.button_margin};
    padding: {self.components.button_padding};
}}

Button:focus {{
    border: {self.borders.focus_border} {self.colors.border_focus};
    background: {self.colors.primary};
    color: {self.colors.text_inverse};
}}

Button.-primary {{
    background: {self.colors.primary};
    color: {self.colors.text_inverse};
}}

Button.-success {{
    background: {self.colors.success};
    color: {self.colors.text_inverse};
}}

Button.-warning {{
    background: {self.colors.warning};
    color: {self.colors.text_primary};
}}

Button.-error {{
    background: {self.colors.error};
    color: {self.colors.text_inverse};
}}""")

        # Input styles
        rules.append(f"""
Input {{
    background: {self.colors.surface};
    color: {self.colors.text_primary};
    border: {self.borders.default_border} {self.colors.border};
    margin: {self.components.input_margin};
    padding: {self.components.input_padding};
}}

Input:focus {{
    border: {self.borders.focus_border} {self.colors.border_focus};
}}""")

        # Select styles
        rules.append(f"""
Select {{
    background: {self.colors.surface};
    color: {self.colors.text_primary};
    border: {self.borders.default_border} {self.colors.border};
}}

Select:focus {{
    border: {self.borders.focus_border} {self.colors.border_focus};
}}""")

        # Container styles
        rules.append(f"""
Container {{
    background: {self.colors.background};
    padding: {self.components.container_padding};
    margin: {self.components.container_margin};
}}""")

        # Label styles
        rules.append(f"""
Label {{
    color: {self.colors.text_primary};
}}

Label.-title {{
    color: {self.colors.primary};
    text-style: {self.typography.title_style};
}}

Label.-subtitle {{
    color: {self.colors.text_secondary};
    text-style: {self.typography.subtitle_style};
}}

Label.-caption {{
    color: {self.colors.text_disabled};
    text-style: {self.typography.caption_style};
}}""")

        # Status styles
        rules.append(f"""
.status-success {{
    color: {self.colors.success};
}}

.status-warning {{
    color: {self.colors.warning};
}}

.status-error {{
    color: {self.colors.error};
}}

.status-info {{
    color: {self.colors.info};
}}""")

        # DataTable styles
        rules.append(f"""
DataTable {{
    background: {self.colors.surface};
    color: {self.colors.text_primary};
    border: {self.borders.default_border} {self.colors.border};
}}

DataTable > .datatable--header {{
    background: {self.colors.primary};
    color: {self.colors.text_inverse};
    text-style: {self.components.table_header_style};
}}

DataTable > .datatable--row:hover {{
    background: {self.colors.highlight};
}}""")

        # Progress bar styles
        rules.append(f"""
ProgressBar {{
    height: {self.components.progress_height};
    border: {self.borders.default_border} {self.colors.border};
}}

ProgressBar > .bar--bar {{
    background: {self.colors.primary};
}}

ProgressBar > .bar--complete {{
    background: {self.colors.success};
}}""")

        return rules

    def get_responsive_css(self, terminal_width: int) -> str:
        """Get responsive CSS based on terminal width."""
        breakpoint = self._get_current_breakpoint(terminal_width)

        responsive_rules = []

        if breakpoint == "xs":
            # Very small terminals - minimal layout
            responsive_rules.append("""
.sidebar { width: 100%; height: 30%; }
.content { width: 100%; height: 70%; }
.hide-on-xs { display: none; }
""")
        elif breakpoint == "sm":
            # Small terminals - compact layout
            responsive_rules.append("""
.sidebar { width: 40%; }
.content { width: 60%; }
.hide-on-sm { display: none; }
""")
        elif breakpoint == "md":
            # Medium terminals - default layout
            responsive_rules.append("""
.sidebar { width: 30%; }
.content { width: 70%; }
""")
        elif breakpoint == "lg":
            # Large terminals - expanded layout
            responsive_rules.append("""
.sidebar { width: 25%; }
.content { width: 75%; }
.show-on-lg { display: block; }
""")
        else:  # xl
            # Extra large terminals - full layout
            responsive_rules.append("""
.sidebar { width: 20%; }
.content { width: 80%; }
.show-on-xl { display: block; }
""")

        return "\n".join(responsive_rules)

    def _get_current_breakpoint(self, width: int) -> str:
        """Determine current breakpoint based on terminal width."""
        if width < self.breakpoints["sm"]:
            return "xs"
        elif width < self.breakpoints["md"]:
            return "sm"
        elif width < self.breakpoints["lg"]:
            return "md"
        elif width < self.breakpoints["xl"]:
            return "lg"
        else:
            return "xl"
