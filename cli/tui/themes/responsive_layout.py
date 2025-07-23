"""Responsive Layout Engine for ICARUS CLI

Provides responsive layout capabilities that adapt to different terminal sizes
and orientations, ensuring optimal user experience across various environments.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple


class LayoutMode(Enum):
    """Layout modes for different screen sizes."""

    MINIMAL = "minimal"  # Very small terminals (< 50 cols)
    COMPACT = "compact"  # Small terminals (50-79 cols)
    STANDARD = "standard"  # Standard terminals (80-119 cols)
    EXPANDED = "expanded"  # Large terminals (120-159 cols)
    WIDE = "wide"  # Very large terminals (160+ cols)


class Orientation(Enum):
    """Terminal orientation based on aspect ratio."""

    PORTRAIT = "portrait"  # Height > Width * 1.5
    SQUARE = "square"  # Height ≈ Width
    LANDSCAPE = "landscape"  # Width > Height * 1.5


@dataclass
class LayoutBreakpoints:
    """Defines breakpoints for responsive layout."""

    # Width breakpoints (columns)
    minimal: int = 40
    compact: int = 60
    standard: int = 80
    expanded: int = 120
    wide: int = 160

    # Height breakpoints (rows)
    short: int = 20
    medium: int = 30
    tall: int = 50

    def get_layout_mode(self, width: int) -> LayoutMode:
        """Determine layout mode based on terminal width."""
        if width < self.compact:
            return LayoutMode.MINIMAL
        elif width < self.standard:
            return LayoutMode.COMPACT
        elif width < self.expanded:
            return LayoutMode.STANDARD
        elif width < self.wide:
            return LayoutMode.EXPANDED
        else:
            return LayoutMode.WIDE

    def get_orientation(self, width: int, height: int) -> Orientation:
        """Determine orientation based on terminal dimensions."""
        aspect_ratio = width / height if height > 0 else 1.0

        if aspect_ratio < 0.67:  # Height > Width * 1.5
            return Orientation.PORTRAIT
        elif aspect_ratio > 1.5:  # Width > Height * 1.5
            return Orientation.LANDSCAPE
        else:
            return Orientation.SQUARE


@dataclass
class LayoutConfig:
    """Configuration for a specific layout."""

    # Container dimensions (as percentages or fixed values)
    sidebar_width: str = "30%"
    content_width: str = "70%"
    header_height: str = "3"
    footer_height: str = "1"

    # Visibility settings
    show_sidebar: bool = True
    show_header: bool = True
    show_footer: bool = True
    show_status_bar: bool = True

    # Layout behavior
    stack_vertically: bool = False
    collapse_sidebar: bool = False
    hide_secondary_panels: bool = False

    # Spacing and padding
    container_padding: int = 1
    widget_spacing: int = 1
    section_spacing: int = 2

    # Component priorities (for hiding in small layouts)
    component_priorities: Dict[str, int] = None

    def __post_init__(self):
        if self.component_priorities is None:
            self.component_priorities = {
                "main_content": 1,  # Always visible
                "navigation": 2,  # High priority
                "status_info": 3,  # Medium priority
                "secondary_panels": 4,  # Low priority
                "decorative": 5,  # Lowest priority
            }


class ResponsiveLayout:
    """Manages responsive layout for the ICARUS CLI."""

    def __init__(self, breakpoints: Optional[LayoutBreakpoints] = None):
        self.breakpoints = breakpoints or LayoutBreakpoints()
        self._current_width: int = 80
        self._current_height: int = 24
        self._current_mode: LayoutMode = LayoutMode.STANDARD
        self._current_orientation: Orientation = Orientation.LANDSCAPE
        self._layout_configs: Dict[LayoutMode, LayoutConfig] = {}
        self._resize_callbacks: List[Callable[[LayoutMode, Orientation], None]] = []

        self._setup_default_layouts()

    def _setup_default_layouts(self) -> None:
        """Setup default layout configurations for each mode."""

        # Minimal layout (< 50 cols)
        self._layout_configs[LayoutMode.MINIMAL] = LayoutConfig(
            sidebar_width="100%",
            content_width="100%",
            header_height="2",
            footer_height="1",
            show_sidebar=False,
            show_status_bar=False,
            stack_vertically=True,
            hide_secondary_panels=True,
            container_padding=0,
            widget_spacing=0,
            section_spacing=1,
        )

        # Compact layout (50-79 cols)
        self._layout_configs[LayoutMode.COMPACT] = LayoutConfig(
            sidebar_width="40%",
            content_width="60%",
            header_height="2",
            footer_height="1",
            show_sidebar=True,
            show_status_bar=True,
            collapse_sidebar=True,
            hide_secondary_panels=True,
            container_padding=1,
            widget_spacing=1,
            section_spacing=1,
        )

        # Standard layout (80-119 cols)
        self._layout_configs[LayoutMode.STANDARD] = LayoutConfig(
            sidebar_width="30%",
            content_width="70%",
            header_height="3",
            footer_height="1",
            show_sidebar=True,
            show_status_bar=True,
            collapse_sidebar=False,
            hide_secondary_panels=False,
            container_padding=1,
            widget_spacing=1,
            section_spacing=2,
        )

        # Expanded layout (120-159 cols)
        self._layout_configs[LayoutMode.EXPANDED] = LayoutConfig(
            sidebar_width="25%",
            content_width="75%",
            header_height="3",
            footer_height="1",
            show_sidebar=True,
            show_status_bar=True,
            collapse_sidebar=False,
            hide_secondary_panels=False,
            container_padding=1,
            widget_spacing=2,
            section_spacing=2,
        )

        # Wide layout (160+ cols)
        self._layout_configs[LayoutMode.WIDE] = LayoutConfig(
            sidebar_width="20%",
            content_width="60%",  # Leave space for additional panels
            header_height="3",
            footer_height="1",
            show_sidebar=True,
            show_status_bar=True,
            collapse_sidebar=False,
            hide_secondary_panels=False,
            container_padding=2,
            widget_spacing=2,
            section_spacing=3,
        )

    def update_dimensions(self, width: int, height: int) -> bool:
        """Update terminal dimensions and recalculate layout."""
        old_mode = self._current_mode
        old_orientation = self._current_orientation

        self._current_width = width
        self._current_height = height
        self._current_mode = self.breakpoints.get_layout_mode(width)
        self._current_orientation = self.breakpoints.get_orientation(width, height)

        # Check if layout changed
        layout_changed = (
            old_mode != self._current_mode
            or old_orientation != self._current_orientation
        )

        if layout_changed:
            self._notify_resize_callbacks()

        return layout_changed

    def get_current_layout(self) -> LayoutConfig:
        """Get the current layout configuration."""
        return self._layout_configs.get(
            self._current_mode,
            self._layout_configs[LayoutMode.STANDARD],
        )

    def get_layout_for_mode(self, mode: LayoutMode) -> LayoutConfig:
        """Get layout configuration for a specific mode."""
        return self._layout_configs.get(mode, self._layout_configs[LayoutMode.STANDARD])

    def set_layout_config(self, mode: LayoutMode, config: LayoutConfig) -> None:
        """Set custom layout configuration for a mode."""
        self._layout_configs[mode] = config

    def get_current_mode(self) -> LayoutMode:
        """Get the current layout mode."""
        return self._current_mode

    def get_current_orientation(self) -> Orientation:
        """Get the current orientation."""
        return self._current_orientation

    def get_dimensions(self) -> Tuple[int, int]:
        """Get current terminal dimensions."""
        return self._current_width, self._current_height

    def should_show_component(self, component_name: str) -> bool:
        """Check if a component should be visible in current layout."""
        layout = self.get_current_layout()

        # Check specific visibility rules
        if component_name == "sidebar" and not layout.show_sidebar:
            return False
        if component_name == "header" and not layout.show_header:
            return False
        if component_name == "footer" and not layout.show_footer:
            return False
        if component_name == "status_bar" and not layout.show_status_bar:
            return False

        # Check priority-based visibility
        priority = layout.component_priorities.get(component_name, 5)

        if self._current_mode == LayoutMode.MINIMAL:
            return priority <= 2  # Only highest priority components
        elif self._current_mode == LayoutMode.COMPACT:
            return priority <= 3  # High and medium priority
        else:
            return True  # All components visible in larger layouts

    def get_component_dimensions(self, component_name: str) -> Dict[str, str]:
        """Get dimensions for a specific component."""
        layout = self.get_current_layout()

        dimensions = {}

        if component_name == "sidebar":
            dimensions["width"] = layout.sidebar_width if layout.show_sidebar else "0"
            dimensions["height"] = "100%"
        elif component_name == "content":
            dimensions["width"] = layout.content_width
            dimensions["height"] = "100%"
        elif component_name == "header":
            dimensions["width"] = "100%"
            dimensions["height"] = layout.header_height
        elif component_name == "footer":
            dimensions["width"] = "100%"
            dimensions["height"] = layout.footer_height

        return dimensions

    def get_layout_css(self) -> str:
        """Generate CSS for the current layout."""
        layout = self.get_current_layout()
        css_rules = []

        # Container layout
        if layout.stack_vertically:
            css_rules.append("""
.main-container {
    layout: vertical;
}

.sidebar {
    width: 100%;
    height: 30%;
}

.content {
    width: 100%;
    height: 70%;
}""")
        else:
            css_rules.append(f"""
.main-container {{
    layout: horizontal;
}}

.sidebar {{
    width: {layout.sidebar_width};
    height: 100%;
    display: {'block' if layout.show_sidebar else 'none'};
}}

.content {{
    width: {layout.content_width};
    height: 100%;
}}""")

        # Header and footer
        css_rules.append(f"""
.header {{
    width: 100%;
    height: {layout.header_height};
    display: {'block' if layout.show_header else 'none'};
}}

.footer {{
    width: 100%;
    height: {layout.footer_height};
    display: {'block' if layout.show_footer else 'none'};
}}""")

        # Spacing and padding
        css_rules.append(f"""
.container {{
    padding: {layout.container_padding};
}}

.widget {{
    margin: {layout.widget_spacing};
}}

.section {{
    margin: {layout.section_spacing};
}}""")

        # Responsive visibility classes
        css_rules.append(f"""
.hide-on-minimal {{ display: {'none' if self._current_mode == LayoutMode.MINIMAL else 'block'}; }}
.hide-on-compact {{ display: {'none' if self._current_mode == LayoutMode.COMPACT else 'block'}; }}
.show-on-expanded {{ display: {'block' if self._current_mode in [LayoutMode.EXPANDED, LayoutMode.WIDE] else 'none'}; }}
.show-on-wide {{ display: {'block' if self._current_mode == LayoutMode.WIDE else 'none'}; }}""")

        return "\n".join(css_rules)

    def add_resize_callback(
        self,
        callback: Callable[[LayoutMode, Orientation], None],
    ) -> None:
        """Add a callback to be called when layout changes."""
        self._resize_callbacks.append(callback)

    def remove_resize_callback(
        self,
        callback: Callable[[LayoutMode, Orientation], None],
    ) -> None:
        """Remove a resize callback."""
        if callback in self._resize_callbacks:
            self._resize_callbacks.remove(callback)

    def _notify_resize_callbacks(self) -> None:
        """Notify all resize callbacks of layout change."""
        for callback in self._resize_callbacks:
            try:
                callback(self._current_mode, self._current_orientation)
            except Exception as e:
                # Log error but don't fail layout update
                print(f"Resize callback error: {e}")

    def get_optimal_widget_count(self, widget_type: str) -> int:
        """Get optimal number of widgets to display based on current layout."""
        layout = self.get_current_layout()

        # Define optimal counts for different widget types and layouts
        optimal_counts = {
            LayoutMode.MINIMAL: {
                "tabs": 2,
                "buttons": 3,
                "form_fields": 4,
                "list_items": 5,
            },
            LayoutMode.COMPACT: {
                "tabs": 4,
                "buttons": 5,
                "form_fields": 6,
                "list_items": 8,
            },
            LayoutMode.STANDARD: {
                "tabs": 6,
                "buttons": 8,
                "form_fields": 10,
                "list_items": 12,
            },
            LayoutMode.EXPANDED: {
                "tabs": 8,
                "buttons": 10,
                "form_fields": 15,
                "list_items": 20,
            },
            LayoutMode.WIDE: {
                "tabs": 10,
                "buttons": 12,
                "form_fields": 20,
                "list_items": 30,
            },
        }

        return optimal_counts.get(self._current_mode, {}).get(widget_type, 10)

    def get_layout_info(self) -> Dict[str, any]:
        """Get comprehensive information about current layout."""
        layout = self.get_current_layout()

        return {
            "mode": self._current_mode.value,
            "orientation": self._current_orientation.value,
            "dimensions": {
                "width": self._current_width,
                "height": self._current_height,
            },
            "layout_config": {
                "sidebar_width": layout.sidebar_width,
                "content_width": layout.content_width,
                "show_sidebar": layout.show_sidebar,
                "stack_vertically": layout.stack_vertically,
                "hide_secondary_panels": layout.hide_secondary_panels,
            },
            "breakpoints": {
                "minimal": self.breakpoints.minimal,
                "compact": self.breakpoints.compact,
                "standard": self.breakpoints.standard,
                "expanded": self.breakpoints.expanded,
                "wide": self.breakpoints.wide,
            },
        }
