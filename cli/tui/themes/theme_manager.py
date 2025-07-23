"""Theme Manager for ICARUS CLI

Manages theme loading, switching, and application across the TUI interface.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from .aerospace_themes import AerospaceThemes
from .theme_config import ThemeConfig
from .theme_config import ThemeType


@dataclass
class Theme:
    """Theme wrapper with metadata and CSS content."""

    config: ThemeConfig
    css_content: str
    responsive_css: Dict[str, str]  # Breakpoint -> CSS mapping

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def type(self) -> ThemeType:
        return self.config.type

    @property
    def description(self) -> str:
        return self.config.description


class ThemeManager:
    """Manages themes for the ICARUS CLI application."""

    def __init__(self):
        self._themes: Dict[str, Theme] = {}
        self._current_theme: Optional[Theme] = None
        self._theme_change_callbacks: List[Callable[[Theme], None]] = []
        self._terminal_width: int = 80  # Default terminal width
        self._load_default_themes()

    def _load_default_themes(self) -> None:
        """Load all default aerospace themes."""
        aerospace_themes = AerospaceThemes.get_all_themes()

        for theme_id, theme_config in aerospace_themes.items():
            theme = Theme(
                config=theme_config,
                css_content=theme_config.to_css(),
                responsive_css=self._generate_responsive_css(theme_config),
            )
            self._themes[theme_id] = theme

        # Set default theme
        if "aerospace_dark" in self._themes:
            self._current_theme = self._themes["aerospace_dark"]

    def _generate_responsive_css(self, theme_config: ThemeConfig) -> Dict[str, str]:
        """Generate responsive CSS for different breakpoints."""
        responsive_css = {}

        for breakpoint in ["xs", "sm", "md", "lg", "xl"]:
            # Get the minimum width for this breakpoint
            min_width = theme_config.breakpoints.get(breakpoint, 80)
            responsive_css[breakpoint] = theme_config.get_responsive_css(min_width)

        return responsive_css

    def get_available_themes(self) -> List[str]:
        """Get list of available theme IDs."""
        return list(self._themes.keys())

    def get_theme_info(self, theme_id: str) -> Optional[Dict[str, str]]:
        """Get information about a specific theme."""
        if theme_id not in self._themes:
            return None

        theme = self._themes[theme_id]
        return {
            "id": theme_id,
            "name": theme.name,
            "type": theme.type.value,
            "description": theme.description,
            "author": theme.config.author,
            "version": theme.config.version,
            "tags": ", ".join(theme.config.tags),
        }

    def get_themes_by_type(self, theme_type: ThemeType) -> List[str]:
        """Get themes filtered by type."""
        return [
            theme_id
            for theme_id, theme in self._themes.items()
            if theme.type == theme_type
        ]

    def get_current_theme(self) -> Optional[Theme]:
        """Get the currently active theme."""
        return self._current_theme

    def get_current_theme_id(self) -> Optional[str]:
        """Get the ID of the currently active theme."""
        if not self._current_theme:
            return None

        for theme_id, theme in self._themes.items():
            if theme == self._current_theme:
                return theme_id

        return None

    def set_theme(self, theme_id: str) -> bool:
        """Set the active theme by ID."""
        if theme_id not in self._themes:
            return False

        old_theme = self._current_theme
        self._current_theme = self._themes[theme_id]

        # Notify callbacks of theme change
        for callback in self._theme_change_callbacks:
            try:
                callback(self._current_theme)
            except Exception as e:
                # Log error but don't fail theme change
                print(f"Theme change callback error: {e}")

        return True

    def get_current_css(self) -> str:
        """Get CSS for the current theme."""
        if not self._current_theme:
            return ""

        base_css = self._current_theme.css_content
        responsive_css = self.get_responsive_css()

        return f"{base_css}\n\n/* Responsive Styles */\n{responsive_css}"

    def get_responsive_css(self) -> str:
        """Get responsive CSS for current terminal width."""
        if not self._current_theme:
            return ""

        breakpoint = self._get_current_breakpoint()
        return self._current_theme.responsive_css.get(breakpoint, "")

    def set_terminal_width(self, width: int) -> None:
        """Update terminal width for responsive calculations."""
        old_breakpoint = self._get_current_breakpoint()
        self._terminal_width = width
        new_breakpoint = self._get_current_breakpoint()

        # If breakpoint changed, notify callbacks
        if old_breakpoint != new_breakpoint and self._current_theme:
            for callback in self._theme_change_callbacks:
                try:
                    callback(self._current_theme)
                except Exception:
                    pass

    def _get_current_breakpoint(self) -> str:
        """Get current responsive breakpoint."""
        if not self._current_theme:
            return "md"

        breakpoints = self._current_theme.config.breakpoints

        if self._terminal_width < breakpoints["sm"]:
            return "xs"
        elif self._terminal_width < breakpoints["md"]:
            return "sm"
        elif self._terminal_width < breakpoints["lg"]:
            return "md"
        elif self._terminal_width < breakpoints["xl"]:
            return "lg"
        else:
            return "xl"

    def add_theme_change_callback(self, callback: Callable[[Theme], None]) -> None:
        """Add a callback to be called when theme changes."""
        self._theme_change_callbacks.append(callback)

    def remove_theme_change_callback(self, callback: Callable[[Theme], None]) -> None:
        """Remove a theme change callback."""
        if callback in self._theme_change_callbacks:
            self._theme_change_callbacks.remove(callback)

    def load_custom_theme(self, theme_config: ThemeConfig, theme_id: str) -> bool:
        """Load a custom theme configuration."""
        try:
            theme = Theme(
                config=theme_config,
                css_content=theme_config.to_css(),
                responsive_css=self._generate_responsive_css(theme_config),
            )
            self._themes[theme_id] = theme
            return True
        except Exception as e:
            print(f"Failed to load custom theme: {e}")
            return False

    def save_theme_to_file(self, theme_id: str, file_path: Path) -> bool:
        """Save a theme configuration to a file."""
        if theme_id not in self._themes:
            return False

        try:
            theme = self._themes[theme_id]
            css_content = theme.css_content

            with open(file_path, "w") as f:
                f.write(css_content)

            return True
        except Exception as e:
            print(f"Failed to save theme to file: {e}")
            return False

    def load_theme_from_file(self, file_path: Path, theme_id: str) -> bool:
        """Load a theme from a CSS file."""
        try:
            with open(file_path) as f:
                css_content = f.read()

            # Create a basic theme config for file-based themes
            from .theme_config import ColorPalette

            # Parse basic colors from CSS (simplified)
            colors = ColorPalette(
                primary="#1976d2",
                primary_dark="#0d47a1",
                primary_light="#42a5f5",
                secondary="#4fc3f7",
                secondary_dark="#0288d1",
                secondary_light="#81d4fa",
                background="#181c24",
                background_dark="#151922",
                background_light="#23293a",
                surface="#23293a",
                text_primary="#ffffff",
                text_secondary="#b0bec5",
                text_disabled="#546e7a",
                text_inverse="#000000",
                success="#43a047",
                warning="#fbc02d",
                error="#e53935",
                info="#4fc3f7",
                border="#1976d2",
                border_focus="#ab47bc",
                accent="#ff9800",
                highlight="#263238",
            )

            theme_config = ThemeConfig(
                name=f"Custom Theme ({theme_id})",
                type=ThemeType.DARK,
                description=f"Custom theme loaded from {file_path.name}",
                colors=colors,
                tags=["custom", "file-based"],
            )

            # Override CSS with file content
            theme_config.custom_css = {"*": css_content}

            return self.load_custom_theme(theme_config, theme_id)

        except Exception as e:
            print(f"Failed to load theme from file: {e}")
            return False

    def get_theme_preview(self, theme_id: str) -> Dict[str, str]:
        """Get a preview of theme colors for UI display."""
        if theme_id not in self._themes:
            return {}

        theme = self._themes[theme_id]
        colors = theme.config.colors

        return {
            "primary": colors.primary,
            "secondary": colors.secondary,
            "background": colors.background,
            "surface": colors.surface,
            "text_primary": colors.text_primary,
            "success": colors.success,
            "warning": colors.warning,
            "error": colors.error,
            "info": colors.info,
        }

    def create_theme_variant(
        self,
        base_theme_id: str,
        modifications: Dict[str, str],
        new_theme_id: str,
    ) -> bool:
        """Create a variant of an existing theme with color modifications."""
        if base_theme_id not in self._themes:
            return False

        try:
            base_theme = self._themes[base_theme_id]
            base_config = base_theme.config

            # Create a copy of the base theme config
            import copy

            new_config = copy.deepcopy(base_config)

            # Apply modifications to colors
            for color_name, color_value in modifications.items():
                if hasattr(new_config.colors, color_name):
                    setattr(new_config.colors, color_name, color_value)

            # Update metadata
            new_config.name = f"{base_config.name} (Variant)"
            new_config.description = f"Variant of {base_config.name}"
            new_config.tags.append("variant")

            return self.load_custom_theme(new_config, new_theme_id)

        except Exception as e:
            print(f"Failed to create theme variant: {e}")
            return False
