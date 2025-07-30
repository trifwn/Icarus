"""Streamlined UI Theme System

This module provides a simplified theme management system for the ICARUS CLI,
focusing on essential functionality and performance.
"""

import logging
from enum import Enum
from typing import Dict
from typing import Optional


class Theme(Enum):
    """Available themes."""

    DEFAULT = "default"
    AEROSPACE = "aerospace"
    SCIENTIFIC = "scientific"
    DARK = "dark"
    LIGHT = "light"


class ThemeManager:
    """Manages UI themes and appearance."""

    def __init__(self):
        self.current_theme = Theme.DEFAULT
        self.logger = logging.getLogger(__name__)

        # Theme CSS definitions
        self._theme_css: Dict[Theme, str] = {}
        self._initialize_themes()

    def _initialize_themes(self) -> None:
        """Initialize built-in themes."""
        # Default theme
        self._theme_css[Theme.DEFAULT] = """
        Screen {
            background: $background;
        }

        Header {
            background: $primary-background;
            color: $primary-color;
        }

        Footer {
            background: $primary-background;
            color: $primary-color;
        }
        """

        # Aerospace theme
        self._theme_css[Theme.AEROSPACE] = """
        Screen {
            background: $background;
        }

        Header {
            background: #000080;
            color: #ffffff;
        }

        Footer {
            background: #000080;
            color: #ffffff;
        }

        .title {
            color: #00bfff;
            text-style: bold;
        }

        .section-title {
            color: #00bfff;
            text-style: bold;
        }
        """

        # Scientific theme
        self._theme_css[Theme.SCIENTIFIC] = """
        Screen {
            background: #000000;
        }

        Header {
            background: #222222;
            color: #00ff00;
        }

        Footer {
            background: #222222;
            color: #00ff00;
        }

        .title {
            color: #00ff00;
            text-style: bold;
        }

        .section-title {
            color: #00ff00;
            text-style: bold;
        }

        Button {
            background: #222222;
            color: #ffffff;
        }

        Button:hover {
            background: #444444;
        }
        """

        # Dark theme
        self._theme_css[Theme.DARK] = """
        Screen {
            background: #121212;
        }

        Header {
            background: #1e1e1e;
            color: #ffffff;
        }

        Footer {
            background: #1e1e1e;
            color: #ffffff;
        }

        .title {
            color: #bb86fc;
            text-style: bold;
        }

        .section-title {
            color: #bb86fc;
            text-style: bold;
        }

        Button {
            background: #2d2d2d;
            color: #ffffff;
        }

        Button:hover {
            background: #3d3d3d;
        }
        """

        # Light theme
        self._theme_css[Theme.LIGHT] = """
        Screen {
            background: #f5f5f5;
            color: #121212;
        }

        Header {
            background: #e0e0e0;
            color: #121212;
        }

        Footer {
            background: #e0e0e0;
            color: #121212;
        }

        .title {
            color: #0066cc;
            text-style: bold;
        }

        .section-title {
            color: #0066cc;
            text-style: bold;
        }

        Button {
            background: #e0e0e0;
            color: #121212;
        }

        Button:hover {
            background: #d0d0d0;
        }
        """

    def apply_theme(self, app, theme_name: str) -> bool:
        """Apply a theme to the application."""
        try:
            # Convert string to Theme enum
            theme = None
            for t in Theme:
                if t.value == theme_name.lower():
                    theme = t
                    break

            if not theme:
                theme = Theme.DEFAULT

            self.current_theme = theme

            # Apply theme CSS to app
            if hasattr(app, "load_css"):
                app.load_css(self._theme_css[theme])
                self.logger.info(f"Applied theme: {theme.value}")
                return True
            else:
                self.logger.warning("App does not support CSS loading")
                return False

        except Exception as e:
            self.logger.error(f"Failed to apply theme: {e}")
            return False

    def get_current_theme(self) -> str:
        """Get the current theme name."""
        return self.current_theme.value

    def get_available_themes(self) -> Dict[str, str]:
        """Get available themes with descriptions."""
        return {
            Theme.DEFAULT.value: "Default theme with balanced colors",
            Theme.AEROSPACE.value: "Aerospace-focused theme with blue accents",
            Theme.SCIENTIFIC.value: "High-contrast theme for scientific applications",
            Theme.DARK.value: "Dark theme for low-light environments",
            Theme.LIGHT.value: "Light theme for high-light environments",
        }


# Global theme manager instance
theme_manager: Optional[ThemeManager] = None


def get_theme_manager() -> ThemeManager:
    """Get or create the global theme manager instance."""
    global theme_manager
    if theme_manager is None:
        theme_manager = ThemeManager()
    return theme_manager
