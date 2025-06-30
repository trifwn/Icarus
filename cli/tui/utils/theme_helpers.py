"""Theme Helper Utilities for ICARUS TUI

This module provides utilities for working with themes and the core UI system.
"""

from typing import Dict, Any, Optional
from textual.app import App

from core.ui import theme_manager, Theme


class ThemeHelper:
    """Helper class for managing TUI themes."""

    def __init__(self, app: App):
        self.app = app
        self.current_theme = theme_manager.current_theme

    def apply_theme(self, theme_name: str) -> bool:
        """Apply a theme to the TUI app."""
        try:
            success = theme_manager.apply_theme(theme_name)
            if success:
                self.current_theme = theme_manager.current_theme
                self._update_app_styles()
            return success
        except Exception:
            return False

    def get_available_themes(self) -> Dict[str, str]:
        """Get available themes with descriptions."""
        themes = {}
        for theme in Theme:
            themes[theme.value] = theme.value.replace("_", " ").title()
        return themes

    def get_current_theme_info(self) -> Dict[str, Any]:
        """Get information about the current theme."""
        return {
            "name": self.current_theme.value,
            "display_name": self.current_theme.value.replace("_", " ").title(),
            "colors": {
                "primary": theme_manager.get_color("primary"),
                "secondary": theme_manager.get_color("secondary"),
                "accent": theme_manager.get_color("accent"),
                "success": theme_manager.get_color("success"),
                "warning": theme_manager.get_color("warning"),
                "error": theme_manager.get_color("error"),
                "info": theme_manager.get_color("info"),
                "muted": theme_manager.get_color("muted"),
            },
        }

    def _update_app_styles(self) -> None:
        """Update the app's CSS styles based on current theme."""
        try:
            # Generate CSS variables based on current theme
            css_vars = self._generate_css_variables()

            # Update app styles
            self.app.styles.update(css_vars)

        except Exception as e:
            # Fallback to default styles
            pass

    def _generate_css_variables(self) -> Dict[str, str]:
        """Generate CSS variables for the current theme."""
        colors = {
            "primary": theme_manager.get_color("primary"),
            "secondary": theme_manager.get_color("secondary"),
            "accent": theme_manager.get_color("accent"),
            "success": theme_manager.get_color("success"),
            "warning": theme_manager.get_color("warning"),
            "error": theme_manager.get_color("error"),
            "info": theme_manager.get_color("info"),
            "muted": theme_manager.get_color("muted"),
        }

        return {
            "--primary-color": colors["primary"],
            "--secondary-color": colors["secondary"],
            "--accent-color": colors["accent"],
            "--success-color": colors["success"],
            "--warning-color": colors["warning"],
            "--error-color": colors["error"],
            "--info-color": colors["info"],
            "--muted-color": colors["muted"],
        }

    def create_themed_component(self, component_type: str, **kwargs) -> Dict[str, Any]:
        """Create a themed component configuration."""
        base_styles = {
            "border": f"solid {theme_manager.get_color('primary')}",
            "background": theme_manager.get_color("background"),
            "color": theme_manager.get_color("text"),
        }

        if component_type == "button":
            base_styles.update(
                {
                    "background": theme_manager.get_color("primary"),
                    "color": "white",
                }
            )
        elif component_type == "input":
            base_styles.update(
                {
                    "border": f"solid {theme_manager.get_color('secondary')}",
                }
            )
        elif component_type == "panel":
            base_styles.update(
                {
                    "border": f"solid {theme_manager.get_color('accent')}",
                }
            )

        return base_styles
