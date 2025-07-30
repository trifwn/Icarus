"""Aerospace-Focused Theme Definitions

This module contains predefined themes specifically designed for aerospace
engineering applications, with color schemes inspired by aviation and space.
"""

from .theme_config import ColorPalette
from .theme_config import ThemeConfig
from .theme_config import ThemeType


class AerospaceThemes:
    """Collection of aerospace-focused themes."""

    @staticmethod
    def get_all_themes() -> dict[str, ThemeConfig]:
        """Get all available aerospace themes."""
        return {
            "aerospace_dark": AerospaceThemes.aerospace_dark(),
            "aerospace_light": AerospaceThemes.aerospace_light(),
            "aviation_blue": AerospaceThemes.aviation_blue(),
            "space_dark": AerospaceThemes.space_dark(),
            "cockpit_green": AerospaceThemes.cockpit_green(),
            "high_contrast": AerospaceThemes.high_contrast(),
            "classic_terminal": AerospaceThemes.classic_terminal(),
        }

    @staticmethod
    def aerospace_dark() -> ThemeConfig:
        """Dark theme with aerospace blue accents."""
        colors = ColorPalette(
            # Primary aerospace blue palette
            primary="#1976d2",
            primary_dark="#0d47a1",
            primary_light="#42a5f5",
            # Secondary cyan accents
            secondary="#4fc3f7",
            secondary_dark="#0288d1",
            secondary_light="#81d4fa",
            # Dark backgrounds
            background="#181c24",
            background_dark="#151922",
            background_light="#23293a",
            surface="#23293a",
            # Text colors
            text_primary="#ffffff",
            text_secondary="#b0bec5",
            text_disabled="#546e7a",
            text_inverse="#000000",
            # Status colors
            success="#43a047",
            warning="#fbc02d",
            error="#e53935",
            info="#4fc3f7",
            # Borders and accents
            border="#1976d2",
            border_focus="#ab47bc",
            accent="#ff9800",
            highlight="#263238",
            # Aerospace-specific
            aerospace_blue="#1976d2",
            aerospace_cyan="#4fc3f7",
            aerospace_orange="#ff9800",
            aerospace_green="#43a047",
        )

        return ThemeConfig(
            name="Aerospace Dark",
            type=ThemeType.AEROSPACE,
            description="Dark theme with aerospace blue accents, inspired by modern cockpit displays",
            colors=colors,
            tags=["dark", "aerospace", "professional", "cockpit"],
        )

    @staticmethod
    def aerospace_light() -> ThemeConfig:
        """Light theme with aerospace blue accents."""
        colors = ColorPalette(
            # Primary aerospace blue palette
            primary="#1976d2",
            primary_dark="#0d47a1",
            primary_light="#42a5f5",
            # Secondary cyan accents
            secondary="#0288d1",
            secondary_dark="#01579b",
            secondary_light="#4fc3f7",
            # Light backgrounds
            background="#f5f5f5",
            background_dark="#e0e0e0",
            background_light="#ffffff",
            surface="#ffffff",
            # Text colors
            text_primary="#212121",
            text_secondary="#757575",
            text_disabled="#bdbdbd",
            text_inverse="#ffffff",
            # Status colors
            success="#2e7d32",
            warning="#f57c00",
            error="#c62828",
            info="#1976d2",
            # Borders and accents
            border="#1976d2",
            border_focus="#ab47bc",
            accent="#ff9800",
            highlight="#e3f2fd",
            # Aerospace-specific
            aerospace_blue="#1976d2",
            aerospace_cyan="#0288d1",
            aerospace_orange="#f57c00",
            aerospace_green="#2e7d32",
        )

        return ThemeConfig(
            name="Aerospace Light",
            type=ThemeType.LIGHT,
            description="Light theme with aerospace blue accents, suitable for bright environments",
            colors=colors,
            tags=["light", "aerospace", "professional", "bright"],
        )

    @staticmethod
    def aviation_blue() -> ThemeConfig:
        """Aviation-inspired blue theme."""
        colors = ColorPalette(
            # Aviation blue palette
            primary="#0d47a1",
            primary_dark="#002171",
            primary_light="#5472d3",
            # Sky blue accents
            secondary="#03a9f4",
            secondary_dark="#0277bd",
            secondary_light="#4fc3f7",
            # Blue-tinted backgrounds
            background="#1a237e",
            background_dark="#000051",
            background_light="#3949ab",
            surface="#283593",
            # Text colors
            text_primary="#ffffff",
            text_secondary="#c5cae9",
            text_disabled="#7986cb",
            text_inverse="#000000",
            # Status colors
            success="#00c853",
            warning="#ffc107",
            error="#f44336",
            info="#03a9f4",
            # Borders and accents
            border="#3f51b5",
            border_focus="#ff4081",
            accent="#ffc107",
            highlight="#3949ab",
            # Aerospace-specific
            aerospace_blue="#0d47a1",
            aerospace_cyan="#03a9f4",
            aerospace_orange="#ffc107",
            aerospace_green="#00c853",
        )

        return ThemeConfig(
            name="Aviation Blue",
            type=ThemeType.AEROSPACE,
            description="Deep blue theme inspired by aviation and sky colors",
            colors=colors,
            tags=["dark", "blue", "aviation", "sky"],
        )

    @staticmethod
    def space_dark() -> ThemeConfig:
        """Space-inspired dark theme with cosmic colors."""
        colors = ColorPalette(
            # Space purple/blue palette
            primary="#673ab7",
            primary_dark="#4527a0",
            primary_light="#9575cd",
            # Cosmic cyan accents
            secondary="#00bcd4",
            secondary_dark="#00838f",
            secondary_light="#4dd0e1",
            # Deep space backgrounds
            background="#0a0a0a",
            background_dark="#000000",
            background_light="#1c1c1c",
            surface="#212121",
            # Text colors
            text_primary="#ffffff",
            text_secondary="#b0bec5",
            text_disabled="#616161",
            text_inverse="#000000",
            # Status colors
            success="#4caf50",
            warning="#ff9800",
            error="#f44336",
            info="#00bcd4",
            # Borders and accents
            border="#673ab7",
            border_focus="#e91e63",
            accent="#ff5722",
            highlight="#424242",
            # Aerospace-specific
            aerospace_blue="#3f51b5",
            aerospace_cyan="#00bcd4",
            aerospace_orange="#ff9800",
            aerospace_green="#4caf50",
        )

        return ThemeConfig(
            name="Space Dark",
            type=ThemeType.DARK,
            description="Deep space theme with cosmic purple and cyan accents",
            colors=colors,
            tags=["dark", "space", "cosmic", "purple"],
        )

    @staticmethod
    def cockpit_green() -> ThemeConfig:
        """Classic cockpit green theme inspired by military avionics."""
        colors = ColorPalette(
            # Military green palette
            primary="#2e7d32",
            primary_dark="#1b5e20",
            primary_light="#4caf50",
            # Amber accents (classic avionics)
            secondary="#ff8f00",
            secondary_dark="#e65100",
            secondary_light="#ffb74d",
            # Dark military backgrounds
            background="#1b2e1b",
            background_dark="#0d1f0d",
            background_light="#2e4e2e",
            surface="#2e4e2e",
            # Text colors
            text_primary="#c8e6c9",
            text_secondary="#a5d6a7",
            text_disabled="#66bb6a",
            text_inverse="#000000",
            # Status colors
            success="#4caf50",
            warning="#ff8f00",
            error="#d32f2f",
            info="#2e7d32",
            # Borders and accents
            border="#4caf50",
            border_focus="#ff8f00",
            accent="#ff5722",
            highlight="#388e3c",
            # Aerospace-specific
            aerospace_blue="#1976d2",
            aerospace_cyan="#00bcd4",
            aerospace_orange="#ff8f00",
            aerospace_green="#2e7d32",
        )

        return ThemeConfig(
            name="Cockpit Green",
            type=ThemeType.DARK,
            description="Military-inspired green theme reminiscent of classic avionics displays",
            colors=colors,
            tags=["dark", "military", "green", "classic", "avionics"],
        )

    @staticmethod
    def high_contrast() -> ThemeConfig:
        """High contrast theme for accessibility."""
        colors = ColorPalette(
            # High contrast palette
            primary="#ffffff",
            primary_dark="#e0e0e0",
            primary_light="#ffffff",
            # Yellow accents for visibility
            secondary="#ffff00",
            secondary_dark="#fbc02d",
            secondary_light="#ffff8d",
            # Pure black/white backgrounds
            background="#000000",
            background_dark="#000000",
            background_light="#1a1a1a",
            surface="#1a1a1a",
            # High contrast text
            text_primary="#ffffff",
            text_secondary="#e0e0e0",
            text_disabled="#808080",
            text_inverse="#000000",
            # High visibility status colors
            success="#00ff00",
            warning="#ffff00",
            error="#ff0000",
            info="#00ffff",
            # High contrast borders
            border="#ffffff",
            border_focus="#ffff00",
            accent="#ff00ff",
            highlight="#333333",
            # Aerospace-specific (high contrast)
            aerospace_blue="#0080ff",
            aerospace_cyan="#00ffff",
            aerospace_orange="#ff8000",
            aerospace_green="#00ff00",
        )

        return ThemeConfig(
            name="High Contrast",
            type=ThemeType.HIGH_CONTRAST,
            description="High contrast theme for improved accessibility and visibility",
            colors=colors,
            tags=["accessibility", "high-contrast", "visibility"],
        )

    @staticmethod
    def classic_terminal() -> ThemeConfig:
        """Classic terminal theme with retro green-on-black styling."""
        colors = ColorPalette(
            # Classic terminal green
            primary="#00ff00",
            primary_dark="#00cc00",
            primary_light="#66ff66",
            # Amber accents
            secondary="#ffaa00",
            secondary_dark="#cc8800",
            secondary_light="#ffcc66",
            # Classic black background
            background="#000000",
            background_dark="#000000",
            background_light="#0a0a0a",
            surface="#0a0a0a",
            # Classic terminal text
            text_primary="#00ff00",
            text_secondary="#00cc00",
            text_disabled="#006600",
            text_inverse="#000000",
            # Retro status colors
            success="#00ff00",
            warning="#ffaa00",
            error="#ff0000",
            info="#00ffff",
            # Simple borders
            border="#00ff00",
            border_focus="#ffaa00",
            accent="#ff00ff",
            highlight="#003300",
            # Aerospace-specific (retro style)
            aerospace_blue="#0080ff",
            aerospace_cyan="#00ffff",
            aerospace_orange="#ffaa00",
            aerospace_green="#00ff00",
        )

        return ThemeConfig(
            name="Classic Terminal",
            type=ThemeType.CLASSIC,
            description="Retro green-on-black terminal theme for nostalgic computing",
            colors=colors,
            tags=["retro", "classic", "terminal", "green"],
        )
