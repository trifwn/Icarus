import pytest

def test_theme_types():
    try:
        from cli.tui.themes import ThemeManager
        from cli.tui.themes.theme_config import ThemeType
        theme_manager = ThemeManager()
        dark_themes = theme_manager.get_themes_by_type(ThemeType.DARK)
        light_themes = theme_manager.get_themes_by_type(ThemeType.LIGHT)
        aerospace_themes = theme_manager.get_themes_by_type(ThemeType.AEROSPACE)
        assert isinstance(dark_themes, list)
        assert isinstance(light_themes, list)
        assert isinstance(aerospace_themes, list)
    except ImportError:
        pytest.skip("ThemeManager or ThemeType not available")
