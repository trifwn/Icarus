import pytest

def test_theme_css_generation():
    try:
        from cli.tui.themes.theme_manager import ThemeManager
        theme_manager = ThemeManager()
        css = theme_manager.get_current_css()
        assert css and len(css) > 100, "Should generate valid CSS"
    except ImportError:
        pytest.skip("ThemeManager module not available")
