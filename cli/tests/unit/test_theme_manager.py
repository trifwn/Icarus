import pytest


@pytest.mark.asyncio
async def test_theme_manager():
    try:
        from cli.tui.themes.theme_manager import ThemeManager

        theme_manager = ThemeManager()
        themes = theme_manager.get_available_themes()
        assert len(themes) > 0, "Should have available themes"
        original_theme = theme_manager.get_current_theme_id()
        for theme_id in themes[:2]:
            success = theme_manager.set_theme(theme_id)
            assert success, f"Should successfully switch to {theme_id}"
            current_id = theme_manager.get_current_theme_id()
            assert current_id == theme_id, f"Current theme should be {theme_id}"
        css = theme_manager.get_current_css()
        assert css and len(css) > 100, "Should generate valid CSS"
    except ImportError:
        pytest.skip("ThemeManager module not available")
