import pytest


@pytest.mark.asyncio
async def test_responsive_layout_calculations():
    try:
        from cli.tui.themes.responsive_layout import LayoutBreakpoints, ResponsiveLayout

        layout = ResponsiveLayout()
        breakpoints = LayoutBreakpoints()
        test_cases = [(30, "MINIMAL"), (80, "STANDARD"), (150, "WIDE")]
        for width, expected_mode in test_cases:
            mode = breakpoints.get_layout_mode(width)
            assert mode.value == expected_mode, (
                f"Width {width} should be {expected_mode}"
            )
        layout.update_dimensions(100, 30)
        info = layout.get_layout_info()
        required_keys = ["mode", "orientation", "dimensions"]
        for key in required_keys:
            assert key in info, f"Layout info should contain {key}"
    except ImportError:
        pytest.skip("ResponsiveLayout module not available")
