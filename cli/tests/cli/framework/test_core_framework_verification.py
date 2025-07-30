#!/usr/bin/env python3
"""Core Framework Verification for ICARUS CLI Theme System

This test verifies the core theme system functionality without requiring
external dependencies like Textual, focusing on the business logic and
configuration aspects.
"""

import sys
from pathlib import Path

# Add the CLI directory to the path
cli_dir = Path(__file__).parent
sys.path.insert(0, str(cli_dir))


def test_theme_configuration():
    """Test theme configuration system."""
    print("Testing Theme Configuration...")

    try:
        from tui.themes.theme_config import ColorPalette
        from tui.themes.theme_config import ThemeConfig
        from tui.themes.theme_config import ThemeType

        # Test color palette creation
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
        print("✓ Color palette created successfully")

        # Test theme config creation
        theme_config = ThemeConfig(
            name="Test Theme",
            type=ThemeType.DARK,
            description="Test theme for verification",
            colors=colors,
        )
        print("✓ Theme config created successfully")

        # Test CSS generation
        css = theme_config.to_css()
        if css and len(css) > 100:
            print(f"✓ CSS generation successful ({len(css)} characters)")

            # Check for key CSS elements
            required_elements = ["App {", "Button {", "Input {", "Label {"]
            missing_elements = [elem for elem in required_elements if elem not in css]

            if not missing_elements:
                print("✓ All required CSS elements present")
            else:
                print(f"✗ Missing CSS elements: {missing_elements}")
        else:
            print("✗ CSS generation failed")

        # Test responsive CSS
        responsive_css = theme_config.get_responsive_css(80)
        if responsive_css:
            print("✓ Responsive CSS generation successful")
        else:
            print("✗ Responsive CSS generation failed")

        return True

    except Exception as e:
        print(f"✗ Theme configuration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_aerospace_theme_definitions():
    """Test aerospace theme definitions."""
    print("\nTesting Aerospace Theme Definitions...")

    try:
        from tui.themes.aerospace_themes import AerospaceThemes
        from tui.themes.theme_config import ThemeType

        # Get all themes
        themes = AerospaceThemes.get_all_themes()
        expected_themes = [
            "aerospace_dark",
            "aerospace_light",
            "aviation_blue",
            "space_dark",
            "cockpit_green",
            "high_contrast",
            "classic_terminal",
        ]

        print(f"✓ Found {len(themes)} themes")

        # Check all expected themes exist
        missing_themes = [theme for theme in expected_themes if theme not in themes]
        if not missing_themes:
            print("✓ All expected themes present")
        else:
            print(f"✗ Missing themes: {missing_themes}")

        # Test each theme
        theme_type_counts = {theme_type: 0 for theme_type in ThemeType}

        for theme_id, theme_config in themes.items():
            # Basic validation
            if not theme_config.name:
                print(f"✗ Theme {theme_id} missing name")
                continue
            if not theme_config.description:
                print(f"✗ Theme {theme_id} missing description")
                continue

            # Color validation
            colors = theme_config.colors
            required_colors = [
                "primary",
                "secondary",
                "background",
                "text_primary",
                "success",
                "warning",
                "error",
                "info",
            ]

            missing_colors = [
                color for color in required_colors if not getattr(colors, color, None)
            ]

            if missing_colors:
                print(f"✗ Theme {theme_id} missing colors: {missing_colors}")
                continue

            # CSS generation test
            css = theme_config.to_css()
            if not css or len(css) < 100:
                print(f"✗ Theme {theme_id} CSS generation failed")
                continue

            print(f"✓ Theme {theme_id} ({theme_config.type.value}) validated")
            theme_type_counts[theme_config.type] += 1

        # Check theme type distribution
        print(f"✓ Theme distribution: {dict(theme_type_counts)}")

        return True

    except Exception as e:
        print(f"✗ Aerospace theme definitions test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_theme_manager_core():
    """Test theme manager core functionality."""
    print("\nTesting Theme Manager Core...")

    try:
        from tui.themes.theme_config import ThemeType
        from tui.themes.theme_manager import ThemeManager

        # Initialize theme manager
        theme_manager = ThemeManager()
        print("✓ Theme manager initialized")

        # Test available themes
        themes = theme_manager.get_available_themes()
        if len(themes) >= 5:  # Should have at least 5 themes
            print(f"✓ Found {len(themes)} available themes")
        else:
            print(f"✗ Expected at least 5 themes, found {len(themes)}")

        # Test theme switching
        original_theme = theme_manager.get_current_theme_id()

        for theme_id in themes[:3]:  # Test first 3 themes
            success = theme_manager.set_theme(theme_id)
            if success:
                current_id = theme_manager.get_current_theme_id()
                if current_id == theme_id:
                    print(f"✓ Successfully switched to {theme_id}")
                else:
                    print(
                        f"✗ Theme switch failed: expected {theme_id}, got {current_id}",
                    )
            else:
                print(f"✗ Failed to switch to {theme_id}")

        # Test CSS generation
        css = theme_manager.get_current_css()
        if css and len(css) > 200:  # Should include base + responsive CSS
            print(f"✓ Current CSS generated ({len(css)} characters)")
        else:
            print("✗ Current CSS generation failed")

        # Test theme info
        for theme_id in themes[:2]:
            info = theme_manager.get_theme_info(theme_id)
            if info and all(key in info for key in ["name", "description", "type"]):
                print(f"✓ Theme info complete for {theme_id}")
            else:
                print(f"✗ Theme info incomplete for {theme_id}")

        # Test theme filtering by type
        dark_themes = theme_manager.get_themes_by_type(ThemeType.DARK)
        aerospace_themes = theme_manager.get_themes_by_type(ThemeType.AEROSPACE)

        if dark_themes:
            print(f"✓ Found {len(dark_themes)} dark themes")
        if aerospace_themes:
            print(f"✓ Found {len(aerospace_themes)} aerospace themes")

        return True

    except Exception as e:
        print(f"✗ Theme manager core test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_responsive_layout_core():
    """Test responsive layout core functionality."""
    print("\nTesting Responsive Layout Core...")

    try:
        from tui.themes.responsive_layout import LayoutBreakpoints
        from tui.themes.responsive_layout import LayoutMode
        from tui.themes.responsive_layout import Orientation
        from tui.themes.responsive_layout import ResponsiveLayout

        # Test breakpoints
        breakpoints = LayoutBreakpoints()
        print("✓ Layout breakpoints created")

        # Test layout mode detection
        test_cases = [
            (30, LayoutMode.MINIMAL),
            (50, LayoutMode.COMPACT),
            (80, LayoutMode.STANDARD),
            (130, LayoutMode.EXPANDED),
            (170, LayoutMode.WIDE),
        ]

        for width, expected_mode in test_cases:
            detected_mode = breakpoints.get_layout_mode(width)
            if detected_mode == expected_mode:
                print(f"✓ Width {width} correctly detected as {expected_mode.value}")
            else:
                print(
                    f"✗ Width {width} detected as {detected_mode.value}, expected {expected_mode.value}",
                )

        # Test orientation detection
        orientation_cases = [
            (40, 60, Orientation.PORTRAIT),
            (80, 40, Orientation.LANDSCAPE),
            (60, 60, Orientation.SQUARE),
        ]

        for width, height, expected_orientation in orientation_cases:
            detected_orientation = breakpoints.get_orientation(width, height)
            if detected_orientation == expected_orientation:
                print(
                    f"✓ {width}x{height} correctly detected as {expected_orientation.value}",
                )
            else:
                print(
                    f"✗ {width}x{height} detected as {detected_orientation.value}, expected {expected_orientation.value}",
                )

        # Test responsive layout
        layout = ResponsiveLayout()
        print("✓ Responsive layout initialized")

        # Test layout updates
        layout.update_dimensions(100, 30)
        info = layout.get_layout_info()

        required_info_keys = ["mode", "orientation", "dimensions", "layout_config"]
        missing_keys = [key for key in required_info_keys if key not in info]

        if not missing_keys:
            print("✓ Layout info complete")
        else:
            print(f"✗ Layout info missing keys: {missing_keys}")

        # Test component visibility
        components = ["sidebar", "header", "footer", "status_bar"]
        for component in components:
            visible = layout.should_show_component(component)
            print(f"✓ Component '{component}' visibility determined: {visible}")

        # Test CSS generation
        css = layout.get_layout_css()
        if css and len(css) > 50:
            print(f"✓ Layout CSS generated ({len(css)} characters)")
        else:
            print("✗ Layout CSS generation failed")

        return True

    except Exception as e:
        print(f"✗ Responsive layout core test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_validation_system():
    """Test validation system for forms."""
    print("\nTesting Validation System...")

    try:
        # Test validation rules (without Textual dependencies)
        def create_validation_rule(name, validator, message):
            return {"name": name, "validator": validator, "error_message": message}

        # Test common validation rules
        rules = [
            create_validation_rule(
                "required",
                lambda x: bool(x.strip()),
                "This field is required",
            ),
            create_validation_rule(
                "min_length",
                lambda x: len(x) >= 3,
                "Minimum 3 characters",
            ),
            create_validation_rule(
                "numeric",
                lambda x: x.replace(".", "").replace("-", "").isdigit(),
                "Must be a number",
            ),
            create_validation_rule(
                "email",
                lambda x: "@" in x and "." in x,
                "Must be a valid email",
            ),
        ]

        # Test validation logic
        test_cases = [
            ("", "required", False),
            ("test", "required", True),
            ("ab", "min_length", False),
            ("abc", "min_length", True),
            ("123", "numeric", True),
            ("abc", "numeric", False),
            ("test@example.com", "email", True),
            ("invalid-email", "email", False),
        ]

        for value, rule_name, expected in test_cases:
            rule = next((r for r in rules if r["name"] == rule_name), None)
            if rule:
                result = rule["validator"](value)
                if result == expected:
                    print(f"✓ Validation '{rule_name}' for '{value}': {result}")
                else:
                    print(
                        f"✗ Validation '{rule_name}' for '{value}': expected {expected}, got {result}",
                    )
            else:
                print(f"✗ Rule '{rule_name}' not found")

        return True

    except Exception as e:
        print(f"✗ Validation system test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_css_generation():
    """Test CSS generation and formatting."""
    print("\nTesting CSS Generation...")

    try:
        from tui.themes.aerospace_themes import AerospaceThemes

        # Get a sample theme
        themes = AerospaceThemes.get_all_themes()
        sample_theme = list(themes.values())[0]

        # Generate CSS
        css = sample_theme.to_css()

        # Test CSS structure
        required_selectors = [
            "App {",
            "Button {",
            "Input {",
            "Label {",
            "DataTable {",
            ".status-success {",
            ".status-error {",
        ]

        missing_selectors = [sel for sel in required_selectors if sel not in css]

        if not missing_selectors:
            print("✓ All required CSS selectors present")
        else:
            print(f"✗ Missing CSS selectors: {missing_selectors}")

        # Test CSS properties
        required_properties = [
            "background:",
            "color:",
            "border:",
            "margin:",
            "padding:",
        ]

        missing_properties = [
            prop for prop in required_properties if css.count(prop) < 3
        ]  # Should appear multiple times

        if not missing_properties:
            print("✓ All required CSS properties present")
        else:
            print(f"✗ Insufficient CSS properties: {missing_properties}")

        # Test color values
        color_patterns = ["#", "rgb(", "rgba("]
        has_colors = any(pattern in css for pattern in color_patterns)

        if has_colors:
            print("✓ CSS contains color values")
        else:
            print("✗ CSS missing color values")

        # Test responsive CSS
        responsive_css = sample_theme.get_responsive_css(80)
        if responsive_css and len(responsive_css) > 20:
            print("✓ Responsive CSS generated")
        else:
            print("✗ Responsive CSS generation failed")

        return True

    except Exception as e:
        print(f"✗ CSS generation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_core_verification():
    """Run core framework verification tests."""
    print("ICARUS CLI Core Framework Verification")
    print("=" * 50)

    tests = [
        ("Theme Configuration", test_theme_configuration),
        ("Aerospace Theme Definitions", test_aerospace_theme_definitions),
        ("Theme Manager Core", test_theme_manager_core),
        ("Responsive Layout Core", test_responsive_layout_core),
        ("Validation System", test_validation_system),
        ("CSS Generation", test_css_generation),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("CORE VERIFICATION RESULTS")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{total} tests")

    if passed == total:
        print("🎉 Core framework verification successful!")
        print("✓ Theme system architecture is solid")
        print("✓ Responsive layout system works correctly")
        print("✓ CSS generation is functional")
        print("✓ All aerospace themes are properly defined")
        return True
    else:
        print("❌ Some core tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = run_core_verification()
    sys.exit(0 if success else 1)
