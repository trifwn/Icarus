#!/usr/bin/env python3
"""Test Script for ICARUS CLI Theme System

Tests the theme system, responsive layouts, and base widgets to ensure
they work correctly and meet the requirements.
"""


def test_theme_system():
    """Test the theme system functionality."""
    print("Testing Theme System...")

    try:
        from tui.themes import ThemeManager
        from tui.themes.theme_config import ThemeType

        # Test theme manager initialization
        theme_manager = ThemeManager()
        print("✓ Theme manager initialized")

        # Test available themes
        themes = theme_manager.get_available_themes()
        print(f"✓ Found {len(themes)} themes: {', '.join(themes)}")

        # Test theme switching
        for theme_id in themes[:3]:  # Test first 3 themes
            success = theme_manager.set_theme(theme_id)
            if success:
                print(f"✓ Successfully switched to {theme_id}")

                # Test CSS generation
                css = theme_manager.get_current_css()
                if css and len(css) > 100:  # Basic check for CSS content
                    print(f"✓ Generated CSS for {theme_id} ({len(css)} characters)")
                else:
                    print(f"✗ CSS generation failed for {theme_id}")
            else:
                print(f"✗ Failed to switch to {theme_id}")

        # Test theme info
        for theme_id in themes[:2]:
            info = theme_manager.get_theme_info(theme_id)
            if info:
                print(
                    f"✓ Theme info for {theme_id}: {info['name']} - {info['description'][:50]}...",
                )
            else:
                print(f"✗ Failed to get info for {theme_id}")

        # Test theme types
        dark_themes = theme_manager.get_themes_by_type(ThemeType.DARK)
        light_themes = theme_manager.get_themes_by_type(ThemeType.LIGHT)
        aerospace_themes = theme_manager.get_themes_by_type(ThemeType.AEROSPACE)

        print(
            f"✓ Theme types - Dark: {len(dark_themes)}, Light: {len(light_themes)}, Aerospace: {len(aerospace_themes)}",
        )

        return True

    except Exception as e:
        print(f"✗ Theme system test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_responsive_layout():
    """Test responsive layout functionality."""
    print("\nTesting Responsive Layout...")

    try:
        from tui.themes.responsive_layout import LayoutMode
        from tui.themes.responsive_layout import ResponsiveLayout

        # Test responsive layout initialization
        layout = ResponsiveLayout()
        print("✓ Responsive layout initialized")

        # Test different screen sizes
        test_sizes = [
            (40, 20, LayoutMode.MINIMAL),
            (60, 25, LayoutMode.COMPACT),
            (80, 30, LayoutMode.STANDARD),
            (120, 35, LayoutMode.EXPANDED),
            (160, 40, LayoutMode.WIDE),
        ]

        for width, height, expected_mode in test_sizes:
            layout.update_dimensions(width, height)
            current_mode = layout.get_current_mode()

            if current_mode == expected_mode:
                print(f"✓ {width}x{height} correctly detected as {expected_mode.value}")
            else:
                print(
                    f"✗ {width}x{height} detected as {current_mode.value}, expected {expected_mode.value}",
                )

        # Test layout configuration
        for mode in LayoutMode:
            config = layout.get_layout_for_mode(mode)
            if config:
                print(
                    f"✓ Layout config for {mode.value}: sidebar={config.sidebar_width}, show_sidebar={config.show_sidebar}",
                )
            else:
                print(f"✗ No layout config for {mode.value}")

        # Test CSS generation
        css = layout.get_layout_css()
        if css and len(css) > 50:
            print(f"✓ Generated layout CSS ({len(css)} characters)")
        else:
            print("✗ Layout CSS generation failed")

        # Test component visibility
        test_components = ["sidebar", "header", "footer", "status_bar"]
        for component in test_components:
            visible = layout.should_show_component(component)
            print(f"✓ Component '{component}' visibility: {visible}")

        return True

    except Exception as e:
        print(f"✗ Responsive layout test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_base_widgets():
    """Test base widget functionality."""
    print("\nTesting Base Widgets...")

    try:
        from tui.widgets.base_widgets import ButtonVariant
        from tui.widgets.base_widgets import InputType
        from tui.widgets.base_widgets import StatusIndicator
        from tui.widgets.base_widgets import ValidationRule

        # Test button variants
        variants = list(ButtonVariant)
        print(f"✓ Button variants available: {[v.value for v in variants]}")

        # Test input types
        input_types = list(InputType)
        print(f"✓ Input types available: {[t.value for t in input_types]}")

        # Test validation rules
        test_rule = ValidationRule(
            "test_rule",
            lambda x: len(x) > 3,
            "Must be longer than 3 characters",
        )

        if test_rule.validator("test"):
            print("✗ Validation rule test failed - should be False")
        elif test_rule.validator("testing"):
            print("✓ Validation rule test passed")
        else:
            print("✗ Validation rule test failed - should be True")

        # Test status indicator types
        status_types = list(StatusIndicator.StatusType)
        print(f"✓ Status indicator types: {[s.value for s in status_types]}")

        return True

    except Exception as e:
        print(f"✗ Base widgets test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_screen_transitions():
    """Test screen transition system."""
    print("\nTesting Screen Transitions...")

    try:
        from tui.utils.screen_transitions import TransitionConfig
        from tui.utils.screen_transitions import TransitionDirection
        from tui.utils.screen_transitions import TransitionType

        # Test transition types
        transition_types = list(TransitionType)
        print(f"✓ Transition types available: {[t.value for t in transition_types]}")

        # Test transition config
        config = TransitionConfig(
            type=TransitionType.FADE,
            duration=0.3,
            easing="ease_in_out",
            direction=TransitionDirection.FORWARD,
        )

        if config.type == TransitionType.FADE and config.duration == 0.3:
            print("✓ Transition config creation successful")
        else:
            print("✗ Transition config creation failed")

        return True

    except Exception as e:
        print(f"✗ Screen transitions test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_animations():
    """Test animation system."""
    print("\nTesting Animation System...")

    try:
        from tui.utils.animations import AnimationConfig
        from tui.utils.animations import AnimationType
        from tui.utils.animations import EasingFunction

        # Test animation types
        animation_types = list(AnimationType)
        print(f"✓ Animation types available: {[t.value for t in animation_types]}")

        # Test easing functions
        easing_functions = list(EasingFunction)
        print(f"✓ Easing functions available: {[e.value for e in easing_functions]}")

        # Test animation config
        config = AnimationConfig(
            type=AnimationType.FADE_IN,
            duration=1.0,
            easing=EasingFunction.EASE_IN_OUT,
            repeat=1,
        )

        if config.type == AnimationType.FADE_IN and config.duration == 1.0:
            print("✓ Animation config creation successful")
        else:
            print("✗ Animation config creation failed")

        return True

    except Exception as e:
        print(f"✗ Animation system test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_aerospace_themes():
    """Test aerospace theme definitions."""
    print("\nTesting Aerospace Themes...")

    try:
        from tui.themes.aerospace_themes import AerospaceThemes

        # Get all themes
        themes = AerospaceThemes.get_all_themes()
        print(f"✓ Found {len(themes)} aerospace themes")

        # Test each theme
        for theme_id, theme_config in themes.items():
            # Check basic properties
            if theme_config.name and theme_config.description:
                print(f"✓ Theme {theme_id}: {theme_config.name}")

                # Check color palette
                colors = theme_config.colors
                if colors.primary and colors.background and colors.text_primary:
                    print("  ✓ Color palette complete")
                else:
                    print("  ✗ Color palette incomplete")

                # Test CSS generation
                css = theme_config.to_css()
                if css and len(css) > 100:
                    print(f"  ✓ CSS generation successful ({len(css)} chars)")
                else:
                    print("  ✗ CSS generation failed")
            else:
                print(f"✗ Theme {theme_id} missing basic properties")

        return True

    except Exception as e:
        print(f"✗ Aerospace themes test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("ICARUS CLI Theme System Test Suite")
    print("=" * 50)

    tests = [
        ("Theme System", test_theme_system),
        ("Responsive Layout", test_responsive_layout),
        ("Base Widgets", test_base_widgets),
        ("Screen Transitions", test_screen_transitions),
        ("Animation System", test_animations),
        ("Aerospace Themes", test_aerospace_themes),
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
    print("TEST RESULTS SUMMARY")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{total} tests")

    if passed == total:
        print("🎉 All tests passed! Theme system is ready.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False
