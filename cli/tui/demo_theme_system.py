#!/usr/bin/env python3
"""Demo Application for ICARUS CLI Theme System

This demo showcases the aerospace-focused theme system, responsive layouts,
base widgets, and screen transitions working together.
"""

import asyncio

from textual.app import App
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.containers import Horizontal
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Label

# Import our theme system
from themes import ThemeManager
from themes.responsive_layout import ResponsiveLayout
from utils.animations import AnimationManager
from utils.screen_transitions import ScreenTransitionManager

# Import base widgets
from widgets.base_widgets import AerospaceButton
from widgets.base_widgets import AerospaceDataTable
from widgets.base_widgets import AerospaceProgressBar
from widgets.base_widgets import ButtonVariant
from widgets.base_widgets import FormContainer
from widgets.base_widgets import InputType
from widgets.base_widgets import NotificationPanel
from widgets.base_widgets import StatusIndicator
from widgets.base_widgets import ValidatedInput
from widgets.base_widgets import ValidationRule


class ThemePreviewWidget(Container):
    """Widget to preview theme colors and components."""

    def __init__(self, theme_name: str, **kwargs):
        super().__init__(**kwargs)
        self.theme_name = theme_name

    def compose(self) -> ComposeResult:
        yield Label(f"Theme: {self.theme_name}", classes="form-title")

        # Button showcase
        with Horizontal():
            yield AerospaceButton("Primary", variant=ButtonVariant.PRIMARY)
            yield AerospaceButton("Success", variant=ButtonVariant.SUCCESS)
            yield AerospaceButton("Warning", variant=ButtonVariant.WARNING)
            yield AerospaceButton("Error", variant=ButtonVariant.ERROR)

        # Input showcase
        yield ValidatedInput(
            "Sample Input",
            placeholder="Enter some text...",
            validation_rules=[
                ValidationRule(
                    "min_length",
                    lambda x: len(x) >= 3,
                    "Minimum 3 characters",
                ),
            ],
        )

        # Progress bar showcase
        yield AerospaceProgressBar(
            total=100,
            show_percentage=True,
            label="Sample Progress",
        )

        # Status indicators
        with Horizontal():
            yield StatusIndicator(StatusIndicator.StatusType.SUCCESS, "Success")
            yield StatusIndicator(StatusIndicator.StatusType.WARNING, "Warning")
            yield StatusIndicator(StatusIndicator.StatusType.ERROR, "Error")
            yield StatusIndicator(StatusIndicator.StatusType.INFO, "Info")


class ResponsiveLayoutDemo(Container):
    """Demonstrates responsive layout capabilities."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.responsive_layout = ResponsiveLayout()

    def compose(self) -> ComposeResult:
        yield Label("Responsive Layout Demo", classes="form-title")

        # Layout info
        layout_info = self.responsive_layout.get_layout_info()
        yield Label(f"Current Mode: {layout_info['mode']}")
        yield Label(f"Orientation: {layout_info['orientation']}")
        yield Label(
            f"Dimensions: {layout_info['dimensions']['width']}x{layout_info['dimensions']['height']}",
        )

        # Responsive components
        yield Container(
            Label("This content adapts to screen size", classes="hide-on-minimal"),
            Label("Minimal layout active", classes="show-on-minimal"),
            classes="responsive-content",
        )


class AnimationDemo(Container):
    """Demonstrates animation capabilities."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.animation_manager = AnimationManager()

    def compose(self) -> ComposeResult:
        yield Label("Animation Demo", classes="form-title")

        with Horizontal():
            yield AerospaceButton("Fade In", id="fade_in_btn")
            yield AerospaceButton("Pulse", id="pulse_btn")
            yield AerospaceButton("Shake", id="shake_btn")

        yield Label("Animated Text", id="animated_text", classes="animated-element")
        yield AerospaceProgressBar(id="animated_progress", label="Animated Progress")

    async def on_button_pressed(self, event) -> None:
        """Handle animation button presses."""
        if event.button.id == "fade_in_btn":
            text_widget = self.query_one("#animated_text")
            await self.animation_manager.fade_in(text_widget, duration=1.0)

        elif event.button.id == "pulse_btn":
            text_widget = self.query_one("#animated_text")
            await self.animation_manager.pulse(text_widget, duration=2.0, repeat=3)

        elif event.button.id == "shake_btn":
            text_widget = self.query_one("#animated_text")
            await self.animation_manager.shake(text_widget, duration=0.5, amplitude=2.0)


class DataTableDemo(Container):
    """Demonstrates aerospace data table capabilities."""

    def compose(self) -> ComposeResult:
        yield Label("Aerospace Data Table", classes="form-title")

        # Create aerospace data table
        table = AerospaceDataTable(sortable=True, filterable=True)

        # Add aerospace-specific columns
        table.add_aerospace_columns(
            [
                {"label": "Aircraft", "key": "aircraft", "width": 15},
                {"label": "Altitude", "key": "altitude", "width": 10},
                {"label": "Speed", "key": "speed", "width": 10},
                {"label": "Mach", "key": "mach", "width": 8},
                {"label": "Angle", "key": "angle", "width": 8},
            ],
        )

        # Add sample data
        sample_data = [
            {
                "aircraft": "Boeing 737",
                "altitude": 35000,
                "speed": 450,
                "mach": 0.78,
                "angle": 2.5,
            },
            {
                "aircraft": "Airbus A320",
                "altitude": 37000,
                "speed": 460,
                "mach": 0.80,
                "angle": 1.8,
            },
            {
                "aircraft": "F-16 Falcon",
                "altitude": 25000,
                "speed": 800,
                "mach": 1.2,
                "angle": 15.0,
            },
            {
                "aircraft": "Cessna 172",
                "altitude": 8000,
                "speed": 120,
                "mach": 0.18,
                "angle": 5.0,
            },
        ]

        for data in sample_data:
            table.add_aerospace_row(data)

        yield table


class FormDemo(Container):
    """Demonstrates form capabilities with validation."""

    def compose(self) -> ComposeResult:
        # Create form container
        form = FormContainer(
            title="Aircraft Configuration",
            submit_label="Analyze",
            cancel_label="Reset",
        )

        # Add form fields
        aircraft_name = ValidatedInput(
            "Aircraft Name",
            placeholder="Enter aircraft name",
            validation_rules=[
                ValidationRule(
                    "required",
                    lambda x: bool(x.strip()),
                    "Aircraft name is required",
                ),
                ValidationRule(
                    "min_length",
                    lambda x: len(x.strip()) >= 3,
                    "Name must be at least 3 characters",
                ),
            ],
            id="aircraft_name",
        )

        altitude = ValidatedInput(
            "Cruise Altitude (ft)",
            placeholder="35000",
            input_type=InputType.NUMBER,
            validation_rules=[
                ValidationRule(
                    "required",
                    lambda x: bool(x.strip()),
                    "Altitude is required",
                ),
                ValidationRule(
                    "numeric",
                    lambda x: x.replace(".", "").isdigit(),
                    "Must be a number",
                ),
                ValidationRule(
                    "range",
                    lambda x: 1000 <= float(x) <= 60000
                    if x.replace(".", "").isdigit()
                    else False,
                    "Altitude must be between 1,000 and 60,000 ft",
                ),
            ],
            id="altitude",
        )

        mach_number = ValidatedInput(
            "Mach Number",
            placeholder="0.78",
            input_type=InputType.NUMBER,
            validation_rules=[
                ValidationRule(
                    "required",
                    lambda x: bool(x.strip()),
                    "Mach number is required",
                ),
                ValidationRule(
                    "numeric",
                    lambda x: x.replace(".", "").isdigit(),
                    "Must be a number",
                ),
                ValidationRule(
                    "range",
                    lambda x: 0.1 <= float(x) <= 5.0
                    if x.replace(".", "").replace("-", "").isdigit()
                    else False,
                    "Mach must be between 0.1 and 5.0",
                ),
            ],
            id="mach_number",
        )

        form.add_field(aircraft_name)
        form.add_field(altitude)
        form.add_field(mach_number)

        yield form

    def on_form_container_form_submitted(self, event) -> None:
        """Handle form submission."""
        if event.is_valid:
            self.notify("Form submitted successfully!", severity="information")
        else:
            self.notify("Please fix form errors", severity="error")


class MainDemoScreen(Screen):
    """Main demo screen showcasing all components."""

    BINDINGS = [
        Binding("1", "switch_theme('aerospace_dark')", "Aerospace Dark"),
        Binding("2", "switch_theme('aerospace_light')", "Aerospace Light"),
        Binding("3", "switch_theme('aviation_blue')", "Aviation Blue"),
        Binding("4", "switch_theme('space_dark')", "Space Dark"),
        Binding("5", "switch_theme('cockpit_green')", "Cockpit Green"),
        Binding("6", "switch_theme('high_contrast')", "High Contrast"),
        Binding("7", "switch_theme('classic_terminal')", "Classic Terminal"),
        Binding("r", "toggle_responsive", "Toggle Responsive"),
        Binding("a", "demo_animations", "Demo Animations"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, theme_manager: ThemeManager, **kwargs):
        super().__init__(**kwargs)
        self.theme_manager = theme_manager
        self.responsive_layout = ResponsiveLayout()
        self.animation_manager = AnimationManager()
        self.current_theme = "aerospace_dark"

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(classes="main-container"):
            # Sidebar with theme preview
            with Container(classes="sidebar"):
                yield Label("Theme System Demo", classes="panel-title")
                yield ThemePreviewWidget(self.current_theme)

                yield Label("Controls", classes="panel-title")
                yield Label("1-7: Switch Themes")
                yield Label("R: Toggle Responsive")
                yield Label("A: Demo Animations")
                yield Label("Q: Quit")

            # Main content area
            with Container(classes="content"):
                with Vertical():
                    yield ResponsiveLayoutDemo()
                    yield DataTableDemo()
                    yield FormDemo()
                    yield AnimationDemo()

                    # Notification panel
                    yield NotificationPanel(id="notifications")

        yield Footer()

    def action_switch_theme(self, theme_id: str) -> None:
        """Switch to a different theme."""
        if self.theme_manager.set_theme(theme_id):
            self.current_theme = theme_id
            self.notify(f"Switched to {theme_id} theme", severity="information")

            # Update theme preview
            theme_preview = self.query_one(ThemePreviewWidget)
            theme_preview.theme_name = theme_id
            theme_preview.refresh()
        else:
            self.notify(f"Failed to switch to {theme_id}", severity="error")

    def action_toggle_responsive(self) -> None:
        """Toggle responsive layout demonstration."""
        # Simulate different screen sizes
        current_width, current_height = self.responsive_layout.get_dimensions()

        if current_width >= 120:
            new_width = 60  # Switch to compact
        elif current_width >= 80:
            new_width = 40  # Switch to minimal
        else:
            new_width = 120  # Switch to expanded

        self.responsive_layout.update_dimensions(new_width, current_height)
        layout_info = self.responsive_layout.get_layout_info()

        self.notify(
            f"Layout: {layout_info['mode']} ({new_width}x{current_height})",
            severity="information",
        )

    async def action_demo_animations(self) -> None:
        """Demonstrate various animations."""
        # Get notification panel
        notifications = self.query_one("#notifications", NotificationPanel)

        # Animate different elements
        notifications.add_notification(
            "Starting animation demo...",
            NotificationPanel.NotificationType.INFO,
        )

        # Fade in animation
        await asyncio.sleep(0.5)
        notifications.add_notification(
            "Fade in animation",
            NotificationPanel.NotificationType.SUCCESS,
        )

        # Pulse animation
        await asyncio.sleep(1.0)
        notifications.add_notification(
            "Pulse animation",
            NotificationPanel.NotificationType.WARNING,
        )

        # Shake animation
        await asyncio.sleep(1.0)
        notifications.add_notification(
            "Shake animation",
            NotificationPanel.NotificationType.ERROR,
        )

        notifications.add_notification(
            "Animation demo complete!",
            NotificationPanel.NotificationType.SUCCESS,
        )


class ThemeSystemDemoApp(App):
    """Demo application for the ICARUS CLI theme system."""

    TITLE = "ICARUS CLI Theme System Demo"
    SUB_TITLE = "Aerospace-Focused TUI Components"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme_manager = ThemeManager()
        self.transition_manager = ScreenTransitionManager(self)

        # Set initial theme
        self.theme_manager.set_theme("aerospace_dark")

    def on_mount(self) -> None:
        """Setup the application."""
        # Apply current theme CSS
        self.stylesheet.add_source(self.theme_manager.get_current_css())

        # Setup theme change callback
        self.theme_manager.add_theme_change_callback(self._on_theme_changed)

        # Setup responsive layout
        self._setup_responsive_layout()

    def _on_theme_changed(self, theme) -> None:
        """Handle theme changes."""
        # Update stylesheet
        self.stylesheet.clear()
        self.stylesheet.add_source(self.theme_manager.get_current_css())
        self.refresh()

    def _setup_responsive_layout(self) -> None:
        """Setup responsive layout monitoring."""
        # In a real app, this would monitor terminal size changes
        # For demo purposes, we'll use fixed dimensions
        pass

    def compose(self) -> ComposeResult:
        yield MainDemoScreen(self.theme_manager)


def main():
    """Run the theme system demo."""
    app = ThemeSystemDemoApp()
    app.run()


if __name__ == "__main__":
    main()
