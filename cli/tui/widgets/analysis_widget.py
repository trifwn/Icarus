"""Analysis Widget for ICARUS TUI

This widget provides analysis forms and validation using the core services.
"""

from textual.containers import Container
from textual.widgets import Input, Select, Button, Label
from textual.reactive import reactive
from typing import Dict, Any, List

from core.services import validation_service
from core.tui_integration import TUIEvent, TUIEventType


class AnalysisWidget(Container):
    """Widget for analysis configuration and execution."""

    analysis_type = reactive("airfoil")
    is_valid = reactive(False)
    validation_errors = reactive({})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analysis_config = {}

    def compose(self):
        """Compose the analysis form."""
        yield Label("Analysis Configuration", classes="title")

        # Analysis type selector
        yield Label("Analysis Type:")
        yield Select(
            [("Airfoil Analysis", "airfoil"), ("Airplane Analysis", "airplane")], value="airfoil", id="analysis_type"
        )

        # Airfoil analysis fields
        yield Label("Airfoil Name:", id="airfoil_name_label")
        yield Input(placeholder="Enter airfoil name", id="airfoil_name")

        yield Label("Solver:", id="solver_label")
        yield Select(
            [("XFOIL", "xfoil"), ("Foil2Wake", "foil2wake"), ("OpenFOAM", "openfoam")], value="xfoil", id="solver"
        )

        yield Label("Angle Range:", id="angles_label")
        yield Input(placeholder="0:15:16", value="0:15:16", id="angles")

        yield Label("Reynolds Number:", id="reynolds_label")
        yield Input(placeholder="1000000", value="1000000", id="reynolds")

        # Action buttons
        yield Button("Validate", id="validate_btn", variant="primary")
        yield Button("Run Analysis", id="run_btn", variant="success")
        yield Button("Clear", id="clear_btn", variant="default")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes."""
        if event.select.id == "analysis_type":
            self.analysis_type = event.value
            self.update_form_fields()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes."""
        self.update_analysis_config()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "validate_btn":
            self.validate_config()
        elif event.button.id == "run_btn":
            self.run_analysis()
        elif event.button.id == "clear_btn":
            self.clear_form()

    def update_form_fields(self) -> None:
        """Update form fields based on analysis type."""
        # Show/hide fields based on analysis type
        if self.analysis_type == "airfoil":
            self.query_one("#airfoil_name_label").display = True
            self.query_one("#airfoil_name").display = True
            self.query_one("#solver_label").display = True
            self.query_one("#solver").display = True
            self.query_one("#angles_label").display = True
            self.query_one("#angles").display = True
            self.query_one("#reynolds_label").display = True
            self.query_one("#reynolds").display = True
        else:
            # Airplane analysis fields
            self.query_one("#airfoil_name_label").display = False
            self.query_one("#airfoil_name").display = False
            self.query_one("#solver_label").display = False
            self.query_one("#solver").display = False
            self.query_one("#angles_label").display = False
            self.query_one("#angles").display = False
            self.query_one("#reynolds_label").display = False
            self.query_one("#reynolds").display = False

    def update_analysis_config(self) -> None:
        """Update the analysis configuration from form fields."""
        self.analysis_config = {
            "type": self.analysis_type,
            "airfoil_name": self.query_one("#airfoil_name").value,
            "solver": self.query_one("#solver").value,
            "angles": self.query_one("#angles").value,
            "reynolds": self.query_one("#reynolds").value,
        }

    def validate_config(self) -> None:
        """Validate the current configuration."""
        try:
            errors = validation_service.validate_data(self.analysis_config, self.analysis_type)
            self.validation_errors = errors
            self.is_valid = len(errors) == 0

            if self.is_valid:
                self.notify("Configuration is valid!", severity="information")
            else:
                self.notify("Configuration has errors!", severity="error")

        except Exception as e:
            self.notify(f"Validation error: {e}", severity="error")

    def run_analysis(self) -> None:
        """Run the analysis with current configuration."""
        if not self.is_valid:
            self.notify("Please validate configuration first!", severity="warning")
            return

        try:
            # Emit analysis started event
            from core.tui_integration import tui_integration

            if tui_integration:
                tui_integration.event_manager.emit(
                    TUIEvent(
                        type=TUIEventType.ANALYSIS_STARTED,
                        data=self.analysis_config,
                        timestamp=0.0,  # Will be set by event manager
                        source="analysis_widget",
                    )
                )

            self.notify("Analysis started!", severity="information")

        except Exception as e:
            self.notify(f"Failed to start analysis: {e}", severity="error")

    def clear_form(self) -> None:
        """Clear the form."""
        self.query_one("#airfoil_name").value = ""
        self.query_one("#angles").value = "0:15:16"
        self.query_one("#reynolds").value = "1000000"
        self.analysis_config = {}
        self.is_valid = False
        self.validation_errors = {}
        self.notify("Form cleared!", severity="information")
