"""Analysis Configuration Screen

This screen provides comprehensive analysis configuration forms with real-time validation,
parameter suggestions, and dynamic form updates based on analysis type and solver selection.
"""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.containers import Horizontal
from textual.containers import ScrollableContainer
from textual.containers import Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.validation import ValidationResult
from textual.validation import Validator
from textual.widgets import Button
from textual.widgets import Collapsible
from textual.widgets import DataTable
from textual.widgets import Input
from textual.widgets import Label
from textual.widgets import Select
from textual.widgets import Switch

# Import integration modules
try:
    from ...integration.analysis_service import AnalysisService
    from ...integration.models import AnalysisConfig
    from ...integration.models import AnalysisType
    from ...integration.models import SolverType
    from ...integration.models import ValidationResult as IntegrationValidationResult
    from ...integration.parameter_validator import ParameterValidator
    from ...integration.solver_manager import SolverManager
except ImportError:
    # Fallback for testing
    AnalysisService = None
    AnalysisConfig = None
    AnalysisType = None
    SolverType = None
    IntegrationValidationResult = None
    ParameterValidator = None
    SolverManager = None


class RealTimeValidator(Validator):
    """Custom validator for real-time parameter validation."""

    def __init__(
        self,
        param_name: str,
        analysis_type: str,
        validator_service: Optional[Any] = None,
    ):
        self.param_name = param_name
        self.analysis_type = analysis_type
        self.validator_service = validator_service
        super().__init__()

    def validate(self, value: str) -> ValidationResult:
        """Validate parameter value in real-time."""
        if not value.strip():
            return self.success()

        try:
            # Basic type validation
            if self.param_name in ["reynolds", "mach", "velocity", "altitude"]:
                if self.param_name == "reynolds" and "," in value:
                    # Handle Reynolds number lists
                    reynolds_list = [float(x.strip()) for x in value.split(",")]
                    for re in reynolds_list:
                        if re <= 0 or re > 1e8:
                            return self.failure(
                                f"Reynolds number {re} out of range (1e3 - 1e8)",
                            )
                else:
                    num_value = float(value)
                    if self.param_name == "reynolds" and (
                        num_value <= 0 or num_value > 1e8
                    ):
                        return self.failure(
                            "Reynolds number must be between 1e3 and 1e8",
                        )
                    elif self.param_name == "mach" and (
                        num_value < 0 or num_value > 0.9
                    ):
                        return self.failure("Mach number must be between 0.0 and 0.9")
                    elif self.param_name == "velocity" and (
                        num_value <= 0 or num_value > 300
                    ):
                        return self.failure("Velocity must be between 1 and 300 m/s")
                    elif self.param_name == "altitude" and (
                        num_value < 0 or num_value > 20000
                    ):
                        return self.failure("Altitude must be between 0 and 20000 m")

            elif self.param_name in ["min_aoa", "max_aoa", "aoa_step"]:
                num_value = float(value)
                if self.param_name in ["min_aoa", "max_aoa"] and (
                    num_value < -180 or num_value > 180
                ):
                    return self.failure(
                        "Angle of attack must be between -180 and 180 degrees",
                    )
                elif self.param_name == "aoa_step" and (
                    num_value <= 0 or num_value > 10
                ):
                    return self.failure("AoA step must be between 0.1 and 10 degrees")

            return self.success()

        except ValueError:
            return self.failure(f"Invalid number format for {self.param_name}")
        except Exception as e:
            return self.failure(f"Validation error: {str(e)}")


class ParameterInput(Container):
    """Enhanced parameter input with validation and suggestions."""

    def __init__(
        self,
        param_name: str,
        label: str,
        placeholder: str = "",
        default_value: str = "",
        description: str = "",
        analysis_type: str = "",
        validator_service: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.param_name = param_name
        self.label_text = label
        self.placeholder = placeholder
        self.default_value = default_value
        self.description = description
        self.analysis_type = analysis_type
        self.validator_service = validator_service
        self.is_valid = reactive(True)
        self.validation_message = reactive("")

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self.label_text, classes="param-label")
            if self.description:
                yield Label(self.description, classes="param-description")
            yield Input(
                placeholder=self.placeholder,
                value=self.default_value,
                validators=[
                    RealTimeValidator(
                        self.param_name,
                        self.analysis_type,
                        self.validator_service,
                    ),
                ],
                id=f"input_{self.param_name}",
            )
            yield Label(
                "",
                id=f"validation_{self.param_name}",
                classes="validation-message",
            )

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes and update validation."""
        if event.input.id == f"input_{self.param_name}":
            validation_label = self.query_one(f"#validation_{self.param_name}", Label)

            if event.validation_result and not event.validation_result.is_valid:
                self.is_valid = False
                self.validation_message = event.validation_result.failure_descriptions[
                    0
                ]
                validation_label.update(self.validation_message)
                validation_label.add_class("error")
            else:
                self.is_valid = True
                self.validation_message = ""
                validation_label.update("")
                validation_label.remove_class("error")

    @property
    def value(self) -> str:
        """Get the current input value."""
        try:
            return self.query_one(f"#input_{self.param_name}", Input).value
        except:
            return ""


class SolverCapabilityDisplay(Container):
    """Display solver capabilities and recommendations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_analysis_type = None
        self.available_solvers = []

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Solver Information", classes="section-title")
            yield DataTable(id="solver_capabilities_table")
            yield Label("", id="solver_recommendation", classes="recommendation")

    def update_solvers(self, analysis_type: str, solvers: List[Dict[str, Any]]) -> None:
        """Update solver display for the given analysis type."""
        self.current_analysis_type = analysis_type
        self.available_solvers = solvers

        table = self.query_one("#solver_capabilities_table", DataTable)
        table.clear(columns=True)

        if not solvers:
            table.add_column("Message")
            table.add_row("No solvers available for this analysis type")
            return

        # Add columns
        table.add_columns("Solver", "Fidelity", "Status", "Description")

        # Add solver rows
        for solver in solvers:
            status = (
                "✓ Available" if solver.get("is_available", False) else "✗ Unavailable"
            )
            fidelity = "●" * solver.get("fidelity", 1) + "○" * (
                3 - solver.get("fidelity", 1)
            )

            table.add_row(
                solver.get("name", "Unknown"),
                fidelity,
                status,
                solver.get("description", "No description")[:50] + "...",
            )

        # Update recommendation
        self._update_recommendation(solvers)

    def _update_recommendation(self, solvers: List[Dict[str, Any]]) -> None:
        """Update solver recommendation."""
        recommendation_label = self.query_one("#solver_recommendation", Label)

        available_solvers = [s for s in solvers if s.get("is_available", False)]

        if not available_solvers:
            recommendation_label.update(
                "⚠️ No solvers are currently available. Please check solver installations.",
            )
            recommendation_label.add_class("warning")
        else:
            # Find highest fidelity available solver
            best_solver = max(available_solvers, key=lambda s: s.get("fidelity", 0))
            recommendation_label.update(
                f"💡 Recommended: {best_solver['name']} (High fidelity, available)",
            )
            recommendation_label.remove_class("warning")
            recommendation_label.add_class("success")


class AnalysisConfigForm(Container):
    """Main analysis configuration form with dynamic fields."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analysis_service = AnalysisService() if AnalysisService else None
        self.current_analysis_type = reactive("AIRFOIL_POLAR")
        self.current_solver = reactive("")
        self.parameter_inputs = {}
        self.is_form_valid = reactive(False)
        self.validation_errors = reactive([])

    def compose(self) -> ComposeResult:
        with ScrollableContainer():
            with Vertical():
                # Analysis type selection
                yield Label("Analysis Configuration", classes="form-title")

                with Horizontal():
                    yield Label("Analysis Type:", classes="field-label")
                    yield Select(
                        options=[
                            ("Airfoil Polar Analysis", "AIRFOIL_POLAR"),
                            ("Airplane Polar Analysis", "AIRPLANE_POLAR"),
                            ("Airplane Stability Analysis", "AIRPLANE_STABILITY"),
                        ],
                        value="AIRFOIL_POLAR",
                        id="analysis_type_select",
                    )

                # Target specification
                with Horizontal():
                    yield Label("Target:", classes="field-label")
                    yield Input(
                        placeholder="Enter airfoil name, file path, or select from namespace",
                        id="target_input",
                    )
                    yield Button("Browse", id="browse_target_btn", variant="default")

                # Solver selection with capabilities
                yield Label("Solver Selection", classes="section-title")
                with Horizontal():
                    yield Label("Solver:", classes="field-label")
                    yield Select(options=[("Loading...", "")], id="solver_select")

                yield SolverCapabilityDisplay(id="solver_capabilities")

                # Dynamic parameter section
                yield Label("Analysis Parameters", classes="section-title")
                yield Container(id="parameters_container")

                # Advanced options
                with Collapsible(title="Advanced Options", collapsed=True):
                    yield Container(id="advanced_options_container")

                # Validation summary
                yield Label("Configuration Status", classes="section-title")
                yield Container(id="validation_summary")

                # Action buttons
                with Horizontal(classes="action-buttons"):
                    yield Button(
                        "Validate Configuration",
                        id="validate_btn",
                        variant="primary",
                    )
                    yield Button(
                        "Load Template",
                        id="load_template_btn",
                        variant="default",
                    )
                    yield Button(
                        "Save Template",
                        id="save_template_btn",
                        variant="default",
                    )
                    yield Button("Reset Form", id="reset_btn", variant="default")

    def on_mount(self) -> None:
        """Initialize form when mounted."""
        self._load_initial_data()
        self._update_form_for_analysis_type("AIRFOIL_POLAR")

    @work(exclusive=True)
    async def _load_initial_data(self) -> None:
        """Load initial data for the form."""
        if not self.analysis_service:
            return

        try:
            # Load available solvers
            solvers = self.analysis_service.get_available_solvers()
            self._update_solver_options("AIRFOIL_POLAR", solvers)
        except Exception as e:
            self.notify(f"Error loading initial data: {e}", severity="error")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select widget changes."""
        if event.select.id == "analysis_type_select":
            self.current_analysis_type = event.value
            self._update_form_for_analysis_type(event.value)
        elif event.select.id == "solver_select":
            self.current_solver = event.value
            self._update_advanced_options(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "validate_btn":
            self._validate_configuration()
        elif event.button.id == "load_template_btn":
            self._load_template()
        elif event.button.id == "save_template_btn":
            self._save_template()
        elif event.button.id == "reset_btn":
            self._reset_form()
        elif event.button.id == "browse_target_btn":
            self._browse_target()

    def _update_form_for_analysis_type(self, analysis_type: str) -> None:
        """Update form fields based on selected analysis type."""
        # Update solver options
        if self.analysis_service:
            try:
                if hasattr(AnalysisType, analysis_type):
                    analysis_enum = getattr(AnalysisType, analysis_type)
                    solvers = self.analysis_service.get_solvers_for_analysis(
                        analysis_enum,
                    )
                    self._update_solver_options(analysis_type, solvers)

                    # Update solver capabilities display
                    solver_capabilities = self.query_one(
                        "#solver_capabilities",
                        SolverCapabilityDisplay,
                    )
                    solver_capabilities.update_solvers(analysis_type, solvers)
            except Exception as e:
                self.notify(f"Error updating solvers: {e}", severity="error")

        # Update parameter fields
        self._update_parameter_fields(analysis_type)

    def _update_solver_options(
        self,
        analysis_type: str,
        solvers: List[Dict[str, Any]],
    ) -> None:
        """Update solver select options."""
        solver_select = self.query_one("#solver_select", Select)

        if not solvers:
            solver_select.set_options([("No solvers available", "")])
            return

        options = []
        for solver in solvers:
            status_icon = "✓" if solver.get("is_available", False) else "✗"
            options.append((f"{status_icon} {solver['name']}", solver["name"]))

        solver_select.set_options(options)

        # Select first available solver
        available_solvers = [s for s in solvers if s.get("is_available", False)]
        if available_solvers:
            solver_select.value = available_solvers[0]["name"]
            self.current_solver = available_solvers[0]["name"]

    def _update_parameter_fields(self, analysis_type: str) -> None:
        """Update parameter input fields based on analysis type."""
        container = self.query_one("#parameters_container", Container)
        container.remove_children()
        self.parameter_inputs.clear()

        # Get parameter suggestions if service is available
        parameter_suggestions = {}
        if self.analysis_service and hasattr(AnalysisType, analysis_type):
            try:
                analysis_enum = getattr(AnalysisType, analysis_type)
                parameter_suggestions = self.analysis_service.get_parameter_suggestions(
                    analysis_enum,
                )
            except Exception:
                pass

        # Define parameters for each analysis type
        if analysis_type == "AIRFOIL_POLAR":
            params = [
                (
                    "reynolds",
                    "Reynolds Number",
                    "1000000",
                    "Reynolds number(s) for analysis",
                ),
                (
                    "mach",
                    "Mach Number",
                    "0.0",
                    "Mach number (typically 0.0 for low-speed)",
                ),
                (
                    "min_aoa",
                    "Min Angle of Attack",
                    "-10.0",
                    "Minimum angle of attack in degrees",
                ),
                (
                    "max_aoa",
                    "Max Angle of Attack",
                    "15.0",
                    "Maximum angle of attack in degrees",
                ),
                ("aoa_step", "AoA Step", "0.5", "Angle of attack increment in degrees"),
                ("ncrit", "N-Critical", "9.0", "Critical amplification factor"),
            ]
        elif analysis_type == "AIRPLANE_POLAR":
            params = [
                ("velocity", "Velocity", "50.0", "Flight velocity in m/s"),
                ("altitude", "Altitude", "0.0", "Flight altitude in meters"),
                (
                    "min_aoa",
                    "Min Angle of Attack",
                    "-5.0",
                    "Minimum angle of attack in degrees",
                ),
                (
                    "max_aoa",
                    "Max Angle of Attack",
                    "15.0",
                    "Maximum angle of attack in degrees",
                ),
                ("aoa_step", "AoA Step", "1.0", "Angle of attack increment in degrees"),
                ("beta", "Sideslip Angle", "0.0", "Sideslip angle in degrees"),
            ]
        elif analysis_type == "AIRPLANE_STABILITY":
            params = [
                ("velocity", "Velocity", "50.0", "Flight velocity in m/s"),
                ("altitude", "Altitude", "0.0", "Flight altitude in meters"),
                ("trim_aoa", "Trim AoA", "0.0", "Trim angle of attack in degrees"),
            ]
        else:
            params = []

        # Create parameter inputs
        for param_name, label, default, description in params:
            # Use suggestion if available
            suggested_value = parameter_suggestions.get(param_name, default)
            if isinstance(suggested_value, (int, float)):
                suggested_value = str(suggested_value)

            param_input = ParameterInput(
                param_name=param_name,
                label=label,
                placeholder=f"Enter {label.lower()}",
                default_value=suggested_value,
                description=description,
                analysis_type=analysis_type,
                validator_service=self.analysis_service,
            )

            container.mount(param_input)
            self.parameter_inputs[param_name] = param_input

    def _update_advanced_options(self, solver_name: str) -> None:
        """Update advanced options based on selected solver."""
        container = self.query_one("#advanced_options_container", Container)
        container.remove_children()

        if not solver_name:
            return

        # Define solver-specific advanced options
        if solver_name.lower() == "xfoil":
            options = [
                (
                    "max_iter",
                    "Max Iterations",
                    "100",
                    "Maximum iterations for convergence",
                ),
                ("print", "Print Output", "false", "Print solver output to console"),
            ]
        elif solver_name.lower() == "avl":
            options = [
                ("iysym", "Y-Symmetry", "0", "Y-symmetry flag (0=no, 1=yes)"),
                ("izsym", "Z-Symmetry", "0", "Z-symmetry flag (0=no, 1=yes)"),
            ]
        else:
            options = []

        for param_name, label, default, description in options:
            if param_name == "print":
                # Boolean switch for print option
                container.mount(
                    Horizontal(
                        Label(label + ":", classes="field-label"),
                        Switch(value=default == "true", id=f"advanced_{param_name}"),
                        Label(description, classes="param-description"),
                    ),
                )
            else:
                container.mount(
                    ParameterInput(
                        param_name=f"advanced_{param_name}",
                        label=label,
                        placeholder=f"Enter {label.lower()}",
                        default_value=default,
                        description=description,
                        analysis_type=self.current_analysis_type,
                        validator_service=self.analysis_service,
                    ),
                )

    @work(exclusive=True)
    async def _validate_configuration(self) -> None:
        """Validate the current configuration."""
        if not self.analysis_service:
            self.notify("Analysis service not available", severity="error")
            return

        try:
            # Collect form data
            config_data = self._collect_form_data()

            if not config_data:
                self.notify("Please fill in required fields", severity="warning")
                return

            # Create analysis config
            analysis_config = self._create_analysis_config(config_data)

            # Validate configuration
            validation_result = self.analysis_service.validate_analysis_config(
                analysis_config,
            )

            # Update validation summary
            self._update_validation_summary(validation_result)

            if validation_result.is_valid:
                self.is_form_valid = True
                self.notify("Configuration is valid!", severity="success")
            else:
                self.is_form_valid = False
                self.notify(
                    "Configuration has errors. Please check the validation summary.",
                    severity="error",
                )

        except Exception as e:
            self.notify(f"Validation error: {e}", severity="error")

    def _collect_form_data(self) -> Dict[str, Any]:
        """Collect data from all form fields."""
        data = {}

        # Basic fields
        data["analysis_type"] = self.current_analysis_type
        data["solver"] = self.current_solver
        data["target"] = self.query_one("#target_input", Input).value

        # Parameter fields
        data["parameters"] = {}
        for param_name, param_input in self.parameter_inputs.items():
            value = param_input.value
            if value:
                # Try to convert to appropriate type
                try:
                    if param_name == "reynolds" and "," in value:
                        data["parameters"][param_name] = [
                            float(x.strip()) for x in value.split(",")
                        ]
                    elif param_name in [
                        "reynolds",
                        "mach",
                        "velocity",
                        "altitude",
                        "min_aoa",
                        "max_aoa",
                        "aoa_step",
                        "ncrit",
                        "beta",
                        "trim_aoa",
                    ]:
                        data["parameters"][param_name] = float(value)
                    else:
                        data["parameters"][param_name] = value
                except ValueError:
                    data["parameters"][param_name] = value

        # Advanced options
        data["solver_parameters"] = {}
        try:
            advanced_container = self.query_one(
                "#advanced_options_container",
                Container,
            )
            for widget in advanced_container.children:
                if isinstance(widget, ParameterInput):
                    param_name = widget.param_name.replace("advanced_", "")
                    value = widget.value
                    if value:
                        try:
                            data["solver_parameters"][param_name] = float(value)
                        except ValueError:
                            data["solver_parameters"][param_name] = value
                elif isinstance(widget, Horizontal):
                    # Handle switches
                    for child in widget.children:
                        if isinstance(child, Switch):
                            param_name = (
                                child.id.replace("advanced_", "") if child.id else None
                            )
                            if param_name:
                                data["solver_parameters"][param_name] = child.value
        except Exception:
            pass

        return data

    def _create_analysis_config(self, data: Dict[str, Any]) -> Any:
        """Create AnalysisConfig object from form data."""
        if not AnalysisConfig or not AnalysisType or not SolverType:
            return None

        try:
            analysis_type = getattr(AnalysisType, data["analysis_type"])
            solver_type = getattr(SolverType, data["solver"].upper())

            return AnalysisConfig(
                analysis_type=analysis_type,
                solver_type=solver_type,
                target=data["target"],
                parameters=data.get("parameters", {}),
                solver_parameters=data.get("solver_parameters", {}),
                output_format="json",
            )
        except Exception as e:
            raise ValueError(f"Error creating analysis config: {e}")

    def _update_validation_summary(self, validation_result: Any) -> None:
        """Update the validation summary display."""
        container = self.query_one("#validation_summary", Container)
        container.remove_children()

        if validation_result.is_valid:
            container.mount(
                Label(
                    "✅ Configuration is valid and ready for execution",
                    classes="success",
                ),
            )
        else:
            container.mount(Label("❌ Configuration has errors:", classes="error"))

            # Show errors
            for error in validation_result.errors:
                container.mount(
                    Label(f"• {error.field}: {error.message}", classes="error-detail"),
                )

            # Show warnings
            for warning in validation_result.warnings:
                container.mount(Label(f"⚠️ {warning}", classes="warning"))

            # Show suggestions
            for suggestion in validation_result.suggestions:
                container.mount(Label(f"💡 {suggestion}", classes="suggestion"))

    def _load_template(self) -> None:
        """Load a configuration template."""
        # TODO: Implement template loading
        self.notify("Template loading not yet implemented", severity="info")

    def _save_template(self) -> None:
        """Save current configuration as template."""
        # TODO: Implement template saving
        self.notify("Template saving not yet implemented", severity="info")

    def _reset_form(self) -> None:
        """Reset form to default values."""
        # Reset basic fields
        self.query_one("#analysis_type_select", Select).value = "AIRFOIL_POLAR"
        self.query_one("#target_input", Input).value = ""

        # Reset parameter fields
        for param_input in self.parameter_inputs.values():
            input_widget = param_input.query_one(
                f"#input_{param_input.param_name}",
                Input,
            )
            input_widget.value = param_input.default_value

        # Clear validation
        self.is_form_valid = False
        self._update_validation_summary(
            type(
                "ValidationResult",
                (),
                {"is_valid": True, "errors": [], "warnings": [], "suggestions": []},
            )(),
        )

        self.notify("Form reset to default values", severity="info")

    def _browse_target(self) -> None:
        """Browse for target file."""
        # TODO: Implement file browser
        self.notify("File browser not yet implemented", severity="info")

    def get_configuration(self) -> Optional[Dict[str, Any]]:
        """Get the current configuration if valid."""
        if self.is_form_valid:
            return self._collect_form_data()
        return None


class AnalysisScreen(Screen):
    """Main analysis configuration screen."""

    BINDINGS = [
        Binding("ctrl+v", "validate", "Validate"),
        Binding("ctrl+r", "reset", "Reset"),
        Binding("escape", "back", "Back"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config_form = None

    def compose(self) -> ComposeResult:
        with Container():
            yield Label("Analysis Configuration", classes="screen-title")
            yield AnalysisConfigForm(id="analysis_config_form")

            with Horizontal(classes="screen-actions"):
                yield Button(
                    "Continue to Execution",
                    id="continue_btn",
                    variant="success",
                )
                yield Button("Back to Main", id="back_btn", variant="default")

    def on_mount(self) -> None:
        """Initialize screen when mounted."""
        self.config_form = self.query_one("#analysis_config_form", AnalysisConfigForm)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "continue_btn":
            self._continue_to_execution()
        elif event.button.id == "back_btn":
            self.app.pop_screen()

    def _continue_to_execution(self) -> None:
        """Continue to execution screen if configuration is valid."""
        if not self.config_form:
            self.notify("Configuration form not available", severity="error")
            return

        config = self.config_form.get_configuration()
        if config:
            # Pass configuration to execution screen
            from .execution_screen import ExecutionScreen

            self.app.push_screen(ExecutionScreen(config))
        else:
            self.notify("Please validate configuration first", severity="warning")

    def action_validate(self) -> None:
        """Validate configuration action."""
        if self.config_form:
            self.config_form._validate_configuration()

    def action_reset(self) -> None:
        """Reset form action."""
        if self.config_form:
            self.config_form._reset_form()

    def action_back(self) -> None:
        """Go back action."""
        self.app.pop_screen()
