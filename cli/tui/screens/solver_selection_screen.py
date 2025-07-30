"""Solver Selection Screen

This screen provides an advanced solver selection interface with capability detection,
performance comparison, and detailed solver information.
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
from textual.widgets import Button
from textual.widgets import Collapsible
from textual.widgets import DataTable
from textual.widgets import Label
from textual.widgets import Select
from textual.widgets import Static
from textual.widgets import Switch

# Import integration modules
try:
    from ...integration.models import AnalysisType
    from ...integration.models import SolverType
    from ...integration.solver_manager import SolverManager
except ImportError:
    SolverManager = None
    AnalysisType = None
    SolverType = None


class SolverStatusIndicator(Static):
    """Visual indicator for solver status."""

    def __init__(self, solver_info: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.solver_info = solver_info
        self.is_available = solver_info.get("is_available", False)
        self.fidelity = solver_info.get("fidelity", 1)

    def render(self) -> str:
        """Render the status indicator."""
        status_icon = "🟢" if self.is_available else "🔴"
        fidelity_stars = "⭐" * self.fidelity + "☆" * (3 - self.fidelity)

        return f"{status_icon} {fidelity_stars}"


class SolverDetailPanel(Container):
    """Detailed information panel for a selected solver."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_solver = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Solver Details", classes="section-title")
            yield Container(id="solver_info_container")
            yield Label("Requirements", classes="subsection-title")
            yield Container(id="requirements_container")
            yield Label("Capabilities", classes="subsection-title")
            yield Container(id="capabilities_container")
            yield Label("Performance Characteristics", classes="subsection-title")
            yield Container(id="performance_container")

    def update_solver_details(self, solver_info: Dict[str, Any]) -> None:
        """Update the panel with solver details."""
        self.current_solver = solver_info

        # Update basic info
        info_container = self.query_one("#solver_info_container", Container)
        info_container.remove_children()

        info_container.mount(
            Vertical(
                Label(
                    f"Name: {solver_info.get('name', 'Unknown')}",
                    classes="info-item",
                ),
                Label(
                    f"Version: {solver_info.get('version', 'Unknown')}",
                    classes="info-item",
                ),
                Label(
                    f"Type: {solver_info.get('type', 'Unknown')}",
                    classes="info-item",
                ),
                Label(
                    f"Fidelity: {'⭐' * solver_info.get('fidelity', 1)}",
                    classes="info-item",
                ),
                Label(
                    f"Status: {'Available' if solver_info.get('is_available') else 'Unavailable'}",
                    classes="info-item",
                ),
            ),
        )

        # Update requirements
        self._update_requirements(solver_info)

        # Update capabilities
        self._update_capabilities(solver_info)

        # Update performance characteristics
        self._update_performance(solver_info)

    def _update_requirements(self, solver_info: Dict[str, Any]) -> None:
        """Update requirements section."""
        req_container = self.query_one("#requirements_container", Container)
        req_container.remove_children()

        requirements = solver_info.get("requirements", [])
        if not requirements:
            req_container.mount(Label("No special requirements", classes="info-item"))
            return

        for req in requirements:
            status_icon = "✅" if self._check_requirement(req) else "❌"
            req_container.mount(
                Label(f"{status_icon} {req}", classes="requirement-item"),
            )

    def _update_capabilities(self, solver_info: Dict[str, Any]) -> None:
        """Update capabilities section."""
        cap_container = self.query_one("#capabilities_container", Container)
        cap_container.remove_children()

        supported_analyses = solver_info.get("supported_analyses", [])
        if not supported_analyses:
            cap_container.mount(
                Label("No analysis types specified", classes="info-item"),
            )
            return

        for analysis in supported_analyses:
            analysis_name = analysis if isinstance(analysis, str) else analysis.value
            cap_container.mount(
                Label(
                    f"• {analysis_name.replace('_', ' ').title()}",
                    classes="capability-item",
                ),
            )

    def _update_performance(self, solver_info: Dict[str, Any]) -> None:
        """Update performance characteristics."""
        perf_container = self.query_one("#performance_container", Container)
        perf_container.remove_children()

        fidelity = solver_info.get("fidelity", 1)

        # Performance characteristics based on fidelity and solver type
        characteristics = self._get_performance_characteristics(solver_info)

        for char_name, char_value in characteristics.items():
            perf_container.mount(
                Label(f"{char_name}: {char_value}", classes="performance-item"),
            )

    def _check_requirement(self, requirement: str) -> bool:
        """Check if a requirement is met."""
        # Simplified requirement checking
        # In a real implementation, this would check for actual executables, libraries, etc.
        return True  # Placeholder

    def _get_performance_characteristics(
        self,
        solver_info: Dict[str, Any],
    ) -> Dict[str, str]:
        """Get performance characteristics for the solver."""
        solver_name = solver_info.get("name", "").lower()
        fidelity = solver_info.get("fidelity", 1)

        base_chars = {
            "Computational Speed": "Fast"
            if fidelity <= 2
            else "Moderate"
            if fidelity == 3
            else "Slow",
            "Memory Usage": "Low"
            if fidelity <= 2
            else "Moderate"
            if fidelity == 3
            else "High",
            "Accuracy": "Good"
            if fidelity <= 2
            else "High"
            if fidelity == 3
            else "Very High",
        }

        # Solver-specific characteristics
        if "xfoil" in solver_name:
            base_chars.update(
                {
                    "Convergence": "Generally Good",
                    "Stall Prediction": "Limited",
                    "Viscous Effects": "Included",
                },
            )
        elif "avl" in solver_name:
            base_chars.update(
                {
                    "Convergence": "Excellent",
                    "Compressibility": "Limited",
                    "Stability Analysis": "Excellent",
                },
            )
        elif "genuvp" in solver_name:
            base_chars.update(
                {
                    "Unsteady Effects": "Excellent",
                    "Wake Modeling": "Advanced",
                    "Computational Cost": "High",
                },
            )
        elif "openfoam" in solver_name:
            base_chars.update(
                {
                    "Turbulence Modeling": "Advanced",
                    "Mesh Requirements": "High",
                    "Parallel Scaling": "Excellent",
                },
            )

        return base_chars


class SolverComparisonTable(DataTable):
    """Table for comparing multiple solvers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.solvers_data = []

    def update_comparison(
        self,
        solvers: List[Dict[str, Any]],
        analysis_type: str,
    ) -> None:
        """Update the comparison table with solver data."""
        self.solvers_data = solvers
        self.clear(columns=True)

        if not solvers:
            self.add_column("Message")
            self.add_row("No solvers available for comparison")
            return

        # Add columns
        self.add_columns(
            "Solver",
            "Status",
            "Fidelity",
            "Speed",
            "Accuracy",
            "Memory",
            "Recommended",
        )

        # Sort solvers by availability and fidelity
        sorted_solvers = sorted(
            solvers,
            key=lambda s: (s.get("is_available", False), s.get("fidelity", 0)),
            reverse=True,
        )

        for solver in sorted_solvers:
            status = (
                "✅ Available"
                if solver.get("is_available", False)
                else "❌ Unavailable"
            )
            fidelity = "⭐" * solver.get("fidelity", 1)

            # Performance ratings based on fidelity
            fidelity_level = solver.get("fidelity", 1)
            speed = (
                "Fast"
                if fidelity_level <= 2
                else "Moderate"
                if fidelity_level == 3
                else "Slow"
            )
            accuracy = (
                "Good"
                if fidelity_level <= 2
                else "High"
                if fidelity_level == 3
                else "Excellent"
            )
            memory = (
                "Low"
                if fidelity_level <= 2
                else "Moderate"
                if fidelity_level == 3
                else "High"
            )

            # Recommendation based on availability and suitability
            recommended = (
                "⭐"
                if solver.get("is_available")
                and self._is_recommended_for_analysis(solver, analysis_type)
                else ""
            )

            self.add_row(
                solver.get("name", "Unknown"),
                status,
                fidelity,
                speed,
                accuracy,
                memory,
                recommended,
            )

    def _is_recommended_for_analysis(
        self,
        solver: Dict[str, Any],
        analysis_type: str,
    ) -> bool:
        """Check if solver is recommended for the analysis type."""
        supported_analyses = solver.get("supported_analyses", [])

        # Convert analysis type to match supported analyses format
        for analysis in supported_analyses:
            analysis_name = (
                analysis
                if isinstance(analysis, str)
                else getattr(analysis, "value", str(analysis))
            )
            if analysis_type.upper() in analysis_name.upper():
                return True

        return False


class SolverSelectionForm(Container):
    """Main solver selection form."""

    def __init__(self, analysis_type: str = "AIRFOIL_POLAR", **kwargs):
        super().__init__(**kwargs)
        self.analysis_type = analysis_type
        self.solver_manager = SolverManager() if SolverManager else None
        self.available_solvers = []
        self.selected_solver = reactive("")
        self.auto_select_best = reactive(True)

    def compose(self) -> ComposeResult:
        with ScrollableContainer():
            with Vertical():
                yield Label("Solver Selection", classes="form-title")

                # Analysis type display
                yield Label(
                    f"Analysis Type: {self.analysis_type.replace('_', ' ').title()}",
                    classes="analysis-type",
                )

                # Auto-selection option
                with Horizontal():
                    yield Label("Auto-select best solver:", classes="field-label")
                    yield Switch(value=True, id="auto_select_switch")

                # Solver selection
                with Horizontal():
                    yield Label("Selected Solver:", classes="field-label")
                    yield Select(options=[("Loading...", "")], id="solver_select")
                    yield Button("Refresh", id="refresh_solvers_btn", variant="default")

                # Solver comparison table
                yield Label("Solver Comparison", classes="section-title")
                yield SolverComparisonTable(id="solver_comparison")

                # Detailed solver information
                yield Label("Solver Details", classes="section-title")
                yield SolverDetailPanel(id="solver_details")

                # Advanced options
                with Collapsible(title="Advanced Solver Options", collapsed=True):
                    yield Container(id="advanced_solver_options")

    def on_mount(self) -> None:
        """Initialize the form when mounted."""
        self._load_solvers()

    @work(exclusive=True)
    async def _load_solvers(self) -> None:
        """Load available solvers for the analysis type."""
        if not self.solver_manager:
            self.notify("Solver manager not available", severity="error")
            return

        try:
            # Get solvers for the specific analysis type
            if hasattr(AnalysisType, self.analysis_type):
                analysis_enum = getattr(AnalysisType, self.analysis_type)
                solvers = self.solver_manager.get_solvers_for_analysis(analysis_enum)
            else:
                solvers = self.solver_manager.get_all_solvers()

            # Convert to dict format
            self.available_solvers = [
                {
                    "name": solver.name,
                    "type": solver.solver_type.value
                    if hasattr(solver.solver_type, "value")
                    else str(solver.solver_type),
                    "version": solver.version,
                    "is_available": solver.is_available,
                    "fidelity": solver.fidelity_level,
                    "description": solver.description,
                    "supported_analyses": solver.supported_analyses,
                    "requirements": getattr(solver, "requirements", []),
                    "capabilities": getattr(solver, "capabilities", {}),
                }
                for solver in solvers
            ]

            # Update UI components
            self._update_solver_options()
            self._update_comparison_table()

            # Auto-select best solver if enabled
            if self.auto_select_best:
                self._auto_select_best_solver()

        except Exception as e:
            self.notify(f"Error loading solvers: {e}", severity="error")

    def _update_solver_options(self) -> None:
        """Update solver select options."""
        solver_select = self.query_one("#solver_select", Select)

        if not self.available_solvers:
            solver_select.set_options([("No solvers available", "")])
            return

        options = []
        for solver in self.available_solvers:
            status_icon = "✅" if solver["is_available"] else "❌"
            fidelity_stars = "⭐" * solver["fidelity"]
            options.append(
                (f"{status_icon} {solver['name']} {fidelity_stars}", solver["name"]),
            )

        solver_select.set_options(options)

    def _update_comparison_table(self) -> None:
        """Update the solver comparison table."""
        comparison_table = self.query_one("#solver_comparison", SolverComparisonTable)
        comparison_table.update_comparison(self.available_solvers, self.analysis_type)

    def _auto_select_best_solver(self) -> None:
        """Automatically select the best available solver."""
        if not self.available_solvers:
            return

        # Filter available solvers
        available_solvers = [s for s in self.available_solvers if s["is_available"]]

        if not available_solvers:
            self.notify("No solvers are currently available", severity="warning")
            return

        # Select highest fidelity available solver
        best_solver = max(available_solvers, key=lambda s: s["fidelity"])

        solver_select = self.query_one("#solver_select", Select)
        solver_select.value = best_solver["name"]
        self.selected_solver = best_solver["name"]

        # Update details panel
        self._update_solver_details(best_solver)

        self.notify(
            f"Auto-selected: {best_solver['name']} (Best available)",
            severity="success",
        )

    def _update_solver_details(self, solver_info: Dict[str, Any]) -> None:
        """Update the solver details panel."""
        details_panel = self.query_one("#solver_details", SolverDetailPanel)
        details_panel.update_solver_details(solver_info)

        # Update advanced options
        self._update_advanced_options(solver_info)

    def _update_advanced_options(self, solver_info: Dict[str, Any]) -> None:
        """Update advanced solver options."""
        options_container = self.query_one("#advanced_solver_options", Container)
        options_container.remove_children()

        solver_name = solver_info["name"].lower()

        # Define solver-specific options
        if "xfoil" in solver_name:
            options_container.mount(
                Vertical(
                    Label("XFoil Specific Options:", classes="subsection-title"),
                    Horizontal(
                        Label("Max Iterations:", classes="field-label"),
                        Select(
                            options=[
                                ("50", "50"),
                                ("100", "100"),
                                ("200", "200"),
                                ("500", "500"),
                            ],
                            value="100",
                            id="xfoil_max_iter",
                        ),
                    ),
                    Horizontal(
                        Label("Print Output:", classes="field-label"),
                        Switch(value=False, id="xfoil_print"),
                    ),
                    Horizontal(
                        Label("Transition Points:", classes="field-label"),
                        Select(
                            options=[("Auto", "auto"), ("Fixed (0.1, 0.1)", "fixed")],
                            value="auto",
                            id="xfoil_transition",
                        ),
                    ),
                ),
            )
        elif "avl" in solver_name:
            options_container.mount(
                Vertical(
                    Label("AVL Specific Options:", classes="subsection-title"),
                    Horizontal(
                        Label("Y-Symmetry:", classes="field-label"),
                        Switch(value=False, id="avl_ysym"),
                    ),
                    Horizontal(
                        Label("Z-Symmetry:", classes="field-label"),
                        Switch(value=False, id="avl_zsym"),
                    ),
                    Horizontal(
                        Label("Mach Number:", classes="field-label"),
                        Select(
                            options=[
                                ("0.0", "0.0"),
                                ("0.1", "0.1"),
                                ("0.2", "0.2"),
                                ("0.3", "0.3"),
                            ],
                            value="0.0",
                            id="avl_mach",
                        ),
                    ),
                ),
            )
        elif "genuvp" in solver_name:
            options_container.mount(
                Vertical(
                    Label("GenuVP Specific Options:", classes="subsection-title"),
                    Horizontal(
                        Label("Wake Length:", classes="field-label"),
                        Select(
                            options=[
                                ("5 chords", "5"),
                                ("10 chords", "10"),
                                ("20 chords", "20"),
                            ],
                            value="10",
                            id="genuvp_wake_length",
                        ),
                    ),
                    Horizontal(
                        Label("Time Steps:", classes="field-label"),
                        Select(
                            options=[("100", "100"), ("200", "200"), ("500", "500")],
                            value="200",
                            id="genuvp_time_steps",
                        ),
                    ),
                ),
            )

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select widget changes."""
        if event.select.id == "solver_select":
            self.selected_solver = event.value

            # Find solver info and update details
            solver_info = next(
                (s for s in self.available_solvers if s["name"] == event.value),
                None,
            )
            if solver_info:
                self._update_solver_details(solver_info)

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch changes."""
        if event.switch.id == "auto_select_switch":
            self.auto_select_best = event.value
            if event.value:
                self._auto_select_best_solver()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "refresh_solvers_btn":
            self._load_solvers()

    def get_selected_solver(self) -> Optional[Dict[str, Any]]:
        """Get the currently selected solver information."""
        if not self.selected_solver:
            return None

        return next(
            (s for s in self.available_solvers if s["name"] == self.selected_solver),
            None,
        )

    def get_solver_options(self) -> Dict[str, Any]:
        """Get the current solver-specific options."""
        options = {}

        if not self.selected_solver:
            return options

        solver_name = self.selected_solver.lower()

        try:
            if "xfoil" in solver_name:
                options["max_iter"] = int(
                    self.query_one("#xfoil_max_iter", Select).value,
                )
                options["print"] = self.query_one("#xfoil_print", Switch).value
                options["transition"] = self.query_one(
                    "#xfoil_transition",
                    Select,
                ).value
            elif "avl" in solver_name:
                options["iysym"] = 1 if self.query_one("#avl_ysym", Switch).value else 0
                options["izsym"] = 1 if self.query_one("#avl_zsym", Switch).value else 0
                options["mach"] = float(self.query_one("#avl_mach", Select).value)
            elif "genuvp" in solver_name:
                options["wake_length"] = int(
                    self.query_one("#genuvp_wake_length", Select).value,
                )
                options["time_steps"] = int(
                    self.query_one("#genuvp_time_steps", Select).value,
                )
        except Exception:
            pass

        return options


class SolverSelectionScreen(Screen):
    """Main solver selection screen."""

    BINDINGS = [
        Binding("ctrl+r", "refresh", "Refresh"),
        Binding("ctrl+a", "auto_select", "Auto Select"),
        Binding("escape", "back", "Back"),
    ]

    def __init__(self, analysis_type: str = "AIRFOIL_POLAR", **kwargs):
        super().__init__(**kwargs)
        self.analysis_type = analysis_type
        self.selection_form = None

    def compose(self) -> ComposeResult:
        with Container():
            yield Label("Solver Selection", classes="screen-title")
            yield SolverSelectionForm(
                analysis_type=self.analysis_type,
                id="solver_selection_form",
            )

            with Horizontal(classes="screen-actions"):
                yield Button(
                    "Use Selected Solver",
                    id="use_solver_btn",
                    variant="success",
                )
                yield Button("Back", id="back_btn", variant="default")

    def on_mount(self) -> None:
        """Initialize screen when mounted."""
        self.selection_form = self.query_one(
            "#solver_selection_form",
            SolverSelectionForm,
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "use_solver_btn":
            self._use_selected_solver()
        elif event.button.id == "back_btn":
            self.app.pop_screen()

    def _use_selected_solver(self) -> None:
        """Use the selected solver and return to previous screen."""
        if not self.selection_form:
            self.notify("Selection form not available", severity="error")
            return

        selected_solver = self.selection_form.get_selected_solver()
        solver_options = self.selection_form.get_solver_options()

        if not selected_solver:
            self.notify("Please select a solver", severity="warning")
            return

        if not selected_solver["is_available"]:
            self.notify("Selected solver is not available", severity="error")
            return

        # Return solver selection to parent screen
        self.app.pop_screen({"solver": selected_solver, "options": solver_options})

    def action_refresh(self) -> None:
        """Refresh solvers action."""
        if self.selection_form:
            self.selection_form._load_solvers()

    def action_auto_select(self) -> None:
        """Auto select best solver action."""
        if self.selection_form:
            self.selection_form._auto_select_best_solver()

    def action_back(self) -> None:
        """Go back action."""
        self.app.pop_screen()
