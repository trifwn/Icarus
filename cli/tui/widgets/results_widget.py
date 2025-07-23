"""Results Widget for ICARUS TUI

This widget displays and manages analysis results using the core export services.
"""

from typing import Any
from typing import Dict

from core.services import export_service
from core.tui_integration import TUIEvent
from core.tui_integration import TUIEventType
from textual.reactive import reactive
from textual.widgets import DataTable


class ResultsWidget(DataTable):
    """Widget for displaying analysis results."""

    results_count = reactive(0)
    current_results = reactive({})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_columns("Parameter", "Value", "Unit")
        self.results_history = []

    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        self.add_row("Status", "Ready", "")

    def add_result(self, parameter: str, value: str, unit: str = "") -> None:
        """Add a result to the table."""
        self.add_row(parameter, value, unit)
        self.results_count += 1

    def clear_results(self) -> None:
        """Clear all results."""
        self.clear()
        self.add_columns("Parameter", "Value", "Unit")
        self.add_row("Status", "Ready", "")
        self.results_count = 0
        self.current_results = {}

    def _on_analysis_completed(self, event: TUIEvent) -> None:
        """Handle analysis completion events."""
        if event.type == TUIEventType.ANALYSIS_COMPLETED:
            self.update_with_results(event.data)

    def update_with_results(self, results: Dict[str, Any]) -> None:
        """Update the widget with new results."""
        self.clear_results()
        self.current_results = results

        if not results:
            self.add_row("Status", "No results available", "")
            return

        # Add results to table
        for key, value in results.items():
            if isinstance(value, dict):
                # Handle nested results
                for sub_key, sub_value in value.items():
                    self.add_row(f"{key}.{sub_key}", str(sub_value), "")
            else:
                self.add_row(key, str(value), "")

        self.results_count = len(results)
        self.add_row("Status", "Results loaded", "")

    def export_results(self, format_type: str = "json") -> bool:
        """Export current results."""
        if not self.current_results:
            return False

        try:
            # Generate filename based on timestamp
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"icarus_results_{timestamp}.{format_type}"

            # Export using core service
            success = export_service.export_data(
                self.current_results,
                filename,
                format_type,
            )

            if success:
                self.add_row("Export", f"Saved to {filename}", "")
                return True
            else:
                self.add_row("Export", "Failed", "")
                return False

        except Exception as e:
            self.add_row("Export", f"Error: {e}", "")
            return False

    def generate_report(self, report_type: str = "summary") -> str:
        """Generate a report from current results."""
        if not self.current_results:
            return "No results available for report generation."

        try:
            return export_service.create_report(self.current_results, report_type)
        except Exception as e:
            return f"Error generating report: {e}"

    def get_results_summary(self) -> Dict[str, Any]:
        """Get a summary of current results."""
        return {
            "count": self.results_count,
            "has_results": bool(self.current_results),
            "keys": list(self.current_results.keys()) if self.current_results else [],
        }
