"""Data Helper Utilities for ICARUS TUI

This module provides utilities for working with data and the core state/export services.
"""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from core.services import export_service
from core.state import config_manager
from core.state import session_manager


class DataHelper:
    """Helper class for managing data in the TUI."""

    def __init__(self):
        self.session = session_manager
        self.config = config_manager
        self.export = export_service

    def get_session_data(self) -> Dict[str, Any]:
        """Get current session data."""
        try:
            return {
                "session_info": self.session.get_session_info(),
                "current_session": self.session.current_session.__dict__
                if self.session.current_session
                else {},
                "config": self.config.config,
            }
        except Exception:
            return {}

    def save_session_data(self, data: Dict[str, Any]) -> bool:
        """Save data to the current session."""
        try:
            for key, value in data.items():
                self.session.set_result(key, value)
            return True
        except Exception:
            return False

    def export_session_data(self, filepath: str, format_type: str = "json") -> bool:
        """Export session data to a file."""
        try:
            session_data = self.get_session_data()
            return self.export.export_data(session_data, filepath, format_type)
        except Exception:
            return False

    def import_session_data(
        self,
        filepath: str,
        format_type: str = "json",
    ) -> Optional[Dict[str, Any]]:
        """Import session data from a file."""
        try:
            return self.export.import_data(filepath, format_type)
        except Exception:
            return None

    def get_analysis_results(self, analysis_id: str = None) -> Dict[str, Any]:
        """Get analysis results from session."""
        if analysis_id:
            return self.session.get_result(analysis_id, {})
        else:
            return (
                self.session.current_session.last_results
                if self.session.current_session
                else {}
            )

    def save_analysis_results(self, analysis_id: str, results: Dict[str, Any]) -> bool:
        """Save analysis results to session."""
        try:
            self.session.set_result(analysis_id, results)
            return True
        except Exception:
            return False

    def get_airfoil_list(self) -> List[str]:
        """Get list of airfoils from session."""
        return (
            self.session.current_session.airfoils
            if self.session.current_session
            else []
        )

    def get_airplane_list(self) -> List[str]:
        """Get list of airplanes from session."""
        return (
            self.session.current_session.airplanes
            if self.session.current_session
            else []
        )

    def add_airfoil(self, airfoil_name: str) -> bool:
        """Add airfoil to session."""
        try:
            self.session.add_airfoil(airfoil_name)
            return True
        except Exception:
            return False

    def add_airplane(self, airplane_name: str) -> bool:
        """Add airplane to session."""
        try:
            self.session.add_airplane(airplane_name)
            return True
        except Exception:
            return False

    def generate_report(
        self,
        data: Dict[str, Any],
        report_type: str = "summary",
    ) -> str:
        """Generate a report from data."""
        try:
            return self.export.create_report(data, report_type)
        except Exception as e:
            return f"Error generating report: {e}"

    def get_data_summary(self) -> Dict[str, Any]:
        """Get a summary of all data."""
        return {
            "session": {
                "id": self.session.current_session.session_id
                if self.session.current_session
                else "None",
                "duration": self.session.get_session_info().get("duration", "Unknown"),
                "airfoils": len(self.get_airfoil_list()),
                "airplanes": len(self.get_airplane_list()),
                "results": len(self.get_analysis_results()),
            },
            "config": {
                "theme": self.config.get("theme", "default"),
                "database_path": self.config.get_database_path(),
                "auto_save": self.config.get("auto_save", True),
            },
        }
