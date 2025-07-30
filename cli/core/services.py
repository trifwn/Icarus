"""Core Services for ICARUS CLI

This module provides validation, export/import, and other core services
for the enhanced CLI functionality.
"""

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import yaml
from rich.console import Console
from rich.table import Table

from .ui import notification_system
from .ui import theme_manager

console = Console()


@dataclass
class ValidationRule:
    """Represents a validation rule."""

    field: str
    rule_type: str
    parameters: Dict[str, Any]
    message: str
    required: bool = True


class ValidationService:
    """Service for validating user inputs and data."""

    def __init__(self):
        self.rules: Dict[str, List[ValidationRule]] = {}
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup default validation rules."""
        # Airfoil validation rules
        self.rules["airfoil"] = [
            ValidationRule(
                field="name",
                rule_type="string",
                parameters={"min_length": 1, "max_length": 50},
                message="Airfoil name must be between 1 and 50 characters",
                required=True,
            ),
            ValidationRule(
                field="reynolds",
                rule_type="range",
                parameters={"min": 1e4, "max": 1e8},
                message="Reynolds number must be between 10,000 and 100,000,000",
                required=True,
            ),
            ValidationRule(
                field="angles",
                rule_type="angle_range",
                parameters={"min": -20, "max": 30},
                message="Angle of attack range must be between -20 and 30 degrees",
                required=True,
            ),
        ]

        # Airplane validation rules
        self.rules["airplane"] = [
            ValidationRule(
                field="name",
                rule_type="string",
                parameters={"min_length": 1, "max_length": 50},
                message="Airplane name must be between 1 and 50 characters",
                required=True,
            ),
            ValidationRule(
                field="altitude",
                rule_type="range",
                parameters={"min": 0, "max": 50000},
                message="Altitude must be between 0 and 50,000 feet",
                required=True,
            ),
            ValidationRule(
                field="mach",
                rule_type="range",
                parameters={"min": 0.1, "max": 2.0},
                message="Mach number must be between 0.1 and 2.0",
                required=True,
            ),
        ]

        # Solver validation rules
        self.rules["solver"] = [
            ValidationRule(
                field="name",
                rule_type="choice",
                parameters={
                    "choices": [
                        "xfoil",
                        "foil2wake",
                        "openfoam",
                        "avl",
                        "gnvp3",
                        "gnvp7",
                        "lspt",
                    ],
                },
                message="Invalid solver name",
                required=True,
            ),
        ]

    def validate_data(
        self,
        data: Dict[str, Any],
        data_type: str,
    ) -> Dict[str, List[str]]:
        """Validate data against rules for a specific type."""
        errors = {}

        if data_type not in self.rules:
            return errors

        for rule in self.rules[data_type]:
            field_value = data.get(rule.field)

            # Check if required field is present
            if rule.required and field_value is None:
                if rule.field not in errors:
                    errors[rule.field] = []
                errors[rule.field].append(rule.message)
                continue

            # Skip validation if field is not required and not present
            if not rule.required and field_value is None:
                continue

            # Apply validation rule
            if not self._apply_rule(field_value, rule):
                if rule.field not in errors:
                    errors[rule.field] = []
                errors[rule.field].append(rule.message)

        return errors

    def _apply_rule(self, value: Any, rule: ValidationRule) -> bool:
        """Apply a validation rule to a value."""
        if rule.rule_type == "string":
            return self._validate_string(value, rule.parameters)
        elif rule.rule_type == "range":
            return self._validate_range(value, rule.parameters)
        elif rule.rule_type == "choice":
            return self._validate_choice(value, rule.parameters)
        elif rule.rule_type == "angle_range":
            return self._validate_angle_range(value, rule.parameters)
        elif rule.rule_type == "file_path":
            return self._validate_file_path(value, rule.parameters)
        elif rule.rule_type == "regex":
            return self._validate_regex(value, rule.parameters)
        else:
            return True  # Unknown rule type, assume valid

    def _validate_string(self, value: Any, params: Dict[str, Any]) -> bool:
        """Validate string value."""
        if not isinstance(value, str):
            return False

        min_length = params.get("min_length", 0)
        max_length = params.get("max_length", float("inf"))

        return min_length <= len(value) <= max_length

    def _validate_range(self, value: Any, params: Dict[str, Any]) -> bool:
        """Validate numeric range."""
        try:
            num_value = float(value)
            min_val = params.get("min", float("-inf"))
            max_val = params.get("max", float("inf"))
            return min_val <= num_value <= max_val
        except (ValueError, TypeError):
            return False

    def _validate_choice(self, value: Any, params: Dict[str, Any]) -> bool:
        """Validate choice from list."""
        choices = params.get("choices", [])
        return value in choices

    def _validate_angle_range(self, value: Any, params: Dict[str, Any]) -> bool:
        """Validate angle range string (e.g., '0:15:16')."""
        if not isinstance(value, str):
            return False

        try:
            parts = value.split(":")
            if len(parts) != 3:
                return False

            start, end, steps = map(float, parts)
            min_angle = params.get("min", -90)
            max_angle = params.get("max", 90)

            return (
                min_angle <= start <= max_angle
                and min_angle <= end <= max_angle
                and start < end
                and steps > 0
            )
        except (ValueError, TypeError):
            return False

    def _validate_file_path(self, value: Any, params: Dict[str, Any]) -> bool:
        """Validate file path."""
        if not isinstance(value, str):
            return False

        path = Path(value)
        must_exist = params.get("must_exist", False)

        if must_exist:
            return path.exists()
        else:
            return True  # Path format validation could be added here

    def _validate_regex(self, value: Any, params: Dict[str, Any]) -> bool:
        """Validate against regex pattern."""
        if not isinstance(value, str):
            return False

        pattern = params.get("pattern", "")
        try:
            return bool(re.match(pattern, value))
        except re.error:
            return False

    def add_rule(self, data_type: str, rule: ValidationRule):
        """Add a custom validation rule."""
        if data_type not in self.rules:
            self.rules[data_type] = []
        self.rules[data_type].append(rule)

    def get_validation_summary(self, errors: Dict[str, List[str]]) -> str:
        """Get a formatted summary of validation errors."""
        if not errors:
            return "✓ All validations passed"

        summary = "✗ Validation errors found:\n"
        for field, field_errors in errors.items():
            summary += f"  • {field}: {', '.join(field_errors)}\n"

        return summary


class ExportService:
    """Service for exporting and importing data in various formats."""

    def __init__(self):
        self.supported_formats = ["json", "csv", "yaml", "txt"]

    def export_data(self, data: Any, filepath: str, format: str = None) -> bool:
        """Export data to file in specified format."""
        if format is None:
            format = Path(filepath).suffix.lstrip(".")

        if format not in self.supported_formats:
            notification_system.error(f"Unsupported export format: {format}")
            return False

        try:
            if format == "json":
                self._export_json(data, filepath)
            elif format == "csv":
                self._export_csv(data, filepath)
            elif format == "yaml":
                self._export_yaml(data, filepath)
            elif format == "txt":
                self._export_txt(data, filepath)

            notification_system.success(f"Data exported to {filepath}")
            return True

        except Exception as e:
            notification_system.error(f"Export failed: {e}")
            return False

    def _export_json(self, data: Any, filepath: str):
        """Export data as JSON."""
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _export_csv(self, data: Any, filepath: str):
        """Export data as CSV."""
        if isinstance(data, list) and data:
            # Assume list of dictionaries
            fieldnames = data[0].keys()
            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
        else:
            # Single item or other structure
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                if isinstance(data, dict):
                    for key, value in data.items():
                        writer.writerow([key, value])
                else:
                    writer.writerow([data])

    def _export_yaml(self, data: Any, filepath: str):
        """Export data as YAML."""
        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)

    def _export_txt(self, data: Any, filepath: str):
        """Export data as plain text."""
        with open(filepath, "w") as f:
            if isinstance(data, dict):
                for key, value in data.items():
                    f.write(f"{key}: {value}\n")
            elif isinstance(data, list):
                for item in data:
                    f.write(f"{item}\n")
            else:
                f.write(str(data))

    def import_data(self, filepath: str, format: str = None) -> Optional[Any]:
        """Import data from file."""
        if format is None:
            format = Path(filepath).suffix.lstrip(".")

        if format not in self.supported_formats:
            notification_system.error(f"Unsupported import format: {format}")
            return None

        try:
            if format == "json":
                return self._import_json(filepath)
            elif format == "csv":
                return self._import_csv(filepath)
            elif format == "yaml":
                return self._import_yaml(filepath)
            elif format == "txt":
                return self._import_txt(filepath)

        except Exception as e:
            notification_system.error(f"Import failed: {e}")
            return None

    def _import_json(self, filepath: str) -> Any:
        """Import data from JSON file."""
        with open(filepath) as f:
            return json.load(f)

    def _import_csv(self, filepath: str) -> List[Dict[str, Any]]:
        """Import data from CSV file."""
        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _import_yaml(self, filepath: str) -> Any:
        """Import data from YAML file."""
        with open(filepath) as f:
            return yaml.safe_load(f)

    def _import_txt(self, filepath: str) -> str:
        """Import data from text file."""
        with open(filepath) as f:
            return f.read()

    def create_report(self, data: Dict[str, Any], report_type: str = "summary") -> str:
        """Create a formatted report from data."""
        if report_type == "summary":
            return self._create_summary_report(data)
        elif report_type == "detailed":
            return self._create_detailed_report(data)
        elif report_type == "table":
            return self._create_table_report(data)
        else:
            return str(data)

    def _create_summary_report(self, data: Dict[str, Any]) -> str:
        """Create a summary report."""
        report = f"Report generated on {data.get('timestamp', 'unknown')}\n"
        report += "=" * 50 + "\n\n"

        if "summary" in data:
            for key, value in data["summary"].items():
                report += f"{key.replace('_', ' ').title()}: {value}\n"

        return report

    def _create_detailed_report(self, data: Dict[str, Any]) -> str:
        """Create a detailed report."""
        report = f"Detailed Report - {data.get('timestamp', 'unknown')}\n"
        report += "=" * 50 + "\n\n"

        for section, content in data.items():
            if section != "timestamp":
                report += f"{section.replace('_', ' ').title()}:\n"
                report += "-" * 30 + "\n"
                report += str(content) + "\n\n"

        return report

    def _create_table_report(self, data: Dict[str, Any]) -> str:
        """Create a table-formatted report."""
        table = Table(title="Data Report")
        table.add_column("Property", style=theme_manager.get_color("text"))
        table.add_column("Value", style=theme_manager.get_color("accent"))

        for key, value in data.items():
            table.add_row(key.replace("_", " ").title(), str(value))

        return table


# Global service instances
validation_service = ValidationService()
export_service = ExportService()
