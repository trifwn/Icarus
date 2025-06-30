"""Validation Helper Utilities for ICARUS TUI

This module provides utilities for working with validation and the core validation service.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from core.services import validation_service


@dataclass
class ValidationResult:
    """Represents a validation result."""

    is_valid: bool
    errors: Dict[str, List[str]]
    warnings: List[str]


class ValidationHelper:
    """Helper class for managing validation in the TUI."""

    def __init__(self):
        self.validation_service = validation_service

    def validate_airfoil_data(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate airfoil analysis data."""
        try:
            errors = self.validation_service.validate_data(data, "airfoil")
            warnings = self._check_airfoil_warnings(data)

            return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)
        except Exception as e:
            return ValidationResult(is_valid=False, errors={"validation": [f"Validation error: {e}"]}, warnings=[])

    def validate_airplane_data(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate airplane analysis data."""
        try:
            errors = self.validation_service.validate_data(data, "airplane")
            warnings = self._check_airplane_warnings(data)

            return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)
        except Exception as e:
            return ValidationResult(is_valid=False, errors={"validation": [f"Validation error: {e}"]}, warnings=[])

    def validate_solver_config(self, solver_name: str, config: Dict[str, Any]) -> ValidationResult:
        """Validate solver configuration."""
        try:
            # Validate solver name
            solver_data = {"name": solver_name}
            errors = self.validation_service.validate_data(solver_data, "solver")

            # Add solver-specific validation
            solver_errors = self._validate_solver_specific_config(solver_name, config)
            errors.update(solver_errors)

            warnings = self._check_solver_warnings(solver_name, config)

            return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)
        except Exception as e:
            return ValidationResult(is_valid=False, errors={"validation": [f"Validation error: {e}"]}, warnings=[])

    def _check_airfoil_warnings(self, data: Dict[str, Any]) -> List[str]:
        """Check for airfoil-specific warnings."""
        warnings = []

        # Check Reynolds number range
        reynolds = data.get("reynolds")
        if reynolds:
            try:
                reynolds_num = float(reynolds)
                if reynolds_num < 1e4:
                    warnings.append("Very low Reynolds number - results may be unreliable")
                elif reynolds_num > 1e7:
                    warnings.append("Very high Reynolds number - consider compressibility effects")
            except (ValueError, TypeError):
                pass

        # Check angle range
        angles = data.get("angles")
        if angles:
            try:
                parts = angles.split(":")
                if len(parts) == 3:
                    start, end, steps = map(float, parts)
                    if end - start > 30:
                        warnings.append("Large angle range - analysis may take longer")
                    if steps > 50:
                        warnings.append("Many angle steps - analysis may take longer")
            except (ValueError, TypeError):
                pass

        return warnings

    def _check_airplane_warnings(self, data: Dict[str, Any]) -> List[str]:
        """Check for airplane-specific warnings."""
        warnings = []

        # Check altitude
        altitude = data.get("altitude")
        if altitude:
            try:
                alt = float(altitude)
                if alt > 40000:
                    warnings.append("High altitude - consider compressibility effects")
            except (ValueError, TypeError):
                pass

        # Check Mach number
        mach = data.get("mach")
        if mach:
            try:
                mach_num = float(mach)
                if mach_num > 0.8:
                    warnings.append("High Mach number - consider compressibility effects")
            except (ValueError, TypeError):
                pass

        return warnings

    def _validate_solver_specific_config(self, solver_name: str, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate solver-specific configuration."""
        errors = {}

        if solver_name == "xfoil":
            # XFOIL-specific validation
            if "iterations" in config:
                try:
                    iterations = int(config["iterations"])
                    if iterations < 10 or iterations > 1000:
                        errors["iterations"] = ["XFOIL iterations must be between 10 and 1000"]
                except (ValueError, TypeError):
                    errors["iterations"] = ["Invalid iterations value"]

        elif solver_name == "openfoam":
            # OpenFOAM-specific validation
            if "mesh_quality" in config:
                try:
                    quality = float(config["mesh_quality"])
                    if quality < 0.1 or quality > 1.0:
                        errors["mesh_quality"] = ["Mesh quality must be between 0.1 and 1.0"]
                except (ValueError, TypeError):
                    errors["mesh_quality"] = ["Invalid mesh quality value"]

        return errors

    def _check_solver_warnings(self, solver_name: str, config: Dict[str, Any]) -> List[str]:
        """Check for solver-specific warnings."""
        warnings = []

        if solver_name == "xfoil":
            if config.get("iterations", 100) > 500:
                warnings.append("High iteration count - XFOIL may take longer to converge")

        elif solver_name == "openfoam":
            if config.get("mesh_quality", 0.5) < 0.3:
                warnings.append("Low mesh quality - results may be inaccurate")

        return warnings

    def format_validation_result(self, result: ValidationResult) -> str:
        """Format validation result for display."""
        lines = []

        if result.is_valid:
            lines.append("✓ Validation passed")
        else:
            lines.append("✗ Validation failed")

        if result.errors:
            lines.append("\nErrors:")
            for field, field_errors in result.errors.items():
                for error in field_errors:
                    lines.append(f"  • {field}: {error}")

        if result.warnings:
            lines.append("\nWarnings:")
            for warning in result.warnings:
                lines.append(f"  ⚠ {warning}")

        return "\n".join(lines)
