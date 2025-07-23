"""
Parameter validation system with comprehensive error handling.

This module provides validation for analysis parameters, solver parameters,
and configuration settings with detailed error reporting and suggestions.
"""

import logging
import re
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

from .models import AnalysisConfig
from .models import AnalysisType
from .models import SolverType
from .models import ValidationResult


class ParameterValidator:
    """Comprehensive parameter validation for ICARUS analyses."""

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._validation_rules = self._initialize_validation_rules()

    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize validation rules for different analysis types and parameters."""
        return {
            AnalysisType.AIRFOIL_POLAR: {
                "required_params": ["reynolds", "mach"],
                "optional_params": ["min_aoa", "max_aoa", "aoa_step", "ncrit"],
                "param_types": {
                    "reynolds": (int, float, list),
                    "mach": (int, float),
                    "min_aoa": (int, float),
                    "max_aoa": (int, float),
                    "aoa_step": (int, float),
                    "ncrit": (int, float),
                },
                "param_ranges": {
                    "reynolds": (1e3, 1e8),
                    "mach": (0.0, 0.9),
                    "min_aoa": (-180, 180),
                    "max_aoa": (-180, 180),
                    "aoa_step": (0.1, 10.0),
                    "ncrit": (1.0, 20.0),
                },
                "param_defaults": {
                    "min_aoa": -10.0,
                    "max_aoa": 15.0,
                    "aoa_step": 0.5,
                    "ncrit": 9.0,
                },
            },
            AnalysisType.AIRPLANE_POLAR: {
                "required_params": ["velocity", "altitude"],
                "optional_params": ["min_aoa", "max_aoa", "aoa_step", "beta"],
                "param_types": {
                    "velocity": (int, float),
                    "altitude": (int, float),
                    "min_aoa": (int, float),
                    "max_aoa": (int, float),
                    "aoa_step": (int, float),
                    "beta": (int, float),
                },
                "param_ranges": {
                    "velocity": (1.0, 300.0),  # m/s
                    "altitude": (0.0, 20000.0),  # m
                    "min_aoa": (-30, 30),
                    "max_aoa": (-30, 30),
                    "aoa_step": (0.1, 5.0),
                    "beta": (-30, 30),
                },
                "param_defaults": {
                    "min_aoa": -5.0,
                    "max_aoa": 15.0,
                    "aoa_step": 1.0,
                    "beta": 0.0,
                },
            },
            AnalysisType.AIRPLANE_STABILITY: {
                "required_params": ["velocity", "altitude"],
                "optional_params": ["trim_aoa", "cg_location"],
                "param_types": {
                    "velocity": (int, float),
                    "altitude": (int, float),
                    "trim_aoa": (int, float),
                    "cg_location": (list, tuple),
                },
                "param_ranges": {
                    "velocity": (1.0, 300.0),
                    "altitude": (0.0, 20000.0),
                    "trim_aoa": (-30, 30),
                },
                "param_defaults": {
                    "trim_aoa": 0.0,
                },
            },
        }

    def validate_analysis_config(self, config: AnalysisConfig) -> ValidationResult:
        """Validate a complete analysis configuration."""
        result = ValidationResult(is_valid=True)

        # Validate basic configuration
        self._validate_basic_config(config, result)

        # Validate target file/object
        self._validate_target(config, result)

        # Validate analysis parameters
        self._validate_analysis_parameters(config, result)

        # Validate solver parameters
        self._validate_solver_parameters(config, result)

        # Cross-validation checks
        self._validate_parameter_consistency(config, result)

        return result

    def _validate_basic_config(
        self,
        config: AnalysisConfig,
        result: ValidationResult,
    ) -> None:
        """Validate basic configuration fields."""
        # Check analysis type
        if not isinstance(config.analysis_type, AnalysisType):
            result.add_error("analysis_type", "Invalid analysis type", "type_error")

        # Check solver type
        if not isinstance(config.solver_type, SolverType):
            result.add_error("solver_type", "Invalid solver type", "type_error")

        # Check target
        if not config.target or not isinstance(config.target, str):
            result.add_error("target", "Target must be a non-empty string", "required")

        # Check output format
        valid_formats = ["json", "csv", "hdf5", "matlab", "pickle"]
        if config.output_format not in valid_formats:
            result.add_error(
                "output_format",
                f"Output format must be one of: {', '.join(valid_formats)}",
                "value_error",
                suggested_value="json",
            )

    def _validate_target(
        self,
        config: AnalysisConfig,
        result: ValidationResult,
    ) -> None:
        """Validate the analysis target (file, airfoil name, etc.)."""
        target = config.target

        if config.analysis_type in [
            AnalysisType.AIRPLANE_POLAR,
            AnalysisType.AIRPLANE_STABILITY,
        ]:
            # For airplane analyses, target should be a file path
            if not self._is_valid_file_path(target):
                result.add_error(
                    "target",
                    f"Target file '{target}' does not exist or is not accessible",
                    "file_error",
                )
            elif not self._is_valid_airplane_file(target):
                result.add_error(
                    "target",
                    f"Target file '{target}' is not a valid airplane definition file",
                    "format_error",
                )

        elif config.analysis_type == AnalysisType.AIRFOIL_POLAR:
            # For airfoil analyses, target can be a file or airfoil name
            if self._looks_like_file_path(target):
                if not self._is_valid_file_path(target):
                    result.add_error(
                        "target",
                        f"Airfoil file '{target}' does not exist or is not accessible",
                        "file_error",
                    )
                elif not self._is_valid_airfoil_file(target):
                    result.add_error(
                        "target",
                        f"File '{target}' is not a valid airfoil coordinate file",
                        "format_error",
                    )
            else:
                # Validate airfoil name format
                if not self._is_valid_airfoil_name(target):
                    result.add_warning(
                        f"Airfoil name '{target}' may not be recognized. "
                        "Consider using a standard NACA designation or coordinate file.",
                    )

    def _validate_analysis_parameters(
        self,
        config: AnalysisConfig,
        result: ValidationResult,
    ) -> None:
        """Validate analysis-specific parameters."""
        analysis_type = config.analysis_type
        parameters = config.parameters

        if analysis_type not in self._validation_rules:
            result.add_warning(
                f"No validation rules defined for analysis type: {analysis_type.value}",
            )
            return

        rules = self._validation_rules[analysis_type]

        # Check required parameters
        for param in rules.get("required_params", []):
            if param not in parameters:
                default_value = rules.get("param_defaults", {}).get(param)
                result.add_error(
                    f"parameters.{param}",
                    f"Required parameter '{param}' is missing",
                    "required",
                    suggested_value=default_value,
                )

        # Validate parameter types and ranges
        for param, value in parameters.items():
            self._validate_parameter(param, value, rules, result)

    def _validate_parameter(
        self,
        param: str,
        value: Any,
        rules: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate a single parameter."""
        # Check if parameter is recognized
        all_params = rules.get("required_params", []) + rules.get("optional_params", [])
        if param not in all_params:
            result.add_warning(
                f"Unknown parameter '{param}' - will be passed to solver as-is",
            )
            return

        # Check parameter type
        expected_types = rules.get("param_types", {}).get(param)
        if expected_types and not isinstance(value, expected_types):
            type_names = [
                t.__name__
                for t in (
                    expected_types
                    if isinstance(expected_types, tuple)
                    else (expected_types,)
                )
            ]
            result.add_error(
                f"parameters.{param}",
                f"Parameter '{param}' must be of type {' or '.join(type_names)}, got {type(value).__name__}",
                "type_error",
            )
            return

        # Check parameter range
        param_range = rules.get("param_ranges", {}).get(param)
        if param_range and isinstance(value, (int, float)):
            min_val, max_val = param_range
            if value < min_val or value > max_val:
                result.add_error(
                    f"parameters.{param}",
                    f"Parameter '{param}' value {value} is outside valid range [{min_val}, {max_val}]",
                    "range_error",
                    suggested_value=(min_val + max_val) / 2,
                )

        # Special validations for list parameters
        if isinstance(value, list):
            self._validate_list_parameter(param, value, rules, result)

    def _validate_list_parameter(
        self,
        param: str,
        value: List[Any],
        rules: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate list-type parameters."""
        if not value:
            result.add_error(
                f"parameters.{param}",
                f"Parameter '{param}' cannot be an empty list",
                "value_error",
            )
            return

        # Validate each element in the list
        param_range = rules.get("param_ranges", {}).get(param)
        if param_range:
            min_val, max_val = param_range
            for i, item in enumerate(value):
                if isinstance(item, (int, float)) and (
                    item < min_val or item > max_val
                ):
                    result.add_error(
                        f"parameters.{param}[{i}]",
                        f"List item {item} is outside valid range [{min_val}, {max_val}]",
                        "range_error",
                    )

    def _validate_solver_parameters(
        self,
        config: AnalysisConfig,
        result: ValidationResult,
    ) -> None:
        """Validate solver-specific parameters."""
        solver_type = config.solver_type
        solver_params = config.solver_parameters

        # Define solver-specific validation rules
        solver_rules = {
            SolverType.XFOIL: {
                "max_iter": {"type": int, "range": (1, 1000), "default": 100},
                "ncrit": {"type": (int, float), "range": (1.0, 20.0), "default": 9.0},
                "xtr": {"type": (list, tuple), "length": 2, "default": [0.1, 0.1]},
                "print": {"type": bool, "default": False},
            },
            SolverType.AVL: {
                "mach": {"type": (int, float), "range": (0.0, 0.9), "default": 0.0},
                "iysym": {"type": int, "range": (0, 1), "default": 0},
                "izsym": {"type": int, "range": (0, 1), "default": 0},
            },
        }

        if solver_type not in solver_rules:
            if solver_params:
                result.add_warning(
                    f"No validation rules for {solver_type.value} solver parameters",
                )
            return

        rules = solver_rules[solver_type]

        # Validate each solver parameter
        for param, value in solver_params.items():
            if param not in rules:
                result.add_warning(
                    f"Unknown solver parameter '{param}' for {solver_type.value}",
                )
                continue

            rule = rules[param]

            # Type validation
            expected_type = rule["type"]
            if not isinstance(value, expected_type):
                type_names = [
                    t.__name__
                    for t in (
                        expected_type
                        if isinstance(expected_type, tuple)
                        else (expected_type,)
                    )
                ]
                result.add_error(
                    f"solver_parameters.{param}",
                    f"Solver parameter '{param}' must be of type {' or '.join(type_names)}",
                    "type_error",
                    suggested_value=rule.get("default"),
                )
                continue

            # Range validation
            if "range" in rule and isinstance(value, (int, float)):
                min_val, max_val = rule["range"]
                if value < min_val or value > max_val:
                    result.add_error(
                        f"solver_parameters.{param}",
                        f"Solver parameter '{param}' value {value} is outside valid range [{min_val}, {max_val}]",
                        "range_error",
                        suggested_value=rule.get("default"),
                    )

            # Length validation for lists/tuples
            if "length" in rule and hasattr(value, "__len__"):
                expected_length = rule["length"]
                if len(value) != expected_length:
                    result.add_error(
                        f"solver_parameters.{param}",
                        f"Solver parameter '{param}' must have length {expected_length}, got {len(value)}",
                        "length_error",
                        suggested_value=rule.get("default"),
                    )

    def _validate_parameter_consistency(
        self,
        config: AnalysisConfig,
        result: ValidationResult,
    ) -> None:
        """Validate consistency between parameters."""
        params = config.parameters

        # Angle of attack consistency
        if "min_aoa" in params and "max_aoa" in params:
            if params["min_aoa"] >= params["max_aoa"]:
                result.add_error(
                    "parameters.min_aoa",
                    "Minimum angle of attack must be less than maximum angle of attack",
                    "consistency_error",
                )

        # Reynolds number list validation
        if "reynolds" in params and isinstance(params["reynolds"], list):
            reynolds_list = params["reynolds"]
            if len(set(reynolds_list)) != len(reynolds_list):
                result.add_warning("Duplicate Reynolds numbers detected in list")

            if not all(isinstance(re, (int, float)) for re in reynolds_list):
                result.add_error(
                    "parameters.reynolds",
                    "All Reynolds numbers in list must be numeric",
                    "type_error",
                )

        # Solver compatibility checks
        self._validate_solver_compatibility(config, result)

    def _validate_solver_compatibility(
        self,
        config: AnalysisConfig,
        result: ValidationResult,
    ) -> None:
        """Validate solver compatibility with analysis type and parameters."""
        analysis_type = config.analysis_type
        solver_type = config.solver_type

        # Define compatibility matrix
        compatibility = {
            AnalysisType.AIRFOIL_POLAR: [
                SolverType.XFOIL,
                SolverType.XFLR5,
                SolverType.OPENFOAM,
                SolverType.FOIL2WAKE,
            ],
            AnalysisType.AIRPLANE_POLAR: [
                SolverType.AVL,
                SolverType.GENUVP,
                SolverType.XFLR5,
                SolverType.ICARUS_LSPT,
            ],
            AnalysisType.AIRPLANE_STABILITY: [SolverType.AVL, SolverType.GENUVP],
        }

        if analysis_type in compatibility:
            if solver_type not in compatibility[analysis_type]:
                compatible_solvers = [s.value for s in compatibility[analysis_type]]
                result.add_error(
                    "solver_type",
                    f"Solver '{solver_type.value}' is not compatible with analysis type '{analysis_type.value}'",
                    "compatibility_error",
                    suggested_value=compatible_solvers[0]
                    if compatible_solvers
                    else None,
                )
                result.add_suggestion(
                    f"Compatible solvers: {', '.join(compatible_solvers)}",
                )

    def _is_valid_file_path(self, path: str) -> bool:
        """Check if a file path exists and is accessible."""
        try:
            return Path(path).exists() and Path(path).is_file()
        except (OSError, ValueError):
            return False

    def _looks_like_file_path(self, target: str) -> bool:
        """Check if target looks like a file path."""
        return "/" in target or "\\" in target or "." in target

    def _is_valid_airplane_file(self, path: str) -> bool:
        """Validate airplane definition file format."""
        try:
            # Basic checks for common airplane file formats
            path_obj = Path(path)
            valid_extensions = [".xml", ".json", ".avl", ".dat", ".txt"]
            return path_obj.suffix.lower() in valid_extensions
        except Exception:
            return False

    def _is_valid_airfoil_file(self, path: str) -> bool:
        """Validate airfoil coordinate file format."""
        try:
            path_obj = Path(path)
            valid_extensions = [".dat", ".txt", ".coord", ".csv"]
            if path_obj.suffix.lower() not in valid_extensions:
                return False

            # Try to read first few lines to check format
            with open(path) as f:
                lines = [line.strip() for line in f.readlines()[:10] if line.strip()]
                if len(lines) < 3:
                    return False

                # Check if lines contain coordinate pairs
                for line in lines[1:3]:  # Skip potential header
                    parts = line.split()
                    if len(parts) != 2:
                        continue
                    try:
                        float(parts[0])
                        float(parts[1])
                        return True
                    except ValueError:
                        continue

            return False
        except Exception:
            return False

    def _is_valid_airfoil_name(self, name: str) -> bool:
        """Validate airfoil name format."""
        # NACA 4-digit
        if re.match(r"^NACA\s*\d{4}$", name.upper()):
            return True

        # NACA 5-digit
        if re.match(r"^NACA\s*\d{5}$", name.upper()):
            return True

        # Common airfoil naming patterns
        common_patterns = [
            r"^[A-Z]+\s*\d+",  # Like "CLARK Y", "EPPLER 387"
            r"^\w+[-_]\w+",  # Like "S1223", "FX-63-137"
        ]

        return any(re.match(pattern, name.upper()) for pattern in common_patterns)

    def get_parameter_suggestions(self, analysis_type: AnalysisType) -> Dict[str, Any]:
        """Get parameter suggestions for an analysis type."""
        if analysis_type not in self._validation_rules:
            return {}

        rules = self._validation_rules[analysis_type]
        suggestions = {}

        # Add defaults
        suggestions.update(rules.get("param_defaults", {}))

        # Add parameter descriptions
        descriptions = {
            "reynolds": "Reynolds number(s) for the analysis",
            "mach": "Mach number (typically 0.0 for low-speed)",
            "min_aoa": "Minimum angle of attack in degrees",
            "max_aoa": "Maximum angle of attack in degrees",
            "aoa_step": "Angle of attack increment in degrees",
            "ncrit": "Critical amplification factor (9.0 typical)",
            "velocity": "Flight velocity in m/s",
            "altitude": "Flight altitude in meters",
            "beta": "Sideslip angle in degrees",
        }

        for param in rules.get("required_params", []) + rules.get(
            "optional_params",
            [],
        ):
            if param in descriptions:
                suggestions[f"{param}_description"] = descriptions[param]

        return suggestions
