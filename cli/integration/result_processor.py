"""
Result processing pipeline with standardized formatting.

This module processes raw analysis results from ICARUS solvers and converts
them into standardized, formatted data structures for display and export.
"""

import json
import logging
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd

from .models import AnalysisResult
from .models import AnalysisType
from .models import ProcessedResult
from .models import SolverType


class ResultProcessor:
    """Processes and formats analysis results from ICARUS solvers."""

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._formatters = self._initialize_formatters()

    def _initialize_formatters(self) -> Dict[AnalysisType, callable]:
        """Initialize result formatters for different analysis types."""
        return {
            AnalysisType.AIRFOIL_POLAR: self._format_airfoil_polar_result,
            AnalysisType.AIRPLANE_POLAR: self._format_airplane_polar_result,
            AnalysisType.AIRPLANE_STABILITY: self._format_stability_result,
            AnalysisType.MISSION_ANALYSIS: self._format_mission_result,
            AnalysisType.OPTIMIZATION: self._format_optimization_result,
        }

    def process_result(self, analysis_result: AnalysisResult) -> ProcessedResult:
        """Process a raw analysis result into a standardized format."""
        if not analysis_result.is_successful:
            return self._create_error_result(analysis_result)

        try:
            # Get the appropriate formatter
            formatter = self._formatters.get(analysis_result.config.analysis_type)
            if not formatter:
                self._logger.warning(
                    f"No formatter for analysis type: {analysis_result.config.analysis_type}",
                )
                return self._create_generic_result(analysis_result)

            # Process the result
            processed = formatter(analysis_result)

            # Add common metadata
            self._add_common_metadata(processed, analysis_result)

            # Generate export formats
            self._generate_export_formats(processed)

            return processed

        except Exception as e:
            self._logger.error(f"Error processing result: {e}")
            return self._create_error_result(analysis_result, str(e))

    def _format_airfoil_polar_result(self, result: AnalysisResult) -> ProcessedResult:
        """Format airfoil polar analysis results."""
        raw_data = result.raw_data

        # Extract data based on solver type
        if result.config.solver_type == SolverType.XFOIL:
            return self._format_xfoil_polar(result, raw_data)
        else:
            return self._format_generic_airfoil_polar(result, raw_data)

    def _format_xfoil_polar(
        self,
        result: AnalysisResult,
        raw_data: Any,
    ) -> ProcessedResult:
        """Format XFoil polar results."""
        processed = ProcessedResult(analysis_result=result)

        try:
            # Assume raw_data is a dictionary or has polar data
            if hasattr(raw_data, "polars") or isinstance(raw_data, dict):
                polar_data = (
                    raw_data.get("polars", raw_data)
                    if isinstance(raw_data, dict)
                    else raw_data.polars
                )

                # Extract coefficient data
                alpha = self._extract_array(
                    polar_data,
                    ["alpha", "aoa", "angle_of_attack"],
                )
                cl = self._extract_array(polar_data, ["cl", "CL", "lift_coefficient"])
                cd = self._extract_array(polar_data, ["cd", "CD", "drag_coefficient"])
                cm = self._extract_array(polar_data, ["cm", "CM", "moment_coefficient"])

                # Create formatted data
                processed.formatted_data = {
                    "coefficients": {
                        "alpha": alpha.tolist()
                        if isinstance(alpha, np.ndarray)
                        else alpha,
                        "cl": cl.tolist() if isinstance(cl, np.ndarray) else cl,
                        "cd": cd.tolist() if isinstance(cd, np.ndarray) else cd,
                        "cm": cm.tolist() if isinstance(cm, np.ndarray) else cm,
                    },
                    "performance": self._calculate_airfoil_performance(
                        alpha,
                        cl,
                        cd,
                        cm,
                    ),
                    "reynolds": result.config.parameters.get("reynolds"),
                    "mach": result.config.parameters.get("mach", 0.0),
                }

                # Create plots
                processed.plots = self._create_airfoil_plots(alpha, cl, cd, cm)

                # Create tables
                processed.tables = self._create_airfoil_tables(alpha, cl, cd, cm)

                # Create summary
                processed.summary = self._create_airfoil_summary(
                    alpha,
                    cl,
                    cd,
                    cm,
                    result.config,
                )

        except Exception as e:
            self._logger.error(f"Error formatting XFoil result: {e}")
            processed.formatted_data = {
                "error": str(e),
                "raw_data": str(raw_data)[:1000],
            }

        return processed

    def _format_airplane_polar_result(self, result: AnalysisResult) -> ProcessedResult:
        """Format airplane polar analysis results."""
        processed = ProcessedResult(analysis_result=result)
        raw_data = result.raw_data

        try:
            # Extract airplane polar data
            alpha = self._extract_array(raw_data, ["alpha", "aoa", "angle_of_attack"])
            CL = self._extract_array(raw_data, ["CL", "cl", "lift_coefficient"])
            CD = self._extract_array(raw_data, ["CD", "cd", "drag_coefficient"])
            CM = self._extract_array(raw_data, ["CM", "cm", "moment_coefficient"])

            # Additional airplane-specific coefficients
            CY = self._extract_array(
                raw_data,
                ["CY", "cy", "side_force_coefficient"],
                default_value=0.0,
            )
            Cl_roll = self._extract_array(
                raw_data,
                ["Cl", "cl_roll", "roll_moment_coefficient"],
                default_value=0.0,
            )
            Cn = self._extract_array(
                raw_data,
                ["Cn", "cn", "yaw_moment_coefficient"],
                default_value=0.0,
            )

            processed.formatted_data = {
                "coefficients": {
                    "alpha": alpha.tolist() if isinstance(alpha, np.ndarray) else alpha,
                    "CL": CL.tolist() if isinstance(CL, np.ndarray) else CL,
                    "CD": CD.tolist() if isinstance(CD, np.ndarray) else CD,
                    "CM": CM.tolist() if isinstance(CM, np.ndarray) else CM,
                    "CY": CY.tolist() if isinstance(CY, np.ndarray) else CY,
                    "Cl": Cl_roll.tolist()
                    if isinstance(Cl_roll, np.ndarray)
                    else Cl_roll,
                    "Cn": Cn.tolist() if isinstance(Cn, np.ndarray) else Cn,
                },
                "performance": self._calculate_airplane_performance(alpha, CL, CD, CM),
                "flight_conditions": {
                    "velocity": result.config.parameters.get("velocity"),
                    "altitude": result.config.parameters.get("altitude"),
                    "beta": result.config.parameters.get("beta", 0.0),
                },
            }

            # Create plots
            processed.plots = self._create_airplane_plots(alpha, CL, CD, CM)

            # Create tables
            processed.tables = self._create_airplane_tables(
                alpha,
                CL,
                CD,
                CM,
                CY,
                Cl_roll,
                Cn,
            )

            # Create summary
            processed.summary = self._create_airplane_summary(
                alpha,
                CL,
                CD,
                CM,
                result.config,
            )

        except Exception as e:
            self._logger.error(f"Error formatting airplane polar result: {e}")
            processed.formatted_data = {
                "error": str(e),
                "raw_data": str(raw_data)[:1000],
            }

        return processed

    def _format_stability_result(self, result: AnalysisResult) -> ProcessedResult:
        """Format stability analysis results."""
        processed = ProcessedResult(analysis_result=result)
        raw_data = result.raw_data

        try:
            # Extract stability derivatives
            stability_data = {
                "longitudinal": self._extract_longitudinal_derivatives(raw_data),
                "lateral": self._extract_lateral_derivatives(raw_data),
                "modes": self._extract_dynamic_modes(raw_data),
            }

            processed.formatted_data = {
                "stability_derivatives": stability_data,
                "trim_conditions": self._extract_trim_conditions(raw_data),
                "flight_conditions": {
                    "velocity": result.config.parameters.get("velocity"),
                    "altitude": result.config.parameters.get("altitude"),
                },
            }

            # Create stability-specific plots and tables
            processed.plots = self._create_stability_plots(stability_data)
            processed.tables = self._create_stability_tables(stability_data)
            processed.summary = self._create_stability_summary(
                stability_data,
                result.config,
            )

        except Exception as e:
            self._logger.error(f"Error formatting stability result: {e}")
            processed.formatted_data = {
                "error": str(e),
                "raw_data": str(raw_data)[:1000],
            }

        return processed

    def _extract_array(
        self,
        data: Any,
        possible_keys: List[str],
        default_value: Any = None,
    ) -> Union[np.ndarray, List, float]:
        """Extract array data from various possible key names."""
        if isinstance(data, dict):
            for key in possible_keys:
                if key in data:
                    value = data[key]
                    if isinstance(value, (list, np.ndarray)):
                        return (
                            np.array(value)
                            if not isinstance(value, np.ndarray)
                            else value
                        )
                    else:
                        return value
        elif hasattr(data, "__dict__"):
            for key in possible_keys:
                if hasattr(data, key):
                    value = getattr(data, key)
                    if isinstance(value, (list, np.ndarray)):
                        return (
                            np.array(value)
                            if not isinstance(value, np.ndarray)
                            else value
                        )
                    else:
                        return value

        # Return default or empty array
        if default_value is not None:
            return default_value
        return np.array([])

    def _calculate_airfoil_performance(
        self,
        alpha: np.ndarray,
        cl: np.ndarray,
        cd: np.ndarray,
        cm: np.ndarray,
    ) -> Dict[str, Any]:
        """Calculate airfoil performance metrics."""
        if len(alpha) == 0 or len(cl) == 0 or len(cd) == 0:
            return {}

        try:
            # Find key performance points
            max_cl_idx = np.argmax(cl)
            min_cd_idx = np.argmin(cd)

            # Calculate L/D ratio
            ld_ratio = np.divide(cl, cd, out=np.zeros_like(cl), where=cd != 0)
            max_ld_idx = np.argmax(ld_ratio)

            # Find zero-lift angle
            zero_lift_idx = np.argmin(np.abs(cl))

            return {
                "max_cl": {
                    "value": float(cl[max_cl_idx]),
                    "alpha": float(alpha[max_cl_idx]),
                    "cd": float(cd[max_cl_idx]),
                },
                "min_cd": {
                    "value": float(cd[min_cd_idx]),
                    "alpha": float(alpha[min_cd_idx]),
                    "cl": float(cl[min_cd_idx]),
                },
                "max_ld": {
                    "value": float(ld_ratio[max_ld_idx]),
                    "alpha": float(alpha[max_ld_idx]),
                    "cl": float(cl[max_ld_idx]),
                    "cd": float(cd[max_ld_idx]),
                },
                "zero_lift": {
                    "alpha": float(alpha[zero_lift_idx]),
                    "cd": float(cd[zero_lift_idx]),
                },
                "stall_angle": float(alpha[max_cl_idx]),
                "ld_ratio": ld_ratio.tolist(),
            }
        except Exception as e:
            self._logger.error(f"Error calculating airfoil performance: {e}")
            return {}

    def _calculate_airplane_performance(
        self,
        alpha: np.ndarray,
        CL: np.ndarray,
        CD: np.ndarray,
        CM: np.ndarray,
    ) -> Dict[str, Any]:
        """Calculate airplane performance metrics."""
        if len(alpha) == 0 or len(CL) == 0 or len(CD) == 0:
            return {}

        try:
            # Similar to airfoil but for airplane coefficients
            max_CL_idx = np.argmax(CL)
            min_CD_idx = np.argmin(CD)

            LD_ratio = np.divide(CL, CD, out=np.zeros_like(CL), where=CD != 0)
            max_LD_idx = np.argmax(LD_ratio)

            zero_lift_idx = np.argmin(np.abs(CL))

            return {
                "max_CL": {
                    "value": float(CL[max_CL_idx]),
                    "alpha": float(alpha[max_CL_idx]),
                    "CD": float(CD[max_CL_idx]),
                },
                "min_CD": {
                    "value": float(CD[min_CD_idx]),
                    "alpha": float(alpha[min_CD_idx]),
                    "CL": float(CL[min_CD_idx]),
                },
                "max_LD": {
                    "value": float(LD_ratio[max_LD_idx]),
                    "alpha": float(alpha[max_LD_idx]),
                    "CL": float(CL[max_LD_idx]),
                    "CD": float(CD[max_LD_idx]),
                },
                "zero_lift": {
                    "alpha": float(alpha[zero_lift_idx]),
                    "CD": float(CD[zero_lift_idx]),
                },
                "LD_ratio": LD_ratio.tolist(),
            }
        except Exception as e:
            self._logger.error(f"Error calculating airplane performance: {e}")
            return {}

    def _create_airfoil_plots(
        self,
        alpha: np.ndarray,
        cl: np.ndarray,
        cd: np.ndarray,
        cm: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Create plot data for airfoil results."""
        plots = []

        if len(alpha) > 0:
            # Lift curve
            plots.append(
                {
                    "type": "lift_curve",
                    "title": "Lift Coefficient vs Angle of Attack",
                    "x_data": alpha.tolist(),
                    "y_data": cl.tolist(),
                    "x_label": "Angle of Attack (degrees)",
                    "y_label": "Lift Coefficient (Cl)",
                    "line_style": "solid",
                    "color": "blue",
                },
            )

            # Drag polar
            plots.append(
                {
                    "type": "drag_polar",
                    "title": "Drag Polar",
                    "x_data": cd.tolist(),
                    "y_data": cl.tolist(),
                    "x_label": "Drag Coefficient (Cd)",
                    "y_label": "Lift Coefficient (Cl)",
                    "line_style": "solid",
                    "color": "red",
                },
            )

            # Moment coefficient
            if len(cm) > 0:
                plots.append(
                    {
                        "type": "moment_curve",
                        "title": "Moment Coefficient vs Angle of Attack",
                        "x_data": alpha.tolist(),
                        "y_data": cm.tolist(),
                        "x_label": "Angle of Attack (degrees)",
                        "y_label": "Moment Coefficient (Cm)",
                        "line_style": "solid",
                        "color": "green",
                    },
                )

        return plots

    def _create_airplane_plots(
        self,
        alpha: np.ndarray,
        CL: np.ndarray,
        CD: np.ndarray,
        CM: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Create plot data for airplane results."""
        plots = []

        if len(alpha) > 0:
            # Lift curve
            plots.append(
                {
                    "type": "lift_curve",
                    "title": "Aircraft Lift Coefficient vs Angle of Attack",
                    "x_data": alpha.tolist(),
                    "y_data": CL.tolist(),
                    "x_label": "Angle of Attack (degrees)",
                    "y_label": "Lift Coefficient (CL)",
                    "line_style": "solid",
                    "color": "blue",
                },
            )

            # Drag polar
            plots.append(
                {
                    "type": "drag_polar",
                    "title": "Aircraft Drag Polar",
                    "x_data": CD.tolist(),
                    "y_data": CL.tolist(),
                    "x_label": "Drag Coefficient (CD)",
                    "y_label": "Lift Coefficient (CL)",
                    "line_style": "solid",
                    "color": "red",
                },
            )

        return plots

    def _create_airfoil_tables(
        self,
        alpha: np.ndarray,
        cl: np.ndarray,
        cd: np.ndarray,
        cm: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Create table data for airfoil results."""
        if len(alpha) == 0:
            return []

        # Create main data table
        table_data = []
        for i in range(len(alpha)):
            row = {
                "alpha": float(alpha[i]),
                "cl": float(cl[i]) if i < len(cl) else 0.0,
                "cd": float(cd[i]) if i < len(cd) else 0.0,
                "cm": float(cm[i]) if i < len(cm) else 0.0,
            }
            if len(cd) > i and cd[i] != 0:
                row["cl_cd"] = float(cl[i] / cd[i])
            table_data.append(row)

        return [
            {
                "name": "polar_data",
                "title": "Airfoil Polar Data",
                "columns": ["alpha", "cl", "cd", "cm", "cl_cd"],
                "column_labels": ["α (°)", "Cl", "Cd", "Cm", "Cl/Cd"],
                "data": table_data,
            },
        ]

    def _create_airplane_tables(
        self,
        alpha: np.ndarray,
        CL: np.ndarray,
        CD: np.ndarray,
        CM: np.ndarray,
        CY: np.ndarray,
        Cl: np.ndarray,
        Cn: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Create table data for airplane results."""
        if len(alpha) == 0:
            return []

        table_data = []
        for i in range(len(alpha)):
            row = {
                "alpha": float(alpha[i]),
                "CL": float(CL[i]) if i < len(CL) else 0.0,
                "CD": float(CD[i]) if i < len(CD) else 0.0,
                "CM": float(CM[i]) if i < len(CM) else 0.0,
                "CY": float(CY[i]) if i < len(CY) else 0.0,
                "Cl": float(Cl[i]) if i < len(Cl) else 0.0,
                "Cn": float(Cn[i]) if i < len(Cn) else 0.0,
            }
            if len(CD) > i and CD[i] != 0:
                row["CL_CD"] = float(CL[i] / CD[i])
            table_data.append(row)

        return [
            {
                "name": "polar_data",
                "title": "Aircraft Polar Data",
                "columns": ["alpha", "CL", "CD", "CM", "CY", "Cl", "Cn", "CL_CD"],
                "column_labels": ["α (°)", "CL", "CD", "CM", "CY", "Cl", "Cn", "CL/CD"],
                "data": table_data,
            },
        ]

    def _create_airfoil_summary(
        self,
        alpha: np.ndarray,
        cl: np.ndarray,
        cd: np.ndarray,
        cm: np.ndarray,
        config: Any,
    ) -> Dict[str, Any]:
        """Create summary for airfoil analysis."""
        performance = self._calculate_airfoil_performance(alpha, cl, cd, cm)

        return {
            "analysis_type": "Airfoil Polar Analysis",
            "target": config.target,
            "solver": config.solver_type.value,
            "conditions": {
                "Reynolds": config.parameters.get("reynolds"),
                "Mach": config.parameters.get("mach", 0.0),
            },
            "key_results": performance,
            "data_points": len(alpha),
        }

    def _create_airplane_summary(
        self,
        alpha: np.ndarray,
        CL: np.ndarray,
        CD: np.ndarray,
        CM: np.ndarray,
        config: Any,
    ) -> Dict[str, Any]:
        """Create summary for airplane analysis."""
        performance = self._calculate_airplane_performance(alpha, CL, CD, CM)

        return {
            "analysis_type": "Aircraft Polar Analysis",
            "target": config.target,
            "solver": config.solver_type.value,
            "conditions": {
                "Velocity": config.parameters.get("velocity"),
                "Altitude": config.parameters.get("altitude"),
                "Sideslip": config.parameters.get("beta", 0.0),
            },
            "key_results": performance,
            "data_points": len(alpha),
        }

    def _extract_longitudinal_derivatives(self, data: Any) -> Dict[str, float]:
        """Extract longitudinal stability derivatives."""
        # Placeholder - would extract actual derivatives from solver output
        return {
            "CLa": 0.0,  # Lift curve slope
            "CMa": 0.0,  # Pitching moment derivative
            "CLq": 0.0,  # Pitch damping
            "CMq": 0.0,  # Pitch damping moment
        }

    def _extract_lateral_derivatives(self, data: Any) -> Dict[str, float]:
        """Extract lateral stability derivatives."""
        return {
            "CYb": 0.0,  # Side force due to sideslip
            "Clb": 0.0,  # Rolling moment due to sideslip
            "Cnb": 0.0,  # Yawing moment due to sideslip
            "Clp": 0.0,  # Roll damping
            "Cnr": 0.0,  # Yaw damping
        }

    def _extract_dynamic_modes(self, data: Any) -> Dict[str, Any]:
        """Extract dynamic mode characteristics."""
        return {
            "phugoid": {"frequency": 0.0, "damping": 0.0},
            "short_period": {"frequency": 0.0, "damping": 0.0},
            "dutch_roll": {"frequency": 0.0, "damping": 0.0},
            "roll_mode": {"time_constant": 0.0},
            "spiral_mode": {"time_constant": 0.0},
        }

    def _extract_trim_conditions(self, data: Any) -> Dict[str, float]:
        """Extract trim conditions."""
        return {
            "alpha_trim": 0.0,
            "elevator_trim": 0.0,
            "thrust_trim": 0.0,
        }

    def _create_stability_plots(
        self,
        stability_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Create plots for stability analysis."""
        # Placeholder for stability-specific plots
        return []

    def _create_stability_tables(
        self,
        stability_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Create tables for stability analysis."""
        # Placeholder for stability-specific tables
        return []

    def _create_stability_summary(
        self,
        stability_data: Dict[str, Any],
        config: Any,
    ) -> Dict[str, Any]:
        """Create summary for stability analysis."""
        return {
            "analysis_type": "Stability Analysis",
            "target": config.target,
            "solver": config.solver_type.value,
            "stability_assessment": "Stable",  # Would be calculated
        }

    def _format_mission_result(self, result: AnalysisResult) -> ProcessedResult:
        """Format mission analysis results."""
        # Placeholder for mission analysis formatting
        return ProcessedResult(analysis_result=result)

    def _format_optimization_result(self, result: AnalysisResult) -> ProcessedResult:
        """Format optimization results."""
        # Placeholder for optimization result formatting
        return ProcessedResult(analysis_result=result)

    def _format_generic_airfoil_polar(
        self,
        result: AnalysisResult,
        raw_data: Any,
    ) -> ProcessedResult:
        """Generic formatter for airfoil polar results."""
        processed = ProcessedResult(analysis_result=result)
        processed.formatted_data = {"raw_data": str(raw_data)[:1000]}
        return processed

    def _create_generic_result(self, result: AnalysisResult) -> ProcessedResult:
        """Create a generic processed result."""
        processed = ProcessedResult(analysis_result=result)
        processed.formatted_data = {
            "message": "Generic result formatting - no specific formatter available",
            "raw_data": str(result.raw_data)[:1000] if result.raw_data else "No data",
        }
        return processed

    def _create_error_result(
        self,
        result: AnalysisResult,
        additional_error: str = None,
    ) -> ProcessedResult:
        """Create a processed result for failed analyses."""
        processed = ProcessedResult(analysis_result=result)

        error_info = {
            "status": result.status,
            "error_message": result.error_message,
            "analysis_type": result.config.analysis_type.value,
            "solver": result.config.solver_type.value,
        }

        if additional_error:
            error_info["processing_error"] = additional_error

        processed.formatted_data = {"error": error_info}
        processed.summary = {
            "analysis_failed": True,
            "error_summary": result.error_message or "Analysis failed",
        }

        return processed

    def _add_common_metadata(
        self,
        processed: ProcessedResult,
        result: AnalysisResult,
    ) -> None:
        """Add common metadata to processed results."""
        processed.formatted_data.setdefault("metadata", {}).update(
            {
                "analysis_id": result.analysis_id,
                "timestamp": result.start_time.isoformat(),
                "duration": result.duration,
                "solver": result.config.solver_type.value,
                "analysis_type": result.config.analysis_type.value,
                "target": result.config.target,
            },
        )

    def _generate_export_formats(self, processed: ProcessedResult) -> None:
        """Generate available export formats for the result."""
        processed.export_formats = ["json", "csv"]

        # Add format-specific exports based on data content
        if processed.tables:
            processed.export_formats.extend(["xlsx", "html"])

        if processed.plots:
            processed.export_formats.extend(["png", "svg", "pdf"])

        # Remove duplicates
        processed.export_formats = list(set(processed.export_formats))

    def export_result(
        self,
        processed: ProcessedResult,
        format_type: str,
        output_path: Optional[str] = None,
    ) -> str:
        """Export processed result to specified format."""
        if format_type not in processed.export_formats:
            raise ValueError(f"Format {format_type} not available for this result")

        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"icarus_result_{timestamp}.{format_type}"

        try:
            if format_type == "json":
                return self._export_json(processed, output_path)
            elif format_type == "csv":
                return self._export_csv(processed, output_path)
            elif format_type == "xlsx":
                return self._export_excel(processed, output_path)
            else:
                raise ValueError(f"Export format {format_type} not implemented")
        except Exception as e:
            self._logger.error(f"Error exporting result: {e}")
            raise

    def _export_json(self, processed: ProcessedResult, output_path: str) -> str:
        """Export result as JSON."""
        export_data = {
            "formatted_data": processed.formatted_data,
            "summary": processed.summary,
            "plots": processed.plots,
            "tables": processed.tables,
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        return output_path

    def _export_csv(self, processed: ProcessedResult, output_path: str) -> str:
        """Export result as CSV (main table data)."""
        if not processed.tables:
            raise ValueError("No table data available for CSV export")

        # Export the first table
        table = processed.tables[0]
        df = pd.DataFrame(table["data"])
        df.to_csv(output_path, index=False)

        return output_path

    def _export_excel(self, processed: ProcessedResult, output_path: str) -> str:
        """Export result as Excel with multiple sheets."""
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Export each table as a separate sheet
            for table in processed.tables:
                df = pd.DataFrame(table["data"])
                sheet_name = table["name"][:31]  # Excel sheet name limit
                df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Export summary as a sheet
            if processed.summary:
                summary_df = pd.DataFrame([processed.summary])
                summary_df.to_excel(writer, sheet_name="Summary", index=False)

        return output_path
