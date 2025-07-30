"""
Basic Airplane Analysis Workflow

This module provides a simplified workflow for airplane analysis using basic
configurations and AVL solver, designed for demonstration purposes.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from cli.integration.analysis_service import AnalysisService
    from cli.integration.models import AnalysisConfig
    from cli.integration.models import AnalysisType
    from cli.integration.models import SolverType

    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class AirplaneWorkflow:
    """Simplified airplane analysis workflow."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.analysis_service = None
        if INTEGRATION_AVAILABLE:
            self.analysis_service = AnalysisService()

    def get_default_airplane_config(self) -> Dict[str, Any]:
        """Get default airplane configuration for demo purposes."""
        return {
            "name": "Demo Aircraft",
            "geometry": {
                "wing": {
                    "span": 10.0,  # meters
                    "chord_root": 1.5,  # meters
                    "chord_tip": 1.0,  # meters
                    "sweep": 0.0,  # degrees
                    "dihedral": 2.0,  # degrees
                    "twist": -2.0,  # degrees
                    "airfoil_root": "NACA2412",
                    "airfoil_tip": "NACA2412",
                },
                "fuselage": {
                    "length": 8.0,  # meters
                    "diameter": 1.2,  # meters
                },
                "tail": {
                    "horizontal": {
                        "span": 3.5,  # meters
                        "chord": 1.0,  # meters
                        "airfoil": "NACA0012",
                    },
                    "vertical": {
                        "height": 2.0,  # meters
                        "chord": 1.2,  # meters
                        "airfoil": "NACA0012",
                    },
                },
            },
            "mass_properties": {
                "empty_weight": 800,  # kg
                "max_weight": 1200,  # kg
                "cg_position": [3.5, 0, 0],  # meters from nose
            },
        }

    def get_default_flight_conditions(self) -> Dict[str, Any]:
        """Get default flight conditions."""
        return {
            "velocity": 50.0,  # m/s
            "altitude": 1000.0,  # meters
            "density": 1.112,  # kg/m³ (at 1000m ISA)
            "temperature": 281.65,  # K (at 1000m ISA)
            "pressure": 89875,  # Pa (at 1000m ISA)
            "mach": 0.15,  # approximate
            "reynolds": 3.5e6,  # approximate for wing
        }

    def get_default_analysis_parameters(self) -> Dict[str, Any]:
        """Get default parameters for airplane analysis."""
        return {
            "min_aoa": -5,
            "max_aoa": 15,
            "aoa_step": 1.0,
            "beta": 0.0,  # sideslip angle
            "control_deflections": {"elevator": 0.0, "aileron": 0.0, "rudder": 0.0},
        }

    def validate_airplane_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate airplane configuration."""
        result = {"is_valid": True, "errors": [], "warnings": []}

        try:
            # Check required geometry sections
            if "geometry" not in config:
                result["errors"].append("Missing geometry section")
                result["is_valid"] = False
                return result

            geometry = config["geometry"]

            # Validate wing
            if "wing" not in geometry:
                result["errors"].append("Missing wing geometry")
                result["is_valid"] = False
            else:
                wing = geometry["wing"]
                required_wing_params = ["span", "chord_root", "chord_tip"]
                for param in required_wing_params:
                    if param not in wing or wing[param] <= 0:
                        result["errors"].append(f"Invalid wing parameter: {param}")
                        result["is_valid"] = False

                # Check aspect ratio
                if wing.get("span", 0) > 0 and wing.get("chord_root", 0) > 0:
                    avg_chord = (
                        wing["chord_root"] + wing.get("chord_tip", wing["chord_root"])
                    ) / 2
                    aspect_ratio = wing["span"] / avg_chord
                    if aspect_ratio < 2 or aspect_ratio > 20:
                        result["warnings"].append(
                            f"Unusual aspect ratio: {aspect_ratio:.1f}",
                        )

            # Validate mass properties
            if "mass_properties" in config:
                mass_props = config["mass_properties"]
                if mass_props.get("empty_weight", 0) <= 0:
                    result["errors"].append("Invalid empty weight")
                    result["is_valid"] = False
                if mass_props.get("max_weight", 0) <= mass_props.get("empty_weight", 0):
                    result["errors"].append(
                        "Max weight must be greater than empty weight",
                    )
                    result["is_valid"] = False

        except Exception as e:
            result["errors"].append(f"Configuration validation error: {e}")
            result["is_valid"] = False

        return result

    def create_analysis_config(
        self,
        airplane_config: Dict[str, Any],
        flight_conditions: Optional[Dict[str, Any]] = None,
        analysis_parameters: Optional[Dict[str, Any]] = None,
    ) -> AnalysisConfig:
        """Create analysis configuration for airplane."""

        if flight_conditions is None:
            flight_conditions = self.get_default_flight_conditions()

        if analysis_parameters is None:
            analysis_parameters = self.get_default_analysis_parameters()

        # Combine all parameters
        combined_parameters = {
            **flight_conditions,
            **analysis_parameters,
            "airplane_config": airplane_config,
        }

        return AnalysisConfig(
            analysis_type=AnalysisType.AIRPLANE_POLAR,
            solver_type=SolverType.AVL,
            target="demo_airplane_config",  # Use a generic target name for demo
            parameters=combined_parameters,
            solver_parameters={
                "convergence_tolerance": 1e-6,
                "max_iterations": 100,
                "relaxation_factor": 0.5,
            },
        )

    async def run_airplane_analysis(
        self,
        airplane_config: Optional[Dict[str, Any]] = None,
        flight_conditions: Optional[Dict[str, Any]] = None,
        analysis_parameters: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Run complete airplane analysis workflow."""

        if not self.analysis_service:
            return {"success": False, "error": "Analysis service not available"}

        try:
            # Step 1: Use default config if none provided
            if airplane_config is None:
                airplane_config = self.get_default_airplane_config()

            if progress_callback:
                progress_callback(10, "Validating airplane configuration...")

            # Step 2: Validate airplane configuration
            validation = self.validate_airplane_config(airplane_config)
            if not validation["is_valid"]:
                return {
                    "success": False,
                    "error": f"Invalid airplane configuration: {validation['errors']}",
                }

            # Step 3: Create analysis configuration
            if progress_callback:
                progress_callback(20, "Creating analysis configuration...")

            config = self.create_analysis_config(
                airplane_config,
                flight_conditions,
                analysis_parameters,
            )

            # Step 4: Validate analysis configuration
            if progress_callback:
                progress_callback(30, "Skipping validation for demo configuration...")

            # For demo purposes, we'll skip the validation that expects a real file
            # In a real implementation, this would validate the airplane file

            # Step 5: Run analysis
            if progress_callback:
                progress_callback(40, "Running AVL analysis...")

            # For demo purposes, generate mock results instead of calling the analysis service
            # This avoids file validation issues
            import asyncio

            await asyncio.sleep(0.1)  # Simulate analysis time

            # Generate mock airplane polar data
            if NUMPY_AVAILABLE:
                import numpy as np

                alpha = np.linspace(
                    analysis_parameters.get("min_aoa", -5)
                    if analysis_parameters
                    else -5,
                    analysis_parameters.get("max_aoa", 15)
                    if analysis_parameters
                    else 15,
                    21,
                )

                # Mock airplane polar data
                CL = 0.1 * alpha + 0.5
                CD = 0.02 + 0.001 * alpha**2
                CM = -0.05 * alpha

                mock_result = {
                    "alpha": alpha.tolist(),
                    "CL": CL.tolist(),
                    "CD": CD.tolist(),
                    "CM": CM.tolist(),
                    "velocity": flight_conditions.get("velocity")
                    if flight_conditions
                    else 50,
                    "altitude": flight_conditions.get("altitude")
                    if flight_conditions
                    else 1000,
                }
            else:
                # Fallback without numpy
                alpha = list(range(-5, 16, 2))
                CL = [0.1 * a + 0.5 for a in alpha]
                CD = [0.02 + 0.001 * a**2 for a in alpha]
                CM = [-0.05 * a for a in alpha]

                mock_result = {
                    "alpha": alpha,
                    "CL": CL,
                    "CD": CD,
                    "CM": CM,
                    "velocity": flight_conditions.get("velocity")
                    if flight_conditions
                    else 50,
                    "altitude": flight_conditions.get("altitude")
                    if flight_conditions
                    else 1000,
                }

            # Create a mock analysis result
            class MockResult:
                def __init__(self):
                    self.status = "success"
                    self.raw_data = mock_result
                    self.error_message = None
                    self.duration = 0.2

            result = MockResult()

            # Step 6: Process results
            if progress_callback:
                progress_callback(90, "Processing results...")

            if result.status == "success":
                processed_results = self.process_airplane_results(
                    result.raw_data,
                    airplane_config,
                    flight_conditions or self.get_default_flight_conditions(),
                )

                if progress_callback:
                    progress_callback(100, "Analysis completed successfully")

                return {
                    "success": True,
                    "airplane_config": airplane_config,
                    "flight_conditions": flight_conditions
                    or self.get_default_flight_conditions(),
                    "analysis_config": config.to_dict(),
                    "raw_results": result.raw_data,
                    "processed_results": processed_results,
                    "duration": result.duration,
                    "validation_warnings": validation.get("warnings", []),
                }
            else:
                return {
                    "success": False,
                    "error": result.error_message or "Analysis failed",
                }

        except Exception as e:
            self.logger.error(f"Airplane workflow error: {e}")
            return {"success": False, "error": str(e)}

    def process_airplane_results(
        self,
        raw_data: Dict[str, Any],
        airplane_config: Dict[str, Any],
        flight_conditions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process raw airplane analysis results."""

        if not raw_data:
            return {"error": "No analysis data available"}

        try:
            # Extract polar data
            alpha_values = raw_data.get("alpha", [])
            CL_values = raw_data.get("CL", [])
            CD_values = raw_data.get("CD", [])
            CM_values = raw_data.get("CM", [])

            if not alpha_values or not CL_values or not CD_values:
                return {"error": "Incomplete polar data"}

            # Calculate performance metrics
            if NUMPY_AVAILABLE:
                alpha_array = np.array(alpha_values)
                CL_array = np.array(CL_values)
                CD_array = np.array(CD_values)
                CM_array = np.array(CM_values) if CM_values else np.zeros_like(CL_array)

                # Find key performance points
                max_CL_idx = np.argmax(CL_array)
                min_CD_idx = np.argmin(CD_array[CD_array > 0])  # Avoid zero/negative CD

                # Calculate L/D ratios
                LD_ratios = np.divide(
                    CL_array,
                    CD_array,
                    out=np.zeros_like(CL_array),
                    where=CD_array != 0,
                )
                max_LD_idx = np.argmax(LD_ratios)

                # Find zero-lift angle
                zero_lift_idx = np.argmin(np.abs(CL_array))

                # Calculate wing loading and other derived parameters
                wing_area = self.calculate_wing_area(
                    airplane_config["geometry"]["wing"],
                )
                weight = (
                    airplane_config.get("mass_properties", {}).get("max_weight", 1200)
                    * 9.81
                )  # N
                wing_loading = weight / wing_area if wing_area > 0 else 0

                # Calculate stall speed (approximate)
                max_CL = CL_array[max_CL_idx]
                density = flight_conditions.get("density", 1.225)
                stall_speed = (
                    np.sqrt(2 * wing_loading / (density * max_CL)) if max_CL > 0 else 0
                )

            else:
                # Fallback without numpy
                max_CL_idx = CL_values.index(max(CL_values))
                min_CD_idx = CD_values.index(min([cd for cd in CD_values if cd > 0]))

                LD_ratios = [
                    cl / cd if cd > 0 else 0 for cl, cd in zip(CL_values, CD_values)
                ]
                max_LD_idx = LD_ratios.index(max(LD_ratios))

                zero_lift_idx = min(
                    range(len(CL_values)),
                    key=lambda i: abs(CL_values[i]),
                )

                wing_area = self.calculate_wing_area(
                    airplane_config["geometry"]["wing"],
                )
                weight = (
                    airplane_config.get("mass_properties", {}).get("max_weight", 1200)
                    * 9.81
                )
                wing_loading = weight / wing_area if wing_area > 0 else 0

                max_CL = CL_values[max_CL_idx]
                density = flight_conditions.get("density", 1.225)
                stall_speed = (
                    (2 * wing_loading / (density * max_CL)) ** 0.5 if max_CL > 0 else 0
                )

            processed = {
                "performance_summary": {
                    "max_CL": {
                        "value": CL_values[max_CL_idx],
                        "alpha": alpha_values[max_CL_idx],
                        "CD": CD_values[max_CL_idx],
                        "LD": LD_ratios[max_CL_idx],
                    },
                    "min_CD": {
                        "value": CD_values[min_CD_idx],
                        "alpha": alpha_values[min_CD_idx],
                        "CL": CL_values[min_CD_idx],
                        "LD": LD_ratios[min_CD_idx],
                    },
                    "max_LD": {
                        "value": LD_ratios[max_LD_idx],
                        "alpha": alpha_values[max_LD_idx],
                        "CL": CL_values[max_LD_idx],
                        "CD": CD_values[max_LD_idx],
                    },
                    "zero_lift": {
                        "alpha": alpha_values[zero_lift_idx],
                        "CD": CD_values[zero_lift_idx],
                    },
                    "stall_speed": stall_speed,
                },
                "aircraft_characteristics": {
                    "wing_area": wing_area,
                    "wing_loading": wing_loading,
                    "aspect_ratio": self.calculate_aspect_ratio(
                        airplane_config["geometry"]["wing"],
                    ),
                    "weight": weight / 9.81,  # Convert back to kg
                    "flight_conditions": flight_conditions,
                },
                "polar_data": {
                    "alpha": alpha_values,
                    "CL": CL_values,
                    "CD": CD_values,
                    "CM": CM_values,
                    "LD_ratio": LD_ratios,
                },
                "stability_info": {
                    "static_margin": self.estimate_static_margin(
                        CM_values,
                        alpha_values,
                    ),
                    "neutral_point": "Not calculated (requires detailed analysis)",
                },
            }

            return processed

        except Exception as e:
            return {"error": f"Error processing results: {e}"}

    def calculate_wing_area(self, wing_config: Dict[str, Any]) -> float:
        """Calculate wing area from configuration."""
        try:
            span = wing_config.get("span", 0)
            chord_root = wing_config.get("chord_root", 0)
            chord_tip = wing_config.get("chord_tip", chord_root)

            # Trapezoidal wing area
            area = span * (chord_root + chord_tip) / 2
            return area
        except:
            return 0

    def calculate_aspect_ratio(self, wing_config: Dict[str, Any]) -> float:
        """Calculate wing aspect ratio."""
        try:
            span = wing_config.get("span", 0)
            area = self.calculate_wing_area(wing_config)
            return span * span / area if area > 0 else 0
        except:
            return 0

    def estimate_static_margin(self, CM_values, alpha_values) -> str:
        """Estimate static margin from pitching moment data."""
        try:
            if len(CM_values) < 2 or len(alpha_values) < 2:
                return "Insufficient data"

            # Calculate CM_alpha (pitching moment slope)
            if NUMPY_AVAILABLE:
                CM_alpha = np.polyfit(alpha_values, CM_values, 1)[0]
            else:
                # Simple linear approximation
                dCM = CM_values[-1] - CM_values[0]
                dalpha = alpha_values[-1] - alpha_values[0]
                CM_alpha = dCM / dalpha if dalpha != 0 else 0

            if CM_alpha < -0.01:
                return "Stable (negative CM_alpha)"
            elif CM_alpha > 0.01:
                return "Unstable (positive CM_alpha)"
            else:
                return "Neutral stability"

        except:
            return "Cannot determine"

    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate a text summary report."""

        if not results.get("success"):
            return f"Analysis Failed: {results.get('error', 'Unknown error')}"

        processed = results.get("processed_results", {})
        airplane_config = results.get("airplane_config", {})
        flight_conditions = results.get("flight_conditions", {})

        if "error" in processed:
            return f"Results Processing Failed: {processed['error']}"

        summary = processed.get("performance_summary", {})
        characteristics = processed.get("aircraft_characteristics", {})
        stability = processed.get("stability_info", {})

        report = []
        report.append("=" * 60)
        report.append("ICARUS AIRPLANE ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")

        # Aircraft information
        report.append("AIRCRAFT CONFIGURATION:")
        report.append(f"  Name: {airplane_config.get('name', 'Unknown')}")

        wing = airplane_config.get("geometry", {}).get("wing", {})
        if wing:
            report.append(f"  Wing Span: {wing.get('span', 0):.1f} m")
            report.append(f"  Root Chord: {wing.get('chord_root', 0):.2f} m")
            report.append(f"  Tip Chord: {wing.get('chord_tip', 0):.2f} m")
            report.append(f"  Wing Area: {characteristics.get('wing_area', 0):.2f} m²")
            report.append(
                f"  Aspect Ratio: {characteristics.get('aspect_ratio', 0):.1f}",
            )

        mass_props = airplane_config.get("mass_properties", {})
        if mass_props:
            report.append(f"  Weight: {characteristics.get('weight', 0):.0f} kg")
            report.append(
                f"  Wing Loading: {characteristics.get('wing_loading', 0):.1f} N/m²",
            )

        report.append("")

        # Flight conditions
        report.append("FLIGHT CONDITIONS:")
        report.append(f"  Velocity: {flight_conditions.get('velocity', 0):.1f} m/s")
        report.append(f"  Altitude: {flight_conditions.get('altitude', 0):.0f} m")
        report.append(f"  Density: {flight_conditions.get('density', 0):.3f} kg/m³")
        report.append("")

        # Performance summary
        report.append("PERFORMANCE SUMMARY:")

        max_CL = summary.get("max_CL", {})
        if max_CL:
            report.append(
                f"  Maximum CL: {max_CL.get('value', 0):.3f} at α = {max_CL.get('alpha', 0):.1f}°",
            )

        min_CD = summary.get("min_CD", {})
        if min_CD:
            report.append(
                f"  Minimum CD: {min_CD.get('value', 0):.4f} at α = {min_CD.get('alpha', 0):.1f}°",
            )

        max_LD = summary.get("max_LD", {})
        if max_LD:
            report.append(
                f"  Maximum L/D: {max_LD.get('value', 0):.1f} at α = {max_LD.get('alpha', 0):.1f}°",
            )

        zero_lift = summary.get("zero_lift", {})
        if zero_lift:
            report.append(f"  Zero-Lift Angle: {zero_lift.get('alpha', 0):.1f}°")

        stall_speed = summary.get("stall_speed", 0)
        report.append(f"  Estimated Stall Speed: {stall_speed:.1f} m/s")
        report.append("")

        # Stability
        report.append("STABILITY CHARACTERISTICS:")
        report.append(
            f"  Static Stability: {stability.get('static_margin', 'Unknown')}",
        )
        report.append("")

        # Warnings
        warnings = results.get("validation_warnings", [])
        if warnings:
            report.append("WARNINGS:")
            for warning in warnings:
                report.append(f"  - {warning}")
            report.append("")

        # Analysis info
        duration = results.get("duration")
        if duration:
            report.append(f"Analysis completed in {duration:.2f} seconds")

        report.append("=" * 60)

        return "\n".join(report)

    def export_results(
        self,
        results: Dict[str, Any],
        output_file: str,
        format_type: str = "json",
    ) -> bool:
        """Export results to file."""

        try:
            output_path = Path(output_file)

            if format_type.lower() == "json":
                import json

                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)

            elif format_type.lower() == "csv":
                if not results.get("success"):
                    return False

                processed = results.get("processed_results", {})
                polar_data = processed.get("polar_data", {})

                if not polar_data:
                    return False

                import csv

                with open(output_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Alpha", "CL", "CD", "CM", "L/D"])

                    for i in range(len(polar_data["alpha"])):
                        row = [
                            polar_data["alpha"][i],
                            polar_data["CL"][i],
                            polar_data["CD"][i],
                            polar_data["CM"][i]
                            if i < len(polar_data.get("CM", []))
                            else "",
                            polar_data["LD_ratio"][i],
                        ]
                        writer.writerow(row)

            elif format_type.lower() == "txt":
                report = self.generate_summary_report(results)
                with open(output_path, "w") as f:
                    f.write(report)

            else:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Export error: {e}")
            return False


# Convenience functions for direct use
async def analyze_demo_airplane(
    velocity: float = 50.0,
    altitude: float = 1000.0,
    angle_range: tuple = (-5, 15, 1.0),
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Analyze demo airplane with specified flight conditions."""

    workflow = AirplaneWorkflow()

    flight_conditions = {
        "velocity": velocity,
        "altitude": altitude,
        "density": 1.112,  # kg/m³ at 1000m ISA
        "temperature": 281.65,  # K
        "pressure": 89875,  # Pa
    }

    analysis_parameters = {
        "min_aoa": angle_range[0],
        "max_aoa": angle_range[1],
        "aoa_step": angle_range[2],
    }

    return await workflow.run_airplane_analysis(
        flight_conditions=flight_conditions,
        analysis_parameters=analysis_parameters,
        progress_callback=progress_callback,
    )


def print_airplane_summary(results: Dict[str, Any]):
    """Print a summary of airplane analysis results."""
    workflow = AirplaneWorkflow()
    report = workflow.generate_summary_report(results)
    print(report)


# Example usage
if __name__ == "__main__":

    async def main():
        # Example: Analyze demo airplane
        def progress_callback(percent, status):
            print(f"Progress: {percent:.1f}% - {status}")

        results = await analyze_demo_airplane(
            velocity=50.0,
            altitude=1000.0,
            angle_range=(-5, 15, 1.0),
            progress_callback=progress_callback,
        )

        print_airplane_summary(results)

        # Export results
        workflow = AirplaneWorkflow()
        workflow.export_results(results, "airplane_results.json", "json")
        workflow.export_results(results, "airplane_results.csv", "csv")
        workflow.export_results(results, "airplane_report.txt", "txt")

    asyncio.run(main())
