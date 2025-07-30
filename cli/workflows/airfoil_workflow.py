"""
Basic Airfoil Analysis Workflow

This module provides a simplified workflow for airfoil analysis using NACA airfoils
and XFoil solver, designed for demonstration purposes.
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


class AirfoilWorkflow:
    """Simplified airfoil analysis workflow."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.analysis_service = None
        if INTEGRATION_AVAILABLE:
            self.analysis_service = AnalysisService()

    def validate_naca_airfoil(self, naca_code: str) -> Dict[str, Any]:
        """Validate NACA airfoil code."""
        result = {
            "is_valid": False,
            "airfoil_type": None,
            "parameters": {},
            "errors": [],
        }

        try:
            # Remove NACA prefix if present
            code = naca_code.upper().replace("NACA", "").strip()

            if len(code) == 4:
                # 4-digit NACA airfoil
                result["airfoil_type"] = "4-digit"
                result["parameters"] = {
                    "max_camber": int(code[0]) / 100.0,
                    "camber_position": int(code[1]) / 10.0,
                    "thickness": int(code[2:4]) / 100.0,
                }
                result["is_valid"] = True

            elif len(code) == 5:
                # 5-digit NACA airfoil
                result["airfoil_type"] = "5-digit"
                result["parameters"] = {
                    "design_cl": int(code[0]) * 0.15 / 10.0,
                    "camber_position": int(code[1]) / 20.0,
                    "reflex": code[2] == "1",
                    "thickness": int(code[3:5]) / 100.0,
                }
                result["is_valid"] = True

            else:
                result["errors"].append(f"Invalid NACA code length: {len(code)}")

        except ValueError as e:
            result["errors"].append(f"Invalid NACA code format: {e}")

        return result

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for airfoil analysis."""
        return {
            "reynolds": 1000000,
            "mach": 0.0,
            "min_aoa": -10,
            "max_aoa": 15,
            "aoa_step": 0.5,
            "n_crit": 9,
            "max_iterations": 100,
            "convergence_tolerance": 1e-6,
        }

    def create_analysis_config(
        self,
        airfoil: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> AnalysisConfig:
        """Create analysis configuration for airfoil."""
        if parameters is None:
            parameters = self.get_default_parameters()
        else:
            # Ensure all required parameters are present
            defaults = self.get_default_parameters()
            for key, value in defaults.items():
                if key not in parameters:
                    parameters[key] = value

        return AnalysisConfig(
            analysis_type=AnalysisType.AIRFOIL_POLAR,
            solver_type=SolverType.XFOIL,
            target=airfoil,
            parameters=parameters,
            solver_parameters={
                "n_crit": parameters.get("n_crit", 9),
                "max_iterations": parameters.get("max_iterations", 100),
                "convergence_tolerance": parameters.get("convergence_tolerance", 1e-6),
            },
        )

    async def run_airfoil_analysis(
        self,
        airfoil: str,
        parameters: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Run complete airfoil analysis workflow."""

        if not self.analysis_service:
            return {"success": False, "error": "Analysis service not available"}

        try:
            # Step 1: Validate airfoil
            if progress_callback:
                progress_callback(10, "Validating airfoil...")

            validation = self.validate_naca_airfoil(airfoil)
            if not validation["is_valid"]:
                return {
                    "success": False,
                    "error": f"Invalid airfoil: {validation['errors']}",
                }

            # Step 2: Create configuration
            if progress_callback:
                progress_callback(20, "Creating analysis configuration...")

            config = self.create_analysis_config(airfoil, parameters)

            # Step 3: Validate configuration
            if progress_callback:
                progress_callback(30, "Validating configuration...")

            validation_result = self.analysis_service.validate_analysis_config(config)
            if not validation_result.is_valid:
                return {
                    "success": False,
                    "error": f"Configuration validation failed: {validation_result.errors[0].message}",
                }

            # Step 4: Run analysis
            if progress_callback:
                progress_callback(40, "Running XFoil analysis...")

            def analysis_progress_callback(progress):
                if progress_callback:
                    # Map analysis progress to overall workflow progress (40-90%)
                    overall_progress = 40 + (progress.progress_percent * 0.5)
                    progress_callback(overall_progress, progress.current_step)

            result = await self.analysis_service.run_analysis(
                config,
                progress_callback=analysis_progress_callback,
            )

            # Step 5: Process results
            if progress_callback:
                progress_callback(90, "Processing results...")

            if result.status == "success":
                processed_results = self.process_airfoil_results(
                    result.raw_data,
                    validation,
                )

                if progress_callback:
                    progress_callback(100, "Analysis completed successfully")

                return {
                    "success": True,
                    "airfoil_info": validation,
                    "analysis_config": config.to_dict(),
                    "raw_results": result.raw_data,
                    "processed_results": processed_results,
                    "duration": result.duration,
                }
            else:
                return {
                    "success": False,
                    "error": result.error_message or "Analysis failed",
                }

        except Exception as e:
            self.logger.error(f"Airfoil workflow error: {e}")
            return {"success": False, "error": str(e)}

    def process_airfoil_results(
        self,
        raw_data: Dict[str, Any],
        airfoil_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process raw airfoil analysis results."""

        if not raw_data or "polars" not in raw_data:
            return {"error": "No polar data available"}

        polars = raw_data["polars"]

        try:
            # Basic performance metrics
            cl_values = polars["cl"]
            cd_values = polars["cd"]
            alpha_values = polars["alpha"]

            # Find key performance points
            max_cl_idx = (
                np.argmax(cl_values)
                if NUMPY_AVAILABLE
                else cl_values.index(max(cl_values))
            )
            min_cd_idx = (
                np.argmin(cd_values)
                if NUMPY_AVAILABLE
                else cd_values.index(min(cd_values))
            )

            # Calculate L/D ratios
            if NUMPY_AVAILABLE:
                ld_ratios = np.array(cl_values) / np.array(cd_values)
                max_ld_idx = np.argmax(ld_ratios)
            else:
                ld_ratios = [
                    cl / cd if cd > 0 else 0 for cl, cd in zip(cl_values, cd_values)
                ]
                max_ld_idx = ld_ratios.index(max(ld_ratios))

            # Find stall angle (approximate)
            stall_idx = max_cl_idx

            processed = {
                "performance_summary": {
                    "max_cl": {
                        "value": cl_values[max_cl_idx],
                        "alpha": alpha_values[max_cl_idx],
                        "cd": cd_values[max_cl_idx],
                    },
                    "min_cd": {
                        "value": cd_values[min_cd_idx],
                        "alpha": alpha_values[min_cd_idx],
                        "cl": cl_values[min_cd_idx],
                    },
                    "max_ld": {
                        "value": ld_ratios[max_ld_idx],
                        "alpha": alpha_values[max_ld_idx],
                        "cl": cl_values[max_ld_idx],
                        "cd": cd_values[max_ld_idx],
                    },
                    "stall_angle": alpha_values[stall_idx],
                },
                "airfoil_characteristics": {
                    "type": airfoil_info.get("airfoil_type", "Unknown"),
                    "parameters": airfoil_info.get("parameters", {}),
                    "reynolds": raw_data.get("reynolds", 0),
                    "mach": raw_data.get("mach", 0),
                },
                "polar_data": {
                    "alpha": alpha_values,
                    "cl": cl_values,
                    "cd": cd_values,
                    "cm": polars.get("cm", []),
                    "ld_ratio": ld_ratios,
                },
            }

            return processed

        except Exception as e:
            return {"error": f"Error processing results: {e}"}

    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate a text summary report."""

        if not results.get("success"):
            return f"Analysis Failed: {results.get('error', 'Unknown error')}"

        processed = results.get("processed_results", {})
        airfoil_info = results.get("airfoil_info", {})

        if "error" in processed:
            return f"Results Processing Failed: {processed['error']}"

        summary = processed.get("performance_summary", {})
        characteristics = processed.get("airfoil_characteristics", {})

        report = []
        report.append("=" * 50)
        report.append("ICARUS AIRFOIL ANALYSIS REPORT")
        report.append("=" * 50)
        report.append("")

        # Airfoil information
        report.append("AIRFOIL INFORMATION:")
        report.append(f"  Type: {characteristics.get('type', 'Unknown')}")

        params = characteristics.get("parameters", {})
        if params:
            if "max_camber" in params:
                report.append(f"  Max Camber: {params['max_camber']:.1%}")
                report.append(f"  Camber Position: {params['camber_position']:.1%}")
            report.append(f"  Thickness: {params.get('thickness', 0):.1%}")

        report.append(f"  Reynolds Number: {characteristics.get('reynolds', 0):,.0f}")
        report.append(f"  Mach Number: {characteristics.get('mach', 0):.3f}")
        report.append("")

        # Performance summary
        report.append("PERFORMANCE SUMMARY:")

        max_cl = summary.get("max_cl", {})
        if max_cl:
            report.append(
                f"  Maximum CL: {max_cl.get('value', 0):.3f} at α = {max_cl.get('alpha', 0):.1f}°",
            )

        min_cd = summary.get("min_cd", {})
        if min_cd:
            report.append(
                f"  Minimum CD: {min_cd.get('value', 0):.4f} at α = {min_cd.get('alpha', 0):.1f}°",
            )

        max_ld = summary.get("max_ld", {})
        if max_ld:
            report.append(
                f"  Maximum L/D: {max_ld.get('value', 0):.1f} at α = {max_ld.get('alpha', 0):.1f}°",
            )

        stall_angle = summary.get("stall_angle", 0)
        report.append(f"  Stall Angle: {stall_angle:.1f}°")
        report.append("")

        # Analysis info
        duration = results.get("duration")
        if duration:
            report.append(f"Analysis completed in {duration:.2f} seconds")

        report.append("=" * 50)

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
                            polar_data["cl"][i],
                            polar_data["cd"][i],
                            polar_data["cm"][i]
                            if i < len(polar_data.get("cm", []))
                            else "",
                            polar_data["ld_ratio"][i],
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
async def analyze_naca_airfoil(
    naca_code: str,
    reynolds: float = 1000000,
    angle_range: tuple = (-10, 15, 0.5),
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Analyze a NACA airfoil with default parameters."""

    workflow = AirfoilWorkflow()

    parameters = {
        "reynolds": reynolds,
        "min_aoa": angle_range[0],
        "max_aoa": angle_range[1],
        "aoa_step": angle_range[2],
    }

    return await workflow.run_airfoil_analysis(naca_code, parameters, progress_callback)


def print_airfoil_summary(results: Dict[str, Any]):
    """Print a summary of airfoil analysis results."""
    workflow = AirfoilWorkflow()
    report = workflow.generate_summary_report(results)
    print(report)


# Example usage
if __name__ == "__main__":

    async def main():
        # Example: Analyze NACA 2412 airfoil
        def progress_callback(percent, status):
            print(f"Progress: {percent:.1f}% - {status}")

        results = await analyze_naca_airfoil(
            "NACA2412",
            reynolds=1000000,
            angle_range=(-10, 15, 0.5),
            progress_callback=progress_callback,
        )

        print_airfoil_summary(results)

        # Export results
        workflow = AirfoilWorkflow()
        workflow.export_results(results, "naca2412_results.json", "json")
        workflow.export_results(results, "naca2412_results.csv", "csv")
        workflow.export_results(results, "naca2412_report.txt", "txt")

    asyncio.run(main())
