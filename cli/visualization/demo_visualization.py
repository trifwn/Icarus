"""
Demo Visualization and Export Functionality

This module provides simplified visualization and export capabilities for
analysis results, designed for demonstration purposes.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class DemoVisualizer:
    """Simplified visualization for demo purposes."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.console = Console() if RICH_AVAILABLE else None

    def plot_airfoil_polar(
        self,
        results: Dict[str, Any],
        output_file: Optional[str] = None,
        show_plot: bool = True,
    ) -> bool:
        """Plot airfoil polar curves."""

        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("Matplotlib not available for plotting")
            return False

        if not results.get("success"):
            self.logger.error("Cannot plot failed analysis results")
            return False

        try:
            processed = results.get("processed_results", {})
            polar_data = processed.get("polar_data", {})

            if not polar_data:
                self.logger.error("No polar data available for plotting")
                return False

            alpha = polar_data["alpha"]
            cl = polar_data["cl"]
            cd = polar_data["cd"]
            ld_ratio = polar_data.get("ld_ratio", [])

            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle("Airfoil Analysis Results", fontsize=16, fontweight="bold")

            # CL vs Alpha
            ax1.plot(alpha, cl, "b-", linewidth=2, label="CL")
            ax1.set_xlabel("Angle of Attack (degrees)")
            ax1.set_ylabel("Lift Coefficient (CL)")
            ax1.set_title("Lift Curve")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Mark max CL point
            summary = processed.get("performance_summary", {})
            max_cl_info = summary.get("max_cl", {})
            if max_cl_info:
                ax1.plot(
                    max_cl_info["alpha"],
                    max_cl_info["value"],
                    "ro",
                    markersize=8,
                    label=f"Max CL = {max_cl_info['value']:.3f}",
                )
                ax1.legend()

            # CD vs Alpha
            ax2.plot(alpha, cd, "r-", linewidth=2, label="CD")
            ax2.set_xlabel("Angle of Attack (degrees)")
            ax2.set_ylabel("Drag Coefficient (CD)")
            ax2.set_title("Drag Curve")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            # Mark min CD point
            min_cd_info = summary.get("min_cd", {})
            if min_cd_info:
                ax2.plot(
                    min_cd_info["alpha"],
                    min_cd_info["value"],
                    "go",
                    markersize=8,
                    label=f"Min CD = {min_cd_info['value']:.4f}",
                )
                ax2.legend()

            # Drag Polar (CL vs CD)
            ax3.plot(cd, cl, "g-", linewidth=2)
            ax3.set_xlabel("Drag Coefficient (CD)")
            ax3.set_ylabel("Lift Coefficient (CL)")
            ax3.set_title("Drag Polar")
            ax3.grid(True, alpha=0.3)

            # L/D vs Alpha
            if ld_ratio:
                ax4.plot(alpha, ld_ratio, "m-", linewidth=2, label="L/D")
                ax4.set_xlabel("Angle of Attack (degrees)")
                ax4.set_ylabel("Lift-to-Drag Ratio (L/D)")
                ax4.set_title("L/D Ratio")
                ax4.grid(True, alpha=0.3)
                ax4.legend()

                # Mark max L/D point
                max_ld_info = summary.get("max_ld", {})
                if max_ld_info:
                    ax4.plot(
                        max_ld_info["alpha"],
                        max_ld_info["value"],
                        "co",
                        markersize=8,
                        label=f"Max L/D = {max_ld_info['value']:.1f}",
                    )
                    ax4.legend()

            plt.tight_layout()

            # Save plot if requested
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches="tight")
                self.logger.info(f"Plot saved to: {output_file}")

            # Show plot if requested
            if show_plot:
                plt.show()
            else:
                plt.close()

            return True

        except Exception as e:
            self.logger.error(f"Error plotting airfoil polar: {e}")
            return False

    def plot_airplane_polar(
        self,
        results: Dict[str, Any],
        output_file: Optional[str] = None,
        show_plot: bool = True,
    ) -> bool:
        """Plot airplane polar curves."""

        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("Matplotlib not available for plotting")
            return False

        if not results.get("success"):
            self.logger.error("Cannot plot failed analysis results")
            return False

        try:
            processed = results.get("processed_results", {})
            polar_data = processed.get("polar_data", {})

            if not polar_data:
                self.logger.error("No polar data available for plotting")
                return False

            alpha = polar_data["alpha"]
            CL = polar_data["CL"]
            CD = polar_data["CD"]
            CM = polar_data.get("CM", [])
            LD_ratio = polar_data.get("LD_ratio", [])

            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle("Airplane Analysis Results", fontsize=16, fontweight="bold")

            # CL vs Alpha
            ax1.plot(alpha, CL, "b-", linewidth=2, label="CL")
            ax1.set_xlabel("Angle of Attack (degrees)")
            ax1.set_ylabel("Lift Coefficient (CL)")
            ax1.set_title("Lift Curve")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Mark max CL point
            summary = processed.get("performance_summary", {})
            max_CL_info = summary.get("max_CL", {})
            if max_CL_info:
                ax1.plot(
                    max_CL_info["alpha"],
                    max_CL_info["value"],
                    "ro",
                    markersize=8,
                    label=f"Max CL = {max_CL_info['value']:.3f}",
                )
                ax1.legend()

            # CD vs Alpha
            ax2.plot(alpha, CD, "r-", linewidth=2, label="CD")
            ax2.set_xlabel("Angle of Attack (degrees)")
            ax2.set_ylabel("Drag Coefficient (CD)")
            ax2.set_title("Drag Curve")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            # Mark min CD point
            min_CD_info = summary.get("min_CD", {})
            if min_CD_info:
                ax2.plot(
                    min_CD_info["alpha"],
                    min_CD_info["value"],
                    "go",
                    markersize=8,
                    label=f"Min CD = {min_CD_info['value']:.4f}",
                )
                ax2.legend()

            # Drag Polar (CL vs CD)
            ax3.plot(CD, CL, "g-", linewidth=2)
            ax3.set_xlabel("Drag Coefficient (CD)")
            ax3.set_ylabel("Lift Coefficient (CL)")
            ax3.set_title("Drag Polar")
            ax3.grid(True, alpha=0.3)

            # CM vs Alpha or L/D vs Alpha
            if CM:
                ax4.plot(alpha, CM, "orange", linewidth=2, label="CM")
                ax4.set_xlabel("Angle of Attack (degrees)")
                ax4.set_ylabel("Pitching Moment Coefficient (CM)")
                ax4.set_title("Pitching Moment")
                ax4.grid(True, alpha=0.3)
                ax4.legend()
                ax4.axhline(y=0, color="k", linestyle="--", alpha=0.5)
            elif LD_ratio:
                ax4.plot(alpha, LD_ratio, "m-", linewidth=2, label="L/D")
                ax4.set_xlabel("Angle of Attack (degrees)")
                ax4.set_ylabel("Lift-to-Drag Ratio (L/D)")
                ax4.set_title("L/D Ratio")
                ax4.grid(True, alpha=0.3)
                ax4.legend()

                # Mark max L/D point
                max_LD_info = summary.get("max_LD", {})
                if max_LD_info:
                    ax4.plot(
                        max_LD_info["alpha"],
                        max_LD_info["value"],
                        "co",
                        markersize=8,
                        label=f"Max L/D = {max_LD_info['value']:.1f}",
                    )
                    ax4.legend()

            plt.tight_layout()

            # Save plot if requested
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches="tight")
                self.logger.info(f"Plot saved to: {output_file}")

            # Show plot if requested
            if show_plot:
                plt.show()
            else:
                plt.close()

            return True

        except Exception as e:
            self.logger.error(f"Error plotting airplane polar: {e}")
            return False

    def create_comparison_plot(
        self,
        results_list: List[Dict[str, Any]],
        labels: List[str],
        output_file: Optional[str] = None,
        show_plot: bool = True,
    ) -> bool:
        """Create comparison plot for multiple analysis results."""

        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("Matplotlib not available for plotting")
            return False

        if len(results_list) != len(labels):
            self.logger.error("Number of results must match number of labels")
            return False

        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle("Analysis Comparison", fontsize=16, fontweight="bold")

            colors = ["b", "r", "g", "m", "c", "y", "k"]

            for i, (results, label) in enumerate(zip(results_list, labels)):
                if not results.get("success"):
                    continue

                processed = results.get("processed_results", {})
                polar_data = processed.get("polar_data", {})

                if not polar_data:
                    continue

                color = colors[i % len(colors)]

                # Determine if airfoil or airplane data
                if "cl" in polar_data:  # Airfoil data
                    alpha = polar_data["alpha"]
                    cl = polar_data["cl"]
                    cd = polar_data["cd"]
                    ax1.plot(alpha, cl, color=color, linewidth=2, label=f"{label} CL")
                    ax2.plot(cd, cl, color=color, linewidth=2, label=label)
                elif "CL" in polar_data:  # Airplane data
                    alpha = polar_data["alpha"]
                    CL = polar_data["CL"]
                    CD = polar_data["CD"]
                    ax1.plot(alpha, CL, color=color, linewidth=2, label=f"{label} CL")
                    ax2.plot(CD, CL, color=color, linewidth=2, label=label)

            ax1.set_xlabel("Angle of Attack (degrees)")
            ax1.set_ylabel("Lift Coefficient")
            ax1.set_title("Lift Curves")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            ax2.set_xlabel("Drag Coefficient")
            ax2.set_ylabel("Lift Coefficient")
            ax2.set_title("Drag Polars")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()

            # Save plot if requested
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches="tight")
                self.logger.info(f"Comparison plot saved to: {output_file}")

            # Show plot if requested
            if show_plot:
                plt.show()
            else:
                plt.close()

            return True

        except Exception as e:
            self.logger.error(f"Error creating comparison plot: {e}")
            return False

    def display_results_table(self, results: Dict[str, Any]) -> bool:
        """Display results in a formatted table."""

        if not RICH_AVAILABLE:
            # Fallback to simple text output
            return self._display_results_text(results)

        if not results.get("success"):
            self.console.print(
                f"[red]Analysis Failed: {results.get('error', 'Unknown error')}[/red]",
            )
            return False

        try:
            processed = results.get("processed_results", {})
            summary = processed.get("performance_summary", {})
            characteristics = processed.get(
                "aircraft_characteristics",
                {},
            ) or processed.get("airfoil_characteristics", {})

            # Create performance summary table
            table = Table(
                title="Performance Summary",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Parameter", style="cyan", no_wrap=True)
            table.add_column("Value", style="green")
            table.add_column("Angle (°)", style="yellow")
            table.add_column("Additional Info", style="white")

            # Add performance metrics
            if "max_cl" in summary or "max_CL" in summary:
                max_lift = summary.get("max_cl", summary.get("max_CL", {}))
                table.add_row(
                    "Maximum Lift Coefficient",
                    f"{max_lift.get('value', 0):.3f}",
                    f"{max_lift.get('alpha', 0):.1f}",
                    f"CD = {max_lift.get('cd', max_lift.get('CD', 0)):.4f}",
                )

            if "min_cd" in summary or "min_CD" in summary:
                min_drag = summary.get("min_cd", summary.get("min_CD", {}))
                table.add_row(
                    "Minimum Drag Coefficient",
                    f"{min_drag.get('value', 0):.4f}",
                    f"{min_drag.get('alpha', 0):.1f}",
                    f"CL = {min_drag.get('cl', min_drag.get('CL', 0)):.3f}",
                )

            if "max_ld" in summary or "max_LD" in summary:
                max_ld = summary.get("max_ld", summary.get("max_LD", {}))
                table.add_row(
                    "Maximum L/D Ratio",
                    f"{max_ld.get('value', 0):.1f}",
                    f"{max_ld.get('alpha', 0):.1f}",
                    f"CL/CD = {max_ld.get('cl', max_ld.get('CL', 0)):.3f}/{max_ld.get('cd', max_ld.get('CD', 0)):.4f}",
                )

            if "stall_angle" in summary:
                table.add_row(
                    "Stall Angle",
                    "-",
                    f"{summary['stall_angle']:.1f}",
                    "Approximate",
                )

            if "stall_speed" in summary:
                table.add_row(
                    "Stall Speed",
                    f"{summary['stall_speed']:.1f} m/s",
                    "-",
                    "Estimated",
                )

            self.console.print(table)

            # Create characteristics table
            if characteristics:
                char_table = Table(
                    title="Configuration",
                    show_header=True,
                    header_style="bold blue",
                )
                char_table.add_column("Property", style="cyan")
                char_table.add_column("Value", style="white")

                if "type" in characteristics:
                    char_table.add_row("Airfoil Type", characteristics["type"])

                if "reynolds" in characteristics:
                    char_table.add_row(
                        "Reynolds Number",
                        f"{characteristics['reynolds']:,.0f}",
                    )

                if "wing_area" in characteristics:
                    char_table.add_row(
                        "Wing Area",
                        f"{characteristics['wing_area']:.2f} m²",
                    )

                if "aspect_ratio" in characteristics:
                    char_table.add_row(
                        "Aspect Ratio",
                        f"{characteristics['aspect_ratio']:.1f}",
                    )

                if "wing_loading" in characteristics:
                    char_table.add_row(
                        "Wing Loading",
                        f"{characteristics['wing_loading']:.1f} N/m²",
                    )

                if "weight" in characteristics:
                    char_table.add_row("Weight", f"{characteristics['weight']:.0f} kg")

                self.console.print(char_table)

            return True

        except Exception as e:
            self.logger.error(f"Error displaying results table: {e}")
            return False

    def _display_results_text(self, results: Dict[str, Any]) -> bool:
        """Fallback text display when Rich is not available."""

        if not results.get("success"):
            print(f"Analysis Failed: {results.get('error', 'Unknown error')}")
            return False

        try:
            processed = results.get("processed_results", {})
            summary = processed.get("performance_summary", {})

            print("\n" + "=" * 50)
            print("PERFORMANCE SUMMARY")
            print("=" * 50)

            if "max_cl" in summary or "max_CL" in summary:
                max_lift = summary.get("max_cl", summary.get("max_CL", {}))
                print(
                    f"Maximum Lift Coefficient: {max_lift.get('value', 0):.3f} at {max_lift.get('alpha', 0):.1f}°",
                )

            if "min_cd" in summary or "min_CD" in summary:
                min_drag = summary.get("min_cd", summary.get("min_CD", {}))
                print(
                    f"Minimum Drag Coefficient: {min_drag.get('value', 0):.4f} at {min_drag.get('alpha', 0):.1f}°",
                )

            if "max_ld" in summary or "max_LD" in summary:
                max_ld = summary.get("max_ld", summary.get("max_LD", {}))
                print(
                    f"Maximum L/D Ratio: {max_ld.get('value', 0):.1f} at {max_ld.get('alpha', 0):.1f}°",
                )

            print("=" * 50)

            return True

        except Exception as e:
            self.logger.error(f"Error displaying text results: {e}")
            return False


class DemoExporter:
    """Simplified export functionality for demo purposes."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def export_to_json(self, results: Dict[str, Any], output_file: str) -> bool:
        """Export results to JSON format."""
        try:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Results exported to JSON: {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting to JSON: {e}")
            return False

    def export_to_csv(self, results: Dict[str, Any], output_file: str) -> bool:
        """Export polar data to CSV format."""
        try:
            if not results.get("success"):
                return False

            processed = results.get("processed_results", {})
            polar_data = processed.get("polar_data", {})

            if not polar_data:
                return False

            import csv

            with open(output_file, "w", newline="") as f:
                writer = csv.writer(f)

                # Determine data type and write appropriate headers
                if "cl" in polar_data:  # Airfoil data
                    writer.writerow(["Alpha", "CL", "CD", "CM", "L/D"])
                    alpha = polar_data["alpha"]
                    cl = polar_data["cl"]
                    cd = polar_data["cd"]
                    cm = polar_data.get("cm", [])
                    ld = polar_data.get("ld_ratio", [])

                    for i in range(len(alpha)):
                        row = [
                            alpha[i],
                            cl[i],
                            cd[i],
                            cm[i] if i < len(cm) else "",
                            ld[i] if i < len(ld) else "",
                        ]
                        writer.writerow(row)

                elif "CL" in polar_data:  # Airplane data
                    writer.writerow(["Alpha", "CL", "CD", "CM", "L/D"])
                    alpha = polar_data["alpha"]
                    CL = polar_data["CL"]
                    CD = polar_data["CD"]
                    CM = polar_data.get("CM", [])
                    LD = polar_data.get("LD_ratio", [])

                    for i in range(len(alpha)):
                        row = [
                            alpha[i],
                            CL[i],
                            CD[i],
                            CM[i] if i < len(CM) else "",
                            LD[i] if i < len(LD) else "",
                        ]
                        writer.writerow(row)

            self.logger.info(f"Polar data exported to CSV: {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {e}")
            return False

    def export_to_matlab(self, results: Dict[str, Any], output_file: str) -> bool:
        """Export results to MATLAB .m file format."""
        try:
            if not results.get("success"):
                return False

            processed = results.get("processed_results", {})
            polar_data = processed.get("polar_data", {})

            if not polar_data:
                return False

            with open(output_file, "w") as f:
                f.write("% ICARUS Analysis Results\n")
                f.write("% Generated by ICARUS CLI Demo\n\n")

                # Write polar data
                if "cl" in polar_data:  # Airfoil data
                    f.write("% Airfoil Polar Data\n")
                    f.write(f"alpha = {polar_data['alpha']};\n")
                    f.write(f"cl = {polar_data['cl']};\n")
                    f.write(f"cd = {polar_data['cd']};\n")
                    if "cm" in polar_data:
                        f.write(f"cm = {polar_data['cm']};\n")
                    if "ld_ratio" in polar_data:
                        f.write(f"ld_ratio = {polar_data['ld_ratio']};\n")

                elif "CL" in polar_data:  # Airplane data
                    f.write("% Airplane Polar Data\n")
                    f.write(f"alpha = {polar_data['alpha']};\n")
                    f.write(f"CL = {polar_data['CL']};\n")
                    f.write(f"CD = {polar_data['CD']};\n")
                    if "CM" in polar_data:
                        f.write(f"CM = {polar_data['CM']};\n")
                    if "LD_ratio" in polar_data:
                        f.write(f"LD_ratio = {polar_data['LD_ratio']};\n")

                # Write performance summary
                summary = processed.get("performance_summary", {})
                if summary:
                    f.write("\n% Performance Summary\n")
                    for key, value in summary.items():
                        if isinstance(value, dict):
                            f.write(f"% {key}: {value}\n")

            self.logger.info(f"Results exported to MATLAB: {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting to MATLAB: {e}")
            return False

    def create_summary_report(self, results: Dict[str, Any], output_file: str) -> bool:
        """Create a comprehensive text summary report."""
        try:
            # Use the workflow's report generation if available
            if "airfoil_info" in results:
                from cli.workflows.airfoil_workflow import AirfoilWorkflow

                workflow = AirfoilWorkflow()
                report = workflow.generate_summary_report(results)
            elif "airplane_config" in results:
                from cli.workflows.airplane_workflow import AirplaneWorkflow

                workflow = AirplaneWorkflow()
                report = workflow.generate_summary_report(results)
            else:
                # Generic report
                report = self._generate_generic_report(results)

            with open(output_file, "w") as f:
                f.write(report)

            self.logger.info(f"Summary report created: {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error creating summary report: {e}")
            return False

    def _generate_generic_report(self, results: Dict[str, Any]) -> str:
        """Generate a generic report for unknown result types."""
        if not results.get("success"):
            return f"Analysis Failed: {results.get('error', 'Unknown error')}"

        report = []
        report.append("ICARUS ANALYSIS REPORT")
        report.append("=" * 40)
        report.append("")

        if "processed_results" in results:
            processed = results["processed_results"]
            if "performance_summary" in processed:
                report.append("Performance Summary:")
                summary = processed["performance_summary"]
                for key, value in summary.items():
                    report.append(f"  {key}: {value}")

        report.append("")
        report.append("Analysis completed successfully")

        return "\n".join(report)


# Convenience functions
def visualize_results(results: Dict[str, Any], output_dir: Optional[str] = None):
    """Visualize analysis results with plots and tables."""
    visualizer = DemoVisualizer()

    # Display table
    visualizer.display_results_table(results)

    # Create plots
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        if "airfoil_info" in results:
            plot_file = output_path / "airfoil_polar.png"
            visualizer.plot_airfoil_polar(results, str(plot_file), show_plot=False)
        elif "airplane_config" in results:
            plot_file = output_path / "airplane_polar.png"
            visualizer.plot_airplane_polar(results, str(plot_file), show_plot=False)
    else:
        # Show plots
        if "airfoil_info" in results:
            visualizer.plot_airfoil_polar(results)
        elif "airplane_config" in results:
            visualizer.plot_airplane_polar(results)


def export_all_formats(results: Dict[str, Any], base_filename: str):
    """Export results in all available formats."""
    exporter = DemoExporter()

    base_path = Path(base_filename)
    stem = base_path.stem
    parent = base_path.parent

    # Export to different formats
    formats = [
        ("json", exporter.export_to_json),
        ("csv", exporter.export_to_csv),
        ("m", exporter.export_to_matlab),
        ("txt", exporter.create_summary_report),
    ]

    for ext, export_func in formats:
        output_file = parent / f"{stem}.{ext}"
        try:
            export_func(results, str(output_file))
        except Exception as e:
            logging.error(f"Failed to export {ext}: {e}")


# Example usage
if __name__ == "__main__":
    # Example with mock data
    mock_airfoil_results = {
        "success": True,
        "processed_results": {
            "performance_summary": {
                "max_cl": {"value": 1.234, "alpha": 12.5, "cd": 0.0234},
                "min_cd": {"value": 0.0123, "alpha": 2.0, "cl": 0.456},
                "max_ld": {"value": 45.6, "alpha": 4.0, "cl": 0.789, "cd": 0.0173},
            },
            "airfoil_characteristics": {"type": "4-digit", "reynolds": 1000000},
            "polar_data": {
                "alpha": list(range(-10, 16)),
                "cl": [0.1 * i for i in range(-10, 16)],
                "cd": [0.01 + 0.001 * i**2 for i in range(-10, 16)],
                "ld_ratio": [10 + i for i in range(-10, 16)],
            },
        },
    }

    # Test visualization
    visualizer = DemoVisualizer()
    visualizer.display_results_table(mock_airfoil_results)

    # Test export
    exporter = DemoExporter()
    exporter.export_to_json(mock_airfoil_results, "test_results.json")
    exporter.export_to_csv(mock_airfoil_results, "test_results.csv")
    exporter.create_summary_report(mock_airfoil_results, "test_report.txt")
