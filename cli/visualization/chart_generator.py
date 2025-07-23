"""Chart Generator - Creates charts from analysis results

This module generates specific chart types from ICARUS analysis results,
providing standardized visualization for different analysis types.
"""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from rich.console import Console


class ChartGenerator:
    """Generates charts from analysis results."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize the chart generator.

        Args:
            console: Rich console for output (optional)
        """
        self.console = console or Console()

        # Chart templates for different analysis types
        self.chart_templates = {
            "airfoil_polar": {
                "plots": ["cl_alpha", "cd_alpha", "cm_alpha", "cl_cd"],
                "layout": (2, 2),
                "title": "Airfoil Polar Analysis",
            },
            "airplane_polar": {
                "plots": ["cl_alpha", "cd_alpha", "cm_alpha", "cl_cd", "efficiency"],
                "layout": (2, 3),
                "title": "Airplane Polar Analysis",
            },
            "pressure_distribution": {
                "plots": ["cp_distribution"],
                "layout": (1, 1),
                "title": "Pressure Distribution",
            },
            "geometry": {
                "plots": ["airfoil_shape", "wing_planform"],
                "layout": (1, 2),
                "title": "Geometry Visualization",
            },
            "convergence": {
                "plots": ["residuals", "forces"],
                "layout": (2, 1),
                "title": "Convergence Analysis",
            },
        }

    def generate_chart(
        self,
        results: Dict[str, Any],
        chart_type: str = "polar",
        **options,
    ) -> Figure:
        """Generate a chart from analysis results.

        Args:
            results: Analysis results dictionary
            chart_type: Type of chart to generate
            **options: Chart generation options

        Returns:
            Matplotlib Figure object
        """
        try:
            if chart_type == "airfoil_polar":
                return self._generate_airfoil_polar_chart(results, **options)
            elif chart_type == "airplane_polar":
                return self._generate_airplane_polar_chart(results, **options)
            elif chart_type == "pressure_distribution":
                return self._generate_pressure_chart(results, **options)
            elif chart_type == "geometry":
                return self._generate_geometry_chart(results, **options)
            elif chart_type == "convergence":
                return self._generate_convergence_chart(results, **options)
            elif chart_type == "comparison":
                return self._generate_comparison_chart(results, **options)
            elif chart_type == "sensitivity":
                return self._generate_sensitivity_chart(results, **options)
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")

        except Exception as e:
            self.console.print(
                f"[red]✗[/red] Failed to generate {chart_type} chart: {e}",
            )
            raise

    def _generate_airfoil_polar_chart(
        self,
        results: Dict[str, Any],
        **options,
    ) -> Figure:
        """Generate airfoil polar chart."""
        template = self.chart_templates["airfoil_polar"]
        fig, axes = plt.subplots(
            *template["layout"],
            figsize=options.get("figsize", (12, 10)),
        )
        fig.suptitle(
            options.get("title", template["title"]),
            fontsize=14,
            fontweight="bold",
        )

        # Flatten axes for easier indexing
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        else:
            axes = axes.flatten()

        # Extract data
        alpha = results.get("alpha", [])
        cl = results.get("cl", [])
        cd = results.get("cd", [])
        cm = results.get("cm", [])

        # CL vs Alpha
        if len(axes) > 0:
            axes[0].plot(alpha, cl, "b-o", linewidth=2, markersize=4)
            axes[0].set_xlabel("Angle of Attack (deg)")
            axes[0].set_ylabel("CL")
            axes[0].set_title("Lift Coefficient")
            axes[0].grid(True, alpha=0.3)

        # CD vs Alpha
        if len(axes) > 1:
            axes[1].plot(alpha, cd, "r-o", linewidth=2, markersize=4)
            axes[1].set_xlabel("Angle of Attack (deg)")
            axes[1].set_ylabel("CD")
            axes[1].set_title("Drag Coefficient")
            axes[1].grid(True, alpha=0.3)

        # CM vs Alpha
        if len(axes) > 2:
            axes[2].plot(alpha, cm, "g-o", linewidth=2, markersize=4)
            axes[2].set_xlabel("Angle of Attack (deg)")
            axes[2].set_ylabel("CM")
            axes[2].set_title("Moment Coefficient")
            axes[2].grid(True, alpha=0.3)

        # CL vs CD (Drag Polar)
        if len(axes) > 3:
            axes[3].plot(cd, cl, "m-o", linewidth=2, markersize=4)
            axes[3].set_xlabel("CD")
            axes[3].set_ylabel("CL")
            axes[3].set_title("Drag Polar")
            axes[3].grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def _generate_airplane_polar_chart(
        self,
        results: Dict[str, Any],
        **options,
    ) -> Figure:
        """Generate airplane polar chart."""
        template = self.chart_templates["airplane_polar"]
        fig, axes = plt.subplots(
            *template["layout"],
            figsize=options.get("figsize", (15, 10)),
        )
        fig.suptitle(
            options.get("title", template["title"]),
            fontsize=14,
            fontweight="bold",
        )

        axes = axes.flatten()

        # Extract data
        alpha = results.get("alpha", [])
        cl = results.get("cl", [])
        cd = results.get("cd", [])
        cm = results.get("cm", [])

        # Calculate efficiency (L/D)
        efficiency = np.array(cl) / np.array(cd) if len(cl) == len(cd) else []

        # CL vs Alpha
        axes[0].plot(alpha, cl, "b-o", linewidth=2, markersize=4)
        axes[0].set_xlabel("Angle of Attack (deg)")
        axes[0].set_ylabel("CL")
        axes[0].set_title("Lift Coefficient")
        axes[0].grid(True, alpha=0.3)

        # CD vs Alpha
        axes[1].plot(alpha, cd, "r-o", linewidth=2, markersize=4)
        axes[1].set_xlabel("Angle of Attack (deg)")
        axes[1].set_ylabel("CD")
        axes[1].set_title("Drag Coefficient")
        axes[1].grid(True, alpha=0.3)

        # CM vs Alpha
        axes[2].plot(alpha, cm, "g-o", linewidth=2, markersize=4)
        axes[2].set_xlabel("Angle of Attack (deg)")
        axes[2].set_ylabel("CM")
        axes[2].set_title("Moment Coefficient")
        axes[2].grid(True, alpha=0.3)

        # CL vs CD (Drag Polar)
        axes[3].plot(cd, cl, "m-o", linewidth=2, markersize=4)
        axes[3].set_xlabel("CD")
        axes[3].set_ylabel("CL")
        axes[3].set_title("Drag Polar")
        axes[3].grid(True, alpha=0.3)

        # L/D vs Alpha
        if len(efficiency) > 0:
            axes[4].plot(alpha, efficiency, "c-o", linewidth=2, markersize=4)
            axes[4].set_xlabel("Angle of Attack (deg)")
            axes[4].set_ylabel("L/D")
            axes[4].set_title("Lift-to-Drag Ratio")
            axes[4].grid(True, alpha=0.3)

        # Hide unused subplot
        if len(axes) > 5:
            axes[5].set_visible(False)

        fig.tight_layout()
        return fig

    def _generate_pressure_chart(self, results: Dict[str, Any], **options) -> Figure:
        """Generate pressure distribution chart."""
        fig, ax = plt.subplots(figsize=options.get("figsize", (10, 6)))
        fig.suptitle(
            options.get("title", "Pressure Distribution"),
            fontsize=14,
            fontweight="bold",
        )

        # Extract data
        x = results.get("x", [])
        cp_upper = results.get("cp_upper", [])
        cp_lower = results.get("cp_lower", [])

        # Plot pressure distribution
        if cp_upper:
            ax.plot(x, cp_upper, "b-", linewidth=2, label="Upper Surface")
        if cp_lower:
            ax.plot(x, cp_lower, "r-", linewidth=2, label="Lower Surface")

        # Formatting
        ax.invert_yaxis()  # Typical for pressure plots
        ax.set_xlabel("x/c")
        ax.set_ylabel("Cp")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        return fig

    def _generate_geometry_chart(self, results: Dict[str, Any], **options) -> Figure:
        """Generate geometry visualization chart."""
        fig, axes = plt.subplots(1, 2, figsize=options.get("figsize", (12, 6)))
        fig.suptitle(
            options.get("title", "Geometry Visualization"),
            fontsize=14,
            fontweight="bold",
        )

        # Airfoil shape
        if "airfoil" in results:
            airfoil_data = results["airfoil"]
            x = airfoil_data.get("x", [])
            y = airfoil_data.get("y", [])

            axes[0].plot(x, y, "b-", linewidth=2)
            axes[0].set_xlabel("x/c")
            axes[0].set_ylabel("y/c")
            axes[0].set_title("Airfoil Shape")
            axes[0].grid(True, alpha=0.3)
            axes[0].set_aspect("equal")

        # Wing planform
        if "wing" in results:
            wing_data = results["wing"]
            y_span = wing_data.get("y", [])
            chord = wing_data.get("chord", [])

            axes[1].plot(y_span, chord, "r-", linewidth=2)
            axes[1].set_xlabel("Span Position")
            axes[1].set_ylabel("Chord Length")
            axes[1].set_title("Wing Planform")
            axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def _generate_convergence_chart(self, results: Dict[str, Any], **options) -> Figure:
        """Generate convergence analysis chart."""
        fig, axes = plt.subplots(2, 1, figsize=options.get("figsize", (10, 8)))
        fig.suptitle(
            options.get("title", "Convergence Analysis"),
            fontsize=14,
            fontweight="bold",
        )

        # Residuals
        if "residuals" in results:
            residuals = results["residuals"]
            iterations = residuals.get("iterations", [])

            for var_name, values in residuals.items():
                if var_name != "iterations":
                    axes[0].semilogy(iterations, values, linewidth=2, label=var_name)

            axes[0].set_xlabel("Iteration")
            axes[0].set_ylabel("Residual")
            axes[0].set_title("Residual Convergence")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

        # Forces convergence
        if "forces" in results:
            forces = results["forces"]
            iterations = forces.get("iterations", [])

            for force_name, values in forces.items():
                if force_name != "iterations":
                    axes[1].plot(iterations, values, linewidth=2, label=force_name)

            axes[1].set_xlabel("Iteration")
            axes[1].set_ylabel("Force Coefficient")
            axes[1].set_title("Force Convergence")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

        fig.tight_layout()
        return fig

    def _generate_comparison_chart(self, results: Dict[str, Any], **options) -> Figure:
        """Generate comparison chart for multiple cases."""
        fig, axes = plt.subplots(2, 2, figsize=options.get("figsize", (12, 10)))
        fig.suptitle(
            options.get("title", "Case Comparison"),
            fontsize=14,
            fontweight="bold",
        )

        axes = axes.flatten()
        colors = ["b", "r", "g", "m", "c", "y", "k"]

        # Compare multiple cases
        cases = results.get("cases", {})

        for i, (case_name, case_data) in enumerate(cases.items()):
            color = colors[i % len(colors)]
            alpha = case_data.get("alpha", [])
            cl = case_data.get("cl", [])
            cd = case_data.get("cd", [])

            # CL vs Alpha
            axes[0].plot(
                alpha,
                cl,
                f"{color}-o",
                linewidth=2,
                markersize=4,
                label=case_name,
            )

            # CD vs Alpha
            axes[1].plot(
                alpha,
                cd,
                f"{color}-o",
                linewidth=2,
                markersize=4,
                label=case_name,
            )

            # CL vs CD
            axes[2].plot(
                cd,
                cl,
                f"{color}-o",
                linewidth=2,
                markersize=4,
                label=case_name,
            )

            # L/D vs Alpha
            if len(cl) == len(cd) and len(cd) > 0:
                ld_ratio = np.array(cl) / np.array(cd)
                axes[3].plot(
                    alpha,
                    ld_ratio,
                    f"{color}-o",
                    linewidth=2,
                    markersize=4,
                    label=case_name,
                )

        # Set labels and titles
        axes[0].set_xlabel("Alpha (deg)")
        axes[0].set_ylabel("CL")
        axes[0].set_title("Lift Coefficient Comparison")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].set_xlabel("Alpha (deg)")
        axes[1].set_ylabel("CD")
        axes[1].set_title("Drag Coefficient Comparison")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        axes[2].set_xlabel("CD")
        axes[2].set_ylabel("CL")
        axes[2].set_title("Drag Polar Comparison")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        axes[3].set_xlabel("Alpha (deg)")
        axes[3].set_ylabel("L/D")
        axes[3].set_title("L/D Ratio Comparison")
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()

        fig.tight_layout()
        return fig

    def _generate_sensitivity_chart(self, results: Dict[str, Any], **options) -> Figure:
        """Generate sensitivity analysis chart."""
        fig, axes = plt.subplots(2, 2, figsize=options.get("figsize", (12, 10)))
        fig.suptitle(
            options.get("title", "Sensitivity Analysis"),
            fontsize=14,
            fontweight="bold",
        )

        axes = axes.flatten()

        # Extract sensitivity data
        parameter = results.get("parameter", "Parameter")
        parameter_values = results.get("parameter_values", [])

        # Plot sensitivity of different outputs
        outputs = ["cl_max", "cd_min", "ld_max", "cm_zero"]
        output_labels = ["CL Max", "CD Min", "L/D Max", "CM at Zero Lift"]

        for i, (output, label) in enumerate(zip(outputs, output_labels)):
            if output in results:
                values = results[output]
                axes[i].plot(parameter_values, values, "bo-", linewidth=2, markersize=6)
                axes[i].set_xlabel(parameter)
                axes[i].set_ylabel(label)
                axes[i].set_title(f"{label} Sensitivity")
                axes[i].grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def get_supported_chart_types(self) -> List[str]:
        """Get list of supported chart types.

        Returns:
            List of supported chart type names
        """
        return [
            "airfoil_polar",
            "airplane_polar",
            "pressure_distribution",
            "geometry",
            "convergence",
            "comparison",
            "sensitivity",
        ]

    def get_chart_template(self, chart_type: str) -> Optional[Dict[str, Any]]:
        """Get chart template information.

        Args:
            chart_type: Type of chart

        Returns:
            Template dictionary or None if not found
        """
        return self.chart_templates.get(chart_type)
