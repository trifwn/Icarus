"""Interactive Plotter - Creates interactive plots with customization options

This module provides interactive plotting capabilities for the ICARUS CLI visualization system.
It supports various plot types and provides real-time interaction capabilities.
"""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rich.console import Console


class InteractivePlotter:
    """Creates and manages interactive plots."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize the interactive plotter.

        Args:
            console: Rich console for output (optional)
        """
        self.console = console or Console()
        self.plot_styles = {
            "aerospace": {
                "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
                "linestyles": ["-", "--", "-.", ":"],
                "markers": ["o", "s", "^", "v", "D"],
            },
            "publication": {
                "colors": ["#000000", "#333333", "#666666", "#999999", "#cccccc"],
                "linestyles": ["-", "--", "-.", ":"],
                "markers": ["o", "s", "^", "v", "D"],
            },
            "colorful": {
                "colors": ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"],
                "linestyles": ["-", "--", "-.", ":"],
                "markers": ["o", "s", "^", "v", "D"],
            },
        }

    def create_plot(
        self,
        data: Dict[str, Any],
        plot_type: str = "line",
        title: Optional[str] = None,
        style: str = "aerospace",
        **kwargs,
    ) -> Figure:
        """Create an interactive plot.

        Args:
            data: Data to plot (format depends on plot_type)
            plot_type: Type of plot (line, scatter, bar, polar, contour, etc.)
            title: Plot title
            style: Plot style theme
            **kwargs: Additional plot parameters

        Returns:
            Matplotlib Figure object
        """
        try:
            # Create figure and axis
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 6)))

            # Apply style
            self._apply_style(fig, ax, style)

            # Create the appropriate plot type
            if plot_type == "line":
                self._create_line_plot(ax, data, style, **kwargs)
            elif plot_type == "scatter":
                self._create_scatter_plot(ax, data, style, **kwargs)
            elif plot_type == "bar":
                self._create_bar_plot(ax, data, style, **kwargs)
            elif plot_type == "polar":
                self._create_polar_plot(fig, data, style, **kwargs)
            elif plot_type == "contour":
                self._create_contour_plot(ax, data, style, **kwargs)
            elif plot_type == "surface":
                self._create_surface_plot(fig, data, style, **kwargs)
            elif plot_type == "airfoil_polar":
                self._create_airfoil_polar_plot(fig, data, style, **kwargs)
            elif plot_type == "pressure_distribution":
                self._create_pressure_plot(ax, data, style, **kwargs)
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")

            # Set title
            if title:
                fig.suptitle(title, fontsize=14, fontweight="bold")

            # Add grid and formatting
            if hasattr(ax, "grid"):
                ax.grid(True, alpha=0.3)

            # Tight layout
            fig.tight_layout()

            self.console.print(f"[green]✓[/green] Created {plot_type} plot")
            return fig

        except Exception as e:
            self.console.print(f"[red]✗[/red] Failed to create plot: {e}")
            raise

    def _apply_style(self, fig: Figure, ax: Axes, style: str) -> None:
        """Apply styling to the plot."""
        if style in self.plot_styles:
            # Set background colors
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            # Set spine colors
            for spine in ax.spines.values():
                spine.set_color("#333333")
                spine.set_linewidth(0.8)

    def _create_line_plot(
        self,
        ax: Axes,
        data: Dict[str, Any],
        style: str,
        **kwargs,
    ) -> None:
        """Create a line plot."""
        x_data = data.get("x", [])
        y_data = data.get("y", [])

        if isinstance(y_data, dict):
            # Multiple series
            colors = self.plot_styles[style]["colors"]
            linestyles = self.plot_styles[style]["linestyles"]

            for i, (label, y_values) in enumerate(y_data.items()):
                color = colors[i % len(colors)]
                linestyle = linestyles[i % len(linestyles)]

                ax.plot(
                    x_data,
                    y_values,
                    color=color,
                    linestyle=linestyle,
                    linewidth=kwargs.get("linewidth", 2),
                    label=label,
                )

            ax.legend()
        else:
            # Single series
            ax.plot(
                x_data,
                y_data,
                color=self.plot_styles[style]["colors"][0],
                linewidth=kwargs.get("linewidth", 2),
            )

        # Set labels
        ax.set_xlabel(data.get("xlabel", "X"))
        ax.set_ylabel(data.get("ylabel", "Y"))

    def _create_scatter_plot(
        self,
        ax: Axes,
        data: Dict[str, Any],
        style: str,
        **kwargs,
    ) -> None:
        """Create a scatter plot."""
        x_data = data.get("x", [])
        y_data = data.get("y", [])

        colors = self.plot_styles[style]["colors"]
        markers = self.plot_styles[style]["markers"]

        if isinstance(y_data, dict):
            # Multiple series
            for i, (label, y_values) in enumerate(y_data.items()):
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]

                ax.scatter(
                    x_data,
                    y_values,
                    color=color,
                    marker=marker,
                    s=kwargs.get("markersize", 50),
                    alpha=kwargs.get("alpha", 0.7),
                    label=label,
                )

            ax.legend()
        else:
            # Single series
            ax.scatter(
                x_data,
                y_data,
                color=colors[0],
                marker=markers[0],
                s=kwargs.get("markersize", 50),
                alpha=kwargs.get("alpha", 0.7),
            )

        # Set labels
        ax.set_xlabel(data.get("xlabel", "X"))
        ax.set_ylabel(data.get("ylabel", "Y"))

    def _create_bar_plot(
        self,
        ax: Axes,
        data: Dict[str, Any],
        style: str,
        **kwargs,
    ) -> None:
        """Create a bar plot."""
        x_data = data.get("x", [])
        y_data = data.get("y", [])

        colors = self.plot_styles[style]["colors"]

        if isinstance(y_data, dict):
            # Multiple series (grouped bars)
            x_pos = np.arange(len(x_data))
            width = 0.8 / len(y_data)

            for i, (label, y_values) in enumerate(y_data.items()):
                offset = (i - len(y_data) / 2 + 0.5) * width
                ax.bar(
                    x_pos + offset,
                    y_values,
                    width=width,
                    color=colors[i % len(colors)],
                    alpha=kwargs.get("alpha", 0.8),
                    label=label,
                )

            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_data)
            ax.legend()
        else:
            # Single series
            ax.bar(x_data, y_data, color=colors[0], alpha=kwargs.get("alpha", 0.8))

        # Set labels
        ax.set_xlabel(data.get("xlabel", "X"))
        ax.set_ylabel(data.get("ylabel", "Y"))

    def _create_polar_plot(
        self,
        fig: Figure,
        data: Dict[str, Any],
        style: str,
        **kwargs,
    ) -> None:
        """Create a polar plot."""
        # Clear the figure and create polar subplot
        fig.clear()
        ax = fig.add_subplot(111, projection="polar")

        theta = data.get("theta", [])
        r = data.get("r", [])

        colors = self.plot_styles[style]["colors"]

        if isinstance(r, dict):
            # Multiple series
            for i, (label, r_values) in enumerate(r.items()):
                ax.plot(
                    theta,
                    r_values,
                    color=colors[i % len(colors)],
                    linewidth=kwargs.get("linewidth", 2),
                    label=label,
                )
            ax.legend()
        else:
            # Single series
            ax.plot(theta, r, color=colors[0], linewidth=kwargs.get("linewidth", 2))

        # Set labels
        ax.set_title(data.get("title", "Polar Plot"))

    def _create_contour_plot(
        self,
        ax: Axes,
        data: Dict[str, Any],
        style: str,
        **kwargs,
    ) -> None:
        """Create a contour plot."""
        X = data.get("X", [])
        Y = data.get("Y", [])
        Z = data.get("Z", [])

        levels = kwargs.get("levels", 20)

        if kwargs.get("filled", True):
            contour = ax.contourf(X, Y, Z, levels=levels, cmap="viridis")
            plt.colorbar(contour, ax=ax)
        else:
            contour = ax.contour(X, Y, Z, levels=levels, colors="black")
            ax.clabel(contour, inline=True, fontsize=8)

        # Set labels
        ax.set_xlabel(data.get("xlabel", "X"))
        ax.set_ylabel(data.get("ylabel", "Y"))

    def _create_surface_plot(
        self,
        fig: Figure,
        data: Dict[str, Any],
        style: str,
        **kwargs,
    ) -> None:
        """Create a 3D surface plot."""
        # Clear the figure and create 3D subplot
        fig.clear()
        ax = fig.add_subplot(111, projection="3d")

        X = data.get("X", [])
        Y = data.get("Y", [])
        Z = data.get("Z", [])

        surface = ax.plot_surface(
            X,
            Y,
            Z,
            cmap="viridis",
            alpha=kwargs.get("alpha", 0.8),
        )

        fig.colorbar(surface, ax=ax, shrink=0.5)

        # Set labels
        ax.set_xlabel(data.get("xlabel", "X"))
        ax.set_ylabel(data.get("ylabel", "Y"))
        ax.set_zlabel(data.get("zlabel", "Z"))

    def _create_airfoil_polar_plot(
        self,
        fig: Figure,
        data: Dict[str, Any],
        style: str,
        **kwargs,
    ) -> None:
        """Create an airfoil polar plot (CL vs CD, CL vs Alpha, etc.)."""
        # Create subplots for multiple polar curves
        fig.clear()

        # Determine subplot layout
        num_plots = len(data.get("plots", ["cl_cd", "cl_alpha", "cd_alpha"]))
        if num_plots <= 2:
            rows, cols = 1, num_plots
        elif num_plots <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 3, 2

        colors = self.plot_styles[style]["colors"]

        for i, plot_type in enumerate(
            data.get("plots", ["cl_cd", "cl_alpha", "cd_alpha"]),
        ):
            ax = fig.add_subplot(rows, cols, i + 1)

            if plot_type == "cl_cd":
                x_data = data.get("cd", [])
                y_data = data.get("cl", [])
                ax.set_xlabel("CD")
                ax.set_ylabel("CL")
                ax.set_title("CL vs CD")
            elif plot_type == "cl_alpha":
                x_data = data.get("alpha", [])
                y_data = data.get("cl", [])
                ax.set_xlabel("Alpha (deg)")
                ax.set_ylabel("CL")
                ax.set_title("CL vs Alpha")
            elif plot_type == "cd_alpha":
                x_data = data.get("alpha", [])
                y_data = data.get("cd", [])
                ax.set_xlabel("Alpha (deg)")
                ax.set_ylabel("CD")
                ax.set_title("CD vs Alpha")

            ax.plot(
                x_data,
                y_data,
                color=colors[i % len(colors)],
                linewidth=2,
                marker="o",
                markersize=4,
            )
            ax.grid(True, alpha=0.3)

        fig.tight_layout()

    def _create_pressure_plot(
        self,
        ax: Axes,
        data: Dict[str, Any],
        style: str,
        **kwargs,
    ) -> None:
        """Create a pressure distribution plot."""
        x_data = data.get("x", [])  # Chord position
        cp_upper = data.get("cp_upper", [])  # Upper surface Cp
        cp_lower = data.get("cp_lower", [])  # Lower surface Cp

        colors = self.plot_styles[style]["colors"]

        # Plot upper and lower surface pressure
        ax.plot(x_data, cp_upper, color=colors[0], linewidth=2, label="Upper Surface")
        ax.plot(x_data, cp_lower, color=colors[1], linewidth=2, label="Lower Surface")

        # Invert y-axis (typical for pressure plots)
        ax.invert_yaxis()

        # Set labels and legend
        ax.set_xlabel("x/c")
        ax.set_ylabel("Cp")
        ax.set_title("Pressure Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def update_plot_data(self, fig: Figure, new_data: Dict[str, Any]) -> None:
        """Update plot data for real-time plotting.

        Args:
            fig: Figure to update
            new_data: New data to plot
        """
        try:
            # Get the first axis (assuming single plot for now)
            ax = fig.axes[0]

            # Update line data
            for line, (key, values) in zip(ax.lines, new_data.items()):
                if key in ["x", "y"] and isinstance(values, (list, np.ndarray)):
                    if key == "y":
                        line.set_ydata(values)
                    elif key == "x":
                        line.set_xdata(values)

            # Rescale axes
            ax.relim()
            ax.autoscale_view()

            # Redraw
            fig.canvas.draw()
            fig.canvas.flush_events()

        except Exception as e:
            self.console.print(f"[red]✗[/red] Failed to update plot: {e}")

    def get_supported_plot_types(self) -> List[str]:
        """Get list of supported plot types.

        Returns:
            List of supported plot type names
        """
        return [
            "line",
            "scatter",
            "bar",
            "polar",
            "contour",
            "surface",
            "airfoil_polar",
            "pressure_distribution",
        ]

    def get_plot_info(self, fig: Figure) -> Dict[str, Any]:
        """Get information about a plot.

        Args:
            fig: Figure to analyze

        Returns:
            Dictionary with plot information
        """
        info = {
            "num_axes": len(fig.axes),
            "figure_size": fig.get_size_inches().tolist(),
            "title": fig._suptitle.get_text() if fig._suptitle else None,
            "axes_info": [],
        }

        for ax in fig.axes:
            ax_info = {
                "xlabel": ax.get_xlabel(),
                "ylabel": ax.get_ylabel(),
                "title": ax.get_title(),
                "num_lines": len(ax.lines),
                "has_legend": ax.get_legend() is not None,
                "xlim": ax.get_xlim(),
                "ylim": ax.get_ylim(),
            }
            info["axes_info"].append(ax_info)

        return info
