from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from . import OptimizationCallback

if TYPE_CHECKING:
    from ICARUS.vehicle import Airplane


class DesignVariableVisualizer(OptimizationCallback):
    """Class to visualize the design variables change during optimization."""

    def __init__(
        self,
        plane: Airplane,
        design_variables: list[str],
        bounds: dict[str, tuple[float, float]],
    ):
        """Inputs:
        design_variables: list[str]
            The list of design variables to visualize.
        bounds: dict[str, tuple[float, float]]
            The bounds of each design variable.
        """
        self.design_variables = {
            name: plane.get_property(name) for name in design_variables
        }
        self.bounds = bounds
        self.axes: dict[str, Axes] = {}
        self.lines: dict[str, Line2D] = {}

    def setup(self) -> None:
        """Setup the figure for visualization. The figure should consist of n
        subplots, where n is the number of design variables. Each subplot
        should contain a line plot of the design variable value, and a line
        plot of the design variable bounds. Additionally, each subplot should
        have a title with the design variable name, and a line with the initial
        design variable value.
        """
        px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches

        self.fig, axes = plt.subplots(
            len(self.design_variables),
            1,
            figsize=(1920 * px / 2, 1000 * px),
            sharex=True,
            gridspec_kw={"hspace": 0.5},
        )
        self.fig.show()

        # Add subplots
        for i, var_name in enumerate(self.design_variables.keys()):
            ax = axes[i]
            ax.set_title(var_name)
            ax.grid(True)

            # Add initial design variable values and bounds
            initial_value = self.design_variables[var_name]
            ax.axhline(
                y=initial_value,
                color="orange",
                linestyle="--",
                label="Initial Value",
            )

            lower, upper = self.bounds[var_name]
            ax.axhline(y=lower, color="red", linestyle="--", label="Lower Bound")
            ax.axhline(y=upper, color="red", linestyle="--", label="Upper Bound")

            # Add lines for design variable values over time
            line = Line2D([], [], color="blue", label="Current Value")
            ax.add_line(line)
            self.lines[var_name] = line
            self.axes[var_name] = ax

        # Set common x-axis label and legend
        self.axes[list(self.design_variables.keys())[-1]].set_xlabel("Iteration")

        # Create a separate legend for names with custom handles and labels
        legend_handles = [
            Line2D([0], [0], color="blue", label="Current Value"),
            Line2D([0], [0], color="red", linestyle="--", label="Upper Bound"),
            Line2D([0], [0], color="orange", linestyle="--", label="Initial Value"),
        ]
        legend_labels = ["Current Value", "Upper Bound", "Initial Value"]
        self.fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.97),
            ncol=3,
        )

        # plt.tight_layout(rect=(0, 0, 1, 0.96))  # Adjust the rect parameter to leave space at the top

        # Update the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(
        self,
        iteration: int,
        design_variables: dict[str, float],
        **kwargs: Any,
    ) -> None:
        """Update the figure with the current design variables."""
        if not hasattr(self, "fig"):
            self.setup()

        for name, value in design_variables.items():
            line: Line2D = self.lines[name]
            ax: Axes = self.axes[name]

            x_data, y_data = line.get_data()

            # Add the current value to the line
            x_data = np.append(x_data, iteration)
            y_data = np.append(y_data, value)

            line.set_xdata(x_data)
            line.set_ydata(y_data)
            ax.relim()
            ax.autoscale()

        self.fig.canvas.draw()
        plt.pause(0.01)
