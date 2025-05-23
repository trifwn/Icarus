from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from scipy.optimize import OptimizeResult

from . import OptimizationCallback


class OptimizationProgress(OptimizationCallback):
    """Class to visualize the design variables change during optimization."""

    def __init__(
        self,
        has_jacobian: bool = False,
        has_hessian: bool = False,
    ):
        """Initialize the class."""
        self.lines: dict[str, Line2D] = {}
        self.axes: dict[str, Axes] = {}
        self.has_jacobian = has_jacobian
        self.has_hessian = has_hessian

    def setup(self) -> None:
        """Create a figure to visualize the optimization progress."""
        self.fig = plt.figure(figsize=(10, 10))
        self.fig.show()

        # Add a subplot to show the value of the objective function
        ax = self.fig.add_subplot(1, 1, 1)
        ax.set_title("Objective Function Value")
        ax.set_ylabel("Value")
        ax.set_xlabel("Iteration")
        ax.grid()

        # Add a line to show the value of the objective function
        line = Line2D(
            [],
            [],
            color="orange",
            label="Best Encountered Objective Function Value",
        )
        ax.add_line(line)
        self.lines["Best Objective Function Value"] = line
        self.axes["Best Objective Function Value"] = ax

        line = Line2D(
            [],
            [],
            color="blue",
            label="Current Objective Function Value",
        )
        ax.add_line(line)
        self.lines["Current Objective Function Value"] = line
        self.axes["Current Objective Function Value"] = ax

        # Add plot for penalty function
        line = Line2D(
            [],
            [],
            color="red",
            label="Current Penalty Function Value",
        )
        ax.add_line(line)
        self.lines["Current Penalty Function Value"] = line
        self.axes["Current Penalty Function Value"] = ax

        # Add legend
        ax.legend()

        # If it has a jacobian, add a subplot to show the jacobian norm
        if self.has_jacobian:
            ax = self.fig.add_subplot(2, 1, 1)
            ax.set_title("Jacobian Norm")
            ax.set_ylabel("Value")
            ax.set_xlabel("Iteration")
            ax.grid()

            # Add a line to show the jacobian norm
            line = Line2D(
                [],
                [],
                color="orange",
                label="Initial Value",
            )
            ax.add_line(line)
            self.lines["Jacobian Norm"] = line
            self.axes["Jacobian Norm"] = ax
            # Add legend
            ax.legend()

        # If it has a hessian, add a subplot to show the hessian norm
        if self.has_hessian:
            ax = self.fig.add_subplot(3, 1, 1)
            ax.set_title("Hessian Norm")
            ax.set_ylabel("Value")
            ax.set_xlabel("Iteration")
            ax.grid()

            # Add a line to show the hessian norm
            line = Line2D(
                [],
                [],
                color="orange",
                label="Initial Value",
            )
            ax.add_line(line)
            self.lines["Hessian Norm"] = line
            self.axes["Hessian Norm"] = ax
            # Add legend
            ax.legend()

        # Update the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(
        self,
        result: OptimizeResult,
        iteration: int,
        fitness: float,
        penalty: float,
        **kwargs: Any,
    ) -> None:
        """Update the figure with the current design variables."""
        if not hasattr(self, "fig") or not hasattr(self, "lines"):
            self.setup()

        # Update the objective function value
        line = self.lines["Best Objective Function Value"]
        ax = self.axes["Best Objective Function Value"]
        try:
            # Get the data of the line
            xdata, ydata = line.get_data()
            print(xdata, ydata)
            if result.fun > 1e10:
                return
            # Append the new data
            xdata = np.append(xdata, iteration)
            ydata = np.append(ydata, result.fun)
            # Set the new data of the line
            line.set_data(xdata, ydata)
        except AttributeError:
            pass

        # Update the best objective function value
        line = self.lines["Current Objective Function Value"]
        ax = self.axes["Current Objective Function Value"]
        xdata, ydata = line.get_data()
        xdata = np.append(xdata, iteration)
        ydata = np.append(ydata, fitness)
        line.set_data(xdata, ydata)

        # Enable log scale if the values are too far apart
        if np.max(ydata) / np.min(ydata) > 100:
            ax.set_yscale("log")

        # Update the penalty function value
        line = self.lines["Current Penalty Function Value"]
        ax = self.axes["Current Penalty Function Value"]
        xdata, ydata = line.get_data()
        xdata = np.append(xdata, iteration)
        ydata = np.append(ydata, penalty)
        line.set_data(xdata, ydata)

        ax.relim()
        ax.autoscale()

        # If it has a jacobian, update the jacobian norm
        if self.has_jacobian:
            line = self.lines["Jacobian Norm"]
            ax = self.axes["Jacobian Norm"]

            # Get the data of the line
            xdata, ydata = line.get_data()
            # Append the new data
            xdata = np.append(xdata, iteration)
            ydata = np.append(ydata, np.linalg.norm(result.jac))
            # Set the new data of the line
            line.set_data(xdata, ydata)
            ax.relim()
            ax.autoscale()

        # If it has a hessian, update the hessian norm
        if self.has_hessian:
            line = self.lines["Hessian Norm"]
            ax = self.axes["Hessian Norm"]

            # Get the data of the line
            xdata, ydata = line.get_data()
            # Append the new data
            xdata = np.append(xdata, iteration)
            ydata = np.append(ydata, np.linalg.norm(result.hess))
            # Set the new data of the line
            line.set_data(xdata, ydata)
            ax.relim()
            ax.autoscale()

        self.fig.canvas.draw()

        plt.pause(0.1)
