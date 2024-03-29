import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import OptimizeResult

from ICARUS.optimization.callbacks.optimization_callback import OptimizationCallback
from ICARUS.vehicle.plane import Airplane


class PlaneGeometryVisualization(OptimizationCallback):
    """
    Class to visualize the design variables change during optimization.
    """

    def __init__(
        self,
        plane: Airplane,
    ):
        """
        Initialize the class.
        """
        self.initial_plane = plane

    def setup(self) -> None:
        """
        Create a 3D plot to visualize the plane geometry.
        """
        self.fig = plt.figure(figsize=(10, 10))
        self.fig.show()

        # Add a subplot to show the value of the objective function
        ax: Axes3D = self.fig.add_subplot(1, 1, 1, projection="3d")  # type: ignore
        ax.set_title("Initial Plane Geometry")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.grid()

        # Store the axes
        self.initial_ax = ax

        # Add the plane geometry
        self.initial_plane.visualize(self.fig, self.initial_ax)
        # Add legend
        ax.legend()

        # Add a subplot to show the current plane geometry
        ax: Axes3D = self.fig.add_subplot(2, 1, 1, projection="3d")  # type: ignore
        ax.set_title("Current Plane Geometry")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.grid()

        # Store the axes
        self.current_ax = ax
        # Add the plane geometry
        self.initial_plane.visualize(self.fig, self.current_ax)
        # Add legend
        ax.legend()

        # Update the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, plane: Airplane) -> None:
        """
        Update the visualization.
        """
        # Update the current plane geometry
        plane.visualize(self.fig, self.current_ax)
        # Update the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
