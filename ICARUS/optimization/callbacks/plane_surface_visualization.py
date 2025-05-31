from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from . import OptimizationCallback

if TYPE_CHECKING:
    from ICARUS.vehicle import Airplane


class PlaneSurfaceVisualizer(OptimizationCallback):
    """Class to visualize the design variables change during optimization."""

    def __init__(
        self,
        plane: Airplane,
        surface_name: str,
    ):
        """Initialize the class."""
        self.initial_plane = plane
        self.surface_name = surface_name

    def setup(self) -> None:
        """Create a 3D plot to visualize the plane geometry."""
        self.fig = plt.figure(figsize=(10, 10))
        ax: Axes = self.fig.subplots(1)  # type: ignore
        self.ax = ax
        self.fig.show()

        # Add a plot for the initial plane geometry
        for surface in self.initial_plane.wings:
            print(surface.name)
            if surface.name == self.surface_name:
                surf = surface

        chords = surf._chord_dist
        spans = surf._span_dist

        ax.plot(spans, chords, "b--", label="Initial Plane Geometry")
        ax.plot(spans, np.zeros_like(spans), "b--")
        ax.plot(
            np.ones_like(spans) * spans[-1],
            np.linspace(0, chords[-1], spans.shape[0]),
            "b--",
        )
        ax.plot(
            np.ones_like(spans) * spans[0],
            np.linspace(0, chords[0], spans.shape[0]),
            "b--",
        )
        ax.relim()
        ax.set_aspect("equal", "box")

        # Add a l
        self.up_line = ax.plot(spans, chords, color="r", label="Current Plane Geometry")
        self.down_line = ax.plot(spans, np.zeros_like(spans), "r")
        self.left_line = ax.plot(
            np.ones_like(spans) * spans[-1],
            np.linspace(0, chords[-1], spans.shape[0]),
            color="r",
        )
        self.right_line = ax.plot(
            np.ones_like(spans) * spans[0],
            np.linspace(0, chords[0], spans.shape[0]),
            color="r",
        )

        ax.relim()
        ax.legend()
        # Update the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, plane: Airplane, *args: Any, **kwargs: Any) -> None:
        """Update the visualization."""
        # Update the current plane geometry
        # Add a plot for the initial plane geometry
        for surface in plane.wings:
            print(surface.name)
            if surface.name == self.surface_name:
                surf = surface

        chords = surf._chord_dist
        spans = surf._span_dist

        xdata, ydata = self.up_line[0].get_data()
        xdata = spans
        ydata = chords
        self.up_line[0].set_data(xdata, ydata)

        xdata, ydata = self.down_line[0].get_data()
        xdata = spans
        ydata = np.zeros_like(spans)
        self.down_line[0].set_data(xdata, ydata)

        xdata, ydata = self.left_line[0].get_data()
        xdata = np.ones_like(spans) * spans[-1]
        ydata = np.linspace(0, chords[-1], spans.shape[0])
        self.left_line[0].set_data(xdata, ydata)

        xdata, ydata = self.right_line[0].get_data()
        xdata = np.ones_like(spans) * spans[0]
        ydata = np.linspace(0, chords[0], spans.shape[0])
        self.right_line[0].set_data(xdata, ydata)

        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.set_aspect("equal", "box")
        # Update the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
