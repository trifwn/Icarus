from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Button
from matplotlib.widgets import Slider
from numpy import ndarray
from pandas import DataFrame
from pandas import Series

from ICARUS.Core.struct import Struct
from ICARUS.Database import DB
from ICARUS.Vehicle.plane import Airplane
from ICARUS.Visualization import colors_
from ICARUS.Visualization import markers
from ICARUS.Visualization.airplane.db_polars import plot_airplane_polars


def cg_investigation(
    airplane_name: str,
    solvers: list[str] = ["All"],
    size: tuple[int, int] = (10, 10),
    title: str = "Aero Coefficients",
) -> None:
    plots: list[list[str]] = [["AoA", "CL"], ["AoA", "CD"], ["AoA", "Cm"], ["CL", "CD"]]
    axs, fig = plot_airplane_polars([airplane_name], solvers, plots, size, title)

    # Get the plane from the database
    plane: Airplane = DB.vehicles_db.planes[airplane_name]
    cg_x: float = plane.CG[1]

    # Create a slider to change the CG
    ax_cg = fig.add_axes([0.25, 0.1, 0.65, 0.03])

    cg_slider = Slider(ax=ax_cg, label="CG", valmin=-1, valmax=1, valinit=cg_x)

    # The function to be called anytime a slider's value changes
    def update(val: float) -> None:
        """
        The function to be called anytime a slider's value changes
        Each time the slider is changed, the cg position is updated and the plot is redrawn
        All the forces and moments are recalculated.

        Args:
            val (float): The new cg position in the x direction
        """
        # Remove all the plots
        ax = axs.flatten()[2]
        ax.clear()
        fig.canvas.draw_idle()

    # register the update function with each slider
    cg_slider.on_changed(update)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, "Reset", hovercolor="0.975")

    def reset(event: Any) -> None:
        cg_slider.reset()

    button.on_clicked(reset)
