from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from numpy import ndarray
from pandas import Series

from ICARUS.Core.struct import Struct
from ICARUS.Database import DB
from ICARUS.Visualization import colors_
from ICARUS.Visualization import markers


def plot_airfoil_reynolds(
    airfoil_name: str,
    reynolds: str,
    solvers: list[str] = ["All"],
    plots: list[list[str]] = [
        ["AoA", "CL"],
        ["AoA", "CD"],
        ["AoA", "Cm"],
        ["CL", "CD"],
    ],
    size: tuple[int, int] = (10, 10),
    aoa_bounds: list[float] | None = None,
    title: str = "Aero Coefficients",
) -> tuple[ndarray[Any, Any], Figure]:
    """Function to plot airfoil polars for a given list of airfoils and solvers.

    Args:
        airfoil_name (str): Airfoil name (e.g. naca0012)
        reynolds (str): Reynolds number (e.g. 100000)
        solvers (list[str], optional): Can be either all or individual solver names. Defaults to ["All"].
        plots (list[list[str]], optional): List of plots to plot. Defaults to [["AoA", "CL"], ["AoA", "CD"], ["AoA", "Cm"], ["CL", "CD"]].
        size (tuple[int, int], optional): Figure Size. Defaults to (10, 10).
        aoa_bounds (_type_, optional): Angle of Attack Bounds. Defaults to None.
        title (str, optional): Figure Title. Defaults to "Aero Coefficients".
    """

    # Get the number of plots
    number_of_plots = len(plots) + 1

    # Divide the plots equally
    sqrt_num = number_of_plots**0.5
    i: int = int(np.ceil(sqrt_num))
    j: int = int(np.floor(sqrt_num))

    fig: Figure = plt.figure(figsize=size)
    axs: ndarray = fig.subplots(j, i)  # type: ignore

    if title == "Aero Coefficients":
        fig.suptitle(
            f"NACA {airfoil_name[4:]}- Reynolds={reynolds}\n Aero Coefficients",
            fontsize=16,
        )
    else:
        fig.suptitle(title, fontsize=16)

    # Get the solvers
    if solvers == ["All"]:
        solvers = ["Xfoil", "Foil2Wake", "OpenFoam", "XFLR"]

    # Get the data from the database
    data: Struct = DB.foils_db._data

    for j, solver in enumerate(solvers):
        try:
            polar = data[airfoil_name][solver][reynolds]
            if aoa_bounds is not None:
                # Get data where AoA is in AoA bounds
                polar = polar.loc[(polar["AoA"] >= aoa_bounds[0]) & (polar["AoA"] <= aoa_bounds[1])]
            for plot, ax in zip(plots, axs.flatten()[: len(plots)]):
                key0 = f"{plot[0]}"
                key1 = f"{plot[1]}"

                if plot[0] == "AoA":
                    key0 = "AoA"
                if plot[1] == "AoA":
                    key1 = "AoA"

                x: Series = polar[f"{key0}"]
                y: Series = polar[f"{key1}"]
                c = colors_(j / len(solvers))
                m = markers[i].get_marker()
                label: str = f"{airfoil_name}: {reynolds} - {solver}"
                try:
                    ax.plot(
                        x,
                        y,
                        ls="--",
                        color=c,
                        marker=m,
                        label=label,
                        markersize=3.5,
                        linewidth=1,
                    )
                except ValueError as e:
                    raise e
        except KeyError as solver:
            print(f"Run Doesn't Exist: {airfoil_name},{reynolds},{solver}")

    # Remove empty plots
    for ax in axs.flatten()[len(plots) :]:
        ax.remove()

    # Take the legend of all plots (they are the same) and add them to the empty space below
    # where we removed the empty plots
    handles, labels = axs.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncol=2)

    # Adjust the plots
    fig.tight_layout()
    fig.subplots_adjust(top=0.9, bottom=0.1)

    plt.show()
    return axs, fig
