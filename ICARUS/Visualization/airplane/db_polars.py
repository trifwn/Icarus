from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame
from pandas import Series

from ICARUS.Database import DB
from ICARUS.Visualization import colors_
from ICARUS.Visualization import markers


def plot_airplane_polars(
    airplane_names: list[str],
    solvers: list[str] = ["All"],
    plots: list[list[str]] = [["AoA", "CL"], ["AoA", "CD"], ["AoA", "Cm"], ["CL", "CD"]],
    size: tuple[int, int] = (10, 10),
    title: str = "Aero Coefficients",
) -> tuple[ndarray[Any, Any], Figure]:
    """Function to plot airplane polars for a given list of airplanes and solvers

    Args:
        data (dict[str, DataFrame]): Dictionary of airplane polars
        airplanes (list[str]): List of airplanes to plot
        solvers (list[str], optional): List of Solvers to plot. Defaults to ["All"].
        plots (list[list[str]], optional): List of plots to plot. Defaults to [["AoA", "CL"], ["AoA", "CD"], ["AoA", "Cm"], ["CL", "CD"]].
        size (tuple[int, int], optional): Figure Size. Defaults to (10, 10).
        title (str, optional): Figure Title. Defaults to "Aero Coefficients".

    Returns:
        tuple[ndarray, Figure]: Array of Axes and Figure
    """
    number_of_plots = len(plots) + 1

    # Divide the plots equally
    sqrt_num = number_of_plots**0.5
    i: int = int(np.ceil(sqrt_num))
    j: int = int(np.floor(sqrt_num))

    fig: Figure = plt.figure(figsize=size)
    axs: ndarray[Axes] = fig.subplots(i, j)  # type: ignore
    fig.suptitle(f"{title}", fontsize=16)

    for plot, ax in zip(plots, axs.flatten()[: len(plots)]):
        ax.set_xlabel(plot[0])
        ax.set_ylabel(plot[1])
        ax.set_title(f"{plot[1]} vs {plot[0]}")
        ax.grid()
        ax.axhline(y=0, color="k")
        ax.axvline(x=0, color="k")

    if solvers == ["All"]:
        solvers = ["GNVP3 Potential", "GNVP3 2D", "GNVP7 Potential", "GNVP7 2D", "LSPT Potential", "LSPT 2D"]

    for i, airplane in enumerate(airplane_names):
        flag = False
        for j, solver in enumerate(solvers):
            try:
                polar: DataFrame = DB.vehicles_db.data[airplane]
                for plot, ax in zip(plots, axs.flatten()[: len(plots)]):
                    if airplane.startswith("XFLR"):
                        key0 = f"{plot[0]}"
                        key1 = f"{plot[1]}"
                        ax.plot(
                            polar[f"{key0}"],
                            polar[f"{key1}"],
                            label=f"{airplane} XFLR",
                            markersize=1.5,
                            color="m",
                            linewidth=1,
                        )
                        flag = True

                    else:
                        key0 = f"{solver} {plot[0]}"
                        key1 = f"{solver} {plot[1]}"

                        if plot[0] == "AoA":
                            key0 = "AoA"
                        if plot[1] == "AoA":
                            key1 = "AoA"

                        x: Series = polar[f"{key0}"]
                        y: Series = polar[f"{key1}"]
                        c = colors_(j / len(solvers))
                        m = markers[i].get_marker()
                        label: str = f"{airplane} - {solver}"
                        try:
                            ax.plot(x, y, ls="--", color=c, marker=m, label=label, markersize=3.5, linewidth=1)
                        except ValueError as e:
                            raise e
                if flag:
                    break
            except KeyError as e:
                print(f"Run Doesn't Exist: {airplane},{e} ")

    # In the plots we created there is either one or two empty plots
    # depending on the number of plots we demanded
    # We need to remove the empty plots and add the legend to
    # the empty space

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
