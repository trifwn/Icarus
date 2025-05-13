from typing import Any

import distinctipy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame
from pandas import Series

from ICARUS.database import Database
from ICARUS.vehicle.airplane import Airplane
from ICARUS.visualization import markers


def plot_airplane_polars(
    airplanes: list[str] | list[Airplane] | str | Airplane,
    prefixes: list[str] = ["All"],
    plots: list[list[str]] = [
        ["AoA", "CL"],
        ["AoA", "CD"],
        ["AoA", "Cm"],
        ["AoA", "CL/CD"],
    ],
    size: tuple[float, float] = (10.0, 10.0),
    title: str = "Aerodynamic Coefficients",
    operating_point: dict[str, float] = {},
) -> tuple[ndarray[Any, Any], Figure]:
    """Function to plot airplane polars for a given list of airplanes and solvers

    Args:
        airplanes (list[str] | list[Airplane] | str | Airplane): List of airplanes to plot.
        prefixes (list[str], optional): List of solvers to plot. Defaults to ["All"].
        plots (list[list[str]], optional): List of plots to plot. Defaults to [["AoA", "CL"], ["AoA", "CD"], ["AoA", "Cm"], ["CL", "CD"]].
        size (tuple[int, int], optional): Figure Size. Defaults to (10, 10).
        title (str, optional): Figure Title. Defaults to "Aero Coefficients".
        operating_point (dict[str, float], optional): Operating points to plot. Defaults to {}.

    Returns:
        tuple[ndarray, Figure]: Array of Axes and Figure

    """
    if isinstance(airplanes, str) or isinstance(airplanes, Airplane):
        airplanes = [airplanes]

    number_of_plots = len(plots)
    DB = Database.get_instance()
    # Divide the plots equally
    sqrt_num = number_of_plots**0.5
    i: int = int(np.ceil(sqrt_num))
    j: int = int(np.floor(sqrt_num))

    fig: Figure = plt.figure(figsize=size)
    axs = fig.subplots(i, j)  # type: ignore
    fig.suptitle(f"{title}", fontsize=16)

    if isinstance(axs, Axes):
        axs = np.array([axs])
    elif isinstance(axs, list):
        axs = np.array(axs)

    for plot, ax in zip(plots, axs.flatten()[: len(plots)]):
        ax.set_xlabel(plot[0])
        ax.set_ylabel(plot[1])
        ax.set_title(f"{plot[1]} vs {plot[0]}")
        ax.grid()
        ax.axhline(y=0, color="k")
        ax.axvline(x=0, color="k")

    if prefixes == ["All"]:
        prefixes = [
            "GenuVP3 Potential",
            "GenuVP3 2D",
            "GenuVP3 ONERA",
            "GenuVP7 Potential",
            "GenuVP7 2D",
            "LSPT Potential",
            "LSPT 2D",
            "AVL",
        ]

    colors_ = distinctipy.get_colors(len(airplanes) * len(prefixes))
    for i, airplane in enumerate(airplanes):
        if isinstance(airplane, Airplane):
            airplane = airplane.name

        polar: DataFrame = DB.get_vehicle_polars(airplane)
        for j, prefix in enumerate(prefixes):
            if len(airplanes) == 1:
                c = colors_[j]
                m = "o"
            else:
                c = colors_[i]
                m = markers[j].get_marker()

            try:
                for plot, ax in zip(plots, axs.flatten()[: len(plots)]):
                    if plot[0] == "CL/CD" or plot[1] == "CL/CD":
                        polar[f"{prefix} CL/CD"] = polar[f"{prefix} CL"] / polar[f"{prefix} CD"]
                    if plot[0] == "CD/CL" or plot[1] == "CD/CL":
                        polar[f"{prefix} CD/CL"] = polar[f"{prefix} CD"] / polar[f"{prefix} CL"]

                    key0 = f"{prefix} {plot[0]}"
                    key1 = f"{prefix} {plot[1]}"

                    if plot[0] == "AoA":
                        key0 = "AoA"
                    if plot[1] == "AoA":
                        key1 = "AoA"

                    # Drop the NaN values
                    df = polar[[f"{key0}", f"{key1}"]].dropna(axis="index")
                    x: Series[float] = df[f"{key0}"]
                    y: Series[float] = df[f"{key1}"]

                    ax.plot(
                        x,
                        y,
                        color=c,
                        marker=m,
                        markersize=5,
                        linewidth=1,
                        label=f"{airplane} - {prefix}",
                    )

                    if plot[0] == "AoA":
                        # Annotate the operating points in the plots
                        for op in operating_point:
                            ax.axvline(
                                x=operating_point[op],
                                color="r",
                                linestyle="--",
                                label=f"{op}",
                            )
                    elif plot[1] == "AoA":
                        for op in operating_point:
                            ax.axhline(
                                y=operating_point[op],
                                color="r",
                                linestyle="--",
                                label=f"{op}",
                            )
            except KeyError as e:
                print(f"For plane {airplane}: run {e} Does not exist")

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
