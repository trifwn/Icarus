from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame
from pandas import Series

from ICARUS.core.struct import Struct
from ICARUS.database import DB

from .. import colors_
from .. import markers


def plot_airfoil_polars(
    airfoil_name: str,
    solvers: list[str] | str = "All",
    plots: list[list[str]] = [
        ["AoA", "CL"],
        ["AoA", "CD"],
        ["AoA", "Cm"],
        ["CD", "CL"],
    ],
    size: tuple[int, int] = (10, 10),
    aoa_bounds: list[float] | None = None,
    title: str = "Aero Coefficients",
) -> tuple[ndarray[Any, Any], Figure]:
    """
    Args:
        airfoil_name (str): Airfoil names
        solvers (list[str] | str, optional): List of solver Names. Defaults to "All".
        plots (list[list[str]], optional): List of plots to plot. Defaults to [["AoA", "CL"], ["AoA", "CD"], ["AoA", "Cm"], ["CL", "CD"]].
        size (tuple[int, int], optional): Fig Size. Defaults to (10, 10).
        aoa_bounds (_type_, optional): Angle of Attack Bounds. Defaults to None.
        title (str, optional): Figure Title. Defaults to "Aero Coefficients".
    """
    number_of_plots = len(plots) + 1

    # Divide the plots equally
    sqrt_num = number_of_plots**0.5
    i: int = int(np.ceil(sqrt_num))
    j: int = int(np.floor(sqrt_num))

    fig: Figure = plt.figure(figsize=size)
    axs: ndarray = fig.subplots(j, i)  # type: ignore

    fig.suptitle(f"{title}", fontsize=16)

    for plot, ax in zip(plots, axs.flatten()[: len(plots)]):
        ax.set_xlabel(plot[0])
        ax.set_ylabel(plot[1])
        ax.set_title(f"{plot[1]} vs {plot[0]}")
        ax.grid()
        ax.axhline(y=0, color="k")
        ax.axvline(x=0, color="k")

    if solvers == "All" or solvers == ["All"]:
        solvers = ["Xfoil", "Foil2Wake", "OpenFoam", "XFLR"]

    # Get the data from the database
    data: Struct = DB.foils_db._raw_data
    db_solvers = data[airfoil_name]

    for i, solver in enumerate(db_solvers):
        if solver not in solvers:
            print(f"Skipping {solver} is not in {solvers}")
            continue

        for j, reynolds in enumerate(db_solvers[solver].keys()):
            print(reynolds, solver)
            try:
                polar: DataFrame = db_solvers[solver][reynolds]

                # Sort the data by AoA
                polar = polar.sort_values(by="AoA")
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

                    x: Series[float] = polar[f"{key0}"]
                    y: Series[float] = polar[f"{key1}"]
                    c = colors_(j / len(db_solvers[solver].keys()))
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
            except (KeyError, ValueError) as solv:
                print(f"Run Doesn't Exist: {airfoil_name},{reynolds},{solv}")

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
