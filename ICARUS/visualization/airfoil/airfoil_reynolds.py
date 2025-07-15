from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import Series

from ICARUS.airfoils import AirfoilPolarMap
from ICARUS.database import AirfoilNotFoundError
from ICARUS.database import Database
from ICARUS.database import PolarsNotFoundError
from ICARUS.visualization.utils import get_distinct_colors
from ICARUS.visualization.utils import get_distinct_markers


def plot_airfoils_at_reynolds(
    airfoil_names: list[str],
    reynolds: float | None = None,
    solvers: list[str] | str = ["All"],
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
    number_of_plots = len(plots) + 1

    # Divide the plots equally
    sqrt_num = number_of_plots**0.5
    i: int = int(np.ceil(sqrt_num))
    j: int = int(np.floor(sqrt_num))

    fig: Figure = plt.figure(figsize=size)
    axs: ndarray = fig.subplots(2, 2)  # noqa

    fig.suptitle(f"{title}", fontsize=16)

    for plot, ax in zip(plots, axs.flatten()[: len(plots)]):
        ax.set_xlabel(plot[0])
        ax.set_ylabel(plot[1])
        ax.set_title(f"{plot[1]} vs {plot[0]}")
        ax.grid()
        ax.axhline(y=0, color="k")
        ax.axvline(x=0, color="k")

    if isinstance(solvers, str):
        solvers = [solvers]

    if solvers == ["All"]:
        solvers = ["Xfoil", "Foil2Wake", "OpenFoam", "XFLR"]

    DB = Database.get_instance()

    colors = get_distinct_colors(len(airfoil_names))
    for j, airfoil_name in enumerate(airfoil_names):
        markers = get_distinct_markers(len(solvers))
        for i, solver in enumerate(solvers):
            try:
                polar_map: AirfoilPolarMap = DB.get_airfoil_polars(
                    airfoil=airfoil_name,
                    solver=solver,
                )
            except (AirfoilNotFoundError, PolarsNotFoundError):
                print(
                    f"Airfoil {airfoil_name} Solver: {solver} doesn't exist in the database",
                )
                continue
            try:
                if reynolds is None:
                    reyn_idx = int(len(polar_map.reynolds_numbers) // 2)
                    reyn = polar_map.reynolds_numbers[reyn_idx]
                    print(
                        f"Reynolds number not provided for {airfoil_name}. Selecting Reynolds number: {reyn:,}",
                    )
                else:
                    available_reynolds = polar_map.reynolds_numbers
                    # Find the closest reynolds number to the given reynolds
                    reyn = min(
                        available_reynolds,
                        key=lambda x: abs(float(x) - reynolds),
                    )
                polar = polar_map.get_polar(reyn)

                # Sort the data by AoA
                polar_df = polar.df.sort_values(by="AoA")
                if aoa_bounds is not None:
                    # Get data where AoA is in AoA bounds
                    polar_df = polar_df.loc[
                        (polar_df["AoA"] >= aoa_bounds[0])
                        & (polar_df["AoA"] <= aoa_bounds[1])
                    ]
                for plot, ax in zip(plots, axs.flatten()[: len(plots)]):
                    if plot[1] == "CL/CD" or plot[1] == "CL/CD":
                        polar_df["CL/CD"] = polar_df["CL"] / polar_df["CD"]
                        # Get the index of the values that are greater than 200
                        idx = polar_df[polar_df["CL/CD"] > 200].index
                        idx2 = polar_df[polar_df["CL/CD"] < -200].index
                        # Replace the values with 0
                        polar_df.loc[idx, "CL/CD"] = 0
                        polar_df.loc[idx2, "CL/CD"] = 0

                    if plot[1] == "CD/CL" or plot[1] == "CD/CL":
                        polar_df["CD/CL"] = polar_df["CD"] / polar_df["CL"]
                        # If any value is infinite (or greater than 200), replace it with 0
                        # Get the index of the values that are greater than 200
                        idx = polar_df[polar_df["CD/CL"] > 200].index
                        idx2 = polar_df[polar_df["CD/CL"] < -200].index
                        # Replace the values with 0
                        polar_df.loc[idx, "CD/CL"] = 0
                        polar_df.loc[idx2, "CD/CL"] = 0

                    key0 = f"{plot[0]}"
                    key1 = f"{plot[1]}"

                    if plot[0] == "AoA":
                        key0 = "AoA"
                    if plot[1] == "AoA":
                        key1 = "AoA"

                    x: Series[float] = polar_df[f"{key0}"]
                    y: Series[float] = polar_df[f"{key1}"]
                    c = colors[j]
                    m = markers[i].get_marker()
                    label: str = f"{airfoil_name}: {reyn:,} - {solver}"
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
