import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame
from pandas import Series

from .. import colors
from .. import markers
from ICARUS.Core.struct import Struct


def plot_airfoil_polars(
    data: dict[str, dict[str, dict[str, DataFrame]]] | Struct,
    airfoil: str,
    solvers: list[str] | str = "All",
    plots=[["AoA", "CL"], ["AoA", "CD"], ["AoA", "Cm"], ["CL", "CD"]],
    size: tuple[int, int] = (10, 10),
    aoa_bounds: list[float] | None = None,
    title: str = "Aero Coefficients",
) -> tuple[ndarray, Figure]:
    """
    Args:
        data (dict[str, dict[str, dict[str, DataFrame]]]): Nested Dictionary with the airfoil polars
        airfoil (str): airfoil names
        solvers (list[str] | str, optional): List of solver Names. Defaults to "All".
        size (tuple[int, int], optional): Fig Size. Defaults to (10, 10).
        AoA_bounds (_type_, optional): Angle of Attack Bounds. Defaults to None.
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

    for i, solver in enumerate(data[airfoil].keys()):
        if solver not in solvers:
            print(f"Skipping {solver} is not in {solvers}")
            continue

        for j, reynolds in enumerate(data[airfoil][solver].keys()):
            try:
                polar: DataFrame = data[airfoil][solver][reynolds]
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
                    c = colors(j / len(data[airfoil][solver].keys()))
                    m = markers[i].get_marker()
                    label: str = f"{airfoil}: {reynolds} - {solver}"
                    try:
                        ax.plot(x, y, ls='--', color=c, marker=m, label=label, markersize=3.5, linewidth=1)
                    except ValueError as e:
                        raise e
            except (KeyError, ValueError) as solv:
                print(f"Run Doesn't Exist: {airfoil},{reynolds},{solv}")

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
