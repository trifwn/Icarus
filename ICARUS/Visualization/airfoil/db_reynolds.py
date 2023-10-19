import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from numpy import ndarray
from pandas import DataFrame

from .. import colors
from .. import markers
from ICARUS.Core.struct import Struct


def plot_airfoil_reynolds(
    data: dict[str, dict[str, dict[str, DataFrame]]] | Struct,
    airfoil_name: str,
    reynolds: str,
    solvers: list[str] = ["All"],
    size: tuple[int, int] = (10, 10),
) -> None:
    """Function to plot airfoil polars for a given list of airfoils and solvers.

    Args:
        data (dict[str, dict[str, dict[str, DataFrame]]]): Nested dictionary containing the airfoil Polars
        airfoil_name (str): Airfoil name (e.g. naca0012)
        reynolds (str): Reynolds number (e.g. 100000)
        solvers (list[str], optional): Can be either all or individual solver names. Defaults to ["All"].
        size (tuple[int, int], optional): Figure Size. Defaults to (10, 10).
    """

    fig: Figure = plt.figure(figsize=size)
    axs: ndarray = fig.subplots(2, 2)  # type: ignore
    fig.suptitle(
        f"NACA {airfoil_name[4:]}- Reynolds={reynolds}\n Aero Coefficients",
        fontsize=16,
    )
    axs[0, 0].set_title("Cm vs AoA")
    axs[0, 0].set_ylabel("Cm")

    axs[0, 1].set_title("Cd vs AoA")
    axs[0, 1].set_xlabel("AoA")
    axs[0, 1].set_ylabel("Cd")

    axs[1, 0].set_title("Cl vs AoA")
    axs[1, 0].set_xlabel("AoA")
    axs[1, 0].set_ylabel("Cl")

    axs[1, 1].set_title("Cl vs Cd")
    axs[1, 1].set_xlabel("Cd")

    if solvers == ["All"]:
        solvers = ["Xfoil", "Foil2Wake", "OpenFoam", "XFLR"]

    for j, solver in enumerate(solvers):
        try:
            polar = data[airfoil_name][solver][reynolds]
            aoa, cl, cd, cm = polar.T.values
            c: str = colors[j]
            m: MarkerStyle = markers[j].get_marker()
            style: str = f"{c}{m}-"
            label: str = f"{airfoil_name}: {reynolds} - {solver}"
            axs[0, 1].plot(aoa, cd, style, label=label, markersize=3, linewidth=1)
            axs[1, 0].plot(aoa, cl, style, label=label, markersize=3, linewidth=1)
            axs[1, 1].plot(cd, cl, style, label=label, markersize=3, linewidth=1)
            axs[0, 0].plot(aoa, cm, style, label=label, markersize=3, linewidth=1)
        except KeyError as solver:
            print(f"Run Doesn't Exist: {airfoil_name},{reynolds},{solver}")

    fig.tight_layout()

    axs[0, 1].grid()
    axs[1, 0].grid()
    axs[1, 1].grid()
    axs[0, 0].grid()

    axs[1, 0].legend(
        bbox_to_anchor=(-0.1, -0.25),
        ncol=3,
        fancybox=True,
        loc="lower left",
    )
