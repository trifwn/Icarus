import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame

from . import colors
from . import markers


def plot_airplane_polars(
    data: dict[str, DataFrame],
    airplanes: list[str],
    solvers: list[str] = ["All"],
    size: tuple[int, int] = (10, 10),
) -> None:
    """Function to plot airplane polars for a given list of airplanes and solvers

    Args:
        data (dict[str, DataFrame]): Dictionary of airplane polars
        airplanes (list[str]): List of airplanes to plot
        solvers (list[str], optional): List of Solvers to plot. Defaults to ["All"].
        size (tuple[int, int], optional): Figure Size. Defaults to (10, 10).
    """
    fig: Figure = plt.figure(figsize=size)
    axs: ndarray = fig.subplots(2, 2)

    if len(airplanes) == 1:
        fig.suptitle(f"{airplanes[0]} Aero Coefficients", fontsize=16)
    else:
        fig.suptitle("Airplanes Aero Coefficients", fontsize=16)

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
        solvers = ["Potential", "ONERA", "2D"]

    for i, airplane in enumerate(airplanes):
        skip = False
        for j, solver in enumerate(solvers):
            try:
                polar = data[airplane]
                aoa = polar["AoA"]
                if airplane.startswith("XFLR"):
                    cl = polar["CL"]
                    cd = polar["CD"]
                    cm = polar["Cm"]
                    skip = True
                    c: str = "m"
                    m: str = "x"
                    style: str = f"{c}{m}-"

                    label: str = f"{airplane}"
                else:
                    cl = polar[f"CL_{solver}"]
                    cd = polar[f"CD_{solver}"]
                    cm = polar[f"Cm_{solver}"]
                    c = colors[i]
                    m = markers[j]
                    style = f"{c}{m}--"

                    label = f"{airplane} - {solver}"
                axs[0, 1].plot(aoa, cd, style, label=label, markersize=3.5, linewidth=1)
                axs[1, 0].plot(aoa, cl, style, label=label, markersize=3.5, linewidth=1)
                axs[1, 1].plot(cd, cl, style, label=label, markersize=3.5, linewidth=1)
                axs[0, 0].plot(aoa, cm, style, label=label, markersize=3.5, linewidth=1)
            except KeyError as solver:
                print(f"Run Doesn't Exist: {airplane},{solver}")
            if skip:
                break
    fig.tight_layout()
    for axe in axs:
        for ax in axe:
            ax.axhline(y=0, color="k")
            ax.axvline(x=0, color="k")
            ax.grid()

    axs[1, 0].legend()  # (bbox_to_anchor=(-0.1, -0.25),  ncol=3,
    # fancybox=True, loc='lower left')
    plt.show()
