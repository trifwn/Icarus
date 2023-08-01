import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from numpy import ndarray
from pandas import DataFrame

from .. import colors
from .. import markers
from ICARUS.Core.struct import Struct


def plot_airfoil_polars(
    data: dict[str, dict[str, dict[str, DataFrame]]] | Struct,
    airfoil: str,
    solvers: list[str] | str = "All",
    size: tuple[int, int] = (10, 10),
    aoa_bounds: list[float] | None = None,
) -> None:
    """
    # ! TODO make the DB connection handle that

    Args:
        data (dict[str, dict[str, dict[str, DataFrame]]]): Nested Dictionary with the airfoil polars
        airfoil (str): airfoil names
        solvers (list[str] | str, optional): List of solver Names. Defaults to "All".
        size (tuple[int, int], optional): Fig Size. Defaults to (10, 10).
        AoA_bounds (_type_, optional): Angle of Attack Bounds. Defaults to None.
    """
    # Function to plot airfoil polars

    fig: Figure = plt.figure(figsize=size)
    axs: ndarray = fig.subplots(2, 2)  # type: ignore

    fig.suptitle(f"NACA {airfoil[4:]} Aero Coefficients", fontsize=16)
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

                aoa, cl, cd, cm = polar.T.values

                c: str = colors[j]
                m: MarkerStyle = markers[i]
                style: str = f"{c}{m}-"
                label: str = f"{airfoil}: {reynolds} - {solver}"
                axs[0, 1].plot(aoa, cd, style, label=label, markersize=3, linewidth=1)
                axs[1, 0].plot(aoa, cl, style, label=label, markersize=3, linewidth=1)
                axs[1, 1].plot(cd, cl, style, label=label, markersize=3, linewidth=1)
                axs[0, 0].plot(aoa, cm, style, label=label, markersize=3, linewidth=1)
            except KeyError as solv:
                print(f"Run Doesn't Exist: {airfoil},{reynolds},{solv}")

    fig.tight_layout()
    if len(solvers) == 3:
        per: float = -0.85
    elif len(solvers) == 2:
        per = -0.6
    else:
        per = -0.4
    axs[0, 1].grid()
    axs[1, 0].grid()
    axs[1, 1].grid()
    axs[0, 0].grid()

    axs[1, 0].legend(
        bbox_to_anchor=(-0.1, per),
        ncol=3,
        fancybox=True,
        loc="lower left",
    )
