import matplotlib.pyplot as plt

from . import colors
from . import markers


def plotAirfoilReynolds(data, airfoil, reyn, solvers="All", size=(10, 10)):
    fig, axs = plt.subplots(2, 2, figsize=size)
    fig.suptitle(
        f"NACA {airfoil[4:]}- Reynolds={reyn}\n Aero Coefficients",
        fontsize=16,
    )
    axs[0, 0].set_title("Cm vs AoA")  # type: ignore
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
            polar = data[airfoil][solver][reyn]
            aoa, cl, cd, cm = polar.T.values
            c = colors[j]
            m = markers[j]
            style = f"{c}{m}-"
            label = f"{airfoil}: {reyn} - {solver}"
            axs[0, 1].plot(aoa, cd, style, label=label, markersize=3, linewidth=1)
            axs[1, 0].plot(aoa, cl, style, label=label, markersize=3, linewidth=1)
            axs[1, 1].plot(cd, cl, style, label=label, markersize=3, linewidth=1)
            axs[0, 0].plot(aoa, cm, style, label=label, markersize=3, linewidth=1)
        except KeyError as solver:
            print(f"Run Doesn't Exist: {airfoil},{reyn},{solver}")

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
