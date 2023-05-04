import os

import matplotlib.pyplot as plt
import numpy as np

from . import colors
from . import markers


def plotAirplanePolars(data, airplanes, solvers=["All"], size=(10, 10)):
    fig, axs = plt.subplots(2, 2, figsize=size)
    if len(airplanes) == 1:
        fig.suptitle(f"{airplanes[0]} Aero Coefficients", fontsize=16)
    else:
        fig.suptitle(f"Airplanes Aero Coefficients", fontsize=16)

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
                    cl = polar[f"CL"]
                    cd = polar[f"CD"]
                    cm = polar[f"Cm"]
                    skip = True
                    c = "m"
                    m = "x"
                    style = f"{c}{m}-"

                    label = f"{airplane}"
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
            if skip == True:
                break
    fig.tight_layout()
    for axR in axs:
        for ax in axR:
            ax.axhline(y=0, color="k")
            ax.axvline(x=0, color="k")
            ax.grid()

    axs[1, 0].legend()  # (bbox_to_anchor=(-0.1, -0.25),  ncol=3,
    # fancybox=True, loc='lower left')
    plt.show()
