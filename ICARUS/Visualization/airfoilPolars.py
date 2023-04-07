import matplotlib.pyplot as plt
import os
import numpy as np
from . import colors, markers


def plotAirfoilPolars(data, airfoil, solvers='All', size=(10, 10), AoA_bounds=None):
    fig, axs = plt.subplots(2, 2, figsize=size)

    fig.suptitle(f'NACA {airfoil[4:]} Aero Coefficients', fontsize=16)
    axs[0, 0].set_title('Cm vs AoA')
    axs[0, 0].set_ylabel('Cm')

    axs[0, 1].set_title('Cd vs AoA')
    axs[0, 1].set_xlabel('AoA')
    axs[0, 1].set_ylabel('Cd')

    axs[1, 0].set_title('Cl vs AoA')
    axs[1, 0].set_xlabel('AoA')
    axs[1, 0].set_ylabel('Cl')

    axs[1, 1].set_title('Cl vs Cd')
    axs[1, 1].set_xlabel('Cd')

    if solvers == ['All']:
        solvers = ["Xfoil", "Foil2Wake", "OpenFoam", "XFLR"]

    for i, solver in enumerate(data[airfoil].keys()):
        if solver not in solvers:
            continue
        for j, reynolds in enumerate(data[airfoil][solver].keys()):
            try:
                polar = data[airfoil][solver][reynolds]
                if AoA_bounds is not None:
                    # Get data where AoA is in AoA bounds
                    polar = polar.loc[(polar['AoA'] >= AoA_bounds[0]) &
                                      (polar['AoA'] <= AoA_bounds[1])]

                aoa, cl, cd, cm = polar.T.values
                c = colors[j]
                m = markers[i]
                style = f"{c}{m}-"
                label = f"{airfoil}: {reynolds} - {solver}"
                axs[0, 1].plot(aoa, cd, style, label=label,
                               markersize=3, linewidth=1)
                axs[1, 0].plot(aoa, cl, style, label=label,
                               markersize=3, linewidth=1)
                axs[1, 1].plot(cd, cl, style, label=label,
                               markersize=3, linewidth=1)
                axs[0, 0].plot(aoa, cm, style, label=label,
                               markersize=3, linewidth=1)
            except KeyError as solver:
                print(f"Run Doesn't Exist: {airfoil},{reynolds},{solver}")

    fig.tight_layout()
    if len(solvers) == 3:
        per = -.85
    elif len(solvers) == 2:
        per = -.6
    else:
        per = -0.4
    axs[0, 1].grid()
    axs[1, 0].grid()
    axs[1, 1].grid()
    axs[0, 0].grid()

    axs[1, 0].legend(bbox_to_anchor=(-0.1, per),  ncol=3,
                     fancybox=True, loc='lower left')
