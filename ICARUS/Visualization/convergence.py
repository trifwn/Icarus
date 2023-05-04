import os

import matplotlib.pyplot as plt
import numpy as np

from . import colors
from . import markers


def plotConvergence(
    data,
    plane,
    angles=["All"],
    solvers=["All"],
    plotError=True,
    size=(10, 10),
):
    # Define 3 subplots that will be filled with Fx Fz and My vs Iterations
    fig, axs = plt.subplots(3, 3, figsize=size)
    fig.suptitle(f"{plane} Convergence", fontsize=16)

    axs[0, 0].set_title("Fx vs Iterations")
    axs[0, 0].set_ylabel("Fx")

    axs[0, 1].set_title("Fy vs Iterations")
    axs[0, 1].set_ylabel("Fy")

    axs[0, 2].set_title("Fz vs Iterations")
    axs[0, 2].set_ylabel("Fz")

    axs[1, 0].set_title("Mx vs Iterations")
    axs[1, 0].set_ylabel("Mx")
    axs[1, 0].set_xlabel("Iterations")

    axs[1, 1].set_title("My vs Iterations")
    axs[1, 1].set_ylabel("My")
    axs[1, 1].set_xlabel("Iterations")

    axs[1, 2].set_title("Mz vs Iterations")
    axs[1, 2].set_ylabel("Mz")
    axs[1, 2].set_xlabel("Iterations")

    axs[2, 0].set_title("ERROR vs Iterations")
    axs[2, 0].set_ylabel("ERROR")
    axs[2, 0].set_xlabel("Iterations")

    axs[2, 1].set_title("ERRORM vs Iterations")
    axs[2, 1].set_ylabel("ERRORM")
    axs[2, 1].set_xlabel("Iterations")

    fig.delaxes(axs[2, 2])
    # Fill plots with data
    if solvers == ["All"]:
        solvers = ["", "2D", "DS2D"]

    cases = data[plane]
    i = -1
    j = -1
    toomuchData = False
    for ang in cases.keys():
        num = float("".join(c for c in ang if (c.isdigit() or c == ".")))
        if ang.startswith("m"):
            ang_num = -num
        else:
            ang_num = num

        if ang_num in angles or angles is ["All"]:
            runHist = cases[ang]
            i += 1
            j = 0
        else:
            continue
        for solver in solvers:
            try:
                it = runHist["TTIME"].astype(float)
                it = it / it.iloc[0]

                fx = np.abs(runHist[f"TFORC{solver}(1)"].astype(float))
                fy = np.abs(runHist[f"TFORC{solver}(2)"].astype(float))
                fz = np.abs(runHist[f"TFORC{solver}(3)"].astype(float))
                mx = np.abs(runHist[f"TAMOM{solver}(1)"].astype(float))
                my = np.abs(runHist[f"TAMOM{solver}(2)"].astype(float))
                mz = np.abs(runHist[f"TAMOM{solver}(2)"].astype(float))
                error = runHist[f"ERROR"].astype(float)
                errorM = runHist[f"ERRORM"].astype(float)
                it2 = it
                if plotError == True:
                    it = it.iloc[1:].values
                    fx = np.abs(fx.iloc[1:].values - fx.iloc[:-1].values)
                    fy = np.abs(fy.iloc[1:].values - fy.iloc[:-1].values)
                    fz = np.abs(fz.iloc[1:].values - fz.iloc[:-1].values)
                    mx = np.abs(mx.iloc[1:].values - mx.iloc[:-1].values)
                    my = np.abs(my.iloc[1:].values - my.iloc[:-1].values)
                    mz = np.abs(mz.iloc[1:].values - mz.iloc[:-1].values)

                j += 1
                if i > len(colors) - 1:
                    toomuchData = True
                    break
                c = colors[i]
                m = markers[j]
                style = f"{c}{m}--"

                label = f"{plane} - {solver} - {ang_num}"
                axs[0, 0].plot(it, fx, style, label=label, markersize=2.0, linewidth=1)
                axs[0, 1].plot(it, fy, style, label=label, markersize=2.0, linewidth=1)
                axs[0, 2].plot(it, fz, style, label=label, markersize=2.0, linewidth=1)

                axs[1, 0].plot(it, mx, style, label=label, markersize=2.0, linewidth=1)
                axs[1, 1].plot(it, my, style, label=label, markersize=2.0, linewidth=1)
                axs[1, 2].plot(it, mz, style, label=label, markersize=2.0, linewidth=1)

                axs[2, 0].plot(
                    it2,
                    error,
                    style,
                    label=label,
                    markersize=2.0,
                    linewidth=1,
                )
                axs[2, 1].plot(
                    it2,
                    errorM,
                    style,
                    label=label,
                    markersize=2.0,
                    linewidth=1,
                )
            except KeyError as e:
                print(f"Run Doesn't Exist: {plane},{e},{ang}")
    if toomuchData == True:
        print(f"Too much data to plot, only plotting {len(colors)} cases")

    fig.tight_layout()
    for axR in axs:
        for ax in axR:
            ax.grid(which="both", axis="both")
            ax.set_yscale("log")

    axs[2, 1].legend(bbox_to_anchor=(1, -0.25), ncol=2, fancybox=True, loc="lower left")
