import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ICARUS.core.types import FloatArray
from ICARUS.database import Database
from ICARUS.visualization import colors_
from ICARUS.visualization import markers


def plot_convergence(
    plane: str,
    angles: list[float] | FloatArray = [],
    solvers: list[str] = ["All"],
    plot_error: bool = True,
    size: tuple[int, int] = (10, 10),
) -> None:
    """Function to plot the convergence of a given plane simulation given the
    dimensional forces and solver errors

    Args:
        plane (str): Plane Name
        angles (list[str], optional): Angles to show. Defaults to ["All"].
        solvers (list[str], optional): Solvers to show. Defaults to ["All"].
        plot_error (bool, optional): Wheter to plot the relative error or not. Defaults to True.
        size (tuple[int,int], optional): Size of the figure. Defaults to (10, 10).

    """
    DB = Database.get_instance()
    data = DB.vehicles_db.convergence_data

    # Define 3 subplots that will be filled with Fx Fz and My vs Iterations
    fig: Figure = plt.figure(figsize=size)
    axs: np.ndarray = fig.subplots(3, 3)  # type: ignore
    fig.suptitle(f"{plane}", fontsize=16)

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
        solvers = ["", "2D"]#, "DS2D"]

    cases = data[plane]
    i = 0
    for ang in cases.keys():
        num = float("".join(c for c in ang if (c.isdigit() or c == ".")))
        if ang.startswith("m"):
            ang_num: float = -num
        else:
            ang_num = num

        if (ang_num not in angles) and (angles != []):
            continue

        runHist = cases[ang]
        i += 1
        j = 0
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
                error = runHist["ERROR"].astype(float)
                errorM = runHist["ERRORM"].astype(float)
                it2 = it
                if plot_error:
                    it = it.iloc[1:].values
                    fx = np.abs(fx.iloc[1:].values - fx.iloc[:-1].values)
                    fy = np.abs(fy.iloc[1:].values - fy.iloc[:-1].values)
                    fz = np.abs(fz.iloc[1:].values - fz.iloc[:-1].values)
                    mx = np.abs(mx.iloc[1:].values - mx.iloc[:-1].values)
                    my = np.abs(my.iloc[1:].values - my.iloc[:-1].values)
                    mz = np.abs(mz.iloc[1:].values - mz.iloc[:-1].values)

                j += 1
                c = colors_[j]
                m = markers[i].get_marker()

                label: str = f"{plane} - {solver} - {ang_num}"
                axs[0, 0].plot(
                    it,
                    fx,
                    color=c,
                    marker=m,
                    label=label,
                    markersize=2.0,
                    linewidth=1,
                )
                axs[0, 1].plot(
                    it,
                    fy,
                    color=c,
                    marker=m,
                    label=label,
                    markersize=2.0,
                    linewidth=1,
                )
                axs[0, 2].plot(
                    it,
                    fz,
                    color=c,
                    marker=m,
                    label=label,
                    markersize=2.0,
                    linewidth=1,
                )
                axs[1, 0].plot(
                    it,
                    mx,
                    color=c,
                    marker=m,
                    label=label,
                    markersize=2.0,
                    linewidth=1,
                )
                axs[1, 1].plot(
                    it,
                    my,
                    color=c,
                    marker=m,
                    label=label,
                    markersize=2.0,
                    linewidth=1,
                )
                axs[1, 2].plot(
                    it,
                    mz,
                    color=c,
                    marker=m,
                    label=label,
                    markersize=2.0,
                    linewidth=1,
                )

                axs[2, 0].plot(
                    it2,
                    error,
                    color=c,
                    marker=m,
                    label=label,
                    markersize=2.0,
                    linewidth=1,
                )
                axs[2, 1].plot(
                    it2,
                    errorM,
                    color=c,
                    marker=m,
                    label=label,
                    markersize=2.0,
                    linewidth=1,
                )
            except KeyError as e:
                print(f"Run Doesn't Exist: {plane},{e},{ang}")

    fig.tight_layout()
    for axR in axs:
        for ax in axR:
            ax.grid(which="both", axis="both")
            # ax.set_yscale("log")

    axs[2, 1].legend(bbox_to_anchor=(1, -0.25), ncol=2, fancybox=True, loc="lower left")
