from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from ICARUS.visualization import colors_
from ICARUS.visualization import markers

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ICARUS.core.struct import Struct
    from ICARUS.vehicle.plane import Airplane


def plot_sensitivity(
    data: Struct,
    plane: Airplane,
    trim: pd.Series[float],
    relative: bool = False,
    vars2s: list[str] = ["All"],
    solvers: list[str] = ["2D"],
    size: tuple[int, int] = (16, 7),
) -> None:
    fig: Figure = plt.figure(figsize=size)
    axs: list[Axes] = fig.subplots(2, 3).flatten()  # type: ignore
    fig.suptitle(f"{plane.name} Convergence", fontsize=16)

    axs[0].set_title("Fx vs epsilon")
    axs[0].set_ylabel("Fx")

    axs[1].set_title("Fy vs epsilon")
    axs[1].set_ylabel("Fy")

    axs[2].set_title("Fz vs epsilon")
    axs[2].set_ylabel("Fz")

    axs[3].set_title("Mx vs epsilon")
    axs[3].set_ylabel("Mx")
    axs[3].set_xlabel("Epsilon")

    axs[4].set_title("My vs epsilon")
    axs[4].set_ylabel("My")
    axs[4].set_xlabel("Epsilon")

    axs[5].set_title("Mz vs epsilon")
    axs[5].set_ylabel("Mz")
    axs[5].set_xlabel("Epsilon")

    try:
        cases: Struct = data
    except Exception:
        print("No Sensitivity Results")
        return
    i = -1
    j = -1

    trim = trim.astype(float)
    fx_trim: float = trim[f"TFORC{solvers[0]}(1)"]
    fy_trim: float = trim[f"TFORC{solvers[0]}(2)"]
    fz_trim: float = trim[f"TFORC{solvers[0]}(3)"]
    mx_trim: float = trim[f"TAMOM{solvers[0]}(1)"]
    my_trim: float = trim[f"TAMOM{solvers[0]}(2)"]
    mz_trim: float = trim[f"TAMOM{solvers[0]}(2)"]
    if not relative:
        axs[0].axhline(
            fx_trim,
            xmin=-1,
            xmax=1,
            color="k",
            label="Trim",
            linewidth=1,
        )
        axs[1].axhline(
            fy_trim,
            xmin=-1,
            xmax=1,
            color="k",
            label="Trim",
            linewidth=1,
        )
        axs[2].axhline(
            fz_trim,
            xmin=-1,
            xmax=1,
            color="k",
            label="Trim",
            linewidth=1,
        )

        axs[3].axhline(
            mx_trim,
            xmin=-1,
            xmax=1,
            color="k",
            label="Trim",
            linewidth=1,
        )
        axs[4].axhline(
            my_trim,
            xmin=-1,
            xmax=1,
            color="k",
            label="Trim",
            linewidth=1,
        )
        axs[5].axhline(
            mz_trim,
            xmin=-1,
            xmax=1,
            color="k",
            label="Trim",
            linewidth=1,
        )

    for var in list(cases.keys()):
        if var in vars2s or vars2s == ["All"]:
            runHist = cases[var]
            i += 1
            j = 0
        else:
            continue

        for solver in solvers:
            try:
                epsilon = runHist["Epsilon"].astype(float)
                fx = runHist[f"TFORC{solver}(1)"].astype(float)
                fy = runHist[f"TFORC{solver}(2)"].astype(float)
                fz = runHist[f"TFORC{solver}(3)"].astype(float)
                mx = runHist[f"TAMOM{solver}(1)"].astype(float)
                my = runHist[f"TAMOM{solver}(2)"].astype(float)
                mz = runHist[f"TAMOM{solver}(2)"].astype(float)

                if relative:
                    fx = fx - fx_trim
                    fy = fy - fy_trim
                    fz = fz - fz_trim
                    mx = mx - mx_trim
                    my = my - my_trim
                    mz = mz - mz_trim

                j += 1
                c = colors_(i / len(vars2s))
                m = markers[j]
                # style = f"{c}{m}--"

                label = f"{plane.name} - {solver} - {var}"
                axs[0].scatter(
                    epsilon,
                    fx,
                    color=c,
                    marker=m,
                    label=label,
                    linewidth=3.0,
                )
                axs[1].scatter(
                    epsilon,
                    fy,
                    color=c,
                    marker=m,
                    label=label,
                    linewidth=3.0,
                )
                axs[2].scatter(
                    epsilon,
                    fz,
                    color=c,
                    marker=m,
                    label=label,
                    linewidth=3.0,
                )

                axs[3].scatter(
                    epsilon,
                    mx,
                    color=c,
                    marker=m,
                    label=label,
                    linewidth=3.0,
                )
                axs[4].scatter(
                    epsilon,
                    my,
                    color=c,
                    marker=m,
                    label=label,
                    linewidth=3.0,
                )
                axs[5].scatter(
                    epsilon,
                    mz,
                    color=c,
                    marker=m,
                    label=label,
                    linewidth=3.0,
                )

            except KeyError as solver:
                print(f"Run Doesn't Exist: {plane.name},{solver},{var}")

    fig.tight_layout()
    for ax in axs:
        ax.grid(which="both", axis="both")
        ax.set_xscale("log")
        ax.set_yscale("log")

    axs[1].legend()  # (bbox_to_anchor=(-0.1, -0.25),  ncol=3,
    # fancybox=True, loc='lower left')
    plt.show()
