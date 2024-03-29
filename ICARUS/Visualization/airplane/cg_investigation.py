from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.widgets import Button
from matplotlib.widgets import Slider
from numpy import ndarray

from ICARUS.database import DB
from ICARUS.vehicle.plane import Airplane
from ICARUS.visualization.airplane.db_polars import plot_airplane_polars


def setup_plot(
    airplane: str,
    solvers: list[str] = ["All"],
    size: tuple[int, int] = (10, 10),
    title: str = "Aero Coefficients",
) -> tuple[list[Axes], Any]:
    fig = plt.figure(figsize=size)
    axs: ndarray = fig.subplots(2, 2)  # type: ignore

    if len(airplane) == 1:
        fig.suptitle(f"{airplane} Aero Coefficients", fontsize=16)
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

    axs[1, 1].set_title("Cl/Cd vs AoA")
    axs[1, 1].set_xlabel("AoA")

    if solvers == ["All"]:
        solvers = [
            "GNVP3 Potential",
            "GNVP3 2D",
            "GNVP7 Potential",
            "GNVP7 2D",
            "LSPT Potential",
            "LSPT 2D",
        ]

    for j, solver in enumerate(solvers):
        skip = False
        try:
            polar = DB.vehicles_db.polars[airplane]
            aoa = polar["AoA"]
            if airplane.startswith("XFLR"):
                cl = polar[f"{solver} CL"]
                cd = polar[f"{solver} CD"]
                cm = polar[f"{solver} Cm"]
                skip = True

                style: str = f"--"

                label: str = f"{airplane}"
            else:
                cl = polar[f"{solver} CL"]
                cd = polar[f"{solver} CD"]
                cm = polar[f"{solver} Cm"]
                style = f"--"

                label = f"{airplane} - {solver}"
            try:
                axs[0, 1].plot(aoa, cd, style, label=label, markersize=3.5, linewidth=1)
                axs[1, 0].plot(aoa, cl, style, label=label, markersize=3.5, linewidth=1)
                axs[1, 1].plot(aoa, cl / cd, style, label=label, markersize=3.5, linewidth=1)
                axs[0, 0].plot(aoa, cm, style, label=label, markersize=3.5, linewidth=1)
            except ValueError as e:
                print(style)
                raise e
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

    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.show()

    return axs.flatten().tolist(), fig


def cg_investigation(
    airplane_name: str,
    solvers: list[str] = ["All"],
    size: tuple[int, int] = (10, 10),
    title: str = "Aero Coefficients",
) -> None:
    axs, fig = setup_plot(airplane_name, solvers, size, title)

    # Get the plane from the database
    plane: Airplane = DB.vehicles_db.planes[airplane_name]
    cg_x: float = plane.CG[1]

    # Create a slider to change the CG
    ax_cg = fig.add_axes((0.25, 0.1, 0.65, 0.03))

    cg_slider = Slider(ax=ax_cg, label="CG", valmin=-1, valmax=1, valinit=cg_x)

    ## CLS
    lines = list(axs[0].get_lines())
    CLs = {}
    for line in lines:
        name: str = str(line.get_label())
        if name.startswith("_"):
            continue
        CLs[name] = line.get_ydata()

    ## CDs
    lines = list(axs[1].get_lines())
    CDs = {}
    for line in lines:
        name = str(line.get_label())
        if name.startswith("_"):
            continue
        CDs[name] = line.get_ydata()

    ## CMs
    lines = list(axs[2].get_lines())
    CMs = {}
    AoAs = {}
    for line in lines:
        name = str(line.get_label())
        if name.startswith("_"):
            continue
        CMs[name] = line.get_ydata()
        AoAs[name] = line.get_xdata()

    points_cl = []
    points_cd = []
    points_cm = []
    points_clcd = []
    for name in CLs.keys():
        cl = np.array(CLs[name])
        cd = np.array(CDs[name])
        cm = np.array(CMs[name])

        # Find the point where the CM = 0
        id = np.argmin(np.abs(cm))
        aoa = AoAs[name][id]

        points_cm.append(axs[0].scatter(aoa, cm[id], marker="x", color="k"))
        points_cd.append(axs[1].scatter(aoa, cd[id], marker="x", color="k"))
        points_cl.append(axs[2].scatter(aoa, cl[id], marker="x", color="k"))
        points_clcd.append(axs[3].scatter(aoa, cl[id] / cd[id], marker="x", color="k"))

    # The function to be called anytime a slider's value changes
    def update(new_cg: float) -> None:
        """
        The function to be called anytime a slider's value changes
        Each time the slider is changed, the cg position is updated and the plot is redrawn
        All the forces and moments are recalculated.

        Args:
            new_cg (float): The new cg position in the x direction
        """

        # Adjust the CM according to the new CG
        # Update the plot only for the CMs
        axs[0].clear()
        axs[0].set_title("Cm vs AoA")
        axs[0].set_ylabel("Cm")
        axs[0].set_xlabel("AoA")
        axs[0].grid()

        axs[0].axhline(y=0, color="k")
        axs[0].axvline(x=0, color="k")

        for i, name in enumerate(CLs.keys()):
            cl = np.array(CLs[name])
            cd = np.array(CDs[name])
            cm = np.array(CMs[name])

            CM_new = cm + (cg_x - new_cg) * cl / plane.mean_aerodynamic_chord
            CM_new = CM_new + (cg_x - new_cg) * cd / plane.mean_aerodynamic_chord

            axs[0].plot(AoAs[name], CM_new, label=name)

            # Find the point where the CM = 0
            id = np.argmin(np.abs(CM_new))
            aoa = AoAs[name][id]

            # Add the point to the plots
            points_cm[i].set_offsets(np.c_[aoa, CM_new[id]])
            points_cd[i].set_offsets(np.c_[aoa, cd[id]])
            points_cl[i].set_offsets(np.c_[aoa, cl[id]])
            points_clcd[i].set_offsets(np.c_[aoa, cl[id] / cd[id]])

    # register the update function with each slider
    cg_slider.on_changed(update)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = fig.add_axes((0.8, 0.025, 0.1, 0.04))
    button = Button(resetax, "Reset", hovercolor="0.975")

    def reset(event: Any) -> None:
        cg_slider.reset()

    button.on_clicked(reset)
    plt.show()


if __name__ == "__main__":
    planenames: list[str] = DB.vehicles_db.get_planenames()

    cg_investigation(
        planenames[2],
        solvers=[
            "GNVP7 2D",
        ],
        size=(10, 7),
    )
