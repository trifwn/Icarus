from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import Collection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.widgets import Button
from matplotlib.widgets import Slider
from numpy import ndarray
from pandas import Series

from ICARUS import APPHOME
from ICARUS.database import DB
from ICARUS.propulsion.engine import Engine
from ICARUS.vehicle.plane import Airplane


def setup_plot(
    airplanes: list[str],
    planes: list[Airplane],
    solvers: list[str] = ["All"],
    size: tuple[int, int] = (10, 10),
) -> tuple[
    Figure,
    dict[str, Line2D],
    dict[str, Line2D],
    dict[str, Line2D],
    dict[str, Line2D],
    dict[str, dict[str, Collection]],
    dict[str, Any],
]:
    fig = plt.figure(figsize=size)
    axs: ndarray = fig.subplots(2, 2)  # type: ignore

    if len(airplanes) == 1:
        fig.suptitle(f"{airplanes[0]} CG Investigation", fontsize=16)
    else:
        fig.suptitle(f"Aero Coefficients", fontsize=16)

    for ax_row in axs:
        for ax in ax_row:
            ax.axhline(y=0, color="k")
            ax.axvline(x=0, color="k")
            ax.grid(True)

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
            "AVL",
        ]

    polars = [DB.vehicles_db.get_polars(airplane) for airplane in airplanes]
    cm_lines = {}
    cd_lines = {}
    cl_lines = {}
    clcd_lines = {}
    collections = {}
    annots = {}

    for i, polar in enumerate(polars):
        for j, solver in enumerate(solvers):
            try:
                aoa = polar["AoA"]
                cl: Series[bool] = polar[f"{solver} CL"]
                cd: Series[bool] = polar[f"{solver} CD"]
                cm: Series[bool] = polar[f"{solver} Cm"]
                style = f"--"
                label = f"{airplanes[i]} - {solver}"

                try:
                    axs[1, 0].plot(
                        aoa,
                        cl,
                        style,
                        label=f"Original {label}",
                        markersize=3.5,
                        linewidth=1,
                    )
                    axs[0, 0].plot(
                        aoa,
                        cm,
                        style,
                        label=f"Original {label}",
                        markersize=3.5,
                        linewidth=1,
                    )
                    axs[0, 1].plot(
                        aoa,
                        cd,
                        style,
                        label=f"Original {label}",
                        markersize=3.5,
                        linewidth=1,
                    )
                    axs[1, 1].plot(
                        aoa,
                        cl / cd,
                        style,
                        label=f"Original {label}",
                        markersize=3.5,
                        linewidth=1,
                    )
                    (cl_line,) = axs[1, 0].plot(aoa, cl, style, label=label, markersize=3.5, linewidth=1)
                    (cm_line,) = axs[0, 0].plot(aoa, cm, style, label=label, markersize=3.5, linewidth=1)
                    (cd_line,) = axs[0, 1].plot(aoa, cd, style, label=label, markersize=3.5, linewidth=1)
                    (clcd_line,) = axs[1, 1].plot(aoa, cl / cd, style, label=label, markersize=3.5, linewidth=1)

                    cm_lines[airplanes[i]] = cm_line
                    cd_lines[airplanes[i]] = cd_line
                    cl_lines[airplanes[i]] = cl_line
                    clcd_lines[airplanes[i]] = clcd_line
                    break
                except ValueError as e:
                    print(style)
                    raise e
            except KeyError as solver:
                print(f"Run Doesn't Exist: {airplanes[i]},{solver}")

        # Interpolate the aoa for cm = 0 with numpy
        aoas = np.array(aoa)
        cm_sorted_idx = np.argsort(cm)
        aoa_trim = np.interp(0.0, cm[cm_sorted_idx], aoas[cm_sorted_idx])

        clcd_trim = np.interp(aoa_trim, aoas, np.array(clcd_line.get_ydata()))
        cd_trim = np.interp(aoa_trim, aoas, np.array(cd_line.get_ydata()))
        cl_trim = np.interp(aoa_trim, aoas, np.array(cl_line.get_ydata()))

        # Add Red points to all the plots to indicate the trim
        collection = {}
        # Get cl_line color
        color = cl_line.get_color()

        collection["Cm"] = axs[0, 0].scatter(
            aoa_trim,
            0,
            color=color,
        )
        collection["CD"] = axs[0, 1].scatter(aoa_trim, cd_trim, color=color)
        collection["CL"] = axs[1, 0].scatter(aoa_trim, cl_trim, color=color)
        collection["CL/CD"] = axs[1, 1].scatter(aoa_trim, clcd_trim, color=color)

        collections[airplanes[i]] = collection

        # Get the zero lift aoa
        cl_sorted_idx = np.argsort(cl)
        aoa_zero_lift: float = float(np.interp(0.0, cl[cl_sorted_idx], aoas[cl_sorted_idx]))
        cm_zero_lift = np.interp(aoa_zero_lift, aoas, np.array(cm))

        # Scatter the zero lift aoa on the cm plot
        axs[0, 0].scatter(aoa_zero_lift, cm_zero_lift, color="black", marker="o")

        # Based on airplane mass calculate the trim velocity
        rho = 1.225
        S = planes[i].S
        m = planes[i].M
        g = 9.81
        V_trim = np.sqrt(2 * g * m / (rho * S * cl_trim))
        # Add an annotation to the plot with the trim velocity
        annot = axs[1, 0].annotate(
            f"V_trim = {V_trim:.2f} m/s",
            (aoa_trim, cl_trim),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        annot.set_color(color)
        annots[airplanes[i]] = annot
        axs[0, 0].legend()

    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.show()

    return fig, cl_lines, cd_lines, cm_lines, clcd_lines, collections, annots


def cg_investigation(
    airplane_names: str | list[str],
    solvers: list[str] = ["All"],
    size: tuple[int, int] = (10, 10),
    title: str = "Aero Coefficients",
    engine: Engine | None = None,
) -> None:

    if isinstance(airplane_names, str):
        airplane_names = [airplane_names]

    # Get the plane from the database
    planes: list[Airplane] = [DB.get_vehicle(airplane_name) for airplane_name in airplane_names]

    fig, cl_lines, cd_lines, cm_lines, clcd_lines, collections, annots = setup_plot(
        airplane_names,
        planes,
        solvers,
        size,
    )

    CG = planes[0].CG
    cg_x_orig: float = CG[0]

    # Create a slider to change the CG
    # Place the slider at the bottom of the plot outside the plot area
    ax_cg = fig.add_axes((0.1, 0.025, 0.65, 0.01), facecolor="lightgoldenrodyellow")
    cg_slider = Slider(ax=ax_cg, label="CG", valmin=-1, valmax=1, valinit=cg_x_orig)

    # Store initial CMs for each solver
    initial_CMs = {}
    aoas = {}
    for airplane in airplane_names:
        cm_line = cm_lines[airplane]
        initial_CMs[airplane] = np.array(cm_line.get_ydata(orig=True))
        aoas[airplane] = np.array(cm_line.get_xdata(orig=True))

    # Update function for slider
    def update(new_cg: float) -> None:

        for airplane in airplane_names:
            cm_line = cm_lines[airplane]
            cl_line = cl_lines[airplane]
            cd_line = cd_lines[airplane]
            clcd_line = clcd_lines[airplane]
            collection = collections[airplane]
            annot = annots[airplane]

            cl = np.array(cl_line.get_ydata())
            cd = np.array(cd_line.get_ydata())
            initial_CM = initial_CMs[airplane]
            aoas_now = aoas[airplane]

            new_CM = initial_CM + (new_cg - cg_x_orig) * (cl) / planes[0].mean_aerodynamic_chord
            cm_line.set_ydata(new_CM)

            # Interpolate the aoa for cm = 0 with numpy
            # x = aoa
            # y = cm
            # We need to find x such that y= f(x) = 0
            # Sort the cms

            cm_sorted_idx = np.argsort(new_CM)
            aoa_trim = np.interp(0, new_CM[cm_sorted_idx], aoas_now[cm_sorted_idx])

            clcd_trim = np.interp(aoa_trim, aoas_now, np.array(clcd_line.get_ydata()))
            cd_trim = np.interp(aoa_trim, aoas_now, np.array(cd_line.get_ydata()))
            cl_trim = np.interp(aoa_trim, aoas_now, np.array(cl_line.get_ydata()))

            # Update the red points
            collection["Cm"].set_offsets([[aoa_trim, 0]])
            collection["CD"].set_offsets([[aoa_trim, cd_trim]])
            collection["CL"].set_offsets([[aoa_trim, cl_trim]])
            collection["CL/CD"].set_offsets([[aoa_trim, clcd_trim]])

            # Based on airplane mass calculate the trim velocity
            rho = 1.225
            S = planes[0].S
            m = planes[0].M
            g = 9.81
            V_trim = np.sqrt(2 * m * g / (rho * S * cl_trim))
            text = f"V_trim = {V_trim:.2f} m/s"

            DRAG = cd_trim * 0.5 * rho * V_trim**2 * S
            THRUST = DRAG
            if engine is not None:
                V_trim = jnp.array([V_trim])
                THRUST = jnp.array([THRUST])
                amperes = engine.current(V_trim, THRUST)
                # Add to the text
                text += f" and Current = {amperes[0]:.2f} A"

            # Update annotation to the plot with the trim velocity
            annot.set_text(text)
            annot.set_position((aoa_trim, cl_trim))

            fig.canvas.draw_idle()  # Update the plot

    cg_slider.on_changed(update)

    # Button to reset slider
    resetax = fig.add_axes((0.8, 0.025, 0.1, 0.03))
    button = Button(resetax, "Reset", hovercolor="0.975")

    def reset(event: Any) -> None:
        cg_slider.reset()

    button.on_clicked(reset)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # isss =  np.linspace(-2,2,20)
    planenames = [
        "bmark",
    ]

    engine_dir = f"{APPHOME}/Data/Engine/Motor_1/"

    engine = Engine()
    engine.load_data_from_df(engine_dir)

    cg_investigation(
        planenames,
        solvers=[
            "AVL",
        ],
        size=(10, 7),
        engine=engine,
    )
