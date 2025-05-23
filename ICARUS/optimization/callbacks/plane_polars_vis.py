from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import Collection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text

from ICARUS.flight_dynamics import State
from . import OptimizationCallback


class PlanePolarOptimizationVisualizer(OptimizationCallback):
    def __init__(
        self,
        initial_state: State,
    ) -> None:
        self.initial_state: State = initial_state
        self.lines: dict[str, Line2D] = {}
        self.collections: dict[str, Collection] = {}
        self.text: Text | None = None

    def setup(
        self,
    ) -> None:
        fig = plt.figure(figsize=(10, 10))
        fig.show()
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Create 4 sublots (2,2) for the airplane polars and add points for the trim values
        axs: list[Axes] = []
        axs.append(fig.add_subplot(2, 2, 1))
        axs.append(fig.add_subplot(2, 2, 2))
        axs.append(fig.add_subplot(2, 2, 3))
        axs.append(fig.add_subplot(2, 2, 4))

        try:
            aoa = self.initial_state.polar["AoA"].to_numpy()
            CL = self.initial_state.polar["CL"].to_numpy()
            CD = self.initial_state.polar["CD"].to_numpy()
            Cm = self.initial_state.polar["Cm"].to_numpy()
            cl_over_cd = CL / CD
        except AttributeError:
            aoa = np.array([], dtype=float)
            CL = np.array([], dtype=float)
            CD = np.array([], dtype=float)
            Cm = np.array([], dtype=float)
            cl_over_cd = np.array([], dtype=float)

        axs[0].plot(aoa, CL, label="Initial", color="b", linestyle="--", linewidth=1)
        # Set origin axies lines
        axs[0].axhline(0, color="k", linewidth=0.5)
        axs[0].axvline(0, color="k", linewidth=0.5)
        line = Line2D(
            aoa,
            CL,
            color="r",
            label="Current",
        )
        axs[0].add_line(line)
        self.lines["CL"] = line
        axs[0].set_xlabel("AoA [deg]")
        axs[0].set_ylabel("CL")
        axs[0].set_title("CL vs AoA")
        axs[0].legend()
        axs[0].grid()

        axs[1].plot(aoa, CD, label="Initial", color="b", linestyle="--", linewidth=1)
        axs[1].axhline(0, color="k", linewidth=0.5)
        axs[1].axvline(0, color="k", linewidth=0.5)
        line = Line2D(
            aoa,
            CD,
            color="r",
            label="Current",
        )
        self.lines["CD"] = line
        axs[1].add_line(line)
        axs[1].set_xlabel("AoA [deg]")
        axs[1].set_ylabel("CD")
        axs[1].set_title("CD vs AoA")
        axs[1].legend()
        axs[1].grid()

        axs[2].plot(aoa, Cm, label="Initial", color="b", linestyle="--", linewidth=1)
        axs[2].axhline(0, color="k", linewidth=0.5)
        axs[2].axvline(0, color="k", linewidth=0.5)
        line = Line2D(
            aoa,
            Cm,
            color="r",
            label="Current",
        )
        self.lines["Cm"] = line
        axs[2].add_line(line)
        axs[2].set_xlabel("AoA [deg]")
        axs[2].set_ylabel("Cm")
        axs[2].set_title("Cm vs AoA")
        axs[2].legend()
        axs[2].grid()

        axs[3].plot(
            aoa,
            cl_over_cd,
            label="Initial",
            color="b",
            linestyle="--",
            linewidth=1,
        )
        axs[3].axhline(0, color="k", linewidth=0.5)
        axs[3].axvline(0, color="k", linewidth=0.5)
        line = Line2D(
            aoa,
            cl_over_cd,
            color="r",
            label="Current",
        )
        self.lines["CL_CD"] = line
        axs[3].add_line(line)
        axs[3].set_xlabel("AoA")
        axs[3].set_ylabel("CL / CD")
        axs[3].set_title("CL vs CD")
        axs[3].legend()
        axs[3].grid()

        self.text = axs[0].text(
            0,
            0,
            "",
            ha="center",
            va="center",
        )

        if "AoA" in self.initial_state.trim:
            text: Text = axs[0].text(
                x=self.initial_state.trim["AoA"],
                y=self.initial_state.trim["CL"],
                s=f"AoA: {self.initial_state.trim['AoA']:.2f} with V: {self.initial_state.trim['U']:.2f}",
                ha="center",
                va="center",
            )
            self.text = text

            # Add scatter points for the trim values
            axs[0].scatter(
                self.initial_state.trim["AoA"],
                self.initial_state.trim["CL"],
                label="Initial Trim",
            )
            axs[1].scatter(
                self.initial_state.trim["AoA"],
                self.initial_state.trim["CD"],
                label="Initial Trim",
            )
            axs[3].scatter(
                self.initial_state.trim["AoA"],
                self.initial_state.trim["CL"] / self.initial_state.trim["CD"],
                label="Initial Trim",
            )

        # Add dummy scatter points for the current values
        collection = axs[0].scatter(0, 0, label="Current")
        self.collections["CL"] = collection
        collection = axs[1].scatter(0, 0, label="Current")
        self.collections["CD"] = collection
        collection = axs[3].scatter(0, 0, label="Current")
        self.collections["CL_CD"] = collection

        # Update the figure
        fig.canvas.draw()
        fig.canvas.flush_events()

        self.axs: list[Axes] = axs
        self.fig: Figure = fig

    def update(
        self,
        state: State,
        **kwargs: Any,
    ) -> None:
        if not hasattr(self, "fig") or not hasattr(self, "axs"):
            self.setup()

        # Update the plot
        axs = self.axs
        fig = self.fig

        # Update the CL vs AoA plot
        line = self.lines["CL"]
        line.set_xdata(state.polar["AoA"])
        line.set_ydata(state.polar["CL"])
        axs[0].relim()
        axs[0].autoscale_view()

        # Update the CD vs AoA plot
        line = self.lines["CD"]
        line.set_xdata(state.polar["AoA"])
        line.set_ydata(state.polar["CD"])
        axs[1].relim()
        axs[1].autoscale_view()

        # Update the Cm vs AoA plot
        line = self.lines["Cm"]
        line.set_xdata(state.polar["AoA"])
        line.set_ydata(state.polar["Cm"])
        axs[2].relim()
        axs[2].autoscale_view()

        # Update the CL vs CD plot
        line = self.lines["CL_CD"]
        line.set_xdata(state.polar["AoA"])
        line.set_ydata(state.polar["CL"] / state.polar["CD"])
        axs[3].relim()
        axs[3].autoscale_view()

        # Update the scatter points
        if "AoA" in state.trim:
            collection = self.collections["CL"]
            collection.set_offsets(np.array([state.trim["AoA"], state.trim["CL"]]))
            collection = self.collections["CD"]
            collection.set_offsets(np.array([state.trim["AoA"], state.trim["CD"]]))
            collection = self.collections["CL_CD"]
            collection.set_offsets(
                np.array([state.trim["AoA"], state.trim["CL"] / state.trim["CD"]]),
            )
            # Add text annotations for state trim velocity and aoa
            if isinstance(self.text, Text):
                self.text.set_text(
                    f"AoA: {state.trim['AoA']:.2f} with V: {state.trim['U']:.2f}",
                )
            else:
                self.text = axs[0].text(
                    x=state.trim["AoA"],
                    y=state.trim["CL"],
                    s=f"AoA: {state.trim['AoA']:.2f} with V: {state.trim['U']:.2f}",
                    ha="center",
                    va="center",
                )

        # Update the figure
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)
