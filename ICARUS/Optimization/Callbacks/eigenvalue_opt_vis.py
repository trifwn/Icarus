import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import Collection
from matplotlib.figure import Figure

from ICARUS.Core.types import ComplexArray
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Optimization.Callbacks.optimization_callback import OptimizationCallback


class EigenvalueOptimizationVisualizer(OptimizationCallback):
    def __init__(
        self,
        goal_longitudal: ComplexArray,
        goal_lateral: ComplexArray,
        initial_longitudal: ComplexArray,
        initial_lateral: ComplexArray,
    ) -> None:
        self.goal_longitudal: ComplexArray = goal_longitudal
        self.goal_lateral: ComplexArray = goal_lateral
        self.zero_longitudal: ComplexArray = initial_longitudal
        self.zero_lateral: ComplexArray = initial_lateral

    def setup(
        self,
    ) -> None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        fig.show()
        fig.canvas.draw()
        fig.canvas.flush_events()

        # DUMMY scatters
        zeros = np.zeros(4)

        # Longitudal Mode
        x: list[float] = [ele.real for ele in self.goal_longitudal]
        y: list[float] = [ele.imag for ele in self.goal_longitudal]
        axs[0].scatter(x, y, color="r", marker="d", label="Desired")
        x = [ele.real for ele in self.zero_longitudal]
        y = [ele.imag for ele in self.zero_longitudal]
        axs[0].scatter(x, y, color="b", marker="o", label="Start")
        collection_long = axs[0].scatter(zeros, zeros, color="k", marker="x", label="Current")
        axs[0].grid()
        axs[0].set_title("Longitudal Eigenvalues")
        axs[0].legend()
        # x=0 line
        axs[0].axhline(0, color="k", linewidth=0.5)
        axs[0].axvline(0, color="k", linewidth=0.5)

        # Lateral Mode
        x = [ele.real for ele in self.goal_lateral]
        y = [ele.imag for ele in self.goal_lateral]
        axs[1].scatter(x, y, color="r", marker="d", label="Desired")
        x = [ele.real for ele in self.zero_lateral]
        y = [ele.imag for ele in self.zero_lateral]
        axs[1].scatter(x, y, color="b", marker="o", label="Start")
        collection_lat = axs[1].scatter(zeros, zeros, color="k", marker="x", label="Current")
        axs[1].grid()
        axs[1].set_title("Lateral Eigenvalues")
        axs[1].legend()
        # x=0 line
        axs[1].axvline(0, color="k", linewidth=0.5)
        axs[1].axhline(0, color="k", linewidth=0.5)

        # Update the figure
        fig.canvas.draw()
        fig.canvas.flush_events()

        self.fig: Figure = fig
        self.axs: list[Axes] = axs
        self.collection_lat: Collection = collection_lat
        self.collection_long: Collection = collection_long

    def update(
        self,
        state: State,
        iteration: int,
        **kwargs: Any,
    ) -> None:
        if (
            not hasattr(self, "fig")
            or not hasattr(self, "axs")
            or not hasattr(self, "collection_lat")
            or not hasattr(self, "collection_long")
        ):
            self.setup()

        new_eigenvalues_longitudal: ComplexArray = state.state_space.longitudal.eigenvalues
        new_eigenvalues_lateral: ComplexArray = state.state_space.lateral.eigenvalues

        self.fig.canvas.flush_events()
        title: str = f"Eigenvalues at {iteration} of optimization"
        # Make title display the polynomial equation using latex

        self.fig.suptitle(title, fontsize=16)
        ax_longitudal: Axes = self.axs[0]
        ax_lateral: Axes = self.axs[1]

        # scatter Longitudal

        offsets: list[tuple[float, float]] = []
        for item in new_eigenvalues_lateral:
            offsets.append((item.real, item.imag))
        self.collection_lat.set_offsets(np.array(offsets))

        # scatter Longitudal
        offsets = []
        for item in new_eigenvalues_longitudal:
            offsets.append((item.real, item.imag))
        self.collection_long.set_offsets(np.array(offsets))

        # Relim and Scale
        ax_longitudal.relim()
        ax_longitudal.autoscale()
        ax_lateral.relim()
        ax_lateral.autoscale()

        self.axs[0].relim()
        self.axs[0].ignore_existing_data_limits = True
        self.axs[0].update_datalim(self.collection_long.get_datalim(self.axs[0].transData))
        self.axs[0].autoscale_view()

        self.axs[1].relim()
        self.axs[1].ignore_existing_data_limits = True
        self.axs[1].update_datalim(self.collection_lat.get_datalim(self.axs[1].transData))
        self.axs[1].autoscale_view()

        self.fig.canvas.draw()
        time.sleep(0.1)
