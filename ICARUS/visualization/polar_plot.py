from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.figure import SubFigure

from .figure_setup import flatten_axes


def polar_plot(
    default_title: str | None = None,
    default_plots: list[list[str]] = [
        ["AoA", "CL"],
        ["AoA", "CD"],
        ["AoA", "Cm"],
        ["AoA", "CL/CD"],
    ],
    figsize: tuple[float, float] = (10.0, 10.0),
    return_axs: bool = True,
):
    """
    Decorator to prepare or reuse a matplotlib Figure and Axes array for plotting.

    Parameters:
    -----------
    subplots : tuple[int, int]
        Number of rows and columns for subplots when creating a new figure.
    title : str | None
        Suptitle for the figure; falls back to default if None.
    figsize : tuple[float, float]
        Size of the figure when creating a new one.
    return_axs : bool
        If True, return the Axes and Figure objects.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(
            self,
            *args,
            axs: list[Axes] | None = None,
            title: str | None = None,
            plots: list[list[str]] | None = None,
            **kwargs,
        ) -> tuple[list[Axes], Figure | SubFigure] | None:
            # Determine the effective title
            effective_title = title if title is not None else default_title
            effective_plots = plots if plots is not None else default_plots

            number_of_plots = len(effective_plots) + 1

            # Divide the plots equally
            sqrt_num = number_of_plots**0.5
            i: int = int(np.ceil(sqrt_num))
            j: int = int(np.floor(sqrt_num))
            subplots = (i, j)

            need_recreate = False
            # If axes are provided, check if their count matches requested subplots
            if axs is not None:
                flat_axs = axs.flatten() if isinstance(axs, np.ndarray) else axs

                fig = flat_axs[0].figure or plt.figure(figsize=figsize)
                total_axes = subplots[0] * subplots[1] if subplots else 1
                # Check if the number of axes matches the number of plots
                if len(flat_axs) != total_axes:
                    print(f"Warning: {len(flat_axs)} axes provided, but {total_axes} expected. Creating new figure.")
                    need_recreate = True
            else:
                print("Warning: No axes provided. Creating new figure.")
                need_recreate = True

            # Create new figure/axes if needed
            if need_recreate:
                print(f"Creating new figure with size {figsize} and subplots {subplots}.")
                fig = plt.figure(figsize=figsize)
                axs_prod = fig.subplots(*subplots)
                axs_now = flatten_axes(axs_prod)
                if effective_title is not None:
                    fig.suptitle(effective_title)
            else:
                # Reuse provided axes
                if axs is None:
                    raise ValueError("Provided axs is None")

                axs_now = flatten_axes(axs)
                # Ensure suptitle exists
                if getattr(fig, "_suptitle", None) is None and effective_title is not None:
                    fig.suptitle(effective_title)

            # Execute plotting
            func(self, *args, axs=axs_now, **kwargs)

            # Remove empty plots
            for ax in axs_now[len(effective_plots) :]:
                ax.set_visible(False)

            for ax in axs_now[: len(effective_plots)]:
                ax.grid(True)

            handles, labels = axs_now[0].get_legend_handles_labels()
            # if the figure has a legend, remove it
            if fig.legends:
                for legend in fig.legends:
                    legend.remove()
            fig.legend(handles, labels, loc="lower right", ncol=2)

            # Adjust the plots
            fig.subplots_adjust(top=0.9, bottom=0.1)
            # Display if top-level
            if not isinstance(fig, SubFigure):
                fig.tight_layout()
                fig.show()

            # Optionally return objects
            if return_axs:
                return axs_now, fig

        return wrapper

    return decorator
