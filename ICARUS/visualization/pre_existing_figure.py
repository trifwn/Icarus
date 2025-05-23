from functools import wraps

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.figure import SubFigure

from .figure_setup import flatten_axes


def pre_existing_figure(
    subplots: tuple[int, int] = (1, 2),
    default_title: str | None = None,
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
            **kwargs,
        ) -> tuple[list[Axes], Figure | SubFigure] | None:
            # Determine the effective title
            effective_title = title if title is not None else default_title

            need_recreate = False
            # If axes are provided, check if their count matches requested subplots
            if axs is not None:
                fig = axs[0].figure or plt.figure(figsize=figsize)
                total_axes = subplots[0] * subplots[1] if subplots else 1
                if len(axs) != total_axes:
                    need_recreate = True
            else:
                need_recreate = True

            # Create new figure/axes if needed
            if need_recreate:
                fig = plt.figure(figsize=figsize)
                if effective_title is not None:
                    fig.suptitle(effective_title)

                axs_prod = fig.subplots(*subplots)
                axs_now = flatten_axes(axs_prod)
            else:
                if axs is None:
                    raise ValueError("axs cannot be None if not recreating figure")

                # Reuse provided axes
                axs_now = flatten_axes(axs)
                # Ensure suptitle exists
                if getattr(fig, "_suptitle", None) is None and effective_title is not None:
                    fig.suptitle(effective_title)

            # Execute plotting
            func(self, *args, axs=axs_now, **kwargs)

            # Display if top-level
            if not isinstance(fig, SubFigure):
                fig.show()

            # Optionally return objects
            if return_axs:
                return axs_now, fig

        return wrapper

    return decorator
