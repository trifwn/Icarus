from functools import wraps
from typing import Any
from typing import Callable
from typing import Optional
from typing import TypeVar
from typing import Union
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.figure import SubFigure

from .figure_setup import flatten_axes

F = TypeVar("F", bound=Callable[..., Any])


def polar_plot(
    default_title: Optional[str] = None,
    default_plots: list[list[str]] = [
        ["AoA", "CL"],
        ["AoA", "CD"],
        ["AoA", "Cm"],
        ["AoA", "CL/CD"],
    ],
    figsize: tuple[float, float] = (10.0, 10.0),
    return_axs: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to prepare or reuse a matplotlib Figure and Axes array for polar plotting.

    It allows a method to optionally receive pre-created axes (`axs`) and a list of plots.
    If axes are not provided or are insufficient, a new figure is created automatically.
    The figure is post-processed to hide excess axes and display a shared legend.

    Parameters
    ----------
    default_title : str | None, optional
        Default title to use as the suptitle if none is provided at runtime.
    default_plots : list[list[str]], default=[["AoA", "CL"], ...]
        Default list of plots. Each inner list defines an x/y pair to be plotted.
    figsize : tuple[float, float], default=(10.0, 10.0)
        Size of the figure when created.
    return_axs : bool, default=True
        If True, return the list of Axes and the Figure/SubFigure after plotting.

    Returns
    -------
    Callable
        The decorated function, optionally returning (axs, fig).
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(
            self,
            *args: Any,
            axs: Optional[list[Axes]] = None,
            title: Optional[str] = None,
            plots: Optional[list[list[str]]] = None,
            **kwargs: Any,
        ) -> Union[tuple[list[Axes], Union[Figure, SubFigure]], None]:
            effective_title = title if title is not None else default_title
            effective_plots = plots if plots is not None else default_plots

            num_plots = len(effective_plots) + 1  # +1 for CL/CD or similar
            rows = int(np.ceil(np.sqrt(num_plots)))
            cols = int(np.floor(np.sqrt(num_plots)))
            subplots_shape = (rows, cols)

            need_recreate = axs is None or len(axs) != rows * cols

            if axs is not None:
                flat_axs = axs.flatten() if isinstance(axs, np.ndarray) else axs
                fig = flat_axs[0].figure or plt.figure(figsize=figsize)
                if len(flat_axs) != rows * cols:
                    print(
                        f"Warning: {len(flat_axs)} axes provided, but {rows * cols} expected. Creating new figure.",
                    )
                    need_recreate = True
            else:
                print("Warning: No axes provided. Creating new figure.")
                need_recreate = True

            if need_recreate:
                print(
                    f"Creating new figure with size {figsize} and subplots {subplots_shape}.",
                )
                fig = plt.figure(figsize=figsize)
                axs_prod = fig.subplots(*subplots_shape)
                axs_now = flatten_axes(axs_prod)
                if effective_title is not None:
                    fig.suptitle(effective_title)
            else:
                if axs is None:
                    raise ValueError("Provided axs is None")
                fig = axs[0].figure
                axs_now = flatten_axes(axs)
                if (
                    getattr(fig, "_suptitle", None) is None
                    and effective_title is not None
                ):
                    fig.suptitle(effective_title)

            # Call the original plotting function
            func(self, *args, axs=axs_now, plots=effective_plots, **kwargs)

            # Hide unused axes
            for ax in axs_now[len(effective_plots) :]:
                ax.set_visible(False)

            # Configure used axes
            for ax in axs_now[: len(effective_plots)]:
                ax.grid(True)

            # Shared legend
            handles, labels = axs_now[0].get_legend_handles_labels()
            if fig.legends:
                for legend in fig.legends:
                    legend.remove()
            fig.legend(handles, labels, loc="lower right", ncol=2)

            # Adjust layout
            fig.subplots_adjust(top=0.9, bottom=0.1)
            if not isinstance(fig, SubFigure):
                fig.tight_layout()
                fig.show()

            if return_axs:
                return axs_now, fig
            return None

        return cast(F, wrapper)

    return decorator
