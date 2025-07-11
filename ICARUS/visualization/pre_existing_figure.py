from functools import wraps
from typing import Any
from typing import Callable
from typing import Optional
from typing import TypeVar
from typing import Union
from typing import cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.figure import SubFigure

from .figure_setup import flatten_axes

F = TypeVar("F", bound=Callable[..., Any])


def pre_existing_figure(
    subplots: tuple[int, int] = (1, 2),
    default_title: Optional[str] = None,
    figsize: tuple[float, float] = (10.0, 10.0),
    return_axs: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to prepare or reuse a matplotlib Figure and Axes array for plotting.

    It allows a method to accept an optional `axs` argument (list of Axes) and
    handles figure/axes creation or reuse accordingly. If axes are reused, it also
    sets the suptitle if not already present.

    Parameters
    ----------
    subplots : tuple[int, int], default=(1, 2)
        Tuple specifying the (rows, cols) layout for subplots when creating a new figure.
    default_title : str | None, optional
        Suptitle for the figure; used if the `title` argument is not provided.
    figsize : tuple[float, float], default=(10.0, 10.0)
        Size of the figure to create when `axs` is not provided or needs recreation.
    return_axs : bool, default=True
        If True, the decorated function returns a tuple (axs, fig). If False, returns None.

    Returns
    -------
    Callable
        The decorated function, optionally returning (axs, fig).
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(
            self: Any,
            *args: Any,
            axs: Optional[list[Axes]] = None,
            title: Optional[str] = None,
            **kwargs: Any,
        ) -> Union[tuple[list[Axes], Union[Figure, SubFigure]], None]:
            effective_title = title if title is not None else default_title

            need_recreate = axs is None or len(axs) != (subplots[0] * subplots[1])

            if need_recreate:
                fig = plt.figure(figsize=figsize)
                if effective_title is not None:
                    fig.suptitle(effective_title)
                axs_generated = fig.subplots(*subplots)
                axs_now = flatten_axes(axs_generated)
            else:
                if axs is None:
                    raise ValueError("axs cannot be None if not recreating the figure")
                fig = axs[0].figure or plt.figure(figsize=figsize)
                axs_now = flatten_axes(axs)

                # Set suptitle if missing
                if (
                    getattr(fig, "_suptitle", None) is None
                    and effective_title is not None
                ):
                    fig.suptitle(effective_title)

            # Execute user plot function
            func(self, *args, axs=axs_now, **kwargs)

            # Show the figure (unless part of subfigure layout)
            if not isinstance(fig, SubFigure):
                fig.show()

            if return_axs:
                return axs_now, fig
            return None

        return cast(F, wrapper)

    return decorator
