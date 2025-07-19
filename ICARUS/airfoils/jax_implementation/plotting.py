"""
Plotting utilities for JAX airfoil implementation.

This module provides advanced plotting capabilities including batch plotting,
visualization of camber lines and thickness distributions, and debugging tools.
"""

from typing import TYPE_CHECKING
from typing import List
from typing import Optional
from typing import Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from .jax_airfoil import JaxAirfoil


class AirfoilPlotter:
    """
    Advanced plotting utilities for JAX airfoils.

    This class provides methods for:
    - Batch plotting of multiple airfoils
    - Visualization of camber lines and thickness distributions
    - Scatter plot options for debugging
    - Customizable plot styling and layout
    """

    @staticmethod
    def plot_batch(
        airfoils: List["JaxAirfoil"],
        names: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        ax: Optional[Axes] = None,
        scatter: bool = False,
        alpha: float = 0.7,
        linewidth: float = 1.0,
        legend: bool = True,
    ) -> Axes:
        """
        Plot multiple airfoils on the same axes.

        Args:
            airfoils: List of JaxAirfoil instances to plot
            names: Optional list of names for legend (uses airfoil.name if None)
            colors: Optional list of colors for each airfoil
            ax: Matplotlib axes object. If None, creates new figure.
            scatter: Whether to plot as scatter plots
            alpha: Transparency level for plots
            linewidth: Line width for plots
            legend: Whether to show legend

        Returns:
            Matplotlib axes object with the plots

        Raises:
            ValueError: If input lists have mismatched lengths
        """
        if not airfoils:
            raise ValueError("At least one airfoil must be provided")

        # Validate input lengths
        n_airfoils = len(airfoils)
        if names is not None and len(names) != n_airfoils:
            raise ValueError(
                f"Length of names ({len(names)}) must match number of airfoils ({n_airfoils})",
            )
        if colors is not None and len(colors) != n_airfoils:
            raise ValueError(
                f"Length of colors ({len(colors)}) must match number of airfoils ({n_airfoils})",
            )

        # Create axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Default colors if not provided
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, n_airfoils))

        # Default names if not provided
        if names is None:
            names = [airfoil.name for airfoil in airfoils]

        # Plot each airfoil
        for i, airfoil in enumerate(airfoils):
            pts = airfoil.to_selig()
            x, y = pts

            color = colors[i]
            name = names[i]

            if scatter:
                # Plot upper and lower surfaces as scatter plots
                ax.scatter(
                    x[: airfoil._upper_split_idx],
                    y[: airfoil._upper_split_idx],
                    s=10,
                    color=color,
                    alpha=alpha,
                    label=f"{name} (upper)",
                )
                ax.scatter(
                    x[airfoil._upper_split_idx :],
                    y[airfoil._upper_split_idx :],
                    s=10,
                    color=color,
                    alpha=alpha,
                    marker="s",
                    label=f"{name} (lower)",
                )
            else:
                # Plot upper and lower surfaces as lines
                ax.plot(
                    x[: airfoil._upper_split_idx],
                    y[: airfoil._upper_split_idx],
                    color=color,
                    alpha=alpha,
                    linewidth=linewidth,
                    label=f"{name} (upper)",
                )
                ax.plot(
                    x[airfoil._upper_split_idx :],
                    y[airfoil._upper_split_idx :],
                    color=color,
                    alpha=alpha,
                    linewidth=linewidth,
                    linestyle="--",
                    label=f"{name} (lower)",
                )

        # Configure plot
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        ax.set_title(f"Batch Airfoil Plot ({n_airfoils} airfoils)")

        if (
            legend and n_airfoils <= 10
        ):  # Only show legend for reasonable number of airfoils
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        return ax

    @staticmethod
    def plot_camber_comparison(
        airfoils: List["JaxAirfoil"],
        names: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        ax: Optional[Axes] = None,
        n_points: int = 100,
        legend: bool = True,
    ) -> Axes:
        """
        Plot camber lines for multiple airfoils.

        Args:
            airfoils: List of JaxAirfoil instances
            names: Optional list of names for legend
            colors: Optional list of colors for each airfoil
            ax: Matplotlib axes object. If None, creates new figure.
            n_points: Number of points to evaluate camber line
            legend: Whether to show legend

        Returns:
            Matplotlib axes object with the camber line plots
        """
        if not airfoils:
            raise ValueError("At least one airfoil must be provided")

        # Create axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        # Default colors and names
        n_airfoils = len(airfoils)
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, n_airfoils))
        if names is None:
            names = [airfoil.name for airfoil in airfoils]

        # Plot camber lines
        for i, airfoil in enumerate(airfoils):
            # Get coordinate bounds
            pts = airfoil.to_selig()
            x_min = jnp.min(pts[0])
            x_max = jnp.max(pts[0])

            # Generate query points
            x_camber = jnp.linspace(x_min, x_max, n_points)
            y_camber = airfoil.camber_line(x_camber)

            # Plot camber line
            ax.plot(x_camber, y_camber, color=colors[i], linewidth=2, label=names[i])

        # Configure plot
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x/c")
        ax.set_ylabel("Camber")
        ax.set_title("Camber Line Comparison")

        if legend:
            ax.legend()

        return ax

    @staticmethod
    def plot_thickness_comparison(
        airfoils: List["JaxAirfoil"],
        names: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        ax: Optional[Axes] = None,
        n_points: int = 100,
        legend: bool = True,
    ) -> Axes:
        """
        Plot thickness distributions for multiple airfoils.

        Args:
            airfoils: List of JaxAirfoil instances
            names: Optional list of names for legend
            colors: Optional list of colors for each airfoil
            ax: Matplotlib axes object. If None, creates new figure.
            n_points: Number of points to evaluate thickness
            legend: Whether to show legend

        Returns:
            Matplotlib axes object with the thickness distribution plots
        """
        if not airfoils:
            raise ValueError("At least one airfoil must be provided")

        # Create axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        # Default colors and names
        n_airfoils = len(airfoils)
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, n_airfoils))
        if names is None:
            names = [airfoil.name for airfoil in airfoils]

        # Plot thickness distributions
        for i, airfoil in enumerate(airfoils):
            # Get coordinate bounds
            pts = airfoil.to_selig()
            x_min = jnp.min(pts[0])
            x_max = jnp.max(pts[0])

            # Generate query points
            x_thick = jnp.linspace(x_min, x_max, n_points)
            thickness = airfoil.thickness(x_thick)

            # Plot thickness distribution
            ax.plot(x_thick, thickness, color=colors[i], linewidth=2, label=names[i])

        # Configure plot
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x/c")
        ax.set_ylabel("Thickness")
        ax.set_title("Thickness Distribution Comparison")

        if legend:
            ax.legend()

        return ax

    @staticmethod
    def plot_airfoil_analysis(
        airfoil: "JaxAirfoil",
        show_camber: bool = True,
        show_thickness: bool = True,
        show_max_thickness: bool = True,
        show_max_camber: bool = True,
        scatter_points: bool = False,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """
        Create a comprehensive analysis plot for a single airfoil.

        Args:
            airfoil: JaxAirfoil instance to analyze
            show_camber: Whether to show camber line
            show_thickness: Whether to show thickness distribution
            show_max_thickness: Whether to mark maximum thickness location
            show_max_camber: Whether to mark maximum camber location
            scatter_points: Whether to show coordinate points as scatter
            ax: Matplotlib axes object. If None, creates new figure.

        Returns:
            Matplotlib axes object with the analysis plot
        """
        # Create axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        # Plot the basic airfoil
        airfoil.plot(
            camber=show_camber,
            scatter=scatter_points,
            max_thickness=show_max_thickness,
            ax=ax,
        )

        # Add maximum camber location if requested
        if show_max_camber:
            x_max_camber = airfoil.max_camber_location
            max_camber = airfoil.max_camber
            y_camber = airfoil.camber_line(jnp.array([x_max_camber]))[0]

            # Mark the maximum camber point
            ax.plot(x_max_camber, y_camber, "go", markersize=8, label="Max Camber")
            ax.text(
                x_max_camber,
                y_camber + 0.02,
                f"Max Camber: {max_camber:.3f}\nat x = {x_max_camber:.3f}",
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
            )

        # Add thickness distribution plot if requested
        if show_thickness:
            # Create a secondary y-axis for thickness
            ax2 = ax.twinx()

            # Get coordinate bounds
            pts = airfoil.to_selig()
            x_min = jnp.min(pts[0])
            x_max = jnp.max(pts[0])

            # Generate query points for thickness
            x_thick = jnp.linspace(x_min, x_max, 100)
            thickness = airfoil.thickness(x_thick)

            # Plot thickness distribution
            ax2.plot(
                x_thick,
                thickness,
                "g-",
                alpha=0.6,
                linewidth=2,
                label="Thickness",
            )
            ax2.set_ylabel("Thickness", color="g")
            ax2.tick_params(axis="y", labelcolor="g")

        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        if show_thickness:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        else:
            ax.legend()

        # Enhanced title with airfoil properties
        title = f"Airfoil Analysis: {airfoil.name}\n"
        title += f"Points: {airfoil.n_points}, "
        title += f"Max Thickness: {airfoil.max_thickness:.3f} at x={airfoil.max_thickness_location:.3f}, "
        title += f"Max Camber: {airfoil.max_camber:.3f} at x={airfoil.max_camber_location:.3f}"
        ax.set_title(title)

        return ax

    @staticmethod
    def create_subplot_grid(
        airfoils: List["JaxAirfoil"],
        names: Optional[List[str]] = None,
        ncols: int = 3,
        figsize: Optional[Tuple[float, float]] = None,
        show_camber: bool = False,
        scatter: bool = False,
    ) -> Tuple[Figure, List[Axes]]:
        """
        Create a grid of subplots for multiple airfoils.

        Args:
            airfoils: List of JaxAirfoil instances
            names: Optional list of names for subplot titles
            ncols: Number of columns in the grid
            figsize: Figure size (width, height)
            show_camber: Whether to show camber lines
            scatter: Whether to use scatter plots

        Returns:
            Tuple of (figure, list of axes)
        """
        if not airfoils:
            raise ValueError("At least one airfoil must be provided")

        n_airfoils = len(airfoils)
        nrows = (n_airfoils + ncols - 1) // ncols  # Ceiling division

        # Default figure size
        if figsize is None:
            figsize = (4 * ncols, 3 * nrows)

        # Create subplots
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        # Handle single subplot case
        if n_airfoils == 1:
            axes = [axes]
        elif nrows == 1:
            axes = list(axes)
        else:
            axes = axes.flatten()

        # Default names
        if names is None:
            names = [airfoil.name for airfoil in airfoils]

        # Plot each airfoil
        for i, airfoil in enumerate(airfoils):
            ax = axes[i]
            airfoil.plot(camber=show_camber, scatter=scatter, ax=ax)
            ax.set_title(names[i])

        # Hide unused subplots
        for i in range(n_airfoils, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig, axes[:n_airfoils]

    @staticmethod
    def plot_morphing_sequence(
        airfoil1: "JaxAirfoil",
        airfoil2: "JaxAirfoil",
        n_steps: int = 5,
        figsize: Optional[Tuple[float, float]] = None,
        alpha: float = 0.7,
    ) -> Tuple[Figure, Axes]:
        """
        Plot a sequence of morphed airfoils between two base airfoils.

        Args:
            airfoil1: First airfoil
            airfoil2: Second airfoil
            n_steps: Number of morphing steps to show
            figsize: Figure size (width, height)
            alpha: Transparency level for plots

        Returns:
            Tuple of (figure, axes)
        """
        if figsize is None:
            figsize = (12, 6)

        fig, ax = plt.subplots(figsize=figsize)

        # Generate morphing parameters
        eta_values = jnp.linspace(0, 1, n_steps)
        colors = plt.cm.viridis(eta_values)

        # Plot morphed airfoils
        for i, eta in enumerate(eta_values):
            # Create morphed airfoil
            from .jax_airfoil import JaxAirfoil

            morphed = JaxAirfoil.morph_new_from_two_foils(
                airfoil1,
                airfoil2,
                float(eta),
                n_points=200,
            )

            # Plot the morphed airfoil
            pts = morphed.to_selig()
            x, y = pts

            ax.plot(
                x[: morphed._upper_split_idx],
                y[: morphed._upper_split_idx],
                color=colors[i],
                alpha=alpha,
                linewidth=2,
                label=f"η = {eta:.2f}",
            )
            ax.plot(
                x[morphed._upper_split_idx :],
                y[morphed._upper_split_idx :],
                color=colors[i],
                alpha=alpha,
                linewidth=2,
                linestyle="--",
            )

        # Configure plot
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        ax.set_title(f"Morphing Sequence: {airfoil1.name} → {airfoil2.name}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        return fig, ax

    @staticmethod
    def debug_coordinate_plot(
        airfoil: "JaxAirfoil",
        show_indices: bool = True,
        show_buffer: bool = True,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """
        Create a debugging plot showing coordinate indices and buffer information.

        Args:
            airfoil: JaxAirfoil instance to debug
            show_indices: Whether to show point indices
            show_buffer: Whether to show buffer information
            ax: Matplotlib axes object. If None, creates new figure.

        Returns:
            Matplotlib axes object with the debug plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        # Get coordinates
        pts = airfoil.to_selig()
        x, y = pts

        # Plot upper surface points
        upper_x = x[: airfoil._upper_split_idx]
        upper_y = y[: airfoil._upper_split_idx]
        ax.scatter(upper_x, upper_y, c="red", s=50, alpha=0.7, label="Upper Surface")

        # Plot lower surface points
        lower_x = x[airfoil._upper_split_idx :]
        lower_y = y[airfoil._upper_split_idx :]
        ax.scatter(lower_x, lower_y, c="blue", s=50, alpha=0.7, label="Lower Surface")

        # Show indices if requested
        if show_indices:
            for i, (xi, yi) in enumerate(zip(upper_x, upper_y)):
                ax.annotate(
                    f"U{i}",
                    (xi, yi),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

            for i, (xi, yi) in enumerate(zip(lower_x, lower_y)):
                ax.annotate(
                    f"L{i}",
                    (xi, yi),
                    xytext=(5, -15),
                    textcoords="offset points",
                    fontsize=8,
                )

        # Configure plot
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        ax.legend()

        # Add buffer information
        if show_buffer:
            title = f"Debug Plot: {airfoil.name}\n"
            title += f"Valid Points: {airfoil.n_points}, Buffer Size: {airfoil.buffer_size}, "
            title += f"Upper Split: {airfoil._upper_split_idx}"
            ax.set_title(title)

        return ax
