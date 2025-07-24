#!/usr/bin/env python3
"""
Airfoil Plotting and Visualization Examples

This example demonstrates various ways to visualize JAX airfoils:
1. Basic airfoil plotting
2. Multi-airfoil comparisons
3. Thickness and camber distribution plots
4. Surface pressure visualization concepts
5. Animation of morphing airfoils
6. 3D visualization concepts

All plotting uses matplotlib and leverages JAX arrays seamlessly.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ICARUS.airfoils import Airfoil
from ICARUS.airfoils.naca4 import NACA4


def basic_airfoil_plotting() -> None:
    """Demonstrate basic airfoil plotting techniques."""
    print("=== Basic Airfoil Plotting ===")

    # Create a NACA airfoil
    naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)

    # Get surface coordinates
    upper_surface = naca2412.upper_surface
    lower_surface = naca2412.lower_surface

    # Create basic plot
    plt.figure(figsize=(12, 8))

    # Plot 1: Standard airfoil plot
    plt.subplot(2, 2, 1)
    plt.plot(
        upper_surface[0],
        upper_surface[1],
        "b-",
        linewidth=2,
        label="Upper surface",
    )
    plt.plot(
        lower_surface[0],
        lower_surface[1],
        "r-",
        linewidth=2,
        label="Lower surface",
    )
    plt.fill_between(
        upper_surface[0],
        upper_surface[1],
        lower_surface[1],
        alpha=0.3,
        color="lightblue",
    )
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title(f"{naca2412.name} - Standard View")
    plt.legend()

    # Plot 2: Zoomed view of leading edge
    plt.subplot(2, 2, 2)
    plt.plot(
        upper_surface[0],
        upper_surface[1],
        "b-",
        linewidth=2,
        label="Upper surface",
    )
    plt.plot(
        lower_surface[0],
        lower_surface[1],
        "r-",
        linewidth=2,
        label="Lower surface",
    )
    plt.xlim(0, 0.2)
    plt.ylim(-0.05, 0.1)
    plt.grid(True, alpha=0.3)
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title("Leading Edge Detail")
    plt.legend()

    # Plot 3: Trailing edge detail
    plt.subplot(2, 2, 3)
    plt.plot(
        upper_surface[0],
        upper_surface[1],
        "b-",
        linewidth=2,
        label="Upper surface",
    )
    plt.plot(
        lower_surface[0],
        lower_surface[1],
        "r-",
        linewidth=2,
        label="Lower surface",
    )
    plt.xlim(0.8, 1.0)
    plt.ylim(-0.02, 0.02)
    plt.grid(True, alpha=0.3)
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title("Trailing Edge Detail")
    plt.legend()

    # Plot 4: With coordinate points highlighted
    plt.subplot(2, 2, 4)
    plt.plot(
        upper_surface[0],
        upper_surface[1],
        "b-",
        linewidth=2,
        label="Upper surface",
    )
    plt.plot(
        lower_surface[0],
        lower_surface[1],
        "r-",
        linewidth=2,
        label="Lower surface",
    )
    # Highlight every 10th point
    plt.plot(upper_surface[0][::10], upper_surface[1][::10], "bo", markersize=4)
    plt.plot(lower_surface[0][::10], lower_surface[1][::10], "ro", markersize=4)
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title("With Coordinate Points")
    plt.legend()

    plt.tight_layout()
    plt.show()


def multi_airfoil_comparison():
    """Create comparison plots of multiple airfoils."""
    print("\n=== Multi-Airfoil Comparison ===")

    # Create different airfoils for comparison
    airfoils = [
        NACA4(M=0.0, P=0.0, XX=0.09, n_points=100),  # NACA 0009 - thin symmetric
        NACA4(M=0.0, P=0.0, XX=0.12, n_points=100),  # NACA 0012 - medium symmetric
        NACA4(M=0.0, P=0.0, XX=0.18, n_points=100),  # NACA 0018 - thick symmetric
        NACA4(M=0.02, P=0.4, XX=0.12, n_points=100),  # NACA 2412 - cambered
        NACA4(M=0.04, P=0.4, XX=0.15, n_points=100),  # NACA 4415 - high camber
    ]

    colors = ["blue", "green", "red", "orange", "purple"]

    plt.figure(figsize=(15, 10))

    # Plot 1: All airfoils overlaid
    plt.subplot(2, 3, 1)
    for i, airfoil in enumerate(airfoils):
        upper = airfoil.upper_surface
        lower = airfoil.lower_surface
        plt.plot(upper[0], upper[1], color=colors[i], linewidth=2, label=airfoil.name)
        plt.plot(lower[0], lower[1], color=colors[i], linewidth=2, linestyle="--")

    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title("Airfoil Shape Comparison")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Plot 2: Thickness comparison
    plt.subplot(2, 3, 2)
    x_eval = jnp.linspace(0, 1, 100)
    for i, airfoil in enumerate(airfoils):
        thickness = airfoil.thickness(x_eval)
        plt.plot(x_eval, thickness, color=colors[i], linewidth=2, label=airfoil.name)

    plt.grid(True, alpha=0.3)
    plt.xlabel("x/c")
    plt.ylabel("Thickness (t/c)")
    plt.title("Thickness Distribution")
    plt.legend()

    # Plot 3: Camber comparison (for cambered airfoils)
    plt.subplot(2, 3, 3)
    for i, airfoil in enumerate(airfoils):
        if hasattr(airfoil, "camber_line") and airfoil.m > 0:
            camber = airfoil.camber_line(x_eval)
            plt.plot(x_eval, camber, color=colors[i], linewidth=2, label=airfoil.name)

    plt.grid(True, alpha=0.3)
    plt.xlabel("x/c")
    plt.ylabel("Camber (y_c/c)")
    plt.title("Camber Line Comparison")
    plt.legend()

    # Plot 4: Maximum thickness comparison
    plt.subplot(2, 3, 4)
    names = [airfoil.name for airfoil in airfoils]
    max_thickness = [airfoil.max_thickness for airfoil in airfoils]
    bars = plt.bar(names, max_thickness, color=colors)
    plt.ylabel("Maximum Thickness (t/c)")
    plt.title("Maximum Thickness Comparison")
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, value in zip(bars, max_thickness):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    # Plot 5: Leading edge comparison
    plt.subplot(2, 3, 5)
    for i, airfoil in enumerate(airfoils):
        upper = airfoil.upper_surface
        lower = airfoil.lower_surface
        plt.plot(upper[0], upper[1], color=colors[i], linewidth=2, label=airfoil.name)
        plt.plot(lower[0], lower[1], color=colors[i], linewidth=2, linestyle="--")

    plt.xlim(0, 0.15)
    plt.ylim(-0.05, 0.1)
    plt.grid(True, alpha=0.3)
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title("Leading Edge Comparison")

    # Plot 6: Aspect ratio visualization
    plt.subplot(2, 3, 6)
    thickness_locations = [airfoil.max_thickness_location for airfoil in airfoils]
    plt.scatter(max_thickness, thickness_locations, c=colors, s=100)
    for i, airfoil in enumerate(airfoils):
        plt.annotate(
            airfoil.name,
            (max_thickness[i], thickness_locations[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.xlabel("Maximum Thickness (t/c)")
    plt.ylabel("Max Thickness Location (x/c)")
    plt.title("Thickness vs Location")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def thickness_camber_analysis_plots() -> None:
    """Create detailed thickness and camber analysis plots."""
    print("\n=== Thickness and Camber Analysis Plots ===")

    # Create airfoils with different characteristics
    naca0012 = NACA4(M=0.0, P=0.0, XX=0.12, n_points=100)  # Symmetric
    naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)  # Cambered
    naca4415 = NACA4(M=0.04, P=0.4, XX=0.15, n_points=100)  # High camber

    x_eval = jnp.linspace(0, 1, 200)

    plt.figure(figsize=(15, 10))

    # Plot 1: Thickness distributions
    plt.subplot(2, 3, 1)
    for airfoil, color, label in [
        (naca0012, "blue", "NACA 0012"),
        (naca2412, "red", "NACA 2412"),
        (naca4415, "green", "NACA 4415"),
    ]:
        thickness = airfoil.thickness(x_eval)
        plt.plot(x_eval, thickness, color=color, linewidth=2, label=label)

        # Mark maximum thickness
        max_t = airfoil.max_thickness
        max_t_loc = airfoil.max_thickness_location
        plt.plot(max_t_loc, max_t, "o", color=color, markersize=8)
        plt.annotate(
            f"Max: {max_t:.3f}",
            xy=(max_t_loc, max_t),
            xytext=(10, 10),
            textcoords="offset points",
            color=color,
            fontweight="bold",
        )

    plt.grid(True, alpha=0.3)
    plt.xlabel("x/c")
    plt.ylabel("Thickness (t/c)")
    plt.title("Thickness Distribution Analysis")
    plt.legend()

    # Plot 2: Camber lines
    plt.subplot(2, 3, 2)
    for airfoil, color, label in [
        (naca2412, "red", "NACA 2412"),
        (naca4415, "green", "NACA 4415"),
    ]:
        if hasattr(airfoil, "camber_line"):
            camber = airfoil.camber_line(x_eval)
            plt.plot(x_eval, camber, color=color, linewidth=2, label=label)

            # Mark maximum camber
            max_camber_idx = jnp.argmax(camber)
            max_camber = camber[max_camber_idx]
            max_camber_loc = x_eval[max_camber_idx]
            plt.plot(max_camber_loc, max_camber, "o", color=color, markersize=8)

    plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel("x/c")
    plt.ylabel("Camber (y_c/c)")
    plt.title("Camber Line Analysis")
    plt.legend()

    # Plot 3: Surface slopes (derivatives)
    plt.subplot(2, 3, 3)
    for airfoil, color, label in [
        (naca0012, "blue", "NACA 0012"),
        (naca2412, "red", "NACA 2412"),
    ]:
        if hasattr(airfoil, "camber_line_derivative"):
            slope = airfoil.camber_line_derivative(x_eval)
            plt.plot(x_eval, slope, color=color, linewidth=2, label=f"{label} slope")

    plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel("x/c")
    plt.ylabel("dy_c/dx")
    plt.title("Camber Line Slope")
    plt.legend()

    # Plot 4: Thickness-to-chord ratio distribution
    plt.subplot(2, 3, 4)
    x_stations = jnp.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    for airfoil, color, label in [
        (naca0012, "blue", "NACA 0012"),
        (naca2412, "red", "NACA 2412"),
        (naca4415, "green", "NACA 4415"),
    ]:
        thickness_stations = airfoil.thickness(x_stations)
        plt.plot(
            x_stations,
            thickness_stations,
            "o-",
            color=color,
            linewidth=2,
            markersize=6,
            label=label,
        )

    plt.grid(True, alpha=0.3)
    plt.xlabel("x/c")
    plt.ylabel("Thickness (t/c)")
    plt.title("Thickness at Standard Stations")
    plt.legend()

    # Plot 5: Airfoil comparison with thickness visualization
    plt.subplot(2, 3, 5)
    for airfoil, color, label in [
        (naca0012, "blue", "NACA 0012"),
        (naca2412, "red", "NACA 2412"),
    ]:
        upper = airfoil.upper_surface
        lower = airfoil.lower_surface

        # Plot airfoil
        plt.plot(upper[0], upper[1], color=color, linewidth=2, label=f"{label} upper")
        plt.plot(
            lower[0],
            lower[1],
            color=color,
            linewidth=2,
            linestyle="--",
            label=f"{label} lower",
        )

        # Add thickness lines at several stations
        x_thick = jnp.array([0.1, 0.3, 0.5, 0.7])
        for x in x_thick:
            y_u = airfoil.y_upper(jnp.array([x]))[0]
            y_l = airfoil.y_lower(jnp.array([x]))[0]
            plt.plot([x, x], [y_l, y_u], color=color, linewidth=1, alpha=0.7)

    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title("Thickness Visualization")
    plt.legend()

    # Plot 6: Curvature analysis (conceptual)
    plt.subplot(2, 3, 6)
    # This would require second derivatives - simplified version
    for airfoil, color, label in [
        (naca0012, "blue", "NACA 0012"),
        (naca2412, "red", "NACA 2412"),
    ]:
        upper = airfoil.upper_surface
        # Simple curvature approximation using finite differences
        x_u, y_u = upper[0], upper[1]
        dx = jnp.diff(x_u)
        dy = jnp.diff(y_u)
        curvature_approx = jnp.diff(dy / dx) / jnp.diff(x_u[:-1])
        x_curv = x_u[1:-1]

        plt.plot(
            x_curv,
            curvature_approx,
            color=color,
            linewidth=2,
            label=f"{label} upper",
        )

    plt.grid(True, alpha=0.3)
    plt.xlabel("x/c")
    plt.ylabel("Approximate Curvature")
    plt.title("Surface Curvature (Upper)")
    plt.legend()

    plt.tight_layout()
    plt.show()


def morphing_animation():
    """Create an animation of airfoil morphing."""
    print("\n=== Creating Morphing Animation ===")

    # Create two airfoils to morph between
    airfoil1 = NACA4(M=0.0, P=0.0, XX=0.12, n_points=50)  # NACA 0012
    airfoil2 = NACA4(M=0.04, P=0.4, XX=0.18, n_points=50)  # NACA 4418

    fig, ax = plt.subplots(figsize=(12, 6))

    # Initialize empty line objects
    (line_upper,) = ax.plot([], [], "b-", linewidth=3, label="Upper surface")
    (line_lower,) = ax.plot([], [], "r-", linewidth=3, label="Lower surface")
    fill = ax.fill_between([], [], [], alpha=0.3, color="lightblue")

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.15, 0.15)
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect("equal")

    # Text for displaying current blend ratio
    text = ax.text(0.02, 0.12, "", fontsize=14, fontweight="bold")

    def animate(frame):
        # Calculate blend ratio (0 to 1 and back)
        eta = 0.5 * (1 + jnp.sin(2 * jnp.pi * frame / 100))

        # Create morphed airfoil
        morphed = Airfoil.morph_new_from_two_foils(
            airfoil1,
            airfoil2,
            eta=eta,
            n_points=50,
        )

        # Get coordinates
        upper = morphed.upper_surface
        lower = morphed.lower_surface

        # Update lines
        line_upper.set_data(upper[0], upper[1])
        line_lower.set_data(lower[0], lower[1])

        # Update fill (remove old fill and create new one)
        ax.collections.clear()
        ax.fill_between(upper[0], upper[1], lower[1], alpha=0.3, color="lightblue")

        # Update text
        text.set_text(f"Morphing: η = {eta:.2f}\\n{airfoil1.name} → {airfoil2.name}")

        return line_upper, line_lower, text

    # Create animation
    anim = FuncAnimation(fig, animate, frames=200, interval=50, blit=False, repeat=True)

    plt.title("Airfoil Morphing Animation")
    plt.show()

    # Note: To save animation, uncomment the following line
    # anim.save('airfoil_morphing.gif', writer='pillow', fps=20)

    return anim


def advanced_visualization_concepts() -> None:
    """Demonstrate advanced visualization concepts."""
    print("\n=== Advanced Visualization Concepts ===")

    naca4415 = NACA4(M=0.04, P=0.4, XX=0.15, n_points=100)

    plt.figure(figsize=(15, 12))

    # Plot 1: Airfoil with flow field visualization concept
    plt.subplot(2, 3, 1)
    upper = naca4415.upper_surface
    lower = naca4415.lower_surface

    # Plot airfoil
    plt.fill_between(
        upper[0],
        upper[1],
        lower[1],
        alpha=0.8,
        color="gray",
        label="Airfoil",
    )
    plt.plot(upper[0], upper[1], "k-", linewidth=2)
    plt.plot(lower[0], lower[1], "k-", linewidth=2)

    # Add conceptual streamlines
    y_stream = jnp.linspace(-0.3, 0.3, 7)
    x_stream = jnp.linspace(-0.5, 1.5, 50)

    for y in y_stream:
        if abs(y) > 0.05:  # Avoid airfoil region
            # Simple streamline approximation
            y_line = y * jnp.ones_like(x_stream)
            # Add some curvature around airfoil
            mask = (x_stream > 0) & (x_stream < 1)
            y_line = jnp.where(
                mask,
                y_line + 0.1 * y * jnp.sin(jnp.pi * x_stream),
                y_line,
            )
            plt.plot(x_stream, y_line, "b-", alpha=0.6, linewidth=1)

    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.3, 0.3)
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title("Conceptual Flow Field")
    plt.grid(True, alpha=0.3)

    # Plot 2: Pressure distribution concept
    plt.subplot(2, 3, 2)
    x_eval = jnp.linspace(0, 1, 100)

    # Conceptual pressure coefficient (inverted for typical presentation)
    cp_upper = -2 * (naca4415.y_upper(x_eval) - naca4415.camber_line(x_eval))
    cp_lower = -2 * (naca4415.y_lower(x_eval) - naca4415.camber_line(x_eval))

    plt.plot(x_eval, cp_upper, "b-", linewidth=2, label="Upper surface")
    plt.plot(x_eval, cp_lower, "r-", linewidth=2, label="Lower surface")
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    plt.xlabel("x/c")
    plt.ylabel("Cp (conceptual)")
    plt.title("Conceptual Pressure Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()  # Typical Cp plot convention

    # Plot 3: Airfoil with coordinate system
    plt.subplot(2, 3, 3)
    plt.plot(upper[0], upper[1], "b-", linewidth=2, label="Upper surface")
    plt.plot(lower[0], lower[1], "r-", linewidth=2, label="Lower surface")

    # Add coordinate system
    plt.arrow(
        0,
        -0.1,
        0.2,
        0,
        head_width=0.01,
        head_length=0.02,
        fc="black",
        ec="black",
    )
    plt.arrow(
        0,
        -0.1,
        0,
        0.1,
        head_width=0.02,
        head_length=0.01,
        fc="black",
        ec="black",
    )
    plt.text(0.1, -0.12, "x", fontsize=12, ha="center")
    plt.text(-0.03, -0.05, "y", fontsize=12, ha="center")

    # Mark special points
    plt.plot(0, 0, "go", markersize=8, label="Leading edge")
    plt.plot(1, 0, "ro", markersize=8, label="Trailing edge")

    # Mark maximum thickness location
    max_t_loc = naca4415.max_thickness_location
    y_upper_max = naca4415.y_upper(jnp.array([max_t_loc]))[0]
    y_lower_max = naca4415.y_lower(jnp.array([max_t_loc]))[0]
    plt.plot([max_t_loc, max_t_loc], [y_lower_max, y_upper_max], "g--", linewidth=2)
    plt.plot(max_t_loc, (y_upper_max + y_lower_max) / 2, "go", markersize=6)
    plt.text(max_t_loc, (y_upper_max + y_lower_max) / 2 + 0.02, "Max t", ha="center")

    plt.axis("equal")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title("Airfoil Coordinate System")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 4: Surface normal vectors
    plt.subplot(2, 3, 4)
    plt.plot(upper[0], upper[1], "b-", linewidth=2, label="Upper surface")
    plt.plot(lower[0], lower[1], "r-", linewidth=2, label="Lower surface")

    # Add normal vectors at several points
    x_normals = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
    for x in x_normals:
        y_u = naca4415.y_upper(jnp.array([x]))[0]
        y_l = naca4415.y_lower(jnp.array([x]))[0]

        # Simple normal approximation (perpendicular to surface)
        if hasattr(naca4415, "camber_line_derivative"):
            slope = naca4415.camber_line_derivative(jnp.array([x]))[0]
            normal_x = -slope / jnp.sqrt(1 + slope**2)
            normal_y = 1 / jnp.sqrt(1 + slope**2)
        else:
            normal_x, normal_y = 0, 1

        # Scale normal vectors
        scale = 0.05
        plt.arrow(
            x,
            y_u,
            normal_x * scale,
            normal_y * scale,
            head_width=0.01,
            head_length=0.01,
            fc="blue",
            ec="blue",
            alpha=0.7,
        )
        plt.arrow(
            x,
            y_l,
            normal_x * scale,
            -normal_y * scale,
            head_width=0.01,
            head_length=0.01,
            fc="red",
            ec="red",
            alpha=0.7,
        )

    plt.axis("equal")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title("Surface Normal Vectors")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 5: Airfoil family comparison
    plt.subplot(2, 3, 5)
    thicknesses = [0.09, 0.12, 0.15, 0.18]
    colors = ["blue", "green", "red", "orange"]

    for i, thickness in enumerate(thicknesses):
        airfoil = NACA4(M=0.02, P=0.4, XX=thickness, n_points=50)
        upper = airfoil.upper_surface
        lower = airfoil.lower_surface
        plt.plot(
            upper[0],
            upper[1],
            color=colors[i],
            linewidth=2,
            label=f"NACA 24{int(thickness * 100):02d}",
        )
        plt.plot(lower[0], lower[1], color=colors[i], linewidth=2, linestyle="--")

    plt.axis("equal")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title("NACA 24XX Family")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 6: Design space visualization
    plt.subplot(2, 3, 6)
    cambers = jnp.linspace(0, 0.06, 20)
    thicknesses = jnp.linspace(0.08, 0.20, 20)

    C, T = jnp.meshgrid(cambers, thicknesses)

    # Create a design metric (e.g., lift-to-drag ratio proxy)
    design_metric = C * 10 + (0.15 - jnp.abs(T - 0.12)) * 5

    contour = plt.contourf(C, T, design_metric, levels=20, cmap="viridis")
    plt.colorbar(contour, label="Design Metric")

    # Mark some actual NACA airfoils
    naca_points = [(0.0, 0.12), (0.02, 0.12), (0.04, 0.15)]
    naca_names = ["0012", "2412", "4415"]

    for (c, t), name in zip(naca_points, naca_names):
        plt.plot(c, t, "ro", markersize=8)
        plt.annotate(
            f"NACA {name}",
            (c, t),
            xytext=(5, 5),
            textcoords="offset points",
            color="white",
            fontweight="bold",
        )

    plt.xlabel("Maximum Camber")
    plt.ylabel("Maximum Thickness")
    plt.title("Airfoil Design Space")

    plt.tight_layout()
    plt.show()


def main():
    """Main function demonstrating all visualization techniques."""
    print("JAX Airfoil Plotting and Visualization Examples")
    print("=" * 60)

    # Demonstrate various plotting techniques
    basic_airfoil_plotting()
    multi_airfoil_comparison()
    thickness_camber_analysis_plots()

    # Create morphing animation
    print("Creating morphing animation...")
    anim = morphing_animation()

    # Advanced visualization concepts
    advanced_visualization_concepts()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("1. JAX arrays work seamlessly with matplotlib")
    print("2. Multiple visualization styles enhance understanding")
    print("3. Comparative plots reveal airfoil characteristics")
    print("4. Animation can show dynamic behavior")
    print("5. Advanced plots support engineering analysis")
    print("6. Visualization aids in design space exploration")


if __name__ == "__main__":
    main()
