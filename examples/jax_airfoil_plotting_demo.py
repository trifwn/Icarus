#!/usr/bin/env python3
"""
Demonstration of JAX Airfoil plotting capabilities.

This script showcases the various plotting features available in the JAX Airfoil
implementation, including basic plotting, batch plotting, camber line visualization,
thickness distribution plots, and advanced analysis features.
"""

import matplotlib.pyplot as plt

from ICARUS.airfoils.jax_implementation import AirfoilPlotter
from ICARUS.airfoils.jax_implementation import JaxAirfoil


def demo_basic_plotting():
    """Demonstrate basic airfoil plotting functionality."""
    print("=== Basic Plotting Demo ===")

    # Create a NACA 2412 airfoil
    airfoil = JaxAirfoil.naca("2412", n_points=100)

    # Create a figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Basic plot
    airfoil.plot(ax=ax1)
    ax1.set_title("Basic Airfoil Plot")

    # Plot with camber line
    airfoil.plot(ax=ax2, camber=True, max_thickness=True)
    ax2.set_title("With Camber Line and Max Thickness")

    # Scatter plot
    airfoil.plot(ax=ax3, scatter=True, markersize=3.0, alpha=0.7)
    ax3.set_title("Scatter Plot Mode")

    # Custom styling
    airfoil.plot(
        ax=ax4,
        camber=True,
        overide_color="purple",
        linewidth=2.5,
        alpha=0.8,
        show_legend=True,
    )
    ax4.set_title("Custom Styling")

    plt.tight_layout()
    plt.savefig("demo_basic_plotting.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Basic plotting demo completed!\n")


def demo_batch_plotting():
    """Demonstrate batch plotting functionality."""
    print("=== Batch Plotting Demo ===")

    # Create multiple airfoils
    airfoils = [
        JaxAirfoil.naca("0012", n_points=80),
        JaxAirfoil.naca("2412", n_points=80),
        JaxAirfoil.naca("4412", n_points=80),
        JaxAirfoil.naca("6412", n_points=80),
    ]

    names = ["NACA 0012 (Symmetric)", "NACA 2412", "NACA 4412", "NACA 6412"]
    colors = ["blue", "red", "green", "orange"]

    # Create batch plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Line plot
    AirfoilPlotter.plot_batch(
        airfoils,
        names=names,
        colors=colors,
        ax=ax1,
        alpha=0.8,
        linewidth=2.0,
        legend=True,
    )
    ax1.set_title("Batch Line Plot")

    # Scatter plot
    AirfoilPlotter.plot_batch(
        airfoils,
        names=names,
        colors=colors,
        ax=ax2,
        scatter=True,
        alpha=0.6,
        legend=True,
    )
    ax2.set_title("Batch Scatter Plot")

    plt.tight_layout()
    plt.savefig("demo_batch_plotting.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Batch plotting demo completed!\n")


def demo_analysis_plots():
    """Demonstrate analysis plotting functionality."""
    print("=== Analysis Plots Demo ===")

    # Create airfoils for comparison
    airfoils = [
        JaxAirfoil.naca("0012", n_points=100),
        JaxAirfoil.naca("2412", n_points=100),
        JaxAirfoil.naca("4412", n_points=100),
    ]

    names = ["NACA 0012", "NACA 2412", "NACA 4412"]

    # Create analysis plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Camber comparison
    AirfoilPlotter.plot_camber_comparison(airfoils, names=names, ax=ax1)
    ax1.set_title("Camber Line Comparison")

    # Thickness comparison
    AirfoilPlotter.plot_thickness_comparison(airfoils, names=names, ax=ax2)
    ax2.set_title("Thickness Distribution Comparison")

    # Comprehensive analysis of single airfoil
    AirfoilPlotter.plot_airfoil_analysis(
        airfoils[1],  # NACA 2412
        show_camber=True,
        show_thickness=True,
        show_max_thickness=True,
        show_max_camber=True,
        ax=ax3,
    )
    ax3.set_title("Comprehensive Analysis")

    # Debug plot
    AirfoilPlotter.debug_coordinate_plot(
        airfoils[1],
        show_indices=False,  # Too cluttered with indices
        show_buffer=True,
        ax=ax4,
    )
    ax4.set_title("Debug Coordinate Plot")

    plt.tight_layout()
    plt.savefig("demo_analysis_plots.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Analysis plots demo completed!\n")


def demo_advanced_features():
    """Demonstrate advanced plotting features."""
    print("=== Advanced Features Demo ===")

    # Create base airfoils for morphing
    airfoil1 = JaxAirfoil.naca("0012", n_points=80)
    airfoil2 = JaxAirfoil.naca("4412", n_points=80)

    # Morphing sequence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot morphing sequence
    AirfoilPlotter.plot_morphing_sequence(airfoil1, airfoil2, n_steps=7, alpha=0.7)
    plt.savefig("demo_morphing_sequence.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Subplot grid
    morphed_airfoils = []
    eta_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    for eta in eta_values:
        morphed = JaxAirfoil.morph_new_from_two_foils(
            airfoil1,
            airfoil2,
            eta,
            n_points=100,
        )
        morphed_airfoils.append(morphed)

    names = [f"η = {eta:.2f}" for eta in eta_values]

    fig, axes = AirfoilPlotter.create_subplot_grid(
        morphed_airfoils,
        names=names,
        ncols=3,
        figsize=(15, 8),
        show_camber=True,
    )
    plt.savefig("demo_subplot_grid.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Advanced features demo completed!\n")


def demo_flap_visualization():
    """Demonstrate flap operation visualization."""
    print("=== Flap Visualization Demo ===")

    # Create base airfoil
    base_airfoil = JaxAirfoil.naca("2412", n_points=100)

    # Create flapped versions
    flap_angles = [0, 10, 20, 30]
    flapped_airfoils = []

    for angle in flap_angles:
        if angle == 0:
            flapped_airfoils.append(base_airfoil)
        else:
            flapped = base_airfoil.flap(
                flap_hinge_chord_percentage=0.75,
                flap_angle=angle,
            )
            flapped_airfoils.append(flapped)

    names = [f"Flap: {angle}°" for angle in flap_angles]
    colors = ["blue", "green", "orange", "red"]

    # Plot flap sequence
    ax = AirfoilPlotter.plot_batch(
        flapped_airfoils,
        names=names,
        colors=colors,
        alpha=0.8,
        linewidth=2.0,
        legend=True,
    )
    ax.set_title("Flap Deflection Sequence (NACA 2412)")

    plt.savefig("demo_flap_visualization.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Flap visualization demo completed!\n")


def main():
    """Run all plotting demonstrations."""
    print("JAX Airfoil Plotting Demonstration")
    print("=" * 40)

    # Set matplotlib backend for better compatibility
    plt.rcParams["figure.max_open_warning"] = 0

    try:
        # Run all demos
        demo_basic_plotting()
        demo_batch_plotting()
        demo_analysis_plots()
        demo_advanced_features()
        demo_flap_visualization()

        print("All plotting demonstrations completed successfully!")
        print("Check the generated PNG files for the results.")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
