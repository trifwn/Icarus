#!/usr/bin/env python3
"""
Basic Airfoil Operations Examples

This example demonstrates fundamental operations on JAX airfoils:
1. Surface evaluation and interpolation
2. Thickness and camber calculations
3. Geometric properties
4. Coordinate transformations
5. Basic morphing operations

All operations leverage JAX for automatic differentiation and JIT compilation.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import grad
from jax import jit

from ICARUS.airfoils import Airfoil
from ICARUS.airfoils.naca4 import NACA4


def surface_evaluation_demo():
    """Demonstrate surface evaluation and interpolation."""
    print("=== Surface Evaluation and Interpolation ===")

    # Create a NACA airfoil
    naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)

    # Evaluate surfaces at specific x-coordinates
    x_coords = jnp.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])

    print(f"Evaluating {naca2412.name} at x-coordinates: {x_coords}")

    # Get upper and lower surface y-coordinates
    y_upper = naca2412.y_upper(x_coords)
    y_lower = naca2412.y_lower(x_coords)

    print("\nSurface coordinates:")
    print("x/c     y_upper    y_lower    thickness")
    print("-" * 40)
    for i, x in enumerate(x_coords):
        thickness = y_upper[i] - y_lower[i]
        print(f"{x:.2f}    {y_upper[i]:8.5f}   {y_lower[i]:8.5f}   {thickness:8.5f}")

    # Demonstrate vectorized evaluation
    x_fine = jnp.linspace(0, 1, 1000)
    y_upper_fine = naca2412.y_upper(x_fine)
    y_lower_fine = naca2412.y_lower(x_fine)

    print(f"\nVectorized evaluation at {len(x_fine)} points completed successfully")
    print(
        f"Upper surface range: [{jnp.min(y_upper_fine):.5f}, {jnp.max(y_upper_fine):.5f}]",
    )
    print(
        f"Lower surface range: [{jnp.min(y_lower_fine):.5f}, {jnp.max(y_lower_fine):.5f}]",
    )


def thickness_and_camber_analysis():
    """Analyze thickness distribution and camber properties."""
    print("\n=== Thickness and Camber Analysis ===")

    # Create both symmetric and cambered airfoils
    naca0012 = NACA4(M=0.0, P=0.0, XX=0.12, n_points=100)  # Symmetric
    naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)  # Cambered

    airfoils = [naca0012, naca2412]

    for airfoil in airfoils:
        print(f"\nAnalyzing {airfoil.name}:")

        # Basic thickness properties
        print(f"  Maximum thickness: {airfoil.max_thickness:.4f}")
        print(f"  Max thickness location: {airfoil.max_thickness_location:.4f}")

        # Thickness distribution at key locations
        x_eval = jnp.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        thickness_dist = airfoil.thickness(x_eval)

        print("  Thickness distribution:")
        print("    x/c:  ", " ".join(f"{x:.2f}" for x in x_eval))
        print("    t/c:  ", " ".join(f"{t:.3f}" for t in thickness_dist))

        # For NACA 4-digit airfoils, we can also access camber properties
        if hasattr(airfoil, "camber_line"):
            camber_values = airfoil.camber_line(x_eval)
            print("  Camber distribution:")
            print("    y_c:  ", " ".join(f"{c:.4f}" for c in camber_values))


def geometric_properties_demo():
    """Demonstrate calculation of geometric properties."""
    print("\n=== Geometric Properties ===")

    # Create airfoils with different characteristics
    airfoils = [
        NACA4(M=0.0, P=0.0, XX=0.09, n_points=100),  # Thin symmetric
        NACA4(M=0.0, P=0.0, XX=0.18, n_points=100),  # Thick symmetric
        NACA4(M=0.04, P=0.4, XX=0.12, n_points=100),  # High camber
        NACA4(M=0.02, P=0.2, XX=0.12, n_points=100),  # Forward camber
    ]

    print("Airfoil        Max t/c   t/c loc   Camber   Camber loc")
    print("-" * 55)

    for airfoil in airfoils:
        max_t = airfoil.max_thickness
        max_t_loc = airfoil.max_thickness_location

        # For NACA airfoils, we know the design parameters
        if hasattr(airfoil, "m") and hasattr(airfoil, "p"):
            camber = float(airfoil.m)
            camber_loc = float(airfoil.p)
        else:
            camber = 0.0
            camber_loc = 0.0

        print(
            f"{airfoil.name:<12} {max_t:7.4f}   {max_t_loc:7.4f}   {camber:6.3f}    {camber_loc:6.3f}",
        )


def coordinate_access_demo():
    """Demonstrate accessing and working with coordinate arrays."""
    print("\n=== Coordinate Access and Manipulation ===")

    naca4415 = NACA4(M=0.04, P=0.4, XX=0.15, n_points=100)

    # Access surface coordinates
    upper_surface = naca4415.upper_surface  # Shape: (2, n_points)
    lower_surface = naca4415.lower_surface  # Shape: (2, n_points)

    print(f"Upper surface shape: {upper_surface.shape}")
    print(f"Lower surface shape: {lower_surface.shape}")

    # Extract x and y coordinates
    x_upper, y_upper = upper_surface[0], upper_surface[1]
    x_lower, y_lower = lower_surface[0], lower_surface[1]

    print(f"X-coordinate range: [{jnp.min(x_upper):.3f}, {jnp.max(x_upper):.3f}]")
    print(f"Y-coordinate range: [{jnp.min(y_lower):.3f}, {jnp.max(y_upper):.3f}]")

    # Get airfoil in Selig format (trailing edge to trailing edge)
    selig_coords = naca4415.to_selig()
    print(f"Selig format shape: {selig_coords.shape}")
    print(f"First point (TE): ({selig_coords[0, 0]:.3f}, {selig_coords[1, 0]:.3f})")
    print(f"Last point (TE):  ({selig_coords[0, -1]:.3f}, {selig_coords[1, -1]:.3f})")


def morphing_operations_demo():
    """Demonstrate basic morphing operations between airfoils."""
    print("\n=== Basic Morphing Operations ===")

    # Create two different airfoils to morph between
    airfoil1 = NACA4(M=0.0, P=0.0, XX=0.12, n_points=100)  # NACA 0012
    airfoil2 = NACA4(M=0.04, P=0.4, XX=0.15, n_points=100)  # NACA 4415

    print(f"Morphing between {airfoil1.name} and {airfoil2.name}")

    # Create morphed airfoils at different blend ratios
    morph_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    morphed_airfoils = []

    for eta in morph_ratios:
        morphed = Airfoil.morph_new_from_two_foils(
            airfoil1,
            airfoil2,
            eta=eta,
            n_points=100,
        )
        morphed_airfoils.append(morphed)
        print(f"  η = {eta:.2f}: {morphed.name}")
        print(f"    Max thickness: {morphed.max_thickness:.4f}")

    return morphed_airfoils


def jax_specific_features_demo():
    """Demonstrate JAX-specific features like JIT compilation."""
    print("\n=== JAX-Specific Features ===")

    # Create a function that can be JIT compiled
    @jit
    def compute_airfoil_properties(m, p, xx):
        """JIT-compiled function to compute airfoil properties."""
        naca = NACA4(M=m, P=p, XX=xx, n_points=50)

        # Evaluate at standard locations
        x_eval = jnp.linspace(0, 1, 21)
        thickness = naca.thickness(x_eval)

        return {
            "max_thickness": naca.max_thickness,
            "max_thickness_location": naca.max_thickness_location,
            "mean_thickness": jnp.mean(thickness),
        }

    # Test JIT compilation
    print("Testing JIT compilation...")

    # First call will compile
    result1 = compute_airfoil_properties(0.02, 0.4, 0.12)
    print(f"NACA 2412 properties: {result1}")

    # Subsequent calls use compiled version
    result2 = compute_airfoil_properties(0.04, 0.3, 0.15)
    print(f"NACA 4315 properties: {result2}")

    print("JIT compilation successful - subsequent calls will be faster!")


def gradient_computation_demo():
    """Demonstrate automatic differentiation capabilities."""
    print("\n=== Automatic Differentiation Demo ===")

    def thickness_objective(params):
        """Objective function for thickness at a specific location."""
        m, p, xx = params
        naca = NACA4(M=m, P=p, XX=xx, n_points=50)
        # Return thickness at 30% chord
        return naca.thickness(jnp.array([0.3]))[0]

    # Compute gradient of thickness with respect to design parameters
    grad_fn = grad(thickness_objective)

    # Test parameters: NACA 2412
    params = jnp.array([0.02, 0.4, 0.12])

    thickness_value = thickness_objective(params)
    gradients = grad_fn(params)

    print(f"Thickness at 30% chord: {thickness_value:.5f}")
    print(
        f"Gradients w.r.t. [M, P, XX]: [{gradients[0]:.5f}, {gradients[1]:.5f}, {gradients[2]:.5f}]",
    )
    print("This enables gradient-based optimization!")


def plot_operations_results():
    """Create visualizations of the operations demonstrated."""
    print("\n=== Creating Visualization ===")

    # Create airfoils for plotting
    naca0012 = NACA4(M=0.0, P=0.0, XX=0.12, n_points=100)
    naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Airfoil shapes
    ax1 = axes[0, 0]
    for airfoil, color, label in [
        (naca0012, "blue", "NACA 0012"),
        (naca2412, "red", "NACA 2412"),
    ]:
        upper = airfoil.upper_surface
        lower = airfoil.lower_surface
        ax1.plot(upper[0], upper[1], color=color, linewidth=2, label=f"{label} upper")
        ax1.plot(
            lower[0],
            lower[1],
            color=color,
            linewidth=2,
            linestyle="--",
            label=f"{label} lower",
        )

    ax1.set_xlabel("x/c")
    ax1.set_ylabel("y/c")
    ax1.set_title("Airfoil Shapes")
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")
    ax1.legend()

    # Plot 2: Thickness distribution
    ax2 = axes[0, 1]
    x_eval = jnp.linspace(0, 1, 100)
    for airfoil, color, label in [
        (naca0012, "blue", "NACA 0012"),
        (naca2412, "red", "NACA 2412"),
    ]:
        thickness = airfoil.thickness(x_eval)
        ax2.plot(x_eval, thickness, color=color, linewidth=2, label=label)

    ax2.set_xlabel("x/c")
    ax2.set_ylabel("Thickness (t/c)")
    ax2.set_title("Thickness Distribution")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Camber line (for NACA 2412)
    ax3 = axes[1, 0]
    if hasattr(naca2412, "camber_line"):
        camber = naca2412.camber_line(x_eval)
        ax3.plot(x_eval, camber, "red", linewidth=2, label="Camber line")
        ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5)

    ax3.set_xlabel("x/c")
    ax3.set_ylabel("Camber (y_c/c)")
    ax3.set_title("Camber Line (NACA 2412)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Morphing demonstration
    ax4 = axes[1, 1]
    airfoil1 = NACA4(M=0.0, P=0.0, XX=0.12, n_points=50)
    airfoil2 = NACA4(M=0.04, P=0.4, XX=0.15, n_points=50)

    colors = ["blue", "green", "orange", "red"]
    for i, eta in enumerate([0.0, 0.33, 0.67, 1.0]):
        morphed = Airfoil.morph_new_from_two_foils(
            airfoil1,
            airfoil2,
            eta=eta,
            n_points=50,
        )
        upper = morphed.upper_surface
        ax4.plot(
            upper[0],
            upper[1],
            color=colors[i],
            linewidth=2,
            label=f"η = {eta:.2f}",
        )

    ax4.set_xlabel("x/c")
    ax4.set_ylabel("y/c")
    ax4.set_title("Airfoil Morphing")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.show()


def main():
    """Main function demonstrating all basic operations."""
    print("JAX Airfoil Basic Operations Examples")
    print("=" * 50)

    # Demonstrate various operations
    surface_evaluation_demo()
    thickness_and_camber_analysis()
    geometric_properties_demo()
    coordinate_access_demo()
    morphed_airfoils = morphing_operations_demo()
    jax_specific_features_demo()
    gradient_computation_demo()

    # Create visualizations
    plot_operations_results()

    print("\n" + "=" * 50)
    print("Key Takeaways:")
    print("1. Surface evaluation works with both scalar and vector inputs")
    print("2. Thickness and geometric properties are easily accessible")
    print("3. Coordinate arrays are JAX arrays enabling differentiation")
    print("4. Morphing operations create new airfoils with blended properties")
    print("5. JIT compilation accelerates repeated computations")
    print("6. Automatic differentiation enables gradient-based optimization")


if __name__ == "__main__":
    main()
