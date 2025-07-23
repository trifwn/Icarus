#!/usr/bin/env python3
"""
Basic Airfoil Creation Examples

This example demonstrates the fundamental ways to create airfoils using the JAX implementation:
1. Creating airfoils from coordinate arrays
2. Creating NACA 4-digit airfoils
3. Creating NACA 5-digit airfoils
4. Loading airfoils from files
5. Creating airfoils using the unified NACA interface

The JAX implementation provides the same API as the NumPy version while enabling
automatic differentiation and JIT compilation for performance.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from ICARUS.airfoils import Airfoil
from ICARUS.airfoils.naca4 import NACA4
from ICARUS.airfoils.naca5 import NACA5


def create_from_coordinates():
    """Create an airfoil from coordinate arrays."""
    print("=== Creating Airfoil from Coordinates ===")

    # Create a simple symmetric airfoil using coordinate arrays
    n_points = 50
    x = jnp.linspace(0, 1, n_points)

    # Simple parabolic shape for demonstration
    y_upper = 0.1 * jnp.sin(jnp.pi * x)  # Upper surface
    y_lower = -0.1 * jnp.sin(jnp.pi * x)  # Lower surface (symmetric)

    # Stack coordinates into the required format [x, y]
    upper_surface = jnp.stack([x, y_upper])
    lower_surface = jnp.stack([x, y_lower])

    # Create the airfoil
    airfoil = Airfoil(upper_surface, lower_surface, name="custom_symmetric")

    print(f"Created airfoil: {airfoil.name}")
    print(f"Number of points: {airfoil.n_points}")
    print(f"Max thickness: {airfoil.max_thickness:.4f}")
    print(f"Max thickness location: {airfoil.max_thickness_location:.4f}")

    return airfoil


def create_naca4_airfoils():
    """Create NACA 4-digit airfoils using different methods."""
    print("\n=== Creating NACA 4-Digit Airfoils ===")

    # Method 1: Direct parameter specification
    naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)
    print(f"Method 1 - Direct parameters: {naca2412.name}")
    print(
        f"  Camber: {naca2412.m:.3f}, Position: {naca2412.p:.1f}, Thickness: {naca2412.xx:.3f}",
    )

    # Method 2: From name string
    naca0012 = NACA4.from_name("NACA0012")
    print(f"Method 2 - From name: {naca0012.name}")
    print(
        f"  Camber: {naca0012.m:.3f}, Position: {naca0012.p:.1f}, Thickness: {naca0012.xx:.3f}",
    )

    # Method 3: From digit string
    naca4415 = NACA4.from_digits("4415")
    print(f"Method 3 - From digits: {naca4415.name}")
    print(
        f"  Camber: {naca4415.m:.3f}, Position: {naca4415.p:.1f}, Thickness: {naca4415.xx:.3f}",
    )

    # Method 4: Using the unified Airfoil.naca() interface
    naca6409 = Airfoil.naca("6409", n_points=150)
    print(f"Method 4 - Unified interface: {naca6409.name}")

    return [naca2412, naca0012, naca4415, naca6409]


def create_naca5_airfoils():
    """Create NACA 5-digit airfoils."""
    print("\n=== Creating NACA 5-Digit Airfoils ===")

    # NACA 5-digit airfoils have different parameters:
    # L: Type of camber line (2 = simple, 3 = reflexed)
    # P: Position of maximum camber (in tenths of chord)
    # Q: Reflex parameter (0 = no reflex, 1 = reflex)
    # XX: Maximum thickness (in hundredths of chord)

    try:
        # Create a NACA 23012 airfoil
        naca23012 = NACA5(L=2, P=3, Q=0, XX=12, n_points=200)
        print(f"Created NACA 5-digit: {naca23012.name}")

        # Using the unified interface
        naca23015 = Airfoil.naca("23015", n_points=150)
        print(f"Unified interface: {naca23015.name}")

        return [naca23012, naca23015]
    except Exception as e:
        print(f"Note: NACA 5-digit creation failed: {e}")
        print(
            "This may indicate NACA5 class needs to be implemented or imported differently",
        )
        return []


def demonstrate_airfoil_properties():
    """Demonstrate accessing airfoil properties."""
    print("\n=== Airfoil Properties ===")

    # Create a cambered airfoil for demonstration
    naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)

    print(f"Airfoil: {naca2412.name}")
    print(f"Maximum thickness: {naca2412.max_thickness:.4f}")
    print(f"Maximum thickness location: {naca2412.max_thickness_location:.4f}")

    # Access surface coordinates
    upper_surface = naca2412.upper_surface
    lower_surface = naca2412.lower_surface
    print(f"Upper surface shape: {upper_surface.shape}")
    print(f"Lower surface shape: {lower_surface.shape}")

    # Evaluate airfoil at specific x-coordinates
    x_eval = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    y_upper = naca2412.y_upper(x_eval)
    y_lower = naca2412.y_lower(x_eval)
    thickness = naca2412.thickness(x_eval)

    print("\nEvaluation at specific x-coordinates:")
    for i, x in enumerate(x_eval):
        print(
            f"  x={x:.2f}: y_upper={y_upper[i]:.4f}, y_lower={y_lower[i]:.4f}, thickness={thickness[i]:.4f}",
        )


def create_from_file_example():
    """Demonstrate creating airfoil from file (conceptual example)."""
    print("\n=== Creating Airfoil from File (Conceptual) ===")

    # This is a conceptual example - in practice you would have actual airfoil data files
    print("To create an airfoil from a file, use:")
    print("  airfoil = Airfoil.from_file('path/to/airfoil.dat')")
    print("")
    print("The file should contain x,y coordinate pairs, one per line:")
    print("  1.0000  0.0000")
    print("  0.9500  0.0123")
    print("  0.9000  0.0234")
    print("  ...")
    print("")
    print("The coordinates should be in Selig format (trailing edge to trailing edge)")


def plot_created_airfoils():
    """Create a comparison plot of different airfoils."""
    print("\n=== Plotting Created Airfoils ===")

    # Create several airfoils for comparison
    airfoils = [
        NACA4(M=0.0, P=0.0, XX=0.12, n_points=100),  # NACA 0012 (symmetric)
        NACA4(M=0.02, P=0.4, XX=0.12, n_points=100),  # NACA 2412 (cambered)
        NACA4(M=0.04, P=0.4, XX=0.15, n_points=100),  # NACA 4415 (high camber)
    ]

    plt.figure(figsize=(12, 8))

    for i, airfoil in enumerate(airfoils):
        # Get coordinates for plotting
        upper = airfoil.upper_surface
        lower = airfoil.lower_surface

        plt.subplot(2, 2, i + 1)
        plt.plot(upper[0], upper[1], "b-", label="Upper surface", linewidth=2)
        plt.plot(lower[0], lower[1], "r-", label="Lower surface", linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.xlabel("x/c")
        plt.ylabel("y/c")
        plt.title(f"{airfoil.name}\nMax thickness: {airfoil.max_thickness:.3f}")
        plt.legend()

    # Create a comparison plot
    plt.subplot(2, 2, 4)
    colors = ["blue", "red", "green"]
    for i, airfoil in enumerate(airfoils):
        upper = airfoil.upper_surface
        lower = airfoil.lower_surface
        plt.plot(
            upper[0],
            upper[1],
            color=colors[i],
            linewidth=2,
            label=f"{airfoil.name} upper",
        )
        plt.plot(
            lower[0],
            lower[1],
            color=colors[i],
            linewidth=2,
            linestyle="--",
            label=f"{airfoil.name} lower",
        )

    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title("Airfoil Comparison")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


def main():
    """Main function demonstrating all airfoil creation methods."""
    print("JAX Airfoil Creation Examples")
    print("=" * 50)

    # Demonstrate different creation methods
    custom_airfoil = create_from_coordinates()
    naca4_airfoils = create_naca4_airfoils()
    naca5_airfoils = create_naca5_airfoils()

    # Show airfoil properties
    demonstrate_airfoil_properties()

    # File creation example (conceptual)
    create_from_file_example()

    # Create visualization
    plot_created_airfoils()

    print("\n" + "=" * 50)
    print("Key Takeaways:")
    print("1. JAX airfoils use the same API as NumPy versions")
    print("2. Multiple creation methods available (coordinates, NACA, files)")
    print("3. All operations return JAX arrays for automatic differentiation")
    print("4. Properties and methods work identically to NumPy implementation")
    print("5. Ready for JIT compilation and gradient computation")


if __name__ == "__main__":
    main()
