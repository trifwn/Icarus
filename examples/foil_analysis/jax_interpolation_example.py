"""
Example demonstrating the use of the JAX interpolation engine for airfoil analysis.

This script shows how to use the JaxInterpolationEngine to:
1. Interpolate airfoil coordinates
2. Compute thickness distribution
3. Visualize the results
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ICARUS.airfoils import NACA4
from ICARUS.airfoils.jax_implementation.interpolation_engine import (
    JaxInterpolationEngine,
)


def main():
    # Create a NACA 4-digit airfoil
    airfoil = NACA4.from_digits("2412")

    # Get coordinates
    x_upper, y_upper = airfoil.get_upper_surface_points(n_points=50)
    x_lower, y_lower = airfoil.get_lower_surface_points(n_points=50)

    # Create query points
    query_x = np.linspace(0.0, 1.0, 100)

    # Convert to JAX arrays
    upper_coords = jnp.array([x_upper, y_upper])
    lower_coords = jnp.array([x_lower, y_lower])
    query_x_jax = jnp.array(query_x)

    # Interpolate using both linear and cubic methods
    upper_y_linear = JaxInterpolationEngine.interpolate_with_method(
        jnp.array(x_upper),
        jnp.array(y_upper),
        len(x_upper),
        query_x_jax,
        method="linear",
    )

    upper_y_cubic = JaxInterpolationEngine.interpolate_with_method(
        jnp.array(x_upper),
        jnp.array(y_upper),
        len(x_upper),
        query_x_jax,
        method="cubic",
    )

    lower_y_linear = JaxInterpolationEngine.interpolate_with_method(
        jnp.array(x_lower),
        jnp.array(y_lower),
        len(x_lower),
        query_x_jax,
        method="linear",
    )

    lower_y_cubic = JaxInterpolationEngine.interpolate_with_method(
        jnp.array(x_lower),
        jnp.array(y_lower),
        len(x_lower),
        query_x_jax,
        method="cubic",
    )

    # Compute thickness distribution
    thickness = JaxInterpolationEngine.compute_thickness_distribution(
        upper_coords,
        lower_coords,
        len(x_upper),
        query_x_jax,
    )

    # Convert results to numpy for plotting
    upper_y_linear_np = np.array(upper_y_linear)
    upper_y_cubic_np = np.array(upper_y_cubic)
    lower_y_linear_np = np.array(lower_y_linear)
    lower_y_cubic_np = np.array(lower_y_cubic)
    thickness_np = np.array(thickness)

    # Plot the results
    plt.figure(figsize=(12, 8))

    # Plot airfoil shape
    plt.subplot(2, 1, 1)
    plt.plot(x_upper, y_upper, "bo", label="Upper Surface Points")
    plt.plot(x_lower, y_lower, "ro", label="Lower Surface Points")
    plt.plot(query_x, upper_y_linear_np, "b-", label="Linear Interpolation (Upper)")
    plt.plot(query_x, upper_y_cubic_np, "b--", label="Cubic Interpolation (Upper)")
    plt.plot(query_x, lower_y_linear_np, "r-", label="Linear Interpolation (Lower)")
    plt.plot(query_x, lower_y_cubic_np, "r--", label="Cubic Interpolation (Lower)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("NACA 2412 Airfoil - JAX Interpolation")

    # Plot thickness distribution
    plt.subplot(2, 1, 2)
    plt.plot(query_x, thickness_np, "g-", label="Thickness Distribution")
    plt.grid(True)
    plt.xlabel("x/c")
    plt.ylabel("Thickness")
    plt.title("Airfoil Thickness Distribution")
    plt.legend()

    plt.tight_layout()
    plt.savefig("jax_interpolation_example.png")
    plt.show()


if __name__ == "__main__":
    main()
