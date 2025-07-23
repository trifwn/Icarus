#!/usr/bin/env python3
"""
Advanced Morphing Operations with Gradient Computation

This example demonstrates advanced airfoil morphing capabilities using JAX:
1. Complex morphing between multiple airfoils
2. Parametric morphing with gradient computation
3. Shape optimization using morphing parameters
4. Multi-objective morphing with constraints
5. Time-dependent morphing for dynamic applications

The JAX implementation enables automatic differentiation through all morphing
operations, making it ideal for gradient-based optimization workflows.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad
from jax import jit

from ICARUS.airfoils import Airfoil
from ICARUS.airfoils.naca4 import NACA4


def basic_morphing_with_gradients():
    """Demonstrate morphing with gradient computation."""
    print("=== Basic Morphing with Gradients ===")

    # Create source and target airfoils
    source = NACA4(M=0.0, P=0.0, XX=0.12, n_points=100)  # NACA 0012
    target = NACA4(M=0.04, P=0.4, XX=0.15, n_points=100)  # NACA 4415

    print(f"Morphing from {source.name} to {target.name}")

    def morphing_objective(eta):
        """Objective function for morphing parameter optimization."""
        morphed = Airfoil.morph_new_from_two_foils(
            source,
            target,
            eta=eta,
            n_points=100,
        )

        # Example objective: minimize thickness at 50% chord
        thickness_at_mid = morphed.thickness(jnp.array([0.5]))[0]
        return thickness_at_mid

    # Compute gradient of morphing objective
    grad_morphing = grad(morphing_objective)

    # Test at different morphing parameters
    eta_values = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])

    print("\nMorphing parameter analysis:")
    print("η      Thickness@50%   Gradient")
    print("-" * 35)

    for eta in eta_values:
        thickness = morphing_objective(eta)
        gradient = grad_morphing(eta)
        print(f"{eta:.2f}   {thickness:.5f}      {gradient:.5f}")

    return source, target


def parametric_morphing_optimization():
    """Demonstrate parametric morphing with optimization."""
    print("\n=== Parametric Morphing Optimization ===")

    # Create a family of airfoils to morph between
    airfoils = [
        NACA4(M=0.0, P=0.0, XX=0.09, n_points=100),  # Thin symmetric
        NACA4(M=0.02, P=0.4, XX=0.12, n_points=100),  # Moderate camber
        NACA4(M=0.04, P=0.3, XX=0.15, n_points=100),  # High camber
        NACA4(M=0.01, P=0.5, XX=0.18, n_points=100),  # Thick, aft camber
    ]

    def multi_airfoil_morph(weights):
        """Create morphed airfoil from weighted combination of multiple airfoils."""
        # Normalize weights to sum to 1
        weights = weights / jnp.sum(weights)

        # Start with first airfoil
        result_upper = weights[0] * airfoils[0].upper_surface
        result_lower = weights[0] * airfoils[0].lower_surface

        # Add weighted contributions from other airfoils
        for i in range(1, len(airfoils)):
            result_upper += weights[i] * airfoils[i].upper_surface
            result_lower += weights[i] * airfoils[i].lower_surface

        # Create new airfoil from morphed coordinates
        morphed = Airfoil(result_upper, result_lower, name="morphed_multi")
        return morphed

    def optimization_objective(weights):
        """Multi-objective function for airfoil optimization."""
        morphed = multi_airfoil_morph(weights)

        # Objective 1: Target thickness at 30% chord
        target_thickness = 0.13
        thickness_30 = morphed.thickness(jnp.array([0.3]))[0]
        thickness_error = (thickness_30 - target_thickness) ** 2

        # Objective 2: Minimize maximum thickness
        x_eval = jnp.linspace(0.05, 0.95, 50)
        thickness_dist = morphed.thickness(x_eval)
        max_thickness_penalty = jnp.max(thickness_dist) ** 2

        # Combined objective
        return thickness_error + 0.1 * max_thickness_penalty

    # Optimize morphing weights
    grad_objective = grad(optimization_objective)

    # Initial weights (equal contribution)
    weights = jnp.array([0.25, 0.25, 0.25, 0.25])
    learning_rate = 0.1

    print("Optimizing morphing weights...")
    print("Iter   Objective    Weights")
    print("-" * 40)

    for iteration in range(10):
        obj_value = optimization_objective(weights)
        gradients = grad_objective(weights)

        # Gradient descent step
        weights = weights - learning_rate * gradients
        weights = jnp.maximum(weights, 0.01)  # Keep weights positive

        if iteration % 2 == 0:
            print(f"{iteration:2d}     {obj_value:.6f}    {weights}")

    # Create optimized airfoil
    optimized_airfoil = multi_airfoil_morph(weights)
    print(f"\nOptimized airfoil max thickness: {optimized_airfoil.max_thickness:.4f}")

    return airfoils, optimized_airfoil, weights


def dynamic_morphing_sequence():
    """Demonstrate time-dependent morphing for dynamic applications."""
    print("\n=== Dynamic Morphing Sequence ===")

    # Create keyframe airfoils
    keyframes = [
        NACA4(M=0.0, P=0.0, XX=0.12, n_points=100),  # t=0: Symmetric
        NACA4(M=0.03, P=0.3, XX=0.14, n_points=100),  # t=1: Forward camber
        NACA4(M=0.02, P=0.6, XX=0.16, n_points=100),  # t=2: Aft camber
        NACA4(M=0.0, P=0.0, XX=0.10, n_points=100),  # t=3: Return to symmetric
    ]

    def smooth_interpolation(t, keyframe_times):
        """Smooth interpolation between keyframes using cubic splines."""
        # Simple linear interpolation for demonstration
        # In practice, you might use more sophisticated interpolation
        n_keyframes = len(keyframe_times)

        # Find surrounding keyframes
        for i in range(n_keyframes - 1):
            if keyframe_times[i] <= t <= keyframe_times[i + 1]:
                # Linear interpolation between keyframes i and i+1
                dt = keyframe_times[i + 1] - keyframe_times[i]
                alpha = (t - keyframe_times[i]) / dt
                return i, alpha

        # Handle edge cases
        if t <= keyframe_times[0]:
            return 0, 0.0
        else:
            return n_keyframes - 2, 1.0

    def dynamic_morph(t):
        """Create airfoil at time t using dynamic morphing."""
        keyframe_times = jnp.array([0.0, 1.0, 2.0, 3.0])

        # Get interpolation parameters
        i, alpha = smooth_interpolation(t, keyframe_times)
        i = int(i)  # Convert to integer for indexing

        # Morph between adjacent keyframes
        morphed = Airfoil.morph_new_from_two_foils(
            keyframes[i],
            keyframes[i + 1],
            eta=alpha,
            n_points=100,
        )
        return morphed

    # Create morphing sequence
    time_points = jnp.linspace(0, 3, 13)  # 13 time points
    morphed_sequence = []

    print("Creating dynamic morphing sequence...")
    print("Time   Max Thickness   Max Thickness Location")
    print("-" * 45)

    for t in time_points:
        morphed = dynamic_morph(float(t))
        morphed_sequence.append(morphed)
        print(
            f"{t:.1f}    {morphed.max_thickness:.4f}         {morphed.max_thickness_location:.3f}",
        )

    return morphed_sequence, time_points


def gradient_based_shape_optimization():
    """Demonstrate gradient-based shape optimization using morphing."""
    print("\n=== Gradient-Based Shape Optimization ===")

    # Define base airfoils for morphing space
    base_airfoils = [
        NACA4(M=0.0, P=0.0, XX=0.12, n_points=100),  # Symmetric
        NACA4(M=0.04, P=0.2, XX=0.12, n_points=100),  # Forward camber
        NACA4(M=0.04, P=0.6, XX=0.12, n_points=100),  # Aft camber
        NACA4(M=0.0, P=0.0, XX=0.18, n_points=100),  # Thick symmetric
    ]

    @jit
    def create_morphed_airfoil(params):
        """Create airfoil from morphing parameters."""
        # Normalize parameters to create valid weights
        weights = jnp.exp(params) / jnp.sum(jnp.exp(params))

        # Weighted combination of base airfoils
        result_upper = jnp.zeros_like(base_airfoils[0].upper_surface)
        result_lower = jnp.zeros_like(base_airfoils[0].lower_surface)

        for i, airfoil in enumerate(base_airfoils):
            result_upper += weights[i] * airfoil.upper_surface
            result_lower += weights[i] * airfoil.lower_surface

        return result_upper, result_lower

    @jit
    def aerodynamic_objective(params):
        """Simplified aerodynamic objective function."""
        upper_surface, lower_surface = create_morphed_airfoil(params)

        # Create temporary airfoil for analysis
        # In practice, this would interface with CFD or panel methods

        # Simplified objectives:
        # 1. Minimize drag (approximated by thickness)
        x_eval = jnp.linspace(0.1, 0.9, 20)
        thickness = upper_surface[1, :] - lower_surface[1, :]  # Approximate thickness
        thickness_interp = jnp.interp(x_eval, upper_surface[0, :], thickness)
        drag_proxy = jnp.mean(thickness_interp**2)

        # 2. Maintain lift (approximated by camber)
        camber = 0.5 * (upper_surface[1, :] + lower_surface[1, :])
        camber_interp = jnp.interp(x_eval, upper_surface[0, :], camber)
        lift_proxy = jnp.mean(camber_interp)
        target_lift = 0.02
        lift_error = (lift_proxy - target_lift) ** 2

        # Combined objective
        return drag_proxy + 10.0 * lift_error

    # Optimization using gradient descent
    grad_objective = grad(aerodynamic_objective)

    # Initial parameters (equal weights)
    params = jnp.array([0.0, 0.0, 0.0, 0.0])
    learning_rate = 0.1

    print("Gradient-based shape optimization:")
    print("Iter   Objective    Parameters")
    print("-" * 50)

    optimization_history = []

    for iteration in range(15):
        obj_value = aerodynamic_objective(params)
        gradients = grad_objective(params)

        # Store history
        optimization_history.append((float(obj_value), params.copy()))

        # Gradient descent step
        params = params - learning_rate * gradients

        if iteration % 3 == 0:
            weights = jnp.exp(params) / jnp.sum(jnp.exp(params))
            print(f"{iteration:2d}     {obj_value:.6f}    {weights}")

    # Create optimized airfoil
    final_upper, final_lower = create_morphed_airfoil(params)
    optimized_airfoil = Airfoil(final_upper, final_lower, name="optimized_shape")

    print("\nOptimization completed!")
    print(f"Final objective value: {optimization_history[-1][0]:.6f}")
    print(f"Optimized airfoil max thickness: {optimized_airfoil.max_thickness:.4f}")

    return optimized_airfoil, optimization_history


def plot_morphing_results():
    """Create comprehensive visualization of morphing operations."""
    print("\n=== Creating Morphing Visualizations ===")

    # Run demonstrations to get results
    source, target = basic_morphing_with_gradients()
    _, optimized_multi, weights = parametric_morphing_optimization()
    morphed_sequence, time_points = dynamic_morphing_sequence()
    optimized_shape, opt_history = gradient_based_shape_optimization()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Basic morphing sequence
    ax1 = axes[0, 0]
    eta_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    cmap = plt.get_cmap("viridis", len(eta_values))

    for i, eta in enumerate(eta_values):
        morphed = Airfoil.morph_new_from_two_foils(source, target, eta=eta, n_points=50)
        upper = morphed.upper_surface
        ax1.plot(
            upper[0],
            upper[1],
            color=cmap(i / len(eta_values)),
            linewidth=2,
            label=f"η = {eta:.2f}",
        )

    ax1.set_xlabel("x/c")
    ax1.set_ylabel("y/c")
    ax1.set_title("Basic Morphing Sequence")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis("equal")

    # Plot 2: Multi-airfoil morphing result
    ax2 = axes[0, 1]
    ax2.plot(
        *optimized_multi.upper_surface,
        "r-",
        linewidth=3,
        label="Optimized Multi-Morph",
    )
    ax2.plot(*optimized_multi.lower_surface, "r--", linewidth=3)

    # Show contributing airfoils with transparency
    airfoils = [
        NACA4(M=0.0, P=0.0, XX=0.09, n_points=100),
        NACA4(M=0.02, P=0.4, XX=0.12, n_points=100),
        NACA4(M=0.04, P=0.3, XX=0.15, n_points=100),
        NACA4(M=0.01, P=0.5, XX=0.18, n_points=100),
    ]

    for i, airfoil in enumerate(airfoils):
        alpha = float(weights[i])
        ax2.plot(
            *airfoil.upper_surface,
            alpha=alpha,
            linewidth=1,
            label=f"{airfoil.name} (w={alpha:.2f})",
        )

    ax2.set_xlabel("x/c")
    ax2.set_ylabel("y/c")
    ax2.set_title("Multi-Airfoil Morphing")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axis("equal")

    # Plot 3: Dynamic morphing sequence
    ax3 = axes[0, 2]
    cmap = plt.get_cmap("plasma", len(morphed_sequence))

    for i, (morphed, t) in enumerate(zip(morphed_sequence[::3], time_points[::3])):
        upper = morphed.upper_surface
        ax3.plot(
            upper[0],
            upper[1],
            color=cmap(i * 3 / len(morphed_sequence)),
            linewidth=2,
            label=f"t = {t:.1f}",
        )

    ax3.set_xlabel("x/c")
    ax3.set_ylabel("y/c")
    ax3.set_title("Dynamic Morphing Sequence")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.axis("equal")

    # Plot 4: Optimization convergence
    ax4 = axes[1, 0]
    objectives = [hist[0] for hist in opt_history]
    ax4.plot(objectives, "b-", linewidth=2, marker="o")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Objective Value")
    ax4.set_title("Shape Optimization Convergence")
    ax4.grid(True, alpha=0.3)

    # Plot 5: Optimized shape comparison
    ax5 = axes[1, 1]
    ax5.plot(*optimized_shape.upper_surface, "g-", linewidth=3, label="Optimized Shape")
    ax5.plot(*optimized_shape.lower_surface, "g--", linewidth=3)

    # Compare with NACA 0012
    naca0012 = NACA4(M=0.0, P=0.0, XX=0.12, n_points=100)
    ax5.plot(*naca0012.upper_surface, "k-", linewidth=1, alpha=0.7, label="NACA 0012")
    ax5.plot(*naca0012.lower_surface, "k--", linewidth=1, alpha=0.7)

    ax5.set_xlabel("x/c")
    ax5.set_ylabel("y/c")
    ax5.set_title("Gradient-Optimized Shape")
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.axis("equal")

    # Plot 6: Thickness evolution in dynamic morphing
    ax6 = axes[1, 2]
    max_thicknesses = [morphed.max_thickness for morphed in morphed_sequence]
    ax6.plot(time_points, max_thicknesses, "r-", linewidth=2, marker="s")
    ax6.set_xlabel("Time")
    ax6.set_ylabel("Maximum Thickness")
    ax6.set_title("Thickness Evolution")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """Main function demonstrating advanced morphing operations."""
    print("Advanced JAX Airfoil Morphing Operations")
    print("=" * 60)

    # Demonstrate various advanced morphing techniques
    basic_morphing_with_gradients()
    parametric_morphing_optimization()
    dynamic_morphing_sequence()
    gradient_based_shape_optimization()

    # Create comprehensive visualization
    plot_morphing_results()

    print("\n" + "=" * 60)
    print("Key Advanced Morphing Capabilities:")
    print("1. Gradient computation through morphing operations")
    print("2. Multi-airfoil parametric morphing with optimization")
    print("3. Dynamic time-dependent morphing sequences")
    print("4. Gradient-based shape optimization using morphing spaces")
    print("5. JIT compilation for efficient morphing computations")
    print("6. Automatic differentiation enables advanced optimization workflows")


if __name__ == "__main__":
    main()
