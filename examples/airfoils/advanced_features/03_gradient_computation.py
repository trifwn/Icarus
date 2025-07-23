#!/usr/bin/env python3
"""
Advanced Gradient Computation and Automatic Differentiation

This example demonstrates advanced gradient computation capabilities using JAX:
1. Forward and reverse mode automatic differentiation
2. Higher-order derivatives (Hessians)
3. Gradient computation through complex airfoil operations
4. Sensitivity analysis for design parameters
5. Gradient-based optimization workflows

The JAX implementation provides seamless automatic differentiation through
all airfoil operations, enabling sophisticated optimization and sensitivity analysis.
"""

from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad
from jax import hessian
from jax import jacfwd
from jax import jacrev

from ICARUS.airfoils.naca4 import NACA4


def basic_gradient_computation():
    """Demonstrate basic gradient computation for airfoil properties."""
    print("=== Basic Gradient Computation ===")

    def thickness_objective(params):
        """Objective function based on airfoil thickness."""
        m, p, xx = params
        naca = NACA4(M=m, P=p, XX=xx, n_points=100)

        # Return maximum thickness
        return naca.max_thickness

    def thickness_location_objective(params):
        """Objective function based on thickness location."""
        m, p, xx = params
        naca = NACA4(M=m, P=p, XX=xx, n_points=100)

        # Return location of maximum thickness
        return naca.max_thickness_location

    # Test parameters: NACA 2412
    test_params = jnp.array([0.02, 0.4, 0.12])

    # Compute gradients
    grad_thickness = grad(thickness_objective)
    grad_location = grad(thickness_location_objective)

    thickness_value = thickness_objective(test_params)
    location_value = thickness_location_objective(test_params)

    thickness_grads = grad_thickness(test_params)
    location_grads = grad_location(test_params)

    print(
        f"Test airfoil: NACA {int(test_params[0]*100):01d}{int(test_params[1]*10):01d}{int(test_params[2]*100):02d}",
    )
    print(f"Maximum thickness: {thickness_value:.5f}")
    print(f"Thickness gradients [∂t/∂M, ∂t/∂P, ∂t/∂XX]: {thickness_grads}")
    print(f"Thickness location: {location_value:.5f}")
    print(f"Location gradients [∂x/∂M, ∂x/∂P, ∂x/∂XX]: {location_grads}")

    return test_params, thickness_grads, location_grads


def surface_gradient_computation():
    """Demonstrate gradient computation for surface coordinates."""
    print("\n=== Surface Gradient Computation ===")

    def surface_point_objective(params, x_target):
        """Objective function for surface point at specific x-coordinate."""
        m, p, xx = params
        naca = NACA4(M=m, P=p, XX=xx, n_points=100)

        # Return upper surface y-coordinate at target x
        y_upper = naca.y_upper(jnp.array([x_target]))[0]
        return y_upper

    # Test parameters and target x-coordinate
    test_params = jnp.array([0.03, 0.4, 0.15])
    x_targets = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])

    print(f"Computing surface gradients at x-coordinates: {x_targets}")

    # Compute gradients for each x-coordinate
    surface_gradients = []
    surface_values = []

    for x_target in x_targets:
        objective_func = partial(surface_point_objective, x_target=x_target)
        grad_func = grad(objective_func)

        value = objective_func(test_params)
        gradients = grad_func(test_params)

        surface_values.append(float(value))
        surface_gradients.append(gradients)

    surface_gradients = jnp.array(surface_gradients)
    surface_values = jnp.array(surface_values)

    print("\nSurface gradient analysis:")
    print("x/c     y_upper    ∂y/∂M      ∂y/∂P      ∂y/∂XX")
    print("-" * 50)

    for i, x in enumerate(x_targets):
        y = surface_values[i]
        grads = surface_gradients[i]
        print(
            f"{x:.1f}     {y:7.5f}    {grads[0]:8.5f}   {grads[1]:8.5f}   {grads[2]:8.5f}",
        )

    return x_targets, surface_values, surface_gradients


def hessian_computation():
    """Demonstrate second-order derivative (Hessian) computation."""
    print("\n=== Hessian Computation ===")

    def complex_objective(params):
        """Complex objective function for Hessian demonstration."""
        m, p, xx = params
        naca = NACA4(M=m, P=p, XX=xx, n_points=100)

        # Complex objective combining multiple properties
        max_thickness = naca.max_thickness
        thickness_location = naca.max_thickness_location

        # Evaluate thickness at multiple points
        x_eval = jnp.array([0.2, 0.5, 0.8])
        thickness_values = naca.thickness(x_eval)
        mean_thickness = jnp.mean(thickness_values)

        # Combined objective
        objective = (
            (max_thickness - 0.15) ** 2
            + (thickness_location - 0.3) ** 2
            + mean_thickness**2
        )
        return objective

    # Test parameters
    test_params = jnp.array([0.04, 0.3, 0.16])

    # Compute gradient and Hessian
    grad_func = grad(complex_objective)
    hess_func = hessian(complex_objective)

    objective_value = complex_objective(test_params)
    gradient_value = grad_func(test_params)
    hessian_value = hess_func(test_params)

    print(f"Objective value: {objective_value:.6f}")
    print(f"Gradient: {gradient_value}")
    print("Hessian matrix:")
    print(hessian_value)

    # Analyze Hessian properties
    eigenvalues = jnp.linalg.eigvals(hessian_value)
    condition_number = jnp.max(eigenvalues) / jnp.min(eigenvalues)

    print(f"Hessian eigenvalues: {eigenvalues}")
    print(f"Condition number: {condition_number:.2f}")

    return hessian_value, eigenvalues


def jacobian_computation():
    """Demonstrate Jacobian computation for vector-valued functions."""
    print("\n=== Jacobian Computation ===")

    def vector_objective(params):
        """Vector-valued objective function."""
        m, p, xx = params
        naca = NACA4(M=m, P=p, XX=xx, n_points=100)

        # Multiple outputs
        max_thickness = naca.max_thickness
        thickness_location = naca.max_thickness_location

        # Thickness at specific locations
        x_eval = jnp.array([0.25, 0.5, 0.75])
        thickness_values = naca.thickness(x_eval)

        # Return vector of outputs
        return jnp.concatenate(
            [jnp.array([max_thickness, thickness_location]), thickness_values],
        )

    # Test parameters
    test_params = jnp.array([0.025, 0.35, 0.14])

    # Compute Jacobian using forward and reverse mode
    jac_forward = jacfwd(vector_objective)
    jac_reverse = jacrev(vector_objective)

    jacobian_fwd = jac_forward(test_params)
    jacobian_rev = jac_reverse(test_params)

    # Verify they're the same
    jacobian_diff = jnp.max(jnp.abs(jacobian_fwd - jacobian_rev))

    print(f"Jacobian shape: {jacobian_fwd.shape}")
    print(f"Forward vs reverse mode difference: {jacobian_diff:.2e}")

    print("\nJacobian matrix (forward mode):")
    print("Output         ∂/∂M       ∂/∂P       ∂/∂XX")
    print("-" * 45)

    output_names = [
        "Max thickness",
        "Thickness loc",
        "Thickness@0.25",
        "Thickness@0.50",
        "Thickness@0.75",
    ]

    for i, name in enumerate(output_names):
        row = jacobian_fwd[i]
        print(f"{name:<14} {row[0]:9.6f}  {row[1]:9.6f}  {row[2]:9.6f}")

    return jacobian_fwd, jacobian_rev


def sensitivity_analysis():
    """Demonstrate sensitivity analysis for design parameters."""
    print("\n=== Sensitivity Analysis ===")

    def aerodynamic_performance(params):
        """Simplified aerodynamic performance function."""
        m, p, xx = params
        naca = NACA4(M=m, P=p, XX=xx, n_points=100)

        # Simplified performance metrics
        # In practice, these would come from CFD or panel methods

        # Drag coefficient (simplified model)
        drag_coeff = 0.008 + 0.15 * xx**2 + 0.8 * m**2

        # Lift coefficient (simplified model)
        lift_coeff = 1.1 + 12.0 * m - 8.0 * m**2

        # Moment coefficient (simplified model)
        moment_coeff = -0.05 - 2.0 * m + 0.5 * (p - 0.25)

        return jnp.array([drag_coeff, lift_coeff, moment_coeff])

    # Baseline parameters
    baseline_params = jnp.array([0.02, 0.4, 0.12])  # NACA 2412

    # Compute sensitivity matrix (Jacobian)
    sensitivity_matrix = jacfwd(aerodynamic_performance)(baseline_params)

    print("Sensitivity Analysis for NACA 2412:")
    print("Coefficient    ∂/∂M       ∂/∂P       ∂/∂XX")
    print("-" * 45)

    coeff_names = ["Drag (CD)", "Lift (CL)", "Moment (CM)"]
    for i, name in enumerate(coeff_names):
        row = sensitivity_matrix[i]
        print(f"{name:<12}   {row[0]:8.5f}   {row[1]:8.5f}   {row[2]:8.5f}")

    # Compute relative sensitivities
    baseline_values = aerodynamic_performance(baseline_params)
    relative_sensitivity = (
        sensitivity_matrix * baseline_params / baseline_values[:, None]
    )

    print("\nRelative Sensitivities (elasticities):")
    print("Coefficient    ∂/∂M       ∂/∂P       ∂/∂XX")
    print("-" * 45)

    for i, name in enumerate(coeff_names):
        row = relative_sensitivity[i]
        print(f"{name:<12}   {row[0]:8.5f}   {row[1]:8.5f}   {row[2]:8.5f}")

    # Parameter uncertainty analysis
    param_uncertainties = jnp.array(
        [0.002, 0.05, 0.01],
    )  # Typical manufacturing tolerances

    # Propagate uncertainties using linear approximation
    coeff_uncertainties = jnp.sqrt(
        jnp.sum((sensitivity_matrix * param_uncertainties) ** 2, axis=1),
    )

    print("\nUncertainty Propagation:")
    print(
        f"Parameter uncertainties: ΔM={param_uncertainties[0]:.3f}, ΔP={param_uncertainties[1]:.2f}, ΔXX={param_uncertainties[2]:.3f}",
    )
    print("Coefficient uncertainties:")
    for i, name in enumerate(coeff_names):
        print(f"  {name}: ±{coeff_uncertainties[i]:.5f}")

    return sensitivity_matrix, relative_sensitivity, coeff_uncertainties


def optimization_with_gradients():
    """Demonstrate gradient-based optimization workflow."""
    print("\n=== Gradient-Based Optimization ===")

    def optimization_objective(params):
        """Multi-objective optimization function."""
        m, p, xx = params
        naca = NACA4(M=m, P=p, XX=xx, n_points=100)

        # Objectives
        max_thickness = naca.max_thickness
        thickness_location = naca.max_thickness_location

        # Target values
        target_thickness = 0.15
        target_location = 0.35

        # Weighted objective
        thickness_error = (max_thickness - target_thickness) ** 2
        location_error = (thickness_location - target_location) ** 2

        return thickness_error + 2.0 * location_error

    # Optimization using gradient descent
    grad_objective = grad(optimization_objective)

    # Initial guess
    params = jnp.array([0.04, 0.5, 0.18])
    learning_rate = 0.1

    print("Gradient-based optimization:")
    print("Iter   Objective    M        P        XX       Grad Norm")
    print("-" * 60)

    optimization_history = []

    for iteration in range(20):
        obj_value = optimization_objective(params)
        gradients = grad_objective(params)
        grad_norm = jnp.linalg.norm(gradients)

        # Store history
        optimization_history.append(
            {
                "iteration": iteration,
                "objective": float(obj_value),
                "params": params.copy(),
                "gradients": gradients.copy(),
                "grad_norm": float(grad_norm),
            },
        )

        # Print progress
        if iteration % 4 == 0:
            print(
                f"{iteration:2d}     {obj_value:.6f}   {params[0]:.4f}   {params[1]:.4f}   {params[2]:.4f}   {grad_norm:.6f}",
            )

        # Gradient descent step
        params = params - learning_rate * gradients

        # Simple bounds enforcement
        params = jnp.clip(
            params,
            jnp.array([0.0, 0.1, 0.08]),
            jnp.array([0.08, 0.8, 0.25]),
        )

        # Check convergence
        if grad_norm < 1e-6:
            print(f"Converged at iteration {iteration}")
            break

    # Final results
    final_naca = NACA4(M=params[0], P=params[1], XX=params[2], n_points=100)
    final_name = (
        f"NACA {int(params[0]*100):01d}{int(params[1]*10):01d}{int(params[2]*100):02d}"
    )

    print("\nOptimization Results:")
    print(f"Final airfoil: {final_name}")
    print(f"Final parameters: M={params[0]:.4f}, P={params[1]:.4f}, XX={params[2]:.4f}")
    print(f"Max thickness: {final_naca.max_thickness:.4f} (target: 0.15)")
    print(f"Thickness location: {final_naca.max_thickness_location:.4f} (target: 0.35)")

    return optimization_history, final_naca


def plot_gradient_results():
    """Create comprehensive visualization of gradient computation results."""
    print("\n=== Creating Gradient Computation Visualizations ===")

    # Run demonstrations to get results
    test_params, thickness_grads, location_grads = basic_gradient_computation()
    x_targets, surface_values, surface_gradients = surface_gradient_computation()
    hessian_matrix, eigenvalues = hessian_computation()
    jacobian_fwd, jacobian_rev = jacobian_computation()
    sensitivity_matrix, relative_sensitivity, uncertainties = sensitivity_analysis()
    opt_history, final_airfoil = optimization_with_gradients()

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # Plot 1: Basic gradients comparison
    ax1 = axes[0, 0]
    param_names = ["M (Camber)", "P (Position)", "XX (Thickness)"]
    x_pos = range(len(param_names))

    width = 0.35
    ax1.bar(
        [x - width / 2 for x in x_pos],
        thickness_grads,
        width,
        label="Thickness Gradients",
        alpha=0.7,
    )
    ax1.bar(
        [x + width / 2 for x in x_pos],
        location_grads,
        width,
        label="Location Gradients",
        alpha=0.7,
    )

    ax1.set_xlabel("Parameters")
    ax1.set_ylabel("Gradient Value")
    ax1.set_title("Basic Gradient Comparison")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(param_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Surface gradients
    ax2 = axes[0, 1]
    ax2.plot(x_targets, surface_gradients[:, 0], "o-", label="∂y/∂M", linewidth=2)
    ax2.plot(x_targets, surface_gradients[:, 1], "s-", label="∂y/∂P", linewidth=2)
    ax2.plot(x_targets, surface_gradients[:, 2], "^-", label="∂y/∂XX", linewidth=2)

    ax2.set_xlabel("x/c")
    ax2.set_ylabel("Surface Gradient")
    ax2.set_title("Surface Point Gradients")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Hessian eigenvalues
    ax3 = axes[0, 2]
    ax3.bar(range(len(eigenvalues)), eigenvalues, alpha=0.7)
    ax3.set_xlabel("Eigenvalue Index")
    ax3.set_ylabel("Eigenvalue")
    ax3.set_title("Hessian Eigenvalues")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Jacobian heatmap
    ax4 = axes[1, 0]
    im = ax4.imshow(jacobian_fwd, cmap="RdBu_r", aspect="auto")
    ax4.set_xlabel("Parameters (M, P, XX)")
    ax4.set_ylabel("Outputs")
    ax4.set_title("Jacobian Matrix")
    ax4.set_xticks(range(3))
    ax4.set_xticklabels(["M", "P", "XX"])
    ax4.set_yticks(range(5))
    ax4.set_yticklabels(["Max t", "t loc", "t@0.25", "t@0.5", "t@0.75"])
    plt.colorbar(im, ax=ax4)

    # Plot 5: Sensitivity analysis
    ax5 = axes[1, 1]
    coeff_names = ["CD", "CL", "CM"]
    param_names_short = ["M", "P", "XX"]

    x = np.arange(len(coeff_names))
    width = 0.25

    for i, param in enumerate(param_names_short):
        ax5.bar(
            x + i * width,
            sensitivity_matrix[:, i],
            width,
            label=f"∂/∂{param}",
            alpha=0.7,
        )

    ax5.set_xlabel("Aerodynamic Coefficients")
    ax5.set_ylabel("Sensitivity")
    ax5.set_title("Aerodynamic Sensitivity")
    ax5.set_xticks(x + width)
    ax5.set_xticklabels(coeff_names)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Uncertainty propagation
    ax6 = axes[1, 2]
    ax6.bar(coeff_names, uncertainties, alpha=0.7, color="orange")
    ax6.set_xlabel("Aerodynamic Coefficients")
    ax6.set_ylabel("Uncertainty")
    ax6.set_title("Uncertainty Propagation")
    ax6.grid(True, alpha=0.3)

    # Plot 7: Optimization convergence
    ax7 = axes[2, 0]
    objectives = [h["objective"] for h in opt_history]
    ax7.semilogy(objectives, "b-", linewidth=2, marker="o")
    ax7.set_xlabel("Iteration")
    ax7.set_ylabel("Objective Value")
    ax7.set_title("Optimization Convergence")
    ax7.grid(True, alpha=0.3)

    # Plot 8: Parameter evolution
    ax8 = axes[2, 1]
    params_history = jnp.array([h["params"] for h in opt_history])

    ax8.plot(params_history[:, 0], label="M (Camber)", linewidth=2)
    ax8.plot(params_history[:, 1], label="P (Position)", linewidth=2)
    ax8.plot(params_history[:, 2], label="XX (Thickness)", linewidth=2)

    ax8.set_xlabel("Iteration")
    ax8.set_ylabel("Parameter Value")
    ax8.set_title("Parameter Evolution")
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Plot 9: Final optimized airfoil
    ax9 = axes[2, 2]
    upper = final_airfoil.upper_surface
    lower = final_airfoil.lower_surface

    ax9.plot(upper[0], upper[1], "b-", linewidth=3, label="Optimized Upper")
    ax9.plot(lower[0], lower[1], "b--", linewidth=3, label="Optimized Lower")

    # Compare with target
    target_naca = NACA4(M=0.02, P=0.35, XX=0.15, n_points=100)
    ax9.plot(
        target_naca.upper_surface[0],
        target_naca.upper_surface[1],
        "r-",
        linewidth=1,
        alpha=0.7,
        label="Target Shape",
    )
    ax9.plot(
        target_naca.lower_surface[0],
        target_naca.lower_surface[1],
        "r--",
        linewidth=1,
        alpha=0.7,
    )

    ax9.set_xlabel("x/c")
    ax9.set_ylabel("y/c")
    ax9.set_title("Optimized Airfoil Shape")
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    ax9.axis("equal")

    plt.tight_layout()
    plt.show()


def main():
    """Main function demonstrating advanced gradient computation."""
    print("Advanced JAX Airfoil Gradient Computation")
    print("=" * 60)

    # Demonstrate various gradient computation techniques
    basic_gradient_computation()
    surface_gradient_computation()
    hessian_computation()
    jacobian_computation()
    sensitivity_analysis()
    optimization_with_gradients()

    # Create comprehensive visualization
    plot_gradient_results()

    print("\n" + "=" * 60)
    print("Key Gradient Computation Capabilities:")
    print("1. Forward and reverse mode automatic differentiation")
    print("2. Higher-order derivatives (Hessians) for optimization analysis")
    print("3. Jacobian computation for vector-valued functions")
    print("4. Comprehensive sensitivity analysis for design parameters")
    print("5. Uncertainty propagation using gradient information")
    print("6. Gradient-based optimization workflows")
    print("7. JIT compilation of gradient computations for efficiency")
    print("8. Seamless differentiation through complex airfoil operations")


if __name__ == "__main__":
    main()
