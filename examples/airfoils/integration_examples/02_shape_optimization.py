#!/usr/bin/env python3
"""
Shape Optimization Examples Using Gradient-Based Methods

This example demonstrates advanced shape optimization workflows using JAX airfoils:
1. Single-objective optimization (maximize L/D ratio)
2. Multi-objective optimization (Pareto frontier analysis)
3. Constrained optimization with geometric constraints
4. Robust optimization under uncertainty
5. Multi-point optimization across operating conditions
6. Integration with gradient-based optimizers

The JAX implementation enables efficient gradient computation through
complex optimization objectives, making advanced optimization workflows practical.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad
from jax import jit

from ICARUS.airfoils.naca4 import NACA4


def setup_optimization_problem():
    """Set up the optimization problem parameters and constraints."""
    print("=== Setting Up Optimization Problem ===")

    # Design space: NACA 4-digit parameters
    design_bounds = {
        "M": (0.0, 0.08),  # Maximum camber (0-8%)
        "P": (0.2, 0.8),  # Position of maximum camber (20-80% chord)
        "XX": (0.08, 0.20),  # Maximum thickness (8-20% chord)
    }

    # Operating conditions
    operating_conditions = {
        "reynolds": jnp.array([100000.0, 200000.0, 500000.0]),
        "angles": jnp.array([0.0, 2.0, 4.0, 6.0, 8.0]),  # degrees
        "mach": 0.1,
    }

    # Optimization constraints
    constraints = {
        "min_thickness": 0.08,  # Minimum thickness for structural requirements
        "max_thickness": 0.18,  # Maximum thickness for manufacturing
        "min_camber_pos": 0.25,  # Forward limit for camber position
        "max_camber_pos": 0.75,  # Aft limit for camber position
    }

    print(
        f"Design bounds: M∈{design_bounds['M']}, P∈{design_bounds['P']}, XX∈{design_bounds['XX']}",
    )
    print(
        f"Operating conditions: {len(operating_conditions['reynolds'])} Re × {len(operating_conditions['angles'])} α",
    )
    print(
        f"Constraints: thickness ∈ [{constraints['min_thickness']}, {constraints['max_thickness']}]",
    )

    return design_bounds, operating_conditions, constraints


@jit
def aerodynamic_analysis_function(upper_surface, lower_surface, reynolds, angle_rad):
    """
    Aerodynamic analysis function for optimization.

    This uses a simplified panel method approximation. In practice,
    you would interface with actual CFD solvers or higher-fidelity methods.
    """
    # Extract coordinates
    x_coords = upper_surface[0, :]
    y_upper = upper_surface[1, :]
    y_lower = lower_surface[1, :]

    # Calculate geometric properties
    thickness = y_upper - y_lower
    camber = 0.5 * (y_upper + y_lower)

    # Lift coefficient (thin airfoil theory + corrections)
    camber_slope = jnp.gradient(camber, x_coords)
    cl_alpha = 2 * jnp.pi  # 2D lift curve slope
    cl_camber = 2 * jnp.pi * jnp.trapz(camber_slope, x_coords)
    cl = cl_alpha * angle_rad + cl_camber

    # Drag coefficient (form + friction drag)
    max_thickness = jnp.max(thickness)
    reynolds_factor = jnp.log(reynolds) / 15.0
    cd_form = 0.015 * max_thickness**2
    cd_friction = 0.008 / jnp.sqrt(reynolds_factor)
    cd_induced = cl**2 / (jnp.pi * 8.0)  # Simplified induced drag
    cd = cd_form + cd_friction + cd_induced

    # Moment coefficient
    x_ac = 0.25  # Aerodynamic center
    cm = -cl * (x_ac - 0.25)

    return cl, cd, cm


def single_objective_optimization(design_bounds, operating_conditions, constraints):
    """Perform single-objective optimization to maximize L/D ratio."""
    print("\n=== Single-Objective Optimization (Maximize L/D) ===")

    def create_airfoil_from_params(params):
        """Create airfoil from optimization parameters."""
        M, P, XX = params
        return NACA4(M=M, P=P, XX=XX, n_points=100)

    @jit
    def objective_function(params):
        """Objective function: negative L/D ratio (for minimization)."""
        M, P, XX = params

        # Create airfoil
        airfoil = NACA4(M=M, P=P, XX=XX, n_points=100)

        # Analyze at multiple operating conditions
        total_performance = 0.0
        n_conditions = 0

        for reynolds in operating_conditions["reynolds"]:
            for angle_deg in operating_conditions["angles"]:
                angle_rad = jnp.deg2rad(angle_deg)

                cl, cd, cm = aerodynamic_analysis_function(
                    airfoil.upper_surface,
                    airfoil.lower_surface,
                    reynolds,
                    angle_rad,
                )

                # L/D ratio with small epsilon to avoid division by zero
                ld_ratio = cl / (cd + 1e-6)

                # Weight by angle of attack (emphasize cruise conditions)
                weight = jnp.exp(-0.1 * angle_deg**2)  # Gaussian weighting
                total_performance += weight * ld_ratio
                n_conditions += weight

        # Average weighted performance
        avg_performance = total_performance / n_conditions

        # Return negative for minimization
        return -avg_performance

    @jit
    def constraint_function(params):
        """Constraint function: geometric constraints."""
        M, P, XX = params

        # Thickness constraints
        thickness_violation = jnp.maximum(
            0,
            constraints["min_thickness"] - XX,
        ) + jnp.maximum(0, XX - constraints["max_thickness"])

        # Camber position constraints
        camber_pos_violation = jnp.maximum(
            0,
            constraints["min_camber_pos"] - P,
        ) + jnp.maximum(0, P - constraints["max_camber_pos"])

        return thickness_violation + camber_pos_violation

    # Initial guess (NACA 2412)
    x0 = jnp.array([0.02, 0.4, 0.12])

    print(f"Initial guess: M={x0[0]:.3f}, P={x0[1]:.1f}, XX={x0[2]:.3f}")

    # Compute initial performance
    initial_performance = -objective_function(x0)
    print(f"Initial L/D ratio: {initial_performance:.2f}")

    # Gradient-based optimization using JAX
    grad_objective = grad(objective_function)

    # Simple gradient descent optimization
    learning_rate = 0.01
    max_iterations = 100
    tolerance = 1e-6

    x = x0.copy()
    optimization_history = []

    print("\nOptimization progress:")
    print("Iter   L/D Ratio   M      P      XX     Gradient Norm")
    print("-" * 55)

    for iteration in range(max_iterations):
        # Compute objective and gradient
        obj_value = objective_function(x)
        gradients = grad_objective(x)
        grad_norm = jnp.linalg.norm(gradients)

        # Store history
        optimization_history.append(
            {
                "iteration": iteration,
                "params": x.copy(),
                "objective": -obj_value,
                "gradients": gradients,
                "grad_norm": grad_norm,
            },
        )

        # Print progress
        if iteration % 10 == 0:
            print(
                f"{iteration:3d}    {-obj_value:7.3f}   {x[0]:.3f}  {x[1]:.3f}  {x[2]:.3f}  {grad_norm:.2e}",
            )

        # Check convergence
        if grad_norm < tolerance:
            print(f"Converged at iteration {iteration}")
            break

        # Gradient descent step
        x_new = x - learning_rate * gradients

        # Apply bounds constraints
        x_new = jnp.clip(
            x_new,
            jnp.array(
                [design_bounds["M"][0], design_bounds["P"][0], design_bounds["XX"][0]],
            ),
            jnp.array(
                [design_bounds["M"][1], design_bounds["P"][1], design_bounds["XX"][1]],
            ),
        )

        x = x_new

    # Final results
    final_performance = -objective_function(x)
    optimized_airfoil = create_airfoil_from_params(x)

    print("\nOptimization Results:")
    print(f"Final L/D ratio: {final_performance:.3f}")
    print(f"Improvement: {final_performance - initial_performance:.3f}")
    print(f"Optimized parameters: M={x[0]:.4f}, P={x[1]:.3f}, XX={x[2]:.4f}")
    print(
        f"Optimized airfoil name: NACA{int(x[0]*100):01d}{int(x[1]*10):01d}{int(x[2]*100):02d}",
    )

    return {
        "optimized_params": x,
        "optimized_airfoil": optimized_airfoil,
        "final_performance": final_performance,
        "optimization_history": optimization_history,
        "improvement": final_performance - initial_performance,
    }


def multi_objective_optimization(design_bounds, operating_conditions, constraints):
    """Perform multi-objective optimization (L/D vs robustness)."""
    print("\n=== Multi-Objective Optimization (L/D vs Robustness) ===")

    @jit
    def multi_objective_function(params, weight_performance=0.7):
        """Multi-objective function combining performance and robustness."""
        M, P, XX = params

        # Create airfoil
        airfoil = NACA4(M=M, P=P, XX=XX, n_points=100)

        # Objective 1: Average performance (L/D ratio)
        total_ld = 0.0
        ld_values = []
        n_conditions = 0

        for reynolds in operating_conditions["reynolds"]:
            for angle_deg in operating_conditions["angles"]:
                angle_rad = jnp.deg2rad(angle_deg)

                cl, cd, cm = aerodynamic_analysis_function(
                    airfoil.upper_surface,
                    airfoil.lower_surface,
                    reynolds,
                    angle_rad,
                )

                ld_ratio = cl / (cd + 1e-6)
                ld_values.append(ld_ratio)
                total_ld += ld_ratio
                n_conditions += 1

        avg_ld = total_ld / n_conditions

        # Objective 2: Robustness (negative standard deviation of L/D)
        ld_array = jnp.array(ld_values)
        ld_std = jnp.std(ld_array)
        robustness = -ld_std  # Negative because we want low std deviation

        # Combined objective
        combined_objective = (
            weight_performance * avg_ld + (1 - weight_performance) * robustness
        )

        return -combined_objective, avg_ld, ld_std  # Return negative for minimization

    # Pareto frontier analysis
    weight_values = jnp.linspace(
        0.1,
        0.9,
        9,
    )  # Different weights for performance vs robustness
    pareto_results = []

    print("Computing Pareto frontier...")
    print("Weight  L/D Avg  L/D Std   M      P      XX")
    print("-" * 45)

    for weight in weight_values:
        # Optimize for this weight combination
        grad_multi_obj = grad(
            lambda params: multi_objective_function(params, weight)[0],
        )

        # Initial guess
        x = jnp.array([0.02, 0.4, 0.12])
        learning_rate = 0.01

        # Simple optimization loop
        for _ in range(50):
            gradients = grad_multi_obj(x)
            x = x - learning_rate * gradients

            # Apply bounds
            x = jnp.clip(
                x,
                jnp.array(
                    [
                        design_bounds["M"][0],
                        design_bounds["P"][0],
                        design_bounds["XX"][0],
                    ],
                ),
                jnp.array(
                    [
                        design_bounds["M"][1],
                        design_bounds["P"][1],
                        design_bounds["XX"][1],
                    ],
                ),
            )

        # Evaluate final result
        _, avg_ld, ld_std = multi_objective_function(x, weight)

        pareto_results.append(
            {
                "weight": weight,
                "params": x,
                "avg_ld": avg_ld,
                "ld_std": ld_std,
                "airfoil": NACA4(M=x[0], P=x[1], XX=x[2], n_points=100),
            },
        )

        print(
            f"{weight:.1f}     {avg_ld:.3f}   {ld_std:.3f}   {x[0]:.3f}  {x[1]:.3f}  {x[2]:.3f}",
        )

    return pareto_results


def constrained_optimization(design_bounds, operating_conditions, constraints):
    """Perform constrained optimization with geometric and performance constraints."""
    print("\n=== Constrained Optimization ===")

    @jit
    def objective_with_penalties(params, penalty_weight=100.0):
        """Objective function with penalty method for constraints."""
        M, P, XX = params

        # Create airfoil
        airfoil = NACA4(M=M, P=P, XX=XX, n_points=100)

        # Primary objective: L/D at cruise condition
        cruise_reynolds = operating_conditions["reynolds"][1]  # Middle Reynolds number
        cruise_angle = jnp.deg2rad(4.0)  # 4 degrees

        cl, cd, cm = aerodynamic_analysis_function(
            airfoil.upper_surface,
            airfoil.lower_surface,
            cruise_reynolds,
            cruise_angle,
        )

        ld_ratio = cl / (cd + 1e-6)

        # Geometric constraints
        thickness_penalty = (
            jnp.maximum(0, constraints["min_thickness"] - XX) ** 2
            + jnp.maximum(0, XX - constraints["max_thickness"]) ** 2
        )

        camber_pos_penalty = (
            jnp.maximum(0, constraints["min_camber_pos"] - P) ** 2
            + jnp.maximum(0, P - constraints["max_camber_pos"]) ** 2
        )

        # Performance constraints (example: minimum lift coefficient)
        min_cl_constraint = jnp.maximum(0, 0.2 - cl) ** 2  # Minimum CL = 0.2

        # Stability constraint (example: moment coefficient)
        stability_penalty = jnp.maximum(0, jnp.abs(cm) - 0.1) ** 2  # |CM| < 0.1

        # Total penalty
        total_penalty = penalty_weight * (
            thickness_penalty
            + camber_pos_penalty
            + min_cl_constraint
            + stability_penalty
        )

        # Combined objective (negative L/D + penalties)
        return -ld_ratio + total_penalty

    # Optimization with constraints
    grad_constrained = grad(objective_with_penalties)

    # Initial guess
    x = jnp.array([0.03, 0.4, 0.12])
    learning_rate = 0.005
    max_iterations = 200

    print("Constrained optimization progress:")
    print("Iter   Objective   L/D     M      P      XX     Penalties")
    print("-" * 60)

    optimization_history = []

    for iteration in range(max_iterations):
        # Compute objective and components
        total_obj = objective_with_penalties(x)

        # Compute individual components for reporting
        airfoil = NACA4(M=x[0], P=x[1], XX=x[2], n_points=100)
        cruise_reynolds = operating_conditions["reynolds"][1]
        cruise_angle = jnp.deg2rad(4.0)

        cl, cd, cm = aerodynamic_analysis_function(
            airfoil.upper_surface,
            airfoil.lower_surface,
            cruise_reynolds,
            cruise_angle,
        )
        ld_ratio = cl / (cd + 1e-6)

        # Compute penalties
        thickness_penalty = (
            jnp.maximum(0, constraints["min_thickness"] - x[2]) ** 2
            + jnp.maximum(0, x[2] - constraints["max_thickness"]) ** 2
        )
        penalties = thickness_penalty  # Simplified for display

        optimization_history.append(
            {
                "iteration": iteration,
                "params": x.copy(),
                "objective": total_obj,
                "ld_ratio": ld_ratio,
                "penalties": penalties,
            },
        )

        # Print progress
        if iteration % 25 == 0:
            print(
                f"{iteration:3d}    {total_obj:8.3f}   {ld_ratio:.3f}   {x[0]:.3f}  {x[1]:.3f}  {x[2]:.3f}   {penalties:.3f}",
            )

        # Gradient step
        gradients = grad_constrained(x)
        x = x - learning_rate * gradients

        # Apply bounds
        x = jnp.clip(
            x,
            jnp.array(
                [design_bounds["M"][0], design_bounds["P"][0], design_bounds["XX"][0]],
            ),
            jnp.array(
                [design_bounds["M"][1], design_bounds["P"][1], design_bounds["XX"][1]],
            ),
        )

    # Final results
    final_airfoil = NACA4(M=x[0], P=x[1], XX=x[2], n_points=100)
    final_cl, final_cd, final_cm = aerodynamic_analysis_function(
        final_airfoil.upper_surface,
        final_airfoil.lower_surface,
        operating_conditions["reynolds"][1],
        jnp.deg2rad(4.0),
    )
    final_ld = final_cl / (final_cd + 1e-6)

    print("\nConstrained Optimization Results:")
    print(f"Final L/D ratio: {final_ld:.3f}")
    print(f"Final parameters: M={x[0]:.4f}, P={x[1]:.3f}, XX={x[2]:.4f}")
    print("Constraint satisfaction:")
    print(
        f"  Thickness: {x[2]:.3f} ∈ [{constraints['min_thickness']}, {constraints['max_thickness']}]",
    )
    print(
        f"  Camber pos: {x[1]:.3f} ∈ [{constraints['min_camber_pos']}, {constraints['max_camber_pos']}]",
    )
    print(f"  Lift coeff: {final_cl:.3f} (min: 0.2)")
    print(f"  Moment coeff: {final_cm:.3f} (|CM| < 0.1)")

    return {
        "optimized_params": x,
        "optimized_airfoil": final_airfoil,
        "final_performance": final_ld,
        "optimization_history": optimization_history,
    }


def robust_optimization_under_uncertainty(
    design_bounds,
    operating_conditions,
    constraints,
):
    """Perform robust optimization considering uncertainty in operating conditions."""
    print("\n=== Robust Optimization Under Uncertainty ===")

    @jit
    def robust_objective(params, uncertainty_level=0.1):
        """Robust objective considering uncertainty in Reynolds number and angle."""
        M, P, XX = params

        # Create airfoil
        airfoil = NACA4(M=M, P=P, XX=XX, n_points=100)

        # Monte Carlo sampling for uncertainty
        n_samples = 20
        total_performance = 0.0
        performance_values = []

        # Base operating condition
        base_reynolds = operating_conditions["reynolds"][1]
        base_angle = 4.0  # degrees

        for i in range(n_samples):
            # Add uncertainty to operating conditions
            # Using deterministic sampling for reproducibility
            reynolds_factor = 1.0 + uncertainty_level * jnp.sin(i * 0.5)
            angle_factor = 1.0 + uncertainty_level * jnp.cos(i * 0.3)

            perturbed_reynolds = base_reynolds * reynolds_factor
            perturbed_angle = base_angle * angle_factor

            # Analyze performance
            cl, cd, cm = aerodynamic_analysis_function(
                airfoil.upper_surface,
                airfoil.lower_surface,
                perturbed_reynolds,
                jnp.deg2rad(perturbed_angle),
            )

            ld_ratio = cl / (cd + 1e-6)
            performance_values.append(ld_ratio)
            total_performance += ld_ratio

        # Robust metrics
        mean_performance = total_performance / n_samples
        performance_array = jnp.array(performance_values)
        std_performance = jnp.std(performance_array)
        min_performance = jnp.min(performance_array)

        # Robust objective: maximize mean performance while minimizing variability
        robust_metric = (
            mean_performance
            - 0.5 * std_performance
            - 0.2 * (mean_performance - min_performance)
        )

        return -robust_metric, mean_performance, std_performance, min_performance

    # Robust optimization
    grad_robust = grad(lambda params: robust_objective(params)[0])

    # Initial guess
    x = jnp.array([0.025, 0.35, 0.13])
    learning_rate = 0.01
    max_iterations = 100

    print("Robust optimization progress:")
    print("Iter   Mean L/D   Std L/D   Min L/D   M      P      XX")
    print("-" * 55)

    robust_history = []

    for iteration in range(max_iterations):
        _, mean_ld, std_ld, min_ld = robust_objective(x)

        robust_history.append(
            {
                "iteration": iteration,
                "params": x.copy(),
                "mean_ld": mean_ld,
                "std_ld": std_ld,
                "min_ld": min_ld,
            },
        )

        if iteration % 15 == 0:
            print(
                f"{iteration:3d}    {mean_ld:.3f}    {std_ld:.3f}    {min_ld:.3f}   {x[0]:.3f}  {x[1]:.3f}  {x[2]:.3f}",
            )

        # Gradient step
        gradients = grad_robust(x)
        x = x - learning_rate * gradients

        # Apply bounds
        x = jnp.clip(
            x,
            jnp.array(
                [design_bounds["M"][0], design_bounds["P"][0], design_bounds["XX"][0]],
            ),
            jnp.array(
                [design_bounds["M"][1], design_bounds["P"][1], design_bounds["XX"][1]],
            ),
        )

    # Final results
    _, final_mean, final_std, final_min = robust_objective(x)
    robust_airfoil = NACA4(M=x[0], P=x[1], XX=x[2], n_points=100)

    print("\nRobust Optimization Results:")
    print(f"Mean L/D: {final_mean:.3f}")
    print(f"Std L/D:  {final_std:.3f}")
    print(f"Min L/D:  {final_min:.3f}")
    print(f"Robust parameters: M={x[0]:.4f}, P={x[1]:.3f}, XX={x[2]:.4f}")
    print(f"Coefficient of variation: {final_std/final_mean:.3f}")

    return {
        "optimized_params": x,
        "optimized_airfoil": robust_airfoil,
        "mean_performance": final_mean,
        "std_performance": final_std,
        "min_performance": final_min,
        "optimization_history": robust_history,
    }


def plot_optimization_results(
    single_obj_results,
    pareto_results,
    constrained_results,
    robust_results,
):
    """Create comprehensive visualization of optimization results."""
    print("\n=== Creating Optimization Visualizations ===")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Single-objective optimization convergence
    ax1 = axes[0, 0]
    history = single_obj_results["optimization_history"]
    iterations = [h["iteration"] for h in history]
    objectives = [h["objective"] for h in history]

    ax1.plot(iterations, objectives, "b-", linewidth=2, marker="o", markersize=3)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("L/D Ratio")
    ax1.set_title("Single-Objective Optimization Convergence")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Pareto frontier
    ax2 = axes[0, 1]
    avg_ld_values = [r["avg_ld"] for r in pareto_results]
    std_ld_values = [r["ld_std"] for r in pareto_results]
    weights = [r["weight"] for r in pareto_results]

    scatter = ax2.scatter(std_ld_values, avg_ld_values, c=weights, cmap="viridis", s=60)
    ax2.set_xlabel("L/D Standard Deviation")
    ax2.set_ylabel("Average L/D Ratio")
    ax2.set_title("Pareto Frontier: Performance vs Robustness")
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label="Performance Weight")

    # Plot 3: Constrained optimization
    ax3 = axes[0, 2]
    const_history = constrained_results["optimization_history"]
    const_iterations = [h["iteration"] for h in const_history]
    const_ld = [h["ld_ratio"] for h in const_history]
    const_penalties = [h["penalties"] for h in const_history]

    ax3_twin = ax3.twinx()
    line1 = ax3.plot(const_iterations, const_ld, "b-", linewidth=2, label="L/D Ratio")
    line2 = ax3_twin.plot(
        const_iterations,
        const_penalties,
        "r--",
        linewidth=2,
        label="Penalties",
    )

    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("L/D Ratio", color="b")
    ax3_twin.set_ylabel("Penalty Value", color="r")
    ax3.set_title("Constrained Optimization")
    ax3.grid(True, alpha=0.3)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc="upper right")

    # Plot 4: Robust optimization statistics
    ax4 = axes[1, 0]
    robust_history = robust_results["optimization_history"]
    robust_iterations = [h["iteration"] for h in robust_history]
    robust_mean = [h["mean_ld"] for h in robust_history]
    robust_std = [h["std_ld"] for h in robust_history]
    robust_min = [h["min_ld"] for h in robust_history]

    ax4.plot(robust_iterations, robust_mean, "g-", linewidth=2, label="Mean L/D")
    ax4.fill_between(
        robust_iterations,
        np.array(robust_mean) - np.array(robust_std),
        np.array(robust_mean) + np.array(robust_std),
        alpha=0.3,
        color="green",
        label="±1 Std Dev",
    )
    ax4.plot(robust_iterations, robust_min, "r--", linewidth=2, label="Min L/D")

    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("L/D Ratio")
    ax4.set_title("Robust Optimization Statistics")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Optimized airfoil shapes comparison
    ax5 = axes[1, 1]

    # Plot optimized airfoils
    single_airfoil = single_obj_results["optimized_airfoil"]
    constrained_airfoil = constrained_results["optimized_airfoil"]
    robust_airfoil = robust_results["optimized_airfoil"]

    ax5.plot(*single_airfoil.upper_surface, "b-", linewidth=2, label="Single-Objective")
    ax5.plot(*single_airfoil.lower_surface, "b--", linewidth=2)

    ax5.plot(*constrained_airfoil.upper_surface, "r-", linewidth=2, label="Constrained")
    ax5.plot(*constrained_airfoil.lower_surface, "r--", linewidth=2)

    ax5.plot(*robust_airfoil.upper_surface, "g-", linewidth=2, label="Robust")
    ax5.plot(*robust_airfoil.lower_surface, "g--", linewidth=2)

    ax5.set_xlabel("x/c")
    ax5.set_ylabel("y/c")
    ax5.set_title("Optimized Airfoil Shapes")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axis("equal")

    # Plot 6: Parameter evolution
    ax6 = axes[1, 2]

    # Extract parameter evolution from single-objective optimization
    M_evolution = [h["params"][0] for h in history]
    P_evolution = [h["params"][1] for h in history]
    XX_evolution = [h["params"][2] for h in history]

    ax6.plot(iterations, M_evolution, "r-", linewidth=2, label="M (Camber)")
    ax6.plot(iterations, P_evolution, "g-", linewidth=2, label="P (Camber Pos)")
    ax6.plot(iterations, XX_evolution, "b-", linewidth=2, label="XX (Thickness)")

    ax6.set_xlabel("Iteration")
    ax6.set_ylabel("Parameter Value")
    ax6.set_title("Parameter Evolution (Single-Objective)")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """Main function demonstrating shape optimization workflows."""
    print("JAX Airfoil Shape Optimization Examples")
    print("=" * 60)

    # Set up optimization problem
    design_bounds, operating_conditions, constraints = setup_optimization_problem()

    # Single-objective optimization
    single_obj_results = single_objective_optimization(
        design_bounds,
        operating_conditions,
        constraints,
    )

    # Multi-objective optimization
    pareto_results = multi_objective_optimization(
        design_bounds,
        operating_conditions,
        constraints,
    )

    # Constrained optimization
    constrained_results = constrained_optimization(
        design_bounds,
        operating_conditions,
        constraints,
    )

    # Robust optimization
    robust_results = robust_optimization_under_uncertainty(
        design_bounds,
        operating_conditions,
        constraints,
    )

    # Create visualizations
    plot_optimization_results(
        single_obj_results,
        pareto_results,
        constrained_results,
        robust_results,
    )

    print("\n" + "=" * 60)
    print("Key Shape Optimization Capabilities:")
    print("1. Single-objective optimization with gradient descent")
    print("2. Multi-objective Pareto frontier analysis")
    print("3. Constrained optimization with penalty methods")
    print("4. Robust optimization under operating condition uncertainty")
    print("5. Automatic differentiation through complex objectives")
    print("6. JIT compilation for efficient optimization loops")
    print("7. Integration with gradient-based optimizers")

    # Summary of results
    print("\nOptimization Summary:")
    print(f"Single-objective improvement: {single_obj_results['improvement']:.3f}")
    print(f"Constrained optimum L/D: {constrained_results['final_performance']:.3f}")
    print(
        f"Robust optimum mean L/D: {robust_results['mean_performance']:.3f} ± {robust_results['std_performance']:.3f}",
    )

    return {
        "single_objective": single_obj_results,
        "pareto_frontier": pareto_results,
        "constrained": constrained_results,
        "robust": robust_results,
    }


if __name__ == "__main__":
    main()
