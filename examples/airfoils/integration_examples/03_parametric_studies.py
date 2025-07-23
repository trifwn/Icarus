#!/usr/bin/env python3
"""
Parametric Studies with Batch Processing

This example demonstrates comprehensive parametric studies using JAX airfoils:
1. Design space exploration with systematic parameter sweeps
2. Sensitivity analysis across multiple design variables
3. Response surface modeling and visualization
4. Statistical analysis of design parameter effects
5. Batch processing for efficient parameter space exploration
6. Design of experiments (DOE) methodologies

The JAX implementation enables efficient batch processing and automatic
differentiation for comprehensive parametric analysis workflows.
"""

import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad
from jax import jit
from jax import vmap

from ICARUS.airfoils.naca4 import NACA4


def setup_parametric_study():
    """Set up the parametric study design space and analysis conditions."""
    print("=== Setting Up Parametric Study ===")

    # Design space for NACA 4-digit airfoils
    design_space = {
        "M": jnp.linspace(0.0, 0.06, 7),  # Maximum camber: 0-6%
        "P": jnp.linspace(0.2, 0.8, 7),  # Camber position: 20-80%
        "XX": jnp.linspace(0.08, 0.18, 6),  # Thickness: 8-18%
    }

    # Analysis conditions
    analysis_conditions = {
        "reynolds": jnp.array([100000.0, 300000.0, 500000.0]),
        "angles": jnp.linspace(-4, 12, 9),  # degrees
        "mach": 0.1,
    }

    # Performance metrics to evaluate
    metrics = [
        "max_ld_ratio",  # Maximum L/D ratio
        "cruise_ld_ratio",  # L/D at cruise (4 degrees)
        "stall_angle",  # Approximate stall angle
        "zero_lift_angle",  # Zero lift angle of attack
        "max_thickness",  # Maximum thickness
        "drag_coefficient",  # Drag at cruise
    ]

    total_designs = (
        len(design_space["M"]) * len(design_space["P"]) * len(design_space["XX"])
    )
    total_analyses = (
        total_designs
        * len(analysis_conditions["reynolds"])
        * len(analysis_conditions["angles"])
    )

    print("Design space dimensions:")
    print(
        f"  Camber (M): {len(design_space['M'])} values from {design_space['M'][0]:.3f} to {design_space['M'][-1]:.3f}",
    )
    print(
        f"  Position (P): {len(design_space['P'])} values from {design_space['P'][0]:.1f} to {design_space['P'][-1]:.1f}",
    )
    print(
        f"  Thickness (XX): {len(design_space['XX'])} values from {design_space['XX'][0]:.3f} to {design_space['XX'][-1]:.3f}",
    )
    print(f"Total design points: {total_designs}")
    print(f"Total analyses: {total_analyses}")
    print(f"Performance metrics: {len(metrics)}")

    return design_space, analysis_conditions, metrics


@jit
def aerodynamic_analysis_batch(
    upper_surface,
    lower_surface,
    reynolds_array,
    angle_array,
):
    """
    Batch aerodynamic analysis for parametric studies.

    This function is vectorized to handle multiple Reynolds numbers and angles efficiently.
    """

    # Vectorize over both Reynolds numbers and angles
    def single_analysis(reynolds, angle_rad):
        # Extract coordinates
        x_coords = upper_surface[0, :]
        y_upper = upper_surface[1, :]
        y_lower = lower_surface[1, :]

        # Calculate geometric properties
        thickness = y_upper - y_lower
        camber = 0.5 * (y_upper + y_lower)

        # Lift coefficient calculation
        camber_slope = jnp.gradient(camber, x_coords)
        cl_alpha = 2 * jnp.pi
        cl_camber = 2 * jnp.pi * jnp.trapz(camber_slope, x_coords)
        cl = cl_alpha * angle_rad + cl_camber

        # Drag coefficient calculation
        max_thickness = jnp.max(thickness)
        reynolds_factor = jnp.log(reynolds) / 15.0
        cd_form = 0.015 * max_thickness**2
        cd_friction = 0.008 / jnp.sqrt(reynolds_factor)
        cd_induced = cl**2 / (jnp.pi * 8.0)
        cd = cd_form + cd_friction + cd_induced

        # Moment coefficient
        cm = -cl * 0.05  # Simplified moment coefficient

        return cl, cd, cm

    # Vectorize the analysis function
    vectorized_analysis = vmap(
        vmap(single_analysis, in_axes=(None, 0)),
        in_axes=(0, None),
    )

    # Perform batch analysis
    cl_matrix, cd_matrix, cm_matrix = vectorized_analysis(reynolds_array, angle_array)

    return cl_matrix, cd_matrix, cm_matrix


def compute_performance_metrics(cl_matrix, cd_matrix, cm_matrix, angles):
    """Compute performance metrics from aerodynamic analysis results."""

    # Maximum L/D ratio
    ld_matrix = cl_matrix / (cd_matrix + 1e-6)
    max_ld_ratio = jnp.max(ld_matrix)

    # L/D at cruise condition (4 degrees)
    cruise_angle_idx = jnp.argmin(jnp.abs(angles - 4.0))
    cruise_ld_ratio = jnp.mean(
        ld_matrix[:, cruise_angle_idx],
    )  # Average over Reynolds numbers

    # Approximate stall angle (where dCL/dα drops significantly)
    cl_gradients = jnp.gradient(jnp.mean(cl_matrix, axis=0), angles)
    stall_idx = (
        jnp.argmin(cl_gradients[len(cl_gradients) // 2 :]) + len(cl_gradients) // 2
    )
    stall_angle = angles[stall_idx]

    # Zero lift angle
    cl_mean = jnp.mean(cl_matrix, axis=0)  # Average over Reynolds numbers
    zero_lift_idx = jnp.argmin(jnp.abs(cl_mean))
    zero_lift_angle = angles[zero_lift_idx]

    # Drag coefficient at cruise
    cruise_drag = jnp.mean(cd_matrix[:, cruise_angle_idx])

    return {
        "max_ld_ratio": max_ld_ratio,
        "cruise_ld_ratio": cruise_ld_ratio,
        "stall_angle": stall_angle,
        "zero_lift_angle": zero_lift_angle,
        "drag_coefficient": cruise_drag,
    }


def full_factorial_study(design_space, analysis_conditions, metrics):
    """Perform full factorial design space exploration."""
    print("\n=== Full Factorial Design Space Exploration ===")

    # Create all parameter combinations
    M_values = design_space["M"]
    P_values = design_space["P"]
    XX_values = design_space["XX"]

    # Initialize results storage
    results = {
        "parameters": [],
        "airfoils": [],
        "performance_metrics": [],
        "analysis_time": [],
    }

    total_combinations = len(M_values) * len(P_values) * len(XX_values)
    print(f"Analyzing {total_combinations} design combinations...")

    start_time = time.time()

    # Batch process all combinations
    combination_count = 0

    for M in M_values:
        for P in P_values:
            for XX in XX_values:
                combination_count += 1

                # Create airfoil
                airfoil = NACA4(M=M, P=P, XX=XX, n_points=100)

                # Perform aerodynamic analysis
                analysis_start = time.time()

                cl_matrix, cd_matrix, cm_matrix = aerodynamic_analysis_batch(
                    airfoil.upper_surface,
                    airfoil.lower_surface,
                    analysis_conditions["reynolds"],
                    jnp.deg2rad(analysis_conditions["angles"]),
                )

                analysis_time = time.time() - analysis_start

                # Compute performance metrics
                performance = compute_performance_metrics(
                    cl_matrix,
                    cd_matrix,
                    cm_matrix,
                    analysis_conditions["angles"],
                )

                # Add geometric metrics
                performance["max_thickness"] = airfoil.max_thickness
                performance["max_thickness_location"] = airfoil.max_thickness_location

                # Store results
                results["parameters"].append({"M": M, "P": P, "XX": XX})
                results["airfoils"].append(airfoil)
                results["performance_metrics"].append(performance)
                results["analysis_time"].append(analysis_time)

                # Progress reporting
                if (
                    combination_count % 50 == 0
                    or combination_count == total_combinations
                ):
                    elapsed = time.time() - start_time
                    avg_time = elapsed / combination_count
                    remaining = (total_combinations - combination_count) * avg_time
                    print(
                        f"  Progress: {combination_count}/{total_combinations} "
                        f"({100 * combination_count / total_combinations:.1f}%) "
                        f"- ETA: {remaining:.1f}s",
                    )

    total_time = time.time() - start_time
    avg_analysis_time = np.mean(results["analysis_time"])

    print(f"Full factorial study completed in {total_time:.2f} seconds")
    print(f"Average analysis time per design: {avg_analysis_time:.4f} seconds")
    print(
        f"Total speedup from batch processing: {total_combinations * avg_analysis_time / total_time:.1f}x",
    )

    return results


def sensitivity_analysis(design_space, analysis_conditions):
    """Perform sensitivity analysis using gradient computation."""
    print("\n=== Gradient-Based Sensitivity Analysis ===")

    def performance_function(params, reynolds, angle_rad):
        """Performance function for sensitivity analysis."""
        M, P, XX = params

        # Create airfoil
        airfoil = NACA4(M=M, P=P, XX=XX, n_points=100)

        # Analyze performance
        cl_matrix, cd_matrix, cm_matrix = aerodynamic_analysis_batch(
            airfoil.upper_surface,
            airfoil.lower_surface,
            reynolds[None, :],  # Add batch dimension
            angle_rad[None, :],  # Add batch dimension
        )

        # Compute L/D ratio
        ld_matrix = cl_matrix / (cd_matrix + 1e-6)

        # Return mean L/D ratio
        return jnp.mean(ld_matrix)

    # Compute gradients
    grad_performance = grad(performance_function, argnums=0)

    # Test points for sensitivity analysis
    test_points = [
        {"M": 0.00, "P": 0.4, "XX": 0.12, "name": "Symmetric"},
        {"M": 0.02, "P": 0.4, "XX": 0.12, "name": "Low Camber"},
        {"M": 0.04, "P": 0.4, "XX": 0.12, "name": "High Camber"},
        {"M": 0.02, "P": 0.3, "XX": 0.12, "name": "Forward Camber"},
        {"M": 0.02, "P": 0.6, "XX": 0.12, "name": "Aft Camber"},
        {"M": 0.02, "P": 0.4, "XX": 0.09, "name": "Thin"},
        {"M": 0.02, "P": 0.4, "XX": 0.15, "name": "Thick"},
    ]

    sensitivity_results = []

    print("Computing sensitivities at test points:")
    print("Design Point      L/D    ∂L/D/∂M   ∂L/D/∂P   ∂L/D/∂XX")
    print("-" * 60)

    for point in test_points:
        params = jnp.array([point["M"], point["P"], point["XX"]])

        # Compute performance and gradients
        performance = performance_function(
            params,
            analysis_conditions["reynolds"],
            jnp.deg2rad(analysis_conditions["angles"]),
        )

        gradients = grad_performance(
            params,
            analysis_conditions["reynolds"],
            jnp.deg2rad(analysis_conditions["angles"]),
        )

        sensitivity_results.append(
            {
                "name": point["name"],
                "params": params,
                "performance": performance,
                "gradients": gradients,
            },
        )

        print(
            f"{point['name']:15s} {performance:6.2f}  {gradients[0]:+8.2f}  {gradients[1]:+8.2f}  {gradients[2]:+8.2f}",
        )

    return sensitivity_results


def response_surface_modeling(factorial_results):
    """Create response surface models from factorial study results."""
    print("\n=== Response Surface Modeling ===")

    # Extract data for response surface fitting
    parameters = factorial_results["parameters"]
    performance_metrics = factorial_results["performance_metrics"]

    # Create parameter arrays
    M_array = jnp.array([p["M"] for p in parameters])
    P_array = jnp.array([p["P"] for p in parameters])
    XX_array = jnp.array([p["XX"] for p in parameters])

    # Create response arrays
    max_ld_array = jnp.array([pm["max_ld_ratio"] for pm in performance_metrics])
    cruise_ld_array = jnp.array([pm["cruise_ld_ratio"] for pm in performance_metrics])
    drag_array = jnp.array([pm["drag_coefficient"] for pm in performance_metrics])

    print("Response surface data:")
    print(f"  Max L/D range: {jnp.min(max_ld_array):.2f} - {jnp.max(max_ld_array):.2f}")
    print(
        f"  Cruise L/D range: {jnp.min(cruise_ld_array):.2f} - {jnp.max(cruise_ld_array):.2f}",
    )
    print(
        f"  Drag coefficient range: {jnp.min(drag_array):.4f} - {jnp.max(drag_array):.4f}",
    )

    # Find optimal designs
    max_ld_idx = jnp.argmax(max_ld_array)
    min_drag_idx = jnp.argmin(drag_array)

    optimal_max_ld = parameters[max_ld_idx]
    optimal_min_drag = parameters[min_drag_idx]

    print("\nOptimal designs:")
    print(
        f"  Max L/D: M={optimal_max_ld['M']:.3f}, P={optimal_max_ld['P']:.1f}, XX={optimal_max_ld['XX']:.3f} "
        f"(L/D = {max_ld_array[max_ld_idx]:.2f})",
    )
    print(
        f"  Min Drag: M={optimal_min_drag['M']:.3f}, P={optimal_min_drag['P']:.1f}, XX={optimal_min_drag['XX']:.3f} "
        f"(CD = {drag_array[min_drag_idx]:.4f})",
    )

    # Statistical analysis
    print("\nStatistical analysis:")

    # Correlation analysis
    correlation_M_LD = jnp.corrcoef(M_array, max_ld_array)[0, 1]
    correlation_P_LD = jnp.corrcoef(P_array, max_ld_array)[0, 1]
    correlation_XX_LD = jnp.corrcoef(XX_array, max_ld_array)[0, 1]

    print("  Correlations with max L/D:")
    print(f"    M (camber): {correlation_M_LD:+.3f}")
    print(f"    P (position): {correlation_P_LD:+.3f}")
    print(f"    XX (thickness): {correlation_XX_LD:+.3f}")

    return {
        "parameters": {"M": M_array, "P": P_array, "XX": XX_array},
        "responses": {
            "max_ld": max_ld_array,
            "cruise_ld": cruise_ld_array,
            "drag": drag_array,
        },
        "optimal_designs": {"max_ld": optimal_max_ld, "min_drag": optimal_min_drag},
        "correlations": {
            "M": correlation_M_LD,
            "P": correlation_P_LD,
            "XX": correlation_XX_LD,
        },
    }


def design_of_experiments_study(design_space, analysis_conditions):
    """Perform Design of Experiments (DOE) study with Latin Hypercube Sampling."""
    print("\n=== Design of Experiments Study ===")

    # Latin Hypercube Sampling for efficient design space exploration
    n_samples = 50

    print(f"Generating {n_samples} design points using Latin Hypercube Sampling...")

    # Generate LHS samples (simplified implementation)
    # In practice, you would use a proper LHS library
    np.random.seed(42)  # For reproducibility

    # Generate uniform random samples
    lhs_samples = np.random.rand(n_samples, 3)

    # Map to design space bounds
    M_min, M_max = design_space["M"][0], design_space["M"][-1]
    P_min, P_max = design_space["P"][0], design_space["P"][-1]
    XX_min, XX_max = design_space["XX"][0], design_space["XX"][-1]

    M_samples = M_min + lhs_samples[:, 0] * (M_max - M_min)
    P_samples = P_min + lhs_samples[:, 1] * (P_max - P_min)
    XX_samples = XX_min + lhs_samples[:, 2] * (XX_max - XX_min)

    # Analyze DOE samples
    doe_results = {"parameters": [], "performance_metrics": [], "analysis_time": []}

    print("Analyzing DOE samples...")
    start_time = time.time()

    for i in range(n_samples):
        M, P, XX = M_samples[i], P_samples[i], XX_samples[i]

        # Create airfoil
        airfoil = NACA4(M=M, P=P, XX=XX, n_points=100)

        # Perform analysis
        analysis_start = time.time()

        cl_matrix, cd_matrix, cm_matrix = aerodynamic_analysis_batch(
            airfoil.upper_surface,
            airfoil.lower_surface,
            analysis_conditions["reynolds"],
            jnp.deg2rad(analysis_conditions["angles"]),
        )

        analysis_time = time.time() - analysis_start

        # Compute metrics
        performance = compute_performance_metrics(
            cl_matrix,
            cd_matrix,
            cm_matrix,
            analysis_conditions["angles"],
        )
        performance["max_thickness"] = airfoil.max_thickness

        # Store results
        doe_results["parameters"].append({"M": M, "P": P, "XX": XX})
        doe_results["performance_metrics"].append(performance)
        doe_results["analysis_time"].append(analysis_time)

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{n_samples} samples")

    total_time = time.time() - start_time
    print(f"DOE study completed in {total_time:.2f} seconds")

    # Find best designs from DOE
    max_ld_values = [pm["max_ld_ratio"] for pm in doe_results["performance_metrics"]]
    best_idx = np.argmax(max_ld_values)
    best_design = doe_results["parameters"][best_idx]
    best_performance = max_ld_values[best_idx]

    print(
        f"Best design from DOE: M={best_design['M']:.3f}, P={best_design['P']:.3f}, XX={best_design['XX']:.3f}",
    )
    print(f"Best performance: L/D = {best_performance:.2f}")

    return doe_results


def plot_parametric_study_results(
    factorial_results,
    sensitivity_results,
    response_surface_results,
    doe_results,
):
    """Create comprehensive visualization of parametric study results."""
    print("\n=== Creating Parametric Study Visualizations ===")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Response surface - Max L/D vs M and P
    ax1 = axes[0, 0]

    # Extract data for contour plot
    params = response_surface_results["parameters"]
    responses = response_surface_results["responses"]

    # Create grid for contour plot
    M_unique = np.unique(params["M"])
    P_unique = np.unique(params["P"])
    M_grid, P_grid = np.meshgrid(M_unique, P_unique)

    # Reshape max L/D data for contour plot (average over thickness)
    LD_grid = np.zeros_like(M_grid)
    for i, M_val in enumerate(M_unique):
        for j, P_val in enumerate(P_unique):
            # Find indices matching M and P values
            mask = (np.abs(params["M"] - M_val) < 1e-6) & (
                np.abs(params["P"] - P_val) < 1e-6
            )
            if np.any(mask):
                LD_grid[j, i] = np.mean(responses["max_ld"][mask])

    contour = ax1.contourf(M_grid, P_grid, LD_grid, levels=15, cmap="viridis")
    ax1.contour(
        M_grid,
        P_grid,
        LD_grid,
        levels=15,
        colors="black",
        alpha=0.3,
        linewidths=0.5,
    )
    plt.colorbar(contour, ax=ax1, label="Max L/D Ratio")
    ax1.set_xlabel("Camber (M)")
    ax1.set_ylabel("Camber Position (P)")
    ax1.set_title("Response Surface: Max L/D vs M and P")

    # Plot 2: Parameter sensitivity
    ax2 = axes[0, 1]

    names = [sr["name"] for sr in sensitivity_results]
    gradients_M = [sr["gradients"][0] for sr in sensitivity_results]
    gradients_P = [sr["gradients"][1] for sr in sensitivity_results]
    gradients_XX = [sr["gradients"][2] for sr in sensitivity_results]

    x = np.arange(len(names))
    width = 0.25

    ax2.bar(x - width, gradients_M, width, label="∂L/D/∂M", alpha=0.8)
    ax2.bar(x, gradients_P, width, label="∂L/D/∂P", alpha=0.8)
    ax2.bar(x + width, gradients_XX, width, label="∂L/D/∂XX", alpha=0.8)

    ax2.set_xlabel("Design Point")
    ax2.set_ylabel("Sensitivity")
    ax2.set_title("Parameter Sensitivity Analysis")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Thickness vs Performance
    ax3 = axes[0, 2]

    thickness_values = params["XX"]
    max_ld_values = responses["max_ld"]
    drag_values = responses["drag"]

    # Scatter plot with color coding
    scatter = ax3.scatter(
        thickness_values,
        max_ld_values,
        c=drag_values,
        cmap="coolwarm",
        alpha=0.6,
        s=30,
    )
    plt.colorbar(scatter, ax=ax3, label="Drag Coefficient")
    ax3.set_xlabel("Thickness (XX)")
    ax3.set_ylabel("Max L/D Ratio")
    ax3.set_title("Thickness vs Performance (Color: Drag)")
    ax3.grid(True, alpha=0.3)

    # Plot 4: DOE vs Full Factorial comparison
    ax4 = axes[1, 0]

    # Compare performance distributions
    factorial_ld = responses["max_ld"]
    doe_ld = [pm["max_ld_ratio"] for pm in doe_results["performance_metrics"]]

    ax4.hist(
        factorial_ld,
        bins=20,
        alpha=0.7,
        label=f"Full Factorial (n={len(factorial_ld)})",
        density=True,
    )
    ax4.hist(doe_ld, bins=15, alpha=0.7, label=f"DOE (n={len(doe_ld)})", density=True)
    ax4.set_xlabel("Max L/D Ratio")
    ax4.set_ylabel("Density")
    ax4.set_title("Performance Distribution Comparison")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Parameter correlation matrix
    ax5 = axes[1, 1]

    # Create correlation matrix
    correlation_matrix = np.array(
        [
            [1.0, 0.0, 0.0],  # M vs M, P, XX
            [0.0, 1.0, 0.0],  # P vs M, P, XX
            [
                response_surface_results["correlations"]["M"],
                response_surface_results["correlations"]["P"],
                response_surface_results["correlations"]["XX"],
            ],  # L/D vs M, P, XX
        ],
    )

    im = ax5.imshow(correlation_matrix, cmap="RdBu", vmin=-1, vmax=1)
    ax5.set_xticks([0, 1, 2])
    ax5.set_yticks([0, 1, 2])
    ax5.set_xticklabels(["M", "P", "XX"])
    ax5.set_yticklabels(["M", "P", "L/D"])
    ax5.set_title("Parameter Correlation Matrix")

    # Add correlation values as text
    for i in range(3):
        for j in range(3):
            text = ax5.text(
                j,
                i,
                f"{correlation_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )

    plt.colorbar(im, ax=ax5)

    # Plot 6: Optimal designs comparison
    ax6 = axes[1, 2]

    # Show optimal airfoil shapes
    optimal_max_ld = response_surface_results["optimal_designs"]["max_ld"]
    optimal_min_drag = response_surface_results["optimal_designs"]["min_drag"]

    # Create optimal airfoils
    airfoil_max_ld = NACA4(
        M=optimal_max_ld["M"],
        P=optimal_max_ld["P"],
        XX=optimal_max_ld["XX"],
        n_points=100,
    )
    airfoil_min_drag = NACA4(
        M=optimal_min_drag["M"],
        P=optimal_min_drag["P"],
        XX=optimal_min_drag["XX"],
        n_points=100,
    )

    ax6.plot(*airfoil_max_ld.upper_surface, "b-", linewidth=2, label="Max L/D Optimal")
    ax6.plot(*airfoil_max_ld.lower_surface, "b--", linewidth=2)
    ax6.plot(
        *airfoil_min_drag.upper_surface,
        "r-",
        linewidth=2,
        label="Min Drag Optimal",
    )
    ax6.plot(*airfoil_min_drag.lower_surface, "r--", linewidth=2)

    ax6.set_xlabel("x/c")
    ax6.set_ylabel("y/c")
    ax6.set_title("Optimal Airfoil Shapes")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.axis("equal")

    plt.tight_layout()
    plt.show()


def main():
    """Main function demonstrating parametric studies with batch processing."""
    print("JAX Airfoil Parametric Studies with Batch Processing")
    print("=" * 70)

    # Set up parametric study
    design_space, analysis_conditions, metrics = setup_parametric_study()

    # Full factorial design space exploration
    factorial_results = full_factorial_study(design_space, analysis_conditions, metrics)

    # Gradient-based sensitivity analysis
    sensitivity_results = sensitivity_analysis(design_space, analysis_conditions)

    # Response surface modeling
    response_surface_results = response_surface_modeling(factorial_results)

    # Design of experiments study
    doe_results = design_of_experiments_study(design_space, analysis_conditions)

    # Create comprehensive visualizations
    plot_parametric_study_results(
        factorial_results,
        sensitivity_results,
        response_surface_results,
        doe_results,
    )

    print("\n" + "=" * 70)
    print("Key Parametric Study Capabilities:")
    print("1. Full factorial design space exploration with batch processing")
    print("2. Gradient-based sensitivity analysis using automatic differentiation")
    print("3. Response surface modeling and statistical analysis")
    print("4. Design of Experiments (DOE) with Latin Hypercube Sampling")
    print("5. Efficient vectorized analysis across parameter combinations")
    print("6. Comprehensive visualization of design space relationships")
    print("7. Optimal design identification and comparison")

    # Performance summary
    total_factorial_time = sum(factorial_results["analysis_time"])
    total_doe_time = sum(doe_results["analysis_time"])
    factorial_designs = len(factorial_results["parameters"])
    doe_designs = len(doe_results["parameters"])

    print("\nPerformance Summary:")
    print(
        f"Full factorial: {factorial_designs} designs in {total_factorial_time:.2f}s "
        f"({total_factorial_time / factorial_designs:.4f}s per design)",
    )
    print(
        f"DOE study: {doe_designs} designs in {total_doe_time:.2f}s "
        f"({total_doe_time / doe_designs:.4f}s per design)",
    )
    print("JAX batch processing enables efficient parametric exploration")

    return {
        "design_space": design_space,
        "analysis_conditions": analysis_conditions,
        "factorial_results": factorial_results,
        "sensitivity_results": sensitivity_results,
        "response_surface_results": response_surface_results,
        "doe_results": doe_results,
    }


if __name__ == "__main__":
    main()
