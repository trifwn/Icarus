#!/usr/bin/env python3
"""
Aerodynamic Analysis Workflow with JAX Airfoils

This example demonstrates how to integrate JAX airfoils into aerodynamic analysis workflows:
1. Setting up airfoils for CFD analysis using JAX implementation
2. Batch processing multiple airfoils with different Reynolds numbers
3. Gradient-based sensitivity analysis of aerodynamic parameters
4. Integration with existing ICARUS solvers (Xfoil, Foil2Wake)
5. Performance comparison between JAX and NumPy implementations

The JAX implementation enables automatic differentiation through the entire
analysis pipeline, making it ideal for optimization and sensitivity studies.
"""

import time
from typing import Dict
from typing import List

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad
from jax import jit
from jax import vmap

from ICARUS.airfoils import Airfoil
from ICARUS.airfoils.naca4 import NACA4
from ICARUS.core.units import calc_reynolds


def setup_analysis_conditions():
    """Set up realistic aerodynamic analysis conditions."""
    print("=== Setting Up Analysis Conditions ===")

    # Flight conditions
    conditions = {
        "chord_range": (0.06, 0.16),  # meters
        "velocity_range": (5.0, 35.0),  # m/s
        "viscosity": 1.56e-5,  # m²/s
        "mach": 0.085,
        "angles": jnp.linspace(-8, 14, 23),  # degrees
    }

    # Calculate Reynolds number range
    chord_min, chord_max = conditions["chord_range"]
    u_min, u_max = conditions["velocity_range"]
    viscosity = conditions["viscosity"]

    reynolds_min = calc_reynolds(u_min, chord_min, viscosity)
    reynolds_max = calc_reynolds(u_max, chord_max, viscosity)
    reynolds_numbers = jnp.linspace(reynolds_min, reynolds_max, 8)

    conditions["reynolds"] = reynolds_numbers

    print(f"Reynolds range: {reynolds_min:.0f} - {reynolds_max:.0f}")
    print(
        f"Angle range: {conditions['angles'][0]:.1f}° - {conditions['angles'][-1]:.1f}°",
    )
    print(f"Number of Reynolds numbers: {len(reynolds_numbers)}")
    print(f"Number of angles: {len(conditions['angles'])}")

    return conditions


def create_airfoil_family_jax():
    """Create a family of airfoils using JAX implementation."""
    print("\n=== Creating JAX Airfoil Family ===")

    # Define NACA 4-digit parameters for systematic study
    airfoil_specs = [
        {"M": 0.00, "P": 0.0, "XX": 0.09, "name": "NACA0009"},
        {"M": 0.00, "P": 0.0, "XX": 0.12, "name": "NACA0012"},
        {"M": 0.00, "P": 0.0, "XX": 0.15, "name": "NACA0015"},
        {"M": 0.02, "P": 0.4, "XX": 0.12, "name": "NACA2412"},
        {"M": 0.04, "P": 0.4, "XX": 0.15, "name": "NACA4415"},
        {"M": 0.06, "P": 0.4, "XX": 0.09, "name": "NACA6409"},
    ]

    airfoils = []
    for spec in airfoil_specs:
        airfoil = NACA4(M=spec["M"], P=spec["P"], XX=spec["XX"], n_points=200)
        airfoil.name = spec["name"]  # Override name for clarity
        airfoils.append(airfoil)

        print(
            f"Created {airfoil.name}: "
            f"t/c={airfoil.max_thickness:.3f}, "
            f"camber={spec['M']:.3f}",
        )

    return airfoils


@jit
def simplified_aerodynamic_analysis(upper_surface, lower_surface, reynolds, angle_rad):
    """
    Simplified aerodynamic analysis using panel method approximations.

    This is a simplified model for demonstration. In practice, you would
    interface with actual CFD solvers or panel methods.
    """
    # Extract coordinates
    x_coords = upper_surface[0, :]
    y_upper = upper_surface[1, :]
    y_lower = lower_surface[1, :]

    # Calculate geometric properties
    thickness = y_upper - y_lower
    camber = 0.5 * (y_upper + y_lower)

    # Simplified lift coefficient calculation (thin airfoil theory)
    # This is a very simplified model for demonstration
    camber_slope = jnp.gradient(camber, x_coords)
    cl_alpha = 2 * jnp.pi  # 2D lift curve slope
    cl_camber = 2 * jnp.pi * jnp.trapz(camber_slope, x_coords)
    cl = cl_alpha * angle_rad + cl_camber

    # Simplified drag coefficient (form drag approximation)
    max_thickness = jnp.max(thickness)
    reynolds_factor = jnp.log(reynolds) / 15.0  # Rough Reynolds scaling
    cd_form = 0.02 * max_thickness**2
    cd_friction = 0.01 / jnp.sqrt(reynolds_factor)
    cd = cd_form + cd_friction

    # Moment coefficient (simplified)
    x_ac = 0.25  # Aerodynamic center at quarter chord
    cm = -cl * (x_ac - 0.25)  # Moment about quarter chord

    return cl, cd, cm


def batch_aerodynamic_analysis(airfoils: List[Airfoil], conditions: Dict):
    """Perform batch aerodynamic analysis using JAX vectorization."""
    print("\n=== Batch Aerodynamic Analysis ===")

    # Vectorize the analysis function
    vectorized_analysis = vmap(
        vmap(simplified_aerodynamic_analysis, in_axes=(None, None, None, 0)),
        in_axes=(None, None, 0, None),
    )

    results = {}

    for airfoil in airfoils:
        print(f"Analyzing {airfoil.name}...")

        # Convert angles to radians
        angles_rad = jnp.deg2rad(conditions["angles"])

        # Perform vectorized analysis over Reynolds numbers and angles
        start_time = time.time()

        cl_matrix, cd_matrix, cm_matrix = vectorized_analysis(
            airfoil.upper_surface,
            airfoil.lower_surface,
            conditions["reynolds"][:, None],  # Broadcast over angles
            angles_rad[None, :],  # Broadcast over Reynolds numbers
        )

        analysis_time = time.time() - start_time

        results[airfoil.name] = {
            "cl": cl_matrix,
            "cd": cd_matrix,
            "cm": cm_matrix,
            "analysis_time": analysis_time,
            "reynolds": conditions["reynolds"],
            "angles": conditions["angles"],
        }

        print(f"  Completed in {analysis_time:.4f} seconds")
        print(f"  CL range: {jnp.min(cl_matrix):.3f} to {jnp.max(cl_matrix):.3f}")
        print(f"  CD range: {jnp.min(cd_matrix):.4f} to {jnp.max(cd_matrix):.4f}")

    return results


def gradient_sensitivity_analysis(airfoils: List[Airfoil], conditions: Dict):
    """Perform gradient-based sensitivity analysis."""
    print("\n=== Gradient Sensitivity Analysis ===")

    def airfoil_performance_metric(naca_params, reynolds, angle_rad):
        """Performance metric as function of NACA parameters."""
        M, P, XX = naca_params

        # Create airfoil from parameters
        airfoil = NACA4(M=M, P=P, XX=XX, n_points=100)

        # Analyze performance
        cl, cd, cm = simplified_aerodynamic_analysis(
            airfoil.upper_surface,
            airfoil.lower_surface,
            reynolds,
            angle_rad,
        )

        # Performance metric: maximize L/D ratio
        ld_ratio = cl / (cd + 1e-6)  # Add small epsilon to avoid division by zero
        return ld_ratio

    # Compute gradients
    grad_performance = grad(airfoil_performance_metric, argnums=0)

    # Test conditions
    test_reynolds = conditions["reynolds"][4]  # Middle Reynolds number
    test_angle = jnp.deg2rad(4.0)  # 4 degree angle of attack

    sensitivity_results = {}

    for airfoil in airfoils[:3]:  # Analyze first 3 airfoils for demonstration
        print(f"\nSensitivity analysis for {airfoil.name}:")

        # Extract NACA parameters (approximate from airfoil properties)
        # In practice, you'd store these or extract them properly
        if "NACA0009" in airfoil.name:
            params = jnp.array([0.00, 0.0, 0.09])
        elif "NACA0012" in airfoil.name:
            params = jnp.array([0.00, 0.0, 0.12])
        elif "NACA2412" in airfoil.name:
            params = jnp.array([0.02, 0.4, 0.12])
        else:
            continue

        # Compute performance and gradients
        performance = airfoil_performance_metric(params, test_reynolds, test_angle)
        gradients = grad_performance(params, test_reynolds, test_angle)

        sensitivity_results[airfoil.name] = {
            "performance": float(performance),
            "gradients": {
                "dLDdM": float(gradients[0]),  # Sensitivity to camber
                "dLDdP": float(gradients[1]),  # Sensitivity to camber position
                "dLDdXX": float(gradients[2]),  # Sensitivity to thickness
            },
        }

        print(f"  L/D ratio: {performance:.2f}")
        print(f"  ∂(L/D)/∂M:  {gradients[0]:+.2f} (camber sensitivity)")
        print(f"  ∂(L/D)/∂P:  {gradients[1]:+.2f} (camber position sensitivity)")
        print(f"  ∂(L/D)/∂XX: {gradients[2]:+.2f} (thickness sensitivity)")

    return sensitivity_results


def performance_comparison_study():
    """Compare JAX vs NumPy implementation performance."""
    print("\n=== Performance Comparison Study ===")

    # Create test airfoil
    test_airfoil = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

    # Test conditions
    n_reynolds = 10
    n_angles = 20
    reynolds_test = jnp.linspace(50000, 500000, n_reynolds)
    angles_test = jnp.linspace(-10, 15, n_angles)
    angles_rad_test = jnp.deg2rad(angles_test)

    print(
        f"Performance test: {n_reynolds} Reynolds × {n_angles} angles = {n_reynolds * n_angles} evaluations",
    )

    # JAX vectorized version
    vectorized_jax = vmap(
        vmap(simplified_aerodynamic_analysis, in_axes=(None, None, None, 0)),
        in_axes=(None, None, 0, None),
    )

    # Warm up JIT compilation
    print("Warming up JIT compilation...")
    _ = vectorized_jax(
        test_airfoil.upper_surface,
        test_airfoil.lower_surface,
        reynolds_test[:, None],
        angles_rad_test[None, :],
    )

    # Time JAX version
    print("Timing JAX implementation...")
    start_time = time.time()
    for _ in range(10):  # Multiple runs for better timing
        jax_results = vectorized_jax(
            test_airfoil.upper_surface,
            test_airfoil.lower_surface,
            reynolds_test[:, None],
            angles_rad_test[None, :],
        )
    jax_time = (time.time() - start_time) / 10

    # Simulate NumPy version timing (simplified)
    print("Simulating NumPy implementation timing...")
    start_time = time.time()
    numpy_results = []
    for re in reynolds_test:
        for angle in angles_rad_test:
            # Convert to numpy for simulation
            upper_np = np.array(test_airfoil.upper_surface)
            lower_np = np.array(test_airfoil.lower_surface)

            # Simplified numpy calculation (not actual implementation)
            thickness = upper_np[1, :] - lower_np[1, :]
            max_thickness = np.max(thickness)
            cl_simple = 2 * np.pi * float(angle) + 0.1
            cd_simple = 0.02 * max_thickness**2 + 0.01 / np.sqrt(
                np.log(float(re)) / 15.0,
            )
            numpy_results.append((cl_simple, cd_simple))

    numpy_time = time.time() - start_time

    # Performance summary
    speedup = numpy_time / jax_time
    print("\nPerformance Results:")
    print(f"JAX time:   {jax_time:.4f} seconds")
    print(f"NumPy time: {numpy_time:.4f} seconds (simulated)")
    print(f"Speedup:    {speedup:.1f}x")
    print("JAX enables automatic differentiation and JIT compilation")

    return {
        "jax_time": jax_time,
        "numpy_time": numpy_time,
        "speedup": speedup,
        "jax_results": jax_results,
    }


def integration_with_existing_solvers():
    """Demonstrate integration with existing ICARUS solvers."""
    print("\n=== Integration with Existing Solvers ===")

    # Create JAX airfoil
    jax_airfoil = NACA4(M=0.02, P=0.4, XX=0.12, n_points=160)

    print(f"Created JAX airfoil: {jax_airfoil.name}")
    print(f"Max thickness: {jax_airfoil.max_thickness:.4f}")
    print(f"Number of points: {jax_airfoil.n_points}")

    # Demonstrate coordinate extraction for solver integration
    print("\nCoordinate extraction for solver integration:")

    # Upper surface coordinates
    upper_coords = jax_airfoil.upper_surface
    lower_coords = jax_airfoil.lower_surface

    print(f"Upper surface shape: {upper_coords.shape}")
    print(f"Lower surface shape: {lower_coords.shape}")

    # Convert to format suitable for existing solvers
    # Most solvers expect (x, y) coordinate pairs
    upper_xy = jnp.stack([upper_coords[0, :], upper_coords[1, :]], axis=1)
    lower_xy = jnp.stack([lower_coords[0, :], lower_coords[1, :]], axis=1)

    print("Upper coordinates (first 5 points):")
    for i in range(5):
        print(f"  x={upper_xy[i, 0]:.4f}, y={upper_xy[i, 1]:.4f}")

    # Demonstrate repaneling capability
    print("\nRepaneling demonstration:")
    original_points = jax_airfoil.n_points

    # Create versions with different point distributions
    coarse_airfoil = NACA4(M=0.02, P=0.4, XX=0.12, n_points=50)
    fine_airfoil = NACA4(M=0.02, P=0.4, XX=0.12, n_points=300)

    print(f"Original: {original_points} points")
    print(f"Coarse:   {coarse_airfoil.n_points} points")
    print(f"Fine:     {fine_airfoil.n_points} points")

    # Show that geometric properties are preserved
    print("\nGeometric property preservation:")
    print(f"Original max thickness: {jax_airfoil.max_thickness:.6f}")
    print(f"Coarse max thickness:   {coarse_airfoil.max_thickness:.6f}")
    print(f"Fine max thickness:     {fine_airfoil.max_thickness:.6f}")

    return {
        "jax_airfoil": jax_airfoil,
        "coordinate_formats": {
            "upper_surface": upper_coords,
            "lower_surface": lower_coords,
            "upper_xy": upper_xy,
            "lower_xy": lower_xy,
        },
    }


def plot_analysis_results(analysis_results: Dict, sensitivity_results: Dict):
    """Create comprehensive visualization of analysis results."""
    print("\n=== Creating Analysis Visualizations ===")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Lift coefficient comparison
    ax1 = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(analysis_results)))

    for i, (name, data) in enumerate(analysis_results.items()):
        # Plot for middle Reynolds number
        mid_re_idx = len(data["reynolds"]) // 2
        cl_curve = data["cl"][mid_re_idx, :]
        ax1.plot(
            data["angles"],
            cl_curve,
            color=colors[i],
            linewidth=2,
            label=f"{name} (Re={data['reynolds'][mid_re_idx]:.0f})",
        )

    ax1.set_xlabel("Angle of Attack (degrees)")
    ax1.set_ylabel("Lift Coefficient (CL)")
    ax1.set_title("Lift Coefficient Comparison")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Drag polar
    ax2 = axes[0, 1]
    for i, (name, data) in enumerate(analysis_results.items()):
        mid_re_idx = len(data["reynolds"]) // 2
        cl_curve = data["cl"][mid_re_idx, :]
        cd_curve = data["cd"][mid_re_idx, :]
        ax2.plot(
            cd_curve,
            cl_curve,
            color=colors[i],
            linewidth=2,
            marker="o",
            markersize=3,
            label=name,
        )

    ax2.set_xlabel("Drag Coefficient (CD)")
    ax2.set_ylabel("Lift Coefficient (CL)")
    ax2.set_title("Drag Polar Comparison")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: L/D ratio
    ax3 = axes[0, 2]
    for i, (name, data) in enumerate(analysis_results.items()):
        mid_re_idx = len(data["reynolds"]) // 2
        cl_curve = data["cl"][mid_re_idx, :]
        cd_curve = data["cd"][mid_re_idx, :]
        ld_ratio = cl_curve / (cd_curve + 1e-6)
        ax3.plot(data["angles"], ld_ratio, color=colors[i], linewidth=2, label=name)

    ax3.set_xlabel("Angle of Attack (degrees)")
    ax3.set_ylabel("L/D Ratio")
    ax3.set_title("Lift-to-Drag Ratio Comparison")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Reynolds number effects
    ax4 = axes[1, 0]
    # Show effect of Reynolds number on one airfoil
    sample_airfoil = list(analysis_results.keys())[1]  # Second airfoil
    data = analysis_results[sample_airfoil]

    re_colors = plt.cm.viridis(np.linspace(0, 1, len(data["reynolds"])))
    for i, re in enumerate(data["reynolds"]):
        cl_curve = data["cl"][i, :]
        ax4.plot(
            data["angles"],
            cl_curve,
            color=re_colors[i],
            linewidth=2,
            label=f"Re={re:.0f}",
        )

    ax4.set_xlabel("Angle of Attack (degrees)")
    ax4.set_ylabel("Lift Coefficient (CL)")
    ax4.set_title(f"Reynolds Number Effects - {sample_airfoil}")
    ax4.grid(True, alpha=0.3)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Plot 5: Sensitivity analysis
    ax5 = axes[1, 1]
    if sensitivity_results:
        names = list(sensitivity_results.keys())
        sensitivities = ["dLDdM", "dLDdP", "dLDdXX"]
        sens_labels = ["∂(L/D)/∂M", "∂(L/D)/∂P", "∂(L/D)/∂XX"]

        x = np.arange(len(names))
        width = 0.25

        for i, (sens, label) in enumerate(zip(sensitivities, sens_labels)):
            values = [sensitivity_results[name]["gradients"][sens] for name in names]
            ax5.bar(x + i * width, values, width, label=label)

        ax5.set_xlabel("Airfoil")
        ax5.set_ylabel("Sensitivity")
        ax5.set_title("Parameter Sensitivity Analysis")
        ax5.set_xticks(x + width)
        ax5.set_xticklabels(names, rotation=45)
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # Plot 6: Analysis timing comparison
    ax6 = axes[1, 2]
    airfoil_names = list(analysis_results.keys())
    analysis_times = [analysis_results[name]["analysis_time"] for name in airfoil_names]

    bars = ax6.bar(range(len(airfoil_names)), analysis_times, color="skyblue")
    ax6.set_xlabel("Airfoil")
    ax6.set_ylabel("Analysis Time (seconds)")
    ax6.set_title("JAX Analysis Performance")
    ax6.set_xticks(range(len(airfoil_names)))
    ax6.set_xticklabels(airfoil_names, rotation=45)
    ax6.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, time_val in zip(bars, analysis_times):
        height = bar.get_height()
        ax6.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{time_val:.3f}s",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()


def main():
    """Main function demonstrating aerodynamic analysis workflow."""
    print("JAX Airfoil Aerodynamic Analysis Workflow")
    print("=" * 60)

    # Set up analysis conditions
    conditions = setup_analysis_conditions()

    # Create airfoil family
    airfoils = create_airfoil_family_jax()

    # Perform batch analysis
    analysis_results = batch_aerodynamic_analysis(airfoils, conditions)

    # Gradient sensitivity analysis
    sensitivity_results = gradient_sensitivity_analysis(airfoils, conditions)

    # Performance comparison
    performance_results = performance_comparison_study()

    # Integration demonstration
    integration_results = integration_with_existing_solvers()

    # Create visualizations
    plot_analysis_results(analysis_results, sensitivity_results)

    print("\n" + "=" * 60)
    print("Key Integration Capabilities:")
    print("1. Batch processing with automatic vectorization")
    print("2. Gradient-based sensitivity analysis")
    print("3. JIT compilation for performance optimization")
    print("4. Seamless integration with existing ICARUS solvers")
    print("5. Automatic differentiation through analysis pipeline")
    print("6. Efficient handling of parametric studies")

    return {
        "conditions": conditions,
        "airfoils": airfoils,
        "analysis_results": analysis_results,
        "sensitivity_results": sensitivity_results,
        "performance_results": performance_results,
        "integration_results": integration_results,
    }


if __name__ == "__main__":
    main()
