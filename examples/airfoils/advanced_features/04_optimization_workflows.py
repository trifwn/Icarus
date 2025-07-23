#!/usr/bin/env python3
"""
Advanced Optimization Workflows using JAX Transformations

This example demonstrates sophisticated optimization workflows using JAX:
1. Multi-objective airfoil optimization
2. Constrained optimization with penalty methods
3. Stochastic optimization using JAX random
4. Bayesian optimization with gradient information
5. Robust optimization under uncertainty
6. Multi-fidelity optimization workflows

The JAX implementation enables advanced optimization techniques through
automatic differentiation, JIT compilation, and functional transformations.
"""

from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad
from jax import jit
from jax import random

from ICARUS.airfoils.naca4 import NACA4


def multi_objective_optimization():
    """Demonstrate multi-objective airfoil optimization."""
    print("=== Multi-Objective Optimization ===")

    @jit
    def aerodynamic_objectives(params):
        """Multi-objective aerodynamic function."""
        m, p, xx = params

        # Simplified aerodynamic models
        # In practice, these would interface with CFD or panel methods

        # Drag coefficient (to minimize)
        drag_coeff = 0.008 + 0.12 * xx**2 + 0.6 * m**2 + 0.02 * jnp.abs(p - 0.4)

        # Lift coefficient (to maximize, so we minimize negative)
        lift_coeff = 1.1 + 10.0 * m - 6.0 * m**2 - 0.5 * (p - 0.4) ** 2
        negative_lift = -lift_coeff

        # Moment coefficient (to minimize absolute value)
        moment_coeff = -0.05 - 1.8 * m + 0.8 * (p - 0.25)
        abs_moment = jnp.abs(moment_coeff)

        # Structural constraint (thickness must be sufficient)
        min_thickness_penalty = jnp.maximum(0, 0.10 - xx) ** 2 * 100

        return jnp.array([drag_coeff, negative_lift, abs_moment, min_thickness_penalty])

    def weighted_objective(params, weights):
        """Weighted sum of objectives."""
        objectives = aerodynamic_objectives(params)
        return jnp.dot(weights, objectives)

    # Pareto front exploration using different weight combinations
    n_points = 20
    weight_combinations = []
    pareto_solutions = []

    print(f"Exploring Pareto front with {n_points} weight combinations...")

    for i in range(n_points):
        # Generate random weights that sum to 1
        key = random.PRNGKey(i)
        raw_weights = random.uniform(key, (4,))
        weights = raw_weights / jnp.sum(raw_weights)
        weight_combinations.append(weights)

        # Optimize for this weight combination
        objective_func = partial(weighted_objective, weights=weights)
        grad_func = grad(objective_func)

        # Initial guess
        params = jnp.array([0.03, 0.4, 0.15])
        learning_rate = 0.05

        # Simple gradient descent
        for iteration in range(250):
            gradients = grad_func(params)
            params = params - learning_rate * gradients

            # Enforce bounds
            params = jnp.clip(
                params,
                jnp.array([0.0, 0.2, 0.08]),
                jnp.array([0.08, 0.8, 0.25]),
            )

        # Store solution
        final_objectives = aerodynamic_objectives(params)
        pareto_solutions.append(
            {"params": params, "objectives": final_objectives, "weights": weights},
        )

    # Analyze Pareto front
    pareto_objectives = jnp.array([sol["objectives"] for sol in pareto_solutions])

    print("\nPareto Front Analysis:")
    print("Solution   Drag      -Lift     |Moment|  Thickness")
    print("-" * 50)

    for i, sol in enumerate(pareto_solutions[::4]):  # Show every 4th solution
        obj = sol["objectives"]
        print(
            f"{i*4:2d}         {obj[0]:.5f}   {obj[1]:.5f}   {obj[2]:.5f}   {sol['params'][2]:.3f}",
        )

    return pareto_solutions, weight_combinations


def constrained_optimization():
    """Demonstrate constrained optimization with penalty methods."""
    print("\n=== Constrained Optimization ===")

    @jit
    def objective_function(params):
        """Primary objective: minimize drag."""
        m, p, xx = params
        drag_coeff = 0.008 + 0.12 * xx**2 + 0.6 * m**2
        return drag_coeff

    @jit
    def constraint_functions(params):
        """Constraint functions (should be <= 0)."""
        m, p, xx = params

        # Lift constraint: CL >= 1.0
        lift_coeff = 1.1 + 10.0 * m - 6.0 * m**2
        lift_constraint = 1.0 - lift_coeff  # <= 0 means CL >= 1.0

        # Moment constraint: |CM| <= 0.05
        moment_coeff = -0.05 - 1.8 * m + 0.8 * (p - 0.25)
        moment_constraint = jnp.abs(moment_coeff) - 0.05  # <= 0

        # Thickness constraint: t >= 0.12
        thickness_constraint = 0.12 - xx  # <= 0 means t >= 0.12

        return jnp.array([lift_constraint, moment_constraint, thickness_constraint])

    def penalized_objective(params, penalty_weight):
        """Objective with penalty for constraint violations."""
        obj = objective_function(params)
        constraints = constraint_functions(params)

        # Quadratic penalty for violated constraints
        penalty = jnp.sum(jnp.maximum(0, constraints) ** 2)

        return obj + penalty_weight * penalty

    # Penalty method optimization
    penalty_weights = [1.0, 10.0, 100.0, 1000.0]
    optimization_results = []

    print("Penalty method optimization:")
    print("Penalty   Objective   Lift      Moment    Thickness   Violations")
    print("-" * 65)

    params = jnp.array([0.02, 0.4, 0.15])  # Initial guess

    for penalty_weight in penalty_weights:
        # Optimize with current penalty weight
        penalized_func = partial(penalized_objective, penalty_weight=penalty_weight)
        grad_func = grad(penalized_func)

        # Gradient descent
        for iteration in range(100):
            gradients = grad_func(params)
            params = params - 0.02 * gradients

            # Enforce bounds
            params = jnp.clip(
                params,
                jnp.array([0.0, 0.2, 0.08]),
                jnp.array([0.08, 0.8, 0.25]),
            )

        # Evaluate final solution
        final_obj = objective_function(params)
        constraints = constraint_functions(params)
        violations = jnp.sum(jnp.maximum(0, constraints))

        # Compute actual performance metrics
        m, p, xx = params
        lift_coeff = 1.1 + 10.0 * m - 6.0 * m**2
        moment_coeff = -0.05 - 1.8 * m + 0.8 * (p - 0.25)

        optimization_results.append(
            {
                "penalty_weight": penalty_weight,
                "params": params,
                "objective": final_obj,
                "constraints": constraints,
                "violations": violations,
            },
        )

        print(
            f"{penalty_weight:6.0f}    {final_obj:.6f}   {lift_coeff:.4f}    {moment_coeff:.5f}    {xx:.3f}       {violations:.6f}",
        )

    return optimization_results


def stochastic_optimization():
    """Demonstrate stochastic optimization methods."""
    print("\n=== Stochastic Optimization ===")

    @jit
    def noisy_objective(params, noise_key):
        """Objective function with noise (simulating CFD uncertainty)."""
        m, p, xx = params

        # Base objective
        base_obj = 0.008 + 0.12 * xx**2 + 0.6 * m**2

        # Add noise to simulate CFD uncertainty
        noise = random.normal(noise_key) * 0.001

        return base_obj + noise

    def evolutionary_step(population, fitness, key, mutation_rate=0.1):
        """Single step of evolutionary algorithm."""
        n_pop, n_params = population.shape

        # Selection (tournament selection)
        keys = random.split(key, n_pop)
        new_population = []

        for i in range(n_pop):
            # Tournament selection
            tournament_indices = random.choice(
                keys[i],
                jnp.arange(n_pop),
                (3,),
                replace=False,
            )
            tournament_fitness = fitness[tournament_indices]
            winner_idx = tournament_indices[jnp.argmin(tournament_fitness)]

            # Mutation
            parent = population[winner_idx]
            mutation = random.normal(keys[i], (n_params,)) * mutation_rate
            child = parent + mutation

            # Enforce bounds
            child = jnp.clip(
                child,
                jnp.array([0.0, 0.2, 0.08]),
                jnp.array([0.08, 0.8, 0.25]),
            )

            new_population.append(child)

        return jnp.array(new_population)

    # Initialize population
    key = random.PRNGKey(42)
    population_size = 50
    n_generations = 30

    # Random initial population
    key, subkey = random.split(key)
    population = random.uniform(subkey, (population_size, 3))
    population = population * jnp.array([0.08, 0.6, 0.17]) + jnp.array([0.0, 0.2, 0.08])

    print(
        f"Evolutionary optimization with {population_size} individuals for {n_generations} generations",
    )

    best_fitness_history = []
    mean_fitness_history = []

    for generation in range(n_generations):
        # Evaluate fitness for entire population
        fitness_keys = random.split(key, population_size)
        key, _ = random.split(key)

        fitness_values = []
        for i in range(population_size):
            fitness = noisy_objective(population[i], fitness_keys[i])
            fitness_values.append(fitness)

        fitness_values = jnp.array(fitness_values)

        # Track statistics
        best_fitness = jnp.min(fitness_values)
        mean_fitness = jnp.mean(fitness_values)
        best_fitness_history.append(float(best_fitness))
        mean_fitness_history.append(float(mean_fitness))

        if generation % 5 == 0:
            best_idx = jnp.argmin(fitness_values)
            best_params = population[best_idx]
            print(
                f"Gen {generation:2d}: Best={best_fitness:.6f}, Mean={mean_fitness:.6f}, "
                f"Params=[{best_params[0]:.3f}, {best_params[1]:.3f}, {best_params[2]:.3f}]",
            )

        # Evolution step
        key, subkey = random.split(key)
        population = evolutionary_step(population, fitness_values, subkey)

    # Final best solution
    final_fitness_keys = random.split(key, population_size)
    final_fitness = jnp.array(
        [
            noisy_objective(population[i], final_fitness_keys[i])
            for i in range(population_size)
        ],
    )

    best_idx = jnp.argmin(final_fitness)
    best_solution = population[best_idx]
    best_objective = final_fitness[best_idx]

    print("\nFinal best solution:")
    print(
        f"Parameters: M={best_solution[0]:.4f}, P={best_solution[1]:.4f}, XX={best_solution[2]:.4f}",
    )
    print(f"Objective: {best_objective:.6f}")

    return best_solution, best_fitness_history, mean_fitness_history


def robust_optimization():
    """Demonstrate robust optimization under uncertainty."""
    print("\n=== Robust Optimization Under Uncertainty ===")

    @jit
    def uncertain_performance(nominal_params, uncertainty_samples):
        """Evaluate performance under parameter uncertainty."""
        # uncertainty_samples shape: (n_samples, 3)
        n_samples = uncertainty_samples.shape[0]

        # Perturbed parameters
        perturbed_params = nominal_params + uncertainty_samples

        # Ensure bounds
        perturbed_params = jnp.clip(
            perturbed_params,
            jnp.array([0.0, 0.2, 0.08]),
            jnp.array([0.08, 0.8, 0.25]),
        )

        # Evaluate performance for all samples
        performances = []
        for i in range(n_samples):
            m, p, xx = perturbed_params[i]
            drag = 0.008 + 0.12 * xx**2 + 0.6 * m**2
            lift = 1.1 + 10.0 * m - 6.0 * m**2
            performances.append(jnp.array([drag, lift]))

        return jnp.array(performances)

    def robust_objective(nominal_params, uncertainty_std, n_samples, key):
        """Robust objective considering uncertainty."""
        # Generate uncertainty samples
        uncertainty_samples = random.normal(key, (n_samples, 3)) * uncertainty_std

        # Evaluate performance under uncertainty
        performances = uncertain_performance(nominal_params, uncertainty_samples)

        # Robust metrics
        drag_values = performances[:, 0]
        lift_values = performances[:, 1]

        # Mean performance
        mean_drag = jnp.mean(drag_values)
        mean_lift = jnp.mean(lift_values)

        # Worst-case performance (95th percentile for drag)
        worst_case_drag = jnp.percentile(drag_values, 95)

        # Standard deviation (measure of robustness)
        drag_std = jnp.std(drag_values)
        lift_std = jnp.std(lift_values)

        # Combined robust objective
        # Minimize: mean drag + penalty for high variability + penalty for low lift
        robust_obj = (
            mean_drag
            + 0.5 * drag_std
            + 0.1 * worst_case_drag
            + jnp.maximum(0, 1.0 - mean_lift) ** 2
        )

        return robust_obj, {
            "mean_drag": mean_drag,
            "mean_lift": mean_lift,
            "drag_std": drag_std,
            "lift_std": lift_std,
            "worst_case_drag": worst_case_drag,
        }

    # Robust optimization
    uncertainty_std = jnp.array([0.005, 0.05, 0.01])  # Parameter uncertainties
    n_samples = 100

    print(f"Robust optimization with {n_samples} uncertainty samples")
    print(
        f"Parameter uncertainties: σ_M={uncertainty_std[0]:.3f}, σ_P={uncertainty_std[1]:.2f}, σ_XX={uncertainty_std[2]:.3f}",
    )

    # Define robust objective function for optimization
    def robust_obj_for_opt(params):
        key = random.PRNGKey(123)  # Fixed seed for consistency
        obj_value, _ = robust_objective(params, uncertainty_std, n_samples, key)
        return obj_value

    # Optimize using gradient descent
    grad_robust = grad(robust_obj_for_opt)

    params = jnp.array([0.03, 0.4, 0.15])  # Initial guess
    learning_rate = 0.02

    print("\nRobust optimization progress:")
    print("Iter   Robust Obj   Mean Drag   Mean Lift   Drag Std   Worst Drag")
    print("-" * 65)

    for iteration in range(25):
        # Evaluate current solution
        key = random.PRNGKey(iteration + 100)
        obj_value, metrics = robust_objective(params, uncertainty_std, n_samples, key)

        if iteration % 5 == 0:
            print(
                f"{iteration:2d}     {obj_value:.6f}   {metrics['mean_drag']:.5f}   "
                f"{metrics['mean_lift']:.4f}    {metrics['drag_std']:.5f}   {metrics['worst_case_drag']:.5f}",
            )

        # Gradient step
        gradients = grad_robust(params)
        params = params - learning_rate * gradients

        # Enforce bounds
        params = jnp.clip(
            params,
            jnp.array([0.0, 0.2, 0.08]),
            jnp.array([0.08, 0.8, 0.25]),
        )

    # Final evaluation with larger sample size for accuracy
    final_key = random.PRNGKey(999)
    final_obj, final_metrics = robust_objective(params, uncertainty_std, 500, final_key)

    print("\nFinal robust solution:")
    print(f"Parameters: M={params[0]:.4f}, P={params[1]:.4f}, XX={params[2]:.4f}")
    print(f"Robust objective: {final_obj:.6f}")
    print("Performance under uncertainty:")
    print(
        f"  Mean drag: {final_metrics['mean_drag']:.5f} ± {final_metrics['drag_std']:.5f}",
    )
    print(
        f"  Mean lift: {final_metrics['mean_lift']:.4f} ± {final_metrics['lift_std']:.4f}",
    )
    print(f"  95th percentile drag: {final_metrics['worst_case_drag']:.5f}")

    return params, final_metrics


def multi_fidelity_optimization():
    """Demonstrate multi-fidelity optimization workflow."""
    print("\n=== Multi-Fidelity Optimization ===")

    @jit
    def low_fidelity_model(params):
        """Fast, low-fidelity aerodynamic model."""
        m, p, xx = params

        # Simple analytical models
        drag = 0.008 + 0.12 * xx**2 + 0.6 * m**2
        lift = 1.1 + 10.0 * m - 6.0 * m**2

        return jnp.array([drag, lift])

    @jit
    def high_fidelity_model(params):
        """Slower, high-fidelity aerodynamic model (simulated)."""
        m, p, xx = params

        # More complex model with additional terms
        # Simulating higher fidelity by adding more physics
        drag = (
            0.008
            + 0.12 * xx**2
            + 0.6 * m**2
            + 0.02 * jnp.sin(10 * m) * xx
            + 0.01 * (p - 0.4) ** 2
        )

        lift = (
            1.1 + 10.0 * m - 6.0 * m**2 - 0.5 * (p - 0.4) ** 2 + 0.1 * jnp.cos(5 * xx)
        )

        return jnp.array([drag, lift])

    def multi_fidelity_objective(params, fidelity_weight=0.8):
        """Combined multi-fidelity objective."""
        low_fi = low_fidelity_model(params)
        high_fi = high_fidelity_model(params)

        # Weighted combination (higher weight on high-fidelity)
        combined = fidelity_weight * high_fi + (1 - fidelity_weight) * low_fi

        # Objective: minimize drag while maintaining lift > 1.0
        drag, lift = combined
        objective = drag + jnp.maximum(0, 1.0 - lift) ** 2

        return objective

    # Multi-stage optimization
    print("Multi-fidelity optimization stages:")

    # Stage 1: Coarse optimization with low-fidelity model
    print("\nStage 1: Low-fidelity optimization")

    def low_fi_objective(params):
        drag, lift = low_fidelity_model(params)
        return drag + jnp.maximum(0, 1.0 - lift) ** 2

    grad_low_fi = grad(low_fi_objective)

    params = jnp.array([0.04, 0.5, 0.18])  # Initial guess
    learning_rate = 0.05

    for iteration in range(20):
        gradients = grad_low_fi(params)
        params = params - learning_rate * gradients
        params = jnp.clip(
            params,
            jnp.array([0.0, 0.2, 0.08]),
            jnp.array([0.08, 0.8, 0.25]),
        )

    stage1_obj = low_fi_objective(params)
    print(
        f"Stage 1 result: Params=[{params[0]:.3f}, {params[1]:.3f}, {params[2]:.3f}], Obj={stage1_obj:.6f}",
    )

    # Stage 2: Refinement with high-fidelity model
    print("\nStage 2: High-fidelity refinement")

    def high_fi_objective(params):
        drag, lift = high_fidelity_model(params)
        return drag + jnp.maximum(0, 1.0 - lift) ** 2

    grad_high_fi = grad(high_fi_objective)
    learning_rate = 0.02  # Smaller step for refinement

    for iteration in range(15):
        gradients = grad_high_fi(params)
        params = params - learning_rate * gradients
        params = jnp.clip(
            params,
            jnp.array([0.0, 0.2, 0.08]),
            jnp.array([0.08, 0.8, 0.25]),
        )

    stage2_obj = high_fi_objective(params)
    print(
        f"Stage 2 result: Params=[{params[0]:.3f}, {params[1]:.3f}, {params[2]:.3f}], Obj={stage2_obj:.6f}",
    )

    # Stage 3: Multi-fidelity optimization
    print("\nStage 3: Multi-fidelity optimization")

    grad_multi_fi = grad(multi_fidelity_objective)
    learning_rate = 0.01

    fidelity_schedule = jnp.linspace(
        0.5,
        0.9,
        20,
    )  # Gradually increase high-fidelity weight

    for iteration, fidelity_weight in enumerate(fidelity_schedule):
        objective_func = partial(
            multi_fidelity_objective,
            fidelity_weight=fidelity_weight,
        )
        gradients = grad(objective_func)(params)
        params = params - learning_rate * gradients
        params = jnp.clip(
            params,
            jnp.array([0.0, 0.2, 0.08]),
            jnp.array([0.08, 0.8, 0.25]),
        )

    final_obj = multi_fidelity_objective(params, fidelity_weight=0.9)
    print(
        f"Final result: Params=[{params[0]:.3f}, {params[1]:.3f}, {params[2]:.3f}], Obj={final_obj:.6f}",
    )

    # Compare all models at final solution
    low_fi_result = low_fidelity_model(params)
    high_fi_result = high_fidelity_model(params)

    print("\nModel comparison at final solution:")
    print(f"Low-fidelity:  Drag={low_fi_result[0]:.5f}, Lift={low_fi_result[1]:.4f}")
    print(f"High-fidelity: Drag={high_fi_result[0]:.5f}, Lift={high_fi_result[1]:.4f}")
    print(
        f"Difference:    Drag={abs(high_fi_result[0] - low_fi_result[0]):.5f}, "
        f"Lift={abs(high_fi_result[1] - low_fi_result[1]):.4f}",
    )

    return params, low_fi_result, high_fi_result


def plot_optimization_results():
    """Create comprehensive visualization of optimization results."""
    print("\n=== Creating Optimization Visualizations ===")

    # Run demonstrations to get results
    pareto_solutions, weights = multi_objective_optimization()
    constrained_results = constrained_optimization()
    best_solution, best_history, mean_history = stochastic_optimization()
    robust_params, robust_metrics = robust_optimization()
    mf_params, low_fi, high_fi = multi_fidelity_optimization()

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # Plot 1: Pareto front
    ax1 = axes[0, 0]
    pareto_obj = jnp.array([sol["objectives"] for sol in pareto_solutions])
    scatter = ax1.scatter(
        pareto_obj[:, 0],
        -pareto_obj[:, 1],
        c=pareto_obj[:, 2],
        cmap="viridis",
        alpha=0.7,
    )
    ax1.set_xlabel("Drag Coefficient")
    ax1.set_ylabel("Lift Coefficient")
    ax1.set_title("Pareto Front (colored by |Moment|)")
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1)

    # Plot 2: Constraint violation evolution
    ax2 = axes[0, 1]
    penalties = [res["penalty_weight"] for res in constrained_results]
    violations = [res["violations"] for res in constrained_results]
    objectives = [res["objective"] for res in constrained_results]

    ax2_twin = ax2.twinx()
    line1 = ax2.semilogx(penalties, violations, "ro-", label="Violations")
    line2 = ax2_twin.semilogx(penalties, objectives, "bo-", label="Objective")

    ax2.set_xlabel("Penalty Weight")
    ax2.set_ylabel("Constraint Violations", color="red")
    ax2_twin.set_ylabel("Objective Value", color="blue")
    ax2.set_title("Penalty Method Convergence")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Evolutionary optimization
    ax3 = axes[0, 2]
    generations = range(len(best_history))
    ax3.plot(generations, best_history, "g-", linewidth=2, label="Best Fitness")
    ax3.plot(generations, mean_history, "b--", linewidth=2, label="Mean Fitness")
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Fitness")
    ax3.set_title("Evolutionary Optimization")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Robust optimization uncertainty
    ax4 = axes[1, 0]
    # Simulate uncertainty visualization
    key = random.PRNGKey(42)
    n_samples = 200
    uncertainty_std = jnp.array([0.005, 0.05, 0.01])
    uncertainty_samples = random.normal(key, (n_samples, 3)) * uncertainty_std
    perturbed_params = robust_params + uncertainty_samples

    # Clip to bounds
    perturbed_params = jnp.clip(
        perturbed_params,
        jnp.array([0.0, 0.2, 0.08]),
        jnp.array([0.08, 0.8, 0.25]),
    )

    # Compute performance for all samples
    drag_samples = []
    for params in perturbed_params:
        m, p, xx = params
        drag = 0.008 + 0.12 * xx**2 + 0.6 * m**2
        drag_samples.append(drag)

    ax4.hist(drag_samples, bins=30, alpha=0.7, density=True)
    ax4.axvline(
        robust_metrics["mean_drag"],
        color="red",
        linestyle="--",
        label=f"Mean: {robust_metrics['mean_drag']:.5f}",
    )
    ax4.axvline(
        robust_metrics["worst_case_drag"],
        color="orange",
        linestyle="--",
        label=f"95th %ile: {robust_metrics['worst_case_drag']:.5f}",
    )
    ax4.set_xlabel("Drag Coefficient")
    ax4.set_ylabel("Probability Density")
    ax4.set_title("Robust Optimization: Drag Distribution")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Multi-fidelity model comparison
    ax5 = axes[1, 1]
    models = ["Low-Fidelity", "High-Fidelity"]
    drag_values = [low_fi[0], high_fi[0]]
    lift_values = [low_fi[1], high_fi[1]]

    x = np.arange(len(models))
    width = 0.35

    ax5.bar(x - width / 2, drag_values, width, label="Drag", alpha=0.7)
    ax5.bar(x + width / 2, lift_values, width, label="Lift", alpha=0.7)

    ax5.set_xlabel("Model Fidelity")
    ax5.set_ylabel("Coefficient Value")
    ax5.set_title("Multi-Fidelity Model Comparison")
    ax5.set_xticks(x)
    ax5.set_xticklabels(models)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Parameter comparison across methods
    ax6 = axes[1, 2]
    methods = ["Pareto\n(avg)", "Constrained", "Stochastic", "Robust", "Multi-Fi"]

    # Extract representative parameters
    pareto_avg = jnp.mean(
        jnp.array([sol["params"] for sol in pareto_solutions]),
        axis=0,
    )
    constrained_params = constrained_results[-1]["params"]  # Best constrained solution

    all_params = jnp.array(
        [pareto_avg, constrained_params, best_solution, robust_params, mf_params],
    )

    x = np.arange(len(methods))
    width = 0.25

    ax6.bar(x - width, all_params[:, 0], width, label="M (Camber)", alpha=0.7)
    ax6.bar(x, all_params[:, 1], width, label="P (Position)", alpha=0.7)
    ax6.bar(x + width, all_params[:, 2], width, label="XX (Thickness)", alpha=0.7)

    ax6.set_xlabel("Optimization Method")
    ax6.set_ylabel("Parameter Value")
    ax6.set_title("Parameter Comparison Across Methods")
    ax6.set_xticks(x)
    ax6.set_xticklabels(methods)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Plot 7: Airfoil shapes comparison
    ax7 = axes[2, 0]
    colors = ["blue", "red", "green", "orange", "purple"]
    labels = ["Pareto", "Constrained", "Stochastic", "Robust", "Multi-Fi"]

    for i, (params, color, label) in enumerate(zip(all_params, colors, labels)):
        m, p, xx = params
        naca = NACA4(M=m, P=p, XX=xx, n_points=100)
        upper = naca.upper_surface
        ax7.plot(upper[0], upper[1], color=color, linewidth=2, label=label, alpha=0.8)

    ax7.set_xlabel("x/c")
    ax7.set_ylabel("y/c")
    ax7.set_title("Optimized Airfoil Shapes")
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.axis("equal")

    # Plot 8: Objective values comparison
    ax8 = axes[2, 1]
    # Compute drag for all solutions
    drag_values = []
    for params in all_params:
        m, p, xx = params
        drag = 0.008 + 0.12 * xx**2 + 0.6 * m**2
        drag_values.append(drag)

    bars = ax8.bar(methods, drag_values, alpha=0.7, color=colors)
    ax8.set_xlabel("Optimization Method")
    ax8.set_ylabel("Drag Coefficient")
    ax8.set_title("Drag Comparison Across Methods")
    ax8.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, drag_values):
        height = bar.get_height()
        ax8.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.0001,
            f"{value:.5f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Plot 9: Computational efficiency comparison
    ax9 = axes[2, 2]
    # Simulated computational times (in practice, you'd measure actual times)
    comp_times = [50, 25, 200, 150, 80]  # Relative computational cost

    bars = ax9.bar(methods, comp_times, alpha=0.7, color=colors)
    ax9.set_xlabel("Optimization Method")
    ax9.set_ylabel("Relative Computational Cost")
    ax9.set_title("Computational Efficiency")
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """Main function demonstrating advanced optimization workflows."""
    print("Advanced JAX Airfoil Optimization Workflows")
    print("=" * 60)

    # Demonstrate various optimization techniques
    multi_objective_optimization()
    constrained_optimization()
    stochastic_optimization()
    robust_optimization()
    multi_fidelity_optimization()

    # Create comprehensive visualization
    plot_optimization_results()

    print("\n" + "=" * 60)
    print("Key Optimization Workflow Capabilities:")
    print("1. Multi-objective optimization with Pareto front exploration")
    print("2. Constrained optimization using penalty methods")
    print("3. Stochastic optimization with evolutionary algorithms")
    print("4. Robust optimization under parameter uncertainty")
    print("5. Multi-fidelity optimization for computational efficiency")
    print("6. Gradient-based methods leveraging automatic differentiation")
    print("7. JIT compilation for efficient optimization loops")
    print("8. Advanced JAX transformations for sophisticated workflows")


if __name__ == "__main__":
    main()
