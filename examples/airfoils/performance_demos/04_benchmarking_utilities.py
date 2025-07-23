#!/usr/bin/env python3
"""
JAX Airfoil Benchmarking Utilities

This module provides comprehensive benchmarking utilities for JAX airfoil operations,
including automated performance testing, regression detection, and performance
comparison frameworks.

Key features:
- Automated benchmarking framework with statistical analysis
- Performance regression detection and reporting
- Comparative benchmarking between different implementations
- Benchmarking utilities for custom airfoil operations

Requirements: 3.1, 3.2, 5.1, 5.2
"""

import gc
import json
import os
import statistics
import time
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import psutil
from jax import grad
from jax import jit
from jax import vmap

from ICARUS.airfoils.naca4 import NACA4


@dataclass
class BenchmarkResult:
    """Data class for storing benchmark results."""

    name: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    median_time: float
    n_runs: int
    memory_usage: float
    compilation_time: Optional[float] = None
    accuracy_error: Optional[float] = None
    timestamp: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}


class PerformanceBenchmarker:
    """Comprehensive benchmarking framework for JAX airfoil operations."""

    def __init__(self, warmup_runs: int = 5, benchmark_runs: int = 50):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results_history = []
        self.process = psutil.Process(os.getpid())

    def benchmark_function(
        self,
        func: Callable,
        args: Tuple,
        name: str,
        reference_func: Optional[Callable] = None,
        reference_args: Optional[Tuple] = None,
        measure_compilation: bool = False,
    ) -> BenchmarkResult:
        """
        Benchmark a function with comprehensive metrics.

        Args:
            func: Function to benchmark
            args: Arguments for the function
            name: Name identifier for the benchmark
            reference_func: Reference function for accuracy comparison
            reference_args: Arguments for reference function
            measure_compilation: Whether to measure JIT compilation time

        Returns:
            BenchmarkResult with comprehensive metrics
        """
        # Measure compilation time if requested
        compilation_time = None
        if measure_compilation and hasattr(func, "__wrapped__"):
            # For JIT functions, measure first call time
            gc.collect()
            start_time = time.perf_counter()
            _ = func(*args)
            compilation_time = time.perf_counter() - start_time

        # Warm-up runs
        for _ in range(self.warmup_runs):
            _ = func(*args)

        # Force garbage collection
        gc.collect()
        initial_memory = self.process.memory_info().rss / 1024 / 1024

        # Benchmark runs
        times = []
        for _ in range(self.benchmark_runs):
            gc.collect()
            start_time = time.perf_counter()
            result = func(*args)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        final_memory = self.process.memory_info().rss / 1024 / 1024
        memory_usage = final_memory - initial_memory

        # Calculate accuracy error if reference provided
        accuracy_error = None
        if reference_func is not None:
            ref_args = reference_args or args
            reference_result = reference_func(*ref_args)
            if hasattr(result, "shape") and hasattr(reference_result, "shape"):
                accuracy_error = float(jnp.max(jnp.abs(result - reference_result)))

        # Create benchmark result
        benchmark_result = BenchmarkResult(
            name=name,
            mean_time=statistics.mean(times),
            std_time=statistics.stdev(times) if len(times) > 1 else 0.0,
            min_time=min(times),
            max_time=max(times),
            median_time=statistics.median(times),
            n_runs=self.benchmark_runs,
            memory_usage=memory_usage,
            compilation_time=compilation_time,
            accuracy_error=accuracy_error,
            metadata={
                "args_info": str(type(args[0])) if args else "no_args",
                "result_shape": getattr(result, "shape", None),
                "jax_version": jax.__version__,
            },
        )

        self.results_history.append(benchmark_result)
        return benchmark_result

    def benchmark_suite(
        self,
        functions: Dict[str, Tuple[Callable, Tuple]],
        reference_function: Optional[Tuple[Callable, Tuple]] = None,
    ) -> Dict[str, BenchmarkResult]:
        """
        Benchmark a suite of functions.

        Args:
            functions: Dictionary of {name: (function, args)} pairs
            reference_function: Optional reference (function, args) for accuracy

        Returns:
            Dictionary of benchmark results
        """
        results = {}

        for name, (func, args) in functions.items():
            ref_func, ref_args = (
                reference_function if reference_function else (None, None)
            )

            result = self.benchmark_function(
                func=func,
                args=args,
                name=name,
                reference_func=ref_func,
                reference_args=ref_args,
                measure_compilation="jit" in name.lower(),
            )

            results[name] = result

            print(f"Benchmarked {name}:")
            print(
                f"  Time: {result.mean_time * 1000:.2f} ± {result.std_time * 1000:.2f} ms",
            )
            print(f"  Memory: {result.memory_usage:.2f} MB")
            if result.compilation_time:
                print(f"  Compilation: {result.compilation_time * 1000:.2f} ms")
            if result.accuracy_error is not None:
                print(f"  Accuracy: {result.accuracy_error:.2e}")
            print()

        return results

    def save_results(self, filepath: str):
        """Save benchmark results to file."""
        results_data = [asdict(result) for result in self.results_history]

        with open(filepath, "w") as f:
            json.dump(results_data, f, indent=2)

    def load_results(self, filepath: str):
        """Load benchmark results from file."""
        with open(filepath) as f:
            results_data = json.load(f)

        self.results_history = [BenchmarkResult(**data) for data in results_data]

    def compare_with_baseline(
        self,
        baseline_file: str,
        tolerance: float = 0.1,
    ) -> Dict[str, Dict]:
        """
        Compare current results with baseline performance.

        Args:
            baseline_file: Path to baseline results file
            tolerance: Acceptable performance degradation (10% = 0.1)

        Returns:
            Dictionary with comparison results
        """
        if not Path(baseline_file).exists():
            print(f"Baseline file {baseline_file} not found. Creating new baseline.")
            self.save_results(baseline_file)
            return {}

        # Load baseline
        baseline_benchmarker = PerformanceBenchmarker()
        baseline_benchmarker.load_results(baseline_file)

        # Create lookup for baseline results
        baseline_lookup = {
            result.name: result for result in baseline_benchmarker.results_history
        }

        comparisons = {}
        regressions = []

        for current_result in self.results_history:
            if current_result.name in baseline_lookup:
                baseline_result = baseline_lookup[current_result.name]

                time_ratio = current_result.mean_time / baseline_result.mean_time
                memory_ratio = current_result.memory_usage / max(
                    baseline_result.memory_usage,
                    0.001,
                )

                comparison = {
                    "current_time": current_result.mean_time,
                    "baseline_time": baseline_result.mean_time,
                    "time_ratio": time_ratio,
                    "time_change_percent": (time_ratio - 1) * 100,
                    "current_memory": current_result.memory_usage,
                    "baseline_memory": baseline_result.memory_usage,
                    "memory_ratio": memory_ratio,
                    "memory_change_percent": (memory_ratio - 1) * 100,
                    "is_regression": time_ratio > (1 + tolerance),
                }

                comparisons[current_result.name] = comparison

                if comparison["is_regression"]:
                    regressions.append(current_result.name)

        # Print regression report
        if regressions:
            print("PERFORMANCE REGRESSIONS DETECTED:")
            for name in regressions:
                comp = comparisons[name]
                print(f"  {name}: {comp['time_change_percent']:+.1f}% slower")
        else:
            print("No performance regressions detected.")

        return comparisons


class AirfoilBenchmarkSuite:
    """Specialized benchmark suite for airfoil operations."""

    def __init__(self):
        self.benchmarker = PerformanceBenchmarker()
        self.test_airfoil = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

    def benchmark_basic_operations(
        self,
        n_points: int = 1000,
    ) -> Dict[str, BenchmarkResult]:
        """Benchmark basic airfoil operations."""
        print(f"Benchmarking basic operations with {n_points} points...")

        x_points = jnp.linspace(0.001, 1.0, n_points)

        # Create NumPy reference functions
        def numpy_thickness(x, xx=0.12):
            a0, a1, a2, a3, a4 = 0.2969, -0.126, -0.3516, 0.2843, -0.1036
            return (xx / 0.2) * (
                a0 * np.sqrt(x) + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4
            )

        functions = {
            "thickness_distribution": (
                self.test_airfoil.thickness_distribution,
                (x_points,),
            ),
            "thickness_distribution_jit": (
                jit(self.test_airfoil.thickness_distribution),
                (x_points,),
            ),
            "camber_line": (self.test_airfoil.camber_line, (x_points,)),
            "camber_line_jit": (jit(self.test_airfoil.camber_line), (x_points,)),
            "y_upper": (self.test_airfoil.y_upper, (x_points,)),
            "y_upper_jit": (jit(self.test_airfoil.y_upper), (x_points,)),
            "y_lower": (self.test_airfoil.y_lower, (x_points,)),
            "y_lower_jit": (jit(self.test_airfoil.y_lower), (x_points,)),
        }

        reference = (numpy_thickness, (np.array(x_points),))

        return self.benchmarker.benchmark_suite(functions, reference)

    def benchmark_batch_operations(
        self,
        batch_sizes: List[int] = None,
    ) -> Dict[str, BenchmarkResult]:
        """Benchmark batch processing operations."""
        if batch_sizes is None:
            batch_sizes = [1, 10, 50, 100]

        print("Benchmarking batch operations...")

        results = {}
        n_points = 500
        x_eval = jnp.linspace(0.001, 1.0, n_points)

        for batch_size in batch_sizes:
            print(f"  Batch size: {batch_size}")

            # Create batch parameters
            M_values = jnp.array(np.random.uniform(0.01, 0.05, batch_size))
            P_values = jnp.array(np.random.uniform(0.3, 0.5, batch_size))
            XX_values = jnp.array(np.random.uniform(0.08, 0.15, batch_size))

            # Individual processing
            def individual_processing(M_vals, P_vals, XX_vals, x):
                results = []
                for i in range(len(M_vals)):
                    naca = NACA4(
                        M=float(M_vals[i]),
                        P=float(P_vals[i]),
                        XX=float(XX_vals[i]),
                        n_points=100,
                    )
                    result = naca.y_upper(x)
                    results.append(result)
                return jnp.stack(results)

            # Vectorized processing
            def vectorized_processing(M_vals, P_vals, XX_vals, x):
                def single_upper_surface(params):
                    m, p, xx = params
                    naca = NACA4(M=m, P=p, XX=xx, n_points=100)
                    return naca.y_upper(x)

                batch_params = jnp.stack([M_vals, P_vals, XX_vals], axis=1)
                return vmap(single_upper_surface)(batch_params)

            # JIT vectorized processing
            vectorized_jit = jit(vectorized_processing)

            functions = {
                f"individual_batch_{batch_size}": (
                    individual_processing,
                    (M_values, P_values, XX_values, x_eval),
                ),
                f"vectorized_batch_{batch_size}": (
                    vectorized_processing,
                    (M_values, P_values, XX_values, x_eval),
                ),
                f"vectorized_jit_batch_{batch_size}": (
                    vectorized_jit,
                    (M_values, P_values, XX_values, x_eval),
                ),
            }

            batch_results = self.benchmarker.benchmark_suite(functions)
            results.update(batch_results)

        return results

    def benchmark_gradient_operations(self) -> Dict[str, BenchmarkResult]:
        """Benchmark gradient computation operations."""
        print("Benchmarking gradient operations...")

        def airfoil_objective(params):
            m, p, xx = params
            naca = NACA4(M=m, P=p, XX=xx, n_points=100)
            x_points = jnp.linspace(0.001, 1.0, 50)
            y_upper = naca.y_upper(x_points)
            return jnp.sum(y_upper**2)

        def numerical_gradient(func, params, eps=1e-6):
            grad = jnp.zeros_like(params)
            for i in range(len(params)):
                params_plus = params.at[i].add(eps)
                params_minus = params.at[i].add(-eps)
                grad = grad.at[i].set(
                    (func(params_plus) - func(params_minus)) / (2 * eps),
                )
            return grad

        params = jnp.array([0.02, 0.4, 0.12])

        functions = {
            "function_evaluation": (airfoil_objective, (params,)),
            "gradient_ad": (grad(airfoil_objective), (params,)),
            "gradient_ad_jit": (jit(grad(airfoil_objective)), (params,)),
            "gradient_numerical": (numerical_gradient, (airfoil_objective, params)),
        }

        return self.benchmarker.benchmark_suite(functions)

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite."""
        print("RUNNING COMPREHENSIVE AIRFOIL BENCHMARK SUITE")
        print("=" * 60)

        results = {
            "basic_operations": self.benchmark_basic_operations(),
            "batch_operations": self.benchmark_batch_operations(),
            "gradient_operations": self.benchmark_gradient_operations(),
        }

        return results


class PerformanceRegressor:
    """Performance regression detection and analysis."""

    def __init__(self, baseline_dir: str = "benchmarks/baselines"):
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

    def create_baseline(self, name: str):
        """Create a new performance baseline."""
        suite = AirfoilBenchmarkSuite()
        results = suite.run_comprehensive_benchmark()

        baseline_file = self.baseline_dir / f"{name}_baseline.json"
        suite.benchmarker.save_results(str(baseline_file))

        print(f"Baseline '{name}' created at {baseline_file}")
        return results

    def check_regression(self, baseline_name: str, tolerance: float = 0.1):
        """Check for performance regressions against baseline."""
        baseline_file = self.baseline_dir / f"{baseline_name}_baseline.json"

        if not baseline_file.exists():
            print(f"Baseline '{baseline_name}' not found. Creating new baseline.")
            return self.create_baseline(baseline_name)

        # Run current benchmarks
        suite = AirfoilBenchmarkSuite()
        current_results = suite.run_comprehensive_benchmark()

        # Compare with baseline
        comparisons = suite.benchmarker.compare_with_baseline(
            str(baseline_file),
            tolerance,
        )

        return {"current_results": current_results, "comparisons": comparisons}


def create_benchmark_report(
    results: Dict[str, Any],
    output_file: str = "benchmark_report.html",
):
    """Create comprehensive benchmark report."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>JAX Airfoil Performance Benchmark Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .benchmark-table {{ border-collapse: collapse; width: 100%; }}
            .benchmark-table th, .benchmark-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .benchmark-table th {{ background-color: #f2f2f2; }}
            .performance-good {{ color: green; }}
            .performance-bad {{ color: red; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>JAX Airfoil Performance Benchmark Report</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
    """

    # Add basic operations results
    if "basic_operations" in results:
        html_content += """
        <div class="section">
            <h2>Basic Operations Performance</h2>
            <table class="benchmark-table">
                <tr>
                    <th>Operation</th>
                    <th>Mean Time (ms)</th>
                    <th>Std Dev (ms)</th>
                    <th>Memory (MB)</th>
                    <th>Compilation Time (ms)</th>
                </tr>
        """

        for name, result in results["basic_operations"].items():
            compilation_time = (
                f"{result.compilation_time * 1000:.2f}"
                if result.compilation_time
                else "N/A"
            )
            html_content += f"""
                <tr>
                    <td>{name}</td>
                    <td>{result.mean_time * 1000:.2f}</td>
                    <td>{result.std_time * 1000:.2f}</td>
                    <td>{result.memory_usage:.2f}</td>
                    <td>{compilation_time}</td>
                </tr>
            """

        html_content += "</table></div>"

    html_content += """
    </body>
    </html>
    """

    with open(output_file, "w") as f:
        f.write(html_content)

    print(f"Benchmark report saved to {output_file}")


def main():
    """Main demonstration of benchmarking utilities."""
    print("JAX AIRFOIL BENCHMARKING UTILITIES DEMONSTRATION")
    print("=" * 60)

    # Create benchmark suite
    suite = AirfoilBenchmarkSuite()

    # Run comprehensive benchmarks
    results = suite.run_comprehensive_benchmark()

    # Save results
    suite.benchmarker.save_results("benchmark_results.json")

    # Create performance regressor
    regressor = PerformanceRegressor()

    # Create baseline (or check regression if baseline exists)
    regression_results = regressor.check_regression("main", tolerance=0.15)

    # Create benchmark report
    create_benchmark_report(results)

    print("\n" + "=" * 60)
    print("BENCHMARKING DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("Generated files:")
    print("- benchmark_results.json: Raw benchmark data")
    print("- benchmark_report.html: Formatted benchmark report")
    print("- benchmarks/baselines/: Performance baselines for regression testing")


if __name__ == "__main__":
    main()
