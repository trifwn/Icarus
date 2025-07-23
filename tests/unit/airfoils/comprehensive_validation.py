#!/usr/bin/env python3
"""
Comprehensive validation testing for JAX airfoil implementation.

This script performs systematic validation of:
1. Test suite consistency
2. Numerical accuracy against analytical solutions
3. Gradient computation correctness
4. JIT compilation stability

Requirements: 1.1, 1.4, 5.1, 5.2, 7.1, 7.2
"""

import sys
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

from ICARUS.airfoils.naca4 import NACA4


class ComprehensiveValidator:
    """Comprehensive validation suite for JAX airfoil implementation."""

    def __init__(self):
        self.results = {
            "test_consistency": {"passed": 0, "failed": 0, "errors": []},
            "numerical_accuracy": {"passed": 0, "failed": 0, "errors": []},
            "gradient_correctness": {"passed": 0, "failed": 0, "errors": []},
            "jit_stability": {"passed": 0, "failed": 0, "errors": []},
        }
        self.tolerance = 1e-6

    def log_result(
        self,
        category: str,
        test_name: str,
        passed: bool,
        error_msg: str = "",
    ):
        """Log test result to appropriate category."""
        if passed:
            self.results[category]["passed"] += 1
            print(f"  ✓ {test_name}")
        else:
            self.results[category]["failed"] += 1
            self.results[category]["errors"].append(f"{test_name}: {error_msg}")
            print(f"  ✗ {test_name}: {error_msg}")

    def validate_test_consistency(self) -> bool:
        """Validate that all tests pass consistently."""
        print("\n" + "=" * 60)
        print("1. VALIDATING TEST SUITE CONSISTENCY")
        print("=" * 60)

        try:
            # Run pytest programmatically to get detailed results
            import subprocess

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/unit/airfoils/jax_implementation/",
                    "-v",
                    "--tb=no",
                    "-q",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            # Parse results
            output_lines = result.stdout.split("\n")
            passed_count = 0
            failed_count = 0
            failed_tests = []

            for line in output_lines:
                if " PASSED " in line:
                    passed_count += 1
                elif " FAILED " in line:
                    failed_count += 1
                    test_name = (
                        line.split("::")[1].split(" ")[0] if "::" in line else line
                    )
                    failed_tests.append(test_name)

            # Log overall results
            total_tests = passed_count + failed_count
            if total_tests > 0:
                success_rate = (passed_count / total_tests) * 100
                self.log_result(
                    "test_consistency",
                    f"Overall test suite ({total_tests} tests)",
                    failed_count == 0,
                    f"{failed_count} failures, {success_rate:.1f}% success rate",
                )

                # Log individual failures
                for failed_test in failed_tests:
                    self.log_result(
                        "test_consistency",
                        f"Individual test: {failed_test}",
                        False,
                        "Test failed in suite run",
                    )
            else:
                self.log_result(
                    "test_consistency",
                    "Test discovery",
                    False,
                    "No tests found or executed",
                )

        except Exception as e:
            self.log_result("test_consistency", "Test execution", False, str(e))

        return self.results["test_consistency"]["failed"] == 0

    def validate_numerical_accuracy(self) -> bool:
        """Validate numerical accuracy against analytical solutions."""
        print("\n" + "=" * 60)
        print("2. VALIDATING NUMERICAL ACCURACY")
        print("=" * 60)

        try:
            # Test NACA 4-digit analytical accuracy
            self._test_naca4_analytical_accuracy()

            # Test surface interpolation accuracy
            self._test_surface_interpolation_accuracy()

            # Test geometric property accuracy
            self._test_geometric_property_accuracy()

            # Test convergence behavior
            self._test_convergence_behavior()

        except Exception as e:
            self.log_result(
                "numerical_accuracy",
                "Numerical validation setup",
                False,
                str(e),
            )

        return self.results["numerical_accuracy"]["failed"] == 0

    def _test_naca4_analytical_accuracy(self):
        """Test NACA 4-digit airfoil against analytical formulas."""
        try:
            # Test symmetric airfoil (NACA 0012)
            airfoil = NACA4(M=0.0, P=0.0, XX=0.12)

            # Test thickness distribution at known points
            x_test = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
            thickness = airfoil.thickness(x_test)

            # Analytical NACA thickness formula
            t = 0.12  # 12% thickness
            analytical_thickness = t * (
                0.2969 * jnp.sqrt(x_test)
                - 0.1260 * x_test
                - 0.3516 * x_test**2
                + 0.2843 * x_test**3
                - 0.1015 * x_test**4
            )

            # Compare with tolerance
            max_error = jnp.max(jnp.abs(thickness - analytical_thickness))
            self.log_result(
                "numerical_accuracy",
                "NACA 0012 thickness accuracy",
                max_error < self.tolerance,
                f"Max error: {max_error:.2e}",
            )

            # Test cambered airfoil (NACA 2412)
            airfoil_cambered = NACA4(M=0.02, P=0.4, XX=0.12)
            camber_line = airfoil_cambered.camber_line(x_test)

            # Analytical camber line for NACA 2412
            m, p = 0.02, 0.4
            analytical_camber = jnp.where(
                x_test <= p,
                m / p**2 * (2 * p * x_test - x_test**2),
                m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x_test - x_test**2),
            )

            max_camber_error = jnp.max(jnp.abs(camber_line - analytical_camber))
            self.log_result(
                "numerical_accuracy",
                "NACA 2412 camber accuracy",
                max_camber_error < self.tolerance,
                f"Max error: {max_camber_error:.2e}",
            )

        except Exception as e:
            self.log_result(
                "numerical_accuracy",
                "NACA analytical comparison",
                False,
                str(e),
            )

    def _test_surface_interpolation_accuracy(self):
        """Test surface interpolation accuracy."""
        try:
            airfoil = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)

            # Test interpolation at original coordinate points
            x_coords = jnp.linspace(0, 1, 20)

            # Evaluate surface at test coordinates
            upper_surface = airfoil.y_upper(x_coords)
            lower_surface = airfoil.y_lower(x_coords)

            # Check that surface evaluation is consistent
            thickness_calc = upper_surface - lower_surface
            thickness_direct = airfoil.thickness(x_coords)

            # Compare interpolated vs direct calculation (with reasonable tolerance for numerical precision)
            interpolation_tolerance = 1e-4
            thickness_error = jnp.max(jnp.abs(thickness_calc - thickness_direct))

            self.log_result(
                "numerical_accuracy",
                "Surface interpolation consistency",
                thickness_error < interpolation_tolerance,
                f"Thickness error: {thickness_error:.2e}",
            )

        except Exception as e:
            self.log_result(
                "numerical_accuracy",
                "Surface interpolation test",
                False,
                str(e),
            )

    def _test_geometric_property_accuracy(self):
        """Test geometric property calculations."""
        try:
            # Test known geometric properties
            airfoil = NACA4(M=0.0, P=0.0, XX=0.12)  # Symmetric airfoil

            # Maximum thickness should be 12% at approximately 30% chord
            max_thickness = airfoil.max_thickness
            expected_thickness = 0.12

            thickness_error = abs(max_thickness - expected_thickness)
            self.log_result(
                "numerical_accuracy",
                "Maximum thickness calculation",
                thickness_error < 0.01,
                f"Error: {thickness_error:.4f}",
            )

            # For symmetric airfoil, camber should be zero everywhere
            x_test = jnp.linspace(0, 1, 50)
            camber_line = airfoil.camber_line(x_test)
            max_camber_error = jnp.max(jnp.abs(camber_line))

            self.log_result(
                "numerical_accuracy",
                "Symmetric airfoil camber",
                max_camber_error < self.tolerance,
                f"Max camber: {max_camber_error:.2e}",
            )

        except Exception as e:
            self.log_result(
                "numerical_accuracy",
                "Geometric property test",
                False,
                str(e),
            )

    def _test_convergence_behavior(self):
        """Test convergence behavior with increasing resolution."""
        try:
            # Test convergence of surface evaluation with increasing points
            resolutions = [50, 100, 200]
            x_eval = jnp.array([0.25, 0.5, 0.75])

            results = []
            for n_points in resolutions:
                airfoil = NACA4(M=0.02, P=0.4, XX=0.12, n_points=n_points)
                upper_surface = airfoil.y_upper(x_eval)
                results.append(upper_surface)

            # Check convergence (higher resolution should give more accurate results)
            convergence_rate = jnp.max(jnp.abs(results[-1] - results[-2])) / jnp.max(
                jnp.abs(results[-2] - results[-3]),
            )

            self.log_result(
                "numerical_accuracy",
                "Surface evaluation convergence",
                convergence_rate < 1.0,
                f"Convergence rate: {convergence_rate:.4f}",
            )

        except Exception as e:
            self.log_result("numerical_accuracy", "Convergence test", False, str(e))

    def validate_gradient_correctness(self) -> bool:
        """Validate gradient computation correctness."""
        print("\n" + "=" * 60)
        print("3. VALIDATING GRADIENT COMPUTATION")
        print("=" * 60)

        try:
            # Test basic gradient computation
            self._test_basic_gradients()

            # Test gradient accuracy via finite differences
            self._test_gradient_accuracy()

            # Test higher-order derivatives
            self._test_higher_order_derivatives()

            # Test gradient through complex operations
            self._test_complex_gradient_operations()

        except Exception as e:
            self.log_result(
                "gradient_correctness",
                "Gradient validation setup",
                False,
                str(e),
            )

        return self.results["gradient_correctness"]["failed"] == 0

    def _test_basic_gradients(self):
        """Test basic gradient computation functionality."""
        try:

            def surface_function(params):
                m, p, t = params
                airfoil = NACA4(M=m / 100, P=p / 10, XX=t / 100)
                x_eval = jnp.array([0.5])
                return jnp.sum(airfoil.y_upper(x_eval))

            # Test gradient computation
            params = jnp.array([2.0, 4.0, 12.0])
            grad_fn = jax.grad(surface_function)
            gradients = grad_fn(params)

            # Check that gradients are finite and non-zero
            gradient_valid = jnp.all(jnp.isfinite(gradients)) and jnp.any(
                jnp.abs(gradients) > 1e-10,
            )

            self.log_result(
                "gradient_correctness",
                "Basic gradient computation",
                gradient_valid,
                f"Gradients: {gradients}",
            )

        except Exception as e:
            self.log_result(
                "gradient_correctness",
                "Basic gradient test",
                False,
                str(e),
            )

    def _test_gradient_accuracy(self):
        """Test gradient accuracy using finite differences."""
        try:

            def test_function(x):
                airfoil = NACA4(M=0.02, P=0.4, XX=0.12)
                return jnp.sum(airfoil.y_upper(x))

            x = jnp.array([0.3, 0.5, 0.7])

            # Analytical gradient
            analytical_grad = jax.grad(test_function)(x)

            # Numerical gradient (finite differences)
            eps = 1e-6
            numerical_grad = jnp.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.at[i].add(eps)
                x_minus = x.at[i].add(-eps)
                numerical_grad = numerical_grad.at[i].set(
                    (test_function(x_plus) - test_function(x_minus)) / (2 * eps),
                )

            # Compare gradients
            grad_error = jnp.max(jnp.abs(analytical_grad - numerical_grad))
            gradient_tolerance = 1e-4

            self.log_result(
                "gradient_correctness",
                "Gradient accuracy vs finite differences",
                grad_error < gradient_tolerance,
                f"Max error: {grad_error:.2e}",
            )

        except Exception as e:
            self.log_result(
                "gradient_correctness",
                "Gradient accuracy test",
                False,
                str(e),
            )

    def _test_higher_order_derivatives(self):
        """Test higher-order derivative computation."""
        try:

            def test_function(params):
                m, p, t = params
                airfoil = NACA4(M=m / 100, P=p / 10, XX=t / 100)
                x_eval = jnp.array([0.5])
                return jnp.sum(airfoil.thickness(x_eval))

            params = jnp.array([2.0, 4.0, 12.0])

            # First derivative
            first_grad = jax.grad(test_function)(params)

            # Second derivative (Hessian diagonal)
            second_grad = jax.grad(jax.grad(test_function))(params)

            # Check that higher-order derivatives are finite
            higher_order_valid = jnp.all(jnp.isfinite(first_grad)) and jnp.all(
                jnp.isfinite(second_grad),
            )

            self.log_result(
                "gradient_correctness",
                "Higher-order derivatives",
                higher_order_valid,
                f"1st: {jnp.max(jnp.abs(first_grad)):.2e}, 2nd: {jnp.max(jnp.abs(second_grad)):.2e}",
            )

        except Exception as e:
            self.log_result(
                "gradient_correctness",
                "Higher-order derivative test",
                False,
                str(e),
            )

    def _test_complex_gradient_operations(self):
        """Test gradients through complex operations like morphing."""
        try:

            def complex_function(params):
                m1, p1, t1, m2, p2, t2 = params
                airfoil1 = NACA4(M=m1 / 100, P=p1 / 10, XX=t1 / 100)
                airfoil2 = NACA4(M=m2 / 100, P=p2 / 10, XX=t2 / 100)
                x_eval = jnp.array([0.25, 0.5, 0.75])
                # Test gradient through complex combination of operations
                result1 = jnp.sum(airfoil1.y_upper(x_eval))
                result2 = jnp.sum(airfoil2.y_upper(x_eval))
                return result1 + result2

            params = jnp.array([0.0, 0.0, 12.0, 4.0, 4.0, 15.0])
            complex_grad = jax.grad(complex_function)(params)

            # Check gradient is finite and reasonable
            complex_grad_valid = jnp.all(jnp.isfinite(complex_grad)) and jnp.any(
                jnp.abs(complex_grad) > 1e-10,
            )

            self.log_result(
                "gradient_correctness",
                "Gradient through complex operations",
                complex_grad_valid,
                f"Complex gradient max: {jnp.max(jnp.abs(complex_grad)):.2e}",
            )

        except Exception as e:
            self.log_result(
                "gradient_correctness",
                "Complex gradient test",
                False,
                str(e),
            )

    def validate_jit_stability(self) -> bool:
        """Validate JIT compilation stability."""
        print("\n" + "=" * 60)
        print("4. VALIDATING JIT COMPILATION STABILITY")
        print("=" * 60)

        try:
            # Test basic JIT compilation
            self._test_basic_jit_compilation()

            # Test JIT with different input shapes
            self._test_jit_input_shape_stability()

            # Test JIT recompilation behavior
            self._test_jit_recompilation()

            # Test JIT with gradients
            self._test_jit_gradient_stability()

        except Exception as e:
            self.log_result("jit_stability", "JIT validation setup", False, str(e))

        return self.results["jit_stability"]["failed"] == 0

    def _test_basic_jit_compilation(self):
        """Test basic JIT compilation functionality."""
        try:

            @jax.jit
            def jit_surface_eval(params, x):
                m, p, t = params
                airfoil = NACA4(M=m / 100, P=p / 10, XX=t / 100)
                return airfoil.y_upper(x)

            params = jnp.array([2.0, 4.0, 12.0])
            x = jnp.array([0.25, 0.5, 0.75])

            # First call (compilation + execution)
            start_time = time.time()
            result1 = jit_surface_eval(params, x)
            first_call_time = time.time() - start_time

            # Second call (execution only)
            start_time = time.time()
            result2 = jit_surface_eval(params, x)
            second_call_time = time.time() - start_time

            # Results should be identical
            results_identical = jnp.allclose(result1, result2, rtol=1e-10)

            # Second call should be faster (no compilation)
            speedup_achieved = second_call_time < first_call_time

            self.log_result(
                "jit_stability",
                "Basic JIT compilation",
                results_identical and speedup_achieved,
                f"Identical: {results_identical}, Speedup: {speedup_achieved}",
            )

        except Exception as e:
            self.log_result("jit_stability", "Basic JIT test", False, str(e))

    def _test_jit_input_shape_stability(self):
        """Test JIT stability with different input shapes."""
        try:

            @jax.jit
            def jit_batch_eval(x):
                airfoil = NACA4(M=0.02, P=0.4, XX=0.12)
                return airfoil.y_upper(x)

            # Test with different input shapes
            shapes_to_test = [
                jnp.array([0.5]),
                jnp.array([0.25, 0.5, 0.75]),
                jnp.array([0.1, 0.3, 0.5, 0.7, 0.9]),
            ]

            all_shapes_work = True
            for i, x in enumerate(shapes_to_test):
                try:
                    result = jit_batch_eval(x)
                    shape_valid = result.shape == x.shape and jnp.all(
                        jnp.isfinite(result),
                    )
                    if not shape_valid:
                        all_shapes_work = False
                        break
                except Exception:
                    all_shapes_work = False
                    break

            self.log_result(
                "jit_stability",
                "JIT input shape stability",
                all_shapes_work,
                f"Tested {len(shapes_to_test)} different shapes",
            )

        except Exception as e:
            self.log_result("jit_stability", "JIT shape stability test", False, str(e))

    def _test_jit_recompilation(self):
        """Test JIT recompilation behavior."""
        try:

            @jax.jit
            def jit_parameterized_eval(params, x):
                m, p, t = params
                airfoil = NACA4(M=m / 100, P=p / 10, XX=t / 100)
                return airfoil.y_upper(x)

            x = jnp.array([0.5])

            # Test with different parameter values (should not recompile)
            params1 = jnp.array([2.0, 4.0, 12.0])
            params2 = jnp.array([3.0, 5.0, 15.0])

            result1 = jit_parameterized_eval(params1, x)
            result2 = jit_parameterized_eval(params2, x)

            # Results should be different but both finite
            results_different = not jnp.allclose(result1, result2)
            both_finite = jnp.all(jnp.isfinite(result1)) and jnp.all(
                jnp.isfinite(result2),
            )

            self.log_result(
                "jit_stability",
                "JIT recompilation behavior",
                results_different and both_finite,
                f"Different results: {results_different}, Both finite: {both_finite}",
            )

        except Exception as e:
            self.log_result("jit_stability", "JIT recompilation test", False, str(e))

    def _test_jit_gradient_stability(self):
        """Test JIT compilation with gradient computation."""
        try:

            @jax.jit
            def jit_gradient_eval(params):
                m, p, t = params
                airfoil = NACA4(M=m / 100, P=p / 10, XX=t / 100)
                x = jnp.array([0.5])
                return jnp.sum(airfoil.y_upper(x))

            jit_grad_fn = jax.jit(jax.grad(jit_gradient_eval))

            params = jnp.array([2.0, 4.0, 12.0])

            # Test JIT-compiled gradient
            grad_result = jit_grad_fn(params)

            # Compare with non-JIT gradient
            non_jit_grad = jax.grad(jit_gradient_eval)(params)

            gradients_match = jnp.allclose(grad_result, non_jit_grad, rtol=1e-10)

            self.log_result(
                "jit_stability",
                "JIT gradient stability",
                gradients_match,
                f"Gradient match: {gradients_match}",
            )

        except Exception as e:
            self.log_result("jit_stability", "JIT gradient test", False, str(e))

    def print_summary(self):
        """Print comprehensive validation summary."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE VALIDATION SUMMARY")
        print("=" * 80)

        total_passed = 0
        total_failed = 0

        for category, results in self.results.items():
            passed = results["passed"]
            failed = results["failed"]
            total_passed += passed
            total_failed += failed

            status = "✓ PASS" if failed == 0 else "✗ FAIL"
            print(f"\n{category.upper().replace('_', ' ')}: {status}")
            print(f"  Passed: {passed}, Failed: {failed}")

            if results["errors"]:
                print("  Errors:")
                for error in results["errors"][:5]:  # Show first 5 errors
                    print(f"    - {error}")
                if len(results["errors"]) > 5:
                    print(f"    ... and {len(results['errors']) - 5} more")

        overall_success_rate = (
            (total_passed / (total_passed + total_failed) * 100)
            if (total_passed + total_failed) > 0
            else 0
        )
        overall_status = "✓ PASS" if total_failed == 0 else "✗ FAIL"

        print(f"\nOVERALL VALIDATION: {overall_status}")
        print(f"Total tests: {total_passed + total_failed}")
        print(f"Success rate: {overall_success_rate:.1f}%")

        # Production readiness assessment
        print("\nPRODUCTION READINESS ASSESSMENT:")
        if total_failed == 0:
            print("✓ JAX airfoil implementation is READY for production deployment")
        else:
            print(
                "✗ JAX airfoil implementation requires fixes before production deployment",
            )
            print(f"  {total_failed} validation issues need to be addressed")

        return total_failed == 0


def main():
    """Run comprehensive validation."""
    print("JAX Airfoil Implementation - Comprehensive Validation")
    print("=" * 80)
    print("Validating production readiness across all critical aspects...")

    # Suppress JAX warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)

    validator = ComprehensiveValidator()

    # Run all validation categories
    test_consistency_ok = validator.validate_test_consistency()
    numerical_accuracy_ok = validator.validate_numerical_accuracy()
    gradient_correctness_ok = validator.validate_gradient_correctness()
    jit_stability_ok = validator.validate_jit_stability()

    # Print comprehensive summary
    overall_success = validator.print_summary()

    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
