#!/usr/bin/env python3
"""
Production Readiness Check for JAX Airfoil Implementation

This script performs focused validation on the critical issues identified
in the comprehensive validation and provides specific recommendations for
production deployment readiness.

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


class ProductionReadinessChecker:
    """Focused production readiness validation."""

    def __init__(self):
        self.issues = []
        self.recommendations = []

    def add_issue(
        self,
        severity: str,
        category: str,
        description: str,
        recommendation: str,
    ):
        """Add an identified issue with recommendation."""
        self.issues.append(
            {
                "severity": severity,
                "category": category,
                "description": description,
                "recommendation": recommendation,
            },
        )

    def check_critical_numerical_accuracy(self) -> bool:
        """Check critical numerical accuracy issues."""
        print("\n🔍 CHECKING CRITICAL NUMERICAL ACCURACY")
        print("=" * 60)

        issues_found = 0

        # Check NACA thickness formula accuracy
        try:
            airfoil = NACA4(M=0.0, P=0.0, XX=0.12)
            x_test = jnp.array([0.3])  # Test at 30% chord
            computed_thickness = airfoil.thickness_distribution(x_test)[0]

            # Analytical NACA thickness distribution at 30% chord
            x = 0.3
            t = 0.12
            analytical_thickness = t * (
                0.2969 * jnp.sqrt(x)
                - 0.1260 * x
                - 0.3516 * x**2
                + 0.2843 * x**3
                - 0.1036 * x**4
            )

            error = abs(computed_thickness - analytical_thickness)
            tolerance = 0.01  # 1% tolerance

            if error > tolerance:
                issues_found += 1
                self.add_issue(
                    "HIGH",
                    "Numerical Accuracy",
                    f"NACA thickness calculation error: {error:.4f} (>{tolerance:.4f})",
                    "Review thickness calculation implementation and coordinate system consistency",
                )
                print(
                    f"❌ NACA thickness accuracy: Error {error:.4f} > {tolerance:.4f}",
                )
                print(
                    f"   Computed: {computed_thickness:.6f}, Analytical: {analytical_thickness:.6f}",
                )
            else:
                print(
                    f"✅ NACA thickness accuracy: Error {error:.6f} < {tolerance:.4f}",
                )

        except Exception as e:
            issues_found += 1
            self.add_issue(
                "HIGH",
                "Numerical Accuracy",
                f"NACA thickness calculation failed: {str(e)}",
                "Fix implementation errors in thickness calculation",
            )
            print(f"❌ NACA thickness calculation failed: {str(e)}")

        # Check surface continuity
        try:
            airfoil = NACA4(M=0.02, P=0.4, XX=0.12)
            x_test = jnp.linspace(0.01, 0.99, 50)

            upper_surface = airfoil.y_upper(x_test)
            lower_surface = airfoil.y_lower(x_test)

            # Check for NaN or infinite values
            if not (
                jnp.all(jnp.isfinite(upper_surface))
                and jnp.all(jnp.isfinite(lower_surface))
            ):
                issues_found += 1
                self.add_issue(
                    "HIGH",
                    "Numerical Accuracy",
                    "Surface evaluation produces NaN or infinite values",
                    "Fix numerical stability issues in surface interpolation",
                )
                print("❌ Surface continuity: NaN or infinite values detected")
            else:
                print("✅ Surface continuity: All values finite")

        except Exception as e:
            issues_found += 1
            self.add_issue(
                "HIGH",
                "Numerical Accuracy",
                f"Surface continuity check failed: {str(e)}",
                "Fix surface evaluation implementation",
            )
            print(f"❌ Surface continuity check failed: {str(e)}")

        return issues_found == 0

    def check_performance_issues(self) -> bool:
        """Check performance-related issues."""
        print("\n⚡ CHECKING PERFORMANCE ISSUES")
        print("=" * 60)

        issues_found = 0

        # Check JIT compilation overhead
        try:

            @jax.jit
            def jit_surface_eval(params):
                m, p, t = params
                airfoil = NACA4(M=m / 100, P=p / 10, XX=t / 100)
                x = jnp.array([0.25, 0.5, 0.75])
                return airfoil.y_upper(x)

            params = jnp.array([2.0, 4.0, 12.0])

            # Warm up JIT
            _ = jit_surface_eval(params)

            # Time JIT execution
            start_time = time.time()
            for _ in range(100):
                result = jit_surface_eval(params)
            jit_time = (time.time() - start_time) / 100

            # Time non-JIT execution
            def non_jit_surface_eval(params):
                m, p, t = params
                airfoil = NACA4(M=m / 100, P=p / 10, XX=t / 100)
                x = jnp.array([0.25, 0.5, 0.75])
                return airfoil.y_upper(x)

            start_time = time.time()
            for _ in range(100):
                result = non_jit_surface_eval(params)
            non_jit_time = (time.time() - start_time) / 100

            speedup = non_jit_time / jit_time

            if speedup < 1.5:  # Expect at least 1.5x speedup
                issues_found += 1
                self.add_issue(
                    "MEDIUM",
                    "Performance",
                    f"JIT compilation speedup insufficient: {speedup:.2f}x (expected >1.5x)",
                    "Optimize JIT compilation patterns and reduce compilation overhead",
                )
                print(f"❌ JIT speedup: {speedup:.2f}x < 1.5x")
            else:
                print(f"✅ JIT speedup: {speedup:.2f}x > 1.5x")

        except Exception as e:
            issues_found += 1
            self.add_issue(
                "MEDIUM",
                "Performance",
                f"JIT performance check failed: {str(e)}",
                "Fix JIT compilation issues",
            )
            print(f"❌ JIT performance check failed: {str(e)}")

        # Check batch operation efficiency
        try:
            # Individual operations
            airfoils = [NACA4(M=0.02, P=0.4, XX=0.12) for _ in range(10)]
            x = jnp.array([0.5])

            start_time = time.time()
            individual_results = [airfoil.y_upper(x) for airfoil in airfoils]
            individual_time = time.time() - start_time

            # Vectorized operation (simulated batch)
            @jax.jit
            def batch_eval(x_eval):
                results = []
                for i in range(10):
                    airfoil = NACA4(M=0.02, P=0.4, XX=0.12)
                    results.append(airfoil.y_upper(x_eval))
                return jnp.stack(results)

            # Warm up
            _ = batch_eval(x)

            start_time = time.time()
            batch_result = batch_eval(x)
            batch_time = time.time() - start_time

            if (
                batch_time > individual_time * 2
            ):  # Batch shouldn't be more than 2x slower
                issues_found += 1
                self.add_issue(
                    "MEDIUM",
                    "Performance",
                    f"Batch operations too slow: {batch_time:.4f}s vs {individual_time:.4f}s individual",
                    "Optimize batch processing implementation and vectorization",
                )
                print(
                    f"❌ Batch efficiency: {batch_time:.4f}s > {individual_time * 2:.4f}s",
                )
            else:
                print(
                    f"✅ Batch efficiency: {batch_time:.4f}s < {individual_time * 2:.4f}s",
                )

        except Exception as e:
            issues_found += 1
            self.add_issue(
                "MEDIUM",
                "Performance",
                f"Batch operation check failed: {str(e)}",
                "Fix batch processing implementation",
            )
            print(f"❌ Batch operation check failed: {str(e)}")

        return issues_found == 0

    def check_gradient_stability(self) -> bool:
        """Check gradient computation stability."""
        print("\n🎯 CHECKING GRADIENT COMPUTATION STABILITY")
        print("=" * 60)

        issues_found = 0

        # Check basic gradient computation
        try:

            def test_function(params):
                m, p, t = params
                airfoil = NACA4(M=m / 100, P=p / 10, XX=t / 100)
                x = jnp.array([0.5])
                return jnp.sum(airfoil.y_upper(x))  # Ensure scalar output

            params = jnp.array([2.0, 4.0, 12.0])
            grad_fn = jax.grad(test_function)
            gradients = grad_fn(params)

            if not (
                jnp.all(jnp.isfinite(gradients)) and jnp.any(jnp.abs(gradients) > 1e-10)
            ):
                issues_found += 1
                self.add_issue(
                    "HIGH",
                    "Gradient Computation",
                    "Basic gradient computation produces invalid results",
                    "Fix gradient computation implementation",
                )
                print(f"❌ Basic gradients: Invalid results {gradients}")
            else:
                print(f"✅ Basic gradients: Valid results {gradients}")

        except Exception as e:
            issues_found += 1
            self.add_issue(
                "HIGH",
                "Gradient Computation",
                f"Basic gradient computation failed: {str(e)}",
                "Fix gradient computation errors",
            )
            print(f"❌ Basic gradient computation failed: {str(e)}")

        # Check higher-order derivatives
        try:

            def scalar_test_function(param):
                airfoil = NACA4(M=param / 100, P=0.4, XX=0.12)
                x = jnp.array([0.5])
                return airfoil.y_upper(x)[0]  # Return scalar

            param = 2.0
            first_grad = jax.grad(scalar_test_function)(param)
            second_grad = jax.grad(jax.grad(scalar_test_function))(param)

            if not (jnp.isfinite(first_grad) and jnp.isfinite(second_grad)):
                issues_found += 1
                self.add_issue(
                    "MEDIUM",
                    "Gradient Computation",
                    "Higher-order derivatives produce invalid results",
                    "Fix higher-order derivative computation",
                )
                print("❌ Higher-order derivatives: Invalid results")
            else:
                print("✅ Higher-order derivatives: Valid results")

        except Exception as e:
            issues_found += 1
            self.add_issue(
                "MEDIUM",
                "Gradient Computation",
                f"Higher-order derivative check failed: {str(e)}",
                "Fix higher-order derivative implementation",
            )
            print(f"❌ Higher-order derivative check failed: {str(e)}")

        return issues_found == 0

    def check_test_suite_stability(self) -> bool:
        """Check test suite stability and consistency."""
        print("\n🧪 CHECKING TEST SUITE STABILITY")
        print("=" * 60)

        issues_found = 0

        # Run a subset of critical tests
        try:
            import subprocess

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/unit/airfoils/jax_implementation/core/",
                    "-v",
                    "--tb=no",
                    "-q",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                issues_found += 1
                self.add_issue(
                    "HIGH",
                    "Test Suite",
                    "Core functionality tests failing",
                    "Fix core test failures before production deployment",
                )
                print("❌ Core tests: Some tests failing")
            else:
                print("✅ Core tests: All passing")

        except Exception as e:
            issues_found += 1
            self.add_issue(
                "HIGH",
                "Test Suite",
                f"Test execution failed: {str(e)}",
                "Fix test execution environment",
            )
            print(f"❌ Test execution failed: {str(e)}")

        return issues_found == 0

    def generate_production_readiness_report(self) -> bool:
        """Generate final production readiness assessment."""
        print("\n" + "=" * 80)
        print("PRODUCTION READINESS ASSESSMENT")
        print("=" * 80)

        # Run all checks
        numerical_ok = self.check_critical_numerical_accuracy()
        performance_ok = self.check_performance_issues()
        gradient_ok = self.check_gradient_stability()
        test_ok = self.check_test_suite_stability()

        # Categorize issues by severity
        high_issues = [issue for issue in self.issues if issue["severity"] == "HIGH"]
        medium_issues = [
            issue for issue in self.issues if issue["severity"] == "MEDIUM"
        ]
        low_issues = [issue for issue in self.issues if issue["severity"] == "LOW"]

        print("\n📊 ISSUE SUMMARY:")
        print(f"   High severity: {len(high_issues)}")
        print(f"   Medium severity: {len(medium_issues)}")
        print(f"   Low severity: {len(low_issues)}")
        print(f"   Total issues: {len(self.issues)}")

        # Production readiness decision
        production_ready = len(high_issues) == 0 and len(medium_issues) <= 2

        print(
            f"\n🎯 PRODUCTION READINESS: {'✅ READY' if production_ready else '❌ NOT READY'}",
        )

        if not production_ready:
            print("\n🚨 BLOCKING ISSUES:")
            for issue in high_issues:
                print(f"   • {issue['category']}: {issue['description']}")
                print(f"     → {issue['recommendation']}")

            if medium_issues:
                print("\n⚠️  MEDIUM PRIORITY ISSUES:")
                for issue in medium_issues[:3]:  # Show top 3
                    print(f"   • {issue['category']}: {issue['description']}")
                    print(f"     → {issue['recommendation']}")

        print("\n📋 NEXT STEPS:")
        if production_ready:
            print("   1. ✅ Implementation is ready for production deployment")
            print("   2. 🔄 Set up continuous validation monitoring")
            print("   3. 📈 Monitor performance in production environment")
        else:
            print("   1. 🔧 Address all HIGH severity issues")
            print("   2. 🔧 Address critical MEDIUM severity issues")
            print("   3. 🧪 Re-run validation after fixes")
            print("   4. 📝 Update documentation with any changes")

        return production_ready


def main():
    """Run production readiness check."""
    print("JAX Airfoil Implementation - Production Readiness Check")
    print("=" * 80)
    print("Performing focused validation on critical production issues...")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)

    checker = ProductionReadinessChecker()
    production_ready = checker.generate_production_readiness_report()

    return production_ready


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
