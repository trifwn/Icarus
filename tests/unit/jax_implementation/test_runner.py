#!/usr/bin/env python3
"""
Simple test runner for JAX airfoil implementation tests.

This script runs all the test modules and provides a summary of results.
"""

import importlib
import sys
import traceback
from pathlib import Path


def run_test_class(test_class):
    """Run all test methods in a test class."""
    results = {"passed": 0, "failed": 0, "errors": []}

    instance = test_class()

    # Get all test methods
    test_methods = [method for method in dir(instance) if method.startswith("test_")]

    for method_name in test_methods:
        try:
            method = getattr(instance, method_name)
            method()
            results["passed"] += 1
            print(f"  ✓ {method_name}")
        except Exception as e:
            results["failed"] += 1
            error_info = f"{method_name}: {str(e)}"
            results["errors"].append(error_info)
            print(f"  ✗ {method_name}: {str(e)}")

    return results


def run_test_module(module_path):
    """Run all test classes in a module."""
    print(f"\nRunning tests in {module_path}")
    print("=" * 60)

    try:
        # Import the module
        spec = importlib.util.spec_from_file_location("test_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find all test classes
        test_classes = []
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and name.startswith("Test")
                and hasattr(obj, "__module__")
            ):
                test_classes.append(obj)

        total_results = {"passed": 0, "failed": 0, "errors": []}

        for test_class in test_classes:
            print(f"\n{test_class.__name__}:")
            results = run_test_class(test_class)
            total_results["passed"] += results["passed"]
            total_results["failed"] += results["failed"]
            total_results["errors"].extend(results["errors"])

        return total_results

    except Exception as e:
        print(f"Error importing module {module_path}: {str(e)}")
        traceback.print_exc()
        return {"passed": 0, "failed": 1, "errors": [f"Module import error: {str(e)}"]}


def main():
    """Run all tests in the JAX airfoil implementation."""
    print("JAX Airfoil Implementation Test Suite")
    print("=" * 60)

    # Add current directory to Python path
    sys.path.insert(0, str(Path.cwd()))

    # Test modules to run
    test_modules = [
        "tests/unit/airfoils/jax_implementation/core/test_core_functionality.py",
        "tests/unit/airfoils/jax_implementation/core/test_interpolation_surface.py",
        "tests/unit/airfoils/jax_implementation/operations/test_geometric_operations.py",
        "tests/unit/airfoils/jax_implementation/batch/test_batch_operations.py",
        "tests/unit/airfoils/jax_implementation/performance/test_performance_validation.py",
        "tests/unit/airfoils/jax_implementation/compatibility/test_api_compatibility.py",
        "tests/unit/airfoils/jax_implementation/edge_cases/test_edge_cases_errors.py",
    ]

    overall_results = {"passed": 0, "failed": 0, "errors": []}

    for module_path in test_modules:
        if Path(module_path).exists():
            results = run_test_module(module_path)
            overall_results["passed"] += results["passed"]
            overall_results["failed"] += results["failed"]
            overall_results["errors"].extend(results["errors"])
        else:
            print(f"Warning: Test module {module_path} not found")

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests run: {overall_results['passed'] + overall_results['failed']}")
    print(f"Passed: {overall_results['passed']}")
    print(f"Failed: {overall_results['failed']}")

    if overall_results["errors"]:
        print("\nFailed tests:")
        for error in overall_results["errors"][:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(overall_results["errors"]) > 10:
            print(f"  ... and {len(overall_results['errors']) - 10} more")

    success_rate = (
        overall_results["passed"]
        / (overall_results["passed"] + overall_results["failed"])
        * 100
        if (overall_results["passed"] + overall_results["failed"]) > 0
        else 0
    )
    print(f"\nSuccess rate: {success_rate:.1f}%")

    return overall_results["failed"] == 0


if __name__ == "__main__":
    import importlib.util

    success = main()
    sys.exit(0 if success else 1)
