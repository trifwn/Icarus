"""
Integration test runner for JAX airfoil implementation.

This module provides a comprehensive test runner that executes all integration
and validation tests, generates reports, and validates the overall system.

Requirements covered: 1.3, 2.1, 3.1, 8.2
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from typing import Dict

import psutil

# Import all test modules


class IntegrationTestRunner:
    """Comprehensive integration test runner."""

    def __init__(self, output_dir: str = "tests/integration_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.start_time = None
        self.end_time = None

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests and return comprehensive results."""
        self.start_time = time.time()

        print("=" * 80)
        print("JAX AIRFOIL INTEGRATION TEST SUITE")
        print("=" * 80)

        # Test categories to run
        test_categories = [
            {
                "name": "ICARUS Module Integration",
                "module": "test_jax_integration_validation::TestIcarusModuleIntegration",
                "description": "Test integration with existing ICARUS modules",
            },
            {
                "name": "Airfoil Database Validation",
                "module": "test_jax_integration_validation::TestAirfoilDatabaseValidation",
                "description": "Validate against known airfoil databases",
            },
            {
                "name": "Complex Operation Regression",
                "module": "test_jax_integration_validation::TestComplexOperationRegression",
                "description": "Test regression on complex airfoil operations",
            },
            {
                "name": "Memory Usage Validation",
                "module": "test_jax_integration_validation::TestMemoryUsageValidation",
                "description": "Test memory usage under various workloads",
            },
            {
                "name": "Optimization Gradient Accuracy",
                "module": "test_jax_integration_validation::TestOptimizationGradientAccuracy",
                "description": "Validate gradient accuracy for optimization workflows",
            },
            {
                "name": "JIT Compilation Performance",
                "module": "test_jax_performance_validation::TestJITCompilationPerformance",
                "description": "Test JIT compilation performance and caching",
            },
            {
                "name": "Memory Usage Scaling",
                "module": "test_jax_performance_validation::TestMemoryUsageScaling",
                "description": "Test memory usage scaling under various workloads",
            },
            {
                "name": "Gradient Computation Performance",
                "module": "test_jax_performance_validation::TestGradientComputationPerformance",
                "description": "Test performance of gradient computations",
            },
            {
                "name": "Large Scale Workloads",
                "module": "test_jax_performance_validation::TestLargeScaleWorkloads",
                "description": "Test performance under large-scale workloads",
            },
            {
                "name": "NACA Generation Regression",
                "module": "test_jax_regression_validation::TestNACAGenerationRegression",
                "description": "Test regression for NACA airfoil generation",
            },
            {
                "name": "Morphing Operation Regression",
                "module": "test_jax_regression_validation::TestMorphingOperationRegression",
                "description": "Test regression for airfoil morphing operations",
            },
            {
                "name": "Flap Operation Regression",
                "module": "test_jax_regression_validation::TestFlapOperationRegression",
                "description": "Test regression for flap operations",
            },
            {
                "name": "Repaneling Regression",
                "module": "test_jax_regression_validation::TestRepanelingRegression",
                "description": "Test regression for repaneling operations",
            },
            {
                "name": "Batch Operation Regression",
                "module": "test_jax_regression_validation::TestBatchOperationRegression",
                "description": "Test regression for batch operations",
            },
        ]

        # Run each test category
        for category in test_categories:
            self._run_test_category(category)

        self.end_time = time.time()

        # Generate comprehensive report
        self._generate_report()

        return self.results

    def _run_test_category(self, category: Dict[str, str]):
        """Run a specific test category."""
        print(f"\n{'-' * 60}")
        print(f"Running: {category['name']}")
        print(f"Description: {category['description']}")
        print(f"{'-' * 60}")

        start_time = time.time()
        initial_memory = self._get_memory_usage()

        # Run pytest for this specific test class
        test_file = Path(__file__).parent / f"{category['module'].split('::')[0]}.py"
        test_class = category["module"].split("::")[1]

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(test_file) + f"::{test_class}",
            "-v",
            "--tb=short",
            "--no-header",
            f"--junitxml={self.output_dir}/{category['name'].replace(' ', '_')}_results.xml",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per category
            )

            end_time = time.time()
            final_memory = self._get_memory_usage()

            # Parse results
            success = result.returncode == 0
            duration = end_time - start_time
            memory_delta = final_memory - initial_memory

            # Store results
            self.results[category["name"]] = {
                "success": success,
                "duration": duration,
                "memory_delta": memory_delta,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "description": category["description"],
            }

            # Print summary
            status = "PASSED" if success else "FAILED"
            print(f"Status: {status}")
            print(f"Duration: {duration:.2f}s")
            print(f"Memory Delta: {memory_delta:.1f}MB")

            if not success:
                print("STDERR:")
                print(result.stderr)
                print("STDOUT:")
                print(result.stdout[-1000:])  # Last 1000 chars

        except subprocess.TimeoutExpired:
            self.results[category["name"]] = {
                "success": False,
                "duration": 300.0,
                "memory_delta": 0.0,
                "stdout": "",
                "stderr": "Test timed out after 5 minutes",
                "return_code": -1,
                "description": category["description"],
            }
            print("Status: TIMEOUT")

        except Exception as e:
            self.results[category["name"]] = {
                "success": False,
                "duration": 0.0,
                "memory_delta": 0.0,
                "stdout": "",
                "stderr": str(e),
                "return_code": -1,
                "description": category["description"],
            }
            print(f"Status: ERROR - {e}")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def _generate_report(self):
        """Generate comprehensive test report."""
        total_duration = self.end_time - self.start_time

        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r["success"])
        failed_tests = total_tests - passed_tests

        total_test_duration = sum(r["duration"] for r in self.results.values())
        total_memory_delta = sum(r["memory_delta"] for r in self.results.values())

        # Generate summary report
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration": total_duration,
            "test_duration": total_test_duration,
            "overhead_duration": total_duration - total_test_duration,
            "total_memory_delta": total_memory_delta,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            "results": self.results,
        }

        # Save detailed results
        with open(self.output_dir / "integration_test_results.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Generate human-readable report
        self._generate_human_readable_report(summary)

        # Print final summary
        print("\n" + "=" * 80)
        print("INTEGRATION TEST SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Total Duration: {total_duration:.2f}s")
        print(f"Memory Usage: {total_memory_delta:.1f}MB")

        if failed_tests > 0:
            print("\nFAILED TESTS:")
            for name, result in self.results.items():
                if not result["success"]:
                    print(f"  - {name}: {result['stderr'][:100]}...")

        print(f"\nDetailed results saved to: {self.output_dir}")
        print("=" * 80)

    def _generate_human_readable_report(self, summary: Dict[str, Any]):
        """Generate human-readable HTML report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>JAX Airfoil Integration Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .metric {{ text-align: center; padding: 10px; background-color: #e8f4f8; border-radius: 5px; }}
        .test-result {{ margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .passed {{ background-color: #d4edda; border-left: 5px solid #28a745; }}
        .failed {{ background-color: #f8d7da; border-left: 5px solid #dc3545; }}
        .details {{ font-size: 0.9em; color: #666; margin-top: 10px; }}
        pre {{ background-color: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>JAX Airfoil Integration Test Report</h1>
        <p>Generated on: {summary["timestamp"]}</p>
    </div>

    <div class="summary">
        <div class="metric">
            <h3>{summary["total_tests"]}</h3>
            <p>Total Tests</p>
        </div>
        <div class="metric">
            <h3>{summary["passed_tests"]}</h3>
            <p>Passed</p>
        </div>
        <div class="metric">
            <h3>{summary["failed_tests"]}</h3>
            <p>Failed</p>
        </div>
        <div class="metric">
            <h3>{summary["success_rate"]:.1%}</h3>
            <p>Success Rate</p>
        </div>
        <div class="metric">
            <h3>{summary["total_duration"]:.1f}s</h3>
            <p>Total Duration</p>
        </div>
        <div class="metric">
            <h3>{summary["total_memory_delta"]:.1f}MB</h3>
            <p>Memory Usage</p>
        </div>
    </div>

    <h2>Test Results</h2>
"""

        for name, result in summary["results"].items():
            status_class = "passed" if result["success"] else "failed"
            status_text = "PASSED" if result["success"] else "FAILED"

            html_content += f"""
    <div class="test-result {status_class}">
        <h3>{name} - {status_text}</h3>
        <p>{result["description"]}</p>
        <div class="details">
            <strong>Duration:</strong> {result["duration"]:.2f}s |
            <strong>Memory Delta:</strong> {result["memory_delta"]:.1f}MB |
            <strong>Return Code:</strong> {result["return_code"]}
        </div>
"""

            if not result["success"] and result["stderr"]:
                html_content += f"""
        <details>
            <summary>Error Details</summary>
            <pre>{result["stderr"]}</pre>
        </details>
"""

            html_content += "    </div>\n"

        html_content += """
</body>
</html>
"""

        with open(self.output_dir / "integration_test_report.html", "w") as f:
            f.write(html_content)


def run_integration_tests():
    """Main function to run all integration tests."""
    runner = IntegrationTestRunner()
    results = runner.run_all_tests()

    # Return exit code based on results
    failed_tests = sum(1 for r in results.values() if not r["success"])
    return 0 if failed_tests == 0 else 1


if __name__ == "__main__":
    exit_code = run_integration_tests()
    sys.exit(exit_code)
