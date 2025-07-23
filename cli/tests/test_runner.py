"""
Comprehensive Test Runner for ICARUS CLI

This module provides a unified test runner that orchestrates all test types
and provides comprehensive reporting and CI/CD integration.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

# Add CLI directory to path for imports
cli_dir = Path(__file__).parent.parent
sys.path.insert(0, str(cli_dir))

from .e2e_tests import EndToEndTestSuite
from .framework import TestFramework
from .framework import TestType
from .integration_tests import IntegrationTestSuite
from .performance_tests import PerformanceTestSuite
from .unit_tests import UnitTestSuite


class IcarusTestRunner:
    """Comprehensive test runner for ICARUS CLI"""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("cli/testing/reports")
        self.framework = TestFramework(self.output_dir)

        # Register test suites
        self.framework.register_test_suite(TestType.UNIT, UnitTestSuite())
        self.framework.register_test_suite(TestType.INTEGRATION, IntegrationTestSuite())
        self.framework.register_test_suite(TestType.E2E, EndToEndTestSuite())
        self.framework.register_test_suite(TestType.PERFORMANCE, PerformanceTestSuite())

    async def run_tests(
        self,
        test_types: Optional[List[str]] = None,
        include_performance: bool = False,
        verbose: bool = False,
        fail_fast: bool = False,
    ) -> Dict:
        """Run specified test types"""

        # Convert string test types to enum
        if test_types:
            type_mapping = {
                "unit": TestType.UNIT,
                "integration": TestType.INTEGRATION,
                "e2e": TestType.E2E,
                "performance": TestType.PERFORMANCE,
            }

            selected_types = []
            for test_type in test_types:
                if test_type.lower() in type_mapping:
                    selected_types.append(type_mapping[test_type.lower()])
                else:
                    print(f"⚠️  Unknown test type: {test_type}")
        else:
            # Default test types (exclude performance unless explicitly requested)
            selected_types = [TestType.UNIT, TestType.INTEGRATION, TestType.E2E]
            if include_performance:
                selected_types.append(TestType.PERFORMANCE)

        if verbose:
            print(f"🎯 Running test types: {[t.value for t in selected_types]}")

        # Run tests
        results = await self.framework.run_all_tests(selected_types)

        # Check for early exit on failures
        if fail_fast and (results["failed"] > 0 or results["errors"] > 0):
            print("💥 Stopping due to test failures (--fail-fast enabled)")
            return results

        # Generate performance report if performance tests were run
        if TestType.PERFORMANCE in selected_types:
            perf_suite = self.framework.test_suites[TestType.PERFORMANCE]
            if hasattr(perf_suite, "generate_performance_report"):
                perf_report = perf_suite.generate_performance_report()
                await self._save_performance_report(perf_report)

        return results

    async def _save_performance_report(self, report: Dict):
        """Save performance report"""
        report_file = self.output_dir / "performance_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"📊 Performance report saved: {report_file}")

    async def run_specific_test(self, test_name: str, test_type: str = "unit"):
        """Run a specific test by name"""
        type_mapping = {
            "unit": TestType.UNIT,
            "integration": TestType.INTEGRATION,
            "e2e": TestType.E2E,
            "performance": TestType.PERFORMANCE,
        }

        if test_type.lower() not in type_mapping:
            print(f"❌ Invalid test type: {test_type}")
            return False

        selected_type = type_mapping[test_type.lower()]

        if selected_type not in self.framework.test_suites:
            print(f"❌ Test suite not available: {test_type}")
            return False

        print(f"🎯 Running specific test: {test_name} ({test_type})")

        # This would require modification to test suites to support running individual tests
        # For now, we'll run the full suite and filter results
        results = await self.framework.run_all_tests([selected_type])

        # Find and display specific test result
        for suite_result in self.framework.results:
            for test_result in suite_result.results:
                if test_name.lower() in test_result.name.lower():
                    print(f"📋 Test Result: {test_result.name}")
                    print(f"   Status: {test_result.status.value}")
                    print(f"   Duration: {test_result.duration:.3f}s")
                    if test_result.error_message:
                        print(f"   Error: {test_result.error_message}")
                    return test_result.status.value == "passed"

        print(f"❌ Test not found: {test_name}")
        return False

    def list_available_tests(self):
        """List all available tests"""
        print("📋 Available Test Suites:")
        print("=" * 50)

        test_info = {
            "unit": "Unit tests for individual components",
            "integration": "Integration tests for component interactions",
            "e2e": "End-to-end tests for complete workflows",
            "performance": "Performance and benchmarking tests",
        }

        for test_type, description in test_info.items():
            print(f"🔹 {test_type.upper()}: {description}")

        print("\nExample usage:")
        print("  python -m cli.testing.test_runner --type unit")
        print("  python -m cli.testing.test_runner --type integration --verbose")
        print("  python -m cli.testing.test_runner --include-performance")

    async def validate_test_environment(self) -> bool:
        """Validate that the test environment is properly set up"""
        print("🔍 Validating test environment...")

        validation_results = []

        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            validation_results.append(
                (
                    "Python Version",
                    True,
                    f"{python_version.major}.{python_version.minor}",
                ),
            )
        else:
            validation_results.append(
                (
                    "Python Version",
                    False,
                    f"Requires Python 3.8+, got {python_version.major}.{python_version.minor}",
                ),
            )

        # Check required directories
        required_dirs = [
            "cli/app",
            "cli/core",
            "cli/integration",
            "cli/data",
            "cli/plugins",
            "cli/collaboration",
        ]

        for dir_path in required_dirs:
            path = Path(dir_path)
            validation_results.append(
                (f"Directory {dir_path}", path.exists(), str(path.absolute())),
            )

        # Check for test output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        validation_results.append(
            (
                "Test Output Directory",
                self.output_dir.exists(),
                str(self.output_dir.absolute()),
            ),
        )

        # Print validation results
        all_valid = True
        for check_name, is_valid, details in validation_results:
            status = "✅" if is_valid else "❌"
            print(f"  {status} {check_name}: {details}")
            if not is_valid:
                all_valid = False

        if all_valid:
            print("✅ Test environment validation passed")
        else:
            print("❌ Test environment validation failed")

        return all_valid

    async def generate_ci_report(self, results: Dict) -> bool:
        """Generate CI/CD compatible report"""
        # Generate JUnit XML (already handled by framework)

        # Generate GitHub Actions summary
        if "GITHUB_ACTIONS" in os.environ:
            await self._generate_github_summary(results)

        # Generate simple exit code for CI
        return results["failed"] == 0 and results["errors"] == 0

    async def _generate_github_summary(self, results: Dict):
        """Generate GitHub Actions job summary"""
        summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
        if not summary_file:
            return

        summary_content = f"""
# 🚀 ICARUS CLI Test Results

## Summary
- **Total Tests**: {results["total"]}
- **✅ Passed**: {results["passed"]}
- **❌ Failed**: {results["failed"]}
- **💥 Errors**: {results["errors"]}
- **⏭️ Skipped**: {results["skipped"]}

## Success Rate
{(results["passed"] / max(results["total"], 1)) * 100:.1f}%

## Duration
{results["duration"]:.2f} seconds
"""

        if results["failed"] > 0 or results["errors"] > 0:
            summary_content += (
                "\n## ⚠️ Issues Found\nCheck the detailed logs for more information."
            )

        with open(summary_file, "w") as f:
            f.write(summary_content)


async def main():
    """Main entry point for test runner"""
    parser = argparse.ArgumentParser(description="ICARUS CLI Test Runner")

    parser.add_argument(
        "--type",
        choices=["unit", "integration", "e2e", "performance"],
        action="append",
        help="Test types to run (can be specified multiple times)",
    )

    parser.add_argument(
        "--include-performance",
        action="store_true",
        help="Include performance tests (slow)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for test reports",
    )

    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List available test suites",
    )

    parser.add_argument(
        "--validate-env",
        action="store_true",
        help="Validate test environment",
    )

    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI/CD mode - generate CI-compatible reports",
    )

    args = parser.parse_args()

    # Create test runner
    runner = IcarusTestRunner(args.output_dir)

    # Handle list tests
    if args.list_tests:
        runner.list_available_tests()
        return

    # Handle environment validation
    if args.validate_env:
        valid = await runner.validate_test_environment()
        sys.exit(0 if valid else 1)

    # Validate environment before running tests
    if not await runner.validate_test_environment():
        print("❌ Environment validation failed. Use --validate-env for details.")
        sys.exit(1)

    # Run tests
    try:
        results = await runner.run_tests(
            test_types=args.type,
            include_performance=args.include_performance,
            verbose=args.verbose,
            fail_fast=args.fail_fast,
        )

        # Handle CI mode
        if args.ci:
            success = await runner.generate_ci_report(results)
            sys.exit(0 if success else 1)
        else:
            # Exit with error code if tests failed
            sys.exit(0 if results["failed"] == 0 and results["errors"] == 0 else 1)

    except KeyboardInterrupt:
        print("\n⏹️  Test run interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"💥 Test runner crashed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Import os here since it's only needed for CI functionality
    import os

    asyncio.run(main())
