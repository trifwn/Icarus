#!/usr/bin/env python3
"""
Comprehensive Test Runner for ICARUS CLI

This script demonstrates the complete testing framework capabilities
including unit tests, integration tests, end-to-end tests, and performance benchmarking.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add CLI directory to path
cli_dir = Path(__file__).parent.parent
sys.path.insert(0, str(cli_dir))

from tests.test_runner import IcarusTestRunner
from tests.test_validator import TestValidator
from tests.coverage_analyzer import TestCoverageAnalyzer
from tests.test_config import TEST_CONFIG


class ComprehensiveTestRunner:
    """Comprehensive test execution with validation and coverage analysis"""

    def __init__(self):
        self.test_runner = IcarusTestRunner()
        self.validator = TestValidator()
        self.coverage_analyzer = TestCoverageAnalyzer()

    async def run_comprehensive_tests(
        self, include_performance: bool = False
    ) -> Dict[str, Any]:
        """Run comprehensive test suite with validation and coverage"""

        print("🚀 ICARUS CLI Comprehensive Test Suite")
        print("=" * 80)
        print(f"Framework Version: 1.0.0")
        print(f"Test Configuration: {TEST_CONFIG.default_timeout}s timeout")
        print("=" * 80)

        results = {
            "validation": None,
            "coverage": None,
            "unit_tests": None,
            "integration_tests": None,
            "e2e_tests": None,
            "performance_tests": None,
            "overall_success": False,
        }

        # Step 1: Validate framework
        print("\n🔍 STEP 1: Framework Validation")
        print("-" * 40)

        try:
            validation_results = await self.validator.validate_all()
            validation_success = self.validator.print_validation_report()
            results["validation"] = validation_results

            if not validation_success:
                print("❌ Framework validation failed - stopping execution")
                return results

        except Exception as e:
            print(f"💥 Validation failed with error: {e}")
            return results

        # Step 2: Coverage analysis
        print("\n📊 STEP 2: Test Coverage Analysis")
        print("-" * 40)

        try:
            coverage_reports = self.coverage_analyzer.analyze_coverage()
            coverage_report = self.coverage_analyzer.generate_coverage_report()
            print(coverage_report)
            results["coverage"] = coverage_reports

        except Exception as e:
            print(f"⚠️  Coverage analysis failed: {e}")

        # Step 3: Unit Tests
        print("\n🧪 STEP 3: Unit Tests")
        print("-" * 40)

        try:
            unit_results = await self.test_runner.run_tests(
                test_types=["unit"], verbose=True
            )
            results["unit_tests"] = unit_results
            self._print_test_results("Unit Tests", unit_results)

        except Exception as e:
            print(f"💥 Unit tests failed: {e}")
            results["unit_tests"] = {"error": str(e)}

        # Step 4: Integration Tests
        print("\n🔗 STEP 4: Integration Tests")
        print("-" * 40)

        try:
            integration_results = await self.test_runner.run_tests(
                test_types=["integration"], verbose=True
            )
            results["integration_tests"] = integration_results
            self._print_test_results("Integration Tests", integration_results)

        except Exception as e:
            print(f"💥 Integration tests failed: {e}")
            results["integration_tests"] = {"error": str(e)}

        # Step 5: End-to-End Tests
        print("\n🎯 STEP 5: End-to-End Tests")
        print("-" * 40)

        try:
            e2e_results = await self.test_runner.run_tests(
                test_types=["e2e"], verbose=True
            )
            results["e2e_tests"] = e2e_results
            self._print_test_results("End-to-End Tests", e2e_results)

        except Exception as e:
            print(f"💥 End-to-end tests failed: {e}")
            results["e2e_tests"] = {"error": str(e)}

        # Step 6: Performance Tests (optional)
        if include_performance:
            print("\n⚡ STEP 6: Performance Tests")
            print("-" * 40)

            try:
                perf_results = await self.test_runner.run_tests(
                    test_types=["performance"], verbose=True
                )
                results["performance_tests"] = perf_results
                self._print_test_results("Performance Tests", perf_results)

                # Generate performance report
                perf_suite = self.test_runner.framework.test_suites.get("performance")
                if perf_suite and hasattr(perf_suite, "generate_performance_report"):
                    perf_report = perf_suite.generate_performance_report()
                    self._print_performance_summary(perf_report)

            except Exception as e:
                print(f"💥 Performance tests failed: {e}")
                results["performance_tests"] = {"error": str(e)}

        # Final Summary
        results["overall_success"] = self._generate_final_summary(results)

        return results

    def _print_test_results(self, test_type: str, results: Dict[str, Any]):
        """Print test results summary"""
        if "error" in results:
            print(f"❌ {test_type} failed with error: {results['error']}")
            return

        total = results.get("total", 0)
        passed = results.get("passed", 0)
        failed = results.get("failed", 0)
        errors = results.get("errors", 0)
        duration = results.get("duration", 0)

        success_rate = (passed / max(total, 1)) * 100

        print(f"📊 {test_type} Results:")
        print(f"   Total: {total}")
        print(f"   ✅ Passed: {passed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   💥 Errors: {errors}")
        print(f"   ⏱️  Duration: {duration:.2f}s")
        print(f"   📈 Success Rate: {success_rate:.1f}%")

        if failed > 0 or errors > 0:
            print(f"   ⚠️  Issues detected in {test_type}")

    def _print_performance_summary(self, perf_report: Dict[str, Any]):
        """Print performance test summary"""
        print("\n⚡ Performance Summary:")
        print("-" * 25)

        if "performance_metrics" in perf_report:
            metrics = perf_report["performance_metrics"]

            # Find slowest tests
            slowest_tests = []
            for test_name, test_metrics in metrics.items():
                exec_time = test_metrics.get("avg_execution_time_ms", 0)
                slowest_tests.append((test_name, exec_time))

            slowest_tests.sort(key=lambda x: x[1], reverse=True)

            print("🐌 Slowest Tests:")
            for test_name, exec_time in slowest_tests[:5]:
                print(f"   {test_name}: {exec_time:.1f}ms")

            # Memory usage
            high_memory_tests = []
            for test_name, test_metrics in metrics.items():
                memory_growth = test_metrics.get("memory_growth_mb", 0)
                if memory_growth > 5:  # More than 5MB growth
                    high_memory_tests.append((test_name, memory_growth))

            if high_memory_tests:
                print("\n🧠 High Memory Usage:")
                for test_name, memory_growth in high_memory_tests:
                    print(f"   {test_name}: +{memory_growth:.1f}MB")

        # Recommendations
        if "recommendations" in perf_report:
            recommendations = perf_report["recommendations"]
            if recommendations:
                print("\n💡 Performance Recommendations:")
                for i, rec in enumerate(recommendations[:5], 1):
                    print(f"   {i}. {rec}")

    def _generate_final_summary(self, results: Dict[str, Any]) -> bool:
        """Generate final test summary"""
        print("\n" + "=" * 80)
        print("🏁 COMPREHENSIVE TEST SUITE COMPLETE")
        print("=" * 80)

        # Validation status
        validation_success = results.get("validation", {}).get("failed", 1) == 0
        print(
            f"🔍 Framework Validation: {'✅ PASSED' if validation_success else '❌ FAILED'}"
        )

        # Coverage status
        if results.get("coverage"):
            coverage_reports = results["coverage"]
            avg_coverage = sum(
                r.coverage_percentage for r in coverage_reports.values()
            ) / max(len(coverage_reports), 1)
            coverage_status = "✅ GOOD" if avg_coverage >= 70 else "⚠️ NEEDS IMPROVEMENT"
            print(f"📊 Test Coverage: {coverage_status} ({avg_coverage:.1f}%)")

        # Test results
        test_types = [
            "unit_tests",
            "integration_tests",
            "e2e_tests",
            "performance_tests",
        ]
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_errors = 0

        for test_type in test_types:
            test_results = results.get(test_type)
            if test_results and "error" not in test_results:
                total_tests += test_results.get("total", 0)
                total_passed += test_results.get("passed", 0)
                total_failed += test_results.get("failed", 0)
                total_errors += test_results.get("errors", 0)

                type_name = test_type.replace("_", " ").title()
                success_rate = (
                    test_results.get("passed", 0) / max(test_results.get("total", 1), 1)
                ) * 100
                status = (
                    "✅ PASSED"
                    if test_results.get("failed", 0) == 0
                    and test_results.get("errors", 0) == 0
                    else "❌ FAILED"
                )
                print(f"🧪 {type_name}: {status} ({success_rate:.1f}%)")

        # Overall statistics
        print(f"\n📈 Overall Statistics:")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {total_passed}")
        print(f"   ❌ Failed: {total_failed}")
        print(f"   💥 Errors: {total_errors}")

        overall_success_rate = (total_passed / max(total_tests, 1)) * 100
        print(f"   📊 Success Rate: {overall_success_rate:.1f}%")

        # Final verdict
        overall_success = (
            validation_success
            and total_failed == 0
            and total_errors == 0
            and total_tests > 0
        )

        if overall_success:
            print("\n🎉 ALL TESTS PASSED - FRAMEWORK IS READY!")
            print("✨ The ICARUS CLI testing framework is comprehensive and functional")
        else:
            print("\n⚠️  ISSUES DETECTED - PLEASE REVIEW")
            if not validation_success:
                print("   - Framework validation failed")
            if total_failed > 0:
                print(f"   - {total_failed} tests failed")
            if total_errors > 0:
                print(f"   - {total_errors} test errors")
            if total_tests == 0:
                print("   - No tests were executed")

        return overall_success

    async def run_quick_validation(self) -> bool:
        """Run quick validation check"""
        print("🔍 Quick Framework Validation")
        print("-" * 30)

        try:
            # Environment validation
            env_valid = await self.test_runner.validate_test_environment()
            if not env_valid:
                print("❌ Environment validation failed")
                return False

            # Framework validation
            await self.validator.validate_all()
            framework_valid = self.validator.print_validation_report()

            return framework_valid

        except Exception as e:
            print(f"💥 Validation failed: {e}")
            return False


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="ICARUS CLI Comprehensive Test Runner")
    parser.add_argument(
        "--quick", action="store_true", help="Run quick validation only"
    )
    parser.add_argument(
        "--include-performance", action="store_true", help="Include performance tests"
    )
    parser.add_argument(
        "--coverage-only", action="store_true", help="Run coverage analysis only"
    )

    args = parser.parse_args()

    runner = ComprehensiveTestRunner()

    try:
        if args.quick:
            success = await runner.run_quick_validation()
            return 0 if success else 1

        elif args.coverage_only:
            print("📊 Running Coverage Analysis Only")
            print("-" * 40)
            coverage_reports = runner.coverage_analyzer.analyze_coverage()
            print(runner.coverage_analyzer.generate_coverage_report())
            runner.coverage_analyzer.save_coverage_report()
            return 0

        else:
            results = await runner.run_comprehensive_tests(
                include_performance=args.include_performance
            )
            return 0 if results["overall_success"] else 1

    except KeyboardInterrupt:
        print("\n⏹️  Test execution interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
