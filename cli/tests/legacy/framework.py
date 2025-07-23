"""
Main Testing Framework for ICARUS CLI

This module provides the core testing framework that orchestrates all test types
and provides comprehensive test reporting and management.
"""

import json
import time
import traceback
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional


class TestStatus(Enum):
    """Test execution status"""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestType(Enum):
    """Types of tests"""

    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "end_to_end"
    PERFORMANCE = "performance"


@dataclass
class TestResult:
    """Individual test result"""

    name: str
    test_type: TestType
    status: TestStatus
    duration: float = 0.0
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TestSuiteResult:
    """Test suite result summary"""

    suite_name: str
    test_type: TestType
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration: float = 0.0
    results: List[TestResult] = field(default_factory=list)


class TestFramework:
    """Main testing framework coordinator"""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("cli/testing/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.test_suites: Dict[TestType, Any] = {}
        self.results: List[TestSuiteResult] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def register_test_suite(self, test_type: TestType, suite_instance):
        """Register a test suite with the framework"""
        self.test_suites[test_type] = suite_instance

    async def run_all_tests(
        self,
        test_types: Optional[List[TestType]] = None,
    ) -> Dict[str, Any]:
        """Run all registered test suites"""
        if test_types is None:
            test_types = list(TestType)

        self.start_time = datetime.now()
        print(f"🚀 Starting ICARUS CLI Test Suite at {self.start_time}")
        print("=" * 80)

        total_results = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "total": 0,
            "duration": 0.0,
        }

        for test_type in test_types:
            if test_type in self.test_suites:
                print(f"\n📋 Running {test_type.value.upper()} tests...")
                suite_result = await self._run_test_suite(test_type)
                self.results.append(suite_result)

                # Update totals
                total_results["passed"] += suite_result.passed
                total_results["failed"] += suite_result.failed
                total_results["skipped"] += suite_result.skipped
                total_results["errors"] += suite_result.errors
                total_results["total"] += suite_result.total_tests
                total_results["duration"] += suite_result.duration

        self.end_time = datetime.now()

        # Generate comprehensive report
        await self._generate_reports(total_results)

        return total_results

    async def _run_test_suite(self, test_type: TestType) -> TestSuiteResult:
        """Run a specific test suite"""
        suite = self.test_suites[test_type]
        suite_start = time.time()

        try:
            if hasattr(suite, "run_all_tests"):
                results = await suite.run_all_tests()
            else:
                results = []

            suite_duration = time.time() - suite_start

            # Process results
            suite_result = TestSuiteResult(
                suite_name=f"{test_type.value}_tests",
                test_type=test_type,
                duration=suite_duration,
            )

            for result in results:
                suite_result.results.append(result)
                suite_result.total_tests += 1

                if result.status == TestStatus.PASSED:
                    suite_result.passed += 1
                elif result.status == TestStatus.FAILED:
                    suite_result.failed += 1
                elif result.status == TestStatus.SKIPPED:
                    suite_result.skipped += 1
                elif result.status == TestStatus.ERROR:
                    suite_result.errors += 1

            self._print_suite_summary(suite_result)
            return suite_result

        except Exception as e:
            print(f"❌ Test suite {test_type.value} crashed: {e}")
            traceback.print_exc()

            return TestSuiteResult(
                suite_name=f"{test_type.value}_tests",
                test_type=test_type,
                duration=time.time() - suite_start,
                errors=1,
                total_tests=1,
            )

    def _print_suite_summary(self, result: TestSuiteResult):
        """Print test suite summary"""
        print(f"\n📊 {result.suite_name.upper()} SUMMARY:")
        print(f"   Total: {result.total_tests}")
        print(f"   ✅ Passed: {result.passed}")
        print(f"   ❌ Failed: {result.failed}")
        print(f"   ⏭️  Skipped: {result.skipped}")
        print(f"   💥 Errors: {result.errors}")
        print(f"   ⏱️  Duration: {result.duration:.2f}s")

        if result.failed > 0 or result.errors > 0:
            print("   🔍 Failed/Error tests:")
            for test_result in result.results:
                if test_result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                    print(f"      - {test_result.name}: {test_result.error_message}")

    async def _generate_reports(self, total_results: Dict[str, Any]):
        """Generate comprehensive test reports"""
        # Console summary
        self._print_final_summary(total_results)

        # JSON report
        await self._generate_json_report(total_results)

        # HTML report
        await self._generate_html_report(total_results)

        # JUnit XML report (for CI/CD)
        await self._generate_junit_report()

    def _print_final_summary(self, total_results: Dict[str, Any]):
        """Print final test summary"""
        duration = (self.end_time - self.start_time).total_seconds()

        print("\n" + "=" * 80)
        print("🏁 ICARUS CLI TEST SUITE COMPLETE")
        print("=" * 80)
        print(f"⏱️  Total Duration: {duration:.2f}s")
        print(f"📊 Total Tests: {total_results['total']}")
        print(f"✅ Passed: {total_results['passed']}")
        print(f"❌ Failed: {total_results['failed']}")
        print(f"⏭️  Skipped: {total_results['skipped']}")
        print(f"💥 Errors: {total_results['errors']}")

        success_rate = (total_results["passed"] / max(total_results["total"], 1)) * 100
        print(f"📈 Success Rate: {success_rate:.1f}%")

        if total_results["failed"] == 0 and total_results["errors"] == 0:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("⚠️  SOME TESTS FAILED - Check reports for details")

    async def _generate_json_report(self, total_results: Dict[str, Any]):
        """Generate JSON test report"""
        report = {
            "summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "duration": (self.end_time - self.start_time).total_seconds(),
                **total_results,
            },
            "test_suites": [],
        }

        for suite_result in self.results:
            suite_data = {
                "name": suite_result.suite_name,
                "type": suite_result.test_type.value,
                "summary": {
                    "total": suite_result.total_tests,
                    "passed": suite_result.passed,
                    "failed": suite_result.failed,
                    "skipped": suite_result.skipped,
                    "errors": suite_result.errors,
                    "duration": suite_result.duration,
                },
                "tests": [],
            }

            for test_result in suite_result.results:
                test_data = {
                    "name": test_result.name,
                    "status": test_result.status.value,
                    "duration": test_result.duration,
                    "timestamp": test_result.timestamp.isoformat(),
                    "error_message": test_result.error_message,
                    "details": test_result.details,
                }
                suite_data["tests"].append(test_data)

            report["test_suites"].append(suite_data)

        # Save JSON report
        report_file = (
            self.output_dir
            / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"📄 JSON report saved: {report_file}")

    async def _generate_html_report(self, total_results: Dict[str, Any]):
        """Generate HTML test report"""
        html_content = self._create_html_report_content(total_results)

        report_file = (
            self.output_dir
            / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        with open(report_file, "w") as f:
            f.write(html_content)

        print(f"🌐 HTML report saved: {report_file}")

    def _create_html_report_content(self, total_results: Dict[str, Any]) -> str:
        """Create HTML report content"""
        duration = (self.end_time - self.start_time).total_seconds()
        success_rate = (total_results["passed"] / max(total_results["total"], 1)) * 100

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ICARUS CLI Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #1976d2; color: white; padding: 20px; border-radius: 5px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .metric {{ background: #f5f5f5; padding: 15px; border-radius: 5px; text-align: center; }}
        .passed {{ background: #4caf50; color: white; }}
        .failed {{ background: #f44336; color: white; }}
        .suite {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
        .suite-header {{ background: #e3f2fd; padding: 10px; font-weight: bold; }}
        .test-item {{ padding: 8px; border-bottom: 1px solid #eee; }}
        .test-passed {{ background: #e8f5e8; }}
        .test-failed {{ background: #ffebee; }}
        .test-error {{ background: #fff3e0; }}
        .test-skipped {{ background: #f3e5f5; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 ICARUS CLI Test Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="summary">
        <div class="metric">
            <h3>Total Tests</h3>
            <div style="font-size: 24px;">{total_results['total']}</div>
        </div>
        <div class="metric passed">
            <h3>Passed</h3>
            <div style="font-size: 24px;">{total_results['passed']}</div>
        </div>
        <div class="metric failed">
            <h3>Failed</h3>
            <div style="font-size: 24px;">{total_results['failed']}</div>
        </div>
        <div class="metric">
            <h3>Success Rate</h3>
            <div style="font-size: 24px;">{success_rate:.1f}%</div>
        </div>
        <div class="metric">
            <h3>Duration</h3>
            <div style="font-size: 24px;">{duration:.1f}s</div>
        </div>
    </div>
"""

        # Add test suites
        for suite_result in self.results:
            html += f"""
    <div class="suite">
        <div class="suite-header">
            {suite_result.suite_name.upper()} - {suite_result.passed}/{suite_result.total_tests} passed
        </div>
"""
            for test_result in suite_result.results:
                status_class = f"test-{test_result.status.value.replace('_', '-')}"
                status_icon = {
                    TestStatus.PASSED: "✅",
                    TestStatus.FAILED: "❌",
                    TestStatus.ERROR: "💥",
                    TestStatus.SKIPPED: "⏭️",
                }.get(test_result.status, "❓")

                html += f"""
        <div class="test-item {status_class}">
            {status_icon} {test_result.name} ({test_result.duration:.3f}s)
"""
                if test_result.error_message:
                    html += f"<br><small>Error: {test_result.error_message}</small>"

                html += "</div>"

            html += "</div>"

        html += """
</body>
</html>"""

        return html

    async def _generate_junit_report(self):
        """Generate JUnit XML report for CI/CD integration"""
        xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
        xml_content += "<testsuites>\n"

        for suite_result in self.results:
            xml_content += f'  <testsuite name="{suite_result.suite_name}" '
            xml_content += f'tests="{suite_result.total_tests}" '
            xml_content += f'failures="{suite_result.failed}" '
            xml_content += f'errors="{suite_result.errors}" '
            xml_content += f'skipped="{suite_result.skipped}" '
            xml_content += f'time="{suite_result.duration:.3f}">\n'

            for test_result in suite_result.results:
                xml_content += f'    <testcase name="{test_result.name}" '
                xml_content += f'time="{test_result.duration:.3f}"'

                if test_result.status == TestStatus.FAILED:
                    xml_content += ">\n"
                    xml_content += f'      <failure message="{test_result.error_message or "Test failed"}"/>\n'
                    xml_content += "    </testcase>\n"
                elif test_result.status == TestStatus.ERROR:
                    xml_content += ">\n"
                    xml_content += f'      <error message="{test_result.error_message or "Test error"}"/>\n'
                    xml_content += "    </testcase>\n"
                elif test_result.status == TestStatus.SKIPPED:
                    xml_content += ">\n"
                    xml_content += "      <skipped/>\n"
                    xml_content += "    </testcase>\n"
                else:
                    xml_content += "/>\n"

            xml_content += "  </testsuite>\n"

        xml_content += "</testsuites>\n"

        # Save JUnit report
        report_file = (
            self.output_dir
            / f"junit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml"
        )
        with open(report_file, "w") as f:
            f.write(xml_content)

        print(f"📋 JUnit XML report saved: {report_file}")
