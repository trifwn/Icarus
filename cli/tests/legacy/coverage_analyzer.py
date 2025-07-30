"""
Test Coverage Analyzer for ICARUS CLI

This module provides comprehensive test coverage analysis and reporting
for the ICARUS CLI testing framework.
"""

import ast
import importlib
import inspect
import sys
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass


@dataclass
class CoverageReport:
    """Test coverage report"""

    module_name: str
    total_functions: int
    tested_functions: int
    total_classes: int
    tested_classes: int
    coverage_percentage: float
    untested_functions: List[str]
    untested_classes: List[str]
    recommendations: List[str]


class TestCoverageAnalyzer:
    """Analyzes test coverage for ICARUS CLI components"""

    def __init__(self):
        self.cli_root = Path("cli")
        self.test_root = Path("cli/testing")
        self.coverage_reports: Dict[str, CoverageReport] = {}

    def analyze_coverage(self) -> Dict[str, CoverageReport]:
        """Analyze test coverage for all CLI modules"""
        print("🔍 Analyzing test coverage...")

        # Get all CLI modules
        cli_modules = self._discover_cli_modules()

        # Analyze each module
        for module_name in cli_modules:
            try:
                report = self._analyze_module_coverage(module_name)
                self.coverage_reports[module_name] = report
            except Exception as e:
                print(f"⚠️  Failed to analyze {module_name}: {e}")

        return self.coverage_reports

    def _discover_cli_modules(self) -> List[str]:
        """Discover all CLI modules to analyze"""
        modules = []

        # Core modules to analyze
        core_modules = [
            "cli.app.main_app",
            "cli.app.screen_manager",
            "cli.app.event_system",
            "cli.app.state_manager",
            "cli.core.config",
            "cli.core.services",
            "cli.core.workflow",
            "cli.integration.analysis_service",
            "cli.integration.solver_manager",
            "cli.integration.parameter_validator",
            "cli.integration.result_processor",
            "cli.data.database",
            "cli.data.import_export",
            "cli.plugins.manager",
            "cli.plugins.api",
            "cli.collaboration.collaboration_manager",
            "cli.visualization.visualization_manager",
        ]

        # Check which modules actually exist
        for module_name in core_modules:
            try:
                module_path = Path(module_name.replace(".", "/") + ".py")
                if module_path.exists():
                    modules.append(module_name)
            except Exception:
                pass

        return modules

    def _analyze_module_coverage(self, module_name: str) -> CoverageReport:
        """Analyze test coverage for a specific module"""
        # Get module functions and classes
        module_functions, module_classes = self._get_module_components(module_name)

        # Get tested functions and classes
        tested_functions, tested_classes = self._get_tested_components(module_name)

        # Calculate coverage
        total_functions = len(module_functions)
        tested_function_count = len(tested_functions)
        total_classes = len(module_classes)
        tested_class_count = len(tested_classes)

        total_components = total_functions + total_classes
        tested_components = tested_function_count + tested_class_count

        coverage_percentage = (tested_components / max(total_components, 1)) * 100

        # Find untested components
        untested_functions = [f for f in module_functions if f not in tested_functions]
        untested_classes = [c for c in module_classes if c not in tested_classes]

        # Generate recommendations
        recommendations = self._generate_recommendations(
            module_name, untested_functions, untested_classes, coverage_percentage
        )

        return CoverageReport(
            module_name=module_name,
            total_functions=total_functions,
            tested_functions=tested_function_count,
            total_classes=total_classes,
            tested_classes=tested_class_count,
            coverage_percentage=coverage_percentage,
            untested_functions=untested_functions,
            untested_classes=untested_classes,
            recommendations=recommendations,
        )

    def _get_module_components(self, module_name: str) -> tuple[List[str], List[str]]:
        """Get functions and classes from a module"""
        functions = []
        classes = []

        try:
            # Try to import the module
            module = importlib.import_module(module_name)

            # Get all members
            for name, obj in inspect.getmembers(module):
                if not name.startswith("_"):  # Skip private members
                    if inspect.isfunction(obj) and obj.__module__ == module_name:
                        functions.append(name)
                    elif inspect.isclass(obj) and obj.__module__ == module_name:
                        classes.append(name)

        except ImportError:
            # If import fails, try to parse the file directly
            module_path = Path(module_name.replace(".", "/") + ".py")
            if module_path.exists():
                functions, classes = self._parse_file_components(module_path)

        return functions, classes

    def _parse_file_components(self, file_path: Path) -> tuple[List[str], List[str]]:
        """Parse functions and classes from a Python file"""
        functions = []
        classes = []

        try:
            with open(file_path, "r") as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                    classes.append(node.name)

        except Exception as e:
            print(f"⚠️  Failed to parse {file_path}: {e}")

        return functions, classes

    def _get_tested_components(self, module_name: str) -> tuple[Set[str], Set[str]]:
        """Get functions and classes that are tested"""
        tested_functions = set()
        tested_classes = set()

        # Look for test methods that reference the module
        test_files = [
            self.test_root / "unit_tests.py",
            self.test_root / "integration_tests.py",
            self.test_root / "e2e_tests.py",
        ]

        for test_file in test_files:
            if test_file.exists():
                functions, classes = self._find_tested_components_in_file(
                    test_file, module_name
                )
                tested_functions.update(functions)
                tested_classes.update(classes)

        return tested_functions, tested_classes

    def _find_tested_components_in_file(
        self, test_file: Path, module_name: str
    ) -> tuple[Set[str], Set[str]]:
        """Find tested components in a test file"""
        tested_functions = set()
        tested_classes = set()

        try:
            with open(test_file, "r") as f:
                content = f.read()

            # Look for import statements and usage patterns
            lines = content.split("\n")
            for line in lines:
                line = line.strip()

                # Check for imports from the module
                if f"from {module_name}" in line:
                    # Extract imported names
                    if " import " in line:
                        import_part = line.split(" import ")[1]
                        names = [name.strip() for name in import_part.split(",")]
                        for name in names:
                            # Simple heuristic: uppercase names are likely classes
                            if name[0].isupper():
                                tested_classes.add(name)
                            else:
                                tested_functions.add(name)

                # Look for direct usage patterns
                module_short = module_name.split(".")[-1]
                if f"{module_short}." in line or f"test_{module_short}" in line:
                    # This is a rough heuristic - could be improved
                    pass

        except Exception as e:
            print(f"⚠️  Failed to analyze {test_file}: {e}")

        return tested_functions, tested_classes

    def _generate_recommendations(
        self,
        module_name: str,
        untested_functions: List[str],
        untested_classes: List[str],
        coverage_percentage: float,
    ) -> List[str]:
        """Generate recommendations for improving test coverage"""
        recommendations = []

        if coverage_percentage < 50:
            recommendations.append(
                f"Critical: {module_name} has very low test coverage ({coverage_percentage:.1f}%)"
            )
        elif coverage_percentage < 80:
            recommendations.append(
                f"Improve: {module_name} needs better test coverage ({coverage_percentage:.1f}%)"
            )

        if untested_functions:
            recommendations.append(
                f"Add unit tests for functions: {', '.join(untested_functions[:5])}"
                + ("..." if len(untested_functions) > 5 else "")
            )

        if untested_classes:
            recommendations.append(
                f"Add unit tests for classes: {', '.join(untested_classes[:3])}"
                + ("..." if len(untested_classes) > 3 else "")
            )

        return recommendations

    def generate_coverage_report(self) -> str:
        """Generate a comprehensive coverage report"""
        if not self.coverage_reports:
            self.analyze_coverage()

        report = []
        report.append("📊 ICARUS CLI Test Coverage Report")
        report.append("=" * 50)

        # Overall statistics
        total_modules = len(self.coverage_reports)
        total_coverage = sum(
            r.coverage_percentage for r in self.coverage_reports.values()
        )
        avg_coverage = total_coverage / max(total_modules, 1)

        report.append(f"📈 Overall Coverage: {avg_coverage:.1f}%")
        report.append(f"📋 Modules Analyzed: {total_modules}")
        report.append("")

        # Module-by-module breakdown
        report.append("📋 Module Coverage:")
        report.append("-" * 30)

        # Sort by coverage percentage
        sorted_reports = sorted(
            self.coverage_reports.items(),
            key=lambda x: x[1].coverage_percentage,
            reverse=True,
        )

        for module_name, coverage_report in sorted_reports:
            status = (
                "✅"
                if coverage_report.coverage_percentage >= 80
                else "⚠️"
                if coverage_report.coverage_percentage >= 50
                else "❌"
            )
            report.append(
                f"{status} {module_name}: {coverage_report.coverage_percentage:.1f}% "
                f"({coverage_report.tested_functions + coverage_report.tested_classes}/"
                f"{coverage_report.total_functions + coverage_report.total_classes})"
            )

        # Recommendations
        report.append("")
        report.append("💡 Recommendations:")
        report.append("-" * 20)

        all_recommendations = []
        for coverage_report in self.coverage_reports.values():
            all_recommendations.extend(coverage_report.recommendations)

        for i, recommendation in enumerate(all_recommendations[:10], 1):
            report.append(f"{i}. {recommendation}")

        if len(all_recommendations) > 10:
            report.append(
                f"... and {len(all_recommendations) - 10} more recommendations"
            )

        return "\n".join(report)

    def save_coverage_report(self, output_path: Optional[Path] = None):
        """Save coverage report to file"""
        if output_path is None:
            output_path = self.test_root / "reports" / "coverage_report.txt"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        report_content = self.generate_coverage_report()
        with open(output_path, "w") as f:
            f.write(report_content)

        print(f"📄 Coverage report saved to: {output_path}")

    def get_low_coverage_modules(self, threshold: float = 50.0) -> List[str]:
        """Get modules with coverage below threshold"""
        if not self.coverage_reports:
            self.analyze_coverage()

        low_coverage = []
        for module_name, report in self.coverage_reports.items():
            if report.coverage_percentage < threshold:
                low_coverage.append(module_name)

        return low_coverage


if __name__ == "__main__":
    # Run coverage analysis
    analyzer = TestCoverageAnalyzer()
    analyzer.analyze_coverage()

    # Print report
    print(analyzer.generate_coverage_report())

    # Save report
    analyzer.save_coverage_report()
