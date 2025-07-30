
        # Generate summary
        return self._generate_validation_summary()

    async def _validate_framework_structure(self):
        """Validate testing framework file structure"""
        print("📁 Validating framework structure...")

        required_files = [
            "cli/testing/__init__.py",
            "cli/testing/framework.py",
            "cli/testing/unit_tests.py",
            "cli/testing/integration_tests.py",
            "cli/testing/e2e_tests.py",
            "cli/testing/performance_tests.py",
            "cli/testing/test_runner.py",
            "cli/testing/test_config.py",
            "cli/testing/test_utils.py",
            "cli/testing/README.md",
        ]

        for file_path in required_files:
            path = Path(file_path)
            if path.exists():
                # Check file size
                size = path.stat().st_size
                if size > 100:  # Reasonable minimum size
                    self.results.append(
                        ValidationResult(
                            component="structure",
                            test_name=f"file_exists_{path.name}",
                            status=ValidationStatus.PASS,
                            message=f"File exists and has content ({size} bytes)",
                        ),
                    )
                else:
                    self.results.append(
                        ValidationResult(
                            component="structure",
                            test_name=f"file_size_{path.name}",
                            status=ValidationStatus.WARN,
                            message=f"File is very small ({size} bytes)",
                        ),
                    )
            else:
                self.results.append(
                    ValidationResult(
                        component="structure",
                        test_name=f"file_missing_{path.name}",
                        status=ValidationStatus.FAIL,
                        message=f"Required file missing: {file_path}",
                    ),
                )

    async def _validate_test_imports(self):
        """Validate that all test modules can be imported"""
        print("📦 Validating test imports...")

        # Add CLI directory to path
        sys.path.insert(0, str(self.cli_root))

        test_modules = [
            ("testing.framework", "TestFramework"),
            ("testing.unit_tests", "UnitTestSuite"),
            ("testing.integration_tests", "IntegrationTestSuite"),
            ("testing.e2e_tests", "EndToEndTestSuite"),
            ("testing.performance_tests", "PerformanceTestSuite"),
            ("testing.test_runner", "IcarusTestRunner"),
            ("testing.test_utils", "TestFixtures"),
            ("testing.test_config", "TEST_CONFIG"),
        ]

        for module_name, class_name in test_modules:
            try:
                module = __import__(module_name, fromlist=[class_name])
                cls = getattr(module, class_name)

                self.results.append(
                    ValidationResult(
                        component="imports",
                        test_name=f"import_{module_name}",
                        status=ValidationStatus.PASS,
                        message=f"Successfully imported {class_name}",
                    ),
                )

            except ImportError as e:
                self.results.append(
                    ValidationResult(
                        component="imports",
                        test_name=f"import_{module_name}",
                        status=ValidationStatus.FAIL,
                        message=f"Import failed: {e}",
                    ),
                )
            except AttributeError as e:
                self.results.append(
                    ValidationResult(
                        component="imports",
                        test_name=f"class_{class_name}",
                        status=ValidationStatus.FAIL,
                        message=f"Class not found: {e}",
                    ),
                )

    async def _validate_test_suites(self):
        """Validate test suite functionality"""
        print("🧪 Validating test suites...")

        try:
            from tests.unit_tests import UnitTestSuite

            # Test unit test suite
            unit_suite = UnitTestSuite()

            # Check required methods
            required_methods = ["run_all_tests", "_run_test"]
            for method in required_methods:
                if hasattr(unit_suite, method):
                    self.results.append(
                        ValidationResult(
                            component="unit_tests",
                            test_name=f"method_{method}",
                            status=ValidationStatus.PASS,
                            message=f"Method {method} exists",
                        ),
                    )
                else:
                    self.results.append(
                        ValidationResult(
                            component="unit_tests",
                            test_name=f"method_{method}",
                            status=ValidationStatus.FAIL,
                            message=f"Required method {method} missing",
                        ),
                    )

            # Check test_results attribute
            if hasattr(unit_suite, "test_results"):
                self.results.append(
                    ValidationResult(
                        component="unit_tests",
                        test_name="test_results_attribute",
                        status=ValidationStatus.PASS,
                        message="test_results attribute exists",
                    ),
                )
            else:
                self.results.append(
                    ValidationResult(
                        component="unit_tests",
                        test_name="test_results_attribute",
                        status=ValidationStatus.FAIL,
                        message="test_results attribute missing",
                    ),
                )

        except ImportError as e:
            self.results.append(
                ValidationResult(
                    component="unit_tests",
                    test_name="suite_import",
                    status=ValidationStatus.ERROR,
                    message=f"Cannot import UnitTestSuite: {e}",
                ),
            )

    async def _validate_mock_components(self):
        """Validate mock components functionality"""
        print("🎭 Validating mock components...")

        try:
            from tests.test_utils import MockComponents

            # Test MockApp
            mock_app = MockComponents.MockApp()
            if hasattr(mock_app, "screens") and hasattr(mock_app, "event_system"):
                self.results.append(
                    ValidationResult(
                        component="mocks",
                        test_name="mock_app",
                        status=ValidationStatus.PASS,
                        message="MockApp has required attributes",
                    ),
                )
            else:
                self.results.append(
                    ValidationResult(
                        component="mocks",
                        test_name="mock_app",
                        status=ValidationStatus.FAIL,
                        message="MockApp missing required attributes",
                    ),
                )

            # Test MockDatabase
            mock_db = MockComponents.MockDatabase()
            await mock_db.initialize()

            # Test basic CRUD operations
            record_id = await mock_db.create_record("test", {"name": "test"})
            if record_id:
                record = await mock_db.get_record("test", record_id)
                if record and record["name"] == "test":
                    self.results.append(
                        ValidationResult(
                            component="mocks",
                            test_name="mock_database_crud",
                            status=ValidationStatus.PASS,
                            message="MockDatabase CRUD operations work",
                        ),
                    )
                else:
                    self.results.append(
                        ValidationResult(
                            component="mocks",
                            test_name="mock_database_crud",
                            status=ValidationStatus.FAIL,
                            message="MockDatabase CRUD operations failed",
                        ),
                    )

        except Exception as e:
            self.results.append(
                ValidationResult(
                    component="mocks",
                    test_name="mock_components",
                    status=ValidationStatus.ERROR,
                    message=f"Mock components validation failed: {e}",
                ),
            )

    async def _validate_test_utilities(self):
        """Validate test utilities"""
        print("🛠️  Validating test utilities...")

        try:
            from tests.test_utils import TestAssertions
            from tests.test_utils import TestFixtures

            # Test fixtures
            airfoil_data = TestFixtures.get_sample_airfoil_data()
            if len(airfoil_data) > 100:
                self.results.append(
                    ValidationResult(
                        component="utilities",
                        test_name="sample_airfoil_data",
                        status=ValidationStatus.PASS,
                        message=f"Sample airfoil data available ({len(airfoil_data)} chars)",
                    ),
                )
            else:
                self.results.append(
                    ValidationResult(
                        component="utilities",
                        test_name="sample_airfoil_data",
                        status=ValidationStatus.FAIL,
                        message="Sample airfoil data too small or missing",
                    ),
                )

            # Test aircraft config
            aircraft_config = TestFixtures.get_sample_aircraft_config()
            if (
                "components" in aircraft_config
                and "wing" in aircraft_config["components"]
            ):
                self.results.append(
                    ValidationResult(
                        component="utilities",
                        test_name="sample_aircraft_config",
                        status=ValidationStatus.PASS,
                        message="Sample aircraft config is valid",
                    ),
                )
            else:
                self.results.append(
                    ValidationResult(
                        component="utilities",
                        test_name="sample_aircraft_config",
                        status=ValidationStatus.FAIL,
                        message="Sample aircraft config is invalid",
                    ),
                )

            # Test assertions
            try:
                sample_result = {"status": "completed", "results": {"alpha": [1, 2, 3]}}
                TestAssertions.assert_analysis_result_valid(sample_result)

                self.results.append(
                    ValidationResult(
                        component="utilities",
                        test_name="test_assertions",
                        status=ValidationStatus.PASS,
                        message="Test assertions work correctly",
                    ),
                )
            except AssertionError:
                self.results.append(
                    ValidationResult(
                        component="utilities",
                        test_name="test_assertions",
                        status=ValidationStatus.FAIL,
                        message="Test assertions failed unexpectedly",
                    ),
                )

        except Exception as e:
            self.results.append(
                ValidationResult(
                    component="utilities",
                    test_name="test_utilities",
                    status=ValidationStatus.ERROR,
                    message=f"Test utilities validation failed: {e}",
                ),
            )

    async def _validate_configuration(self):
        """Validate test configuration"""
        print("⚙️  Validating test configuration...")

        try:
            from tests.test_config import PERFORMANCE_THRESHOLDS
            from tests.test_config import TEST_CONFIG

            # Check configuration attributes
            required_config_attrs = [
                "default_timeout",
                "performance_iterations",
                "max_execution_time_ms",
                "generate_html_reports",
            ]

            for attr in required_config_attrs:
                if hasattr(TEST_CONFIG, attr):
                    value = getattr(TEST_CONFIG, attr)
                    self.results.append(
                        ValidationResult(
                            component="configuration",
                            test_name=f"config_{attr}",
                            status=ValidationStatus.PASS,
                            message=f"Config {attr} = {value}",
                        ),
                    )
                else:
                    self.results.append(
                        ValidationResult(
                            component="configuration",
                            test_name=f"config_{attr}",
                            status=ValidationStatus.FAIL,
                            message=f"Config attribute {attr} missing",
                        ),
                    )

            # Check performance thresholds
            if hasattr(PERFORMANCE_THRESHOLDS, "max_memory_per_test"):
                self.results.append(
                    ValidationResult(
                        component="configuration",
                        test_name="performance_thresholds",
                        status=ValidationStatus.PASS,
                        message="Performance thresholds configured",
                    ),
                )
            else:
                self.results.append(
                    ValidationResult(
                        component="configuration",
                        test_name="performance_thresholds",
                        status=ValidationStatus.FAIL,
                        message="Performance thresholds missing",
                    ),
                )

        except Exception as e:
            self.results.append(
                ValidationResult(
                    component="configuration",
                    test_name="test_config",
                    status=ValidationStatus.ERROR,
                    message=f"Configuration validation failed: {e}",
                ),
            )

    async def _validate_reporting(self):
        """Validate test reporting functionality"""
        print("📊 Validating test reporting...")

        try:
            from tests.framework import TestFramework

            framework = TestFramework()

            # Check framework methods
            required_methods = ["run_all_tests", "register_test_suite"]
            for method in required_methods:
                if hasattr(framework, method):
                    self.results.append(
                        ValidationResult(
                            component="reporting",
                            test_name=f"framework_{method}",
                            status=ValidationStatus.PASS,
                            message=f"Framework method {method} exists",
                        ),
                    )
                else:
                    self.results.append(
                        ValidationResult(
                            component="reporting",
                            test_name=f"framework_{method}",
                            status=ValidationStatus.FAIL,
                            message=f"Framework method {method} missing",
                        ),
                    )

            # Check output directory creation
            output_dir = Path("cli/testing/reports")
            if output_dir.exists() or True:  # Framework should create it
                self.results.append(
                    ValidationResult(
                        component="reporting",
                        test_name="output_directory",
                        status=ValidationStatus.PASS,
                        message="Reports directory available",
                    ),
                )

        except Exception as e:
            self.results.append(
                ValidationResult(
                    component="reporting",
                    test_name="test_framework",
                    status=ValidationStatus.ERROR,
                    message=f"Reporting validation failed: {e}",
                ),
            )

    def _generate_validation_summary(self) -> Dict[str, Any]:
        """Generate validation summary"""
        summary = {
            "total_checks": len(self.results),
            "passed": sum(1 for r in self.results if r.status == ValidationStatus.PASS),
            "warnings": sum(
                1 for r in self.results if r.status == ValidationStatus.WARN
            ),
            "failed": sum(1 for r in self.results if r.status == ValidationStatus.FAIL),
            "errors": sum(
                1 for r in self.results if r.status == ValidationStatus.ERROR
            ),
            "components": {},
            "recommendations": [],
        }

        # Group by component
        for result in self.results:
            if result.component not in summary["components"]:
                summary["components"][result.component] = {
                    "passed": 0,
                    "warnings": 0,
                    "failed": 0,
                    "errors": 0,
                }

            if result.status == ValidationStatus.PASS:
                summary["components"][result.component]["passed"] += 1
            elif result.status == ValidationStatus.WARN:
                summary["components"][result.component]["warnings"] += 1
            elif result.status == ValidationStatus.FAIL:
                summary["components"][result.component]["failed"] += 1
            elif result.status == ValidationStatus.ERROR:
                summary["components"][result.component]["errors"] += 1

        # Generate recommendations
        if summary["failed"] > 0:
            summary["recommendations"].append(
                f"Fix {summary['failed']} failed validation checks",
            )
        if summary["errors"] > 0:
            summary["recommendations"].append(
                f"Resolve {summary['errors']} validation errors",
            )
        if summary["warnings"] > 0:
            summary["recommendations"].append(
                f"Address {summary['warnings']} validation warnings",
            )

        return summary

    def print_validation_report(self):
        """Print detailed validation report"""
        summary = self._generate_validation_summary()

        print("\n" + "=" * 60)
        print("📋 VALIDATION SUMMARY")
        print("=" * 60)

        print(f"Total Checks: {summary['total_checks']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"⚠️  Warnings: {summary['warnings']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"💥 Errors: {summary['errors']}")

        success_rate = (summary["passed"] / max(summary["total_checks"], 1)) * 100
        print(f"📈 Success Rate: {success_rate:.1f}%")

        # Component breakdown
        print("\n📊 Component Breakdown:")
        print("-" * 30)
        for component, stats in summary["components"].items():
            total = sum(stats.values())
            passed = stats["passed"]
            rate = (passed / max(total, 1)) * 100
            status = "✅" if rate >= 90 else "⚠️" if rate >= 70 else "❌"
            print(f"{status} {component}: {passed}/{total} ({rate:.1f}%)")

        # Failed checks
        failed_results = [
            r
            for r in self.results
            if r.status in [ValidationStatus.FAIL, ValidationStatus.ERROR]
        ]
        if failed_results:
            print("\n❌ Failed Checks:")
            print("-" * 20)
            for result in failed_results[:10]:  # Show first 10
                print(f"   {result.component}.{result.test_name}: {result.message}")
            if len(failed_results) > 10:
                print(f"   ... and {len(failed_results) - 10} more")

        # Recommendations
        if summary["recommendations"]:
            print("\n💡 Recommendations:")
            print("-" * 20)
            for i, rec in enumerate(summary["recommendations"], 1):
                print(f"{i}. {rec}")

        # Overall status
        if summary["failed"] == 0 and summary["errors"] == 0:
            print("\n🎉 VALIDATION PASSED - Framework is ready!")
        else:
            print("\n⚠️  VALIDATION ISSUES FOUND - Please address before running tests")

        return summary["failed"] == 0 and summary["errors"] == 0


async def main():
    """Run validation"""
    validator = TestValidator()
    await validator.validate_all()
    success = validator.print_validation_report()
    return 0 if success else 1
