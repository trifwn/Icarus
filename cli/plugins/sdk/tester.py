"""
Plugin testing framework for ICARUS CLI plugins.

This module provides comprehensive testing tools for plugin development.
"""

import importlib.util
import logging
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from unittest.mock import Mock

from ..api import IcarusPlugin
from ..api import PluginAPI
from ..exceptions import PluginError
from ..models import PluginManifest


class MockPluginContext:
    """Mock plugin context for testing."""

    def __init__(self):
        self.app_instance = Mock()
        self.session_manager = Mock()
        self.config_manager = Mock()
        self.event_system = Mock()
        self.ui_manager = Mock()
        self.data_manager = Mock()
        self.logger = Mock()


class PluginTestCase(unittest.TestCase):
    """
    Base test case class for plugin testing.

    Provides common setup and utilities for testing plugins.
    """

    def setUp(self):
        """Set up test environment."""
        self.mock_context = MockPluginContext()
        self.plugin_api = PluginAPI(self.mock_context)
        self.plugin_instance = None
        self.temp_dir = None

    def tearDown(self):
        """Clean up test environment."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def create_temp_plugin_dir(self) -> Path:
        """Create a temporary plugin directory."""
        self.temp_dir = tempfile.mkdtemp()
        return Path(self.temp_dir)

    def load_plugin_from_path(self, plugin_path: str) -> IcarusPlugin:
        """Load a plugin from a file path."""
        plugin_path = Path(plugin_path)

        if plugin_path.is_file():
            # Single file plugin
            spec = importlib.util.spec_from_file_location("test_plugin", plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find plugin class
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, IcarusPlugin)
                    and attr != IcarusPlugin
                ):
                    return attr()

        else:
            # Directory plugin
            init_file = plugin_path / "__init__.py"
            if init_file.exists():
                spec = importlib.util.spec_from_file_location("test_plugin", init_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, "PLUGIN_CLASS"):
                    return module.PLUGIN_CLASS()

        raise PluginError("Could not load plugin class")

    def assert_plugin_manifest_valid(self, manifest: PluginManifest):
        """Assert that a plugin manifest is valid."""
        self.assertIsNotNone(manifest.name)
        self.assertIsNotNone(manifest.version)
        self.assertIsNotNone(manifest.description)
        self.assertIsNotNone(manifest.author)
        self.assertIsNotNone(manifest.plugin_type)
        self.assertIsNotNone(manifest.security_level)
        self.assertIsNotNone(manifest.main_module)
        self.assertIsNotNone(manifest.main_class)

    def assert_plugin_activated(self, plugin: IcarusPlugin):
        """Assert that a plugin was properly activated."""
        # Check that API methods were called
        self.plugin_api.context.event_system.register_handler.assert_called()

    def simulate_event(self, event_name: str, data: Any = None):
        """Simulate an event being emitted."""
        if hasattr(self.plugin_instance, "api"):
            # Find registered event handlers
            for call in self.plugin_instance.api.context.event_system.register_handler.call_args_list:
                if call[0][0] == event_name:
                    handler = call[0][1]
                    handler(data)


class PluginTester:
    """
    Comprehensive plugin testing framework.

    Provides tools for automated testing of plugin functionality,
    performance, and compatibility.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.test_results: Dict[str, Any] = {}

    def run_plugin_tests(
        self,
        plugin_path: str,
        test_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run comprehensive tests on a plugin.

        Args:
            plugin_path: Path to plugin directory or file
            test_config: Optional test configuration

        Returns:
            Dictionary with test results
        """
        self.logger.info(f"Running tests for plugin: {plugin_path}")

        results = {
            "plugin_path": plugin_path,
            "timestamp": None,
            "overall_status": "unknown",
            "tests": {},
        }

        try:
            # Basic functionality tests
            results["tests"]["basic"] = self._test_basic_functionality(plugin_path)

            # API usage tests
            results["tests"]["api"] = self._test_api_usage(plugin_path)

            # Event handling tests
            results["tests"]["events"] = self._test_event_handling(plugin_path)

            # Configuration tests
            results["tests"]["config"] = self._test_configuration(plugin_path)

            # Error handling tests
            results["tests"]["error_handling"] = self._test_error_handling(plugin_path)

            # Performance tests
            if test_config and test_config.get("performance_tests", False):
                results["tests"]["performance"] = self._test_performance(plugin_path)

            # Security tests
            if test_config and test_config.get("security_tests", False):
                results["tests"]["security"] = self._test_security(plugin_path)

            # Determine overall status
            all_passed = all(
                test_result.get("passed", False)
                for test_result in results["tests"].values()
            )
            results["overall_status"] = "passed" if all_passed else "failed"

        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
            results["overall_status"] = "error"
            results["error"] = str(e)

        return results

    def _test_basic_functionality(self, plugin_path: str) -> Dict[str, Any]:
        """Test basic plugin functionality."""
        test_result = {"name": "Basic Functionality", "passed": False, "details": []}

        try:
            # Load plugin
            plugin_path = Path(plugin_path)
            plugin = self._load_test_plugin(plugin_path)

            if not plugin:
                test_result["details"].append("Failed to load plugin")
                return test_result

            test_result["details"].append("Plugin loaded successfully")

            # Test manifest
            try:
                manifest = plugin.get_manifest()
                if manifest:
                    test_result["details"].append("Manifest retrieved successfully")
                else:
                    test_result["details"].append("Manifest is None")
                    return test_result
            except Exception as e:
                test_result["details"].append(f"Manifest retrieval failed: {e}")
                return test_result

            # Test initialization
            try:
                mock_context = MockPluginContext()
                api = PluginAPI(mock_context)
                plugin.initialize(api)
                test_result["details"].append("Plugin initialized successfully")
            except Exception as e:
                test_result["details"].append(f"Initialization failed: {e}")
                return test_result

            # Test activation
            try:
                plugin.activate()
                test_result["details"].append("Plugin activated successfully")
            except Exception as e:
                test_result["details"].append(f"Activation failed: {e}")
                return test_result

            # Test deactivation
            try:
                plugin.deactivate()
                test_result["details"].append("Plugin deactivated successfully")
            except Exception as e:
                test_result["details"].append(f"Deactivation failed: {e}")
                return test_result

            test_result["passed"] = True

        except Exception as e:
            test_result["details"].append(f"Basic functionality test failed: {e}")

        return test_result

    def _test_api_usage(self, plugin_path: str) -> Dict[str, Any]:
        """Test plugin API usage."""
        test_result = {"name": "API Usage", "passed": False, "details": []}

        try:
            plugin = self._load_test_plugin(Path(plugin_path))
            if not plugin:
                test_result["details"].append("Failed to load plugin")
                return test_result

            # Initialize with mock API
            mock_context = MockPluginContext()
            api = PluginAPI(mock_context)
            plugin.initialize(api)

            # Test API access
            if hasattr(plugin, "api") and plugin.api:
                test_result["details"].append("Plugin has API access")
            else:
                test_result["details"].append("Plugin missing API access")
                return test_result

            # Test common API methods
            api_methods_tested = []

            # Test logging
            try:
                plugin.api.log_info("Test message")
                api_methods_tested.append("log_info")
            except Exception as e:
                test_result["details"].append(f"log_info failed: {e}")

            # Test configuration
            try:
                plugin.api.get_config("test_key", "default_value")
                api_methods_tested.append("get_config")
            except Exception as e:
                test_result["details"].append(f"get_config failed: {e}")

            # Test events
            try:
                plugin.api.emit_event("test_event", {"data": "test"})
                api_methods_tested.append("emit_event")
            except Exception as e:
                test_result["details"].append(f"emit_event failed: {e}")

            test_result["details"].append(f"API methods tested: {api_methods_tested}")

            if len(api_methods_tested) >= 2:
                test_result["passed"] = True

        except Exception as e:
            test_result["details"].append(f"API usage test failed: {e}")

        return test_result

    def _test_event_handling(self, plugin_path: str) -> Dict[str, Any]:
        """Test plugin event handling."""
        test_result = {"name": "Event Handling", "passed": False, "details": []}

        try:
            plugin = self._load_test_plugin(Path(plugin_path))
            if not plugin:
                test_result["details"].append("Failed to load plugin")
                return test_result

            # Initialize plugin
            mock_context = MockPluginContext()
            api = PluginAPI(mock_context)
            plugin.initialize(api)
            plugin.activate()

            # Check if plugin registered any event handlers
            register_calls = api.context.event_system.register_handler.call_args_list

            if register_calls:
                test_result["details"].append(
                    f"Plugin registered {len(register_calls)} event handlers",
                )

                # Test event emission
                try:
                    api.emit_event("test_event", {"test": True})
                    test_result["details"].append("Event emission successful")
                except Exception as e:
                    test_result["details"].append(f"Event emission failed: {e}")

                test_result["passed"] = True
            else:
                test_result["details"].append("No event handlers registered")
                test_result["passed"] = True  # Not all plugins need event handlers

        except Exception as e:
            test_result["details"].append(f"Event handling test failed: {e}")

        return test_result

    def _test_configuration(self, plugin_path: str) -> Dict[str, Any]:
        """Test plugin configuration handling."""
        test_result = {"name": "Configuration", "passed": False, "details": []}

        try:
            plugin = self._load_test_plugin(Path(plugin_path))
            if not plugin:
                test_result["details"].append("Failed to load plugin")
                return test_result

            # Initialize plugin
            mock_context = MockPluginContext()
            api = PluginAPI(mock_context)
            plugin.initialize(api)

            # Test configuration method
            test_config = {"test_setting": "test_value"}

            try:
                plugin.configure(test_config)
                test_result["details"].append(
                    "Configuration method executed successfully",
                )
            except Exception as e:
                test_result["details"].append(f"Configuration method failed: {e}")
                return test_result

            # Test configuration access through API
            try:
                value = plugin.api.get_config("test_key", "default")
                test_result["details"].append(
                    "Configuration access through API successful",
                )
            except Exception as e:
                test_result["details"].append(f"Configuration access failed: {e}")

            test_result["passed"] = True

        except Exception as e:
            test_result["details"].append(f"Configuration test failed: {e}")

        return test_result

    def _test_error_handling(self, plugin_path: str) -> Dict[str, Any]:
        """Test plugin error handling."""
        test_result = {"name": "Error Handling", "passed": False, "details": []}

        try:
            plugin = self._load_test_plugin(Path(plugin_path))
            if not plugin:
                test_result["details"].append("Failed to load plugin")
                return test_result

            # Initialize plugin
            mock_context = MockPluginContext()
            api = PluginAPI(mock_context)
            plugin.initialize(api)

            # Test error scenarios
            error_scenarios_passed = 0

            # Test invalid configuration
            try:
                plugin.configure(None)
                test_result["details"].append(
                    "Plugin handled None configuration gracefully",
                )
                error_scenarios_passed += 1
            except Exception as e:
                test_result["details"].append(
                    f"Plugin failed to handle None configuration: {e}",
                )

            # Test activation without initialization (if applicable)
            try:
                fresh_plugin = self._load_test_plugin(Path(plugin_path))
                fresh_plugin.activate()
                test_result["details"].append(
                    "Plugin handled activation without initialization",
                )
            except Exception:
                test_result["details"].append(
                    "Plugin properly rejected activation without initialization",
                )
                error_scenarios_passed += 1

            if error_scenarios_passed > 0:
                test_result["passed"] = True

        except Exception as e:
            test_result["details"].append(f"Error handling test failed: {e}")

        return test_result

    def _test_performance(self, plugin_path: str) -> Dict[str, Any]:
        """Test plugin performance."""
        test_result = {"name": "Performance", "passed": False, "details": []}

        try:
            import time

            plugin = self._load_test_plugin(Path(plugin_path))
            if not plugin:
                test_result["details"].append("Failed to load plugin")
                return test_result

            # Test initialization time
            start_time = time.time()
            mock_context = MockPluginContext()
            api = PluginAPI(mock_context)
            plugin.initialize(api)
            init_time = time.time() - start_time

            test_result["details"].append(f"Initialization time: {init_time:.3f}s")

            # Test activation time
            start_time = time.time()
            plugin.activate()
            activation_time = time.time() - start_time

            test_result["details"].append(f"Activation time: {activation_time:.3f}s")

            # Performance thresholds
            if init_time < 1.0 and activation_time < 1.0:
                test_result["passed"] = True
                test_result["details"].append("Performance within acceptable limits")
            else:
                test_result["details"].append("Performance may be slow")

        except Exception as e:
            test_result["details"].append(f"Performance test failed: {e}")

        return test_result

    def _test_security(self, plugin_path: str) -> Dict[str, Any]:
        """Test plugin security aspects."""
        test_result = {"name": "Security", "passed": False, "details": []}

        try:
            # This would integrate with the security validator
            from .validator import PluginValidator

            validator = PluginValidator()
            validation_result = validator.validate_plugin(plugin_path)

            # Check for security warnings
            security_warnings = [
                warning
                for warning in validation_result.warnings
                if "dangerous" in warning.lower() or "security" in warning.lower()
            ]

            if security_warnings:
                test_result["details"].extend(security_warnings)
                test_result["details"].append("Security concerns found")
            else:
                test_result["details"].append("No obvious security concerns")
                test_result["passed"] = True

        except Exception as e:
            test_result["details"].append(f"Security test failed: {e}")

        return test_result

    def _load_test_plugin(self, plugin_path: Path) -> Optional[IcarusPlugin]:
        """Load a plugin for testing."""
        try:
            if plugin_path.is_file():
                # Single file plugin
                spec = importlib.util.spec_from_file_location(
                    "test_plugin",
                    plugin_path,
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find plugin class
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, IcarusPlugin)
                        and attr != IcarusPlugin
                    ):
                        return attr()

            else:
                # Directory plugin
                init_file = plugin_path / "__init__.py"
                if init_file.exists():
                    spec = importlib.util.spec_from_file_location(
                        "test_plugin",
                        init_file,
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    if hasattr(module, "PLUGIN_CLASS"):
                        return module.PLUGIN_CLASS()

            return None

        except Exception as e:
            self.logger.error(f"Failed to load test plugin: {e}")
            return None

    def generate_test_suite(self, plugin_path: str) -> str:
        """
        Generate a test suite for a plugin.

        Args:
            plugin_path: Path to plugin

        Returns:
            Generated test code
        """
        plugin_path = Path(plugin_path)
        plugin_name = plugin_path.stem

        test_code = f'''"""
Generated test suite for {plugin_name} plugin.
"""

import unittest
from unittest.mock import Mock, patch
from pathlib import Path

from icarus_cli.plugins.sdk.tester import PluginTestCase


class Test{plugin_name.title().replace('_', '')}Plugin(PluginTestCase):
    """Test cases for {plugin_name} plugin."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.plugin_path = Path(__file__).parent.parent / "{plugin_name}"
        self.plugin_instance = self.load_plugin_from_path(str(self.plugin_path))
        self.plugin_instance.initialize(self.plugin_api)

    def test_manifest(self):
        """Test plugin manifest."""
        manifest = self.plugin_instance.get_manifest()
        self.assert_plugin_manifest_valid(manifest)
        self.assertEqual(manifest.name, "{plugin_name}")

    def test_activation(self):
        """Test plugin activation."""
        self.plugin_instance.activate()
        self.assert_plugin_activated(self.plugin_instance)

    def test_deactivation(self):
        """Test plugin deactivation."""
        self.plugin_instance.activate()
        self.plugin_instance.deactivate()
        # Add specific deactivation assertions here

    def test_configuration(self):
        """Test plugin configuration."""
        test_config = {{"test_key": "test_value"}}
        self.plugin_instance.configure(test_config)
        # Add configuration-specific assertions here

    # Add more specific tests based on plugin functionality


if __name__ == '__main__':
    unittest.main()
'''

        return test_code

    def create_test_file(
        self,
        plugin_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Create a test file for a plugin.

        Args:
            plugin_path: Path to plugin
            output_path: Optional output path for test file

        Returns:
            Path to created test file
        """
        plugin_path = Path(plugin_path)
        plugin_name = plugin_path.stem

        if not output_path:
            if plugin_path.is_dir():
                output_path = plugin_path / "tests" / f"test_{plugin_name}.py"
            else:
                output_path = plugin_path.parent / f"test_{plugin_name}.py"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        test_code = self.generate_test_suite(str(plugin_path))

        with open(output_path, "w") as f:
            f.write(test_code)

        self.logger.info(f"Test file created: {output_path}")
        return str(output_path)

    def run_unittest_suite(self, test_path: str) -> Dict[str, Any]:
        """
        Run a unittest suite and return results.

        Args:
            test_path: Path to test file or directory

        Returns:
            Test results dictionary
        """
        try:
            # Discover and run tests
            loader = unittest.TestLoader()

            if Path(test_path).is_file():
                # Single test file
                spec = importlib.util.spec_from_file_location("test_module", test_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                suite = loader.loadTestsFromModule(module)
            else:
                # Test directory
                suite = loader.discover(test_path, pattern="test_*.py")

            # Run tests
            runner = unittest.TextTestRunner(verbosity=2, stream=open("/dev/null", "w"))
            result = runner.run(suite)

            return {
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skipped": len(result.skipped) if hasattr(result, "skipped") else 0,
                "success": result.wasSuccessful(),
                "failure_details": [str(failure) for failure in result.failures],
                "error_details": [str(error) for error in result.errors],
            }

        except Exception as e:
            return {"error": str(e), "success": False}
