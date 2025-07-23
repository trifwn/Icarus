"""
Tests for the Plugin Development SDK components.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from ..generator import PluginGenerator
from ..packager import PluginPackager
from ..tester import PluginTester
from ..validator import PluginValidator


class TestPluginGenerator(unittest.TestCase):
    """Test the plugin generator."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = PluginGenerator()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_list_templates(self):
        """Test listing available templates."""
        templates = self.generator.list_templates()

        self.assertIsInstance(templates, list)
        self.assertGreater(len(templates), 0)
        self.assertIn("basic", templates)

    def test_get_template_info(self):
        """Test getting template information."""
        template_info = self.generator.get_template_info("basic")

        self.assertIsNotNone(template_info)
        self.assertEqual(template_info.name, "Basic Plugin")
        self.assertIsInstance(template_info.files, list)

    def test_generate_basic_plugin(self):
        """Test generating a basic plugin."""
        plugin_name = "test_plugin"

        success = self.generator.generate_plugin(
            plugin_name=plugin_name,
            template_name="basic",
            output_dir=self.temp_dir,
            author_name="Test Author",
            author_email="test@example.com",
        )

        self.assertTrue(success)

        # Check generated files
        plugin_dir = Path(self.temp_dir) / plugin_name
        self.assertTrue(plugin_dir.exists())
        self.assertTrue((plugin_dir / "__init__.py").exists())
        self.assertTrue((plugin_dir / "plugin.py").exists())
        self.assertTrue((plugin_dir / "manifest.json").exists())
        self.assertTrue((plugin_dir / "README.md").exists())

    def test_generate_analysis_plugin(self):
        """Test generating an analysis plugin."""
        plugin_name = "test_analysis"

        success = self.generator.generate_plugin(
            plugin_name=plugin_name,
            template_name="analysis",
            output_dir=self.temp_dir,
            author_name="Test Author",
        )

        self.assertTrue(success)

        # Check analysis-specific files
        plugin_dir = Path(self.temp_dir) / plugin_name
        self.assertTrue((plugin_dir / "analysis.py").exists())
        self.assertTrue((plugin_dir / "tests" / "test_analysis.py").exists())


class TestPluginValidator(unittest.TestCase):
    """Test the plugin validator."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = PluginValidator()
        self.generator = PluginGenerator()

        # Generate a test plugin
        self.plugin_name = "test_plugin"
        self.generator.generate_plugin(
            plugin_name=self.plugin_name,
            template_name="basic",
            output_dir=self.temp_dir,
            author_name="Test Author",
        )
        self.plugin_path = Path(self.temp_dir) / self.plugin_name

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_validate_valid_plugin(self):
        """Test validating a valid plugin."""
        result = self.validator.validate_plugin(str(self.plugin_path))

        self.assertIsNotNone(result)
        self.assertTrue(result.is_valid)
        self.assertIsInstance(result.errors, list)
        self.assertIsInstance(result.warnings, list)

    def test_validate_missing_plugin(self):
        """Test validating a non-existent plugin."""
        result = self.validator.validate_plugin("/nonexistent/path")

        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)

    def test_validate_manifest_only(self):
        """Test validating just a manifest."""
        manifest_data = {
            "name": "test_plugin",
            "version": "1.0.0",
            "description": "Test plugin",
            "author": {"name": "Test Author"},
            "type": "utility",
            "main_module": "test_plugin",
            "main_class": "TestPlugin",
        }

        result = self.validator.validate_manifest_only(manifest_data)

        self.assertTrue(result.is_valid)


class TestPluginTester(unittest.TestCase):
    """Test the plugin tester."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.tester = PluginTester()
        self.generator = PluginGenerator()

        # Generate a test plugin
        self.plugin_name = "test_plugin"
        self.generator.generate_plugin(
            plugin_name=self.plugin_name,
            template_name="basic",
            output_dir=self.temp_dir,
            author_name="Test Author",
        )
        self.plugin_path = Path(self.temp_dir) / self.plugin_name

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_run_plugin_tests(self):
        """Test running plugin tests."""
        results = self.tester.run_plugin_tests(str(self.plugin_path))

        self.assertIsInstance(results, dict)
        self.assertIn("plugin_path", results)
        self.assertIn("overall_status", results)
        self.assertIn("tests", results)

        # Check that basic tests were run
        tests = results.get("tests", {})
        self.assertIn("basic", tests)

    def test_generate_test_suite(self):
        """Test generating a test suite."""
        test_code = self.tester.generate_test_suite(str(self.plugin_path))

        self.assertIsInstance(test_code, str)
        self.assertIn("unittest", test_code)
        self.assertIn("TestCase", test_code)


class TestPluginPackager(unittest.TestCase):
    """Test the plugin packager."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.packager = PluginPackager()
        self.generator = PluginGenerator()

        # Generate a test plugin
        self.plugin_name = "test_plugin"
        self.generator.generate_plugin(
            plugin_name=self.plugin_name,
            template_name="basic",
            output_dir=self.temp_dir,
            author_name="Test Author",
        )
        self.plugin_path = Path(self.temp_dir) / self.plugin_name

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_package_plugin(self):
        """Test packaging a plugin."""
        package_path = Path(self.temp_dir) / "test_plugin.zip"

        success = self.packager.package_plugin(
            plugin_path=str(self.plugin_path),
            output_path=str(package_path),
            format="zip",
            validate=False,  # Skip validation for speed
        )

        self.assertTrue(success)
        self.assertTrue(package_path.exists())
        self.assertGreater(package_path.stat().st_size, 0)

    def test_extract_package(self):
        """Test extracting a package."""
        # First create a package
        package_path = Path(self.temp_dir) / "test_plugin.zip"

        success = self.packager.package_plugin(
            plugin_path=str(self.plugin_path),
            output_path=str(package_path),
            format="zip",
            validate=False,
        )

        self.assertTrue(success)

        # Then extract it
        extract_dir = Path(self.temp_dir) / "extracted"

        success = self.packager.extract_package(str(package_path), str(extract_dir))

        self.assertTrue(success)
        self.assertTrue(extract_dir.exists())

        # Check extracted files
        extracted_plugin = extract_dir / self.plugin_name
        self.assertTrue(extracted_plugin.exists())
        self.assertTrue((extracted_plugin / "plugin.py").exists())

    def test_verify_package(self):
        """Test verifying a package."""
        # Create a package
        package_path = Path(self.temp_dir) / "test_plugin.zip"

        success = self.packager.package_plugin(
            plugin_path=str(self.plugin_path),
            output_path=str(package_path),
            format="zip",
            validate=False,
        )

        self.assertTrue(success)

        # Verify the package
        result = self.packager.verify_package(str(package_path))

        self.assertIsInstance(result, dict)
        self.assertIn("valid", result)
        self.assertIn("errors", result)
        self.assertIn("warnings", result)

    def test_get_package_info(self):
        """Test getting package information."""
        # Create a package
        package_path = Path(self.temp_dir) / "test_plugin.zip"

        success = self.packager.package_plugin(
            plugin_path=str(self.plugin_path),
            output_path=str(package_path),
            format="zip",
            validate=False,
        )

        self.assertTrue(success)

        # Get package info
        info = self.packager.get_package_info(str(package_path))

        self.assertIsInstance(info, dict)
        self.assertTrue(info["exists"])
        self.assertEqual(info["format"], "zip")
        self.assertGreater(info["size"], 0)
        self.assertIsInstance(info["files"], list)


if __name__ == "__main__":
    unittest.main()
