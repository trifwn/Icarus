"""
Plugin validator for ICARUS CLI plugins.

This module provides comprehensive validation tools for plugin development.
"""

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from ..models import PluginManifest
from ..models import PluginType
from ..models import SecurityLevel


class ValidationResult:
    """Result of plugin validation."""

    def __init__(self):
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def add_info(self, message: str) -> None:
        """Add an info message."""
        self.info.append(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
        }


class PluginValidator:
    """
    Comprehensive plugin validator that checks plugin structure,
    code quality, security, and compatibility.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

        # Validation rules
        self.required_files = ["__init__.py", "manifest.json"]
        self.optional_files = ["README.md", "LICENSE", "requirements.txt"]

        # Security patterns to check
        self.dangerous_patterns = [
            r"exec\s*\(",
            r"eval\s*\(",
            r"__import__\s*\(",
            r"subprocess\.",
            r"os\.system",
            r"os\.popen",
            r'open\s*\([^)]*["\']w["\']',  # File writing
        ]

        # Required API methods for plugin classes
        self.required_methods = ["get_manifest", "on_activate"]
        self.optional_methods = ["on_deactivate", "on_configure", "on_initialize"]

    def validate_plugin(self, plugin_path: str) -> ValidationResult:
        """
        Validate a complete plugin.

        Args:
            plugin_path: Path to plugin directory or file

        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult()
        plugin_path = Path(plugin_path)

        try:
            self.logger.info(f"Validating plugin: {plugin_path}")

            # Check if path exists
            if not plugin_path.exists():
                result.add_error(f"Plugin path does not exist: {plugin_path}")
                return result

            # Validate structure
            self._validate_structure(plugin_path, result)

            # Validate manifest
            manifest = self._validate_manifest(plugin_path, result)

            # Validate code
            if manifest:
                self._validate_code(plugin_path, manifest, result)

            # Security validation
            self._validate_security(plugin_path, result)

            # Compatibility validation
            self._validate_compatibility(plugin_path, result)

            # Best practices validation
            self._validate_best_practices(plugin_path, result)

            self.logger.info(f"Validation complete. Valid: {result.is_valid}")

        except Exception as e:
            result.add_error(f"Validation failed with exception: {e}")
            self.logger.error(f"Validation exception: {e}")

        return result

    def _validate_structure(self, plugin_path: Path, result: ValidationResult) -> None:
        """Validate plugin directory structure."""
        if plugin_path.is_file():
            # Single file plugin
            if not plugin_path.suffix == ".py":
                result.add_error("Single file plugin must be a Python file")
            return

        # Directory plugin
        if not plugin_path.is_dir():
            result.add_error("Plugin path must be a directory or Python file")
            return

        # Check required files
        for required_file in self.required_files:
            file_path = plugin_path / required_file
            if not file_path.exists():
                result.add_error(f"Required file missing: {required_file}")

        # Check for Python package structure
        init_file = plugin_path / "__init__.py"
        if init_file.exists():
            result.add_info("Plugin follows Python package structure")

        # Check for documentation
        readme_files = list(plugin_path.glob("README*"))
        if readme_files:
            result.add_info("Documentation found")
        else:
            result.add_warning("No README file found")

        # Check for tests
        test_dirs = list(plugin_path.glob("test*"))
        test_files = list(plugin_path.glob("**/test_*.py"))
        if test_dirs or test_files:
            result.add_info("Test files found")
        else:
            result.add_warning("No test files found")

    def _validate_manifest(
        self,
        plugin_path: Path,
        result: ValidationResult,
    ) -> Optional[PluginManifest]:
        """Validate plugin manifest."""
        manifest_file = plugin_path / "manifest.json"

        if plugin_path.is_file():
            # For single file plugins, look for PLUGIN_MANIFEST variable
            return self._validate_inline_manifest(plugin_path, result)

        if not manifest_file.exists():
            result.add_error("manifest.json file not found")
            return None

        try:
            with open(manifest_file) as f:
                manifest_data = json.load(f)

            # Validate required fields
            required_fields = [
                "name",
                "version",
                "description",
                "author",
                "type",
                "main_module",
                "main_class",
            ]

            for field in required_fields:
                if field not in manifest_data:
                    result.add_error(f"Required manifest field missing: {field}")

            # Validate field types and values
            if "type" in manifest_data:
                try:
                    PluginType(manifest_data["type"])
                except ValueError:
                    result.add_error(f"Invalid plugin type: {manifest_data['type']}")

            if "security_level" in manifest_data:
                try:
                    SecurityLevel(manifest_data["security_level"])
                except ValueError:
                    result.add_error(
                        f"Invalid security level: {manifest_data['security_level']}",
                    )

            # Validate version format
            if "version" in manifest_data:
                version_pattern = r"^\d+\.\d+\.\d+(-\w+)?$"
                if not re.match(version_pattern, manifest_data["version"]):
                    result.add_warning(
                        "Version should follow semantic versioning (x.y.z)",
                    )

            # Validate author information
            if "author" in manifest_data:
                author = manifest_data["author"]
                if isinstance(author, dict):
                    if "name" not in author:
                        result.add_error("Author name is required")
                    if "email" in author:
                        email_pattern = (
                            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                        )
                        if not re.match(email_pattern, author["email"]):
                            result.add_warning(
                                "Invalid email format in author information",
                            )
                else:
                    result.add_error(
                        "Author must be an object with name and optional email/url",
                    )

            # Create manifest object for further validation
            try:
                manifest = PluginManifest.from_dict(manifest_data)
                result.add_info("Manifest structure is valid")
                return manifest
            except Exception as e:
                result.add_error(f"Failed to parse manifest: {e}")
                return None

        except json.JSONDecodeError as e:
            result.add_error(f"Invalid JSON in manifest: {e}")
            return None
        except Exception as e:
            result.add_error(f"Error reading manifest: {e}")
            return None

    def _validate_inline_manifest(
        self,
        plugin_file: Path,
        result: ValidationResult,
    ) -> Optional[PluginManifest]:
        """Validate inline manifest in single file plugin."""
        try:
            with open(plugin_file) as f:
                content = f.read()

            # Look for PLUGIN_MANIFEST variable
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "PLUGIN_MANIFEST"
                ):
                    # Found manifest, try to evaluate it
                    if isinstance(node.value, ast.Dict):
                        manifest_data = ast.literal_eval(node.value)
                        try:
                            manifest = PluginManifest.from_dict(manifest_data)
                            result.add_info("Inline manifest found and valid")
                            return manifest
                        except Exception as e:
                            result.add_error(f"Invalid inline manifest: {e}")
                            return None

            result.add_warning(
                "No PLUGIN_MANIFEST variable found in single file plugin",
            )
            return None

        except Exception as e:
            result.add_error(f"Error parsing plugin file: {e}")
            return None

    def _validate_code(
        self,
        plugin_path: Path,
        manifest: PluginManifest,
        result: ValidationResult,
    ) -> None:
        """Validate plugin code."""
        # Find main plugin file
        if plugin_path.is_file():
            plugin_files = [plugin_path]
        else:
            plugin_files = list(plugin_path.glob("**/*.py"))

        main_plugin_file = None

        # Look for the main plugin class
        for py_file in plugin_files:
            if self._file_contains_class(py_file, manifest.main_class):
                main_plugin_file = py_file
                break

        if not main_plugin_file:
            result.add_error(f"Main plugin class '{manifest.main_class}' not found")
            return

        # Validate main plugin class
        self._validate_plugin_class(main_plugin_file, manifest.main_class, result)

        # Validate all Python files
        for py_file in plugin_files:
            self._validate_python_file(py_file, result)

    def _file_contains_class(self, file_path: Path, class_name: str) -> bool:
        """Check if file contains a specific class."""
        try:
            with open(file_path) as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    return True

            return False

        except Exception:
            return False

    def _validate_plugin_class(
        self,
        file_path: Path,
        class_name: str,
        result: ValidationResult,
    ) -> None:
        """Validate the main plugin class."""
        try:
            with open(file_path) as f:
                content = f.read()

            tree = ast.parse(content)

            # Find the plugin class
            plugin_class = None
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    plugin_class = node
                    break

            if not plugin_class:
                result.add_error(f"Plugin class '{class_name}' not found")
                return

            # Check inheritance
            inherits_from_icarus_plugin = False
            for base in plugin_class.bases:
                if isinstance(base, ast.Name) and base.id == "IcarusPlugin":
                    inherits_from_icarus_plugin = True
                    break

            if not inherits_from_icarus_plugin:
                result.add_error("Plugin class must inherit from IcarusPlugin")

            # Check required methods
            class_methods = [
                node.name
                for node in plugin_class.body
                if isinstance(node, ast.FunctionDef)
            ]

            for required_method in self.required_methods:
                if required_method not in class_methods:
                    result.add_error(
                        f"Required method '{required_method}' not found in plugin class",
                    )

            # Check method signatures
            for node in plugin_class.body:
                if isinstance(node, ast.FunctionDef):
                    if node.name == "get_manifest":
                        # Should return PluginManifest
                        if not node.returns:
                            result.add_warning(
                                "get_manifest method should have return type annotation",
                            )
                    elif node.name in ["on_activate", "on_deactivate", "on_initialize"]:
                        # Should not have parameters (except self)
                        if len(node.args.args) > 1:
                            result.add_warning(
                                f"Method '{node.name}' should only have 'self' parameter",
                            )

            result.add_info("Plugin class structure is valid")

        except Exception as e:
            result.add_error(f"Error validating plugin class: {e}")

    def _validate_python_file(self, file_path: Path, result: ValidationResult) -> None:
        """Validate a Python file for syntax and basic quality."""
        try:
            with open(file_path) as f:
                content = f.read()

            # Check syntax
            try:
                ast.parse(content)
            except SyntaxError as e:
                result.add_error(f"Syntax error in {file_path.name}: {e}")
                return

            # Check for basic quality issues
            lines = content.split("\n")

            # Check line length
            long_lines = [i + 1 for i, line in enumerate(lines) if len(line) > 120]
            if long_lines:
                result.add_warning(
                    f"Long lines (>120 chars) in {file_path.name}: lines {long_lines[:5]}",
                )

            # Check for TODO comments
            todo_lines = [
                i + 1 for i, line in enumerate(lines) if "TODO" in line.upper()
            ]
            if todo_lines:
                result.add_info(
                    f"TODO comments found in {file_path.name}: {len(todo_lines)} items",
                )

            # Check for docstrings
            tree = ast.parse(content)
            classes_without_docstrings = []
            functions_without_docstrings = []

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if not ast.get_docstring(node):
                        classes_without_docstrings.append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    if not node.name.startswith("_") and not ast.get_docstring(node):
                        functions_without_docstrings.append(node.name)

            if classes_without_docstrings:
                result.add_warning(
                    f"Classes without docstrings in {file_path.name}: {classes_without_docstrings}",
                )

            if functions_without_docstrings:
                result.add_warning(
                    f"Public functions without docstrings in {file_path.name}: {functions_without_docstrings}",
                )

        except Exception as e:
            result.add_error(f"Error validating Python file {file_path.name}: {e}")

    def _validate_security(self, plugin_path: Path, result: ValidationResult) -> None:
        """Validate plugin security."""
        # Find all Python files
        if plugin_path.is_file():
            python_files = [plugin_path]
        else:
            python_files = list(plugin_path.glob("**/*.py"))

        for py_file in python_files:
            try:
                with open(py_file) as f:
                    content = f.read()

                # Check for dangerous patterns
                for pattern in self.dangerous_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        result.add_warning(
                            f"Potentially dangerous code pattern in {py_file.name}: {pattern}",
                        )

                # Check imports
                tree = ast.parse(content)
                dangerous_imports = ["subprocess", "os", "sys", "shutil", "tempfile"]

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name in dangerous_imports:
                                result.add_warning(
                                    f"Potentially dangerous import in {py_file.name}: {alias.name}",
                                )
                    elif isinstance(node, ast.ImportFrom):
                        if node.module in dangerous_imports:
                            result.add_warning(
                                f"Potentially dangerous import in {py_file.name}: {node.module}",
                            )

            except Exception as e:
                result.add_error(f"Error checking security for {py_file.name}: {e}")

    def _validate_compatibility(
        self,
        plugin_path: Path,
        result: ValidationResult,
    ) -> None:
        """Validate plugin compatibility."""
        # Check Python version compatibility
        if plugin_path.is_dir():
            manifest_file = plugin_path / "manifest.json"
            if manifest_file.exists():
                try:
                    with open(manifest_file) as f:
                        manifest_data = json.load(f)

                    python_version = manifest_data.get("python_version", ">=3.8")
                    if not python_version.startswith(">=3."):
                        result.add_warning(
                            "Python version requirement should be specified (e.g., '>=3.8')",
                        )

                    icarus_version = manifest_data.get("icarus_version", ">=1.0.0")
                    if not icarus_version:
                        result.add_warning(
                            "ICARUS version requirement should be specified",
                        )

                except Exception:
                    pass

        # Check for requirements.txt
        if plugin_path.is_dir():
            requirements_file = plugin_path / "requirements.txt"
            if requirements_file.exists():
                result.add_info("Requirements file found")
                try:
                    with open(requirements_file) as f:
                        requirements = f.read().strip().split("\n")

                    # Check for version pinning
                    unpinned = [
                        req
                        for req in requirements
                        if req
                        and not any(
                            op in req for op in ["==", ">=", "<=", ">", "<", "~="]
                        )
                    ]
                    if unpinned:
                        result.add_warning(f"Unpinned dependencies: {unpinned}")

                except Exception:
                    result.add_warning("Error reading requirements.txt")

    def _validate_best_practices(
        self,
        plugin_path: Path,
        result: ValidationResult,
    ) -> None:
        """Validate plugin follows best practices."""
        # Check for license file
        if plugin_path.is_dir():
            license_files = list(plugin_path.glob("LICENSE*"))
            if not license_files:
                result.add_warning("No LICENSE file found")
            else:
                result.add_info("License file found")

        # Check for changelog
        if plugin_path.is_dir():
            changelog_files = list(plugin_path.glob("CHANGELOG*")) + list(
                plugin_path.glob("HISTORY*"),
            )
            if changelog_files:
                result.add_info("Changelog file found")

        # Check for proper package structure
        if plugin_path.is_dir():
            init_file = plugin_path / "__init__.py"
            if init_file.exists():
                try:
                    with open(init_file) as f:
                        content = f.read()

                    # Check for version definition
                    if "__version__" in content:
                        result.add_info("Version defined in __init__.py")

                    # Check for author information
                    if "__author__" in content:
                        result.add_info("Author information in __init__.py")

                except Exception:
                    pass

    def validate_manifest_only(self, manifest_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate only the manifest data.

        Args:
            manifest_data: Manifest dictionary

        Returns:
            ValidationResult
        """
        result = ValidationResult()

        try:
            # Try to create manifest object
            manifest = PluginManifest.from_dict(manifest_data)
            result.add_info("Manifest is valid")

            # Additional validation
            if len(manifest.description) < 10:
                result.add_warning("Plugin description is very short")

            if not manifest.keywords:
                result.add_warning("No keywords specified")

            if not manifest.homepage and not manifest.repository:
                result.add_warning("No homepage or repository URL specified")

        except Exception as e:
            result.add_error(f"Invalid manifest: {e}")

        return result

    def get_validation_report(self, plugin_path: str) -> str:
        """
        Get a formatted validation report.

        Args:
            plugin_path: Path to plugin

        Returns:
            Formatted validation report
        """
        result = self.validate_plugin(plugin_path)

        report = f"Plugin Validation Report: {plugin_path}\n"
        report += "=" * 50 + "\n\n"

        if result.is_valid:
            report += "✅ VALIDATION PASSED\n\n"
        else:
            report += "❌ VALIDATION FAILED\n\n"

        if result.errors:
            report += "ERRORS:\n"
            for error in result.errors:
                report += f"  ❌ {error}\n"
            report += "\n"

        if result.warnings:
            report += "WARNINGS:\n"
            for warning in result.warnings:
                report += f"  ⚠️  {warning}\n"
            report += "\n"

        if result.info:
            report += "INFO:\n"
            for info in result.info:
                report += f"  ℹ️  {info}\n"
            report += "\n"

        return report
