"""
Plugin security system for validation and sandboxing.
"""

import ast
import logging
import tempfile
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

from .models import PluginInfo
from .models import PluginManifest
from .models import SecurityLevel


class SecurityValidator:
    """
    Security validator for plugin code analysis.
    """

    # Dangerous imports and functions
    DANGEROUS_IMPORTS = {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "tempfile",
        "socket",
        "urllib",
        "requests",
        "http",
        "ctypes",
        "multiprocessing",
        "threading",
        "pickle",
        "marshal",
        "eval",
        "exec",
        "__import__",
        "importlib",
    }

    DANGEROUS_FUNCTIONS = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "file",
        "input",
        "raw_input",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "globals",
        "locals",
        "vars",
        "dir",
    }

    DANGEROUS_ATTRIBUTES = {
        "__class__",
        "__bases__",
        "__subclasses__",
        "__import__",
        "__builtins__",
        "__globals__",
        "__dict__",
        "__code__",
        "__func__",
    }

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def validate_plugin(self, plugin_info: PluginInfo) -> Tuple[bool, List[str]]:
        """
        Validate plugin security.

        Args:
            plugin_info: Plugin information to validate

        Returns:
            Tuple of (is_safe, list_of_issues)
        """
        issues = []

        try:
            # Validate manifest
            manifest_issues = self._validate_manifest(plugin_info.manifest)
            issues.extend(manifest_issues)

            # Validate code
            code_issues = self._validate_code(plugin_info)
            issues.extend(code_issues)

            # Validate permissions
            permission_issues = self._validate_permissions(plugin_info.manifest)
            issues.extend(permission_issues)

            # Validate dependencies
            dependency_issues = self._validate_dependencies(plugin_info.manifest)
            issues.extend(dependency_issues)

        except Exception as e:
            issues.append(f"Validation error: {e}")

        is_safe = len(issues) == 0 or self._are_issues_acceptable(
            issues,
            plugin_info.manifest.security_level,
        )

        return is_safe, issues

    def _validate_manifest(self, manifest: PluginManifest) -> List[str]:
        """Validate plugin manifest."""
        issues = []

        # Check required fields
        if not manifest.name or not manifest.name.strip():
            issues.append("Plugin name is required")

        if not manifest.description or not manifest.description.strip():
            issues.append("Plugin description is required")

        if not manifest.author.name or not manifest.author.name.strip():
            issues.append("Plugin author name is required")

        # Validate security level
        if manifest.security_level == SecurityLevel.DANGEROUS:
            issues.append("Plugin requests dangerous security level")

        # Check for suspicious patterns in metadata
        suspicious_patterns = ["eval", "exec", "system", "shell", "cmd"]
        for pattern in suspicious_patterns:
            if pattern.lower() in manifest.description.lower():
                issues.append(f"Suspicious pattern '{pattern}' found in description")

        return issues

    def _validate_code(self, plugin_info: PluginInfo) -> List[str]:
        """Validate plugin code for security issues."""
        issues = []

        try:
            plugin_path = Path(plugin_info.path)

            if plugin_path.is_dir():
                # Validate all Python files in directory
                for py_file in plugin_path.rglob("*.py"):
                    file_issues = self._validate_python_file(py_file)
                    issues.extend([f"{py_file.name}: {issue}" for issue in file_issues])
            else:
                # Validate single file
                file_issues = self._validate_python_file(plugin_path)
                issues.extend(file_issues)

        except Exception as e:
            issues.append(f"Code validation error: {e}")

        return issues

    def _validate_python_file(self, file_path: Path) -> List[str]:
        """Validate a single Python file."""
        issues = []

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Parse AST
            tree = ast.parse(content, filename=str(file_path))

            # Analyze AST
            analyzer = SecurityAnalyzer()
            analyzer.visit(tree)

            issues.extend(analyzer.issues)

        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
        except Exception as e:
            issues.append(f"File analysis error: {e}")

        return issues

    def _validate_permissions(self, manifest: PluginManifest) -> List[str]:
        """Validate requested permissions."""
        issues = []

        # Check for excessive permissions
        high_risk_permissions = {
            "file_system_write",
            "network_access",
            "system_commands",
            "process_control",
            "environment_access",
        }

        requested_permissions = {perm.name for perm in manifest.permissions}
        excessive_permissions = requested_permissions & high_risk_permissions

        if excessive_permissions and manifest.security_level == SecurityLevel.SAFE:
            issues.append(
                f"High-risk permissions requested for SAFE security level: {excessive_permissions}",
            )

        return issues

    def _validate_dependencies(self, manifest: PluginManifest) -> List[str]:
        """Validate plugin dependencies."""
        issues = []

        # Check for dangerous dependencies
        dangerous_deps = {
            "subprocess32",
            "pexpect",
            "paramiko",
            "fabric",
            "requests",
            "urllib3",
            "httpx",
            "aiohttp",
        }

        for dep in manifest.dependencies:
            if dep.name.lower() in dangerous_deps:
                issues.append(f"Potentially dangerous dependency: {dep.name}")

        return issues

    def _are_issues_acceptable(
        self,
        issues: List[str],
        security_level: SecurityLevel,
    ) -> bool:
        """Check if security issues are acceptable for the given security level."""
        if security_level == SecurityLevel.SAFE:
            return len(issues) == 0
        elif security_level == SecurityLevel.RESTRICTED:
            # Allow some issues for restricted plugins
            dangerous_issues = [
                issue for issue in issues if "dangerous" in issue.lower()
            ]
            return len(dangerous_issues) == 0
        elif security_level == SecurityLevel.ELEVATED:
            # Allow most issues for elevated plugins
            critical_issues = [issue for issue in issues if "critical" in issue.lower()]
            return len(critical_issues) == 0
        else:  # DANGEROUS
            return True  # Allow all issues for dangerous plugins


class SecurityAnalyzer(ast.NodeVisitor):
    """AST visitor for security analysis."""

    def __init__(self):
        self.issues = []
        self.imports = set()
        self.function_calls = set()
        self.attribute_access = set()

    def visit_Import(self, node):
        """Visit import statements."""
        for alias in node.names:
            self.imports.add(alias.name)
            if alias.name in SecurityValidator.DANGEROUS_IMPORTS:
                self.issues.append(f"Dangerous import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Visit from-import statements."""
        if node.module:
            self.imports.add(node.module)
            if node.module in SecurityValidator.DANGEROUS_IMPORTS:
                self.issues.append(f"Dangerous import: {node.module}")

        for alias in node.names:
            if alias.name in SecurityValidator.DANGEROUS_FUNCTIONS:
                self.issues.append(f"Dangerous function import: {alias.name}")

        self.generic_visit(node)

    def visit_Call(self, node):
        """Visit function calls."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            self.function_calls.add(func_name)

            if func_name in SecurityValidator.DANGEROUS_FUNCTIONS:
                self.issues.append(f"Dangerous function call: {func_name}")

        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if attr_name in SecurityValidator.DANGEROUS_FUNCTIONS:
                self.issues.append(f"Dangerous method call: {attr_name}")

        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Visit attribute access."""
        if isinstance(node.attr, str):
            self.attribute_access.add(node.attr)

            if node.attr in SecurityValidator.DANGEROUS_ATTRIBUTES:
                self.issues.append(f"Dangerous attribute access: {node.attr}")

        self.generic_visit(node)

    def visit_Str(self, node):
        """Visit string literals."""
        # Check for suspicious string patterns
        suspicious_patterns = [
            "rm -rf",
            "del /f",
            "format c:",
            "sudo",
            "passwd",
            "chmod 777",
            "eval(",
            "exec(",
        ]

        for pattern in suspicious_patterns:
            if pattern in node.s.lower():
                self.issues.append(f"Suspicious string pattern: {pattern}")

        self.generic_visit(node)


class PluginSandbox:
    """
    Plugin sandbox for secure execution.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.restricted_modules = set(SecurityValidator.DANGEROUS_IMPORTS)
        self.restricted_builtins = set(SecurityValidator.DANGEROUS_FUNCTIONS)

    def create_sandbox_environment(self, plugin_info: PluginInfo) -> Dict[str, Any]:
        """
        Create a sandboxed environment for plugin execution.

        Args:
            plugin_info: Plugin information

        Returns:
            Sandboxed environment dictionary
        """
        # Create restricted builtins
        safe_builtins = {}
        for name in dir(__builtins__):
            if name not in self.restricted_builtins:
                safe_builtins[name] = getattr(__builtins__, name)

        # Add safe versions of some functions
        safe_builtins["open"] = self._safe_open
        safe_builtins["__import__"] = self._safe_import

        # Create sandbox environment
        sandbox_env = {
            "__builtins__": safe_builtins,
            "__name__": f"plugin_{plugin_info.manifest.name}",
            "__file__": plugin_info.path,
            "__package__": None,
        }

        return sandbox_env

    def _safe_open(self, filename, mode="r", **kwargs):
        """Safe version of open() function."""
        # Only allow reading from plugin directory and temp directory
        file_path = Path(filename).resolve()

        allowed_paths = [
            Path(tempfile.gettempdir()),
            Path.cwd() / "data",  # Allow access to data directory
        ]

        if not any(str(file_path).startswith(str(path)) for path in allowed_paths):
            raise PermissionError(f"Access denied to file: {filename}")

        if "w" in mode or "a" in mode:
            raise PermissionError("Write access not allowed in sandbox")

        return open(filename, mode, **kwargs)

    def _safe_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        """Safe version of __import__() function."""
        if name in self.restricted_modules:
            raise ImportError(f"Import of '{name}' not allowed in sandbox")

        return __import__(name, globals, locals, fromlist, level)


class PluginSecurity:
    """
    Main plugin security system.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.validator = SecurityValidator(logger)
        self.sandbox = PluginSandbox(logger)
        self.trusted_plugins: Set[str] = set()
        self.blocked_plugins: Set[str] = set()

    def validate_plugin_security(self, plugin_info: PluginInfo) -> bool:
        """
        Validate plugin security and determine if it's safe to load.

        Args:
            plugin_info: Plugin information

        Returns:
            True if plugin is safe to load, False otherwise
        """
        plugin_id = plugin_info.id

        # Check if plugin is explicitly trusted or blocked
        if plugin_id in self.trusted_plugins:
            self.logger.info(f"Plugin {plugin_id} is explicitly trusted")
            return True

        if plugin_id in self.blocked_plugins:
            self.logger.warning(f"Plugin {plugin_id} is explicitly blocked")
            return False

        # Validate plugin
        is_safe, issues = self.validator.validate_plugin(plugin_info)

        if not is_safe:
            self.logger.warning(f"Plugin {plugin_id} failed security validation:")
            for issue in issues:
                self.logger.warning(f"  - {issue}")

            # Store error information
            plugin_info.status = PluginStatus.ERROR
            plugin_info.last_error = f"Security validation failed: {'; '.join(issues)}"

        return is_safe

    def create_plugin_sandbox(self, plugin_info: PluginInfo) -> Dict[str, Any]:
        """
        Create a sandbox environment for plugin execution.

        Args:
            plugin_info: Plugin information

        Returns:
            Sandbox environment
        """
        if not plugin_info.manifest.sandbox_enabled:
            self.logger.warning(f"Sandbox disabled for plugin {plugin_info.id}")
            return {}

        return self.sandbox.create_sandbox_environment(plugin_info)

    def trust_plugin(self, plugin_id: str) -> None:
        """
        Mark a plugin as trusted.

        Args:
            plugin_id: Plugin identifier
        """
        self.trusted_plugins.add(plugin_id)
        if plugin_id in self.blocked_plugins:
            self.blocked_plugins.remove(plugin_id)

        self.logger.info(f"Plugin {plugin_id} marked as trusted")

    def block_plugin(self, plugin_id: str) -> None:
        """
        Mark a plugin as blocked.

        Args:
            plugin_id: Plugin identifier
        """
        self.blocked_plugins.add(plugin_id)
        if plugin_id in self.trusted_plugins:
            self.trusted_plugins.remove(plugin_id)

        self.logger.warning(f"Plugin {plugin_id} marked as blocked")

    def get_plugin_risk_level(self, plugin_info: PluginInfo) -> str:
        """
        Get the risk level of a plugin.

        Args:
            plugin_info: Plugin information

        Returns:
            Risk level string (low, medium, high, critical)
        """
        is_safe, issues = self.validator.validate_plugin(plugin_info)

        if plugin_info.id in self.trusted_plugins:
            return "low"

        if plugin_info.id in self.blocked_plugins:
            return "critical"

        if not is_safe:
            dangerous_issues = [
                issue for issue in issues if "dangerous" in issue.lower()
            ]
            if dangerous_issues:
                return "critical"
            elif len(issues) > 5:
                return "high"
            else:
                return "medium"

        return "low"
