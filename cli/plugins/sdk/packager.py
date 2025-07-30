"""
Plugin packager for ICARUS CLI plugins.

This module provides tools for packaging plugins for distribution.
"""

import hashlib
import json
import logging
import shutil
import tarfile
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional


class PluginPackager:
    """
    Plugin packager that creates distributable plugin packages.

    Supports multiple package formats and includes validation,
    dependency management, and metadata generation.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

        # Supported package formats
        self.supported_formats = ["zip", "tar.gz", "tar.bz2"]

        # Files to exclude from packaging
        self.exclude_patterns = [
            "__pycache__",
            "*.pyc",
            "*.pyo",
            ".git",
            ".gitignore",
            ".DS_Store",
            "Thumbs.db",
            "*.tmp",
            "*.temp",
            ".pytest_cache",
            ".coverage",
            "htmlcov",
            "dist",
            "build",
            "*.egg-info",
        ]

    def package_plugin(
        self,
        plugin_path: str,
        output_path: str,
        format: str = "zip",
        include_tests: bool = False,
        include_docs: bool = True,
        validate: bool = True,
    ) -> bool:
        """
        Package a plugin for distribution.

        Args:
            plugin_path: Path to plugin directory or file
            output_path: Output path for package
            format: Package format (zip, tar.gz, tar.bz2)
            include_tests: Whether to include test files
            include_docs: Whether to include documentation
            validate: Whether to validate plugin before packaging

        Returns:
            True if packaging successful, False otherwise
        """
        try:
            plugin_path = Path(plugin_path)
            output_path = Path(output_path)

            self.logger.info(f"Packaging plugin: {plugin_path}")

            # Validate format
            if format not in self.supported_formats:
                self.logger.error(f"Unsupported format: {format}")
                return False

            # Validate plugin if requested
            if validate:
                from .validator import PluginValidator

                validator = PluginValidator()
                result = validator.validate_plugin(str(plugin_path))

                if not result.is_valid:
                    self.logger.error("Plugin validation failed")
                    for error in result.errors:
                        self.logger.error(f"  - {error}")
                    return False

            # Create temporary directory for packaging
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                package_dir = temp_path / plugin_path.name

                # Copy plugin files
                self._copy_plugin_files(
                    plugin_path,
                    package_dir,
                    include_tests=include_tests,
                    include_docs=include_docs,
                )

                # Generate package metadata
                self._generate_package_metadata(package_dir)

                # Create package
                success = self._create_package(package_dir, output_path, format)

                if success:
                    self.logger.info(f"Plugin packaged successfully: {output_path}")

                    # Generate package info
                    self._generate_package_info(output_path)

                return success

        except Exception as e:
            self.logger.error(f"Packaging failed: {e}")
            return False

    def _copy_plugin_files(
        self,
        source_path: Path,
        dest_path: Path,
        include_tests: bool = False,
        include_docs: bool = True,
    ) -> None:
        """Copy plugin files to package directory."""
        dest_path.mkdir(parents=True, exist_ok=True)

        if source_path.is_file():
            # Single file plugin
            shutil.copy2(source_path, dest_path / source_path.name)
            return

        # Directory plugin
        for item in source_path.rglob("*"):
            if item.is_file():
                # Check if file should be excluded
                if self._should_exclude_file(
                    item,
                    source_path,
                    include_tests,
                    include_docs,
                ):
                    continue

                # Calculate relative path
                rel_path = item.relative_to(source_path)
                dest_file = dest_path / rel_path

                # Create parent directories
                dest_file.parent.mkdir(parents=True, exist_ok=True)

                # Copy file
                shutil.copy2(item, dest_file)

    def _should_exclude_file(
        self,
        file_path: Path,
        plugin_root: Path,
        include_tests: bool,
        include_docs: bool,
    ) -> bool:
        """Check if a file should be excluded from packaging."""
        rel_path = file_path.relative_to(plugin_root)

        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if pattern.startswith("*"):
                if file_path.name.endswith(pattern[1:]):
                    return True
            elif pattern.endswith("*"):
                if file_path.name.startswith(pattern[:-1]):
                    return True
            else:
                if pattern in str(rel_path):
                    return True

        # Check test files
        if not include_tests:
            if (
                "test" in str(rel_path).lower()
                or file_path.name.startswith("test_")
                or "tests" in rel_path.parts
            ):
                return True

        # Check documentation files
        if not include_docs:
            doc_patterns = ["doc", "docs", "documentation"]
            if any(pattern in str(rel_path).lower() for pattern in doc_patterns):
                return True

        return False

    def _generate_package_metadata(self, package_dir: Path) -> None:
        """Generate package metadata files."""
        # Read plugin manifest
        manifest_file = package_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest_data = json.load(f)
        else:
            # Try to find inline manifest
            manifest_data = self._extract_inline_manifest(package_dir)

        if not manifest_data:
            self.logger.warning("No manifest found, creating minimal metadata")
            manifest_data = {
                "name": package_dir.name,
                "version": "1.0.0",
                "description": "ICARUS CLI Plugin",
            }

        # Generate package.json with extended metadata
        package_metadata = {
            "name": manifest_data.get("name", package_dir.name),
            "version": manifest_data.get("version", "1.0.0"),
            "description": manifest_data.get("description", ""),
            "author": manifest_data.get("author", {}),
            "license": manifest_data.get("license", "MIT"),
            "keywords": manifest_data.get("keywords", []),
            "homepage": manifest_data.get("homepage"),
            "repository": manifest_data.get("repository"),
            "plugin_type": manifest_data.get("type", "utility"),
            "security_level": manifest_data.get("security_level", "safe"),
            "python_version": manifest_data.get("python_version", ">=3.8"),
            "icarus_version": manifest_data.get("icarus_version", ">=1.0.0"),
            "install_requires": manifest_data.get("install_requires", []),
            "package_info": {
                "packaged_at": datetime.now().isoformat(),
                "packager_version": "1.0.0",
            },
        }

        # Write package metadata
        package_file = package_dir / "package.json"
        with open(package_file, "w") as f:
            json.dump(package_metadata, f, indent=2)

        # Generate file manifest
        self._generate_file_manifest(package_dir)

    def _extract_inline_manifest(self, package_dir: Path) -> Optional[Dict[str, Any]]:
        """Extract inline manifest from Python files."""
        for py_file in package_dir.glob("**/*.py"):
            try:
                with open(py_file) as f:
                    content = f.read()

                # Look for PLUGIN_MANIFEST variable
                if "PLUGIN_MANIFEST" in content:
                    # Simple extraction - in practice, would use AST
                    import ast

                    tree = ast.parse(content)

                    for node in ast.walk(tree):
                        if (
                            isinstance(node, ast.Assign)
                            and len(node.targets) == 1
                            and isinstance(node.targets[0], ast.Name)
                            and node.targets[0].id == "PLUGIN_MANIFEST"
                        ):
                            if isinstance(node.value, ast.Dict):
                                return ast.literal_eval(node.value)

            except Exception:
                continue

        return None

    def _generate_file_manifest(self, package_dir: Path) -> None:
        """Generate file manifest with checksums."""
        file_manifest = {"files": {}, "generated_at": datetime.now().isoformat()}

        for file_path in package_dir.rglob("*"):
            if file_path.is_file() and file_path.name != "files.json":
                rel_path = str(file_path.relative_to(package_dir))

                # Calculate file hash
                with open(file_path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()

                file_manifest["files"][rel_path] = {
                    "size": file_path.stat().st_size,
                    "sha256": file_hash,
                    "modified": datetime.fromtimestamp(
                        file_path.stat().st_mtime,
                    ).isoformat(),
                }

        # Write file manifest
        manifest_file = package_dir / "files.json"
        with open(manifest_file, "w") as f:
            json.dump(file_manifest, f, indent=2)

    def _create_package(
        self,
        package_dir: Path,
        output_path: Path,
        format: str,
    ) -> bool:
        """Create the actual package file."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format == "zip":
                return self._create_zip_package(package_dir, output_path)
            elif format in ["tar.gz", "tar.bz2"]:
                return self._create_tar_package(package_dir, output_path, format)
            else:
                self.logger.error(f"Unsupported format: {format}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to create package: {e}")
            return False

    def _create_zip_package(self, package_dir: Path, output_path: Path) -> bool:
        """Create ZIP package."""
        try:
            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in package_dir.rglob("*"):
                    if file_path.is_file():
                        arc_name = str(file_path.relative_to(package_dir.parent))
                        zf.write(file_path, arc_name)

            return True

        except Exception as e:
            self.logger.error(f"Failed to create ZIP package: {e}")
            return False

    def _create_tar_package(
        self,
        package_dir: Path,
        output_path: Path,
        format: str,
    ) -> bool:
        """Create TAR package."""
        try:
            mode = "w:gz" if format == "tar.gz" else "w:bz2"

            with tarfile.open(output_path, mode) as tf:
                tf.add(package_dir, arcname=package_dir.name)

            return True

        except Exception as e:
            self.logger.error(f"Failed to create TAR package: {e}")
            return False

    def _generate_package_info(self, package_path: Path) -> None:
        """Generate package information file."""
        info = {
            "package_path": str(package_path),
            "package_size": package_path.stat().st_size,
            "package_hash": self._calculate_file_hash(package_path),
            "created_at": datetime.now().isoformat(),
        }

        info_file = package_path.with_suffix(package_path.suffix + ".info")
        with open(info_file, "w") as f:
            json.dump(info, f, indent=2)

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def extract_package(self, package_path: str, extract_path: str) -> bool:
        """
        Extract a plugin package.

        Args:
            package_path: Path to package file
            extract_path: Path to extract to

        Returns:
            True if extraction successful, False otherwise
        """
        try:
            package_path = Path(package_path)
            extract_path = Path(extract_path)

            self.logger.info(f"Extracting package: {package_path}")

            extract_path.mkdir(parents=True, exist_ok=True)

            if package_path.suffix == ".zip":
                return self._extract_zip_package(package_path, extract_path)
            elif package_path.suffix in [".gz", ".bz2"] and ".tar" in package_path.name:
                return self._extract_tar_package(package_path, extract_path)
            else:
                self.logger.error(f"Unsupported package format: {package_path.suffix}")
                return False

        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            return False

    def _extract_zip_package(self, package_path: Path, extract_path: Path) -> bool:
        """Extract ZIP package."""
        try:
            with zipfile.ZipFile(package_path, "r") as zf:
                zf.extractall(extract_path)

            self.logger.info(f"Package extracted to: {extract_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to extract ZIP package: {e}")
            return False

    def _extract_tar_package(self, package_path: Path, extract_path: Path) -> bool:
        """Extract TAR package."""
        try:
            with tarfile.open(package_path, "r:*") as tf:
                tf.extractall(extract_path)

            self.logger.info(f"Package extracted to: {extract_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to extract TAR package: {e}")
            return False

    def verify_package(self, package_path: str) -> Dict[str, Any]:
        """
        Verify package integrity.

        Args:
            package_path: Path to package file

        Returns:
            Verification results
        """
        result = {"valid": False, "errors": [], "warnings": [], "info": []}

        try:
            package_path = Path(package_path)

            if not package_path.exists():
                result["errors"].append("Package file does not exist")
                return result

            # Extract to temporary directory for verification
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                if not self.extract_package(str(package_path), str(temp_path)):
                    result["errors"].append("Failed to extract package")
                    return result

                # Find extracted plugin directory
                extracted_dirs = [d for d in temp_path.iterdir() if d.is_dir()]
                if not extracted_dirs:
                    result["errors"].append("No plugin directory found in package")
                    return result

                plugin_dir = extracted_dirs[0]

                # Verify file manifest if present
                files_manifest = plugin_dir / "files.json"
                if files_manifest.exists():
                    if not self._verify_file_manifest(
                        plugin_dir,
                        files_manifest,
                        result,
                    ):
                        return result

                # Verify plugin structure
                from .validator import PluginValidator

                validator = PluginValidator()
                validation_result = validator.validate_plugin(str(plugin_dir))

                if validation_result.is_valid:
                    result["valid"] = True
                    result["info"].append("Package verification successful")
                else:
                    result["errors"].extend(validation_result.errors)
                    result["warnings"].extend(validation_result.warnings)

        except Exception as e:
            result["errors"].append(f"Verification failed: {e}")

        return result

    def _verify_file_manifest(
        self,
        plugin_dir: Path,
        manifest_file: Path,
        result: Dict[str, Any],
    ) -> bool:
        """Verify file manifest checksums."""
        try:
            with open(manifest_file) as f:
                manifest = json.load(f)

            files_data = manifest.get("files", {})

            for rel_path, file_info in files_data.items():
                file_path = plugin_dir / rel_path

                if not file_path.exists():
                    result["errors"].append(f"Missing file: {rel_path}")
                    return False

                # Verify file size
                actual_size = file_path.stat().st_size
                expected_size = file_info.get("size")

                if expected_size and actual_size != expected_size:
                    result["errors"].append(
                        f"Size mismatch for {rel_path}: expected {expected_size}, got {actual_size}",
                    )
                    return False

                # Verify file hash
                expected_hash = file_info.get("sha256")
                if expected_hash:
                    actual_hash = self._calculate_file_hash(file_path)
                    if actual_hash != expected_hash:
                        result["errors"].append(f"Hash mismatch for {rel_path}")
                        return False

            result["info"].append("File manifest verification successful")
            return True

        except Exception as e:
            result["errors"].append(f"File manifest verification failed: {e}")
            return False

    def get_package_info(self, package_path: str) -> Dict[str, Any]:
        """
        Get information about a package.

        Args:
            package_path: Path to package file

        Returns:
            Package information dictionary
        """
        info = {
            "path": package_path,
            "exists": False,
            "size": 0,
            "format": "unknown",
            "manifest": None,
            "files": [],
        }

        try:
            package_path = Path(package_path)

            if not package_path.exists():
                return info

            info["exists"] = True
            info["size"] = package_path.stat().st_size

            # Determine format
            if package_path.suffix == ".zip":
                info["format"] = "zip"
            elif ".tar" in package_path.name:
                if package_path.suffix == ".gz":
                    info["format"] = "tar.gz"
                elif package_path.suffix == ".bz2":
                    info["format"] = "tar.bz2"

            # Extract manifest information
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                if self.extract_package(str(package_path), str(temp_path)):
                    extracted_dirs = [d for d in temp_path.iterdir() if d.is_dir()]
                    if extracted_dirs:
                        plugin_dir = extracted_dirs[0]

                        # Read manifest
                        manifest_file = plugin_dir / "manifest.json"
                        if manifest_file.exists():
                            with open(manifest_file) as f:
                                info["manifest"] = json.load(f)

                        # List files
                        info["files"] = [
                            str(f.relative_to(plugin_dir))
                            for f in plugin_dir.rglob("*")
                            if f.is_file()
                        ]

        except Exception as e:
            info["error"] = str(e)

        return info
