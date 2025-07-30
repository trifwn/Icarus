"""CAD file integration with geometry validation."""

import asyncio
import logging
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

from .models import CADFile
from .models import CADFormat
from .models import GeometryInfo
from .models import GeometryValidationResult
from .models import ValidationIssue
from .models import ValidationStatus


class CADIntegration:
    """Handles CAD file import and geometry validation."""

    SUPPORTED_FORMATS = {
        ".step": CADFormat.STEP,
        ".stp": CADFormat.STEP,
        ".iges": CADFormat.IGES,
        ".igs": CADFormat.IGES,
        ".stl": CADFormat.STL,
        ".obj": CADFormat.OBJ,
        ".ply": CADFormat.PLY,
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._validators = {}
        self._setup_validators()

    def _setup_validators(self):
        """Set up format-specific validators."""
        self._validators = {
            CADFormat.STEP: self._validate_step_file,
            CADFormat.IGES: self._validate_iges_file,
            CADFormat.STL: self._validate_stl_file,
            CADFormat.OBJ: self._validate_obj_file,
            CADFormat.PLY: self._validate_ply_file,
        }

    def detect_format(self, file_path: Path) -> Optional[CADFormat]:
        """Detect CAD file format from extension and content."""
        if not file_path.exists():
            return None

        # Check by extension first
        extension = file_path.suffix.lower()
        if extension in self.SUPPORTED_FORMATS:
            return self.SUPPORTED_FORMATS[extension]

        # Try to detect by content
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                first_line = f.readline().strip().upper()

                if first_line.startswith("ISO-10303"):
                    return CADFormat.STEP
                elif first_line.startswith("START"):
                    return CADFormat.IGES
                elif "solid" in first_line.lower():
                    return CADFormat.STL
                elif first_line.startswith("ply"):
                    return CADFormat.PLY
        except Exception:
            pass

        return None

    async def import_cad_file(self, file_path: Union[str, Path]) -> CADFile:
        """Import and validate a CAD file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"CAD file not found: {file_path}")

        # Detect format
        cad_format = self.detect_format(file_path)
        if not cad_format:
            raise ValueError(f"Unsupported CAD file format: {file_path}")

        # Get file info
        stat = file_path.stat()

        # Create CAD file object
        cad_file = CADFile(
            path=file_path,
            format=cad_format,
            name=file_path.name,
            size=stat.st_size,
            created_at=stat.st_ctime,
            modified_at=stat.st_mtime,
        )

        # Validate geometry
        cad_file.validation_result = await self.validate_geometry(cad_file)

        self.logger.info(f"Imported CAD file: {file_path} ({cad_format.value})")
        return cad_file

    async def validate_geometry(self, cad_file: CADFile) -> GeometryValidationResult:
        """Validate geometry in CAD file."""
        validator = self._validators.get(cad_file.format)
        if not validator:
            return GeometryValidationResult(
                status=ValidationStatus.UNKNOWN,
                issues=[
                    ValidationIssue(
                        severity=ValidationStatus.WARNING,
                        message=f"No validator available for {cad_file.format.value} format",
                    ),
                ],
            )

        try:
            return await validator(cad_file)
        except Exception as e:
            self.logger.error(f"Validation error for {cad_file.path}: {e}")
            return GeometryValidationResult(
                status=ValidationStatus.ERROR,
                issues=[
                    ValidationIssue(
                        severity=ValidationStatus.ERROR,
                        message=f"Validation failed: {str(e)}",
                    ),
                ],
            )

    async def _validate_step_file(self, cad_file: CADFile) -> GeometryValidationResult:
        """Validate STEP file format."""
        issues = []
        geometry_info = None

        try:
            with open(cad_file.path, encoding="utf-8") as f:
                content = f.read()

            # Basic STEP file validation
            if not content.startswith("ISO-10303"):
                issues.append(
                    ValidationIssue(
                        severity=ValidationStatus.ERROR,
                        message="Invalid STEP file header",
                    ),
                )

            if "ENDSEC;" not in content:
                issues.append(
                    ValidationIssue(
                        severity=ValidationStatus.ERROR,
                        message="Missing ENDSEC marker",
                    ),
                )

            # Count entities (basic geometry analysis)
            vertices = content.count("CARTESIAN_POINT")
            faces = content.count("FACE_SURFACE")
            edges = content.count("EDGE_CURVE")

            if vertices > 0:
                geometry_info = GeometryInfo(
                    vertices=vertices,
                    faces=faces,
                    edges=edges,
                )

            # Check for common issues
            if vertices == 0:
                issues.append(
                    ValidationIssue(
                        severity=ValidationStatus.WARNING,
                        message="No vertices found in geometry",
                    ),
                )

            status = (
                ValidationStatus.ERROR
                if any(issue.severity == ValidationStatus.ERROR for issue in issues)
                else ValidationStatus.VALID
            )

        except UnicodeDecodeError:
            issues.append(
                ValidationIssue(
                    severity=ValidationStatus.ERROR,
                    message="File encoding error - not a valid text STEP file",
                ),
            )
            status = ValidationStatus.ERROR
        except Exception as e:
            issues.append(
                ValidationIssue(
                    severity=ValidationStatus.ERROR,
                    message=f"Validation error: {str(e)}",
                ),
            )
            status = ValidationStatus.ERROR

        return GeometryValidationResult(
            status=status,
            issues=issues,
            geometry_info=geometry_info,
        )

    async def _validate_iges_file(self, cad_file: CADFile) -> GeometryValidationResult:
        """Validate IGES file format."""
        issues = []
        geometry_info = None

        try:
            with open(cad_file.path, encoding="utf-8") as f:
                lines = f.readlines()

            if not lines:
                issues.append(
                    ValidationIssue(
                        severity=ValidationStatus.ERROR,
                        message="Empty IGES file",
                    ),
                )
                return GeometryValidationResult(
                    status=ValidationStatus.ERROR,
                    issues=issues,
                )

            # Check IGES structure
            start_found = False
            global_found = False
            directory_found = False
            parameter_found = False
            terminate_found = False

            for line in lines:
                if line.rstrip().endswith("S"):
                    start_found = True
                elif line.rstrip().endswith("G"):
                    global_found = True
                elif line.rstrip().endswith("D"):
                    directory_found = True
                elif line.rstrip().endswith("P"):
                    parameter_found = True
                elif line.rstrip().endswith("T"):
                    terminate_found = True

            if not all(
                [
                    start_found,
                    global_found,
                    directory_found,
                    parameter_found,
                    terminate_found,
                ],
            ):
                issues.append(
                    ValidationIssue(
                        severity=ValidationStatus.ERROR,
                        message="Invalid IGES file structure",
                    ),
                )

            # Basic geometry counting
            directory_entries = sum(1 for line in lines if line.rstrip().endswith("D"))
            parameter_entries = sum(1 for line in lines if line.rstrip().endswith("P"))

            geometry_info = GeometryInfo(
                vertices=0,  # Would need IGES parser for accurate count
                faces=directory_entries // 2,  # Rough estimate
                edges=0,
            )

            status = (
                ValidationStatus.ERROR
                if any(issue.severity == ValidationStatus.ERROR for issue in issues)
                else ValidationStatus.VALID
            )

        except Exception as e:
            issues.append(
                ValidationIssue(
                    severity=ValidationStatus.ERROR,
                    message=f"Validation error: {str(e)}",
                ),
            )
            status = ValidationStatus.ERROR

        return GeometryValidationResult(
            status=status,
            issues=issues,
            geometry_info=geometry_info,
        )

    async def _validate_stl_file(self, cad_file: CADFile) -> GeometryValidationResult:
        """Validate STL file format."""
        issues = []
        geometry_info = None

        try:
            # Check if binary or ASCII STL
            with open(cad_file.path, "rb") as f:
                header = f.read(80)

            is_binary = not header.startswith(b"solid")

            if is_binary:
                # Binary STL validation
                with open(cad_file.path, "rb") as f:
                    f.seek(80)  # Skip header
                    triangle_count_bytes = f.read(4)
                    if len(triangle_count_bytes) != 4:
                        issues.append(
                            ValidationIssue(
                                severity=ValidationStatus.ERROR,
                                message="Invalid binary STL header",
                            ),
                        )
                    else:
                        triangle_count = int.from_bytes(triangle_count_bytes, "little")
                        expected_size = 80 + 4 + (triangle_count * 50)

                        if cad_file.size != expected_size:
                            issues.append(
                                ValidationIssue(
                                    severity=ValidationStatus.WARNING,
                                    message=f"File size mismatch: expected {expected_size}, got {cad_file.size}",
                                ),
                            )

                        geometry_info = GeometryInfo(
                            vertices=triangle_count * 3,
                            faces=triangle_count,
                            edges=triangle_count * 3,
                        )
            else:
                # ASCII STL validation
                with open(cad_file.path, encoding="utf-8") as f:
                    content = f.read()

                if not content.strip().startswith("solid"):
                    issues.append(
                        ValidationIssue(
                            severity=ValidationStatus.ERROR,
                            message="Invalid ASCII STL header",
                        ),
                    )

                if not content.strip().endswith("endsolid"):
                    issues.append(
                        ValidationIssue(
                            severity=ValidationStatus.WARNING,
                            message="Missing endsolid marker",
                        ),
                    )

                # Count triangles
                triangle_count = content.count("facet normal")
                vertex_count = content.count("vertex")

                if vertex_count != triangle_count * 3:
                    issues.append(
                        ValidationIssue(
                            severity=ValidationStatus.WARNING,
                            message=f"Vertex count mismatch: {vertex_count} vertices for {triangle_count} triangles",
                        ),
                    )

                geometry_info = GeometryInfo(
                    vertices=vertex_count,
                    faces=triangle_count,
                    edges=triangle_count * 3,
                )

            status = (
                ValidationStatus.ERROR
                if any(issue.severity == ValidationStatus.ERROR for issue in issues)
                else ValidationStatus.VALID
            )

        except Exception as e:
            issues.append(
                ValidationIssue(
                    severity=ValidationStatus.ERROR,
                    message=f"Validation error: {str(e)}",
                ),
            )
            status = ValidationStatus.ERROR

        return GeometryValidationResult(
            status=status,
            issues=issues,
            geometry_info=geometry_info,
        )

    async def _validate_obj_file(self, cad_file: CADFile) -> GeometryValidationResult:
        """Validate OBJ file format."""
        issues = []
        geometry_info = None

        try:
            with open(cad_file.path, encoding="utf-8") as f:
                lines = f.readlines()

            vertices = 0
            faces = 0
            normals = 0
            texture_coords = 0

            for line in lines:
                line = line.strip()
                if line.startswith("v "):
                    vertices += 1
                elif line.startswith("f "):
                    faces += 1
                elif line.startswith("vn "):
                    normals += 1
                elif line.startswith("vt "):
                    texture_coords += 1

            if vertices == 0:
                issues.append(
                    ValidationIssue(
                        severity=ValidationStatus.WARNING,
                        message="No vertices found in OBJ file",
                    ),
                )

            if faces == 0:
                issues.append(
                    ValidationIssue(
                        severity=ValidationStatus.WARNING,
                        message="No faces found in OBJ file",
                    ),
                )

            geometry_info = GeometryInfo(
                vertices=vertices,
                faces=faces,
                edges=0,  # Would need face parsing for accurate edge count
            )

            status = ValidationStatus.VALID

        except Exception as e:
            issues.append(
                ValidationIssue(
                    severity=ValidationStatus.ERROR,
                    message=f"Validation error: {str(e)}",
                ),
            )
            status = ValidationStatus.ERROR

        return GeometryValidationResult(
            status=status,
            issues=issues,
            geometry_info=geometry_info,
        )

    async def _validate_ply_file(self, cad_file: CADFile) -> GeometryValidationResult:
        """Validate PLY file format."""
        issues = []
        geometry_info = None

        try:
            with open(cad_file.path, encoding="utf-8") as f:
                lines = f.readlines()

            if not lines or not lines[0].strip().startswith("ply"):
                issues.append(
                    ValidationIssue(
                        severity=ValidationStatus.ERROR,
                        message="Invalid PLY file header",
                    ),
                )
                return GeometryValidationResult(
                    status=ValidationStatus.ERROR,
                    issues=issues,
                )

            # Parse header
            vertices = 0
            faces = 0
            in_header = True

            for line in lines:
                line = line.strip()
                if line == "end_header":
                    break
                elif line.startswith("element vertex"):
                    vertices = int(line.split()[-1])
                elif line.startswith("element face"):
                    faces = int(line.split()[-1])

            geometry_info = GeometryInfo(vertices=vertices, faces=faces, edges=0)

            if vertices == 0:
                issues.append(
                    ValidationIssue(
                        severity=ValidationStatus.WARNING,
                        message="No vertices defined in PLY header",
                    ),
                )

            status = (
                ValidationStatus.ERROR
                if any(issue.severity == ValidationStatus.ERROR for issue in issues)
                else ValidationStatus.VALID
            )

        except Exception as e:
            issues.append(
                ValidationIssue(
                    severity=ValidationStatus.ERROR,
                    message=f"Validation error: {str(e)}",
                ),
            )
            status = ValidationStatus.ERROR

        return GeometryValidationResult(
            status=status,
            issues=issues,
            geometry_info=geometry_info,
        )

    def get_supported_formats(self) -> List[CADFormat]:
        """Get list of supported CAD formats."""
        return list(CADFormat)

    def is_format_supported(self, format_or_extension: Union[CADFormat, str]) -> bool:
        """Check if a format is supported."""
        if isinstance(format_or_extension, CADFormat):
            return format_or_extension in self._validators
        else:
            return format_or_extension.lower() in self.SUPPORTED_FORMATS

    async def batch_import(self, file_paths: List[Union[str, Path]]) -> List[CADFile]:
        """Import multiple CAD files concurrently."""
        tasks = [self.import_cad_file(path) for path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        cad_files = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Failed to import CAD file: {result}")
            else:
                cad_files.append(result)

        return cad_files
