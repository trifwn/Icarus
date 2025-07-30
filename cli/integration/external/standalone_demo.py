"""Standalone demo script for external tool integration system."""

import asyncio
import json
import tempfile
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional


# Simplified models for demo
class CADFormat(Enum):
    STEP = "step"
    IGES = "iges"
    STL = "stl"
    OBJ = "obj"
    PLY = "ply"


class ValidationStatus(Enum):
    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"


class ExportFormatType(Enum):
    JSON = "json"
    CSV = "csv"
    XML = "xml"


@dataclass
class GeometryInfo:
    vertices: int
    faces: int
    edges: int


@dataclass
class ValidationIssue:
    severity: ValidationStatus
    message: str


@dataclass
class GeometryValidationResult:
    status: ValidationStatus
    issues: List[ValidationIssue] = field(default_factory=list)
    geometry_info: Optional[GeometryInfo] = None


@dataclass
class CADFile:
    path: Path
    format: CADFormat
    name: str
    size: int
    validation_result: Optional[GeometryValidationResult] = None

    @property
    def is_valid(self) -> bool:
        return (
            self.validation_result is not None
            and self.validation_result.status != ValidationStatus.ERROR
        )


# Simplified CAD Integration
class SimpleCADIntegration:
    def __init__(self):
        self.supported_formats = {
            ".stl": CADFormat.STL,
            ".ply": CADFormat.PLY,
            ".obj": CADFormat.OBJ,
        }

    def detect_format(self, file_path: Path) -> Optional[CADFormat]:
        extension = file_path.suffix.lower()
        return self.supported_formats.get(extension)

    async def import_cad_file(self, file_path: Path) -> CADFile:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        cad_format = self.detect_format(file_path)
        if not cad_format:
            raise ValueError(f"Unsupported format: {file_path.suffix}")

        stat = file_path.stat()
        cad_file = CADFile(
            path=file_path,
            format=cad_format,
            name=file_path.name,
            size=stat.st_size,
        )

        # Simple validation
        cad_file.validation_result = await self._validate_file(cad_file)
        return cad_file

    async def _validate_file(self, cad_file: CADFile) -> GeometryValidationResult:
        issues = []

        try:
            with open(cad_file.path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if cad_file.format == CADFormat.STL:
                if content.strip().startswith("solid"):
                    # ASCII STL
                    vertices = content.count("vertex")
                    faces = content.count("facet normal")

                    geometry_info = GeometryInfo(
                        vertices=vertices,
                        faces=faces,
                        edges=faces * 3,
                    )

                    if vertices == 0:
                        issues.append(
                            ValidationIssue(
                                severity=ValidationStatus.WARNING,
                                message="No vertices found",
                            ),
                        )

                    status = (
                        ValidationStatus.VALID
                        if not issues
                        else ValidationStatus.WARNING
                    )

                else:
                    # Binary STL - simplified check
                    geometry_info = GeometryInfo(
                        vertices=100,
                        faces=50,
                        edges=150,
                    )  # Mock values
                    status = ValidationStatus.VALID

            elif cad_file.format == CADFormat.PLY:
                if content.startswith("ply"):
                    # Parse header for vertex/face count
                    lines = content.split("\n")
                    vertices = 0
                    faces = 0

                    for line in lines:
                        if line.startswith("element vertex"):
                            vertices = int(line.split()[-1])
                        elif line.startswith("element face"):
                            faces = int(line.split()[-1])

                    geometry_info = GeometryInfo(
                        vertices=vertices,
                        faces=faces,
                        edges=0,
                    )
                    status = ValidationStatus.VALID
                else:
                    issues.append(
                        ValidationIssue(
                            severity=ValidationStatus.ERROR,
                            message="Invalid PLY header",
                        ),
                    )
                    geometry_info = None
                    status = ValidationStatus.ERROR

            else:
                # Default validation
                geometry_info = GeometryInfo(
                    vertices=50,
                    faces=25,
                    edges=75,
                )  # Mock values
                status = ValidationStatus.VALID

        except Exception as e:
            issues.append(
                ValidationIssue(
                    severity=ValidationStatus.ERROR,
                    message=f"Validation error: {str(e)}",
                ),
            )
            geometry_info = None
            status = ValidationStatus.ERROR

        return GeometryValidationResult(
            status=status,
            issues=issues,
            geometry_info=geometry_info,
        )


# Simplified Export Manager
class SimpleExportManager:
    def __init__(self):
        self.supported_formats = [
            ExportFormatType.JSON,
            ExportFormatType.CSV,
            ExportFormatType.XML,
        ]

    async def export_data(
        self,
        data: Any,
        output_path: Path,
        format_type: ExportFormatType,
    ) -> Dict[str, Any]:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format_type == ExportFormatType.JSON:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)

            elif format_type == ExportFormatType.XML:
                import xml.etree.ElementTree as ET

                root = ET.Element("data")
                self._dict_to_xml(data, root)
                tree = ET.ElementTree(root)
                tree.write(output_path, encoding="utf-8", xml_declaration=True)

            elif format_type == ExportFormatType.CSV:
                # Simple CSV export for demo
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write("key,value\n")
                    if isinstance(data, dict):
                        for key, value in data.items():
                            f.write(f"{key},{value}\n")

            file_size = output_path.stat().st_size
            return {"success": True, "output_path": output_path, "file_size": file_size}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _dict_to_xml(self, data: Dict[str, Any], parent):
        import xml.etree.ElementTree as ET

        for key, value in data.items():
            elem = ET.SubElement(parent, str(key))
            if isinstance(value, dict):
                self._dict_to_xml(value, elem)
            else:
                elem.text = str(value)


# Demo functions
async def demo_cad_integration():
    print("=== CAD Integration Demo ===")

    cad_integration = SimpleCADIntegration()

    # Create a sample STL file
    stl_content = """solid demo
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
  endloop
endfacet
facet normal 0 0 1
  outer loop
    vertex 1 0 0
    vertex 1 1 0
    vertex 0 1 0
  endloop
endfacet
endsolid demo"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".stl", delete=False) as f:
        f.write(stl_content)
        f.flush()
        stl_path = Path(f.name)

    try:
        print(f"Importing CAD file: {stl_path.name}")

        cad_file = await cad_integration.import_cad_file(stl_path)

        print(f"✓ Format: {cad_file.format.value}")
        print(f"✓ Size: {cad_file.size} bytes")
        print(f"✓ Valid: {cad_file.is_valid}")

        if cad_file.validation_result:
            print(f"✓ Validation status: {cad_file.validation_result.status.value}")
            if cad_file.validation_result.geometry_info:
                geo = cad_file.validation_result.geometry_info
                print(
                    f"✓ Geometry - Vertices: {geo.vertices}, Faces: {geo.faces}, Edges: {geo.edges}",
                )

            if cad_file.validation_result.issues:
                print("Issues found:")
                for issue in cad_file.validation_result.issues:
                    print(f"  - {issue.severity.value}: {issue.message}")

    finally:
        stl_path.unlink()

    print("CAD integration demo completed!\n")


async def demo_export_manager():
    print("=== Export Manager Demo ===")

    export_manager = SimpleExportManager()

    # Sample data to export
    sample_data = {
        "analysis_results": {
            "lift_coefficient": 0.85,
            "drag_coefficient": 0.02,
            "pressure_distribution": [1.0, 0.8, 0.6, 0.4, 0.2],
        },
        "metadata": {
            "analysis_type": "airfoil_analysis",
            "solver": "xfoil",
            "timestamp": "2023-01-01T12:00:00Z",
        },
    }

    # Export to different formats
    for format_type in [ExportFormatType.JSON, ExportFormatType.XML]:
        with tempfile.NamedTemporaryFile(
            suffix=f".{format_type.value}",
            delete=False,
        ) as f:
            output_path = Path(f.name)

        try:
            print(f"Exporting to {format_type.value.upper()}...")

            result = await export_manager.export_data(
                sample_data,
                output_path,
                format_type,
            )

            if result["success"]:
                print(f"✓ Export successful: {output_path.name}")
                print(f"  File size: {result['file_size']} bytes")

                # Show preview for JSON
                if format_type == ExportFormatType.JSON:
                    with open(output_path) as f:
                        content = f.read()[:200]
                        print(f"  Preview: {content}...")
            else:
                print(f"✗ Export failed: {result['error']}")

        finally:
            if output_path.exists():
                output_path.unlink()

    print("Export manager demo completed!\n")


async def demo_integration_workflow():
    print("=== Complete Integration Workflow Demo ===")

    print("Workflow: CAD Import → Data Processing → Multi-format Export")

    # Step 1: CAD Import
    print("\n1. Importing CAD file...")
    cad_integration = SimpleCADIntegration()

    # Create sample PLY file
    ply_content = """ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 2
property list uchar int vertex_indices
end_header
0.0 0.0 0.0
1.0 0.0 0.0
0.5 1.0 0.0
0.5 0.5 1.0
3 0 1 2
3 0 2 3
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ply", delete=False) as f:
        f.write(ply_content)
        f.flush()
        ply_path = Path(f.name)

    try:
        cad_file = await cad_integration.import_cad_file(ply_path)
        print(
            f"✓ Imported {cad_file.format.value} file with {cad_file.validation_result.geometry_info.vertices} vertices",
        )

        # Step 2: Process data
        print("\n2. Processing geometry data...")
        processed_data = {
            "geometry": {
                "format": cad_file.format.value,
                "vertices": cad_file.validation_result.geometry_info.vertices,
                "faces": cad_file.validation_result.geometry_info.faces,
                "file_size": cad_file.size,
            },
            "validation": {
                "status": cad_file.validation_result.status.value,
                "issues_count": len(cad_file.validation_result.issues),
                "is_valid": cad_file.is_valid,
            },
            "analysis": {"surface_area": 2.5, "volume": 0.33, "quality_score": 0.95},
        }
        print("✓ Processed geometry and generated analysis data")

        # Step 3: Multi-format export
        print("\n3. Exporting to multiple formats...")
        export_manager = SimpleExportManager()

        export_results = []
        for format_type in [ExportFormatType.JSON, ExportFormatType.XML]:
            with tempfile.NamedTemporaryFile(
                suffix=f".{format_type.value}",
                delete=False,
            ) as f:
                output_path = Path(f.name)

            try:
                result = await export_manager.export_data(
                    processed_data,
                    output_path,
                    format_type,
                )

                if result["success"]:
                    export_results.append((format_type.value, result["file_size"]))
                    print(
                        f"✓ Exported {format_type.value.upper()}: {result['file_size']} bytes",
                    )
                else:
                    print(f"✗ Failed to export {format_type.value.upper()}")

            finally:
                if output_path.exists():
                    output_path.unlink()

        # Step 4: Summary
        print("\n4. Workflow Summary:")
        print(f"   - Input: {cad_file.format.value} file ({cad_file.size} bytes)")
        print(f"   - Validation: {cad_file.validation_result.status.value}")
        print(f"   - Exports: {len(export_results)} formats")
        for format_name, size in export_results:
            print(f"     * {format_name.upper()}: {size} bytes")

        print("✓ Complete workflow executed successfully!")

    finally:
        ply_path.unlink()

    print("Integration workflow demo completed!\n")


async def main():
    """Run all demos."""
    print("ICARUS CLI - External Tool Integration System Demo")
    print("=" * 50)

    try:
        await demo_cad_integration()
        await demo_export_manager()
        await demo_integration_workflow()

        print("🎉 All demos completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✓ CAD file import with format detection")
        print("✓ Geometry validation with detailed reporting")
        print("✓ Multi-format data export (JSON, XML)")
        print("✓ Complete integration workflow")
        print("✓ Error handling and validation")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
