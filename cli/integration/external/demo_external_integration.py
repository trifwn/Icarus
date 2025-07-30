"""Demo script for external tool integration system."""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add the CLI directory to the path
cli_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(cli_dir))

from integration.external.api_adapter import APIAdapter
from integration.external.cad_integration import CADIntegration
from integration.external.cloud_integration import CloudIntegration
from integration.external.export_manager import ExportManager
from integration.external.models import APIEndpoint
from integration.external.models import AuthenticationConfig
from integration.external.models import AuthenticationType
from integration.external.models import CloudService
from integration.external.models import CloudServiceType
from integration.external.models import ExternalToolConfig


async def demo_cad_integration():
    """Demonstrate CAD file integration."""
    print("=== CAD Integration Demo ===")

    cad_integration = CADIntegration()

    # Show supported formats
    formats = cad_integration.get_supported_formats()
    print(f"Supported CAD formats: {[f.value for f in formats]}")

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
        print(f"\nImporting CAD file: {stl_path.name}")

        # Import and validate CAD file
        cad_file = await cad_integration.import_cad_file(stl_path)

        print(f"Format: {cad_file.format.value}")
        print(f"Size: {cad_file.size} bytes")
        print(f"Valid: {cad_file.is_valid}")

        if cad_file.validation_result:
            print(f"Validation status: {cad_file.validation_result.status.value}")
            if cad_file.validation_result.geometry_info:
                geo = cad_file.validation_result.geometry_info
                print(
                    f"Geometry - Vertices: {geo.vertices}, Faces: {geo.faces}, Edges: {geo.edges}",
                )

            if cad_file.validation_result.issues:
                print("Issues found:")
                for issue in cad_file.validation_result.issues:
                    print(f"  - {issue.severity.value}: {issue.message}")

    finally:
        stl_path.unlink()

    print("CAD integration demo completed!\n")


async def demo_export_manager():
    """Demonstrate export manager."""
    print("=== Export Manager Demo ===")

    export_manager = ExportManager()

    # Show supported formats
    formats = export_manager.get_supported_formats()
    print(f"Supported export formats: {[f.type.value for f in formats]}")

    # Sample data to export
    sample_data = {
        "analysis_results": {
            "lift_coefficient": 0.85,
            "drag_coefficient": 0.02,
            "pressure_distribution": [1.0, 0.8, 0.6, 0.4, 0.2],
            "velocity_field": [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]],
        },
        "metadata": {
            "analysis_type": "airfoil_analysis",
            "solver": "xfoil",
            "timestamp": "2023-01-01T12:00:00Z",
        },
    }

    # Export to different formats
    formats_to_test = [("JSON", "json"), ("CSV", "csv"), ("XML", "xml")]

    for format_name, format_ext in formats_to_test:
        with tempfile.NamedTemporaryFile(suffix=f".{format_ext}", delete=False) as f:
            output_path = Path(f.name)

        try:
            print(f"\nExporting to {format_name}...")

            from integration.external.models import ExportFormatType

            format_type = getattr(ExportFormatType, format_ext.upper())

            result = await export_manager.export_data(
                sample_data,
                output_path,
                format_type,
            )

            if result.success:
                print(f"✓ Export successful: {output_path.name}")
                print(f"  File size: {result.file_size} bytes")

                # Show a preview of the content
                if format_ext == "json":
                    with open(output_path) as f:
                        content = f.read()[:200]
                        print(f"  Preview: {content}...")
            else:
                print(f"✗ Export failed: {result.errors}")

        finally:
            if output_path.exists():
                output_path.unlink()

    print("Export manager demo completed!\n")


async def demo_cloud_integration():
    """Demonstrate cloud integration."""
    print("=== Cloud Integration Demo ===")

    # Create mock cloud services
    services = [
        CloudService(
            name="aws-s3",
            type=CloudServiceType.AWS_S3,
            auth_config=AuthenticationConfig(
                type=AuthenticationType.API_KEY,
                credentials={"api_key": "demo-aws-key"},
            ),
            base_url="https://s3.amazonaws.com",
            bucket_name="icarus-data",
        ),
        CloudService(
            name="google-cloud",
            type=CloudServiceType.GOOGLE_CLOUD,
            auth_config=AuthenticationConfig(
                type=AuthenticationType.OAUTH2,
                credentials={
                    "client_id": "demo-client-id",
                    "client_secret": "demo-client-secret",
                },
            ),
            base_url="https://storage.googleapis.com",
            bucket_name="icarus-storage",
        ),
    ]

    async with CloudIntegration() as cloud_integration:
        # Register services
        for service in services:
            cloud_integration.register_service(service)
            print(f"Registered cloud service: {service.name} ({service.type.value})")

        # List services
        registered_services = cloud_integration.list_services()
        print(f"\nRegistered services: {[s.name for s in registered_services]}")

        # Note: Authentication would fail with demo credentials
        # In a real scenario, you would have valid credentials
        print("\nNote: Authentication demo skipped (requires valid credentials)")

    print("Cloud integration demo completed!\n")


async def demo_api_adapter():
    """Demonstrate API adapter."""
    print("=== API Adapter Demo ===")

    # Create mock external tool configuration
    auth_config = AuthenticationConfig(
        type=AuthenticationType.API_KEY,
        credentials={"api_key": "demo-api-key"},
    )

    endpoints = [
        APIEndpoint(
            name="version",
            url="https://api.example-cad-tool.com/v1/version",
            auth_config=auth_config,
        ),
        APIEndpoint(
            name="import",
            url="https://api.example-cad-tool.com/v1/import",
            method="POST",
            auth_config=auth_config,
        ),
        APIEndpoint(
            name="export",
            url="https://api.example-cad-tool.com/v1/export",
            method="POST",
            auth_config=auth_config,
        ),
    ]

    tool_config = ExternalToolConfig(
        name="example-cad-tool",
        api_endpoints=endpoints,
        supported_formats=["step", "iges", "stl"],
        version="1.0.0",
        is_available=True,
    )

    async with APIAdapter() as api_adapter:
        # Register tool
        api_adapter.register_tool(tool_config)
        print(f"Registered external tool: {tool_config.name}")

        # Register a version adapter
        def version_adapter(data, direction):
            """Example version adapter."""
            if direction == "request":
                # Adapt request for newer API version
                if "format" in data:
                    data["file_format"] = data.pop(
                        "format",
                    )  # API v2 uses 'file_format'
            elif direction == "response":
                # Adapt response from newer API version
                if "file_format" in data:
                    data["format"] = data.pop(
                        "file_format",
                    )  # Convert back to v1 format
            return data

        api_adapter.register_adapter(
            "example-cad-tool",
            "1.0.0->2.0.0",
            version_adapter,
        )
        print("Registered version adapter for API migration")

        # Get tool status
        status = api_adapter.get_tool_status("example-cad-tool")
        print("\nTool status:")
        for key, value in status.items():
            print(f"  {key}: {value}")

        # List registered tools
        tools = api_adapter.list_registered_tools()
        print(f"\nRegistered tools: {tools}")

        print("\nNote: API calls demo skipped (requires valid endpoints)")

    print("API adapter demo completed!\n")


async def demo_integration_workflow():
    """Demonstrate complete integration workflow."""
    print("=== Complete Integration Workflow Demo ===")

    # Simulate a complete workflow:
    # 1. Import CAD file
    # 2. Process geometry data
    # 3. Export results in multiple formats
    # 4. Prepare for cloud upload

    print("Workflow: CAD Import → Data Processing → Multi-format Export")

    # Step 1: CAD Import
    print("\n1. Importing CAD file...")
    cad_integration = CADIntegration()

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
            "analysis": {
                "surface_area": 2.5,  # Mock analysis result
                "volume": 0.33,  # Mock analysis result
                "quality_score": 0.95,  # Mock quality assessment
            },
        }
        print("✓ Processed geometry and generated analysis data")

        # Step 3: Multi-format export
        print("\n3. Exporting to multiple formats...")
        export_manager = ExportManager()

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

                if result.success:
                    export_results.append((format_type.value, result.file_size))
                    print(
                        f"✓ Exported {format_type.value.upper()}: {result.file_size} bytes",
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
        await demo_cloud_integration()
        await demo_api_adapter()
        await demo_integration_workflow()

        print("🎉 All demos completed successfully!")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
