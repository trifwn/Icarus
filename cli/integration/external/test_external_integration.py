"""Tests for external tool integration system."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from .api_adapter import APIAdapter
from .cad_integration import CADIntegration
from .cloud_integration import CloudIntegration
from .export_manager import ExportManager
from .models import APIEndpoint
from .models import AuthenticationConfig
from .models import AuthenticationType
from .models import CADFile
from .models import CADFormat
from .models import CloudService
from .models import CloudServiceType
from .models import ExportFormatType
from .models import ExternalToolConfig


class TestCADIntegration:
    """Test CAD file integration."""

    @pytest.fixture
    def cad_integration(self):
        return CADIntegration()

    def test_detect_format_by_extension(self, cad_integration):
        """Test format detection by file extension."""
        test_cases = [
            ("test.step", CADFormat.STEP),
            ("test.stp", CADFormat.STEP),
            ("test.iges", CADFormat.IGES),
            ("test.igs", CADFormat.IGES),
            ("test.stl", CADFormat.STL),
            ("test.obj", CADFormat.OBJ),
            ("test.ply", CADFormat.PLY),
            ("test.unknown", None),
        ]

        for filename, expected in test_cases:
            path = Path(filename)
            result = cad_integration.detect_format(path)
            if expected:
                assert result == expected
            else:
                assert result is None

    def test_get_supported_formats(self, cad_integration):
        """Test getting supported formats."""
        formats = cad_integration.get_supported_formats()
        assert len(formats) == 5
        assert CADFormat.STEP in formats
        assert CADFormat.STL in formats

    def test_is_format_supported(self, cad_integration):
        """Test format support checking."""
        assert cad_integration.is_format_supported(CADFormat.STEP)
        assert cad_integration.is_format_supported(".stl")
        assert not cad_integration.is_format_supported(".unknown")

    @pytest.mark.asyncio
    async def test_validate_stl_file(self, cad_integration):
        """Test STL file validation."""
        # Create a simple ASCII STL file
        stl_content = """solid test
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
  endloop
endfacet
endsolid test"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".stl", delete=False) as f:
            f.write(stl_content)
            f.flush()

            try:
                cad_file = CADFile(
                    path=Path(f.name),
                    format=CADFormat.STL,
                    name="test.stl",
                    size=len(stl_content),
                    created_at=0,
                    modified_at=0,
                )

                result = await cad_integration._validate_stl_file(cad_file)
                assert result.status.value in ["valid", "warning"]
                assert result.geometry_info is not None
                assert result.geometry_info.faces == 1
                assert result.geometry_info.vertices == 3
            finally:
                Path(f.name).unlink()


class TestCloudIntegration:
    """Test cloud service integration."""

    @pytest.fixture
    def cloud_integration(self):
        return CloudIntegration()

    @pytest.fixture
    def mock_service(self):
        auth_config = AuthenticationConfig(
            type=AuthenticationType.API_KEY,
            credentials={"api_key": "test-key"},
        )

        return CloudService(
            name="test-service",
            type=CloudServiceType.AWS_S3,
            auth_config=auth_config,
            base_url="https://api.test.com",
            bucket_name="test-bucket",
        )

    def test_register_service(self, cloud_integration, mock_service):
        """Test service registration."""
        cloud_integration.register_service(mock_service)

        retrieved = cloud_integration.get_service("test-service")
        assert retrieved is not None
        assert retrieved.name == "test-service"
        assert retrieved.type == CloudServiceType.AWS_S3

    def test_list_services(self, cloud_integration, mock_service):
        """Test listing services."""
        cloud_integration.register_service(mock_service)

        services = cloud_integration.list_services()
        assert len(services) == 1
        assert services[0].name == "test-service"

    @pytest.mark.asyncio
    async def test_authenticate_api_key(self, cloud_integration, mock_service):
        """Test API key authentication."""
        cloud_integration.register_service(mock_service)

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response

            cloud_integration.session = Mock()
            cloud_integration.session.get = mock_get

            result = await cloud_integration._authenticate_api_key(mock_service)
            assert result is not None
            assert "api_key" in result
            assert result["api_key"] == "test-key"


class TestExportManager:
    """Test export manager."""

    @pytest.fixture
    def export_manager(self):
        return ExportManager()

    def test_get_supported_formats(self, export_manager):
        """Test getting supported export formats."""
        formats = export_manager.get_supported_formats()
        assert len(formats) > 0

        format_types = [f.type for f in formats]
        assert ExportFormatType.JSON in format_types
        assert ExportFormatType.CSV in format_types
        assert ExportFormatType.XML in format_types

    def test_get_format_by_type(self, export_manager):
        """Test getting format by type."""
        json_format = export_manager.get_format_by_type(ExportFormatType.JSON)
        assert json_format is not None
        assert json_format.extension == ".json"
        assert json_format.mime_type == "application/json"

    def test_get_format_by_extension(self, export_manager):
        """Test getting format by extension."""
        json_format = export_manager.get_format_by_extension(".json")
        assert json_format is not None
        assert json_format.type == ExportFormatType.JSON

        csv_format = export_manager.get_format_by_extension("csv")
        assert csv_format is not None
        assert csv_format.type == ExportFormatType.CSV

    @pytest.mark.asyncio
    async def test_export_json(self, export_manager):
        """Test JSON export."""
        test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = Path(f.name)

        try:
            result = await export_manager.export_data(
                test_data,
                output_path,
                ExportFormatType.JSON,
            )

            assert result.success
            assert result.output_path == output_path
            assert output_path.exists()

            # Verify content
            with open(output_path) as f:
                loaded_data = json.load(f)
            assert loaded_data == test_data

        finally:
            if output_path.exists():
                output_path.unlink()

    @pytest.mark.asyncio
    async def test_export_csv(self, export_manager):
        """Test CSV export."""
        test_data = [
            {"name": "Alice", "age": 30, "city": "New York"},
            {"name": "Bob", "age": 25, "city": "San Francisco"},
        ]

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = Path(f.name)

        try:
            result = await export_manager.export_data(
                test_data,
                output_path,
                ExportFormatType.CSV,
            )

            assert result.success
            assert result.output_path == output_path
            assert output_path.exists()

            # Verify content
            import pandas as pd

            df = pd.read_csv(output_path)
            assert len(df) == 2
            assert "name" in df.columns
            assert df.iloc[0]["name"] == "Alice"

        finally:
            if output_path.exists():
                output_path.unlink()

    @pytest.mark.asyncio
    async def test_export_xml(self, export_manager):
        """Test XML export."""
        test_data = {"root": {"item1": "value1", "item2": "value2"}}

        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            output_path = Path(f.name)

        try:
            result = await export_manager.export_data(
                test_data,
                output_path,
                ExportFormatType.XML,
            )

            assert result.success
            assert result.output_path == output_path
            assert output_path.exists()

            # Verify it's valid XML
            import xml.etree.ElementTree as ET

            tree = ET.parse(output_path)
            root = tree.getroot()
            assert root.tag == "data"

        finally:
            if output_path.exists():
                output_path.unlink()


class TestAPIAdapter:
    """Test API adapter."""

    @pytest.fixture
    def api_adapter(self):
        return APIAdapter()

    @pytest.fixture
    def mock_tool_config(self):
        auth_config = AuthenticationConfig(
            type=AuthenticationType.API_KEY,
            credentials={"api_key": "test-key"},
        )

        version_endpoint = APIEndpoint(
            name="version",
            url="https://api.test.com/version",
            auth_config=auth_config,
        )

        return ExternalToolConfig(
            name="test-tool",
            api_endpoints=[version_endpoint],
            version="1.0.0",
            is_available=True,
        )

    def test_register_tool(self, api_adapter, mock_tool_config):
        """Test tool registration."""
        api_adapter.register_tool(mock_tool_config)

        tools = api_adapter.list_registered_tools()
        assert "test-tool" in tools

    def test_register_adapter(self, api_adapter, mock_tool_config):
        """Test adapter registration."""
        api_adapter.register_tool(mock_tool_config)

        def test_adapter(data, direction):
            return data

        api_adapter.register_adapter("test-tool", "1.0.0", test_adapter)

        # Verify adapter is registered
        assert "test-tool" in api_adapter._adapters
        assert "1.0.0" in api_adapter._adapters["test-tool"]

    def test_get_tool_status(self, api_adapter, mock_tool_config):
        """Test getting tool status."""
        api_adapter.register_tool(mock_tool_config)

        status = api_adapter.get_tool_status("test-tool")
        assert status["name"] == "test-tool"
        assert status["configured_version"] == "1.0.0"
        assert status["is_available"] is True

    @pytest.mark.asyncio
    async def test_check_api_version(self, api_adapter, mock_tool_config):
        """Test API version checking."""
        api_adapter.register_tool(mock_tool_config)

        mock_response_data = {
            "version": "1.1.0",
            "release_date": "2023-01-01",
            "deprecated": False,
        }

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_response_data
            mock_get.return_value.__aenter__.return_value = mock_response

            api_adapter.session = Mock()
            api_adapter.session.get = mock_get

            version_info = await api_adapter.check_api_version("test-tool")
            assert version_info is not None
            assert version_info.version == "1.1.0"
            assert not version_info.deprecated


@pytest.mark.asyncio
async def test_integration_workflow():
    """Test complete integration workflow."""
    # Test CAD import -> Export -> Cloud upload workflow

    # 1. Create a simple STL file
    stl_content = """solid test
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
  endloop
endfacet
endsolid test"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".stl", delete=False) as f:
        f.write(stl_content)
        f.flush()
        stl_path = Path(f.name)

    try:
        # 2. Import CAD file
        cad_integration = CADIntegration()
        cad_file = await cad_integration.import_cad_file(stl_path)

        assert cad_file.format == CADFormat.STL
        assert cad_file.is_valid

        # 3. Export to JSON
        export_manager = ExportManager()

        export_data = {
            "geometry": {
                "vertices": cad_file.validation_result.geometry_info.vertices,
                "faces": cad_file.validation_result.geometry_info.faces,
            },
            "metadata": {"format": cad_file.format.value, "file_size": cad_file.size},
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = Path(f.name)

        try:
            result = await export_manager.export_data(
                export_data,
                json_path,
                ExportFormatType.JSON,
            )

            assert result.success
            assert json_path.exists()

            # 4. Verify exported data
            with open(json_path) as f:
                exported_data = json.load(f)

            assert exported_data["geometry"]["vertices"] == 3
            assert exported_data["geometry"]["faces"] == 1
            assert exported_data["metadata"]["format"] == "stl"

        finally:
            if json_path.exists():
                json_path.unlink()

    finally:
        if stl_path.exists():
            stl_path.unlink()


if __name__ == "__main__":
    # Run basic tests
    asyncio.run(test_integration_workflow())
    print("Integration workflow test passed!")
