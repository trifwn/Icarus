import pytest

pytestmark = pytest.mark.asyncio


def test_export_service_import():
    try:
        from core.services import ExportService
    except ImportError:
        pytest.skip("ExportService module not available")


def test_export_service_basic():
    from core.services import ExportService

    export_service = ExportService()
    formats = export_service.get_supported_formats()
    assert len(formats) > 0
    test_data = {"results": [1, 2, 3], "metadata": {"type": "test"}}
    for format_type in formats[:2]:
        exported = export_service.export_data(test_data, format_type)
        assert exported is not None
