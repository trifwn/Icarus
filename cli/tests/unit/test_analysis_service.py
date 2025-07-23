import pytest

pytestmark = pytest.mark.asyncio


def test_analysis_service_import():
    try:
        from integration.analysis_service import AnalysisService
    except ImportError:
        pytest.skip("AnalysisService module not available")


def test_analysis_service_basic():
    from integration.analysis_service import AnalysisService

    service = AnalysisService()
    modules = service.get_available_modules()
    assert isinstance(modules, list)
    if modules:
        solver_info = service.get_solver_info("xfoil")
        assert solver_info is not None
    test_params = {"reynolds": 1000000, "mach": 0.1}
    validation = service.validate_parameters(test_params)
    assert validation is not None
