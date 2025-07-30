import pytest

pytestmark = pytest.mark.asyncio


def test_solver_manager_import():
    try:
        from integration.solver_manager import SolverManager
    except ImportError:
        pytest.skip("SolverManager module not available")


def test_solver_manager_basic():
    from integration.solver_manager import SolverManager

    solver_manager = SolverManager()
    solvers = solver_manager.discover_solvers()
    assert isinstance(solvers, list)
    for solver in solvers[:2]:
        is_available = solver_manager.is_solver_available(solver)
        assert isinstance(is_available, bool)
    if solvers:
        info = solver_manager.get_solver_info(solvers[0])
        assert info is not None
