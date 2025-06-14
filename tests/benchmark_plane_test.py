
from ICARUS.airfoils import NACA4
from ICARUS.flight_dynamics import State
from ICARUS.vehicle import Airplane


def test_benchmark_plane_creation(benchmark_airplane: Airplane, benchmark_state: State) -> None:
    """Test that benchmark plane can be created successfully."""
    # Test airplane properties
    assert benchmark_airplane is not None, "Airplane should be created"
    assert benchmark_airplane.name == "bmark", "Airplane should have correct name"
    assert hasattr(benchmark_airplane, "main_wing"), "Airplane should have main wing"

    # Test state properties
    assert benchmark_state is not None, "State should be created"
    assert benchmark_state.name == "Unstick", "State should have correct name"
    assert benchmark_state.u_freestream == 100, "State should have correct freestream velocity"
    assert benchmark_state.airplane is benchmark_airplane, "State should reference the airplane"


def test_wing_segment_properties(benchmark_airplane: Airplane) -> None:
    """Test wing segment properties."""
    wing = benchmark_airplane.main_wing

    # Test wing properties that we know exist
    assert wing.span == 10.0, "Wing should have correct span"
    assert wing.N == 15, "Wing should have correct N panels"
    assert wing.M == 15, "Wing should have correct M panels"
    assert hasattr(wing, "root_airfoil"), "Wing should have root airfoil"


def test_airfoil_properties(benchmark_airplane: Airplane) -> None:
    """Test airfoil properties of the wing."""
    wing = benchmark_airplane.main_wing
    airfoil = wing.root_airfoil

    # Test NACA4 airfoil properties
    assert isinstance(airfoil, NACA4), "Should be NACA4 airfoil"
    assert airfoil.m == 0.04, "Should have correct max camber"
    assert airfoil.p == 0.4, "Should have correct camber position"
    assert airfoil.xx == 0.15, "Should have correct thickness"
