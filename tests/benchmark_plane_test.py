import numpy as np

from ICARUS.airfoils import NACA4
from ICARUS.core.types import FloatArray
from ICARUS.environment import EARTH_ISA
from ICARUS.flight_dynamics import State
from ICARUS.vehicle import Airplane
from ICARUS.vehicle import SymmetryAxes
from ICARUS.vehicle import WingSegment


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


def get_benchmark_plane(name: str) -> Airplane:
    """Create a benchmark airplane configuration.

    Args:
        name: Name for the airplane

    Returns:
        Tuple of (airplane, state)
    """
    origin: FloatArray = np.array([0.0, 0.0, 0.0], dtype=float)
    wing_position: FloatArray = np.array(
        [0, 0.0, 0.0],
        dtype=float,
    )
    wing_orientation: FloatArray = np.array(
        [0.0, 0.0, 0.0],
        dtype=float,
    )

    Simplewing = WingSegment(
        name=name,
        root_airfoil=NACA4(M=0.04, P=0.4, XX=0.15),  # "NACA4415",
        origin=origin + wing_position,
        orientation=wing_orientation,
        symmetries=SymmetryAxes.Y,
        span=2 * 5,
        sweep_offset=0.0,
        root_chord=1.0,
        tip_chord=1.0,
        N=15,
        M=15,
        mass=1,
    )
    airplane = Airplane(Simplewing.name, main_wing=Simplewing)
    return airplane


def get_benchmark_state(benchmark_plane: Airplane) -> State:
    """Get the benchmark state for a given airplane."""
    return State(
        name="Unstick",
        airplane=benchmark_plane,
        u_freestream=100,  # Example freestream velocity
        environment=EARTH_ISA,
    )
