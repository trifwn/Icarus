import pytest

from ICARUS.flight_dynamics import State
from ICARUS.vehicle import Airplane


@pytest.mark.unit
def test_state_creation(benchmark_airplane: Airplane, benchmark_state: State) -> None:
    """Test the creation of a State instance with a benchmark airplane."""
    # Test state properties
    assert benchmark_state is not None, "State should be created"
    assert benchmark_state.name == "Unstick", "State should have correct name"
    assert benchmark_state.airspeed == 100, (
        "State should have correct freestream velocity"
    )
