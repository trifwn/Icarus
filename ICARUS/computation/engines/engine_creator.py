from ICARUS.computation.core import ExecutionMode

from .async_engine import AsyncEngine
from .base_engine import AbstractEngine
from .multiprocessing_engine import MultiprocessingEngine
from .sequential_engine import SequentialExecutionEngine
from .threading_engine import ThreadingEngine


# Factory function for creating execution engines
def create_execution_engine(
    mode: ExecutionMode,
) -> AbstractEngine:
    """Factory function to create execution engines based on mode"""
    engine_map = {
        ExecutionMode.SEQUENTIAL: SequentialExecutionEngine,
        ExecutionMode.ASYNC: AsyncEngine,
        ExecutionMode.THREADING: ThreadingEngine,
        ExecutionMode.MULTIPROCESSING: MultiprocessingEngine,
    }

    engine_class = engine_map.get(mode)
    if not engine_class:
        raise ValueError(f"Unknown execution mode: {mode}")

    return engine_class()
