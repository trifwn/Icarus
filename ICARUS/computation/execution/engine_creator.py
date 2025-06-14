from ICARUS.computation.core import ExecutionMode

from .adaptive_engine import AdaptiveExecutionEngine
from .async_engine import AsyncExecutionEngine
from .base_engine import BaseExecutionEngine
from .multiprocessing_engine import MultiprocessingExecutionEngine
from .sequential_engine import SequentialExecutionEngine
from .threading_engine import ThreadingExecutionEngine


# Factory function for creating execution engines
def create_execution_engine(mode: ExecutionMode, max_workers: int | None = None) -> BaseExecutionEngine:
    """Factory function to create execution engines based on mode"""
    engine_map = {
        ExecutionMode.SEQUENTIAL: SequentialExecutionEngine,
        ExecutionMode.ASYNC: AsyncExecutionEngine,
        ExecutionMode.THREADING: ThreadingExecutionEngine,
        ExecutionMode.MULTIPROCESSING: MultiprocessingExecutionEngine,
        ExecutionMode.ADAPTIVE: AdaptiveExecutionEngine,
    }

    engine_class = engine_map.get(mode)
    if not engine_class:
        raise ValueError(f"Unknown execution mode: {mode}")

    return engine_class(max_workers)
