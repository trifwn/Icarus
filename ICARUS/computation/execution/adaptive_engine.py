from __future__ import annotations

from ICARUS.computation.core import ExecutionMode
from ICARUS.computation.core import Task
from ICARUS.computation.core import TaskResult

from .async_engine import AsyncExecutionEngine
from .base_engine import AbstractExecutionEngine
from .multiprocessing_engine import MultiprocessingExecutionEngine
from .sequential_engine import SequentialExecutionEngine
from .threading_engine import ThreadingExecutionEngine


class AdaptiveExecutionEngine(AbstractExecutionEngine):
    """
    Adaptive execution engine that chooses the best strategy based on task characteristics
    """

    def __enter__(self) -> AbstractExecutionEngine:
        """Context manager entry point to prepare execution context."""
        ...

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit point to clean up execution context."""
        ...

    def __init__(self):
        super().__init__()
        self.current_execution_mode: ExecutionMode | None = None
        self.engines: dict[ExecutionMode, AbstractExecutionEngine] = {
            ExecutionMode.SEQUENTIAL: SequentialExecutionEngine(),
            ExecutionMode.ASYNC: AsyncExecutionEngine(),
            ExecutionMode.THREADING: ThreadingExecutionEngine(),
            ExecutionMode.MULTIPROCESSING: MultiprocessingExecutionEngine(),
        }

    @property
    def execution_mode(self) -> ExecutionMode:
        """Get the current execution mode"""
        if self.current_execution_mode is None:
            raise ValueError("Execution mode not set. Call execute_tasks first.")
        return self.current_execution_mode

    async def execute_tasks(self) -> list[TaskResult]:
        """Execute tasks using adaptive strategy selection"""
        execution_mode = self._select_execution_mode(self.tasks)

        self.current_execution_mode = execution_mode

        self.logger.info(f"Selected execution mode: {execution_mode.value} for {len(self.tasks)} tasks")

        engine = self.engines[execution_mode]
        return await engine.execute_tasks()

    def _select_execution_mode(self, tasks: list[Task]) -> ExecutionMode:
        """Select the best execution mode based on task characteristics"""
        num_tasks = len(tasks)

        # Simple heuristics for mode selection
        if num_tasks <= 1:
            return ExecutionMode.SEQUENTIAL

        # Analyze task characteristics
        cpu_intensive_tasks = sum(1 for task in tasks if self._is_cpu_intensive(task))
        io_intensive_tasks = sum(1 for task in tasks if self._is_io_intensive(task))

        cpu_ratio = cpu_intensive_tasks / num_tasks if num_tasks > 0 else 0
        io_ratio = io_intensive_tasks / num_tasks if num_tasks > 0 else 0

        # Decision logic
        if num_tasks <= 5:
            return ExecutionMode.SEQUENTIAL
        elif cpu_ratio > 0.7 and num_tasks > 10:
            # CPU-intensive tasks benefit from multiprocessing
            return ExecutionMode.MULTIPROCESSING
        elif io_ratio > 0.7:
            # I/O-intensive tasks benefit from async
            return ExecutionMode.ASYNC
        elif num_tasks <= 20:
            # Medium number of mixed tasks - use threading
            return ExecutionMode.THREADING
        else:
            # Large number of mixed tasks - use async for better resource management
            return ExecutionMode.ASYNC

    def _is_cpu_intensive(self, task: Task) -> bool:
        """Determine if a task is CPU-intensive based on its characteristics"""
        # This is a heuristic - in practice, you might want to analyze:
        # - Task type/category
        # - Historical execution patterns
        # - Task configuration hints
        # - Resource requirements

        # Simple heuristic based on task name/type
        cpu_indicators = ["compute", "calculate", "process", "transform", "analyze"]
        task_name = task.name.lower()
        return any(indicator in task_name for indicator in cpu_indicators)

    def _is_io_intensive(self, task: Task) -> bool:
        """Determine if a task is I/O-intensive based on its characteristics"""
        # Simple heuristic based on task name/type
        io_indicators = ["fetch", "download", "upload", "read", "write", "load", "save", "api", "network"]
        task_name = task.name.lower()
        return any(indicator in task_name for indicator in io_indicators)

    async def _start_progress_monitoring(
        self,
    ) -> None:
        """Start the progress monitor for the selected execution mode"""
        if self.current_execution_mode is None:
            raise ValueError("Execution mode not set. Call execute_tasks first.")

        engine = self.engines[self.current_execution_mode]
        await engine._start_progress_monitoring()

    async def _stop_progress_monitoring(self) -> None:
        """Stop the progress monitor if it was started"""
        for engine in self.engines.values():
            if hasattr(engine, "_stop_progress_monitoring"):
                await engine._stop_progress_monitoring()
