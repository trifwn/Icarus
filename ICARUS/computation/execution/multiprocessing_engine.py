from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import threading
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any

from ICARUS.computation.core import ExecutionContext
from ICARUS.computation.core import Task
from ICARUS.computation.core import TaskResult
from ICARUS.computation.core import TaskState
from ICARUS.computation.core.protocols import ProgressMonitor
from ICARUS.computation.core.types import ExecutionMode

from .base_engine import AbstractExecutionEngine


class MultiprocessingExecutionEngine(AbstractExecutionEngine):
    """Multiprocessing-based execution engine using ProcessPoolExecutor"""

    execution_mode: ExecutionMode = ExecutionMode.MULTIPROCESSING

    def __enter__(self) -> AbstractExecutionEngine:
        """Context manager entry point to prepare execution context."""
        self.logger.info(f"Entering engine: {self.__class__.__name__}")
        concurrent_vars_req = self.request_concurrent_vars()
        self.execution_mode.set_multiprocessing_manager(mp.Manager())
        concurent_vars = self.execution_mode.primitives.get_concurrent_variables(concurrent_vars_req)
        self.set_concurrent_vars(concurent_vars)
        return self

    async def execute_tasks(self) -> list[TaskResult]:
        """Execute tasks using process pool"""
        self.logger.info(
            f"Starting multiprocessing execution of {len(self.tasks)} tasks with max_workers={self.max_workers}",
        )
        max_workers = self.max_workers or min(mp.cpu_count(), len(self.tasks) or 1)

        # Execute in process pool
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=MultiprocessingExecutionEngine.set_concurrent_vars,
            initargs=(
                self,
                self.concurrent_variables if self.concurrent_variables else {},
            ),
        ) as executor:
            results = executor.map(self._execute_task, self.tasks)

        # Convert exceptions to failed results
        processed_results = []
        for i, (task, result) in enumerate(zip(self.tasks, results)):
            if isinstance(result, Exception):
                processed_results.append(TaskResult(task_id=task.id, state=TaskState.FAILED, error=result))
            else:
                processed_results.append(result)

        self.logger.info("Multiprocessing execution completed")
        return processed_results

    async def _start_progress_monitoring(self) -> None:
        """Start the progress monitor in a separate process if enabled."""
        if self.progress_monitor:

            def monitor_runner():
                if self.progress_monitor:
                    with self.progress_monitor:
                        asyncio.run(self.progress_monitor.monitor_loop())

            self.monitor_thread = threading.Thread(target=monitor_runner, daemon=True)
            self.monitor_thread.start()

    async def _stop_progress_monitoring(self) -> None:
        """Stop the progress monitor if it is running."""
        if self.terminate_event:
            self.terminate_event.set()
        if self.monitor_thread and self.monitor_thread.is_alive():
            try:
                self.monitor_thread.join(timeout=1)
                if self.monitor_thread.is_alive():
                    self.logger.warning("Progress monitor process did not stop gracefully")
            except Exception as e:
                self.logger.error(f"Error stopping monitor process: {e}")
        else:
            self.logger.debug("Progress monitor process was not running")

    def __getstate__(self) -> dict[str, Any]:
        """Custom serialization to avoid issues with multiprocessing."""
        state = self.__dict__.copy()
        state.pop("monitor_thread", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Custom deserialization to restore state."""
        self.__dict__.update(state)
        self.monitor_thread = None

    def _execute_task(
        self,
        task: Task,
    ) -> TaskResult:
        """
        Execute a single task in a separate process.
        This function needs to be at module level for multiprocessing to work.
        """
        # Create new event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            context = ExecutionContext(
                task_id=task.id,
                config=task.config,
                execution_mode=ExecutionMode.MULTIPROCESSING,
                progress_reporter=self.progress_reporter,
                resource_manager=self.resource_manager,
                logger=logging.getLogger(f"task.{task.name}"),
            )

            start_time = datetime.now()
            task.state = TaskState.RUNNING

            async def execute():
                try:
                    # Acquire resources
                    await context.acquire_resources()

                    # Validate input
                    if not await task.executor.validate_input(task.input):
                        raise ValueError("Task input validation failed")

                    # Execute with timeout
                    if task.config.timeout:
                        result = await asyncio.wait_for(
                            task.executor.execute(task.input, context),
                            timeout=task.config.timeout.total_seconds(),
                        )
                    else:
                        result = await task.executor.execute(task.input, context)

                    # Create successful result
                    task_result = TaskResult(
                        task_id=task.id,
                        state=TaskState.COMPLETED,
                        output=result,
                        execution_time=datetime.now() - start_time,
                    )

                    task.state = TaskState.COMPLETED
                    if context.progress_reporter is not None:
                        context.progress_reporter.report_completion(task_result)
                    return task_result

                except Exception as e:
                    task_result = TaskResult(
                        task_id=task.id,
                        state=TaskState.FAILED,
                        error=e,
                        execution_time=datetime.now() - start_time,
                    )
                    task.state = TaskState.FAILED
                    if context.progress_reporter is not None:
                        context.progress_reporter.report_completion(task_result)
                    return task_result

                finally:
                    # Always clean up resources
                    await context.release_resources()
                    await task.executor.cleanup()

            return loop.run_until_complete(execute())

        finally:
            loop.close()


def run_monitor_loop(pm: ProgressMonitor) -> None:
    """Run the monitor loop in a separate process."""
    try:
        asyncio.run(pm.monitor_loop())
    except Exception as e:
        logging.error(f"Error in progress monitor loop: {e}")
