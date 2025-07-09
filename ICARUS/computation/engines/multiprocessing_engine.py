from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import threading
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any

from ICARUS import setup_logging
from ICARUS import setup_mp_logging
from ICARUS.computation.core import ConcurrencyFeature
from ICARUS.computation.core import ConcurrentVariable
from ICARUS.computation.core import ExecutionContext
from ICARUS.computation.core import ExecutionMode
from ICARUS.computation.core import QueueLike
from ICARUS.computation.core import Task
from ICARUS.computation.core import TaskResult
from ICARUS.computation.core import TaskState

from .base_engine import AbstractEngine


class MultiprocessingEngine(AbstractEngine):
    """Multiprocessing-based execution engine using ProcessPoolExecutor"""

    execution_mode: ExecutionMode = ExecutionMode.MULTIPROCESSING

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.listener = None
        self.monitor_thread = None

    def request_concurrent_vars(self) -> dict[str, ConcurrencyFeature]:
        """Request concurrent variables required by this engine."""
        vars = super().request_concurrent_vars()
        # We'll manage the queue ourselves using Manager
        vars["Console_Queue"] = ConcurrencyFeature.QUEUE
        return vars

    def set_concurrent_vars(self, vars: dict[str, ConcurrentVariable]) -> None:
        """Set concurrent variables for this engine."""
        super().set_concurrent_vars(vars)
        log_queue = vars["Console_Queue"]
        if not isinstance(log_queue, QueueLike):
            raise TypeError(
                "MultiprocessingEngine requires a multiprocessing Queue for logging",
            )

        self.listener = setup_mp_logging(log_queue)
        if self.listener is not None:
            self.listener.start()

    def __enter__(self) -> AbstractEngine:
        """Context manager entry point to prepare execution context."""
        self.logger.info(f"Entering engine: {self.__class__.__name__}")
        self.mp_manager = mp.Manager()

        concurrent_vars_req = self.request_concurrent_vars()
        self.execution_mode.set_multiprocessing_manager(self.mp_manager)
        concurrent_vars = self.execution_mode.primitives.get_concurrent_variables(
            concurrent_vars_req,
        )
        self.set_concurrent_vars(concurrent_vars)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.logger.info("Shutting down engine")

        # Stop the queue listener
        if hasattr(self, "listener") and self.listener:
            self.listener.stop()
            self.listener = None

        if hasattr(self, "mp_manager") and self.mp_manager:
            self.mp_manager.shutdown()
            self.mp_manager = None
        self.execution_mode.clear_multiprocessing_manager()

        # Revert global logging configuration
        setup_logging()

    async def execute_tasks(self) -> list[TaskResult]:
        """Execute tasks using process pool"""
        self.logger.info(
            f"Starting multiprocessing execution of {len(self.tasks)} tasks with max_workers={self.max_workers}",
        )
        max_workers = self.max_workers or min(mp.cpu_count(), len(self.tasks) or 1)

        # Execute in process pool
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=MultiprocessingEngine.set_concurrent_vars,
            initargs=(
                self,
                self.concurrent_variables if self.concurrent_variables else {},
            ),
        ) as executor:
            # Submit dummy tasks to force process initialization
            dummy_futures = [executor.submit(dummy_task) for _ in range(100)]
            for f in dummy_futures:
                f.result()  # Wait for all workers to start and run initializer

            # Start the progress monitor if enabled
            if self.monitor_thread:
                self.monitor_thread.start()

            # Submit tasks to the executor
            results = executor.map(self._execute_task, self.tasks)

        # Convert exceptions to failed results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(
                    TaskResult(
                        task_id=result.task_id,
                        state=TaskState.FAILED,
                        error=result,
                    ),
                )
            else:
                processed_results.append(result)

        self.logger.info("Multiprocessing execution completed")
        return processed_results

    async def _start_progress_monitoring(self) -> None:
        """Start the progress monitor in a separate process if enabled."""
        if self.progress_monitor:

            def monitor_runner() -> None:
                if self.progress_monitor:
                    with self.progress_monitor:
                        asyncio.run(self.progress_monitor.monitor_loop())

            self.monitor_thread = threading.Thread(target=monitor_runner, daemon=True)
            # self.monitor_thread.start()

    async def _stop_progress_monitoring(self) -> None:
        """Stop the progress monitor if it is running."""
        if self.terminate_event:
            self.terminate_event.set()
        if self.monitor_thread and self.monitor_thread.is_alive():
            try:
                self.monitor_thread.join(timeout=1)
                if self.monitor_thread.is_alive():
                    self.logger.warning(
                        "Progress monitor process did not stop gracefully",
                    )
            except Exception as e:
                self.logger.error(f"Error stopping monitor process: {e}")
        else:
            self.logger.debug("Progress monitor process was not running")

    def __getstate__(self) -> dict[str, Any]:
        """Custom serialization to avoid issues with multiprocessing."""
        state = self.__dict__.copy()
        # Remove non-serializable items
        state.pop("monitor_thread", None)
        state.pop("listener", None)
        state.pop("concurrent_variables", None)
        state.pop("terminate_event", None)
        state.pop("mp_manager", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Custom deserialization to restore state."""
        self.__dict__.update(state)
        self.monitor_thread = None
        self.listener = None
        self.concurrent_variables = None
        self.terminate_event = None
        self.mp_manager = None

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

            async def execute() -> TaskResult[Any]:
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

                    # Log success
                    context.logger.info(f"Task {task.id} completed successfully")
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

                    # Log failure
                    context.logger.error(f"Task {task.id} failed: {e}")
                    return task_result

                finally:
                    # Always clean up resources
                    await context.release_resources()
                    await task.executor.cleanup()

            return loop.run_until_complete(execute())

        except Exception as e:
            # Handle any errors that occur outside the async context
            logger = logging.getLogger(f"task.{task.name}")
            logger.error(f"Critical error in task {task.id}: {e}")
            return TaskResult(
                task_id=task.id,
                state=TaskState.FAILED,
                error=e,
                execution_time=datetime.now() - start_time
                if "start_time" in locals()
                else None,
            )
        finally:
            loop.close()


def dummy_task() -> None:
    """Dummy task to ensure process pool workers are initialized."""
    pass
