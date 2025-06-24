from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial

from ICARUS.computation.core import ExecutionContext
from ICARUS.computation.core import ResourceManager
from ICARUS.computation.core import Task
from ICARUS.computation.core import TaskResult
from ICARUS.computation.core import TaskState
from ICARUS.computation.core.protocols import ProgressMonitor, ProgressReporter
from ICARUS.computation.core.types import ExecutionMode

from .base_engine import BaseExecutionEngine


class MultiprocessingExecutionEngine(BaseExecutionEngine):
    """Multiprocessing-based execution engine using ProcessPoolExecutor"""

    async def execute_tasks(
        self,
        tasks: list[Task],
        progress_reporter: ProgressReporter | None = None,
        resource_manager: ResourceManager | None = None,
    ) -> list[TaskResult]:
        """Execute tasks using process pool"""
        self.logger.info(
            f"Starting multiprocessing execution of {len(tasks)} tasks with max_workers={self.max_workers}",
        )

        max_workers = self.max_workers or min(mp.cpu_count(), len(tasks) or 1)

        # Use the shared event_queue from the monitor (if provided)
        # progress_queue = getattr(progress_reporter, "event_queue", None) if progress_reporter else None

        # Execute in process pool
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            execute_func = partial(
                _execute_task_in_process,
                progress_reporter=progress_reporter,
                resource_manager=resource_manager,
            )
            futures = [loop.run_in_executor(executor, execute_func, task) for task in tasks]
            results = await asyncio.gather(*futures, return_exceptions=True)

        # Convert exceptions to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task = tasks[i]
                processed_results.append(TaskResult(task_id=task.id, state=TaskState.FAILED, error=result))
            else:
                processed_results.append(result)

        self.logger.info("Multiprocessing execution completed")
        return processed_results

    async def _start_progress_monitoring(self, progress_monitor: ProgressMonitor) -> None:
        """Start the progress monitor in a separate process if enabled."""
        queue = mp.Queue()
        self.progress_monitor = progress_monitor
        progress_monitor.set_event_queue(queue)

        # Spawn the monitor Process Thread on the main process
        self.monitor_process = mp.Process(
            target=progress_monitor.monitor_loop,
            name="ProgressMonitor",
            args=(queue,),
        )

    async def _stop_progress_monitoring(self) -> None:
        """Stop the progress monitor if it is running."""
        if self.monitor_process and self.monitor_process.is_alive():
            try:
                # Send sentinel value to stop
                self.progress_monitor.event_queue.put("STOP")
                self.monitor_process.join(timeout=2)
                if self.monitor_process.is_alive():
                    self.monitor_process.terminate()
                    self.logger.warning("Progress monitor process did not stop gracefully")
            except Exception as e:
                self.logger.error(f"Error stopping monitor process: {e}")
        else:
            self.logger.debug("Progress monitor process was not running")


def _execute_task_in_process(
    task: Task,
    progress_reporter: ProgressReporter | None,
    resource_manager: ResourceManager | None,
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
            progress_reporter=progress_reporter,
            resource_manager=resource_manager,
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
                    await context.progress_reporter.report_completion(task_result)
                return task_result

            except asyncio.TimeoutError:
                error = Exception(f"Task timed out after {task.config.timeout}")
                task_result = TaskResult(
                    task_id=task.id,
                    state=TaskState.FAILED,
                    error=error,
                    execution_time=datetime.now() - start_time,
                )
                task.state = TaskState.FAILED
                if context.progress_reporter is not None:
                    await context.progress_reporter.report_completion(task_result)
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
                    await context.progress_reporter.report_completion(task_result)
                return task_result

            finally:
                # Always clean up resources
                await context.release_resources()
                await task.executor.cleanup()

        return loop.run_until_complete(execute())

    finally:
        loop.close()
