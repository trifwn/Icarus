from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Thread

from ICARUS.computation.core import ExecutionContext
from ICARUS.computation.core import ResourceManager
from ICARUS.computation.core import Task
from ICARUS.computation.core import TaskResult
from ICARUS.computation.core import TaskState
from ICARUS.computation.core.protocols import ProgressReporter
from ICARUS.computation.core.types import ExecutionMode

from .base_engine import AbstractEngine


class ThreadingEngine(AbstractEngine):
    """Threading-based execution engine using ThreadPoolExecutor"""

    execution_mode: ExecutionMode = ExecutionMode.THREADING

    def __enter__(self) -> AbstractEngine:
        """Context manager entry point to prepare execution context."""
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit point to clean up execution context."""
        ...

    async def execute_tasks(self) -> list[TaskResult]:
        """Execute tasks using thread pool"""
        self.logger.info(f"Starting threading execution of {len(self.tasks)} tasks with max_workers={self.max_workers}")
        # Prepare execution-specific resources (e.g., locks)
        if not self.tasks:
            self.logger.warning("No tasks provided for threading execution")
            return []

        max_workers = self.max_workers or min(32, (len(self.tasks) or 1) + 4)

        def sync_execute_task(task: Task) -> TaskResult:
            """Synchronous wrapper for task execution"""
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # include exec_ctx into context
                return loop.run_until_complete(
                    self._execute_task_with_context(task, self.progress_reporter, self.resource_manager),
                )
            finally:
                loop.close()

        # Execute in thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [loop.run_in_executor(executor, sync_execute_task, task) for task in self.tasks]
            results = await asyncio.gather(*futures, return_exceptions=True)

        # Convert exceptions to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task = self.tasks[i]
                processed_results.append(TaskResult(task_id=task.id, state=TaskState.FAILED, error=result))
            else:
                processed_results.append(result)

        self.logger.info("Threading execution completed")
        return processed_results

    async def _execute_task_with_context(
        self,
        task: Task,
        progress_reporter: ProgressReporter | None,
        resource_manager: ResourceManager | None,
    ) -> TaskResult:
        """Execute a single task with full context management"""
        # Create execution context and inject mode-specific resources
        context = ExecutionContext(
            task_id=task.id,
            config=task.config,
            execution_mode=ExecutionMode.THREADING,
            progress_reporter=progress_reporter,
            resource_manager=resource_manager,
            logger=logging.getLogger(f"task.{task.name}"),
        )

        start_time = datetime.now()
        task.state = TaskState.RUNNING

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
            if progress_reporter:
                progress_reporter.report_completion(task_result)
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
            if progress_reporter:
                progress_reporter.report_completion(task_result)
            return task_result

        except Exception as e:
            task_result = TaskResult(
                task_id=task.id,
                state=TaskState.FAILED,
                error=e,
                execution_time=datetime.now() - start_time,
            )
            task.state = TaskState.FAILED
            if progress_reporter:
                progress_reporter.report_completion(task_result)
            return task_result

        finally:
            # Always clean up resources
            await context.release_resources()
            await task.executor.cleanup()

    async def _start_progress_monitoring(self) -> None:
        """Start the progress monitor in a separate thread if enabled."""
        # Create a queue for progress events (local & multiproc)
        if self.progress_monitor:

            def monitor_runner():
                if self.progress_monitor:
                    with self.progress_monitor:
                        asyncio.run(self.progress_monitor.monitor_loop())

            # Start the monitor in a separate thread
            self.monitor_thread = Thread(target=monitor_runner, daemon=True)
            self.monitor_thread.start()

    async def _stop_progress_monitoring(self) -> None:
        """Stop the progress monitor if it is running."""
        if self.terminate_event:
            self.terminate_event.set()
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1)
            if self.monitor_thread.is_alive():
                self.logger.warning("Progress monitor thread did not stop gracefully")
        else:
            self.logger.debug("Progress monitor thread was not running")
