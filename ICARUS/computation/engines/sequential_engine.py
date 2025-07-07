from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from threading import Thread

from ICARUS.computation.core import ExecutionContext
from ICARUS.computation.core import ExecutionMode
from ICARUS.computation.core import ResourceManager
from ICARUS.computation.core import Task
from ICARUS.computation.core import TaskResult
from ICARUS.computation.core import TaskState
from ICARUS.computation.core.protocols import ProgressReporter

from .base_engine import AbstractEngine


class SequentialExecutionEngine(AbstractEngine):
    """Sequential execution engine - executes tasks one by one"""

    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL

    def __enter__(self) -> AbstractEngine:
        """Context manager entry point to prepare execution context."""
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit point to clean up execution context."""
        ...

    async def execute_tasks(self) -> list[TaskResult]:
        """Execute tasks sequentially"""
        self.logger.info(f"Starting sequential execution of {len(self.tasks)} tasks")
        results = []

        for task in self.tasks:
            result = await self._execute_task_with_context(
                task,
                self.progress_reporter,
                self.resource_manager,
            )
            results.append(result)

        self.logger.info("Sequential execution completed")
        return results

    async def _execute_task_with_context(
        self,
        task: Task,
        progress_reporter: ProgressReporter | None,
        resource_manager: ResourceManager | None,
    ) -> TaskResult:
        """Execute a single task with full context management"""
        context = ExecutionContext(
            task_id=task.id,
            config=task.config,
            execution_mode=ExecutionMode.SEQUENTIAL,
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

            self.monitor_thread = Thread(target=monitor_runner, daemon=True)
            self.monitor_thread.start()

    async def _stop_progress_monitoring(self) -> None:
        """Stop the progress monitor if it is running."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1)
            if self.monitor_thread.is_alive():
                self.logger.warning("Progress monitor thread did not stop gracefully")
        else:
            self.logger.debug("Progress monitor thread was not running")
