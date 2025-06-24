from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from ICARUS.computation.core import ExecutionContext
from ICARUS.computation.core import ResourceManager
from ICARUS.computation.core import Task
from ICARUS.computation.core import TaskResult
from ICARUS.computation.core import TaskState
from ICARUS.computation.core.protocols import ProgressMonitor
from ICARUS.computation.core.protocols import ProgressReporter
from ICARUS.computation.core.types import ExecutionMode

from .base_engine import BaseExecutionEngine


class AsyncExecutionEngine(BaseExecutionEngine):
    """Async-based execution engine with progress integration"""

    async def execute_tasks(
        self,
        tasks: list[Task],
        progress_reporter: ProgressReporter,
        resource_manager: ResourceManager | None = None,
    ) -> list[TaskResult]:
        """Execute tasks concurrently using asyncio"""
        self.logger.info(f"Starting async execution of {len(tasks)} tasks with max_workers={self.max_workers}")
        semaphore = asyncio.Semaphore(self.max_workers or 10)

        async def execute_single_task(task: Task) -> TaskResult:
            async with semaphore:
                return await self._execute_task_with_context(task, progress_reporter, resource_manager)

        results = await asyncio.gather(*[execute_single_task(task) for task in tasks], return_exceptions=True)

        # Convert exceptions to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task = tasks[i]
                processed_results.append(TaskResult(task_id=task.id, state=TaskState.FAILED, error=result))
            else:
                processed_results.append(result)

        self.logger.info("Async execution completed")
        return processed_results

    async def _execute_task_with_context(
        self,
        task: Task,
        progress_reporter: ProgressReporter,
        resource_manager: ResourceManager | None,
    ) -> TaskResult:
        """Execute a single task with full context management"""
        context = ExecutionContext(
            task_id=task.id,
            config=task.config,
            execution_mode=ExecutionMode.ASYNC,
            progress_reporter=progress_reporter,
            resource_manager=resource_manager,
            logger=logging.getLogger(f"task.{task.name}"),
        )

        start_time = datetime.now()
        task.state = TaskState.RUNNING

        # Prepare monitoring primitives
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
            await progress_reporter.report_completion(task_result)
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
            await progress_reporter.report_completion(task_result)
            return task_result

        except Exception as e:
            task_result = TaskResult(
                task_id=task.id,
                state=TaskState.FAILED,
                error=e,
                execution_time=datetime.now() - start_time,
            )
            task.state = TaskState.FAILED
            await progress_reporter.report_completion(task_result)
            return task_result

        finally:
            # Always clean up resources
            await context.release_resources()
            await task.executor.cleanup()

    async def _start_progress_monitoring(self, progress_monitor: ProgressMonitor) -> None:
        """Start the progress monitor in an asyncio task if enabled."""

        async def monitor_runner():
            if progress_monitor:
                with progress_monitor:
                    await progress_monitor.monitor_loop()

        self.monitor_task = asyncio.create_task(monitor_runner())

    async def _stop_progress_monitoring(self) -> None:
        """Stop the progress monitor if it is running."""
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await asyncio.wait_for(self.monitor_task, timeout=1.0)
            except asyncio.TimeoutError:
                self.logger.warning("Progress monitor task did not stop gracefully")
            except asyncio.CancelledError:
                pass
        else:
            self.logger.debug("Progress monitor task was not running")
