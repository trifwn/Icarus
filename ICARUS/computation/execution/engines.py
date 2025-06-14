from __future__ import annotations

import asyncio
import logging
from abc import ABC
from abc import abstractmethod
from datetime import datetime

from ICARUS.computation.core import ExecutionContext
from ICARUS.computation.core import ResourceManager
from ICARUS.computation.core import Task
from ICARUS.computation.core import TaskResult
from ICARUS.computation.core import TaskState
from ICARUS.computation.monitoring.progress import TqdmProgressMonitor


class BaseExecutionEngine(ABC):
    """Abstract base for execution engines"""

    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def execute_tasks(
        self,
        tasks: list[Task],
        progress_monitor: TqdmProgressMonitor,
        resource_manager: ResourceManager | None = None,
    ) -> list[TaskResult]:
        """Execute tasks and return results"""
        pass


class AsyncExecutionEngine(BaseExecutionEngine):
    """Async-based execution engine with progress integration"""

    async def execute_tasks(
        self,
        tasks: list[Task],
        progress_monitor: TqdmProgressMonitor,
        resource_manager: ResourceManager | None = None,
    ) -> list[TaskResult]:
        """Execute tasks concurrently using asyncio"""
        semaphore = asyncio.Semaphore(self.max_workers or 10)

        async def execute_single_task(task: Task) -> TaskResult:
            async with semaphore:
                return await self._execute_task_with_context(task, progress_monitor, resource_manager)

        results = await asyncio.gather(*[execute_single_task(task) for task in tasks], return_exceptions=True)

        # Convert exceptions to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task = tasks[i]
                processed_results.append(TaskResult(task_id=task.id, state=TaskState.FAILED, error=result))
            else:
                processed_results.append(result)

        return processed_results

    async def _execute_task_with_context(
        self,
        task: Task,
        progress_monitor: TqdmProgressMonitor,
        resource_manager: ResourceManager | None,
    ) -> TaskResult:
        """Execute a single task with full context management"""
        context = ExecutionContext(
            task.id,
            task.config,
            progress_monitor,
            resource_manager,
            logging.getLogger(f"task.{task.name}"),
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
                result=result,
                execution_time=datetime.now() - start_time,
            )

            task.state = TaskState.COMPLETED
            await progress_monitor.report_completion(task_result)
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
            await progress_monitor.report_completion(task_result)
            return task_result

        except Exception as e:
            task_result = TaskResult(
                task_id=task.id,
                state=TaskState.FAILED,
                error=e,
                execution_time=datetime.now() - start_time,
            )
            task.state = TaskState.FAILED
            await progress_monitor.report_completion(task_result)
            return task_result

        finally:
            # Always clean up resources
            await context.release_resources()
            await task.executor.cleanup()
