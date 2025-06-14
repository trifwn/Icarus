"""
Comprehensive test suite for the ICARUS simulation framework.

This module provides extensive unit tests covering all components
of the simulation framework including core functionality, edge cases,
and integration scenarios.
"""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from typing import Any

from ICARUS.computation.core import (
                    TaskState, Priority, ExecutionMode, TaskId, TaskConfiguration,
                    Task,
                    ExecutionContext,
                    SimulationConfig,
                    ConfigurationError,
)
from ICARUS.computation.executors import SummationExecutor
from ICARUS.computation.runners import SimulationRunner
from ICARUS.computation.resources.manager import SimpleResourceManager


class TestTaskExecutor:
    """Mock task executor for testing."""

    def __init__(self, execution_time: float = 0.1, should_fail: bool = False):
        self.execution_time = execution_time
        self.should_fail = should_fail
        self.call_count = 0

    async def execute(self, task_input: Any, context: ExecutionContext) -> Any:
        self.call_count += 1

        if self.should_fail:
            raise ValueError(f"Test failure for input: {task_input}")

        # Simulate work with progress reporting
        steps = 10
        for i in range(steps):
            await asyncio.sleep(self.execution_time / steps)
            await context.report_progress(i + 1, steps, f"Step {i + 1}/{steps}")

        return f"Result for {task_input}"


class TestCoreTypes:
    """Test core type definitions and enums."""

    def test_task_id_generation(self):
        """Test TaskId generation and uniqueness."""
        task_id1 = TaskId()
        task_id2 = TaskId()

        assert task_id1.value != task_id2.value
        assert str(task_id1) == task_id1.value

    def test_task_id_custom_value(self):
        """Test TaskId with custom value."""
        custom_id = "custom-task-id"
        task_id = TaskId(custom_id)
        assert task_id.value == custom_id
        assert str(task_id) == custom_id

    def test_task_configuration_defaults(self):
        """Test TaskConfiguration default values."""
        config = TaskConfiguration()

        assert config.max_retries == 3
        assert config.timeout is None
        assert config.priority == Priority.NORMAL
        assert config.resources == {}
        assert config.dependencies == []
        assert config.tags == []

    def test_task_configuration_merge(self):
        """Test TaskConfiguration merging."""
        config1 = TaskConfiguration(max_retries=5, priority=Priority.HIGH, resources={"cpu": 2}, tags=["test"])

        config2 = TaskConfiguration(max_retries=7, resources={"memory": "1GB"}, tags=["important"])

        merged = config1.merge(config2)

        assert merged.max_retries == 7  # config2 takes precedence
        assert merged.priority == Priority.HIGH  # from config1
        assert merged.resources == {"cpu": 2, "memory": "1GB"}  # merged
        assert set(merged.tags) == {"test", "important"}  # merged


class TestTask:
    """Test Task class functionality."""

    def test_task_creation(self):
        """Test basic task creation."""
        executor = TestTaskExecutor()
        task = Task("test_task", executor, "test_input")

        assert task.name == "test_task"
        assert task.input == "test_input"
        assert task.state == TaskState.PENDING
        assert task.executor == executor
        assert isinstance(task.id, TaskId)
        assert isinstance(task.created_at, datetime)

    def test_task_with_config(self):
        """Test task creation with custom configuration."""
        executor = TestTaskExecutor()
        config = TaskConfiguration(max_retries=5, priority=Priority.HIGH)
        task = Task("test_task", executor, "test_input", config)

        assert task.config.max_retries == 5
        assert task.config.priority == Priority.HIGH

    def test_task_state_changes(self):
        """Test task state transitions."""
        executor = TestTaskExecutor()
        task = Task("test_task", executor, "test_input")

        # Test state changes
        task.state = TaskState.QUEUED
        assert task.state == TaskState.QUEUED

        task.state = TaskState.RUNNING
        assert task.state == TaskState.RUNNING

        task.state = TaskState.COMPLETED
        assert task.state == TaskState.COMPLETED

    def test_task_progress_tracking(self):
        """Test task progress tracking."""
        executor = TestTaskExecutor()
        task = Task("test_task", executor, "test_input")

        # Test initial progress
        assert task.get_progress() == 0

        # Test progress updates
        task.update_progress(25, "Quarter complete")
        assert task.get_progress() == 25
        assert task.get_progress_message() == "Quarter complete"

        task.update_progress(100, "Complete")
        assert task.get_progress() == 100


class TestSimulationConfig:
    """Test configuration management."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SimulationConfig()

        assert config.execution_mode == ExecutionMode.ASYNC
        assert config.enable_progress_monitoring is True
        assert config.max_retry_attempts == 3
        assert config.debug_mode is False

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid max_workers
        with pytest.raises(ConfigurationError):
            SimulationConfig(max_workers=-1)

        # Test invalid timeout
        with pytest.raises(ConfigurationError):
            SimulationConfig(task_timeout_seconds=-5.0)

    def test_config_merge(self):
        """Test configuration merging."""
        config1 = SimulationConfig(max_workers=4, debug_mode=True)
        config2 = SimulationConfig(max_workers=8, batch_size=100)

        merged = config1.merge(config2)

        assert merged.max_workers == 8  # config2 takes precedence
        assert merged.debug_mode is True  # from config1
        assert merged.batch_size == 100  # from config2

    @patch.dict("os.environ", {"ICARUS_SIM_MAX_WORKERS": "16", "ICARUS_SIM_DEBUG": "true"})
    def test_environment_overrides(self):
        """Test environment variable overrides."""
        config = SimulationConfig()

        assert config.max_workers == 16
        assert config.debug_mode is True


class TestSummationExecutor:
    """Test the example SummationExecutor."""

    @pytest.mark.asyncio
    async def test_summation_executor_basic(self):
        """Test basic summation execution."""
        executor = SummationExecutor()
        context = Mock(spec=ExecutionContext)
        context.is_cancelled = False
        context.report_progress = AsyncMock()

        result = await executor.execute(10, context)

        assert result == sum(range(10))  # 0+1+2+...+9 = 45
        assert context.report_progress.call_count > 0

    @pytest.mark.asyncio
    async def test_summation_executor_cancellation(self):
        """Test executor handling of cancellation."""
        executor = SummationExecutor()
        context = Mock(spec=ExecutionContext)
        context.is_cancelled = True
        context.report_progress = AsyncMock()

        with pytest.raises(asyncio.CancelledError):
            await executor.execute(1000, context)

    @pytest.mark.asyncio
    async def test_summation_executor_validation(self):
        """Test input validation."""
        executor = SummationExecutor()

        # Valid input
        assert await executor.validate_input(10) is True

        # Invalid inputs
        assert await executor.validate_input(-5) is False
        assert await executor.validate_input("not_int") is False


class TestSimulationRunner:
    """Test SimulationRunner functionality."""

    def test_runner_creation(self):
        """Test runner initialization."""
        runner = SimulationRunner()

        assert runner.execution_mode == ExecutionMode.ASYNC
        assert runner.enable_progress_monitoring is True
        assert isinstance(runner.resource_manager, SimpleResourceManager)

    def test_runner_with_config(self):
        """Test runner with custom configuration."""
        config = SimulationConfig(
            execution_mode=ExecutionMode.THREADING, max_workers=8, enable_progress_monitoring=False
        )

        runner = SimulationRunner(
            execution_mode=config.execution_mode,
            max_workers=config.max_workers,
            enable_progress_monitoring=config.enable_progress_monitoring,
        )

        assert runner.execution_mode == ExecutionMode.THREADING
        assert runner.max_workers == 8
        assert runner.enable_progress_monitoring is False

    @pytest.mark.asyncio
    async def test_runner_single_task(self):
        """Test running a single task."""
        runner = SimulationRunner()
        executor = TestTaskExecutor(execution_time=0.01)
        task = Task("test_task", executor, "test_input")

        results = await runner.run_tasks([task])

        assert len(results) == 1
        assert results[0].state == TaskState.COMPLETED
        assert results[0].task_id == task.id
        assert results[0].result == "Result for test_input"

    @pytest.mark.asyncio
    async def test_runner_multiple_tasks(self):
        """Test running multiple tasks concurrently."""
        runner = SimulationRunner()
        executor = TestTaskExecutor(execution_time=0.01)

        tasks = [Task(f"task_{i}", executor, f"input_{i}") for i in range(5)]

        start_time = time.time()
        results = await runner.run_tasks(tasks)
        execution_time = time.time() - start_time

        assert len(results) == 5
        assert all(r.state == TaskState.COMPLETED for r in results)
        # Should execute concurrently, so total time < sum of individual times
        assert execution_time < 5 * 0.01 * 1.5  # Allow some overhead

    @pytest.mark.asyncio
    async def test_runner_task_failure(self):
        """Test handling of task failures."""
        runner = SimulationRunner()
        failing_executor = TestTaskExecutor(should_fail=True)
        task = Task("failing_task", failing_executor, "test_input")

        results = await runner.run_tasks([task])

        assert len(results) == 1
        assert results[0].state == TaskState.FAILED
        assert results[0].error is not None
        assert isinstance(results[0].error, ValueError)


class TestIntegration:
    """Integration tests for the complete simulation framework."""

    @pytest.mark.asyncio
    async def test_end_to_end_simulation(self):
        """Test complete end-to-end simulation workflow."""
        # Create configuration
        config = SimulationConfig(max_workers=4, enable_progress_monitoring=True, debug_mode=True)

        # Create runner
        runner = SimulationRunner(
            execution_mode=config.execution_mode,
            max_workers=config.max_workers,
            enable_progress_monitoring=config.enable_progress_monitoring,
        )

        # Create mixed tasks (some fast, some slow, some failing)
        tasks = []

        # Fast tasks
        fast_executor = TestTaskExecutor(execution_time=0.01)
        for i in range(3):
            tasks.append(Task(f"fast_task_{i}", fast_executor, i))

        # Slow tasks
        slow_executor = TestTaskExecutor(execution_time=0.05)
        for i in range(2):
            tasks.append(Task(f"slow_task_{i}", slow_executor, i * 10))

        # One failing task
        failing_executor = TestTaskExecutor(should_fail=True)
        tasks.append(Task("failing_task", failing_executor, "fail_input"))

        # Execute simulation
        results = await runner.run_tasks(tasks)

        # Verify results
        assert len(results) == 6

        completed_results = [r for r in results if r.state == TaskState.COMPLETED]
        failed_results = [r for r in results if r.state == TaskState.FAILED]

        assert len(completed_results) == 5
        assert len(failed_results) == 1

        # Verify fast task results
        fast_results = [r for r in completed_results if "fast_task" in str(r.task_id)]
        assert len(fast_results) == 3

        # Verify slow task results
        slow_results = [r for r in completed_results if "slow_task" in str(r.task_id)]
        assert len(slow_results) == 2

    @pytest.mark.asyncio
    async def test_progress_monitoring_integration(self):
        """Test progress monitoring integration."""
        runner = SimulationRunner(enable_progress_monitoring=True)
        executor = TestTaskExecutor(execution_time=0.02)

        tasks = [Task(f"monitored_task_{i}", executor, i) for i in range(3)]

        # Track progress updates
        progress_updates = []

        def mock_progress_update(update):
            progress_updates.append(update)

        with patch.object(runner, "_handle_progress_update", side_effect=mock_progress_update):
            results = await runner.run_tasks(tasks)

        assert len(results) == 3
        assert all(r.state == TaskState.COMPLETED for r in results)
        # Should have received progress updates (exact count may vary)
        assert len(progress_updates) > 0


# Fixtures for common test objects
@pytest.fixture
def sample_config():
    """Provide a sample configuration for tests."""
    return SimulationConfig(max_workers=4, task_timeout_seconds=30.0, enable_progress_monitoring=True, debug_mode=True)


@pytest.fixture
def test_executor():
    """Provide a test executor for tests."""
    return TestTaskExecutor(execution_time=0.01)


@pytest.fixture
def simulation_runner(sample_config):
    """Provide a configured simulation runner for tests."""
    return SimulationRunner(
        execution_mode=sample_config.execution_mode,
        max_workers=sample_config.max_workers,
        enable_progress_monitoring=sample_config.enable_progress_monitoring,
    )


# Performance benchmarks
class TestPerformance:
    """Performance benchmarks for the simulation framework."""

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_task_creation_performance(self, benchmark):
        """Benchmark task creation performance."""
        executor = TestTaskExecutor()

        def create_tasks():
            return [Task(f"task_{i}", executor, i) for i in range(1000)]

        tasks = benchmark(create_tasks)
        assert len(tasks) == 1000

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_concurrent_execution_performance(self, benchmark):
        """Benchmark concurrent task execution."""
        runner = SimulationRunner(max_workers=8)
        executor = TestTaskExecutor(execution_time=0.001)

        async def run_simulation():
            tasks = [Task(f"perf_task_{i}", executor, i) for i in range(100)]
            return await runner.run_tasks(tasks)

        results = await benchmark(run_simulation)
        assert len(results) == 100
        assert all(r.state == TaskState.COMPLETED for r in results)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
