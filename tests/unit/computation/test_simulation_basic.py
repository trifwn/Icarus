"""
Simplified test suite for the ICARUS simulation framework.

This module provides basic unit tests that comply with the current
implementation and can be extended as the framework evolves.
"""

from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest

from ICARUS.computation.core import ConfigurationError
from ICARUS.computation.core import ExecutionContext
from ICARUS.computation.core import ExecutionMode
from ICARUS.computation.core import Priority
from ICARUS.computation.core import SimulationConfig
from ICARUS.computation.core import Task
from ICARUS.computation.core import TaskConfiguration
from ICARUS.computation.core import TaskId
from ICARUS.computation.core import TaskState
from ICARUS.computation.executors import SummationExecutor
from ICARUS.computation.resources.manager import SimpleResourceManager
from ICARUS.computation.runners import SimulationRunner


class MockTaskExecutor:
    """Simple mock task executor that complies with the protocol."""

    def __init__(self, result_value="mock_result", should_fail=False):
        self.result_value = result_value
        self.should_fail = should_fail
        self.execute_count = 0

    async def execute(self, task_input, context: ExecutionContext):
        """Execute task and return result."""
        self.execute_count += 1

        if self.should_fail:
            raise ValueError(f"Mock failure for input: {task_input}")

        # Simulate some progress
        await context.report_progress(50, 100, "Half done")
        await context.report_progress(100, 100, "Complete")

        return f"{self.result_value}_{task_input}"

    async def validate_input(self, task_input):
        """Validate input."""
        return True

    async def cleanup(self):
        """Cleanup resources."""
        pass


class TestCoreTypes:
    """Test core type definitions."""

    def test_task_id_creation(self):
        """Test TaskId creation and uniqueness."""
        task_id1 = TaskId()
        task_id2 = TaskId()

        assert task_id1.value != task_id2.value
        assert len(task_id1.value) > 0
        assert str(task_id1) == task_id1.value

    def test_task_configuration_defaults(self):
        """Test default configuration values."""
        config = TaskConfiguration()

        assert config.max_retries == 3
        assert config.priority == Priority.NORMAL
        assert config.resources == {}
        assert config.dependencies == []

    def test_task_configuration_merge(self):
        """Test configuration merging."""
        config1 = TaskConfiguration(max_retries=5, priority=Priority.HIGH)
        config2 = TaskConfiguration(max_retries=7, resources={"cpu": 2})

        merged = config1.merge(config2)

        assert merged.max_retries == 7
        assert merged.priority == Priority.HIGH
        assert merged.resources == {"cpu": 2}


class TestTask:
    """Test Task functionality."""

    def test_task_creation(self):
        """Test basic task creation."""
        executor = MockTaskExecutor()
        task = Task("test_task", executor, "test_input")

        assert task.name == "test_task"
        assert task.input == "test_input"
        assert task.state == TaskState.PENDING
        assert isinstance(task.id, TaskId)

    def test_task_state_transitions(self):
        """Test task state changes."""
        executor = MockTaskExecutor()
        task = Task("test_task", executor, "test_input")

        task.state = TaskState.RUNNING
        assert task.state == TaskState.RUNNING

        task.state = TaskState.COMPLETED
        assert task.state == TaskState.COMPLETED

    def test_task_progress_tracking(self):
        """Test progress tracking."""
        executor = MockTaskExecutor()
        task = Task("test_task", executor, "test_input")

        # Test progress updates
        task.update_progress(25, 100, "Quarter done")
        assert task.get_progress() == 25
        assert task.get_progress_message() == "Quarter done"

        task.update_progress(100, 100, "Complete")
        assert task.get_progress() == 100


class TestSimulationConfig:
    """Test configuration management."""

    def test_default_config(self):
        """Test default configuration."""
        config = SimulationConfig()

        assert config.execution_mode == ExecutionMode.ASYNC
        assert config.enable_progress_monitoring is True
        assert config.max_retry_attempts == 3

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ConfigurationError):
            SimulationConfig(max_workers=-1)

        with pytest.raises(ConfigurationError):
            SimulationConfig(task_timeout_seconds=-1.0)

    def test_config_merge(self):
        """Test configuration merging."""
        config1 = SimulationConfig(max_workers=4, debug_mode=True)
        config2 = SimulationConfig(max_workers=8)

        merged = config1.merge(config2)
        assert merged.max_workers == 8
        assert merged.debug_mode is True


class TestSummationExecutor:
    """Test the SummationExecutor."""

    @pytest.mark.asyncio
    async def test_summation_basic(self):
        """Test basic summation."""
        executor = SummationExecutor()
        context = Mock(spec=ExecutionContext)
        context.is_cancelled = False
        context.report_progress = AsyncMock()

        result = await executor.execute(5, context)

        # Sum of 0+1+2+3+4 = 10
        assert result == 10
        assert context.report_progress.call_count > 0

    @pytest.mark.asyncio
    async def test_summation_validation(self):
        """Test input validation."""
        executor = SummationExecutor()

        assert await executor.validate_input(10) is True
        assert await executor.validate_input(-5) is False


class TestSimulationRunner:
    """Test SimulationRunner functionality."""

    def test_runner_creation(self):
        """Test runner initialization."""
        runner = SimulationRunner()

        assert runner.execution_mode == ExecutionMode.ASYNC
        assert runner.enable_progress_monitoring is True
        assert isinstance(runner.resource_manager, SimpleResourceManager)

    @pytest.mark.asyncio
    async def test_single_task_execution(self):
        """Test executing a single task."""
        runner = SimulationRunner()
        executor = MockTaskExecutor("success")
        task = Task("test_task", executor, "input")

        results = await runner.run_tasks([task])

        assert len(results) == 1
        assert results[0].state == TaskState.COMPLETED
        assert "success_input" in str(results[0].result)

    @pytest.mark.asyncio
    async def test_multiple_task_execution(self):
        """Test executing multiple tasks."""
        runner = SimulationRunner()
        executor = MockTaskExecutor()

        tasks = [Task(f"task_{i}", executor, f"input_{i}") for i in range(3)]

        results = await runner.run_tasks(tasks)

        assert len(results) == 3
        assert all(r.state == TaskState.COMPLETED for r in results)

    @pytest.mark.asyncio
    async def test_task_failure_handling(self):
        """Test handling of failed tasks."""
        runner = SimulationRunner()
        failing_executor = MockTaskExecutor(should_fail=True)
        task = Task("failing_task", failing_executor, "input")

        results = await runner.run_tasks([task])

        assert len(results) == 1
        assert results[0].state == TaskState.FAILED
        assert results[0].error is not None


class TestIntegration:
    """Basic integration tests."""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete workflow."""
        # Create runner with custom config
        config = SimulationConfig(max_workers=2)
        runner = SimulationRunner(execution_mode=config.execution_mode, max_workers=config.max_workers)

        # Create tasks
        executor = MockTaskExecutor()
        tasks = [Task(f"task_{i}", executor, i) for i in range(3)]

        # Execute
        results = await runner.run_tasks(tasks)

        # Verify
        assert len(results) == 3
        assert all(r.state == TaskState.COMPLETED for r in results)

        # Check executor was called
        assert executor.execute_count == 3


# Test fixtures
@pytest.fixture
def mock_executor():
    """Provide a mock executor."""
    return MockTaskExecutor()


@pytest.fixture
def sample_task(mock_executor):
    """Provide a sample task."""
    return Task("sample_task", mock_executor, "sample_input")


@pytest.fixture
def runner():
    """Provide a simulation runner."""
    return SimulationRunner()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
