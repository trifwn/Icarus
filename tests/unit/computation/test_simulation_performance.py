"""
Performance benchmarks for the simulation framework.

This module provides performance benchmarks to ensure the simulation
framework scales appropriately and maintains good performance characteristics.
"""

import time

import pytest

from ICARUS.computation.core import SimulationConfig
from ICARUS.computation.core import Task
from ICARUS.computation.executors import SummationExecutor
from ICARUS.computation.runners import SimulationRunner


class FastMockExecutor:
    """High-performance mock executor for benchmarks."""

    async def execute(self, task_input, context):
        """Fast execution with minimal overhead."""
        # Minimal computation
        result = task_input * 2
        await context.report_progress(100, 100, "Done")
        return result

    async def validate_input(self, task_input):
        return True

    async def cleanup(self):
        pass


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    @pytest.mark.benchmark
    def test_task_creation_performance(self, benchmark):
        """Benchmark task creation speed."""
        executor = FastMockExecutor()

        def create_tasks():
            return [Task(f"task_{i}", executor, i) for i in range(1000)]

        tasks = benchmark(create_tasks)
        assert len(tasks) == 1000

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_small_batch_execution(self, benchmark):
        """Benchmark execution of small task batches."""
        runner = SimulationRunner(max_workers=4)
        executor = FastMockExecutor()

        async def run_small_batch():
            tasks = [Task(f"task_{i}", executor, i) for i in range(10)]
            return await runner.run_tasks(tasks)

        results = await benchmark(run_small_batch)
        assert len(results) == 10

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_large_batch_execution(self, benchmark):
        """Benchmark execution of large task batches."""
        runner = SimulationRunner(max_workers=8)
        executor = FastMockExecutor()

        async def run_large_batch():
            tasks = [Task(f"task_{i}", executor, i) for i in range(100)]
            return await runner.run_tasks(tasks)

        results = await benchmark(run_large_batch)
        assert len(results) == 100

    @pytest.mark.benchmark
    def test_configuration_creation(self, benchmark):
        """Benchmark configuration object creation."""

        def create_config():
            return SimulationConfig(max_workers=4, enable_progress_monitoring=True, debug_mode=False)

        config = benchmark(create_config)
        assert config.max_workers == 4

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrency_scaling(self):
        """Test how performance scales with concurrency."""
        executor = SummationExecutor()
        task_count = 50

        # Test different worker counts
        worker_counts = [1, 2, 4, 8]
        execution_times = {}

        for workers in worker_counts:
            runner = SimulationRunner(max_workers=workers)
            tasks = [
                Task(f"task_{i}", executor, 100)  # Sum 0..99
                for i in range(task_count)
            ]

            start_time = time.time()
            results = await runner.run_tasks(tasks)
            execution_time = time.time() - start_time

            execution_times[workers] = execution_time

            # Verify all tasks completed
            assert len(results) == task_count
            assert all(r.state.name == "COMPLETED" for r in results)

        # Performance should improve with more workers (up to a point)
        print(f"Execution times by worker count: {execution_times}")

        # Ensure we can handle at least basic scaling
        assert execution_times[8] <= execution_times[1] * 1.5  # Allow some overhead

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self):
        """Test that memory usage remains stable during execution."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        runner = SimulationRunner(max_workers=4)
        executor = FastMockExecutor()

        # Run multiple batches to test memory stability
        for batch in range(5):
            tasks = [Task(f"batch_{batch}_task_{i}", executor, i) for i in range(50)]

            results = await runner.run_tasks(tasks)
            assert len(results) == 50

            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory

            # Memory shouldn't grow excessively (allow 50MB increase)
            assert memory_increase < 50, f"Memory usage increased by {memory_increase:.2f}MB"

    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_stress_high_task_count(self):
        """Stress test with high task count."""
        runner = SimulationRunner(max_workers=8)
        executor = FastMockExecutor()

        # Create a large number of tasks
        task_count = 1000
        tasks = [Task(f"stress_task_{i}", executor, i) for i in range(task_count)]

        start_time = time.time()
        results = await runner.run_tasks(tasks)
        execution_time = time.time() - start_time

        # Verify all tasks completed
        assert len(results) == task_count
        completed_count = sum(1 for r in results if r.state.name == "COMPLETED")
        assert completed_count == task_count

        # Performance expectation: should complete within reasonable time
        max_expected_time = task_count * 0.01  # 10ms per task max
        assert execution_time < max_expected_time, f"Took {execution_time:.2f}s for {task_count} tasks"

        print(f"Executed {task_count} tasks in {execution_time:.2f}s ({task_count / execution_time:.0f} tasks/sec)")


class TestScalabilityMetrics:
    """Test scalability characteristics."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_throughput_measurement(self):
        """Measure task throughput under different conditions."""
        executor = FastMockExecutor()

        # Test configurations
        configs = [
            {"workers": 1, "tasks": 50},
            {"workers": 4, "tasks": 200},
            {"workers": 8, "tasks": 400},
        ]

        throughput_results = {}

        for config in configs:
            runner = SimulationRunner(max_workers=config["workers"])
            tasks = [Task(f"throughput_task_{i}", executor, i) for i in range(config["tasks"])]

            start_time = time.time()
            results = await runner.run_tasks(tasks)
            execution_time = time.time() - start_time

            throughput = len(results) / execution_time
            throughput_results[f"{config['workers']}_workers"] = throughput

            print(f"Workers: {config['workers']}, Tasks: {config['tasks']}, Throughput: {throughput:.2f} tasks/sec")

        # Verify throughput scaling makes sense
        assert throughput_results["4_workers"] > throughput_results["1_workers"]
        assert throughput_results["8_workers"] >= throughput_results["4_workers"] * 0.8  # Allow some overhead


if __name__ == "__main__":
    # Run benchmarks
    pytest.main([__file__, "-v", "--benchmark-only", "--benchmark-sort=mean"])
