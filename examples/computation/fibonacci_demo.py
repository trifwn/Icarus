#!/usr/bin/env python3
"""
Comprehensive Fibonacci Simulation Demo

This demo showcases the ICARUS simulation framework by running 15 Fibonacci
calculation jobs across all supported execution modes. It demonstrates:

1. Task creation and configuration
2. Progress monitoring with visual progress bars
3. Different execution strategies (Sequential, Threading, Async)
4. Performance comparison between modes
5. Error handling and cancellation
6. Results aggregation and reporting

Usage:
    python fibonacci_demo.py [--numbers N1 N2 ...] [--delay DELAY] [--mode MODE]
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Add the examples directory to the path so we can import our executor
sys.path.insert(0, str(Path(__file__).parent))

from fibonacci_executor import FibonacciExecutor

from ICARUS.computation import RichProgressMonitor
from ICARUS.computation import SimulationRunner
from ICARUS.computation.core import ExecutionMode
from ICARUS.computation.core import Priority
from ICARUS.computation.core import SimulationConfig
from ICARUS.computation.core import Task
from ICARUS.computation.core import TaskConfiguration
from ICARUS.computation.core import TaskId
from ICARUS.computation.core import TaskResult
from ICARUS.computation.core import TaskState


class FibonacciDemo:
    """Demo class that orchestrates Fibonacci calculations across different execution modes."""

    def __init__(self, numbers: list[int] | None = None, delay_per_step: float = 0.2):
        """Initialize the demo."""
        self.numbers = numbers or list(
            range(8, 23),
        )  # F(8) through F(22) for good visualization
        self.delay_per_step = delay_per_step
        self.results: dict[str, dict[str, Any]] = {}

        # Configure logging
        self.logger = logging.getLogger("FibonacciDemo")

    def create_tasks(self) -> list[Task]:
        """Create Fibonacci calculation tasks."""
        tasks = []

        # Choose executor type
        executor = FibonacciExecutor(delay_per_step=self.delay_per_step)

        for i, n in enumerate(self.numbers):
            # Create task configuration with varying priorities
            if n <= 10:
                priority = Priority.LOW
            elif n <= 15:
                priority = Priority.NORMAL
            elif n <= 20:
                priority = Priority.HIGH
            else:
                priority = Priority.CRITICAL

            config = TaskConfiguration(
                priority=priority,
                max_retries=2,
                tags=[
                    "fibonacci",
                    f"batch_{i // 5}",
                    f"priority_{priority.name.lower()}",
                ],
            )

            # Create task
            task = Task(
                task_id=TaskId(),
                name=f"Fibonacci_F({n})",
                executor=executor,
                task_input=n,
                config=config,
            )

            tasks.append(task)

        return tasks

    async def run_execution_mode(self, mode: ExecutionMode) -> dict[str, Any]:
        """Run the demo for a specific execution mode."""
        print(f"\n{'=' * 60}")
        print(f"🚀 Running Fibonacci Demo - {mode.value.upper()} Mode")
        print(f"{'=' * 60}")

        # Create configuration for this execution mode
        config = SimulationConfig(
            execution_mode=mode,
            max_workers=4 if mode != ExecutionMode.SEQUENTIAL else 1,
            progress_refresh_rate=0.1,  # Faster refresh for demo
            debug_mode=False,
        )

        # Create runner
        runner = SimulationRunner(
            execution_mode=config.execution_mode,
            max_workers=config.max_workers,
            progress_monitor=RichProgressMonitor(1.0)
            if config.enable_progress_monitoring
            else None,
        )

        # Create tasks
        tasks = self.create_tasks()

        print(f"📊 Executing {len(tasks)} Fibonacci tasks...")
        print(f"🔢 Numbers to calculate: {self.numbers}")
        print(f"⚙️  Execution mode: {mode.value}")
        print(f"👥 Max workers: {config.max_workers}")
        print(f"⏱️  Delay per step: {self.delay_per_step}s")

        # Record start time
        start_time = time.time()

        try:
            # Execute tasks
            results = await runner.run_tasks(tasks)
            execution_time = time.time() - start_time

            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Task execution failed with exception: {result}")
                    import traceback

                    traceback.print_exc()

                if isinstance(result, TaskResult) and result.state == TaskState.FAILED:
                    self.logger.error(
                        f"Task {result.task_id} failed with error: {result.error}",
                    )
                    if result.error:
                        raise (result.error)

            # Analyze results
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]

            # Extract Fibonacci values
            fibonacci_values = {}
            for result in successful_results:
                if result.output and isinstance(result.output, tuple):
                    n, fib_value = result.output
                    fibonacci_values[n] = fib_value

            # Print results summary
            print("\n✅ Execution completed!")
            print(f"⏱️  Total time: {execution_time:.2f} seconds")
            print(f"✅ Successful: {len(successful_results)}/{len(tasks)}")
            if failed_results:
                print(f"❌ Failed: {len(failed_results)}")

            # Show some Fibonacci results
            print("\n🔢 Fibonacci Results:")
            for n in sorted(fibonacci_values.keys())[:20]:  # Show first 8
                print(f"   F({n:2d}) = {fibonacci_values[n]:,}")
            if len(fibonacci_values) > 20:
                print(f"   ... and {len(fibonacci_values) - 20} more")

            # Calculate performance metrics
            if successful_results:
                avg_task_time = execution_time / len(successful_results)
                throughput = len(successful_results) / execution_time
            else:
                avg_task_time = 0
                throughput = 0

            return {
                "mode": mode.value,
                "execution_time": execution_time,
                "total_tasks": len(tasks),
                "successful_tasks": len(successful_results),
                "failed_tasks": len(failed_results),
                "success_rate": len(successful_results) / len(tasks) * 100,
                "avg_task_time": avg_task_time,
                "throughput": throughput,
                "fibonacci_values": fibonacci_values,
                "worker_count": config.max_workers,
            }

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ Execution failed: {e}")
            import traceback

            traceback.print_exc()

            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Task execution failed with exception: {result}")
                if isinstance(result, TaskResult) and result.state == TaskState.FAILED:
                    self.logger.error(
                        f"Task {result.task_id} failed with error: {result.error}",
                    )
                    if result.error:
                        raise (result.error)

            return {
                "mode": mode.value,
                "execution_time": execution_time,
                "error": str(e),
                "total_tasks": len(tasks),
                "successful_tasks": 0,
                "failed_tasks": len(tasks),
                "success_rate": 0.0,
            }

    async def run_all_modes(
        self,
        modes: list[ExecutionMode] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Run the demo across all specified execution modes."""
        if modes is None:
            modes = [
                ExecutionMode.SEQUENTIAL,
                ExecutionMode.ASYNC,
                ExecutionMode.THREADING,
                ExecutionMode.MULTIPROCESSING,
                ExecutionMode.ADAPTIVE,
            ]

        all_results = {}

        print("🎯 ICARUS Simulation Framework - Fibonacci Calculation Demo")
        print("=" * 70)
        print(
            f"This demo will calculate Fibonacci numbers F(n) for n in: {self.numbers}",
        )
        print("Each calculation includes artificial delays for visualization.")
        print("Progress bars will show real-time execution status.")

        for mode in modes:
            try:
                result = await self.run_execution_mode(mode)
                all_results[mode.value] = result
                # Brief pause between modes
                await asyncio.sleep(1)
            except Exception as e:
                print(f"❌ Failed to run mode {mode.value}: {e}")
                all_results[mode.value] = {
                    "mode": mode.value,
                    "error": str(e),
                    "successful_tasks": 0,
                    "failed_tasks": len(self.numbers),
                    "success_rate": 0.0,
                }
        return all_results


async def main():
    """Main demo function."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Fibonacci Simulation Framework Demo")
    parser.add_argument(
        "--numbers",
        type=int,
        nargs="+",
        default=list(range(8, 23)),  # F(8) through F(22) for good visualization
        help="Fibonacci numbers to calculate (default: 8-22)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay per calculation step in seconds (default: 0.015)",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "sequential",
            "async",
            "threading",
            "multiprocessing",
            "adaptive",
            "all",
        ],
        default="all",
        help="Execution mode to run (default: all)",
    )

    args = parser.parse_args()

    # Create demo instance
    demo = FibonacciDemo(numbers=args.numbers, delay_per_step=args.delay)

    # Determine modes to run
    if args.mode == "all":
        modes = [
            ExecutionMode.SEQUENTIAL,
            ExecutionMode.ASYNC,
            ExecutionMode.THREADING,
            ExecutionMode.MULTIPROCESSING,
            ExecutionMode.ADAPTIVE,
        ]
    else:
        mode_map = {
            "sequential": ExecutionMode.SEQUENTIAL,
            "async": ExecutionMode.ASYNC,
            "threading": ExecutionMode.THREADING,
            "multiprocessing": ExecutionMode.MULTIPROCESSING,
            "adaptive": ExecutionMode.ADAPTIVE,
        }
        modes = [mode_map[args.mode]]

    try:
        # Run the demo
        results = await demo.run_all_modes(modes)

        # Show performance comparison if multiple modes
        if len(modes) > 1:
            print_performance_comparison(results)

        print("\n🎉 Demo completed successfully!")
        print(
            f"📋 Summary: Calculated Fibonacci numbers for {len(args.numbers)} values",
        )
        print("🕒 Total execution across all modes completed")

    except KeyboardInterrupt:
        print("\n⛔ Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        sys.exit(1)


def print_performance_comparison(results: dict[str, dict[str, Any]]):
    """Print a comprehensive performance comparison between execution modes."""
    print(f"\n{'=' * 70}")
    print("📊 PERFORMANCE COMPARISON")
    print(f"{'=' * 70}")

    # Create performance table
    headers = ["Mode", "Time (s)", "Success Rate", "Throughput (tasks/s)", "Workers"]
    row_format = "{:<12} {:>10} {:>12} {:>18} {:>8}"

    print(row_format.format(*headers))
    print("-" * 70)

    successful_modes = []
    for mode_name, result in results.items():
        if "error" not in result and result["successful_tasks"] > 0:
            successful_modes.append((mode_name, result))
            print(
                row_format.format(
                    mode_name.upper(),
                    f"{result['execution_time']:.2f}",
                    f"{result['success_rate']:.1f}%",
                    f"{result['throughput']:.2f}",
                    f"{result.get('worker_count', 'N/A')}",
                ),
            )
        else:
            print(row_format.format(mode_name.upper(), "FAILED", "0.0%", "0.00", "N/A"))

    # Performance insights
    if successful_modes:
        print("\n🏆 PERFORMANCE INSIGHTS:")

        # Find fastest mode
        fastest_mode = min(successful_modes, key=lambda x: x[1]["execution_time"])
        print(
            f"   ⚡ Fastest: {fastest_mode[0].upper()} ({fastest_mode[1]['execution_time']:.2f}s)",
        )

        # Find highest throughput
        highest_throughput = max(successful_modes, key=lambda x: x[1]["throughput"])
        print(
            f"   🚀 Highest throughput: {highest_throughput[0].upper()} ({highest_throughput[1]['throughput']:.2f} tasks/s)",
        )

        # Show scaling efficiency
        sequential_time = None
        for mode_name, result in successful_modes:
            if mode_name == "sequential":
                sequential_time = result["execution_time"]
                break

        if sequential_time:
            print("\n📈 SCALING EFFICIENCY (vs Sequential):")
            for mode_name, result in successful_modes:
                if mode_name != "sequential":
                    speedup = sequential_time / result["execution_time"]
                    workers = result.get("worker_count", 1)
                    efficiency = (speedup / workers) * 100 if workers > 1 else 100
                    print(
                        f"   {mode_name.upper()}: {speedup:.2f}x speedup, {efficiency:.1f}% efficiency",
                    )


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
