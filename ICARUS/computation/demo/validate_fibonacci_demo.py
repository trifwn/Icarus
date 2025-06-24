#!/usr/bin/env python3
"""
Validation script for the Fibonacci Demo

This script performs basic validation to ensure the demo components
are working correctly before running the full demo.
"""

import sys
import asyncio
from pathlib import Path

# Add the examples directory to the path
sys.path.insert(0, str(Path(__file__).parent))


async def test_fibonacci_executor():
    """Test the Fibonacci executor independently."""
    print("🧪 Testing Fibonacci Executor...")

    try:
        from fibonacci_executor import FibonacciExecutor
        from unittest.mock import AsyncMock, Mock

        # Create mock context
        context = Mock()
        context.is_cancelled = False
        context.report_progress = AsyncMock()

        # Create executor
        executor = FibonacciExecutor(delay_per_step=0.001)  # Fast for testing

        # Test small Fibonacci calculation
        result = await executor.execute(10, context)

        # Validate result
        expected = (10, 55)  # F(10) = 55
        assert result == expected, f"Expected {expected}, got {result}"

        # Validate progress was reported
        assert context.report_progress.call_count > 0, "Progress should be reported"

        print("   ✅ Fibonacci executor test passed")
        return True

    except Exception as e:
        print(f"   ❌ Fibonacci executor test failed: {e}")
        return False


async def test_simulation_framework():
    """Test basic simulation framework functionality."""
    print("🧪 Testing Simulation Framework...")

    try:
        from ICARUS.computation.core.types import ExecutionMode, Priority, TaskConfiguration
        from ICARUS.computation.core.task import Task
        from ICARUS.computation.runners import SimulationRunner
        from fibonacci_executor import FibonacciExecutor

        # Create a simple task
        executor = FibonacciExecutor(delay_per_step=0.001)
        config = TaskConfiguration(priority=Priority.NORMAL)
        task = Task("test_fib", executor, 5, config)

        # Create runner
        runner = SimulationRunner(
            execution_mode=ExecutionMode.ASYNC,
            max_workers=2,
            progress_monitor=None,  # Disable for clean testing
        )

        # Run single task
        results = await runner.run_tasks([task])

        # Validate results        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        assert results[0].state.name == "COMPLETED", "Task should complete successfully"
        assert results[0].output == (5, 5), "F(5) should equal 5"  # F(5) = 5

        print("   ✅ Simulation framework test passed")
        return True

    except Exception as e:
        print(f"   ❌ Simulation framework test failed: {e}")
        return False


async def test_demo_components():
    """Test demo components."""
    print("🧪 Testing Demo Components...")

    try:
        from examples.computation.fibonacci_demo import FibonacciDemo
        from ICARUS.computation.core.types import ExecutionMode

        # Create demo instance
        demo = FibonacciDemo(numbers=[5, 6, 7], delay_per_step=0.001)

        # Test task creation
        tasks = demo.create_tasks()
        assert len(tasks) == 3, f"Expected 3 tasks, got {len(tasks)}"

        # Test single execution mode
        result = await demo.run_execution_mode(ExecutionMode.ASYNC)
        assert result["successful_tasks"] == 3, "All tasks should succeed"
        assert result["success_rate"] == 100.0, "Success rate should be 100%"

        print("   ✅ Demo components test passed")
        return True

    except Exception as e:
        print(f"   ❌ Demo components test failed: {e}")
        return False


def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing Imports...")

    try:
        # Test ICARUS computation imports
        from ICARUS.computation.core.types import ExecutionMode, Priority, TaskConfiguration
        from ICARUS.computation.core.task import Task
        from ICARUS.computation.core.config import SimulationConfig
        from ICARUS.computation.runners import SimulationRunner
        from ICARUS.computation.core.data_structures import TaskState

        # Test local imports
        from fibonacci_executor import FibonacciExecutor

        print("   ✅ All imports successful")
        return True

    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False


async def main():
    """Main validation function."""
    print("🔍 Fibonacci Demo Validation")
    print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("Fibonacci Executor", test_fibonacci_executor),
        ("Simulation Framework", test_simulation_framework),
        ("Demo Components", test_demo_components),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'=' * 50}")
    print("📋 Validation Summary:")

    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1

    print(f"\n📊 Results: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 All validations passed! Demo is ready to run.")
        print("\nNext steps:")
        print("   python run_fibonacci_demo.py quick")
        print("   python run_fibonacci_demo.py standard")
        return True
    else:
        print("❌ Some validations failed. Please fix issues before running demo.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
